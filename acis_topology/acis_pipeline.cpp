//=========================================================================================
// acis_pipeline.cpp
// T4 M1：ACIS 侧「网络输入特征」重建 —— 端到端验证的几何/拓扑基础。
//
// 目标：在 ACIS 内核上，为每条 STEP 生成 BRepNet 端到端前向所需的 4 类输入，逐一对齐
// OCCT BRepPipeline 的语义（不改动任何 OCCT 生产代码）：
//   - 拓扑   : faces(面序)、coedges(id/face_idx/edge_idx/mate_idx/orientation)、edges
//   - FaceGridsGlobal   [Nf, 9, 20, 20]  通道 [x,y,z,nx,ny,nz,mask,u,v]
//   - LCS    : 每条 coedge 一个 [4,4]（列主序=u 法线,v 切线,w=u×v, 平移=中点），奇异/退化→无效
//   - FaceGridsLocal    [Nc, 2, 9, 20, 20]（父面/配偶面网格在各自 coedge LCS 下）
//   - CoedgeGridsLocal  [Nc, 9, 20, 20]（配偶面网格在当前 coedge LCS 下）
//
// LCS 语义（对齐 OCCT compute_coedge_lcs）：
//   u = face 在该边中点处的【面向】法线（du×dv，face REVERSED 则取反）
//   v = edge 在弧长中点处的切线（coedge REVERSED 则取反），归一化
//   若 v∥u（|u×v|<1e-6）→ 奇异，LCS 无效(bool=false)，局部网格清零
//   退化边（无有效 3D 曲线）→ origin=-2000，LCS 无效
// 本文件为诊断探针：输出拓扑/valid/正交性/局部网格统计，先验证 ACIS 几何链路，供后续接网络。
//
// 用法: acis_pipeline.exe <input.step> <out.txt> [input_format]
//=========================================================================================

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _WINDOWS_SOURCE
#include <wtypes.h>
#include <winnt.h>
#undef GetMessage
#endif

// InterOp (Connect 接口)
#include "SPAIDocument.h"
#include "SPAIConverter.h"
#include "SPAIFile.h"
#include "SPAIOptions.h"
#include "SPAIOptionName.h"
#include "SPAIResult.h"
#include "SPAIUnit.h"
#include "SPAIValue.h"
#include "SPAISystemInitGuard.h"
#include "SPAIAcisDocument.h"

// ACIS 内核 / 几何
#include "savres.hxx"
#include "fileinfo.hxx"
#include "lists.hxx"
#include "kernapi.hxx"
#include "license.hxx"
#include "spa_unlock_result.hxx"
#include "spatial_license.h"

// 拓扑
#include "body.hxx"
#include "lump.hxx"
#include "shell.hxx"
#include "face.hxx"
#include "edge.hxx"
#include "vertex.hxx"
#include "point.hxx"
#include "coedge.hxx"
#include "loop.hxx"

// 几何
#include "curve.hxx"
#include "pcurve.hxx"
#include "surdef.hxx"
#include "curdef.hxx"
#include "surface.hxx"
#include "param.hxx"
#include "position.hxx"
#include "vector.hxx"

// 点-面包含(mask) / 变换 / 参数区间
#include "intrapi.hxx"
#include "transf.hxx"
#include "interval.hxx"

#include <vector>
#include <string>
#include <map>
#include <set>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <algorithm>

//-----------------------------------------------------------------------------------------
// 与 BRepUtils::GetParamStrict 一致：index∈[0,total-1]，min→max（reverse=false）
static double param_strict(int index, int total, double lo, double hi) {
    if (total <= 1) return lo;
    return lo + (hi - lo) * (double)index / (double)(total - 1);
}

// 曲面 (u,v) 处自然法线 n = normalize(du × dv)；退化输出全零(与 SLProps.IsNormalDefined()==false 一致)
static bool normal_cross(const surface& srf, double u, double v, double out_n[3])
{
    SPAvector d[2];
    SPAvector* derivs[1];
    derivs[0] = d;
    SPAposition pt;
    int nb = 0;
    try { nb = srf.evaluate(SPApar_pos(u, v), pt, derivs, 1); }
    catch (...) { nb = 0; }
    if (nb < 1) return false;
    SPAvector nv = d[0] * d[1];
    double len2 = nv % nv;
    if (len2 < 1e-24) { out_n[0]=0; out_n[1]=0; out_n[2]=0; return true; }
    SPAunit_vector n = normalise(nv);
    out_n[0]=n.x(); out_n[1]=n.y(); out_n[2]=n.z();
    return true;
}

// 弧长中点参数：离散采样 100 段累加弦长，二分/顺推找到累计=半长的参数（与 OCCT ArcLengthParamFinder 同构）
static double arc_length_mid_param(const curve& cu, double u0, double u1)
{
    const int S = 100;
    std::vector<double> L(S + 1, 0.0);
    for (int k = 0; k <= S; ++k) {
        double t = u0 + (u1 - u0) * (double)k / (double)S;
        SPAposition p; cu.eval(t, p);
        if (k > 0) {
            double dx = p.x(), dy = p.y(), dz = p.z();
            // 需用上一时刻；用双缓冲简化：先存位置
            (void)dx; (void)dy; (void)dz;
        }
    }
    // 简化实现：直接在循环内累加
    std::vector<SPAposition> P(S + 1);
    for (int k = 0; k <= S; ++k) {
        double t = u0 + (u1 - u0) * (double)k / (double)S;
        cu.eval(t, P[k]);
    }
    std::vector<double> cum(S + 1, 0.0);
    for (int k = 1; k <= S; ++k) {
        double d = hypot(hypot(P[k].x()-P[k-1].x(), P[k].y()-P[k-1].y()), P[k].z()-P[k-1].z());
        cum[k] = cum[k-1] + d;
    }
    double total = cum[S];
    if (total <= 0) return 0.5 * (u0 + u1);
    double half = total * 0.5;
    for (int k = 1; k <= S; ++k) {
        if (cum[k] >= half) {
            double f = (half - cum[k-1]) / ((cum[k]-cum[k-1]) < 1e-12 ? 1e-12 : (cum[k]-cum[k-1]));
            return u0 + (u1 - u0) * (double)(k - 1 + f) / (double)S;
        }
    }
    return u0 + 0.5 * (u1 - u0);
}

// 数值投影：目标点 target 到曲面 srf 的最短参数 (u,v)（近似 GeomAPI_ProjectPointOnSurf）
static bool project_param(const surface& srf, const SPApar_box& pb, const SPAposition& target, SPApar_pos& out)
{
    double u = 0.5 * (pb.low().u + pb.high().u);
    double v = 0.5 * (pb.low().v + pb.high().v);
    for (int it = 0; it < 40; ++it) {
        SPAposition pos; SPAvector d[2]; SPAvector* dd[1] = { d };
        int nb = 0;
        try { nb = srf.evaluate(SPApar_pos(u, v), pos, dd, 1); } catch (...) { nb = 0; }
        if (nb < 1) return false;
        double rx = target.x() - pos.x();
        double ry = target.y() - pos.y();
        double rz = target.z() - pos.z();
        // 列 J = [d/du, d/dv] (SPAvector d[0]=du, d[1]=dv)
        double J[3][2] = { {d[0].x(), d[1].x()},
                           {d[0].y(), d[1].y()},
                           {d[0].z(), d[1].z()} };
        // 正规方程 (J^T J) [du,dv] = J^T r
        double a = J[0][0]*J[0][0]+J[1][0]*J[1][0]+J[2][0]*J[2][0];
        double b = J[0][0]*J[0][1]+J[1][0]*J[1][1]+J[2][0]*J[2][1];
        double c = J[0][1]*J[0][1]+J[1][1]*J[1][1]+J[2][1]*J[2][1];
        double gx = J[0][0]*rx+J[1][0]*ry+J[2][0]*rz;
        double gy = J[0][1]*rx+J[1][1]*ry+J[2][1]*rz;
        double det = a*c - b*b;
        if (fabs(det) < 1e-20) break;
        double du = (c*gx - b*gy)/det;
        double dv = (-b*gx + a*gy)/det;
        u += du; v += dv;
        if (u < pb.low().u) u = pb.low().u; if (u > pb.high().u) u = pb.high().u;
        if (v < pb.low().v) v = pb.low().v; if (v > pb.high().v) v = pb.high().v;
        double err = sqrt(rx*rx+ry*ry+rz*rz);
        if (err < 1e-9) break;
    }
    out = SPApar_pos(u, v);
    return true;
}

// 变换网格：对 [C,N] float 数组做 M 仿射(位置) + 旋转(向量)。M 为 [4,4] 列主序。
static void transform_grid_to_local(float* data, int C, int N, const float* M, bool is_face)
{
    for (int i = 0; i < N; ++i) {
        float x = data[0*N+i], y = data[1*N+i], z = data[2*N+i];
        float xn = M[0]*x + M[1]*y + M[2]*z + M[3];
        float yn = M[4]*x + M[5]*y + M[6]*z + M[7];
        float zn = M[8]*x + M[9]*y + M[10]*z + M[11];
        data[0*N+i]=xn; data[1*N+i]=yn; data[2*N+i]=zn;
        int nvec = is_face ? 1 : 3;
        for (int k = 0; k < nvec; ++k) {
            int cb = 3 + k*3;
            float vx = data[(cb+0)*N+i], vy = data[(cb+1)*N+i], vz = data[(cb+2)*N+i];
            data[(cb+0)*N+i] = M[0]*vx + M[1]*vy + M[2]*vz;
            data[(cb+1)*N+i] = M[4]*vx + M[5]*vy + M[6]*vz;
            data[(cb+2)*N+i] = M[8]*vx + M[9]*vy + M[10]*vz;
        }
    }
}

// 3x3 矩阵的逆（用于 LCS [4,4]，第4行=eye）
static bool invert_4x4(const float* M, float* inv)
{
    float A[9] = { M[0],M[1],M[2], M[4],M[5],M[6], M[8],M[9],M[10] };
    float det = A[0]*(A[4]*A[8]-A[5]*A[7]) - A[1]*(A[3]*A[8]-A[5]*A[6]) + A[2]*(A[3]*A[7]-A[4]*A[6]);
    if (fabs(det) < 1e-12f) return false;
    float inv3[9];
    inv3[0] = (A[4]*A[8]-A[5]*A[7])/det; inv3[1] = (A[2]*A[7]-A[1]*A[8])/det; inv3[2] = (A[1]*A[5]-A[2]*A[4])/det;
    inv3[3] = (A[5]*A[6]-A[3]*A[8])/det; inv3[4] = (A[0]*A[8]-A[2]*A[6])/det; inv3[5] = (A[2]*A[3]-A[0]*A[5])/det;
    inv3[6] = (A[3]*A[7]-A[4]*A[6])/det; inv3[7] = (A[1]*A[6]-A[0]*A[7])/det; inv3[8] = (A[0]*A[4]-A[1]*A[3])/det;
    for (int r=0;r<3;r++) for(int c=0;c<3;c++) inv[r*4+c] = inv3[r*3+c];
    for (int r=0;r<3;r++) inv[r*4+3] = -(inv3[r*3+0]*M[3] + inv3[r*3+1]*M[7] + inv3[r*3+2]*M[11]);
    inv[3*4+0]=0; inv[3*4+1]=0; inv[3*4+2]=0; inv[3*4+3]=1;
    return true;
}

//-----------------------------------------------------------------------------------------
// 2D UV 空间点-面分类（近似 OCCT FClass2d）：
//   loops = 面各 LOOP 的边界折线（由 coedge pcurve 采样），even-odd 射线法判断 (u,v) 是否在 trim 内。
//   点在边界上（距离 < tol）按「在内」处理（FClass2d 的 ON/边界情况）。
//   多 loop 异或平价自动处理孔洞。
static int classify_uv_inside(double u, double v,
                              const std::vector<std::vector<SPApar_pos> >& loops,
                              double tol) {
    int parity = 0;
    bool onbnd = false;
    for (size_t L = 0; L < loops.size(); ++L) {
        const std::vector<SPApar_pos>& loop = loops[L];
        size_t n = loop.size();
        if (n < 2) continue;
        int cnt = 0;
        for (size_t i = 0; i < n; ++i) {
            size_t j = (i + 1) % n;
            double ax = loop[i].u, ay = loop[i].v;
            double bx = loop[j].u, by = loop[j].v;
            // 边界检测：点到线段最近距离
            {
                double dx = bx - ax, dy = by - ay;
                double L2 = dx*dx + dy*dy;
                double tsh = (L2 > 0) ? ((u-ax)*dx + (v-ay)*dy)/L2 : 0;
                if (tsh < 0) tsh = 0; else if (tsh > 1) tsh = 1;
                double px = ax + dx*tsh, py = ay + dy*tsh;
                double dd = (u-px)*(u-px) + (v-py)*(v-py);
                if (dd <= tol*tol) onbnd = true;
            }
            // even-odd：射线 +u，统计跨越水平线 v 的边界段
            bool aB = ay > v, bB = by > v;
            if (aB != bB) {
                double xint = ax + (bx - ax) * (v - ay) / (by - ay);
                if (xint > u) cnt++;
            }
        }
        parity ^= (cnt & 1);
    }
    // mask 语义与 OCCT BRepTopAdaptor_FClass2d 对齐：仅 IN=1，ON(边界)/OUT=0
    if (onbnd) return 0;
    return parity;
}

// 面各 LOOP 的 coedge 边界采样成 UV 折线（每 loop 闭合成一周）。
// 优先用 coedge 已存储的 pcurve；解析曲面 coedge 常无显式 pcurve（geometry()==NULL），
// 则退化为用 3D 边曲线采样点 + project_param 逆投影回 UV，重建同等的 trim 折线。
static void build_face_uv_loops(FACE* face, const surface* srf, const SPApar_box* pb,
                                std::vector<std::vector<SPApar_pos> >& loops) {
    for (LOOP* loop = face->loop(); loop != NULL; loop = loop->next(PAT_NO_CREATE)) {
        COEDGE* start = loop->start();
        COEDGE* co = start;
        if (co == NULL) continue;
        std::vector<SPApar_pos> verts;
        bool first = true;
        while (co != NULL && (first || co != start)) {
            first = false;
            PCURVE* pc = co->geometry();
            if (pc) {
                pcurve eq = pc->equation();
                SPAinterval rg = eq.param_range();
                double t0 = rg.start_pt(), t1 = rg.end_pt();
                if (t1 < t0) { double tt = t0; t0 = t1; t1 = tt; }
                if (t1 - t0 < 1e-12) { co = co->next(PAT_NO_CREATE); continue; }
                const int S = 64;
                for (int k = 0; k <= S; ++k) {
                    double t = t0 + (t1 - t0) * (double)k / (double)S;
                    SPApar_pos uv;
                    try { eq.eval(t, uv); } catch (...) { continue; }
                    if (!verts.empty()) {
                        double du = uv.u - verts.back().u, dv = uv.v - verts.back().v;
                        if (du*du + dv*dv < 1e-18) { if (k < S) continue; }
                    }
                    verts.push_back(uv);
                }
            } else if (srf) {
                // 无显式 pcurve：从 3D 边曲线重建 UV 边界。
                // 有有效参数盒(pb)时用 newton 投影并钳位；无盒(平面/样条无界，pb==NULL)
                // 用 surface::param()（对平面为解析精确逆映射）。
                EDGE* e = co->edge();
                CURVE* c3 = e ? e->geometry() : NULL;
                if (!c3) { co = co->next(PAT_NO_CREATE); continue; }
                const curve& cu = c3->equation();
                SPAinterval r3;
                try { r3 = cu.param_range(); } catch (...) { co = co->next(PAT_NO_CREATE); continue; }
                double t0 = r3.start_pt(), t1 = r3.end_pt();
                if (t1 < t0) { double tt = t0; t0 = t1; t1 = tt; }
                if (t1 - t0 < 1e-12) { co = co->next(PAT_NO_CREATE); continue; }
                const int S = 64;
                SPApar_pos guess = SPApar_pos(0, 0);
                if (pb) guess = SPApar_pos(0.5 * (pb->low().u + pb->high().u),
                                           0.5 * (pb->low().v + pb->high().v));
                for (int k = 0; k <= S; ++k) {
                    double t = t0 + (t1 - t0) * (double)k / (double)S;
                    SPAposition pos;
                    try { cu.eval(t, pos); } catch (...) { continue; }
                    SPApar_pos uv;
                    if (pb) {
                        if (!project_param(*srf, *pb, pos, uv)) continue;
                    } else {
                        try { uv = srf->param(pos, guess); } catch (...) { continue; }
                    }
                    if (!verts.empty()) {
                        double du = uv.u - verts.back().u, dv = uv.v - verts.back().v;
                        if (du*du + dv*dv < 1e-18) { if (k < S) continue; }
                    }
                    verts.push_back(uv);
                    guess = uv;
                }
            }
            co = co->next(PAT_NO_CREATE);
        }
        if (verts.size() >= 2) loops.push_back(verts);
    }
    // 注意：曾添加「若折线未闭合则降级为用面顶点构建 UV 多边形」的分支，但经实测该
    // 分支会在 B 样条过渡面(rev=0, topo_srf=10)等面上一键命中：把这些面本已有效的
    // pcurve 折线全部替换成按顶点 param() 重建的退化多边形（v 维坍缩），导致 mask 全 0、
    // 面类别一致率从 24/29 掉到 21/29。故已回滚，保持 pcurve/3D 重建采样链原样输出，
    // 交给 even-odd 用首尾闭合段近似。
}

//-----------------------------------------------------------------------------------------

struct CoedgeInfo {
    int id, face_idx, edge_idx, mate_idx, mate_face_idx;
    int orientation;          // 1 = FORWARD(与 edge 同向), 0 = REVERSED
    float origin[3];
    float lcs[16];            // 列主序 [4,4]
    float lcs_inv[16];
    bool valid;
    long long edge_ptr;
};

//-----------------------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    if (argc < 3) { printf("Usage: acis_pipeline.exe <input.step> <out.txt> [input_format]\n"); return 1; }
    const char* inFile = argv[1];
    const char* outFile = argv[2];
    const char* inFmt  = (argc >= 4) ? argv[3] : "STEP";

    const int NU=20, NV=20;

    spa_unlock_result ulock = spa_unlock_products(SPATIAL_LICENSE);
    printf("[pipe] license unlock state: %d\n", ulock.get_state());
    api_start_modeller(0);
    SPAISystemInitGuard initGuard;

    SPAIResult result = SPAI_S_OK;
    ENTITY_LIST* pAcisEntities = NULL;
    {
        SPAIAcisDocument dst;
        SPAIDocument src(inFile);
        src.SetType(inFmt);
        SPAIOptions options;
        SPAIValue representation("BRep+Assembly");
        result &= options.Add(SPAIOptionName::Representation, representation);
        SPAIConverter converter;
        result &= converter.Convert(src, dst);
        dst.GetEntities(pAcisEntities);
    }
    bool ok = (result == SPAI_S_OK);
    printf("[pipe] Convert result: %d (0=OK)\n", (int)result);

    FILE* fp = fopen(outFile, "w");
    if (!fp) { printf("[pipe] ERROR: cannot open %s\n", outFile); ok = false; }

    if (ok && fp && pAcisEntities != NULL)
    {
        // ---------- 拓扑 ----------
        std::vector<EDGE*> edges;
        std::map<long long,int> edge_map;
        std::vector<CoedgeInfo> coedges;
        std::vector<std::vector<int> > face_coedges;

        int coedge_id = 0;
        for (int i = 0; i < pAcisEntities->count(); ++i) {
            ENTITY* ent = (*pAcisEntities)[i];
            if (ent == NULL || !is_BODY(ent)) continue;
            BODY* body = (BODY*)ent;
            for (LUMP* lump = body->lump(); lump != NULL; lump = lump->next(PAT_NO_CREATE)) {
                for (SHELL* shell = lump->shell(); shell != NULL; shell = shell->next(PAT_NO_CREATE)) {
                    for (FACE* face = shell->face_list(); face != NULL; face = face->next_in_list(PAT_NO_CREATE)) {
                        int f_idx = (int)face_coedges.size();
                        face_coedges.push_back(std::vector<int>());
                        for (LOOP* loop = face->loop(); loop != NULL; loop = loop->next(PAT_NO_CREATE)) {
                            COEDGE* start = loop->start();
                            COEDGE* co = start;
                            if (co == NULL) continue;
                            bool first = true;
                            while (co != NULL && (first || co != start)) {
                                first = false;
                                CoedgeInfo ci;
                                ci.id = coedge_id;
                                ci.face_idx = f_idx;
                                ci.mate_idx = -1;
                                ci.mate_face_idx = -1;
                                ci.orientation = (co->sense() == FORWARD) ? 1 : 0;
                                ci.valid = false;
                                for (int k=0;k<16;k++) ci.lcs[k]=0;
                                for (int k=0;k<16;k++) ci.lcs_inv[k]=0;
                                ci.origin[0]=0; ci.origin[1]=0; ci.origin[2]=0;
                                EDGE* e = co->edge();
                                if (e == NULL) { ci.edge_idx=-1; ci.edge_ptr=0; }
                                else {
                                    long long key = (long long)(long*)e;
                                    std::map<long long,int>::iterator it = edge_map.find(key);
                                    if (it == edge_map.end()) { ci.edge_idx=(int)edges.size(); edges.push_back(e); edge_map[key]=ci.edge_idx; }
                                    else ci.edge_idx = it->second;
                                    ci.edge_ptr = key;
                                }
                                coedges.push_back(ci);
                                face_coedges[f_idx].push_back(ci.id);
                                coedge_id++;
                                co = co->next(PAT_NO_CREATE);
                            }
                        }
                    }
                }
            }
        }

        // ---------- mate（按 EDGE 分组；单面边 mate 指向自己）----------
        {
            std::map<int,std::vector<int> > edge_to_ci;
            for (size_t k=0;k<coedges.size();++k) if (coedges[k].edge_idx>=0) edge_to_ci[coedges[k].edge_idx].push_back((int)k);
            for (std::map<int,std::vector<int> >::iterator it=edge_to_ci.begin(); it!=edge_to_ci.end(); ++it) {
                const std::vector<int>& ids = it->second;
                for (size_t a=0;a<ids.size();++a) {
                    int mate=-1;
                    for (size_t b=0;b<ids.size();++b)
                        if (a!=b && coedges[ids[b]].face_idx != coedges[ids[a]].face_idx) { mate=ids[b]; break; }
                    if (mate>=0) { coedges[ids[a]].mate_idx=mate; coedges[ids[a]].mate_face_idx=coedges[mate].face_idx; }
                    else { coedges[ids[a]].mate_idx=ids[a]; coedges[ids[a]].mate_face_idx=coedges[ids[a]].face_idx; }
                }
            }
        }

        fprintf(fp, "CFG faces=%d coedges=%d edges=%d uv=%d %d\n",
                (int)face_coedges.size(), (int)coedges.size(), (int)edges.size(), NU, NV);

        // ---------- FaceGridsGlobal（复用 acis_grid 逻辑，含 mask channel）----------
        std::vector<std::vector<float> > global_grid;
        std::vector<SPApar_box> face_pb;
        std::vector<char> face_pb_ok;
        std::vector<FACE*> face_ptr;
        global_grid.resize(face_coedges.size());
        face_pb.resize(face_coedges.size());
        face_pb_ok.resize(face_coedges.size(), 0);
        face_ptr.resize(face_coedges.size());

        int fgi=0;
        for (int i = 0; i < pAcisEntities->count(); ++i) {
            ENTITY* ent = (*pAcisEntities)[i];
            if (ent == NULL || !is_BODY(ent)) continue;
            BODY* body=(BODY*)ent;
            for (LUMP* lump=body->lump(); lump!=NULL; lump=lump->next(PAT_NO_CREATE))
              for (SHELL* shell=lump->shell(); shell!=NULL; shell=shell->next(PAT_NO_CREATE))
                for (FACE* face=shell->face_list(); face!=NULL; face=face->next_in_list(PAT_NO_CREATE))
                {
                    int fid = fgi++;
                    face_ptr[fid] = face;
                    // 采样域取面 uv_bound()（对平面/规则面与 OCCT UVBounds 一致）；
                    // 但解析曲面（圆柱/锥等）uv_bound() 常返回空盒 [1,0]x[1,0]，
                    // 此时回退到 surface::param_range()（对齐 OCCT GetParamStrict 的物理范围）。
                    // mask 用边界折线 in UV 空间 even-odd（≈OCCT FClass2d）
                    SPApar_box* pb = face->uv_bound();
                    double umin=0,umax=0,vmin=0,vmax=0; bool hr=false;
                    bool pb_valid = (pb && pb->low().u < pb->high().u && pb->low().v < pb->high().v);
                    if (pb_valid) { umin = pb->low().u; umax = pb->high().u; vmin = pb->low().v; vmax = pb->high().v; hr = true; face_pb[fid] = *pb; }
                    else {
                        SURFACE* se = face->geometry();
                        if (se) {
                            const surface& sf = se->equation();
                            try {
                                SPApar_box pr = sf.param_range();
                                if (pr.low().u < pr.high().u && pr.low().v < pr.high().v) {
                                    umin = pr.low().u; umax = pr.high().u; vmin = pr.low().v; vmax = pr.high().v; hr = true; face_pb[fid] = pr;
                                }
                            } catch(...) { hr = false; }
                            if (!hr) {
                                // 某些曲面(样条/扫描) param_range() 为空，但 u/v 各自范围有效
                                try {
                                    SPAinterval uu = sf.param_range_u(), vv = sf.param_range_v();
                                    if (uu.start_pt() < uu.end_pt() && vv.start_pt() < vv.end_pt()) {
                                        umin = uu.start_pt(); umax = uu.end_pt(); vmin = vv.start_pt(); vmax = vv.end_pt();
                                        hr = true; face_pb[fid] = SPApar_box(SPApar_pos(umin,vmin), SPApar_pos(umax,vmax));
                                    }
                                } catch(...) { hr = false; }
                            }
                        }
                    }
                    face_pb_ok[fid] = hr ? 1 : 0;
                    SURFACE* se0 = face->geometry();
                    const surface* srf = se0 ? &(se0->equation()) : NULL;
                    std::vector<std::vector<SPApar_pos> > uv_loops;
                    { const SPApar_box* valid_pb = face_pb_ok[fid] ? &face_pb[fid] : NULL;
                      if (face) build_face_uv_loops(face, srf, valid_pb, uv_loops); }
                    fprintf(fp, "LOOPS face %d loops=%d rev=%d pbok=%d umin=%.6g umax=%.6g vmin=%.6g vmax=%.6g topo_srf=%d\n",
                            fid, (int)uv_loops.size(), (int)(face->sense() == REVERSED), face_pb_ok[fid],
                            umin, umax, vmin, vmax, (se0 ? se0->equation().type() : -1));
                    // 无有效采样域（平面/样条无界面 uv_bound 与 param_range 均空），
                    // 但重建出了 UV 折线：用折线包围盒(+微扩张)作为采样域
                    if (!face_pb_ok[fid] && !uv_loops.empty()) {
                        double bu0=0,bu1=0,bv0=0,bv1=0; bool bhave=false;
                        for (size_t L=0; L<uv_loops.size(); ++L)
                            for (size_t kk=0; kk<uv_loops[L].size(); ++kk) {
                                double u=uv_loops[L][kk].u, v=uv_loops[L][kk].v;
                                if (!bhave){ bu0=bu1=u; bv0=bv1=v; bhave=true; }
                                else { if(u<bu0)bu0=u; if(u>bu1)bu1=u; if(v<bv0)bv0=v; if(v>bv1)bv1=v; }
                            }
                        if (bhave && bu1>bu0 && bv1>bv0) {
                            double du=bu1-bu0, dv=bv1-bv0, m=(du>dv?du:dv);
                            double pad = (m>0)? 0.0*m : 1e-9;
                            umin=bu0-pad; umax=bu1+pad; vmin=bv0-pad; vmax=bv1+pad;
                            face_pb[fid] = SPApar_box(SPApar_pos(umin,vmin), SPApar_pos(umax,vmax));
                            face_pb_ok[fid] = 1;
                        }
                    }
                    // 诊断：uv_bound vs pcurve 包围盒
                    {
                        double bu0=0,bu1=0,bv0=0,bv1=0; bool bhave=false;
                        for (size_t L=0; L<uv_loops.size(); ++L)
                            for (size_t k=0; k<uv_loops[L].size(); ++k) {
                                double u=uv_loops[L][k].u, v=uv_loops[L][k].v;
                                if (!bhave){ bu0=bu1=u; bv0=bv1=v; bhave=true; }
                                else { if(u<bu0)bu0=u; if(u>bu1)bu1=u; if(v<bv0)bv0=v; if(v>bv1)bv1=v; }
                            }
                        if (bhave)
                            fprintf(fp, "DBGUV face %d uvbound=[%.6g,%.6g]x[%.6g,%.6g] pcbbox=[%.6g,%.6g]x[%.6g,%.6g]\n",
                                    fid, umin, umax, vmin, vmax, bu0, bu1, bv0, bv1);
                    }
                    // 诊断折线封闭性/点数
                    {
                        double du=-1, dv=-1;
                        for (size_t L=0; L<uv_loops.size(); ++L)
                            if (uv_loops[L].size() >= 2) {
                                const SPApar_pos& a = uv_loops[L].front();
                                const SPApar_pos& b = uv_loops[L].back();
                                du = hypot(a.u-b.u, a.v-b.v);
                            }
                        fprintf(fp, "DBGLOOP face %d loops=%d lastfirstgap=%.6g\n",
                                fid, (int)uv_loops.size(), du);
                    }
                    std::vector<float> g(9*NU*NV,0.0f);
                    REVBIT fs = face->sense();
                    bool u_reverse = (fs == REVERSED);   // 对齐 OCCT: REVERSED 面 u 轴反向采样
                    double uvd_scale = ((umax - umin) > (vmax - vmin)) ? (umax - umin) : (vmax - vmin);
                    double mask_tol = (uvd_scale > 0) ? 1e-6 * uvd_scale : 1e-6;
                    for (int ii=0;ii<NU;++ii){
                        double u = u_reverse
                                   ? param_strict(NU-1-ii,NU,umin,umax)
                                   : param_strict(ii,NU,umin,umax);
                        for (int jj=0;jj<NV;++jj){
                            double v = param_strict(jj,NV,vmin,vmax);
                            double px=0,py=0,pz=0,n[3]={0,0,0}; int mask=0;
                            if (srf){
                                try { SPApar_pos uv(u,v); SPAposition pt; srf->evaluate(uv,pt,(SPAvector**)NULL,0); px=pt.x();py=pt.y();pz=pt.z(); } catch(...){}
                                normal_cross(*srf,u,v,n);
                                if (fs==REVERSED){ n[0]*=-1; n[1]*=-1; n[2]*=-1; }
                            }
                            // mask：UV 空间 even-odd（≈OCCT FClass2d）；无 pcurve 边界时退回 3D 判断
                            if (srf){
                                if (!uv_loops.empty()) {
                                    mask = classify_uv_inside(u, v, uv_loops, mask_tol);
                                } else {
                                    // 无 pcurve（解析曲面如圆柱/锥）：用 ACIS 原生 3D 点-面包含，
                                    // 参数按正确签名传（uv_guess），边界/外部一律 0（对齐 FClass2d 的 IN-only）
                                    SPAposition pt(px,py,pz);
                                    try {
                                        point_face_containment cont = point_unknown_face;
                                        outcome o = api_point_in_face(pt, face, SPAtransf(), cont,
                                                                      SPApar_pos(u, v));
                                        if (o.ok() && cont == point_inside_face) mask = 1;
                                    } catch(...){}
                                }
                            }
                            int idx = ii*NV+jj;
                            g[0*NU*NV+idx]=(float)px; g[1*NU*NV+idx]=(float)py; g[2*NU*NV+idx]=(float)pz;
                            g[3*NU*NV+idx]=(float)n[0]; g[4*NU*NV+idx]=(float)n[1]; g[5*NU*NV+idx]=(float)n[2];
                            g[6*NU*NV+idx]=(float)mask;
                            g[7*NU*NV+idx]=(float)u; g[8*NU*NV+idx]=(float)v;
                        }
                    }
                    global_grid[fid]=g;
                }
        }
        int num_faces = (int)global_grid.size();
        (void)num_faces;

        // ---------- LCS per coedge ----------
        for (size_t c=0;c<coedges.size();++c){
            CoedgeInfo& ci = coedges[c];
            int eidx = ci.edge_idx;
            int fidx = ci.face_idx;
            ci.valid=false;
            if (eidx<0 || fidx<0 || fidx>=(int)face_ptr.size()) { ci.origin[0]=ci.origin[1]=ci.origin[2]=0; continue; }
            EDGE* e = edges[eidx];
            FACE* face = face_ptr[fidx];
            if (!e || !face) continue;
            CURVE* c3d = e->geometry();
            if (!c3d) { ci.origin[0]=ci.origin[1]=ci.origin[2]=-2000; continue; } // 退化/无3D曲线
            const curve& cu = c3d->equation();
            double u0 = cu.param_range().start_pt(), u1 = cu.param_range().end_pt();
            double e0 = e->param_range().start_pt(), e1 = e->param_range().end_pt();
            if (e1 > e0) { u0 = e0; u1 = e1; }       // 优先用边自身的参数域（subset range）
            if (u1 < u0) { double t = u0; u0 = u1; u1 = t; }
            double tmid = arc_length_mid_param(cu, u0, u1);
            SPAposition p; SPAvector tanv;
            try { cu.eval(tmid, p, tanv); } catch (...) { ci.origin[0]=ci.origin[1]=ci.origin[2]=-2000; continue; }
            ci.origin[0]=(float)p.x(); ci.origin[1]=(float)p.y(); ci.origin[2]=(float)p.z();

            // 法线：将中点投影到面 surface，求该处【面向】法线（du×dv，face REVERSED 取反）
            double n[3]={0,0,0};
            SURFACE* sf = face->geometry();
            const surface* srf = sf ? &(sf->equation()) : NULL;
            REVBIT frv = face->sense();
            if (srf) {
                SPApar_pos par;
                if (fidx < (int)face_pb_ok.size() && face_pb_ok[fidx]
                       && project_param(*srf, face_pb[fidx], p, par)) {
                    double tmp[3]; if (normal_cross(*srf, par.u, par.v, tmp)){
                        n[0]=tmp[0]; n[1]=tmp[1]; n[2]=tmp[2];
                        if (frv==REVERSED){ n[0]*=-1; n[1]*=-1; n[2]*=-1; }
                    }
                }
            }

            // 组装 LCS：u=法线, v=切线(按 coedge 方向), w=u×v
            float mv[3] = { (float)tanv.x(), (float)tanv.y(), (float)tanv.z() };
            if (ci.orientation==0) { mv[0]*=-1; mv[1]*=-1; mv[2]*=-1; }
            float uv_[] = { (float)n[0], (float)n[1], (float)n[2] };
            float vv_[3] = { mv[0], mv[1], mv[2] };
            double un = sqrt(uv_[0]*uv_[0]+uv_[1]*uv_[1]+uv_[2]*uv_[2]);
            double vn = sqrt(vv_[0]*vv_[0]+vv_[1]*vv_[1]+vv_[2]*vv_[2]);
            if (un<1e-10) un=1e-10; if (vn<1e-10) vn=1e-10;
            float uvec[3] = { uv_[0]/(float)un, uv_[1]/(float)un, uv_[2]/(float)un };
            float vvec[3] = { vv_[0]/(float)vn, vv_[1]/(float)vn, vv_[2]/(float)vn };
            double cross2 = pow(uvec[1]*vvec[2]-uvec[2]*vvec[1],2)+pow(uvec[2]*vvec[0]-uvec[0]*vvec[2],2)+pow(uvec[0]*vvec[1]-uvec[1]*vvec[0],2);
            if (un<1e-8 || vn<1e-8 || sqrt(cross2)<1e-6) { ci.origin[0]=ci.origin[1]=ci.origin[2]=0; continue; } // 奇异
            float wvec[3] = { uvec[1]*vvec[2]-uvec[2]*vvec[1], uvec[2]*vvec[0]-uvec[0]*vvec[2], uvec[0]*vvec[1]-uvec[1]*vvec[0] };
            // 列主序 [4,4]
            float M[16]; for(int k=0;k<16;k++) M[k]=0; M[15]=1;
            M[0]=uvec[0]; M[1]=vvec[0]; M[2]=wvec[0]; M[3]=ci.origin[0];
            M[4]=uvec[1]; M[5]=vvec[1]; M[6]=wvec[1]; M[7]=ci.origin[1];
            M[8]=uvec[2]; M[9]=vvec[2]; M[10]=wvec[2]; M[11]=ci.origin[2];
            for(int k=0;k<16;k++) ci.lcs[k]=M[k];
            ci.valid = true;
            invert_4x4(M, ci.lcs_inv);
        }

        // ---------- 构建局部网格并转储（T4 网络输入） ----------
        // FaceGridsLocal [Nc,2,9,20,20]: [c][0]=父面网格→lcs_inv[c], [c][1]=配偶面网格→lcs_inv[mate]
        // CoedgeGridsLocal [Nc,9,20,20]: 配偶面网格→lcs_inv[c]
        // 均按 OCCT 语义：仅当 coedge valid（bool_array）才写入；否则全 0（奇异/退化）
        const size_t Nc = coedges.size();
        const size_t Nf = face_coedges.size();
        const size_t G  = (size_t)NU * NV;
        const size_t C9 = 9;
        // 通道主序存储：[c][层][通道][uv]，用于直接按 from_blob 排列成 [Nc,2,9,20,20] / [Nc,9,20,20]
        std::vector<float> face_local(Nc * 2 * C9 * G, 0.0f);
        std::vector<float> coedge_local(Nc * C9 * G, 0.0f);
        int valid_cnt = 0, local_finite = 0, local_nonzero = 0;
        for (size_t c = 0; c < Nc; ++c) {
            const CoedgeInfo& ci = coedges[c];
            if (!ci.valid) continue;
            ++valid_cnt;
            int fidx = ci.face_idx;
            int m = ci.mate_idx;
            int mfidx = (m >= 0 && m < (int)coedges.size()) ? coedges[m].face_idx : -1;
            bool finite_all = true, any_nonzero = false;

            // 父面（is_face=true：点仿射 + 法线旋转）
            if (fidx >= 0 && fidx < (int)global_grid.size()) {
                std::vector<float> tg(global_grid[fidx]);
                transform_grid_to_local(&tg[0], (int)C9, (int)G, ci.lcs_inv, true);
                memcpy(&face_local[c * (2 * C9 * G) + 0 * (C9 * G)], &tg[0], C9 * G * sizeof(float));
                for (size_t k = 0; k < C9 * G; ++k) {
                    if (!std::isfinite(tg[k])) finite_all = false;
                    if (tg[k] != 0.0f) any_nonzero = true;
                }
            }
            // 配偶面（face_local 右侧用 lcs_inv[mate]；coedge_local 用 lcs_inv[c]）
            if (mfidx >= 0 && mfidx < (int)global_grid.size()) {
                std::vector<float> mg(global_grid[mfidx]);
                const float* Mmate = (m >= 0 && m < (int)coedges.size()) ? coedges[m].lcs_inv : ci.lcs_inv;
                transform_grid_to_local(&mg[0], (int)C9, (int)G, Mmate, true);               // face_local 右侧
                memcpy(&face_local[c * (2 * C9 * G) + 1 * (C9 * G)], &mg[0], C9 * G * sizeof(float));
                std::vector<float> cg(global_grid[mfidx]);
                transform_grid_to_local(&cg[0], (int)C9, (int)G, ci.lcs_inv, true);          // coedge_local
                memcpy(&coedge_local[c * (C9 * G)], &cg[0], C9 * G * sizeof(float));
                for (size_t k = 0; k < C9 * G; ++k) {
                    if (!std::isfinite(mg[k])) finite_all = false;
                    if (mg[k] != 0.0f) any_nonzero = true;
                    if (!std::isfinite(cg[k])) finite_all = false;
                    if (cg[k] != 0.0f) any_nonzero = true;
                }
            }
            if (finite_all) ++local_finite;
            if (any_nonzero) ++local_nonzero;
        }

        // 转储 network 输入（拓扑 + 两个张量，供 acis_eval.exe 端到端前向）
        {
            std::string binFile(outFile);
            std::string::size_type pos = binFile.rfind('.');
            if (pos != std::string::npos) binFile = binFile.substr(0, pos);
            binFile += "_inputs.bin";
            FILE* bf = fopen(binFile.c_str(), "wb");
            if (bf) {
                int32_t magic = 0x54344231;       // "T4B1"
                int32_t hNf = (int32_t)Nf, hNc = (int32_t)Nc;
                fwrite(&magic, sizeof(int32_t), 1, bf);
                fwrite(&hNf, sizeof(int32_t), 1, bf);
                fwrite(&hNc, sizeof(int32_t), 1, bf);
                for (size_t c = 0; c < Nc; ++c) {
                    int32_t fidx = coedges[c].face_idx;
                    int32_t midx = coedges[c].mate_idx;
                    int32_t mfidx = (midx >= 0 && midx < (int)Nc) ? coedges[midx].face_idx : -1;
                    fwrite(&fidx, sizeof(int32_t), 1, bf);
                    fwrite(&midx, sizeof(int32_t), 1, bf);
                    fwrite(&mfidx, sizeof(int32_t), 1, bf);
                }
                fwrite(coedge_local.data(), sizeof(float), coedge_local.size(), bf);
                fwrite(face_local.data(), sizeof(float), face_local.size(), bf);
                fclose(bf);
                printf("[pipe] wrote network inputs: %s (Nf=%d Nc=%d)\n", binFile.c_str(), (int)Nf, (int)Nc);
            } else {
                printf("[pipe] WARN: cannot open %s\n", binFile.c_str());
            }
        }

        // ---------- 输出诊断 ----------
        fprintf(fp, "\n=== FACE COEDGE LISTS ===\n");
        for (size_t f=0; f<face_coedges.size(); ++f){
            fprintf(fp, "Face %d (%d coedges):", (int)f, (int)face_coedges[f].size());
            for (size_t j=0;j<face_coedges[f].size();++j) fprintf(fp, " %d", face_coedges[f][j]);
            fprintf(fp, "\n");
        }
        fprintf(fp, "\n=== COEDGE INFO ===\n");
        fprintf(fp, "id, face, edge, mate, mface, orient, valid, ox, oy, oz, ortho_err\n");
        for (size_t k=0;k<coedges.size();++k){
            const CoedgeInfo& ci=coedges[k];
            double err=-1;
            if (ci.valid){
                double a=ci.lcs[0]*ci.lcs[0]+ci.lcs[1]*ci.lcs[1]+ci.lcs[2]*ci.lcs[2];
                double b=ci.lcs[4]*ci.lcs[4]+ci.lcs[5]*ci.lcs[5]+ci.lcs[6]*ci.lcs[6];
                double c=ci.lcs[8]*ci.lcs[8]+ci.lcs[9]*ci.lcs[9]+ci.lcs[10]*ci.lcs[10];
                double u2=ci.lcs[0]*ci.lcs[4]+ci.lcs[1]*ci.lcs[5]+ci.lcs[2]*ci.lcs[6];
                err = fabs(a-1)+fabs(b-1)+fabs(c-1)+fabs(u2);
            }
            fprintf(fp, "%d, %d, %d, %d, %d, %d, %d, %.6f, %.6f, %.6f, %.6f\n",
                    ci.id, ci.face_idx, ci.edge_idx, ci.mate_idx, ci.mate_face_idx, ci.orientation,
                    ci.valid?1:0, ci.origin[0], ci.origin[1], ci.origin[2], err);
        }
        fprintf(fp, "\n=== GLOBAL GRID MASK ===\n");
        for (size_t f=0; f<global_grid.size(); ++f){
            int ins=0; int off=0;
            for (int k=0;k<NU*NV;++k){ if (global_grid[f][6*NU*NV+k]>0.5f) ins++; else off++; }
            fprintf(fp, "Face %d mask: inside=%d outside=%d\n", (int)f, ins, off);
        }
        fprintf(fp, "\n=== LOCAL GRID STATS ===\n");
        fprintf(fp, "valid coedges=%d/%d, local_grid_finite=%d, local_grid_nonzero=%d\n",
                valid_cnt, (int)coedges.size(), local_finite, local_nonzero);

        fclose(fp);
        printf("[pipe] wrote: %s\n", outFile);
        if (pAcisEntities) { ACIS_DELETE pAcisEntities; pAcisEntities=NULL; }
    }
    else if (fp) fclose(fp);

    api_stop_modeller();
    fflush(stdout);
    printf("[pipe] exit: %s\n", ok?"SUCCESS":"FAILED");
    _exit(ok?0:1);
}