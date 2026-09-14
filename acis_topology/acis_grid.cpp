//=========================================================================================
// acis_grid.cpp
// T3 Step A：ACIS 版 generate_global_face_grid 的「数值面」等价实现。
//
// 目标：对 ACIS 读入的 STEP，逐 face 生成 OCCT BRepPipeline::generate_global_face_grid()
// 的中性近似——9 通道 [num_u=20, num_v=20] 里的前 6 通道(p=xyz, n=xyz)以及 uv 通道(第 7/8 通道)。
// 通道 6(mask, FClass2d) 本轮不做，留到 T3 Step B。
//
// 关键约定（与 BRepPipeline.h:248 对齐，但不改它）：
//   - UVBounds      : 用 ACIS FACE::uv_bound() 的 SPApar_box（OCCT 用 BRepTools::UVBounds）
//   - 采样方向        : 本工具输出「自然序」u∈[min..max]、v∈[min..max]（不随 face 方向翻转），
//                      并把 face 的 sense(FORWARD/REVERSED) 一并输出，供对比脚本做两种约定还原。
//   - 位置 p         : ACIS surface::evaluate(uv, pt, NULL, 0)（ACIS 几何即模型坐标，无需 loc）
//   - 法线 n_cross   : normalize(du × dv)，用有限差分求 du,dv（规避 surface::evaluate 的
//                      高阶导数 vec_array 布局歧义）。自然方向，不随 face sense 翻转，由脚本测符号。
//   - 退化点/奇点     : |n|≈0 时输出 (0,0,0)，与 OCCT SLProps.IsNormalDefined()==false 一致。
//
// 输出文本格式（每 face 一段）：
//   F <fi> sense=<F|R> uv=<u0 u1 v0 v1> c=<cx cy cz>
//   <i> <j> <u> <v> <px> <py> <pz> <nx> <ny> <nz>
//   （i/j 为 0..19 的 u/v 采样序，n 为 n_cross 自然方向）
//
// 用法: acis_grid.exe <input.step> <out.txt> [input_format]
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
#include "kernapi.hxx"
#include "license.hxx"
#include "spa_unlock_result.hxx"
#include "spatial_license.h"

// 拓扑
#include "body.hxx"
#include "lump.hxx"
#include "shell.hxx"
#include "face.hxx"
#include "loop.hxx"
#include "coedge.hxx"

// 几何（surface 基类完整定义在 surdef.hxx）
#include "surdef.hxx"
#include "surface.hxx"
#include "param.hxx"
#include "position.hxx"
#include "vector.hxx"

#include <vector>
#include <cstddef>

//-----------------------------------------------------------------------------------------
// 与 BRepUtils::GetParamStrict 一致：index∈[0,total-1]，min→max（reverse=false）
static double param_strict(int index, int total, double lo, double hi) {
    if (total <= 1) return lo;
    return lo + (hi - lo) * (double)index / (double)(total - 1);
}

// 用 surface::evaluate(nd=1) 求 (u,v) 处自然法线 n = normalize(du × dv)；退化时输出 0
static bool normal_cross(const surface& srf, double u, double v, double out_n[3])
{
    SPAvector d[2];           // d[0]=dP/du, d[1]=dP/dv
    SPAvector* derivs[1];     // 数组长度 = nd = 1
    derivs[0] = d;
    SPAposition pt;
    int nb = 0;
    try {
        nb = srf.evaluate(SPApar_pos(u, v), pt, derivs, 1);
    } catch (...) { nb = 0; }
    if (nb < 1) return false;

    SPAvector n_vec = d[0] * d[1];
    double len2 = n_vec % n_vec;
    if (len2 < 1e-24) { out_n[0]=0.f; out_n[1]=0.f; out_n[2]=0.f; return true; }
    SPAunit_vector n = normalise(n_vec);
    out_n[0] = n.x(); out_n[1] = n.y(); out_n[2] = n.z();
    return true;
}

//-----------------------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    if (argc < 3) {
        printf("Usage: acis_grid.exe <input.step> <out.txt> [input_format]\n");
        return 1;
    }
    const char* inFile = argv[1];
    const char* outFile = argv[2];
    const char* inFmt  = (argc >= 4) ? argv[3] : "STEP";

    const int num_u = 20;
    const int num_v = 20;

    spa_unlock_result ulock = spa_unlock_products(SPATIAL_LICENSE);
    printf("[grid] license unlock state: %d\n", ulock.get_state());

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
    printf("[grid] Convert result: %d (0=OK)\n", (int)result);

    FILE* fp = NULL;
    if (ok && pAcisEntities != NULL) {
        fp = fopen(outFile, "w");
        if (!fp) { printf("[grid] ERROR: cannot open %s\n", outFile); ok = false; }
    } else {
        printf("[grid] ERROR: Convert failed or no entities\n");
        ok = false;
    }

    if (fp) {
        fprintf(fp, "=== ACIS FACE GRID (T3 Step A) ===\n");
        fprintf(fp, "CFG uv=%d %d\n", num_u, num_v);

        int face_idx = 0;
        for (int i = 0; i < pAcisEntities->count(); ++i) {
            ENTITY* ent = (*pAcisEntities)[i];
            if (ent == NULL) continue;
            if (!is_BODY(ent)) continue;

            BODY* body = (BODY*)ent;
            for (LUMP* lump = body->lump(); lump != NULL; lump = lump->next(PAT_NO_CREATE)) {
                for (SHELL* shell = lump->shell(); shell != NULL; shell = shell->next(PAT_NO_CREATE)) {
                    for (FACE* face = shell->face_list(); face != NULL; face = face->next_in_list(PAT_NO_CREATE)) {
                        // --- UVBounds ---
                        double umin=0, umax=0, vmin=0, vmax=0;
                        bool have_range = false;
                        SPApar_box* pb = face->uv_bound();
                        if (pb) {
                            umin = pb->low().u; umax = pb->high().u;
                            vmin = pb->low().v; vmax = pb->high().v;
                            have_range = true;
                        } else {
                            SURFACE* se = face->geometry();
                            if (se) {
                                try {
                                    SPApar_box pr = se->equation().param_range();
                                    umin = pr.low().u; umax = pr.high().u;
                                    vmin = pr.low().v; vmax = pr.high().v;
                                    have_range = true;
                                } catch (...) { have_range = false; }
                            }
                        }
                        if (!have_range) {
                            fprintf(fp, "F %d sense=U uv=0 0 0 0 c=0 0 0\n", face_idx);
                            fprintf(fp, "# NO_UV_RANGE (face skipped)\n");
                            face_idx++;
                            continue;
                        }

                        SURFACE* se = face->geometry();
                        const surface* srf = se ? &(se->equation()) : NULL;
                        bool has_surf = (srf != NULL);

                        REVBIT fs = face->sense();
                        const char* sense_txt = (fs == REVERSED) ? "R" : "F";

                        // 生成数据到内存（先算 centroid，再写头）
                        std::vector<std::vector<double> > rows; // [0]=i,[1]=j,[2]=u,[3]=v,[4..6]=p,[7..9]=n
                        rows.reserve(num_u*num_v);
                        double cx=0, cy=0, cz=0;
                        int cnt=0;
                        for (int ii = 0; ii < num_u; ++ii) {
                            double u = param_strict(ii, num_u, umin, umax);
                            for (int jj = 0; jj < num_v; ++jj) {
                                double v = param_strict(jj, num_v, vmin, vmax);
                                double px=0,py=0,pz=0,nx=0,ny=0,nz=0;
                                if (has_surf) {
                                    try {
                                        SPApar_pos uv(u, v);
                                        SPAposition pt;
                                        srf->evaluate(uv, pt, (SPAvector**)NULL, 0);
                                        px = pt.x(); py = pt.y(); pz = pt.z();
                                    } catch (...) { px=0;py=0;pz=0; }
                                    double n[3]={0,0,0};
                                    normal_cross(*srf, u, v, n);
                                    nx=n[0]; ny=n[1]; nz=n[2];
                                }
                                std::vector<double> r;
                                r.push_back((double)ii); r.push_back((double)jj);
                                r.push_back(u); r.push_back(v);
                                r.push_back(px); r.push_back(py); r.push_back(pz);
                                r.push_back(nx); r.push_back(ny); r.push_back(nz);
                                rows.push_back(r);
                                cx += px; cy += py; cz += pz; cnt++;
                            }
                        }
                        if (cnt > 0) { cx /= cnt; cy /= cnt; cz /= cnt; }

                        fprintf(fp, "F %d sense=%s uv=%.9f %.9f %.9f %.9f c=%.9f %.9f %.9f\n",
                                face_idx, sense_txt, umin, umax, vmin, vmax, cx, cy, cz);
                        for (size_t k = 0; k < rows.size(); ++k) {
                            const std::vector<double>& r = rows[k];
                            fprintf(fp, "%d %d %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f\n",
                                    (int)r[0], (int)r[1], r[2], r[3],
                                    r[4], r[5], r[6], r[7], r[8], r[9]);
                        }
                        fprintf(fp, "# end_face_%d\n", face_idx);
                        face_idx++;
                    }
                }
            }
        }
        fclose(fp);
        printf("[grid] done: face_idx=%d wrote %s\n", face_idx, outFile);
    }

    api_stop_modeller();
    fflush(stdout);
    printf("[grid] exit: %s\n", ok ? "SUCCESS" : "FAILED");
    _exit(ok ? 0 : 1);
}