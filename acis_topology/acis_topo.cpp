//=========================================================================================
// acis_topo.cpp
// T2 拓扑 A/B 对比：把 ACIS 读入的 STEP 拓扑导出为「中立结构」文本，供与 OCCT 侧
// (before_acis/cpp_topology/*_result_topology.txt *_result_face_coedges.txt) 逐项对比。
//
// 由于 ACIS 与 OCCT 的 face/coedge 枚举顺序未必一致，本工具为每条 coedge 附上
// 几何锚点（中点坐标）与方向(sense)，对比时按几何位置对齐，再核对：
//   - 每 face 的 coedge 序列（面内环序）
//   - 每条 coedge 的父 face、方向、mate coedge、共享 edge
//   - 面与面的邻接（通过共享 edge / 共享 coedge 中点）
//
// 用法: acis_topo.exe <input.step> <out.txt> [input_format]
//=========================================================================================

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

// ACIS 内核
#include "savres.hxx"
#include "fileinfo.hxx"
#include "lists.hxx"
#include "kernapi.hxx"

// 拓扑遍历
#include "body.hxx"
#include "lump.hxx"
#include "shell.hxx"
#include "face.hxx"
#include "edge.hxx"
#include "vertex.hxx"
#include "coedge.hxx"
#include "loop.hxx"

#include "license.hxx"
#include "spa_unlock_result.hxx"
#include "spatial_license.h"

#include <vector>
#include <map>
#include <string>
#include <cmath>
#include <algorithm>

//-----------------------------------------------------------------------------------------

struct CoedgeEntry {
    int id;              // 全局 coedge id（遍历顺序）
    int face_idx;        // 父 face 索引
    int edge_idx;        // 该 coedge 引用的 edge 索引
    int mate_coedge_id;  // 共享同一 EDGE 的另一条 coedge（无则 -1）
    int mate_face_idx;
    int sense;           // +1 = FORWARD(与edge同向), -1 = REVERSED(与edge反向)
    double midx, midy, midz;   // edge 中点坐标（几何锚点）
    long long edge_ptr;        // EDGE* 指纹，用于识别同一条 edge
};

//-----------------------------------------------------------------------------------------

static bool nearly(double a, double b, double tol = 1e-6) {
    return std::fabs(a - b) < tol;
}

// 匹配给定 EDGE 的既有 edge 索引；匹配失败返回 -1
static inline EDGE* edge_from_coedge(COEDGE* co) { return co ? co->edge() : NULL; }

//-----------------------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    if (argc < 3) {
        printf("Usage: acis_topo.exe <input.step> <out.txt> [input_format]\n");
        return 1;
    }

    const char* inFile = argv[1];
    const char* outFile = argv[2];
    const char* inFmt  = (argc >= 4) ? argv[3] : "STEP";

    spa_unlock_result ulock = spa_unlock_products(SPATIAL_LICENSE);
    printf("[topo] license unlock state: %d\n", ulock.get_state());

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
    printf("[topo] Convert result: %d (0=OK)\n", (int)result);

    if (ok && pAcisEntities != NULL)
    {
        std::vector<EDGE*> edges;            // 去重 edge 表
        std::map<long long, int> edge_map;   // EDGE* -> edge_idx
        std::vector<CoedgeEntry> coedges;
        std::vector<int> face_flag;          // 占位，未用
        std::vector<std::vector<int> > face_coedges; // face_idx -> coedge id 列表
        std::vector<int> face_loop_count;

        int coedge_id = 0;

        for (int i = 0; i < pAcisEntities->count(); ++i)
        {
            ENTITY* ent = (*pAcisEntities)[i];
            if (ent == NULL) continue;
            if (!is_BODY(ent)) continue;

            BODY* body = (BODY*)ent;
            for (LUMP* lump = body->lump(); lump != NULL; lump = lump->next(PAT_NO_CREATE))
            {
                for (SHELL* shell = lump->shell(); shell != NULL; shell = shell->next(PAT_NO_CREATE))
                {
                    for (FACE* face = shell->face_list(); face != NULL; face = face->next_in_list(PAT_NO_CREATE))
                    {
                        int f_idx = (int)face_coedges.size();
                        face_coedges.push_back(std::vector<int>());

                        int loop_count = 0;
                        for (LOOP* loop = face->loop(); loop != NULL; loop = loop->next(PAT_NO_CREATE))
                        {
                            ++loop_count;
                            COEDGE* start = loop->start();
                            COEDGE* co = start;
                            if (co == NULL) continue;
                            bool first = true;
                            while (co != NULL && (first || co != start))
                            {
                                first = false;

                                CoedgeEntry ce;
                                ce.id = coedge_id;
                                ce.face_idx = f_idx;
                                ce.mate_coedge_id = -1;
                                ce.mate_face_idx = -1;

                                EDGE* e = edge_from_coedge(co);
                                if (e == NULL) { ce.edge_idx = -1; ce.sense = 0; ce.midx=0; ce.midy=0; ce.midz=0; ce.edge_ptr=0; }
                                else {
                                    ce.edge_ptr = (long long)(long*)e;
                                    long long key = ce.edge_ptr;
                                    std::map<long long,int>::iterator it = edge_map.find(key);
                                    if (it == edge_map.end()) { ce.edge_idx = (int)edges.size(); edges.push_back(e); edge_map[key]=ce.edge_idx; }
                                    else ce.edge_idx = it->second;

                                    ce.sense = (co->sense() == FORWARD) ? +1 : -1;
                                    SPAposition mid;
                                    try { mid = e->mid_pos(); } catch (...) { mid = e->start_pos(); }
                                    ce.midx = mid.x(); ce.midy = mid.y(); ce.midz = mid.z();
                                }

                                coedges.push_back(ce);
                                face_coedges[f_idx].push_back(ce.id);
                                coedge_id++;

                                co = co->next(PAT_NO_CREATE);
                            }
                        }
                        face_loop_count.push_back(loop_count);
                    }
                }
            }
        }

        // ---- 建立 mate（按 EDGE 分组）----
        {
            std::map<int, std::vector<int> > edge_to_coedges;
            for (size_t k = 0; k < coedges.size(); ++k) {
                if (coedges[k].edge_idx >= 0)
                    edge_to_coedges[coedges[k].edge_idx].push_back((int)k);
            }
            for (std::map<int,std::vector<int> >::iterator it = edge_to_coedges.begin(); it != edge_to_coedges.end(); ++it) {
                const std::vector<int>& ids = it->second;
                for (size_t a = 0; a < ids.size(); ++a) {
                    for (size_t b = 0; b < ids.size(); ++b) {
                        if (a != b && coedges[ids[a]].face_idx != coedges[ids[b]].face_idx) {
                            coedges[ids[a]].mate_coedge_id = ids[b];
                            coedges[ids[a]].mate_face_idx = coedges[ids[b]].face_idx;
                            break;
                        }
                    }
                    if (coedges[ids[a]].mate_coedge_id < 0) {
                        // 单face边：mate 指向自己（对齐 OCCT 语义）
                        coedges[ids[a]].mate_coedge_id = ids[a];
                        coedges[ids[a]].mate_face_idx = coedges[ids[a]].face_idx;
                    }
                }
            }
        }

        // ---- 写中立结构 ----
        FILE* fp = fopen(outFile, "w");
        if (fp)
        {
            fprintf(fp, "=== TOPOLOGY INFORMATION (ACIS) ===\n");
            fprintf(fp, "Num Coedges: %d\n", (int)coedges.size());
            fprintf(fp, "Num Faces: %d\n", (int)face_coedges.size());
            fprintf(fp, "Num Edges: %d\n", (int)edges.size());

            fprintf(fp, "\n=== FACE COEDGE LISTS ===\n");
            for (size_t f = 0; f < face_coedges.size(); ++f) {
                fprintf(fp, "Face %d (%d coedges, %d loops):", (int)f, (int)face_coedges[f].size(), face_loop_count[f]);
                for (size_t j = 0; j < face_coedges[f].size(); ++j)
                    fprintf(fp, " %d", face_coedges[f][j]);
                fprintf(fp, "\n");
            }

            fprintf(fp, "\n=== COEDGE INFO ===\n");
            fprintf(fp, "coedge_id, parent_face_id, edge_id, mate_coedge_id, mate_face_id, sense(+1=FWD), midx, midy, midz\n");
            for (size_t k = 0; k < coedges.size(); ++k) {
                const CoedgeEntry& c = coedges[k];
                fprintf(fp, "%d, %d, %d, %d, %d, %d, %.6f, %.6f, %.6f\n",
                        c.id, c.face_idx, c.edge_idx, c.mate_coedge_id, c.mate_face_idx, c.sense,
                        c.midx, c.midy, c.midz);
            }
            fclose(fp);
            printf("[topo] wrote: %s\n", outFile);
        }
        else {
            printf("[topo] ERROR: cannot open %s for writing\n", outFile);
            ok = false;
        }

        if (pAcisEntities != NULL) { ACIS_DELETE pAcisEntities; pAcisEntities = NULL; }
    }

    api_stop_modeller();
    fflush(stdout);
    printf("[topo] done: %s\n", ok ? "SUCCESS" : "FAILED");
    _exit(ok ? 0 : 1);
}