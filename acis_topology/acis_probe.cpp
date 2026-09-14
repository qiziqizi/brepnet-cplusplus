//=========================================================================================
// acis_probe.cpp
// T1 最小探针：验证 InterOp(Connect 接口) 能把 STEP 读成 ACIS 实体，
// 并完整遍历 BODY->LUMP->SHELL->FACE->LOOP->COEDGE->EDGE 拓扑，打印各级计数。
// 这是 OCC->ACIS 替换的第一道关口。
//
// 参考官方样例: D:\Spatial\Interop_2025.1.0.1\samples\connectacis\AcisImport.cpp
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
#include "coedge.hxx"
#include "loop.hxx"

#include "license.hxx"
#include "spa_unlock_result.hxx"
#include "spatial_license.h"

//-----------------------------------------------------------------------------------------

struct TopoCounts
{
    int bodies, faces, loops, coedges, edges;
    TopoCounts() : bodies(0), faces(0), loops(0), coedges(0), edges(0) {}
};

//-----------------------------------------------------------------------------------------

static void CountTopology(const ENTITY_LIST& entities, TopoCounts& c)
{
    for (int i = 0; i < entities.count(); ++i)
    {
        ENTITY* ent = entities[i];
        if (ent == NULL) continue;
        if (!is_BODY(ent)) continue;

        BODY* body = (BODY*)ent;
        ++c.bodies;

        for (LUMP* lump = body->lump(); lump != NULL; lump = lump->next(PAT_NO_CREATE))
        {
            for (SHELL* shell = lump->shell(); shell != NULL; shell = shell->next(PAT_NO_CREATE))
            {
                for (FACE* face = shell->face_list(); face != NULL; face = face->next_in_list(PAT_NO_CREATE))
                {
                    ++c.faces;

                    for (LOOP* loop = face->loop(); loop != NULL; loop = loop->next(PAT_NO_CREATE))
                    {
                        ++c.loops;

                        // 注意：LOOP 内的 COEDGE 是环状链表（沿 next 回到 start）。
                        COEDGE* start = loop->start();
                        COEDGE* co = start;
                        if (co != NULL)
                        {
                            bool first = true;
                            while (co != NULL && (first || co != start))
                            {
                                first = false;
                                ++c.coedges;
                                EDGE* edge = co->edge();
                                if (edge != NULL) ++c.edges;
                                co = co->next(PAT_NO_CREATE);
                            }
                        }
                    }
                }
            }
        }
    }
}

//-----------------------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        printf("Usage: acis_probe.exe <input.step> [input_format]\n");
        return 1;
    }

    const char* inFile = argv[1];
    const char* inFmt  = (argc >= 3) ? argv[2] : "STEP";

    // ---- 解锁许可证 ----
    spa_unlock_result ulock = spa_unlock_products(SPATIAL_LICENSE);
    printf("[probe] license unlock state: %d\n", ulock.get_state());

    // ---- 启动 ACIS 内核 ----
    api_start_modeller(0);

    // ---- InterOp 初始化（进程内仅一次） ----
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
    printf("[probe] Convert result: %d (0=OK)\n", (int)result);

    if (pAcisEntities != NULL)
    {
        TopoCounts c;
        CountTopology(*pAcisEntities, c);
        printf("[probe] entities_in_list=%d\n", pAcisEntities->count());
        printf("[result] bodies=%d faces=%d loops=%d coedges=%d edges=%d\n",
               c.bodies, c.faces, c.loops, c.coedges, c.edges);
    }

    // ---- 清理 ----
    if (pAcisEntities != NULL)
    {
        ACIS_DELETE pAcisEntities;
        pAcisEntities = NULL;
    }

    api_stop_modeller();
    printf("[probe] done: %s\n", ok ? "SUCCESS" : "FAILED");

    // InterOp 在进程退出时的全局析构在 stdout 被重定向时会挂起（实测）。
    // 功能已全部完成，用 _exit 立即终止，跳过这部分析构器。
    fflush(stdout);
    _exit(ok ? 0 : 1);
}