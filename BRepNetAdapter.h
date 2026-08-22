#pragma once
#include "BRepNet.h"
#include "BRepPipeline.h"
#include "UVNet.h"
#include "DebugControl.h"

// 适配器：将 BRepPipeline 的数据转换为 BRepNet 需要的格式
class BRepNetAdapter {
public:
    static std::vector<CoedgeData> extract_coedges(BRepPipeline& pipeline,
                                                     UVNetSurfaceEncoder& surf_enc,
                                                     UVNetCurveEncoder& curve_enc,
                                                     UVNetSurfaceEncoder& surf_enc2
                                                     ) {
        std::vector<CoedgeData> coedges;

        if (!pipeline.FaceGridsLocal.defined() || !pipeline.CoedgeGridsLocal.defined()) {
            ERR_LOG << "[Error] FaceGridsLocal or CoedgeGridsLocal not defined!" << std::endl;
            return coedges;
        }

        int num_coedges = pipeline.FaceGridsLocal.size(0);
        int num_edges = pipeline.unique_edges.Extent();

        // 1. 提取所有面特征
        Tensor face_grids_cloned = pipeline.FaceGridsLocal.clone();
        Tensor all_face_grids = face_grids_cloned.view({num_coedges * 2, 9, 20, 20});

        // 必须clone()！forward()会修改输入张量
        Tensor all_face_features = surf_enc->forward(all_face_grids.clone());  // (num_coedges * 2, 64)
        Tensor Xf = all_face_features.view({num_coedges, 128});

        // std::cout << "\n[UV-Net] Face features Xf: [" << num_coedges << ", 128]" << std::endl;
        // std::cout << "[Verify] Xf[0, :10]: ";
        // for (int j = 0; j < 10; ++j) printf("%.6f ", Xf.at({0, j}));
        // std::cout << std::endl;

        // 2. 提取所有coedge特征
        // 必须clone()！forward()可能会修改输入张量
        // V4: CoedgeGridsLocal is [num_coedges, 9, 20, 20], use surface_encoder2
        Tensor all_coedge_features = surf_enc2->forward(pipeline.CoedgeGridsLocal.clone());  // (num_coedges, 64)

        // std::cout << "\n[UV-Net] Coedge features Xc: [" << num_coedges << ", 64]" << std::endl;
        // std::cout << "[Verify] Xe[0, :10]: ";
        // for (int j = 0; j < 10; ++j) printf("%.6f ", all_edge_features.at({0, j}));
        // std::cout << std::endl;

        // 3. 构建 CoedgeData（从 pipeline.coedges 获取拓扑信息）
        for (size_t c = 0; c < pipeline.coedges.size(); ++c) {
            const auto& c_info = pipeline.coedges[c];

            CoedgeData coedge;
            coedge.coedge_id = c_info.id;
            coedge.parent_face_id = c_info.face_idx;
            coedge.edge_id = c_info.edge_idx;

            // mate_face_id: 通过 mate coedge 获取
            if (c_info.mate_idx >= 0 && c_info.mate_idx < (int)pipeline.coedges.size()) {
                coedge.mate_face_id = pipeline.coedges[c_info.mate_idx].face_idx;
            } else {
                coedge.mate_face_id = c_info.face_idx;  // 如果没有 mate，使用自己的 face
            }

            // 提取 parent face 特征 (前 64 维)
            for (int i = 0; i < 64; ++i) {
                coedge.parent_face_features.push_back(Xf.at({(int64_t)c, i}));
            }

            // 提取 mate face 特征 (后 64 维)
            for (int i = 64; i < 128; ++i) {
                coedge.mate_face_features.push_back(Xf.at({(int64_t)c, i}));
            }

            // 提取 per-coedge 特征
            for (int i = 0; i < 64; ++i) {
                coedge.edge_features.push_back(all_coedge_features.at({(int64_t)c, i}));
            }

            coedges.push_back(coedge);
        }

        return coedges;
    }

    static std::vector<FaceData> extract_faces(BRepPipeline& pipeline) {
        std::vector<FaceData> faces;

        int num_faces = pipeline.unique_faces.Extent();

        // 为每个 face 收集其 coedges
        std::vector<std::vector<int>> face_to_coedges(num_faces);
        for (const auto& c : pipeline.coedges) {
            if (c.face_idx >= 0 && c.face_idx < num_faces) {
                face_to_coedges[c.face_idx].push_back(c.id);
            }
        }

        // 构建 FaceData
        for (int f = 0; f < num_faces; ++f) {
            FaceData face;
            face.face_id = f;
            face.coedge_ids = face_to_coedges[f];
            faces.push_back(face);
        }

        // std::cout << "\n[Topology] Extracted " << faces.size() << " faces" << std::endl;
        // std::cout << "[Debug] Face 0 has " << faces[0].coedge_ids.size() << " coedges" << std::endl;

        return faces;
    }

};