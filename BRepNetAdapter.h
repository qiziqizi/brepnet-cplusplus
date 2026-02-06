#pragma once
#include "BRepNet.h"
#include "BRepPipeline.h"
#include "UVNet.h"

// 适配器：将 BRepPipeline 的数据转换为 BRepNet 需要的格式
class BRepNetAdapter {
public:
    static std::vector<CoedgeData> extract_coedges(BRepPipeline& pipeline,
                                                     UVNetSurfaceEncoder& surf_enc,
                                                     UVNetCurveEncoder& curve_enc) {
        std::vector<CoedgeData> coedges;

        if (!pipeline.FaceGridsLocal.defined() || !pipeline.EdgeGridsLocal.defined()) {
            std::cerr << "[Error] FaceGridsLocal or EdgeGridsLocal not defined!" << std::endl;
            return coedges;
        }

        int num_coedges = pipeline.FaceGridsLocal.size(0);
        int num_edges = pipeline.EdgeGridsLocal.size(0);

        // 1. 提取所有面特征
        Tensor face_grids_cloned = pipeline.FaceGridsLocal.clone();
        Tensor all_face_grids = face_grids_cloned.view({num_coedges * 2, 9, 10, 10});
        Tensor all_face_features = surf_enc->forward(all_face_grids);  // (num_coedges * 2, 64)
        Tensor Xf = all_face_features.view({num_coedges, 128});

        std::cout << "\n[UV-Net] Face features Xf: [" << num_coedges << ", 128]" << std::endl;
        std::cout << "[Verify] Xf[0, :10]: ";
        for (int j = 0; j < 10; ++j) printf("%.6f ", Xf.at({0, j}));
        std::cout << std::endl;

        // 2. 提取所有边特征
        Tensor all_edge_features = curve_enc->forward(pipeline.EdgeGridsLocal);  // (num_edges, 64)

        std::cout << "\n[UV-Net] Edge features Xe: [" << num_edges << ", 64]" << std::endl;
        std::cout << "[Verify] Xe[0, :10]: ";
        for (int j = 0; j < 10; ++j) printf("%.6f ", all_edge_features.at({0, j}));
        std::cout << std::endl;

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

            // 提取 edge 特征
            if (coedge.edge_id >= 0 && coedge.edge_id < num_edges) {
                for (int i = 0; i < 64; ++i) {
                    coedge.edge_features.push_back(all_edge_features.at({coedge.edge_id, i}));
                }
            } else {
                for (int i = 0; i < 64; ++i) {
                    coedge.edge_features.push_back(0.0f);
                }
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

        std::cout << "\n[Topology] Extracted " << faces.size() << " faces" << std::endl;
        std::cout << "[Debug] Face 0 has " << faces[0].coedge_ids.size() << " coedges" << std::endl;

        return faces;
    }

    static std::vector<EdgeData> extract_edges(BRepPipeline& pipeline) {
        std::vector<EdgeData> edges;

        int num_edges = pipeline.unique_edges.Extent();

        // 为每个 edge 收集其 coedges
        std::vector<std::vector<int>> edge_to_coedges(num_edges);
        for (const auto& c : pipeline.coedges) {
            if (c.edge_idx >= 0 && c.edge_idx < num_edges) {
                edge_to_coedges[c.edge_idx].push_back(c.id);
            }
        }

        // 构建 EdgeData
        for (int e = 0; e < num_edges; ++e) {
            EdgeData edge;
            edge.edge_id = e;
            edge.coedge_ids = edge_to_coedges[e];
            edges.push_back(edge);
        }

        std::cout << "[Topology] Extracted " << edges.size() << " edges" << std::endl;

        return edges;
    }
};
