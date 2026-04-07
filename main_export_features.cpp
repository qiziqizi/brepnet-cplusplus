// 调试输出开关：通过 DebugControl.h 统一管理
// 通过命令行参数控制，永远不需要注释/取消注释代码

#include "DebugControl.h"
#include "BRepNet.h"
#include "BRepNetAdapter.h"
#include "BRepPipeline.h"
#include "FeatureMapExporter.h"
#include "EdgeInputExporter.h"
#include "OutputLogger.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <Windows.h>
#include <psapi.h>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <numeric>
#include <map>

namespace fs = std::filesystem;

/**
 * Feature Map 导出版本的推理脚本
 * 目的：在推理过程中导出每一层的中间结果到 cpp_feature_maps/
 * 格式：与 cpp_logits/cpp_probs 完全一致
 */

// 获取目录下所有 STEP 文件
std::vector<std::string> get_step_files(const std::string& dir_path) {
    std::vector<std::string> step_files;
    try {
        for (const auto& entry : fs::directory_iterator(dir_path)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (ext == ".step" || ext == ".stp") {
                    step_files.push_back(entry.path().string());
                }
            }
        }
    } catch (const fs::filesystem_error& e) {
        ERR_LOG << "[Error] Cannot read directory: " << e.what() << std::endl;
    }
    std::sort(step_files.begin(), step_files.end());
    return step_files;
}

// 检查该文件是否已经处理过
bool is_file_already_processed(const std::string& base_name) {
    fs::path logits_file = fs::path("cpp_logits") / (base_name + ".logits");
    bool exists = fs::exists(logits_file);
    DBG_CERR << "[DEBUG] Checking: " << logits_file.string() << " -> " << (exists ? "EXISTS" : "NOT FOUND") << std::endl;
    return exists;
}

// 对单个 STEP 文件进行推理并导出中间层
void run_inference_with_export(const std::string& step_file,
                               std::shared_ptr<BRepNetImpl> model,
                               FeatureMapExporter& exporter) {
    fs::path step_path(step_file);
    std::string base_name = step_path.stem().string();

    // 设置当前文件（控制调试开关）
    DebugControl::instance().setCurrentFile(base_name);

    // 如果指定了 target，跳过不匹配的文件
    const auto& dc = DebugControl::instance();
    if (!dc.targets.empty() && std::find(dc.targets.begin(), dc.targets.end(), base_name) == dc.targets.end()) {
        return;  // 静默跳过非目标文件
    }

    // 检查是否已经处理过
    if (is_file_already_processed(base_name)) {
        INFO_LOG << base_name << " [SKIPPED]" << std::flush;
        return;
    }

    INFO_LOG << base_name << std::flush;

    // ========================================================================
    // 1. 数据预处理
    // ========================================================================
    BRepPipeline pipeline;
    if (!pipeline.process(step_file)) {
        ERR_LOG << "[Error] Processing failed: " << base_name << std::endl;
        return;
    }

    int num_coedges = (int)pipeline.coedges.size();
    int num_faces = (int)pipeline.unique_faces.Extent();
    int num_edges = (int)pipeline.unique_edges.Extent();

    DBG_LOG << "[Topology] " << num_coedges << " coedges, "
            << num_faces << " faces, "
            << num_edges << " edges" << std::endl;

    // ========================================================================
    // [TOPOLOGY EXPORT] 导出拓扑信息用于调试
    // ========================================================================
    if (EXPORT_ENABLED) {
        fs::create_directories("cpp_topology");
        std::string topology_file = "cpp_topology/" + base_name + "_topology.txt";
        std::ofstream topo_out(topology_file);

        if (topo_out.is_open()) {
            topo_out << "=== TOPOLOGY INFORMATION ===" << std::endl;
            topo_out << "Num Coedges: " << num_coedges << std::endl;
            topo_out << "Num Faces: " << num_faces << std::endl;
            topo_out << "Num Edges: " << num_edges << std::endl;
            topo_out << std::endl;

            // 导出每个coedge的基本信息
            topo_out << "=== COEDGE INFO ===" << std::endl;
            topo_out << "coedge_id, parent_face_id, edge_id, mate_coedge_id, mate_face_id, orientation" << std::endl;
            for (int i = 0; i < num_coedges; ++i) {
                const auto& c = pipeline.coedges[i];
                int mate_face = (c.mate_idx >= 0) ? pipeline.coedges[c.mate_idx].face_idx : -1;
                topo_out << c.id << ", " << c.face_idx << ", " << c.edge_idx << ", "
                         << c.mate_idx << ", " << mate_face << ", " << c.orientation << std::endl;
            }
            topo_out << std::endl;

            topo_out.close();
            DBG_LOG << "[Topology] Exported to " << topology_file << std::endl;
        }
    }

    // ========================================================================
    // 2. 导出UV Grid（送入UVNet之前的原始数据）
    // ========================================================================
    DBG_LOG << "[UV Grid Export] Exporting raw UV grids before UVNet..." << std::endl;

    if (!pipeline.FaceGridsLocal.defined()) {
        ERR_LOG << "[Error] FaceGridsLocal not defined!" << std::endl;
        return;
    }

    // 提取Face UV Grids (Coedge格式: [num_coedges*2, 9, 10, 10])
    int num_coedges_topo = (int)pipeline.coedges.size();
    // 必须clone()，避免数据污染。后续操作可能修改张量，影响原始数据
    Tensor all_face_grids = pipeline.FaceGridsLocal.clone().view({num_coedges_topo * 2, 9, 10, 10});  // [380, 9, 10, 10]

    // 展平为 [380, 900]
    Tensor flattened_face_grids = all_face_grids.view({num_coedges_topo * 2, 9 * 10 * 10});

    if (EXPORT_ENABLED) {
        // 创建输出目录
        fs::create_directories("cpp_uv_grids");

        DBG_LOG << "  all_face_grids shape: [" << all_face_grids.size(0) << ", "
                << all_face_grids.size(1) << ", " << all_face_grids.size(2) << ", "
                << all_face_grids.size(3) << "]" << std::endl;

        // 导出Coedge格式的Face UV Grids
        std::string coedge_uv_file = "cpp_uv_grids/coedge_face_uv_grids_" + base_name + ".txt";
        std::ofstream coedge_uv_out(coedge_uv_file, std::ios::out);
        coedge_uv_out << std::scientific << std::setprecision(20);

        for (int i = 0; i < flattened_face_grids.size(0); ++i) {
            for (int j = 0; j < flattened_face_grids.size(1); ++j) {
                if (j > 0) coedge_uv_out << " ";
                coedge_uv_out << flattened_face_grids.at({i, j});
            }
            coedge_uv_out << "\n";
        }
        coedge_uv_out.close();
        DBG_LOG << "  Exported: " << coedge_uv_file << " (" << flattened_face_grids.size(0) << " rows)" << std::endl;

        // 还需要导出Edge UV Grids
        if (pipeline.EdgeGridsLocal.defined()) {
            int num_edges_topo = (int)pipeline.EdgeGridsLocal.size(0);
            // 必须clone()，避免数据污染
            Tensor flattened_edge_grids = pipeline.EdgeGridsLocal.clone().view({num_edges_topo, 13 * 10});

            std::string edge_uv_file = "cpp_uv_grids/edge_uv_grids_" + base_name + ".txt";
            std::ofstream edge_uv_out(edge_uv_file, std::ios::out);
            edge_uv_out << std::scientific << std::setprecision(20);

            for (int i = 0; i < flattened_edge_grids.size(0); ++i) {
                for (int j = 0; j < flattened_edge_grids.size(1); ++j) {
                    if (j > 0) edge_uv_out << " ";
                    edge_uv_out << flattened_edge_grids.at({i, j});
                }
                edge_uv_out << "\n";
            }
            edge_uv_out.close();
            DBG_LOG << "  Exported: " << edge_uv_file << " (" << flattened_edge_grids.size(0) << " rows)" << std::endl;
        }
    }

    // ========================================================================
    // 3. 提取 UVNet 特征（使用BRepNetAdapter）
    // ========================================================================
    DBG_LOG << "[Feature Extraction] Running UVNet..." << std::endl;

    auto coedges = BRepNetAdapter::extract_coedges(pipeline, model->surf_enc, model->curve_enc);
    auto faces = BRepNetAdapter::extract_faces(pipeline);
    auto edges = BRepNetAdapter::extract_edges(pipeline);

    // ========================================================================
    // 导出 Coedge 拼接后的特征（用于调试 GNN 层）
    // ========================================================================
    if (DebugControl::instance().shouldDebug()) {
        DBG_LOG << "\n[DEBUG] Coedge Feature Concatenation for " << base_name << std::endl;
        DBG_LOG << "[DEBUG] Total coedges: " << coedges.size() << std::endl;
        for (size_t c = 0; c < std::min(size_t(5), coedges.size()); ++c) {
            const auto& ce = coedges[c];
            DBG_LOG << "\n[DEBUG] Coedge " << c << ":" << std::endl;
            DBG_LOG << "  parent_face_features (first 10): ";
            for (int i = 0; i < std::min(10, (int)ce.parent_face_features.size()); ++i) {
                DBG_PRINTF("%.6f ", ce.parent_face_features[i]);
            }
            DBG_LOG << std::endl;
            DBG_LOG << "  mate_face_features (first 10): ";
            for (int i = 0; i < std::min(10, (int)ce.mate_face_features.size()); ++i) {
                DBG_PRINTF("%.6f ", ce.mate_face_features[i]);
            }
            DBG_LOG << std::endl;
            DBG_LOG << "  edge_features (first 10): ";
            for (int i = 0; i < std::min(10, (int)ce.edge_features.size()); ++i) {
                DBG_PRINTF("%.6f ", ce.edge_features[i]);
            }
            DBG_LOG << std::endl;
        }
    }

    // ========================================================================
    // 导出按原始Face ID组织的UV Grid（方便直接对比Face 25）
    // ========================================================================
    if (EXPORT_ENABLED) {
        DBG_LOG << "[UV Grid Export] Creating per-face UV grids..." << std::endl;

        // 为每个Face收集其所有coedges的UV Grid并取平均/第一个
        // 这里使用第一个coedge的parent_face grid作为该face的代表
        std::vector<std::vector<float>> face_uv_grids(num_faces);

        for (const auto& coedge_info : pipeline.coedges) {
            int face_id = coedge_info.face_idx;
            int coedge_id = coedge_info.id;

            if (face_id >= 0 && face_id < num_faces && face_uv_grids[face_id].empty()) {
                // 该face还没有记录UV grid，使用这个coedge的parent_face
                int row_idx = coedge_id * 2;  // parent_face行索引

                for (int c = 0; c < 9; ++c) {
                    for (int i = 0; i < 10; ++i) {
                        for (int j = 0; j < 10; ++j) {
                            face_uv_grids[face_id].push_back(all_face_grids.at({row_idx, c, i, j}));
                        }
                    }
                }
            }
        }

        // 导出按Face ID组织的UV Grid
        std::string face_uv_file = "cpp_uv_grids/face_uv_grids_" + base_name + ".txt";
        std::ofstream face_uv_out(face_uv_file, std::ios::out);
        face_uv_out << std::scientific << std::setprecision(20);

        for (int f = 0; f < num_faces; ++f) {
            if (face_uv_grids[f].empty()) {
                // 该face没有coedge，填充0
                for (int i = 0; i < 900; ++i) {
                    if (i > 0) face_uv_out << " ";
                    face_uv_out << 0.0f;
                }
            } else {
                for (size_t i = 0; i < face_uv_grids[f].size(); ++i) {
                    if (i > 0) face_uv_out << " ";
                    face_uv_out << face_uv_grids[f][i];
                }
            }
            face_uv_out << "\n";
        }
        face_uv_out.close();
        DBG_LOG << "  Exported: " << face_uv_file << " (" << num_faces << " faces)" << std::endl;

        // 输出Face 25的统计信息
        if (num_faces > 25 && !face_uv_grids[25].empty()) {
            double sum = 0.0, sum_sq = 0.0;
            double min_val = 1e9, max_val = -1e9;
            for (float val : face_uv_grids[25]) {
                sum += val;
                sum_sq += val * val;
                min_val = std::min(min_val, (double)val);
                max_val = std::max(max_val, (double)val);
            }
            double mean = sum / face_uv_grids[25].size();
            double variance = sum_sq / face_uv_grids[25].size() - mean * mean;
            double std_dev = std::sqrt(variance);

            DBG_LOG << "\n  [Face 25 UV Grid Statistics]" << std::endl;
            DBG_LOG << "    Size: " << face_uv_grids[25].size() << std::endl;
            DBG_LOG << "    Mean: " << mean << std::endl;
            DBG_LOG << "    Std:  " << std_dev << std::endl;
            DBG_LOG << "    Min:  " << min_val << std::endl;
            DBG_LOG << "    Max:  " << max_val << std::endl;
            DBG_LOG << "    First 10 values: ";
            for (int i = 0; i < 10 && i < (int)face_uv_grids[25].size(); ++i) {
                DBG_LOG << face_uv_grids[25][i] << " ";
            }
            DBG_LOG << std::endl;
        }
    }

    // ========================================================================
    // 创建原始Face ID到推理Face ID的映射（因为faces已经重排序）
    // ========================================================================
    std::map<int, int> original_to_inference_face;
    for (size_t inference_id = 0; inference_id < faces.size(); ++inference_id) {
        int original_id = faces[inference_id].face_id;
        original_to_inference_face[original_id] = inference_id;
    }

    DBG_LOG << "[Mapping] Created original->inference face mapping" << std::endl;

    // [TOPOLOGY EXPORT] 导出每个face的coedge列表和映射关系
    if (EXPORT_ENABLED) {
        fs::create_directories("cpp_topology");
        std::string face_coedge_file = "cpp_topology/" + base_name + "_face_coedges.txt";
        std::ofstream face_coedge_out(face_coedge_file);

        if (face_coedge_out.is_open()) {
            face_coedge_out << "=== FACE COEDGE LISTS (REORDERED) ===" << std::endl;
            face_coedge_out << "格式：推理Face ID (原始Face ID): coedge列表" << std::endl;
            face_coedge_out << std::endl;

            for (size_t inference_id = 0; inference_id < faces.size(); ++inference_id) {
                const auto& face = faces[inference_id];
                face_coedge_out << "Face " << inference_id << " (Original Face " << face.face_id << "): ";
                for (size_t i = 0; i < face.coedge_ids.size(); ++i) {
                    if (i > 0) face_coedge_out << ", ";
                    face_coedge_out << face.coedge_ids[i];
                }
                face_coedge_out << " (total: " << face.coedge_ids.size() << ")" << std::endl;
            }

            face_coedge_out << std::endl;
            face_coedge_out << "=== FACE MAPPING ===" << std::endl;
            face_coedge_out << "格式：推理Face ID -> 原始Face ID" << std::endl;
            for (size_t inference_id = 0; inference_id < faces.size(); ++inference_id) {
                face_coedge_out << inference_id << " -> " << faces[inference_id].face_id << std::endl;
            }

            face_coedge_out.close();
            DBG_LOG << "[Topology] Face coedge lists exported to " << face_coedge_file << std::endl;
        }
    }

    // 导出 UVNet 输出
    // 注意：UVNet的输出已经嵌入在coedges的特征中了
    // 需要从coedges提取出来
    std::vector<std::vector<float>> uvnet_surface_features;
    std::vector<std::vector<float>> uvnet_curve_features;

    // === 修改：使用交错排列格式以匹配Python端 ===
    // Python格式: [num_coedges*2, 64] = [248, 64]
    // 行0: coedge 0的parent_face (64维)
    // 行1: coedge 0的mate_face (64维)
    // 行2: coedge 1的parent_face (64维)
    // 行3: coedge 1的mate_face (64维)
    // ...
    for (const auto& coedge : coedges) {
        // 先写入parent_face特征（64维）-> 行 2*i
        uvnet_surface_features.push_back(coedge.parent_face_features);

        // 再写入mate_face特征（64维）-> 行 2*i+1
        uvnet_surface_features.push_back(coedge.mate_face_features);
    }

    // Curve features: 使用和BRepPipeline一样的edge representative逻辑
    // BRepPipeline.h:993-1001: 优先选择orientation==true的coedge
    std::vector<int> edge_representatives(num_edges, -1);
    for (const auto& coedge_info : pipeline.coedges) {
        int eid = coedge_info.edge_idx;
        if (eid >= 0 && eid < num_edges) {
            if (edge_representatives[eid] == -1 || coedge_info.orientation == true) {
                edge_representatives[eid] = coedge_info.id;
            }
        }
    }

    // 根据edge_representatives收集edge features
    for (int e = 0; e < num_edges; ++e) {
        int representative_coedge_id = edge_representatives[e];
        if (representative_coedge_id >= 0 && representative_coedge_id < (int)coedges.size()) {
            uvnet_curve_features.push_back(coedges[representative_coedge_id].edge_features);

            // Debug: Print Edge 38's representative
            if (DebugControl::instance().shouldDebug() && e == 38) {
                bool orientation = (representative_coedge_id < (int)pipeline.coedges.size()) ?
                                  pipeline.coedges[representative_coedge_id].orientation : false;
                DBG_LOG << "[DEBUG] Edge 38 representative: coedge " << representative_coedge_id
                        << " (orientation=" << (orientation ? "true" : "false") << ")" << std::endl;
                DBG_LOG << "[DEBUG] Edge 38 features[:10]: ";
                for (int i = 0; i < 10 && i < (int)coedges[representative_coedge_id].edge_features.size(); ++i) {
                    DBG_LOG << coedges[representative_coedge_id].edge_features[i] << " ";
                }
                DBG_LOG << std::endl;
            }
        } else {
            // 如果找不到representative，用全0特征
            ERR_LOG << "[Warning] Edge " << e << " has no representative coedge!" << std::endl;
            uvnet_curve_features.push_back(std::vector<float>(64, 0.0f));
        }
    }

    if (EXPORT_ENABLED) {
        exporter.exportVectorData(uvnet_surface_features, "uvnet_surface", base_name);
        exporter.exportVectorData(uvnet_curve_features, "uvnet_curve", base_name);
    }

    // ========================================================================
    // 3. 运行BRepNet forward并导出每一层
    // ========================================================================
    DBG_LOG << "[Inference] Running BRepNet..." << std::endl;

    // 我们需要手动执行forward的每一步，并在每一步之后导出

    // Layer 0: 一阶邻居更新
    DBG_LOG << "  [Layer 0] First-order neighbors..." << std::endl;

    std::vector<std::vector<float>> layer0_input_concat;
    std::vector<std::vector<float>> layer0_mlp_output_data;

    DBG_LOG << "\n================================================================================\n";
    DBG_LOG << "[LAYER 0 MLP DEBUG] Processing all coedges\n";
    DBG_LOG << "================================================================================\n";

    // 诊断输出：对所有 coedge 记录简单的统计信息
    std::ofstream diag_mlp;
    if (EXPORT_ENABLED) {
        fs::create_directories("cpp_feature_maps");
        diag_mlp.open("cpp_feature_maps/layer0_mlp_all_coedges_stats.txt");
        diag_mlp << "Layer 0 MLP 输入/输出统计\n";
        diag_mlp << "格式: Coedge_ID, Parent_Face_ID, Input_Min, Input_Max, Input_Mean, FaceState_Min, FaceState_Max\n\n";
    }

    int processed_coedges = 0;
    for (auto& coedge : coedges) {
        // 构建输入：parent_face (64) + mate_face (64) + edge (64) = 192
        std::vector<float> input;
        input.insert(input.end(), coedge.parent_face_features.begin(), coedge.parent_face_features.end());
        input.insert(input.end(), coedge.mate_face_features.begin(), coedge.mate_face_features.end());
        input.insert(input.end(), coedge.edge_features.begin(), coedge.edge_features.end());

        layer0_input_concat.push_back(input);

        // 通过 MLP
        // 必须clone()！from_blob只是指向input.data()的指针，不拥有所有权
        // 当input向量在下一次循环被重用时，之前的Tensor数据会被污染
        Tensor input_tensor = breptorch::from_blob(input.data(), {1, 192}, breptorch::kFloat32).clone();
        Tensor output = model->layer0_mlp->forward(input_tensor);  // (1, 60)

        // 详细调试：仅对前 3 个 coedge 打印
        if (DebugControl::instance().shouldDebug() && processed_coedges < 3) {
            DBG_LOG << "\n[DEBUG Layer 0 MLP] Coedge " << coedge.coedge_id << std::endl;
            DBG_LOG << "  Input shape: [1, 192]" << std::endl;
            DBG_LOG << "  parent_face_features (first 10): ";
            for (int i = 0; i < 10; ++i) DBG_PRINTF("%.10f ", input[i]);
            DBG_LOG << std::endl;
            DBG_LOG << "  mate_face_features (first 10): ";
            for (int i = 64; i < 74; ++i) DBG_PRINTF("%.10f ", input[i]);
            DBG_LOG << std::endl;
            DBG_LOG << "  edge_features (first 10): ";
            for (int i = 128; i < 138; ++i) DBG_PRINTF("%.10f ", input[i]);
            DBG_LOG << std::endl;
            DBG_LOG << "  MLP output shape: [" << output.size(0) << ", " << output.size(1) << "]" << std::endl;
            DBG_LOG << "  edge_state (dims 0-9): ";
            for (int i = 0; i < 10; ++i) DBG_PRINTF("%.10f ", output.at({0, i}));
            DBG_LOG << std::endl;
            DBG_LOG << "  face_state (dims 30-39): ";
            for (int i = 30; i < 40; ++i) DBG_PRINTF("%.10f ", output.at({0, i}));
            DBG_LOG << std::endl;
        }

        // 保存MLP输出
        std::vector<float> mlp_out;
        for (int i = 0; i < 60; ++i) {
            mlp_out.push_back(output.at({0, i}));
        }
        layer0_mlp_output_data.push_back(mlp_out);

        // 分离 face 和 edge 输出
        coedge.layer0_edge_state.resize(30);
        coedge.layer0_face_state.resize(30);
        for (int i = 0; i < 30; ++i) {
            coedge.layer0_edge_state[i] = output.at({0, i});
            coedge.layer0_face_state[i] = output.at({0, i + 30});
        }

        // 统计输入和输出
        float input_min = *std::min_element(input.begin(), input.end());
        float input_max = *std::max_element(input.begin(), input.end());
        float input_mean = std::accumulate(input.begin(), input.end(), 0.0f) / input.size();

        float face_min = *std::min_element(coedge.layer0_face_state.begin(), coedge.layer0_face_state.end());
        float face_max = *std::max_element(coedge.layer0_face_state.begin(), coedge.layer0_face_state.end());

        // 记录到诊断文件
        if (diag_mlp.is_open()) {
            diag_mlp << coedge.coedge_id << "," << coedge.parent_face_id << ","
                    << input_min << "," << input_max << "," << input_mean << ","
                    << face_min << "," << face_max << "\n";
        }

        processed_coedges++;
    }

    if (diag_mlp.is_open()) diag_mlp.close();

    if (EXPORT_ENABLED) {
        exporter.exportVectorData(layer0_input_concat, "layer0_input_concat", base_name);
        exporter.exportVectorData(layer0_mlp_output_data, "layer0_mlp_output", base_name);
    }

    // Face MaxPooling
    // 诊断文件：记录每个面的 MaxPooling 过程
    std::ofstream diag_pool;
    if (EXPORT_ENABLED) {
        diag_pool.open("cpp_feature_maps/layer0_face_pooling_all_faces_stats.txt");
        diag_pool << "Layer 0 Face MaxPooling 统计\n";
        diag_pool << "格式: Face_ID, Num_Coedges, Coedge_Count, Pooled_Min, Pooled_Max, Pooled_Mean\n\n";
    }

    std::vector<std::vector<float>> layer0_face_pooling_data;
    for (auto& face : faces) {
        face.layer0_state.resize(30, 0.0f);  // 初始化为0，与Python一致（Python添加零向量用于填充）
        int coedge_count = 0;
        for (int coedge_id : face.coedge_ids) {
            if (coedge_id >= 0 && coedge_id < (int)coedges.size()) {
                const auto& coedge_state = coedges[coedge_id].layer0_face_state;
                for (int i = 0; i < 30; ++i) {
                    face.layer0_state[i] = std::max(face.layer0_state[i], coedge_state[i]);
                }
                coedge_count++;
            }
        }

        // 计算统计
        float pooled_min = *std::min_element(face.layer0_state.begin(), face.layer0_state.end());
        float pooled_max = *std::max_element(face.layer0_state.begin(), face.layer0_state.end());
        float pooled_mean = std::accumulate(face.layer0_state.begin(), face.layer0_state.end(), 0.0f) / 30;

        // 记录到诊断文件
        if (diag_pool.is_open()) {
            diag_pool << face.face_id << "," << face.coedge_ids.size() << "," << coedge_count << ","
                     << pooled_min << "," << pooled_max << "," << pooled_mean << "\n";
        }

        layer0_face_pooling_data.push_back(face.layer0_state);
    }

    if (diag_pool.is_open()) diag_pool.close();

    if (EXPORT_ENABLED) {
        exporter.exportVectorData(layer0_face_pooling_data, "layer0_face_pooling", base_name);
    }

    // Edge MaxPooling
    std::vector<std::vector<float>> layer0_edge_pooling_data;
    std::ofstream edge_debug_file;
    if (EXPORT_ENABLED) {
        edge_debug_file.open("cpp_feature_maps/edge_pooling_debug.txt", std::ios::app);
        edge_debug_file << "=== Test: " << base_name << " ===\n";
    }

    for (auto& edge : edges) {
        edge.layer0_state.resize(30, 0.0f);  // 初始化为0，与Python一致
        int max_coedge_id = -1;
        float max_value = -1e9f;

        for (int coedge_id : edge.coedge_ids) {
            if (coedge_id >= 0 && coedge_id < (int)coedges.size()) {
                const auto& coedge_state = coedges[coedge_id].layer0_edge_state;
                float coedge_sum = 0.0f;
                for (int i = 0; i < 30; ++i) {
                    edge.layer0_state[i] = std::max(edge.layer0_state[i], coedge_state[i]);
                    coedge_sum += coedge_state[i];
                }
                if (coedge_sum > max_value) {
                    max_value = coedge_sum;
                    max_coedge_id = coedge_id;
                }
            }
        }

        // Debug output
        if (edge_debug_file.is_open()) {
            edge_debug_file << "Edge " << edge.edge_id << ": coedge_ids=[";
            for (int cid : edge.coedge_ids) edge_debug_file << cid << " ";
            edge_debug_file << "] max_coedge=" << max_coedge_id
                           << " value[0]=" << edge.layer0_state[0]
                           << " sum_first_10=" << std::accumulate(edge.layer0_state.begin(), edge.layer0_state.begin() + std::min(10, 30), 0.0f);
            for (int i = 0; i < std::min(10, 30); ++i) {
                edge_debug_file << " " << edge.layer0_state[i];
            }
            edge_debug_file << "\n";
        }

        layer0_edge_pooling_data.push_back(edge.layer0_state);
    }
    if (edge_debug_file.is_open()) edge_debug_file.close();

    if (EXPORT_ENABLED) {
        exporter.exportVectorData(layer0_edge_pooling_data, "layer0_edge_pooling", base_name);
    }

    // ========================================================================
    // Layer 1: 二阶邻居
    // ========================================================================
    DBG_LOG << "  [Layer 1] Second-order neighbors..." << std::endl;

    std::vector<std::vector<float>> layer1_input_concat;
    std::vector<std::vector<float>> layer1_mlp_output_data;

    for (auto& coedge : coedges) {
        // 构建输入：parent_face (30) + mate_face (30) + edge (30) = 90
        std::vector<float> input;

        // 使用映射：原始face_id -> 推理face索引
        int parent_inference_id = original_to_inference_face[coedge.parent_face_id];
        int mate_inference_id = original_to_inference_face[coedge.mate_face_id];

        input.insert(input.end(), faces[parent_inference_id].layer0_state.begin(),
                     faces[parent_inference_id].layer0_state.end());
        input.insert(input.end(), faces[mate_inference_id].layer0_state.begin(),
                     faces[mate_inference_id].layer0_state.end());
        input.insert(input.end(), edges[coedge.edge_id].layer0_state.begin(),
                     edges[coedge.edge_id].layer0_state.end());

        layer1_input_concat.push_back(input);

        // 通过 MLP
        // 必须clone()！from_blob只是指向input.data()的指针，不拥有所有权
        Tensor input_tensor = breptorch::from_blob(input.data(), {1, 90}, breptorch::kFloat32).clone();
        Tensor output = model->layer1_mlp->forward(input_tensor);  // (1, 60)

        // 保存MLP输出
        std::vector<float> mlp_out;
        for (int i = 0; i < 60; ++i) {
            mlp_out.push_back(output.at({0, i}));
        }
        layer1_mlp_output_data.push_back(mlp_out);

        // 分离 face 和 edge 输出
        coedge.layer1_edge_state.resize(30);
        coedge.layer1_face_state.resize(30);
        for (int i = 0; i < 30; ++i) {
            coedge.layer1_edge_state[i] = output.at({0, i});
            coedge.layer1_face_state[i] = output.at({0, i + 30});
        }
    }

    if (EXPORT_ENABLED) {
        exporter.exportVectorData(layer1_input_concat, "layer1_input_concat", base_name);
        exporter.exportVectorData(layer1_mlp_output_data, "layer1_mlp_output", base_name);
    }

    // Face MaxPooling
    std::vector<std::vector<float>> layer1_face_pooling_data;
    for (auto& face : faces) {
        face.layer1_state.resize(30, 0.0f);  // 初始化为0，与Python一致
        for (int coedge_id : face.coedge_ids) {
            if (coedge_id >= 0 && coedge_id < (int)coedges.size()) {
                const auto& coedge_state = coedges[coedge_id].layer1_face_state;
                for (int i = 0; i < 30; ++i) {
                    face.layer1_state[i] = std::max(face.layer1_state[i], coedge_state[i]);
                }
            }
        }
        layer1_face_pooling_data.push_back(face.layer1_state);
    }

    if (EXPORT_ENABLED) {
        exporter.exportVectorData(layer1_face_pooling_data, "layer1_face_pooling", base_name);
    }

    // Edge MaxPooling
    std::vector<std::vector<float>> layer1_edge_pooling_data;
    for (auto& edge : edges) {
        edge.layer1_state.resize(30, 0.0f);  // 初始化为0，与Python一致
        for (int coedge_id : edge.coedge_ids) {
            if (coedge_id >= 0 && coedge_id < (int)coedges.size()) {
                const auto& coedge_state = coedges[coedge_id].layer1_edge_state;
                for (int i = 0; i < 30; ++i) {
                    edge.layer1_state[i] = std::max(edge.layer1_state[i], coedge_state[i]);
                }
            }
        }
        layer1_edge_pooling_data.push_back(edge.layer1_state);
    }

    if (EXPORT_ENABLED) {
        exporter.exportVectorData(layer1_edge_pooling_data, "layer1_edge_pooling", base_name);
    }

    // ========================================================================
    // Output Layer: 三阶邻居
    // ========================================================================
    DBG_LOG << "  [Output Layer] Third-order neighbors..." << std::endl;

    std::vector<std::vector<float>> output_layer_input_concat;
    std::vector<std::vector<float>> output_layer_mlp_output_data;

    for (auto& coedge : coedges) {
        // 构建输入：parent_face (30) + mate_face (30) + edge (30) = 90
        std::vector<float> input;

        // 使用映射：原始face_id -> 推理face索引
        int parent_inference_id = original_to_inference_face[coedge.parent_face_id];
        int mate_inference_id = original_to_inference_face[coedge.mate_face_id];

        input.insert(input.end(), faces[parent_inference_id].layer1_state.begin(),
                     faces[parent_inference_id].layer1_state.end());
        input.insert(input.end(), faces[mate_inference_id].layer1_state.begin(),
                     faces[mate_inference_id].layer1_state.end());
        input.insert(input.end(), edges[coedge.edge_id].layer1_state.begin(),
                     edges[coedge.edge_id].layer1_state.end());

        output_layer_input_concat.push_back(input);

        // 通过 MLP
        // 必须clone()！from_blob只是指向input.data()的指针，不拥有所有权
        Tensor input_tensor = breptorch::from_blob(input.data(), {1, 90}, breptorch::kFloat32).clone();
        Tensor output = model->output_mlp->forward(input_tensor);  // (1, 30)

        // 保存MLP输出
        std::vector<float> mlp_out;
        for (int i = 0; i < 30; ++i) {
            mlp_out.push_back(output.at({0, i}));
        }
        output_layer_mlp_output_data.push_back(mlp_out);

        // 保存到coedge
        coedge.output_face_state.resize(30);
        for (int i = 0; i < 30; ++i) {
            coedge.output_face_state[i] = output.at({0, i});
        }
    }

    if (EXPORT_ENABLED) {
        exporter.exportVectorData(output_layer_input_concat, "output_layer_input_concat", base_name);
        exporter.exportVectorData(output_layer_mlp_output_data, "output_layer_mlp_output", base_name);
    }

    // Face MaxPooling（最终 embedding）
    std::vector<std::vector<float>> output_layer_face_embedding_data;
    for (size_t inference_id = 0; inference_id < faces.size(); ++inference_id) {
        auto& face = faces[inference_id];

        // === 关键修复：Big faces vs Small faces 的初始化差异 ===
        const int max_coedges_per_face = 30;
        bool is_small_face = (int)face.coedge_ids.size() < max_coedges_per_face;
        float init_value = is_small_face ? 0.0f : -1e9f;
        face.output_state.assign(30, init_value);

        // 调试输出
        if (!is_small_face) {
            DBG_LOG << "[Debug] Face " << face.face_id << " is BIG FACE with "
                    << face.coedge_ids.size() << " coedges, init=" << init_value << std::endl;
        }

        for (int coedge_id : face.coedge_ids) {
            if (coedge_id >= 0 && coedge_id < (int)coedges.size()) {
                const auto& coedge_state = coedges[coedge_id].output_face_state;
                for (int i = 0; i < 30; ++i) {
                    face.output_state[i] = std::max(face.output_state[i], coedge_state[i]);
                }
            }
        }

        // === 重要：只对小面应用ReLU ===
        // 根据验证结果，暂时使用< 30以匹配现有的Python logits文件
        // Python代码使用<= 30，但提供的logits文件似乎是用< 30生成的
        // 可能原因：训练模型时使用了旧版本代码（< 30）
        // 等待Python端重新生成logits后再调整

        if (is_small_face) {
            // 小面：应用ReLU
            for (int i = 0; i < 30; ++i) {
                face.output_state[i] = std::max(0.0f, face.output_state[i]);
            }
        }
        // 大面：不应用ReLU，保留负值

        output_layer_face_embedding_data.push_back(face.output_state);
    }

    if (EXPORT_ENABLED) {
        exporter.exportVectorData(output_layer_face_embedding_data, "output_layer_face_embedding", base_name);
    }

    // ========================================================================
    // Linear 分类层（用于验证，但不导出，因为已经有cpp_logits了）
    // ========================================================================
    DBG_LOG << "  [Classification] Running Linear layer..." << std::endl;

    // 构建face embedding tensor
    Tensor face_embedding = breptorch::Tensor({num_faces, 30}, breptorch::kFloat32);
    for (int f = 0; f < num_faces; ++f) {
        for (int i = 0; i < 30; ++i) {
            face_embedding.at({f, i}) = faces[f].output_state[i];
        }
    }

    auto logits = model->classification_layer->forward(face_embedding);
    breptorch::Tensor probs = breptorch::softmax(logits, 1);

    // 获取预测结果并统计
    std::vector<int> predictions;
    std::map<int, int> class_dist;

    for (int f = 0; f < num_faces; ++f) {
        float max_prob = -1e9f;
        int pred_class = 0;

        for (int c = 0; c < 27; ++c) {
            float p = probs.at({f, c});
            if (p > max_prob) {
                max_prob = p;
                pred_class = c;
            }
        }
        predictions.push_back(pred_class);
        class_dist[pred_class]++;
    }

    // 输出预测结果（简化版）
    if (DebugControl::instance().shouldDebug()) {
        DBG_LOG << "  [Predictions] ";
        for (const auto& pair : class_dist) {
            DBG_LOG << "Class" << pair.first << ":" << pair.second << " ";
        }
        DBG_LOG << std::endl;
    }

    // ========================================================================
    // 导出 Logits 到 cpp_logits/ — 始终执行，不受调试开关控制
    // ========================================================================
    DBG_LOG << "  [Export] Saving logits..." << std::endl;

    // 确保目录存在
    fs::create_directories("cpp_logits");

    // ========================================================================
    // 重要：Python端按原始Face顺序导出logits，我们也需要按原始顺序导出
    // 当前logits tensor是按推理顺序的，需要重新排列
    // ========================================================================

    // 创建原始顺序的logits数组
    std::vector<std::vector<float>> logits_original_order(num_faces);

    for (int inference_id = 0; inference_id < num_faces; ++inference_id) {
        int original_id = faces[inference_id].face_id;

        std::vector<float> logit_row(27);
        for (int c = 0; c < 27; ++c) {
            logit_row[c] = logits.at({inference_id, c});
        }

        logits_original_order[original_id] = logit_row;
    }

    // 导出 logits（按原始顺序）
    std::string logits_path = "cpp_logits/" + base_name + ".logits";
    std::ofstream logits_file(logits_path);
    logits_file << std::scientific << std::setprecision(20);

    for (int original_id = 0; original_id < num_faces; ++original_id) {
        for (int c = 0; c < 27; ++c) {
            if (c > 0) logits_file << " ";
            logits_file << logits_original_order[original_id][c];
        }
        logits_file << "\n";
    }
    logits_file.close();

    DBG_LOG << "  [Export] Logits saved to: " << logits_path << " (original face order)" << std::endl;

    INFO_LOG << " -> [✓] F:" << num_faces << " E:" << num_edges << std::endl;

    // ========================================================================
    // [MEMORY CLEANUP] 显式释放所有临时数据结构，防止内存泄漏
    // ========================================================================
    // 对于长期处理大批量文件（28,548个），必须显式清理每个文件的临时数据
    // 避免 std::vector 和 Tensor 的底层存储累积导致 std::bad_alloc

    // 清理 Layer 0 临时向量
    {
        std::vector<std::vector<float>>().swap(layer0_input_concat);
        std::vector<std::vector<float>>().swap(layer0_mlp_output_data);
        std::vector<std::vector<float>>().swap(layer0_face_pooling_data);
        std::vector<std::vector<float>>().swap(layer0_edge_pooling_data);
    }

    // 清理 Layer 1 临时向量
    {
        std::vector<std::vector<float>>().swap(layer1_input_concat);
        std::vector<std::vector<float>>().swap(layer1_mlp_output_data);
        std::vector<std::vector<float>>().swap(layer1_face_pooling_data);
        std::vector<std::vector<float>>().swap(layer1_edge_pooling_data);
    }

    // 清理 Output Layer 临时向量
    {
        std::vector<std::vector<float>>().swap(output_layer_input_concat);
        std::vector<std::vector<float>>().swap(output_layer_mlp_output_data);
        std::vector<std::vector<float>>().swap(output_layer_face_embedding_data);
    }

    // 清理拓扑数据结构
    {
        std::vector<CoedgeData>().swap(coedges);
        std::vector<FaceData>().swap(faces);
        std::vector<EdgeData>().swap(edges);
    }

    // 清理其他临时数据
    {
        std::vector<std::vector<float>>().swap(uvnet_surface_features);
        std::vector<std::vector<float>>().swap(uvnet_curve_features);
        std::vector<std::vector<float>>().swap(logits_original_order);
        std::vector<int>().swap(edge_representatives);
        std::map<int, int>().swap(original_to_inference_face);
    }
}

int main(int argc, char* argv[]) {
    // 解析命令行参数
    DebugControl::instance().parse(argc, argv);

    // 自动保存所有终端输出到文件
    OutputLogger logger("cpp_inference.txt");

    SetConsoleOutputCP(65001);
    SetConsoleCP(65001);

    INFO_LOG << "=== BRepNet Inference Tool ===" << std::endl;
    DBG_LOG << "Purpose: Export intermediate layer outputs for comparison with Python" << std::endl;
    DBG_LOG << "Output directory: cpp_feature_maps/" << std::endl;

    // ========================================================================
    // 1. 创建导出器
    // ========================================================================
    FeatureMapExporter exporter("cpp_feature_maps");

    // ========================================================================
    // 2. 加载模型
    // ========================================================================
    std::string weights_file = "inference_data/state_dict.npz";
    std::string step_dir = "inference_data/step_files";

    DBG_LOG << "\n[Model] Loading weights: " << weights_file << std::endl;

    auto model = std::make_shared<BRepNetImpl>(27);
    cnpy::npz_t npz = cnpy::npz_load(weights_file);

    // 加载 UV-Net 权重
    std::map<std::string, breptorch::Tensor> surf_weights, curve_weights;
    for (auto& item : npz) {
        if (item.first.find("surface_encoder") != std::string::npos) {
            auto arr = item.second;
            std::vector<int64_t> shape(arr.shape.begin(), arr.shape.end());
            surf_weights[item.first] = breptorch::from_blob(arr.data<float>(), shape, breptorch::kFloat32).clone();
        }
        if (item.first.find("curve_encoder") != std::string::npos) {
            auto arr = item.second;
            std::vector<int64_t> shape(arr.shape.begin(), arr.shape.end());
            curve_weights[item.first] = breptorch::from_blob(arr.data<float>(), shape, breptorch::kFloat32).clone();
        }
    }
    model->surf_enc->load_weights(surf_weights);
    model->curve_enc->load_weights(curve_weights);

    // 加载 BRepNet 权重
    auto params = model->named_parameters();

    // Debug: print all available parameters
    DBG_LOG << "\n[Debug] Available C++ parameters:" << std::endl;
    if (DebugControl::instance().shouldDebug()) {
        for (auto& p : params) {
            DBG_LOG << "  " << p.first << std::endl;
        }
    }

    DBG_LOG << "\n[Debug] Processing NPZ weights..." << std::endl;
    for (auto& item : npz) {
        std::string original_key = item.first;
        std::string key = original_key;
        if (key.find("layers.0.mlp") != std::string::npos) {
            key = "layer_0.mlp" + key.substr(key.find(".mlp") + 4);
        } else if (key.find("layers.1.mlp") != std::string::npos) {
            key = "layer_1.mlp" + key.substr(key.find(".mlp") + 4);
        }
        if (params.find(key) != params.end()) {
            auto arr = item.second;
            std::vector<int64_t> shape(arr.shape.begin(), arr.shape.end());
            *params[key] = breptorch::from_blob(arr.data<float>(), shape, breptorch::kFloat32).clone();

            // 调试：打印所有 classification_layer 的加载
            if (DebugControl::instance().shouldDebug() && key.find("classification_layer") != std::string::npos) {
                DBG_LOG << "  [LOADED] " << original_key << " (shape: ";
                for (size_t i = 0; i < shape.size(); i++) {
                    DBG_LOG << shape[i];
                    if (i < shape.size() - 1) DBG_LOG << ", ";
                }
                DBG_LOG << ")" << std::endl;
            }

            // Debug: print when loading layer 1 linear_1 bias
            if (DebugControl::instance().shouldDebug() && key.find("layer_1.mlp") != std::string::npos && key.find("linear_1.bias") != std::string::npos) {
                DBG_LOG << "  [LOADED] " << original_key << " -> " << key << " (shape: ";
                for (size_t i = 0; i < shape.size(); i++) {
                    DBG_LOG << shape[i];
                    if (i < shape.size() - 1) DBG_LOG << ", ";
                }
                DBG_LOG << ")" << std::endl;
            }
        } else {
            // Debug: print when NOT found
            if (DebugControl::instance().shouldDebug()) {
                if (original_key.find("classification_layer") != std::string::npos) {
                    DBG_LOG << "  [NOT FOUND] " << original_key << " -> " << key << std::endl;
                }
                if (original_key.find("layer_1") != std::string::npos && original_key.find("linear_1.bias") != std::string::npos) {
                    DBG_LOG << "  [NOT FOUND] " << original_key << " -> " << key << std::endl;
                }
            }
        }
    }

    DBG_LOG << "\n[Model] Weights loaded successfully!" << std::endl;

    // 验证 Layer 1 linear_1 bias 是否被加载（主循环已完成加载，此处仅做验证）
    auto layer1_bias_key = "layer_1.mlp.mlp.linear_1.bias";
    if (params.find(layer1_bias_key) != params.end()) {
        DBG_LOG << "\n[Verification] Layer 1 linear_1 bias: loaded" << std::endl;
    } else {
        DBG_LOG << "\n[Verification] Layer 1 linear_1 bias: NOT FOUND" << std::endl;
    }

    // ========================================================================
    // 3. 获取所有 STEP 文件
    // ========================================================================
    DBG_LOG << "\n[Files] Scanning directory: " << step_dir << std::endl;
    auto step_files = get_step_files(step_dir);

    if (step_files.empty()) {
        ERR_LOG << "[Error] No STEP files found" << std::endl;
        return -1;
    }

    DBG_LOG << "[Files] Found " << step_files.size() << " STEP files" << std::endl;

    // ========================================================================
    // 4. 批量推理并导出
    // ========================================================================
    auto total_start = std::chrono::high_resolution_clock::now();

    int total_files = (int)step_files.size();
    int current = 0;

    for (const auto& step_file : step_files) {
        current++;
        DBG_LOG << "\n[" << current << "/" << total_files << "] ";
        run_inference_with_export(step_file, model, exporter);

        // ========================================================================
        // [MEMORY MAINTENANCE] 定期检查和清理内存
        // ========================================================================
        // 每处理100个文件进行一次内存检查，防止长期运行导致内存溢出
        if (current % 100 == 0) {
            // Windows 内存信息
            PROCESS_MEMORY_COUNTERS pmc;
            if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
                double working_set_mb = pmc.WorkingSetSize / 1024.0 / 1024.0;
                double peak_mb = pmc.PeakWorkingSetSize / 1024.0 / 1024.0;
                DBG_LOG << " [Memory] WS: " << std::fixed << std::setprecision(1) << working_set_mb << " MB, Peak: " << peak_mb << " MB" << std::flush;
            }
        }
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);

    // ========================================================================
    // 5. 打印导出清单
    // ========================================================================
    INFO_LOG << "\n" << std::string(70, '=') << std::endl;
    INFO_LOG << "All " << total_files << " files completed!" << std::endl;
    INFO_LOG << "Total time: " << total_duration.count() / 1000.0 << " seconds" << std::endl;
    INFO_LOG << "Average time per file: " << (total_duration.count() / total_files) << " ms" << std::endl;
    INFO_LOG << std::string(70, '=') << std::endl;

    return 0;
}
