// 调试输出开关：通过 DebugControl.h 统一管理
// 通过命令行参数控制，永远不需要注释/取消注释代码

#include "DebugControl.h"
#include "BRepNet.h"
#include "BRepNetAdapter.h"
#include "BRepPipeline.h"
#include "FeatureMapExporter.h"
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
#include <thread>
#include <mutex>
#include <atomic>

namespace fs = std::filesystem;

// 全局互斥锁：用于多线程模式下的控制台输出同步
std::mutex g_console_mutex;

// 全局强制重新处理标志（--force 参数）
bool g_force_overwrite = false;

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

// ============================================================================
// 模型加载函数：从已加载的 NPZ 数据创建一个独立的模型实例
// 每个线程调用此函数获得自己的模型副本，确保线程安全
// ============================================================================
std::shared_ptr<BRepNetImpl> load_model(const cnpy::npz_t& npz) {
    auto model = std::make_shared<BRepNetImpl>(27);

    // 加载 UV-Net 权重
    std::map<std::string, breptorch::Tensor> surf_weights, curve_weights;
    std::map<std::string, breptorch::Tensor> surf2_weights;
    for (auto& item : npz) {
        auto arr = item.second;
        std::vector<int64_t> shape(arr.shape.begin(), arr.shape.end());
        breptorch::Tensor t = breptorch::from_blob(arr.data<float>(), shape, breptorch::kFloat32).clone();

        // V4: Must distinguish surface_encoder. from surface_encoder2.
        if (item.first.substr(0, 17) == "surface_encoder2.") {
            surf2_weights["surface_encoder." + item.first.substr(17)] = t;
        } else
        if (item.first.find("surface_encoder.") != std::string::npos) {
            surf_weights[item.first] = t;
        }
        if (item.first.find("curve_encoder.") != std::string::npos) {
            curve_weights[item.first] = t;
        }
    }
    model->surf_enc->load_weights(surf_weights);
    model->curve_enc->load_weights(curve_weights);
    model->surf_enc2->load_weights(surf2_weights);

    // 加载 BRepNet 权重
    auto params = model->named_parameters();
    for (auto& item : npz) {
        std::string key = item.first;
        if (key.find("layers.0.mlp") != std::string::npos) {
            key = "layer_0.mlp" + key.substr(key.find(".mlp") + 4);
        } else if (key.find("layers.1.mlp") != std::string::npos) {
            key = "layer_1.mlp" + key.substr(key.find(".mlp") + 4);
        }
        if (params.find(key) != params.end()) {
            auto arr = item.second;
            std::vector<int64_t> shape(arr.shape.begin(), arr.shape.end());
            *params[key] = breptorch::from_blob(arr.data<float>(), shape, breptorch::kFloat32).clone();
        }
    }

    return model;
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

    // 检查是否已经处理过（--force 可跳过此检查）
    if (!g_force_overwrite && is_file_already_processed(base_name)) {
        {
            std::lock_guard<std::mutex> lock(g_console_mutex);
            INFO_LOG << fs::absolute(step_path).string() << " [SKIPPED]" << std::endl;
        }
        return;
    }

    auto file_start = std::chrono::high_resolution_clock::now();

    // ========================================================================
    // 1. 数据预处理
    // ========================================================================
    BRepPipeline pipeline;
    if (!pipeline.process(step_file)) {
        {
            std::lock_guard<std::mutex> lock(g_console_mutex);
            ERR_LOG << "[Error] Processing failed: " << base_name << std::endl;
        }
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

    // 提取Face UV Grids (Coedge格式: [num_coedges*2, 9, 20, 20])
    int num_coedges_topo = (int)pipeline.coedges.size();
    // 必须clone()，避免数据污染。后续操作可能修改张量，影响原始数据
    Tensor all_face_grids = pipeline.FaceGridsLocal.clone().view({num_coedges_topo * 2, 9, 20, 20});  // [380, 9, 20, 20]

    // 展平为 [380, 900]
    Tensor flattened_face_grids = all_face_grids.view({num_coedges_topo * 2, 9 * 20 * 20});

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

    }

    // ========================================================================
    // 3. 提取 UVNet 特征（使用BRepNetAdapter）
    // ========================================================================
    DBG_LOG << "[Feature Extraction] Running UVNet..." << std::endl;

    auto coedges = BRepNetAdapter::extract_coedges(pipeline, model->surf_enc, model->curve_enc, model->surf_enc2);
    auto faces = BRepNetAdapter::extract_faces(pipeline);

    // ========================================================================
    // 导出按原始Face ID组织的UV Grid
    // ========================================================================
    if (EXPORT_ENABLED) {
        DBG_LOG << "[UV Grid Export] Creating per-face UV grids..." << std::endl;

        // 直接导出全局Face网格（与Python extract_face_point_grids 一致）
        std::vector<std::vector<float>> face_uv_grids(num_faces);

        if (pipeline.FaceGridsGlobal.defined() && pipeline.FaceGridsGlobal.size(0) >= num_faces) {
            for (int f = 0; f < num_faces; ++f) {
                for (int c = 0; c < 9; ++c) {
                    for (int i = 0; i < 20; ++i) {
                        for (int j = 0; j < 20; ++j) {
                            face_uv_grids[f].push_back(pipeline.FaceGridsGlobal.at({f, c, i, j}));
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
                for (int i = 0; i < 3600; ++i) {
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
    }

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

    // B2: 批量构建输入 [num_coedges, 192] 并一次前向 (裸指针加速)
    int num_coedges_l0 = (int)coedges.size();
    Tensor l0_batch_input({num_coedges_l0, 192}, breptorch::kFloat32);
    float* l0_in_ptr = l0_batch_input.storage_->dataf_.data();
    for (int c = 0; c < num_coedges_l0; ++c) {
        const auto& coedge = coedges[c];
        float* row = l0_in_ptr + c * 192;
        memcpy(row,       coedge.parent_face_features.data(), 64 * sizeof(float));
        memcpy(row + 64,  coedge.mate_face_features.data(),   64 * sizeof(float));
        memcpy(row + 128, coedge.edge_features.data(),         64 * sizeof(float));
    }

    Tensor l0_batch_output = model->layer0_mlp->forward(l0_batch_input);  // [num_coedges, 60]
    const float* l0_out_ptr = l0_batch_output.storage_->dataf_.data();

    for (int c = 0; c < num_coedges_l0; ++c) {
        // 导出数据
        if (EXPORT_ENABLED) {
            layer0_input_concat.emplace_back(l0_in_ptr + c * 192, l0_in_ptr + c * 192 + 192);
            layer0_mlp_output_data.emplace_back(l0_out_ptr + c * 60, l0_out_ptr + c * 60 + 60);
        }

        // 分离 face 和 edge 输出
        coedges[c].layer0_edge_state.resize(30);
        coedges[c].layer0_face_state.resize(30);
        memcpy(coedges[c].layer0_edge_state.data(), l0_out_ptr + c * 60,      30 * sizeof(float));
        memcpy(coedges[c].layer0_face_state.data(), l0_out_ptr + c * 60 + 30, 30 * sizeof(float));
    }

    if (EXPORT_ENABLED) {
        exporter.exportVectorData(layer0_input_concat, "layer0_input_concat", base_name);
        exporter.exportVectorData(layer0_mlp_output_data, "layer0_mlp_output", base_name);
    }

    // Face MaxPooling
    std::vector<std::vector<float>> layer0_face_pooling_data;
    for (auto& face : faces) {
        face.layer0_state.resize(30, 0.0f);  // 初始化为0，与Python一致
        for (int coedge_id : face.coedge_ids) {
            if (coedge_id >= 0 && coedge_id < (int)coedges.size()) {
                const auto& coedge_state = coedges[coedge_id].layer0_face_state;
                for (int i = 0; i < 30; ++i) {
                    face.layer0_state[i] = std::max(face.layer0_state[i], coedge_state[i]);
                }
            }
        }
        layer0_face_pooling_data.push_back(face.layer0_state);
    }

    if (EXPORT_ENABLED) {
        exporter.exportVectorData(layer0_face_pooling_data, "layer0_face_pooling", base_name);
    }



    // ========================================================================
    // Layer 1: 二阶邻居
    // ========================================================================
    DBG_LOG << "  [Layer 1] Second-order neighbors..." << std::endl;

    std::vector<std::vector<float>> layer1_input_concat;
    std::vector<std::vector<float>> layer1_mlp_output_data;

    // B2: 批量构建输入 [num_coedges, 90] 并一次前向 (裸指针加速)
    int num_coedges_l1 = (int)coedges.size();
    Tensor l1_batch_input({num_coedges_l1, 90}, breptorch::kFloat32);
    float* l1_in_ptr = l1_batch_input.storage_->dataf_.data();
    for (int c = 0; c < num_coedges_l1; ++c) {
        const auto& coedge = coedges[c];
        float* row = l1_in_ptr + c * 90;
        memcpy(row,      faces[coedge.parent_face_id].layer0_state.data(), 30 * sizeof(float));
        memcpy(row + 30, faces[coedge.mate_face_id].layer0_state.data(),   30 * sizeof(float));
        memcpy(row + 60, coedge.layer0_edge_state.data(),                  30 * sizeof(float));
    }

    Tensor l1_batch_output = model->layer1_mlp->forward(l1_batch_input);  // [num_coedges, 60]
    const float* l1_out_ptr = l1_batch_output.storage_->dataf_.data();

    for (int c = 0; c < num_coedges_l1; ++c) {
        // 导出数据
        if (EXPORT_ENABLED) {
            layer1_input_concat.emplace_back(l1_in_ptr + c * 90, l1_in_ptr + c * 90 + 90);
            layer1_mlp_output_data.emplace_back(l1_out_ptr + c * 60, l1_out_ptr + c * 60 + 60);
        }

        // 分离 face 和 edge 输出
        coedges[c].layer1_edge_state.resize(30);
        coedges[c].layer1_face_state.resize(30);
        memcpy(coedges[c].layer1_edge_state.data(), l1_out_ptr + c * 60,      30 * sizeof(float));
        memcpy(coedges[c].layer1_face_state.data(), l1_out_ptr + c * 60 + 30, 30 * sizeof(float));
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



    // ========================================================================
    // Output Layer: 三阶邻居
    // ========================================================================
    DBG_LOG << "  [Output Layer] Third-order neighbors..." << std::endl;

    std::vector<std::vector<float>> output_layer_input_concat;
    std::vector<std::vector<float>> output_layer_mlp_output_data;

    // B2: 批量构建输入 [num_coedges, 90] 并一次前向 (裸指针加速)
    int num_coedges_out = (int)coedges.size();
    Tensor out_batch_input({num_coedges_out, 90}, breptorch::kFloat32);
    float* out_in_ptr = out_batch_input.storage_->dataf_.data();
    for (int c = 0; c < num_coedges_out; ++c) {
        const auto& coedge = coedges[c];
        float* row = out_in_ptr + c * 90;
        memcpy(row,      faces[coedge.parent_face_id].layer1_state.data(), 30 * sizeof(float));
        memcpy(row + 30, faces[coedge.mate_face_id].layer1_state.data(),   30 * sizeof(float));
        memcpy(row + 60, coedge.layer1_edge_state.data(),                  30 * sizeof(float));
    }

    Tensor out_batch_output = model->output_mlp->forward(out_batch_input);  // [num_coedges, 30]
    const float* out_out_ptr = out_batch_output.storage_->dataf_.data();

    for (int c = 0; c < num_coedges_out; ++c) {
        // 导出数据
        if (EXPORT_ENABLED) {
            output_layer_input_concat.emplace_back(out_in_ptr + c * 90, out_in_ptr + c * 90 + 90);
            output_layer_mlp_output_data.emplace_back(out_out_ptr + c * 30, out_out_ptr + c * 30 + 30);
        }

        // 保存到coedge
        coedges[c].output_face_state.resize(30);
        memcpy(coedges[c].output_face_state.data(), out_out_ptr + c * 30, 30 * sizeof(float));
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

        for (int coedge_id : face.coedge_ids) {
            if (coedge_id >= 0 && coedge_id < (int)coedges.size()) {
                const auto& coedge_state = coedges[coedge_id].output_face_state;
                for (int i = 0; i < 30; ++i) {
                    face.output_state[i] = std::max(face.output_state[i], coedge_state[i]);
                }
            }
        }

        // 小面(coedges<30)初始化为0并应用ReLU，大面初始化为-1e9不应用ReLU
        if (is_small_face) {
            for (int i = 0; i < 30; ++i) {
                face.output_state[i] = std::max(0.0f, face.output_state[i]);
            }
        }

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

    // ========================================================================
    // 导出预测结果到 cpp_results/ — 始终执行，不受调试开关控制
    // ========================================================================
    DBG_LOG << "  [Export] Saving predictions..." << std::endl;

    // 确保目录存在
    fs::create_directories("cpp_results");

    // 创建原始顺序的预测数据
    std::vector<std::vector<float>> probs_original_order(num_faces);
    std::vector<int> predictions_original_order(num_faces);
    std::vector<float> confidences_original_order(num_faces);

    for (int inference_id = 0; inference_id < num_faces; ++inference_id) {
        int original_id = faces[inference_id].face_id;

        // 获取该 Face 的概率和预测
        std::vector<float> face_probs(27);
        float max_prob = -1e9f;
        int pred_class = 0;

        for (int c = 0; c < 27; ++c) {
            face_probs[c] = probs.at({inference_id, c});
            if (face_probs[c] > max_prob) {
                max_prob = face_probs[c];
                pred_class = c;
            }
        }

        probs_original_order[original_id] = face_probs;
        predictions_original_order[original_id] = pred_class;
        confidences_original_order[original_id] = max_prob;
    }

    // 导出预测结果
    std::string results_path = "cpp_results/" + base_name + ".results";
    std::ofstream results_file(results_path);
    results_file << "# filename: " << base_name << ".step" << "\n";
    results_file << "# topology: " << num_coedges << " coedges, "
                 << num_faces << " faces, " << num_edges << " edges\n";
    results_file << "# format: face_id predicted_class confidence top3_classes\n";
    results_file << std::scientific << std::setprecision(6);

    for (int original_id = 0; original_id < num_faces; ++original_id) {
        int pred_class = predictions_original_order[original_id];
        float confidence = confidences_original_order[original_id];

        // 找 Top 3
        std::vector<std::pair<int, float>> class_probs;
        for (int c = 0; c < 27; ++c) {
            class_probs.push_back({c, probs_original_order[original_id][c]});
        }
        std::sort(class_probs.begin(), class_probs.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });

        // 输出一行
        results_file << "face_" << original_id << " "
                     << pred_class << " ";

        // 置信度（不用科学计数法，便于阅读）
        results_file << std::defaultfloat << std::setprecision(6) << confidence << " ";

        // Top 3 类别
        results_file << std::scientific << std::setprecision(6);
        for (int i = 0; i < 3 && i < 27; ++i) {
            if (i > 0) results_file << " ";
            results_file << class_probs[i].first << ":"
                         << class_probs[i].second;
        }
        results_file << "\n";
    }
    results_file.close();

    DBG_LOG << "  [Export] Results saved to: " << results_path << " (original face order)" << std::endl;

    // ========================================================================
    // 导出 .seg 分类结果到 cpp_results/ — 始终执行
    // 格式：每行一个整数（face 类别 id），从第一行开始，无注释
    // 同时导出 4 类映射版本到 cpp_results_4class/
    // 映射规则: 0=chamfer, 23=round, 1/12=hole, 其余=other(3)
    // ========================================================================
    std::string seg_path = "cpp_results/" + base_name + ".seg";
    std::ofstream seg_file(seg_path);
    for (int original_id = 0; original_id < num_faces; ++original_id) {
        seg_file << predictions_original_order[original_id] << "\n";
    }
    seg_file.close();

    DBG_LOG << "  [Export] Seg saved to: " << seg_path << std::endl;

    // 4 类映射版本
    fs::create_directories("cpp_results_4class");
    std::string seg4_path = "cpp_results_4class/" + base_name + ".seg";
    std::ofstream seg4_file(seg4_path);
    for (int original_id = 0; original_id < num_faces; ++original_id) {
        int cls27 = predictions_original_order[original_id];
        int cls4;
        switch (cls27) {
            case 0:  cls4 = 0; break;  // chamfer
            case 23: cls4 = 1; break;  // round
            case 1:
            case 12: cls4 = 2; break;  // hole
            default: cls4 = 3; break;  // other
        }
        seg4_file << cls4 << "\n";
    }
    seg4_file.close();

    DBG_LOG << "  [Export] 4-class seg saved to: " << seg4_path << std::endl;

    auto file_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> file_elapsed = file_end - file_start;
    {
        std::lock_guard<std::mutex> lock(g_console_mutex);
        INFO_LOG << fs::absolute(step_path).string()
                 << " -> [✓] F:" << num_faces << " E:" << num_edges
                 << " (" << std::fixed << std::setprecision(2) << file_elapsed.count() << "s)" << std::endl;
    }

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
    }

    // 清理 Layer 1 临时向量
    {
        std::vector<std::vector<float>>().swap(layer1_input_concat);
        std::vector<std::vector<float>>().swap(layer1_mlp_output_data);
        std::vector<std::vector<float>>().swap(layer1_face_pooling_data);
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
    }

    // 清理其他临时数据
    {
        std::vector<std::vector<float>>().swap(uvnet_surface_features);
        std::vector<std::vector<float>>().swap(uvnet_curve_features);
        std::vector<std::vector<float>>().swap(logits_original_order);
        std::vector<std::vector<float>>().swap(probs_original_order);
        std::vector<int>().swap(predictions_original_order);
        std::vector<float>().swap(confidences_original_order);
        std::vector<int>().swap(edge_representatives);
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
    // 权重文件搜索顺序：版本化文件名 → 通用文件名
    std::vector<std::string> weights_candidates = {
        "inference_data/state_dict_v4.npz",
        "inference_data/state_dict.npz",
        "bin/inference_data/state_dict_v4.npz",
        "bin/inference_data/state_dict.npz"
    };
    std::string weights_file;
    for (const auto& candidate : weights_candidates) {
        if (fs::exists(candidate)) {
            weights_file = candidate;
            break;
        }
    }
    if (weights_file.empty()) {
        ERR_LOG << "[Error] Cannot find weights file! Searched:" << std::endl;
        for (const auto& c : weights_candidates) {
            ERR_LOG << "  " << c << std::endl;
        }
        return -1;
    }

    // STEP 文件目录搜索
    std::string step_dir;
    for (const auto& d : {"inference_data/step_files", "bin/inference_data/step_files"}) {
        if (fs::exists(d)) { step_dir = d; break; }
    }
    if (step_dir.empty()) step_dir = "inference_data/step_files";

    DBG_LOG << "\n[Model] Loading weights: " << weights_file << std::endl;

    // NPZ 只加载一次，然后为每个线程创建独立的模型副本
    cnpy::npz_t npz = cnpy::npz_load(weights_file);
    DBG_LOG << "[Model] NPZ loaded, " << npz.size() << " keys" << std::endl;

    // 解析线程数参数
    int num_threads = (int)std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 1;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--threads" && i + 1 < argc) {
            num_threads = std::atoi(argv[++i]);
            if (num_threads < 1) num_threads = 1;
        } else if (arg == "--force") {
            g_force_overwrite = true;
        }
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

    int total_files = (int)step_files.size();

    // 线程数不超过文件数
    int effective_threads = std::min(num_threads, total_files);

    // 为每个线程创建独立的模型实例（确保 UVNet 编码器的 params map 线程安全）
    std::vector<std::shared_ptr<BRepNetImpl>> models;
    models.reserve(effective_threads);
    for (int t = 0; t < effective_threads; ++t) {
        models.push_back(load_model(npz));
    }
    INFO_LOG << "[Model] Loaded " << effective_threads << " model instances for "
             << effective_threads << " threads" << std::endl;

    // ========================================================================
    // 4. 并行推理
    // ========================================================================
    INFO_LOG << "[Parallel] Using " << effective_threads << " threads for "
             << total_files << " files" << std::endl;

    auto total_start = std::chrono::high_resolution_clock::now();

    // 原子计数器：线程安全的任务分配
    std::atomic<size_t> file_index(0);

    // 工作线程函数
    auto worker = [&](int thread_id) {
        while (true) {
            size_t idx = file_index.fetch_add(1);
            if (idx >= step_files.size()) break;

            run_inference_with_export(step_files[idx], models[thread_id], exporter);
        }
    };

    // 启动线程
    std::vector<std::thread> threads;
    threads.reserve(effective_threads);
    for (int t = 0; t < effective_threads; ++t) {
        threads.emplace_back(worker, t);
    }

    // 等待所有线程完成
    for (auto& t : threads) {
        t.join();
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);

    // ========================================================================
    // 5. 打印导出清单
    // ========================================================================
    INFO_LOG << "\n" << std::string(70, '=') << std::endl;
    INFO_LOG << "All " << total_files << " files completed!" << std::endl;
    INFO_LOG << "Threads: " << effective_threads << std::endl;
    INFO_LOG << "Total time: " << total_duration.count() / 1000.0 << " seconds" << std::endl;
    INFO_LOG << "Average time per file: " << (total_duration.count() / total_files) << " ms" << std::endl;
    INFO_LOG << std::string(70, '=') << std::endl;

    return 0;
}
