#pragma once
#include "DebugControl.h"
#include "VersionConfig.h"
#include "BRepTorch.h"
#include "UVNet.h"
#include "cnpy.h"
#include <vector>
#include <map>
#include <set>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cmath>

using Tensor = breptorch::Tensor;
using namespace breptorch::nn;

// ============================================================================
// BRepNet C++ 推理引擎
// 按照 refactoring_report.md 的简化逻辑实现
// 核心思想：不构建 Psi 矩阵，直接遍历拓扑结构
// ============================================================================

// 1. 简单的 MLP
struct BRepNetMLPImpl : Module {
    SequentialPtr mlp{ nullptr };
    float dropout_p;

    BRepNetMLPImpl(int input_size, int hidden_size, int output_size, bool final_layer, float dropout = 0.3f)
        : dropout_p(dropout) {
        mlp = register_module("mlp", Sequential());

        // 第一层：linear_0 + dropout_0 + relu_0
        mlp->push_back("linear_0", Linear(LinearOptions(input_size, hidden_size).bias(true)));
        if (dropout_p > 0.0f) {
            mlp->push_back("dropout_0", Dropout(dropout_p));
        }
        mlp->push_back("relu_0", ReLU());

        // 第二层：linear_1 + dropout_1 + relu_1（可选）
        mlp->push_back("linear_1", Linear(LinearOptions(hidden_size, output_size).bias(!final_layer)));
        if (dropout_p > 0.0f && !final_layer) {
            mlp->push_back("dropout_1", Dropout(dropout_p));
        }
        if (!final_layer) {
            mlp->push_back("relu_1", ReLU());
        }
    }

    Tensor forward(Tensor x) {
        return mlp->forward(x);
    }

    // 参数同步（当前为空操作，权重通过 named_parameters 直接加载）
    void sync_parameters() {}
};
TORCH_MODULE(BRepNetMLP)


// 2. Coedge 数据结构
// 导师的话：让 OCC 遍历每一个 coedge，coedge 可以加一个属性，比如 parentFace
struct CoedgeData {
    int coedge_id;
    int parent_face_id;
    int mate_face_id;
    int edge_id;

    // UV-Net 提取的初始特征
    std::vector<float> parent_face_features;  // 64 维
    std::vector<float> mate_face_features;    // 64 维
    std::vector<float> edge_features;         // 64 维

    // Layer 0 的状态（一阶邻居，MLP-G）
    std::vector<float> layer0_face_state;  // 30 维
    std::vector<float> layer0_edge_state;  // 30 维

    // Layer 1 的状态（二阶邻居，MLP-1）
    std::vector<float> layer1_face_state;  // 30 维
    std::vector<float> layer1_edge_state;  // 30 维

    // Output layer 的状态（三阶邻居，MLP-2）
    std::vector<float> output_face_state;  // 30 维
};


// 3. Face 数据结构
// 导师的话：遍历 face，可以找出 face 的所有 coedge
struct FaceData {
    int face_id;
    std::vector<int> coedge_ids;  // 该 face 的所有 coedge

    // Layer 0 的状态（一阶邻居）
    std::vector<float> layer0_state;  // 30 维

    // Layer 1 的状态（二阶邻居）
    std::vector<float> layer1_state;  // 30 维

    // Output layer 的状态（三阶邻居，即最终 embedding）
    std::vector<float> output_state;  // 30 维
};


// 4. Edge 数据结构
struct EdgeData {
    int edge_id;
    std::vector<int> coedge_ids;  // 该 edge 的所有 coedge

    // Layer 0 的状态（一阶邻居）
    std::vector<float> layer0_state;  // 30 维

    // Layer 1 的状态（二阶邻居）
    std::vector<float> layer1_state;  // 30 维
};


// 5. BRepNet 主网络
struct BRepNetImpl : Module {
    UVNetSurfaceEncoder surf_enc;
    UVNetCurveEncoder curve_enc;
#if BREPNET_VERSION == 4
    UVNetSurfaceEncoder surf_enc2;
#endif

    // Layer 0 MLP (一阶邻居，MLP-G-surface/edge)
    BRepNetMLP layer0_mlp{ nullptr };

    // Layer 1 MLP (二阶邻居，MLP-1-surface/edge)
    BRepNetMLP layer1_mlp{ nullptr };

    // Output layer MLP (三阶邻居，MLP-2-surface)
    BRepNetMLP output_mlp{ nullptr };

    // Classification layer
    LinearPtr classification_layer{ nullptr };

    int num_classes;

    BRepNetImpl(int n_classes) : num_classes(n_classes) {
        // UV-Net encoders (output 64-dim features)
        // 直接创建 shared_ptr 并使用基类赋值运算符
        std::shared_ptr<UVNetSurfaceEncoderImpl> surf_enc_ptr(new UVNetSurfaceEncoderImpl());
        std::shared_ptr<UVNetCurveEncoderImpl> curve_enc_ptr(new UVNetCurveEncoderImpl());
        
        // 使用 std::shared_ptr 的基类赋值，避免触发 TORCH_MODULE 的模板构造函数
        static_cast<std::shared_ptr<UVNetSurfaceEncoderImpl>&>(surf_enc) = surf_enc_ptr;
        static_cast<std::shared_ptr<UVNetCurveEncoderImpl>&>(curve_enc) = curve_enc_ptr;
        modules_["surface_encoder"] = surf_enc_ptr;
        modules_["curve_encoder"] = curve_enc_ptr;
#if BREPNET_VERSION == 4
        std::shared_ptr<UVNetSurfaceEncoderImpl> surf_enc2_ptr(new UVNetSurfaceEncoderImpl());
        static_cast<std::shared_ptr<UVNetSurfaceEncoderImpl>&>(surf_enc2) = surf_enc2_ptr;
        modules_["surface_encoder2"] = surf_enc2_ptr;
#endif

        // Layer 0: input 192 -> output 60
        // Input: parent_face (64) + mate_face (64) + edge (64) = 192
        // Output: 60 (split into face:30 + edge:30)
        layer0_mlp = register_module("layer_0.mlp", BRepNetMLP(192, 60, 60, false));

        // Layer 1: input 90 -> output 60
        // Input: parent_face (30) + mate_face (30) + edge (30) = 90
        // Output: 60 (split into face:30 + edge:30)
        layer1_mlp = register_module("layer_1.mlp", BRepNetMLP(90, 60, 60, false));

        // Output layer: input 90 -> output 30
        // Input: parent_face (30) + mate_face (30) + edge (30) = 90
        output_mlp = register_module("output_layer.mlp", BRepNetMLP(90, 30, 30, true));  // final_layer=true

        // Classification: input 30 -> output num_classes
        classification_layer = register_module("classification_layer",
            Linear(LinearOptions(30, num_classes).bias(true)));
    }

    // 主 forward 函数
    // 导师的话：C++ 的流程特别简单，遍历 coedge → 遍历 face → MaxPooling
    Tensor forward(
        std::vector<CoedgeData>& coedges,
        std::vector<FaceData>& faces,
        std::vector<EdgeData>& edges) {

        // 设置输出精度为10位小数，方便与Python对比
        if (DebugControl::instance().shouldDebug()) {
            std::cout << std::fixed << std::setprecision(10);
        }

        DBG_LOG << "\n================================================================================\n";
        DBG_LOG << "Forward Propagation Started\n";
        DBG_LOG << "================================================================================\n";
        DBG_LOG << "[Input Data]\n";
        DBG_LOG << "  Coedges: " << coedges.size() << "\n";
        DBG_LOG << "  Faces: " << faces.size() << "\n";
        DBG_LOG << "  Edges: " << edges.size() << "\n";
        DBG_LOG << std::endl;

        // ====================================================================
        // Layer 0: 一阶邻居更新 (MLP-G-surface/edge)
        // ====================================================================
        DBG_LOG << "\n================================================================================\n";
        DBG_LOG << "Layer 0 - First Order Neighbors (MLP-G)\n";
        DBG_LOG << "================================================================================\n";

        // 步骤1: 遍历每个 coedge，计算其 MLP 输出
        // 导师的话：对每个 coedge，找出其 parent face、mate coedge 的 parent face、以及 edge
        int processed_coedges = 0;

        // 诊断输出：对所有 coedge 记录简单的统计信息
        std::ofstream diag_mlp;
        if (EXPORT_ENABLED) {
            diag_mlp.open("cpp_feature_maps/layer0_mlp_all_coedges_stats.txt");
            diag_mlp << "Layer 0 MLP 输入/输出统计\n";
            diag_mlp << "格式: Coedge_ID, Parent_Face_ID, Input_Min, Input_Max, Input_Mean, FaceState_Min, FaceState_Max\n\n";
        }

        for (auto& coedge : coedges) {
            // 构建输入：parent_face (64) + mate_face (64) + edge (64) = 192
            std::vector<float> input;
            input.insert(input.end(), coedge.parent_face_features.begin(), coedge.parent_face_features.end());
            input.insert(input.end(), coedge.mate_face_features.begin(), coedge.mate_face_features.end());
            input.insert(input.end(), coedge.edge_features.begin(), coedge.edge_features.end());

            // 转换为 Tensor
            Tensor input_tensor = breptorch::from_blob(input.data(), {1, 192}, breptorch::kFloat32).clone();

            // 通过 MLP
            Tensor output = layer0_mlp->forward(input_tensor);  // (1, 60)

            // 分离 face 和 edge 输出 (各30维)
            coedge.layer0_edge_state.resize(30);
            coedge.layer0_face_state.resize(30);
            for (int i = 0; i < 30; ++i) {
                coedge.layer0_edge_state[i] = output.at({0, i});      // 前 30 维是 edge
                coedge.layer0_face_state[i] = output.at({0, i + 30}); // 后 30 维是 face
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

        // 调试：仅对前 3 个 coedge 打印详细信息
        processed_coedges = 0;
        for (auto& coedge : coedges) {
            if (DebugControl::instance().shouldDebug() && processed_coedges < 3) {
                DBG_LOG << "\n[DEBUG Layer 0] Coedge " << coedge.coedge_id << " (Face " << coedge.parent_face_id << ")" << std::endl;
                DBG_LOG << "  face_state (first 5): ";
                for (int i = 0; i < 5; ++i) DBG_PRINTF("%.4f ", coedge.layer0_face_state[i]);
                DBG_LOG << std::endl;
            }
            processed_coedges++;
            if (processed_coedges >= 3) break;
        }

        // 步骤2: 遍历每个 face，MaxPooling 其所有 coedge 的状态
        // 恢复为 MaxPooling（与 Python 端一致）
        DBG_LOG << "\n[Layer 0 Face Pooling (MaxPooling)]" << std::endl;

        // 诊断文件：记录每个面的 MaxPooling 过程
        std::ofstream diag_pool;
        if (EXPORT_ENABLED) {
            diag_pool.open("cpp_feature_maps/layer0_face_pooling_all_faces_stats.txt");
            diag_pool << "Layer 0 Face MaxPooling 统计\n";
            diag_pool << "格式: Face_ID, Num_Coedges, Coedge_Count, Pooled_Min, Pooled_Max, Pooled_Mean\n\n";
        }

        for (auto& face : faces) {
            face.layer0_state.resize(30, 0.0f);  // 初始化为0，与Python零填充一致

            int coedge_count = 0;
            for (int coedge_id : face.coedge_ids) {
                if (coedge_id >= 0 && coedge_id < (int)coedges.size()) {
                    const auto& coedge_state = coedges[coedge_id].layer0_face_state;
                    for (int i = 0; i < 30; ++i) {
                        face.layer0_state[i] = std::max(face.layer0_state[i], coedge_state[i]);  // 取最大值
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

            // 调试：打印 Face 0 的 MaxPooling 结果
            if (DebugControl::instance().shouldDebug() && face.face_id == 0) {
                DBG_LOG << "  Face 0 (has " << face.coedge_ids.size() << " coedges):" << std::endl;
                DBG_LOG << "    Coedge IDs: ";
                for (int i = 0; i < std::min(10, (int)face.coedge_ids.size()); ++i) {
                    DBG_LOG << face.coedge_ids[i] << " ";
                }
                DBG_LOG << std::endl;
                DBG_LOG << "    Hf[0, :10] (MaxPooled face_state): ";
                for (int i = 0; i < 10; ++i) DBG_LOG << face.layer0_state[i] << " ";
                DBG_LOG << std::endl;
            }
        }

        if (diag_pool.is_open()) diag_pool.close();


        DBG_LOG << "\n[Layer 0] Completed - Hf: [" << faces.size() << ", 30], He: [" << edges.size() << ", 30]" << std::endl;

        // ====================================================================
        // Layer 1: 二阶邻居更新 (MLP-1-surface/edge)
        // ====================================================================
        DBG_LOG << "\n================================================================================\n";
        DBG_LOG << "Layer 1 - Second Order Neighbors (MLP-1)\n";
        DBG_LOG << "================================================================================\n";

        // 导师的话：用相同的步骤，根据已有的状态生成二阶的邻居状态
        for (auto& coedge : coedges) {
            // 构建输入：parent_face (30) + mate_face (30) + edge (30) = 90
            std::vector<float> input;
            input.insert(input.end(), faces[coedge.parent_face_id].layer0_state.begin(),
                         faces[coedge.parent_face_id].layer0_state.end());
            input.insert(input.end(), faces[coedge.mate_face_id].layer0_state.begin(),
                         faces[coedge.mate_face_id].layer0_state.end());
            // V123: use coedge state directly (no edge maxpool)
            input.insert(input.end(), coedge.layer0_edge_state.begin(),
                         coedge.layer0_edge_state.end());

            // 调试：打印 Coedge 0 的输入（展示三个实体）
            if (DebugControl::instance().shouldDebug() && coedge.coedge_id == 0) {
                DBG_LOG << "\n[Layer 1 MLP Input] Coedge 0:" << std::endl;
                DBG_LOG << "  Input shape: [1, 90]" << std::endl;
                DBG_LOG << "  Input composition: parent_face(30) + mate_face(30) + edge(30)" << std::endl;
                DBG_LOG << "  parent_face[:10]: ";
                for (int i = 0; i < 10; ++i) DBG_LOG << input[i] << " ";
                DBG_LOG << std::endl;
                DBG_LOG << "  mate_face[:10]: ";
                for (int i = 30; i < 40; ++i) DBG_LOG << input[i] << " ";
                DBG_LOG << std::endl;
                DBG_LOG << "  edge[:10]: ";
                for (int i = 60; i < 70; ++i) DBG_LOG << input[i] << " ";
                DBG_LOG << std::endl;
            }

            // 通过 MLP
            Tensor input_tensor = breptorch::from_blob(input.data(), {1, 90}, breptorch::kFloat32).clone();
            Tensor output = layer1_mlp->forward(input_tensor);  // (1, 60)

            // 调试：打印 Coedge 0 的输出
            if (DebugControl::instance().shouldDebug() && coedge.coedge_id == 0) {
                DBG_LOG << "\n[Layer 1 MLP Output] Coedge 0:" << std::endl;
                DBG_LOG << "  Output shape: [1, 60]" << std::endl;
                DBG_LOG << "  Output composition: edge_state(30) + face_state(30)" << std::endl;
                DBG_LOG << "  edge_state[:10]: ";
                for (int i = 0; i < 10; ++i) DBG_LOG << output.at({0, i}) << " ";
                DBG_LOG << std::endl;
                DBG_LOG << "  face_state[:10]: ";
                for (int i = 30; i < 40; ++i) DBG_LOG << output.at({0, i}) << " ";
                DBG_LOG << std::endl;
            }

            // 分离 face 和 edge 输出
            // 注意：Python 的 MLP 输出顺序是 [edge_state, face_state]
            coedge.layer1_edge_state.resize(30);
            coedge.layer1_face_state.resize(30);
            for (int i = 0; i < 30; ++i) {
                coedge.layer1_edge_state[i] = output.at({0, i});      // 前 30 维是 edge
                coedge.layer1_face_state[i] = output.at({0, i + 30}); // 后 30 维是 face
            }
        }

        // MaxPooling
        DBG_LOG << "\n[Layer 1 Face Pooling (MaxPooling)]" << std::endl;
        for (auto& face : faces) {
            face.layer1_state.resize(30, 0.0f);  // 初始化为0，与Python零填充一致

            int coedge_count = 0;
            for (int coedge_id : face.coedge_ids) {
                if (coedge_id >= 0 && coedge_id < (int)coedges.size()) {
                    const auto& coedge_state = coedges[coedge_id].layer1_face_state;
                    for (int i = 0; i < 30; ++i) {
                        face.layer1_state[i] = std::max(face.layer1_state[i], coedge_state[i]);  // 取最大值
                    }
                    coedge_count++;
                }
            }

            // 调试：打印 Face 0 的 MaxPooling 结果
            if (DebugControl::instance().shouldDebug() && face.face_id == 0) {
                DBG_LOG << "  Face 0 (has " << face.coedge_ids.size() << " coedges):" << std::endl;
                DBG_LOG << "    Coedge IDs: ";
                for (int i = 0; i < std::min(10, (int)face.coedge_ids.size()); ++i) {
                    DBG_LOG << face.coedge_ids[i] << " ";
                }
                DBG_LOG << std::endl;
                DBG_LOG << "    Hf[0, :10] (MaxPooled face_state): ";
                for (int i = 0; i < 10; ++i) DBG_LOG << face.layer1_state[i] << " ";
                DBG_LOG << std::endl;
            }
        }


        DBG_LOG << "\n[Layer 1] Completed - Hf: [" << faces.size() << ", 30], He: [" << edges.size() << ", 30]" << std::endl;

        // ====================================================================
        // Output Layer: 三阶邻居更新 (MLP-2-surface)
        // ====================================================================
        DBG_LOG << "\n================================================================================\n";
        DBG_LOG << "Output Layer - Third Order Neighbors (MLP-2)\n";
        DBG_LOG << "================================================================================\n";
        DBG_LOG << "Note: Output Layer only processes Face (no Edge output)\n" << std::endl;

        // 导师的话：重复此过程生成三阶邻居状态
        for (auto& coedge : coedges) {
            // 构建输入：parent_face (30) + mate_face (30) + edge (30) = 90
            std::vector<float> input;
            input.insert(input.end(), faces[coedge.parent_face_id].layer1_state.begin(),
                         faces[coedge.parent_face_id].layer1_state.end());
            input.insert(input.end(), faces[coedge.mate_face_id].layer1_state.begin(),
                         faces[coedge.mate_face_id].layer1_state.end());
            // V123: use coedge state directly (no edge maxpool)
            input.insert(input.end(), coedge.layer1_edge_state.begin(),
                         coedge.layer1_edge_state.end());

            // 调试：打印 Coedge 0 的输入（展示三个实体）
            if (DebugControl::instance().shouldDebug() && coedge.coedge_id == 0) {
                DBG_LOG << "\n[Output Layer MLP Input] Coedge 0:" << std::endl;
                DBG_LOG << "  Input shape: [1, 90]" << std::endl;
                DBG_LOG << "  Input composition: parent_face(30) + mate_face(30) + edge(30)" << std::endl;
                DBG_LOG << "  parent_face[:10]: ";
                for (int i = 0; i < 10; ++i) DBG_LOG << input[i] << " ";
                DBG_LOG << std::endl;
                DBG_LOG << "  mate_face[:10]: ";
                for (int i = 30; i < 40; ++i) DBG_LOG << input[i] << " ";
                DBG_LOG << std::endl;
                DBG_LOG << "  edge[:10]: ";
                for (int i = 60; i < 70; ++i) DBG_LOG << input[i] << " ";
                DBG_LOG << std::endl;
            }

            // 通过 MLP
            Tensor input_tensor = breptorch::from_blob(input.data(), {1, 90}, breptorch::kFloat32).clone();
            Tensor output = output_mlp->forward(input_tensor);  // (1, 30)

            // 调试：打印 Coedge 0 的输出
            if (DebugControl::instance().shouldDebug() && coedge.coedge_id == 0) {
                DBG_LOG << "\n[Output Layer MLP Output] Coedge 0:" << std::endl;
                DBG_LOG << "  Output shape: [1, 30]" << std::endl;
                DBG_LOG << "  Output composition: face_state(30) only" << std::endl;
                DBG_LOG << "  face_state[:10]: ";
                for (int i = 0; i < 10; ++i) DBG_LOG << output.at({0, i}) << " ";
                DBG_LOG << std::endl;
            }

            // 只有 face 输出
            coedge.output_face_state.resize(30);
            for (int i = 0; i < 30; ++i) {
                coedge.output_face_state[i] = output.at({0, i});
            }
        }

        // MaxPooling
        DBG_LOG << "\n[Output Layer Face Pooling (MaxPooling)]" << std::endl;
        for (auto& face : faces) {
            const int max_coedges_per_face = 30;
            bool is_small_face = (int)face.coedge_ids.size() < max_coedges_per_face;
            float init_value = is_small_face ? 0.0f : -1e9f;
            face.output_state.assign(30, init_value);

            if (!is_small_face) {
                DBG_LOG << "[Debug] Face " << face.face_id << " is BIG FACE with "
                          << face.coedge_ids.size() << " coedges, init=" << init_value << std::endl;
            }

            int coedge_count = 0;
            for (int coedge_id : face.coedge_ids) {
                if (coedge_id >= 0 && coedge_id < (int)coedges.size()) {
                    const auto& coedge_state = coedges[coedge_id].output_face_state;
                    for (int i = 0; i < 30; ++i) {
                        face.output_state[i] = std::max(face.output_state[i], coedge_state[i]);
                    }
                    coedge_count++;
                }
            }

            if (is_small_face) {
                for (int i = 0; i < 30; ++i) {
                    face.output_state[i] = std::max(0.0f, face.output_state[i]);
                }
            }

            // 调试：打印 Face 0 的所有 coedge 的 face_state
            if (DebugControl::instance().shouldDebug() && face.face_id == 0) {
                DBG_LOG << "\n[Debug] Face 0 coedge face_states:" << std::endl;
                for (int coedge_id : face.coedge_ids) {
                    if (coedge_id >= 0 && coedge_id < (int)coedges.size()) {
                        DBG_LOG << "  Coedge " << coedge_id << " face_state[:10]: ";
                        for (int i = 0; i < 10; ++i) {
                            DBG_LOG << coedges[coedge_id].output_face_state[i] << " ";
                        }
                        DBG_LOG << std::endl;
                    }
                }
            }

            // 调试：打印 Face 0 的 MaxPooling 结果
            if (DebugControl::instance().shouldDebug() && face.face_id == 0) {
                DBG_LOG << "\n  Face 0 (has " << face.coedge_ids.size() << " coedges):" << std::endl;
                DBG_LOG << "    Coedge IDs: ";
                for (int i = 0; i < std::min(10, (int)face.coedge_ids.size()); ++i) {
                    DBG_LOG << face.coedge_ids[i] << " ";
                }
                DBG_LOG << std::endl;
                DBG_LOG << "    Hf[0, :10] (MaxPooled face_state): ";
                for (int i = 0; i < 10; ++i) DBG_LOG << face.output_state[i] << " ";
                DBG_LOG << std::endl;
            }
        }

        DBG_LOG << "\n[Output Layer] Completed - Face embeddings: [" << faces.size() << ", 30]" << std::endl;

        // ====================================================================
        // Classification Layer
        // ====================================================================
        DBG_LOG << "\n================================================================================\n";
        DBG_LOG << "Classification Layer\n";
        DBG_LOG << "================================================================================\n";

        // 构建 face embeddings tensor
        std::vector<float> face_embeddings_data;
        for (const auto& face : faces) {
            face_embeddings_data.insert(face_embeddings_data.end(),
                                        face.output_state.begin(),
                                        face.output_state.end());
        }

        Tensor face_embeddings = breptorch::from_blob(face_embeddings_data.data(),
                                                      {(int64_t)faces.size(), 30},
                                                      breptorch::kFloat32).clone();

        // 调试：检查 Face 2 和 Face 26 的 embedding
        if (DebugControl::instance().shouldDebug() && faces.size() > 26) {
            DBG_LOG << "\n[DEBUG Classification] Face 2 embedding (first 10): ";
            for (int i = 0; i < 10; ++i) {
                DBG_LOG << face_embeddings.at({2, (int64_t)i}) << " ";
            }
            DBG_LOG << std::endl;

            DBG_LOG << "[DEBUG Classification] Face 26 embedding (first 10): ";
            for (int i = 0; i < 10; ++i) {
                DBG_LOG << face_embeddings.at({26, (int64_t)i}) << " ";
            }
            DBG_LOG << std::endl;
        }

        Tensor logits = classification_layer->forward(face_embeddings);

        // 调试：检查 Classification Layer 的权重
        DBG_LOG << "\n[DEBUG Classification] Weight info: shape [" << classification_layer->weight.size(0)
                  << ", " << classification_layer->weight.size(1) << "]" << std::endl;
        DBG_LOG << "[DEBUG Classification] Bias info: shape [" << classification_layer->bias.size(0) << "]" << std::endl;

        DBG_LOG << "[Classification] Output logits shape: [" << logits.size(0) << ", " << logits.size(1) << "]" << std::endl;

        return logits;
    }
};
TORCH_MODULE(BRepNet)
