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


// 4. BRepNet 主网络
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
        std::vector<FaceData>& faces) {

        // ====================================================================
        // Layer 0: 一阶邻居更新 (MLP-G-surface/edge)
        // ====================================================================

        // 步骤1: 遍历每个 coedge，计算其 MLP 输出
        // 导师的话：对每个 coedge，找出其 parent face、mate coedge 的 parent face、以及 edge
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
        }

        // 步骤2: 遍历每个 face，MaxPooling 其所有 coedge 的状态
        // 恢复为 MaxPooling（与 Python 端一致）
        for (auto& face : faces) {
            face.layer0_state.resize(30, 0.0f);  // 初始化为0，与Python零填充一致

            for (int coedge_id : face.coedge_ids) {
                if (coedge_id >= 0 && coedge_id < (int)coedges.size()) {
                    const auto& coedge_state = coedges[coedge_id].layer0_face_state;
                    for (int i = 0; i < 30; ++i) {
                        face.layer0_state[i] = std::max(face.layer0_state[i], coedge_state[i]);  // 取最大值
                    }
                }
            }
        }

        // ====================================================================
        // Layer 1: 二阶邻居更新 (MLP-1-surface/edge)
        // ====================================================================

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

            // 通过 MLP
            Tensor input_tensor = breptorch::from_blob(input.data(), {1, 90}, breptorch::kFloat32).clone();
            Tensor output = layer1_mlp->forward(input_tensor);  // (1, 60)

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
        for (auto& face : faces) {
            face.layer1_state.resize(30, 0.0f);  // 初始化为0，与Python零填充一致

            for (int coedge_id : face.coedge_ids) {
                if (coedge_id >= 0 && coedge_id < (int)coedges.size()) {
                    const auto& coedge_state = coedges[coedge_id].layer1_face_state;
                    for (int i = 0; i < 30; ++i) {
                        face.layer1_state[i] = std::max(face.layer1_state[i], coedge_state[i]);  // 取最大值
                    }
                }
            }
        }

        // ====================================================================
        // Output Layer: 三阶邻居更新 (MLP-2-surface)
        // ====================================================================

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

            // 通过 MLP
            Tensor input_tensor = breptorch::from_blob(input.data(), {1, 90}, breptorch::kFloat32).clone();
            Tensor output = output_mlp->forward(input_tensor);  // (1, 30)

            // 只有 face 输出
            coedge.output_face_state.resize(30);
            for (int i = 0; i < 30; ++i) {
                coedge.output_face_state[i] = output.at({0, i});
            }
        }

        // MaxPooling
        for (auto& face : faces) {
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

            if (is_small_face) {
                for (int i = 0; i < 30; ++i) {
                    face.output_state[i] = std::max(0.0f, face.output_state[i]);
                }
            }
        }

        // ====================================================================
        // Classification Layer
        // ====================================================================

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

        Tensor logits = classification_layer->forward(face_embeddings);

        return logits;
    }
};
TORCH_MODULE(BRepNet)
