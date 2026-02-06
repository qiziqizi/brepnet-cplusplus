#pragma once
#include "BRepTorch.h"
#include "UVNet.h"
#include "cnpy.h"
#include <vector>
#include <map>

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

    BRepNetMLPImpl(int input_size, int hidden_size, int output_size, bool final_layer) {
        mlp = register_module("mlp", Sequential());

        // 第一层：linear_0 + ReLU（总是有ReLU）
        mlp->push_back("linear_0", Linear(LinearOptions(input_size, hidden_size).bias(true)));
        mlp->push_back("relu_0", ReLU());

        // 第二层：linear_1 + 可选的ReLU
        // 只有当final_layer=true时，linear_1才没有bias和ReLU
        mlp->push_back("linear_1", Linear(LinearOptions(hidden_size, output_size).bias(!final_layer)));
        if (!final_layer) {
            mlp->push_back("relu_1", ReLU());
        }
    }

    Tensor forward(Tensor x) {
        return mlp->forward(x);
    }
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
    bool use_uvnet = false;
    UVNetSurfaceEncoder surf_enc{ nullptr };
    UVNetCurveEncoder curve_enc{ nullptr };

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
        surf_enc = register_module("surface_encoder", std::make_shared<UVNetSurfaceEncoderImpl>());
        curve_enc = register_module("curve_encoder", std::make_shared<UVNetCurveEncoderImpl>());

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

        std::cout << "\n================================================================================\n";
        std::cout << "Forward Propagation Started\n";
        std::cout << "================================================================================\n";
        std::cout << "[Input Data]\n";
        std::cout << "  Coedges: " << coedges.size() << "\n";
        std::cout << "  Faces: " << faces.size() << "\n";
        std::cout << "  Edges: " << edges.size() << "\n";
        std::cout << std::endl;

        // ====================================================================
        // Layer 0: 一阶邻居更新 (MLP-G-surface/edge)
        // ====================================================================
        std::cout << "\n================================================================================\n";
        std::cout << "Layer 0 - First Order Neighbors (MLP-G)\n";
        std::cout << "================================================================================\n";

        // 步骤1: 遍历每个 coedge，计算其 MLP 输出
        // 导师的话：对每个 coedge，找出其 parent face、mate coedge 的 parent face、以及 edge
        for (auto& coedge : coedges) {
            // 构建输入：parent_face (64) + mate_face (64) + edge (64) = 192
            std::vector<float> input;
            input.insert(input.end(), coedge.parent_face_features.begin(), coedge.parent_face_features.end());
            input.insert(input.end(), coedge.mate_face_features.begin(), coedge.mate_face_features.end());
            input.insert(input.end(), coedge.edge_features.begin(), coedge.edge_features.end());

            // 调试：打印 Coedge 0 的输入（展示三个实体）
            if (coedge.coedge_id == 0) {
                std::cout << "\n[Layer 0 MLP Input] Coedge 0:" << std::endl;
                std::cout << "  Input shape: [1, 192]" << std::endl;
                std::cout << "  Input composition: parent_face(64) + mate_face(64) + edge(64)" << std::endl;
                std::cout << "  parent_face[:10]: ";
                for (int i = 0; i < 10; ++i) std::cout << coedge.parent_face_features[i] << " ";
                std::cout << std::endl;
                std::cout << "  mate_face[:10]: ";
                for (int i = 0; i < 10; ++i) std::cout << coedge.mate_face_features[i] << " ";
                std::cout << std::endl;
                std::cout << "  edge[:10]: ";
                for (int i = 0; i < 10; ++i) std::cout << coedge.edge_features[i] << " ";
                std::cout << std::endl;
            }

            // 通过 MLP
            Tensor input_tensor = breptorch::from_blob(input.data(), {1, 192}, breptorch::kFloat32).clone();
            Tensor output = layer0_mlp->forward(input_tensor);  // (1, 60)

            // 分离 face 和 edge 输出 (各30维)
            // 注意：Python 的 MLP 输出顺序是 [edge_state, face_state]
            // 所以前 30 维是 edge_state，后 30 维是 face_state
            coedge.layer0_edge_state.resize(30);
            coedge.layer0_face_state.resize(30);
            for (int i = 0; i < 30; ++i) {
                coedge.layer0_edge_state[i] = output.at({0, i});      // 前 30 维是 edge
                coedge.layer0_face_state[i] = output.at({0, i + 30}); // 后 30 维是 face
            }

            // 调试：打印 Coedge 0 的输出
            if (coedge.coedge_id == 0) {
                std::cout << "\n[Layer 0 MLP Output] Coedge 0:" << std::endl;
                std::cout << "  Output shape: [1, 60]" << std::endl;
                std::cout << "  Output composition: edge_state(30) + face_state(30)" << std::endl;
                std::cout << "  edge_state[:10]: ";
                for (int i = 0; i < 10; ++i) std::cout << coedge.layer0_edge_state[i] << " ";
                std::cout << std::endl;
                std::cout << "  face_state[:10]: ";
                for (int i = 0; i < 10; ++i) std::cout << coedge.layer0_face_state[i] << " ";
                std::cout << std::endl;
            }
        }

        // 步骤2: 遍历每个 face，MaxPooling 其所有 coedge 的状态
        // 导师的话：把该 face 所有 coedge 的状态 maxpooling，生成 face 的一阶邻居状态
        std::cout << "\n[Layer 0 Face Pooling]" << std::endl;
        for (auto& face : faces) {
            face.layer0_state.resize(30, -1e9f);  // 初始化为负无穷

            for (int coedge_id : face.coedge_ids) {
                if (coedge_id >= 0 && coedge_id < (int)coedges.size()) {
                    const auto& coedge_state = coedges[coedge_id].layer0_face_state;
                    for (int i = 0; i < 30; ++i) {
                        face.layer0_state[i] = std::max(face.layer0_state[i], coedge_state[i]);
                    }
                }
            }

            // 调试：打印 Face 0 的 MaxPooling 结果
            if (face.face_id == 0) {
                std::cout << "  Face 0 (has " << face.coedge_ids.size() << " coedges):" << std::endl;
                std::cout << "    Coedge IDs: ";
                for (int i = 0; i < std::min(10, (int)face.coedge_ids.size()); ++i) {
                    std::cout << face.coedge_ids[i] << " ";
                }
                std::cout << std::endl;
                std::cout << "    Hf[0, :10] (MaxPooled face_state): ";
                for (int i = 0; i < 10; ++i) std::cout << face.layer0_state[i] << " ";
                std::cout << std::endl;
            }
        }

        // 步骤3: 遍历每个 edge，MaxPooling 其所有 coedge 的状态
        std::cout << "\n[Layer 0 Edge Pooling]" << std::endl;
        for (auto& edge : edges) {
            edge.layer0_state.resize(30, -1e9f);

            for (int coedge_id : edge.coedge_ids) {
                if (coedge_id >= 0 && coedge_id < (int)coedges.size()) {
                    const auto& coedge_state = coedges[coedge_id].layer0_edge_state;
                    for (int i = 0; i < 30; ++i) {
                        edge.layer0_state[i] = std::max(edge.layer0_state[i], coedge_state[i]);
                    }
                }
            }

            // 调试：打印 Edge 0 的 MaxPooling 结果
            if (edge.edge_id == 0) {
                std::cout << "  Edge 0 (has " << edge.coedge_ids.size() << " coedges):" << std::endl;
                std::cout << "    Coedge IDs: ";
                for (int i = 0; i < std::min(10, (int)edge.coedge_ids.size()); ++i) {
                    std::cout << edge.coedge_ids[i] << " ";
                }
                std::cout << std::endl;
                std::cout << "    He[0, :10] (MaxPooled edge_state): ";
                for (int i = 0; i < 10; ++i) std::cout << edge.layer0_state[i] << " ";
                std::cout << std::endl;
            }
        }

        std::cout << "\n[Layer 0] Completed - Hf: [" << faces.size() << ", 30], He: [" << edges.size() << ", 30]" << std::endl;

        // ====================================================================
        // Layer 1: 二阶邻居更新 (MLP-1-surface/edge)
        // ====================================================================
        std::cout << "\n================================================================================\n";
        std::cout << "Layer 1 - Second Order Neighbors (MLP-1)\n";
        std::cout << "================================================================================\n";

        // 导师的话：用相同的步骤，根据已有的状态生成二阶的邻居状态
        for (auto& coedge : coedges) {
            // 构建输入：parent_face (30) + mate_face (30) + edge (30) = 90
            std::vector<float> input;
            input.insert(input.end(), faces[coedge.parent_face_id].layer0_state.begin(),
                         faces[coedge.parent_face_id].layer0_state.end());
            input.insert(input.end(), faces[coedge.mate_face_id].layer0_state.begin(),
                         faces[coedge.mate_face_id].layer0_state.end());
            input.insert(input.end(), edges[coedge.edge_id].layer0_state.begin(),
                         edges[coedge.edge_id].layer0_state.end());

            // 调试：打印 Coedge 0 的输入（展示三个实体）
            if (coedge.coedge_id == 0) {
                std::cout << "\n[Layer 1 MLP Input] Coedge 0:" << std::endl;
                std::cout << "  Input shape: [1, 90]" << std::endl;
                std::cout << "  Input composition: parent_face(30) + mate_face(30) + edge(30)" << std::endl;
                std::cout << "  parent_face[:10]: ";
                for (int i = 0; i < 10; ++i) std::cout << input[i] << " ";
                std::cout << std::endl;
                std::cout << "  mate_face[:10]: ";
                for (int i = 30; i < 40; ++i) std::cout << input[i] << " ";
                std::cout << std::endl;
                std::cout << "  edge[:10]: ";
                for (int i = 60; i < 70; ++i) std::cout << input[i] << " ";
                std::cout << std::endl;
            }

            // 通过 MLP
            Tensor input_tensor = breptorch::from_blob(input.data(), {1, 90}, breptorch::kFloat32).clone();
            Tensor output = layer1_mlp->forward(input_tensor);  // (1, 60)

            // 调试：打印 Coedge 0 的输出
            if (coedge.coedge_id == 0) {
                std::cout << "\n[Layer 1 MLP Output] Coedge 0:" << std::endl;
                std::cout << "  Output shape: [1, 60]" << std::endl;
                std::cout << "  Output composition: edge_state(30) + face_state(30)" << std::endl;
                std::cout << "  edge_state[:10]: ";
                for (int i = 0; i < 10; ++i) std::cout << output.at({0, i}) << " ";
                std::cout << std::endl;
                std::cout << "  face_state[:10]: ";
                for (int i = 30; i < 40; ++i) std::cout << output.at({0, i}) << " ";
                std::cout << std::endl;
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
        std::cout << "\n[Layer 1 Face Pooling]" << std::endl;
        for (auto& face : faces) {
            face.layer1_state.resize(30, -1e9f);
            for (int coedge_id : face.coedge_ids) {
                if (coedge_id >= 0 && coedge_id < (int)coedges.size()) {
                    const auto& coedge_state = coedges[coedge_id].layer1_face_state;
                    for (int i = 0; i < 30; ++i) {
                        face.layer1_state[i] = std::max(face.layer1_state[i], coedge_state[i]);
                    }
                }
            }

            // 调试：打印 Face 0 的 MaxPooling 结果
            if (face.face_id == 0) {
                std::cout << "  Face 0 (has " << face.coedge_ids.size() << " coedges):" << std::endl;
                std::cout << "    Coedge IDs: ";
                for (int i = 0; i < std::min(10, (int)face.coedge_ids.size()); ++i) {
                    std::cout << face.coedge_ids[i] << " ";
                }
                std::cout << std::endl;
                std::cout << "    Hf[0, :10] (MaxPooled face_state): ";
                for (int i = 0; i < 10; ++i) std::cout << face.layer1_state[i] << " ";
                std::cout << std::endl;
            }
        }

        std::cout << "\n[Layer 1 Edge Pooling]" << std::endl;
        for (auto& edge : edges) {
            edge.layer1_state.resize(30, -1e9f);
            for (int coedge_id : edge.coedge_ids) {
                if (coedge_id >= 0 && coedge_id < (int)coedges.size()) {
                    const auto& coedge_state = coedges[coedge_id].layer1_edge_state;
                    for (int i = 0; i < 30; ++i) {
                        edge.layer1_state[i] = std::max(edge.layer1_state[i], coedge_state[i]);
                    }
                }
            }

            // 调试：打印 Edge 0 的 MaxPooling 结果
            if (edge.edge_id == 0) {
                std::cout << "  Edge 0 (has " << edge.coedge_ids.size() << " coedges):" << std::endl;
                std::cout << "    Coedge IDs: ";
                for (int i = 0; i < std::min(10, (int)edge.coedge_ids.size()); ++i) {
                    std::cout << edge.coedge_ids[i] << " ";
                }
                std::cout << std::endl;
                std::cout << "    He[0, :10] (MaxPooled edge_state): ";
                for (int i = 0; i < 10; ++i) std::cout << edge.layer1_state[i] << " ";
                std::cout << std::endl;
            }
        }

        std::cout << "\n[Layer 1] Completed - Hf: [" << faces.size() << ", 30], He: [" << edges.size() << ", 30]" << std::endl;

        // ====================================================================
        // Output Layer: 三阶邻居更新 (MLP-2-surface)
        // ====================================================================
        std::cout << "\n================================================================================\n";
        std::cout << "Output Layer - Third Order Neighbors (MLP-2)\n";
        std::cout << "================================================================================\n";
        std::cout << "Note: Output Layer only processes Face (no Edge output)\n" << std::endl;

        // 导师的话：重复此过程生成三阶邻居状态
        for (auto& coedge : coedges) {
            // 构建输入：parent_face (30) + mate_face (30) + edge (30) = 90
            std::vector<float> input;
            input.insert(input.end(), faces[coedge.parent_face_id].layer1_state.begin(),
                         faces[coedge.parent_face_id].layer1_state.end());
            input.insert(input.end(), faces[coedge.mate_face_id].layer1_state.begin(),
                         faces[coedge.mate_face_id].layer1_state.end());
            input.insert(input.end(), edges[coedge.edge_id].layer1_state.begin(),
                         edges[coedge.edge_id].layer1_state.end());

            // 调试：打印 Coedge 0 的输入（展示三个实体）
            if (coedge.coedge_id == 0) {
                std::cout << "\n[Output Layer MLP Input] Coedge 0:" << std::endl;
                std::cout << "  Input shape: [1, 90]" << std::endl;
                std::cout << "  Input composition: parent_face(30) + mate_face(30) + edge(30)" << std::endl;
                std::cout << "  parent_face[:10]: ";
                for (int i = 0; i < 10; ++i) std::cout << input[i] << " ";
                std::cout << std::endl;
                std::cout << "  mate_face[:10]: ";
                for (int i = 30; i < 40; ++i) std::cout << input[i] << " ";
                std::cout << std::endl;
                std::cout << "  edge[:10]: ";
                for (int i = 60; i < 70; ++i) std::cout << input[i] << " ";
                std::cout << std::endl;
            }

            // 通过 MLP
            Tensor input_tensor = breptorch::from_blob(input.data(), {1, 90}, breptorch::kFloat32).clone();
            Tensor output = output_mlp->forward(input_tensor);  // (1, 30)

            // 调试：打印 Coedge 0 的输出
            if (coedge.coedge_id == 0) {
                std::cout << "\n[Output Layer MLP Output] Coedge 0:" << std::endl;
                std::cout << "  Output shape: [1, 30]" << std::endl;
                std::cout << "  Output composition: face_state(30) only" << std::endl;
                std::cout << "  face_state[:10]: ";
                for (int i = 0; i < 10; ++i) std::cout << output.at({0, i}) << " ";
                std::cout << std::endl;
            }

            // 只有 face 输出
            coedge.output_face_state.resize(30);
            for (int i = 0; i < 30; ++i) {
                coedge.output_face_state[i] = output.at({0, i});
            }
        }

        // MaxPooling
        std::cout << "\n[Output Layer Face Pooling]" << std::endl;
        for (auto& face : faces) {
            // 注意：使用 0.0f 初始化以匹配 Python 端的 padding 行为
            // Python 端使用 torch.zeros 作为 padding，导致负值被替换为 0
            // 这是 Python 端的一个 bug，但为了与训练模型一致，C++ 端需要复现这个行为
            // 详见：Python端疑问咨询报告-回复.md
            face.output_state.resize(30, 0.0f);  // 使用 0 而不是 -1e9f

            for (int coedge_id : face.coedge_ids) {
                if (coedge_id >= 0 && coedge_id < (int)coedges.size()) {
                    const auto& coedge_state = coedges[coedge_id].output_face_state;
                    for (int i = 0; i < 30; ++i) {
                        face.output_state[i] = std::max(face.output_state[i], coedge_state[i]);
                    }
                }
            }

            // 调试：打印 Face 0 的所有 coedge 的 face_state
            if (face.face_id == 0) {
                std::cout << "\n[Debug] Face 0 coedge face_states:" << std::endl;
                for (int coedge_id : face.coedge_ids) {
                    if (coedge_id >= 0 && coedge_id < (int)coedges.size()) {
                        std::cout << "  Coedge " << coedge_id << " face_state[:10]: ";
                        for (int i = 0; i < 10; ++i) {
                            std::cout << coedges[coedge_id].output_face_state[i] << " ";
                        }
                        std::cout << std::endl;
                    }
                }
            }

            // 调试：打印 Face 0 的 MaxPooling 结果
            if (face.face_id == 0) {
                std::cout << "\n  Face 0 (has " << face.coedge_ids.size() << " coedges):" << std::endl;
                std::cout << "    Coedge IDs: ";
                for (int i = 0; i < std::min(10, (int)face.coedge_ids.size()); ++i) {
                    std::cout << face.coedge_ids[i] << " ";
                }
                std::cout << std::endl;
                std::cout << "    Hf[0, :10] (MaxPooled face_state): ";
                for (int i = 0; i < 10; ++i) std::cout << face.output_state[i] << " ";
                std::cout << std::endl;
            }
        }

        std::cout << "\n[Output Layer] Completed - Face embeddings: [" << faces.size() << ", 30]" << std::endl;

        // ====================================================================
        // Classification Layer
        // ====================================================================
        std::cout << "\n================================================================================\n";
        std::cout << "Classification Layer\n";
        std::cout << "================================================================================\n";

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

        std::cout << "[Classification] Output logits shape: [" << logits.size(0) << ", " << logits.size(1) << "]" << std::endl;

        return logits;
    }
};
TORCH_MODULE(BRepNet)
