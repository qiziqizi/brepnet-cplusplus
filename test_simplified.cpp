#include "BRepNet.h"
#include "BRepNetAdapter.h"
#include "BRepPipeline.h"
// #include "SimpleLogger.h"  // 禁用日志功能，避免 Windows API 依赖
#include <iostream>
#include <iomanip>
#include <chrono>
#include <Windows.h>

int main() {
    SetConsoleOutputCP(936);  // GBK 编码

    std::cout << "=== BRepNet Simplified Version Test ===" << std::endl;

    // ========================================================================
    // 1. 数据预处理
    // ========================================================================
    std::string step_file = "inference_data/test1.step";
    std::string weights_file = "inference_data/state_dict.npz";

    BRepPipeline pipeline;

    // 使用 process 方法完成所有预处理（包括拓扑构建、特征提取、网格生成）
    if (!pipeline.process(step_file)) {
        std::cerr << "[Error] Failed to load and process STEP file!" << std::endl;
        return -1;
    }

    std::cout << "[Simplified] Topology built: "
              << pipeline.coedges.size() << " coedges, "
              << pipeline.unique_faces.Extent() << " faces, "
              << pipeline.unique_edges.Extent() << " edges" << std::endl;

    // ========================================================================
    // 2. 加载模型
    // ========================================================================
    auto model = std::make_shared<BRepNetImpl>(27);

    // 加载权重
    std::cout << "[Simplified] Loading weights from: " << weights_file << std::endl;
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

    // 加载 BRepNet 权重（手动加载）
    auto params = model->named_parameters();
    for (auto& item : npz) {
        std::string key = item.first;

        // 转换 Python 的键名到 C++ 的键名
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

    std::cout << "[Simplified] Weights loaded successfully!" << std::endl;

    // 验证 Layer 0 MLP 权重
    std::cout << "\n[Debug] Verifying Layer 0 MLP weights..." << std::endl;
    auto layer0_params = model->layer0_mlp->named_parameters();
    if (layer0_params.find("mlp.linear_0.weight") != layer0_params.end()) {
        auto w = *layer0_params["mlp.linear_0.weight"];
        std::cout << "[Debug] Layer 0 MLP linear_0.weight shape: [" << w.size(0) << ", " << w.size(1) << "]" << std::endl;
        std::cout << "[Debug] Layer 0 MLP linear_0.weight[0, :10]: ";
        for (int i = 0; i < 10; ++i) {
            std::cout << w.at({0, i}) << " ";
        }
        std::cout << std::endl;
    }
    if (layer0_params.find("mlp.linear_0.bias") != layer0_params.end()) {
        auto b = *layer0_params["mlp.linear_0.bias"];
        std::cout << "[Debug] Layer 0 MLP linear_0.bias[:10]: ";
        for (int i = 0; i < 10; ++i) {
            std::cout << b.at({i}) << " ";
        }
        std::cout << std::endl;
    }

    // 验证 Output Layer MLP 权重
    std::cout << "\n[Debug] Verifying Output Layer MLP weights..." << std::endl;
    auto output_params = model->output_mlp->named_parameters();
    std::cout << "[Debug] Output MLP parameters:" << std::endl;
    for (const auto& p : output_params) {
        std::cout << "  - " << p.first << ": [";
        const auto& shape = p.second->sizes();
        for (size_t i = 0; i < shape.size(); ++i) {
            std::cout << shape[i];
            if (i < shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    if (output_params.find("mlp.linear_0.weight") != output_params.end()) {
        auto w = *output_params["mlp.linear_0.weight"];
        std::cout << "[Debug] Output MLP linear_0.weight shape: [" << w.size(0) << ", " << w.size(1) << "]" << std::endl;
        std::cout << "[Debug] Output MLP linear_0.weight[0, :10]: ";
        for (int i = 0; i < 10; ++i) {
            std::cout << w.at({0, i}) << " ";
        }
        std::cout << std::endl;
    }
    if (output_params.find("mlp.linear_0.bias") != output_params.end()) {
        auto b = *output_params["mlp.linear_0.bias"];
        std::cout << "[Debug] Output MLP linear_0.bias[:10]: ";
        for (int i = 0; i < 10; ++i) {
            std::cout << b.at({i}) << " ";
        }
        std::cout << std::endl;
    }
    if (output_params.find("mlp.linear_1.weight") != output_params.end()) {
        auto w = *output_params["mlp.linear_1.weight"];
        std::cout << "[Debug] Output MLP linear_1.weight shape: [" << w.size(0) << ", " << w.size(1) << "]" << std::endl;
        std::cout << "[Debug] Output MLP linear_1.weight[0, :10]: ";
        for (int i = 0; i < 10; ++i) {
            std::cout << w.at({0, i}) << " ";
        }
        std::cout << std::endl;
    } else {
        std::cout << "[Warning] mlp.linear_1.weight not found!" << std::endl;
    }

    // ========================================================================
    // 3. 转换数据格式
    // ========================================================================
    std::cout << "[Simplified] Converting data format..." << std::endl;

    auto coedges = BRepNetAdapter::extract_coedges(pipeline, model->surf_enc, model->curve_enc);
    auto faces = BRepNetAdapter::extract_faces(pipeline);
    auto edges = BRepNetAdapter::extract_edges(pipeline);

    // ========================================================================
    // 4. 推理
    // ========================================================================
    breptorch::Tensor logits = model->forward(coedges, faces, edges);

    std::cout << "\n=== Inference Complete! ===" << std::endl;

    // ========================================================================
    // 5. 计算 Softmax 概率
    // ========================================================================
    std::cout << "\n[Computing Softmax Probabilities]\n";
    breptorch::Tensor probs = breptorch::softmax(logits, 1);
    std::cout << "  Probabilities shape: [" << probs.size(0) << ", " << probs.size(1) << "]\n";

    // ========================================================================
    // 6. 输出结果
    // ========================================================================
    std::cout << "\n[Output Results]\n";
    std::cout << "  Logits shape: [" << logits.size(0) << ", " << logits.size(1) << "]\n";

    // 打印前 3 个 face 的 logits
    std::cout << "\n[Debug] Final logits (first 3 faces, all 27 dims):\n";
    for (int f = 0; f < std::min(3, (int)logits.size(0)); ++f) {
        std::cout << "  Face " << f << ": [";
        for (int c = 0; c < (int)logits.size(1); ++c) {
            std::cout << logits.at({f, c});
            if (c < (int)logits.size(1) - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    // 打印前 3 个 face 的概率（用于与 Python 对比）
    std::cout << "\n[Debug] Softmax probabilities (first 3 faces, all 27 dims):\n";
    std::cout << std::scientific << std::setprecision(18);  // 使用科学计数法，18位精度
    for (int f = 0; f < std::min(3, (int)probs.size(0)); ++f) {
        std::cout << "  Face " << f << ": ";
        for (int c = 0; c < (int)probs.size(1); ++c) {
            std::cout << probs.at({f, c});
            if (c < (int)probs.size(1) - 1) std::cout << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
