#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <chrono>
#include <cmath>
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")

#include "BRepTorch.h"
namespace torch = breptorch;

#include "BRepNet_Clean.h"
#include "BRepNetAdapter.h"
#include "BRepPipeline.h"
#include "SimpleLogger.h"

// Get current process memory usage
double get_current_memory_mb() {
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return (double)pmc.WorkingSetSize / (1024.0 * 1024.0);
    }
    return 0.0;
}

int main() {
    // 设置控制台为 GBK 编码（CP936），以正确显示中文
    SetConsoleOutputCP(936);

    std::cout << "=== BRepNet Clean Version Test ===" << std::endl;
    std::cout << "[Perf] Initial Memory: " << get_current_memory_mb() << " MB" << std::endl;

    std::string weights_path = "inference_data\\state_dict.npz";
    std::string step_path = "inference_data\\test1.step";

    std::cout << "[Config] Weights File: " << weights_path << std::endl;
    std::cout << "[Config] STEP File   : " << step_path << std::endl;

    auto total_start = std::chrono::high_resolution_clock::now();

    // ========================================================================
    // 1. 数据预处理
    // ========================================================================
    auto start_prep = std::chrono::high_resolution_clock::now();

    BRepPipeline pipeline;
    if (!pipeline.process(step_path)) {
        std::cerr << "Failed to process STEP file!" << std::endl;
        return 1;
    }

    // 加载标准化参数
    cnpy::npz_t npz_weights = cnpy::npz_load(weights_path);

    std::cout << "[Clean] Loading normalization parameters..." << std::endl;

    // 只标准化 face 特征（Xf），不标准化 edge 特征（Xe）
    if (npz_weights.count("mean_f") && npz_weights.count("std_f")) {

        cnpy::NpyArray arr_mean_f = npz_weights["mean_f"];
        std::vector<int64_t> shape_mean_f;
        for (auto s : arr_mean_f.shape) shape_mean_f.push_back(s);
        pipeline.mean_f = torch::from_blob(arr_mean_f.data<float>(), shape_mean_f, torch::kFloat32).clone();

        cnpy::NpyArray arr_std_f = npz_weights["std_f"];
        std::vector<int64_t> shape_std_f;
        for (auto s : arr_std_f.shape) shape_std_f.push_back(s);
        pipeline.std_f = torch::from_blob(arr_std_f.data<float>(), shape_std_f, torch::kFloat32).clone();

        pipeline.has_stats = true;

        std::cout << "[Clean] Normalization parameters loaded:" << std::endl;
        std::cout << "  mean_f shape: " << pipeline.mean_f.sizes() << std::endl;
        std::cout << "  std_f shape: " << pipeline.std_f.sizes() << std::endl;

        // 标准化 face 特征
        std::cout << "[Clean] Normalizing face features..." << std::endl;
        std::cout << "  Xf shape before: " << pipeline.Xf.sizes() << std::endl;

        pipeline.Xf = (pipeline.Xf - pipeline.mean_f) / pipeline.std_f;

        std::cout << "[Clean] Face features normalized successfully!" << std::endl;
    } else {
        std::cout << "[Clean] Warning: Normalization parameters not found in weights file!" << std::endl;
        pipeline.has_stats = false;
    }

    auto end_prep = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> prep_ms = end_prep - start_prep;
    std::cout << "[Perf] Data Preprocessing Time: " << prep_ms.count() << " ms" << std::endl;
    std::cout << "[Perf] Data Preprocessing Memory: " << get_current_memory_mb() << " MB" << std::endl;

    // ========================================================================
    // 2. 创建模型并加载权重（需要先创建以获取 UV-Net）
    // ========================================================================
    auto start_init = std::chrono::high_resolution_clock::now();

    BRepNetClean model(27);  // 27 classes (matching the weight file)
    model->load_weights(weights_path);

    auto end_init = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> init_ms = end_init - start_init;
    std::cout << "[Perf] Model Initialization Time: " << init_ms.count() << " ms" << std::endl;
    std::cout << "[Perf] Model Loaded Memory: " << get_current_memory_mb() << " MB" << std::endl;

    // ========================================================================
    // 3. 转换数据格式（使用模型的 UV-Net）
    // ========================================================================
    std::cout << "[Clean] Converting data format..." << std::endl;

    auto coedges = BRepNetAdapter::extract_coedges(pipeline, model->surf_enc, model->curve_enc);
    auto faces = BRepNetAdapter::extract_faces(pipeline);
    auto edges = BRepNetAdapter::extract_edges(pipeline);

    std::cout << "[Clean] Extracted: " << coedges.size() << " coedges, "
              << faces.size() << " faces, " << edges.size() << " edges" << std::endl;

    // ========================================================================
    // 4. 推理
    // ========================================================================
    auto start_infer = std::chrono::high_resolution_clock::now();

    torch::Tensor logits = model->forward(coedges, faces, edges);

    auto end_infer = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> infer_ms = end_infer - start_infer;

    std::cout << "=== Inference Complete! ===" << std::endl;
    std::cout << "[Perf] Inference Time: " << infer_ms.count() << " ms" << std::endl;
    std::cout << "[Perf] Inference Memory (Peak): " << get_current_memory_mb() << " MB" << std::endl;

    // ========================================================================
    // 5. 输出结果
    // ========================================================================
    std::cout << "\n[Output Results]\n";
    std::cout << "  Logits shape: [" << logits.size(0) << ", " << logits.size(1) << "]\n";

    // 计算 softmax 概率（用于与 Python 对比）
    // softmax(x_i) = exp(x_i) / sum(exp(x_j))
    std::vector<std::vector<float>> probabilities(logits.size(0), std::vector<float>(logits.size(1)));
    for (int f = 0; f < logits.size(0); ++f) {
        // 找到最大值用于数值稳定性
        float max_logit = -1e9f;
        for (int c = 0; c < logits.size(1); ++c) {
            max_logit = std::max(max_logit, logits.at({f, c}));
        }

        // 计算 exp(x - max) 和 sum
        float sum_exp = 0.0f;
        for (int c = 0; c < logits.size(1); ++c) {
            float exp_val = std::exp(logits.at({f, c}) - max_logit);
            probabilities[f][c] = exp_val;
            sum_exp += exp_val;
        }

        // 归一化
        for (int c = 0; c < logits.size(1); ++c) {
            probabilities[f][c] /= sum_exp;
        }
    }

    // 打印前3个面的完整 logits 和 softmax 概率
    int num_faces_to_show = std::min(3, (int)logits.size(0));
    for (int f = 0; f < num_faces_to_show; ++f) {
        std::cout << "\n  Face " << f << " logits: [";
        for (int c = 0; c < logits.size(1); ++c) {
            std::cout << logits.at({f, c});
            if (c < logits.size(1) - 1) std::cout << ", ";
        }
        std::cout << "]\n";

        std::cout << "  Face " << f << " softmax probabilities: [";
        for (int c = 0; c < logits.size(1); ++c) {
            printf("%.15e", probabilities[f][c]);
            if (c < logits.size(1) - 1) std::cout << ", ";
        }
        std::cout << "]\n";
    }

    // 打印统计信息
    float min_val = 1e9f, max_val = -1e9f, sum_val = 0.0f;
    int total_elements = logits.size(0) * logits.size(1);
    for (int i = 0; i < logits.size(0); ++i) {
        for (int j = 0; j < logits.size(1); ++j) {
            float val = logits.at({i, j});
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
            sum_val += val;
        }
    }

    std::cout << "\n[Statistics]\n";
    std::cout << "  Min value: " << min_val << "\n";
    std::cout << "  Max value: " << max_val << "\n";
    std::cout << "  Mean value: " << (sum_val / total_elements) << "\n";

    std::cout << "\n[Info] Inference completed! Please manually compare with Python results.\n";

    // ========================================================================
    // 性能报告
    // ========================================================================
    auto total_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> total_ms = total_end - total_start;

    std::cout << "\n---------------- Performance Report ----------------" << std::endl;
    std::cout << "Total Time: " << total_ms.count() << " ms" << std::endl;
    std::cout << "  - Data Preprocessing: " << prep_ms.count() << " ms" << std::endl;
    std::cout << "  - Model Loading: " << init_ms.count() << " ms" << std::endl;
    std::cout << "  - Inference: " << infer_ms.count() << " ms" << std::endl;
    std::cout << "------------------------------------------" << std::endl;

    return 0;
}
