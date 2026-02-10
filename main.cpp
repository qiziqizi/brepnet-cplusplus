#include "BRepNet.h"
#include "BRepNetAdapter.h"
#include "BRepPipeline.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <Windows.h>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <map>
#include <fstream>
#include <sstream>
#include <io.h>
#include <fcntl.h>
#include <limits>

namespace fs = std::filesystem;

// 全局输出文件流
std::ofstream g_log_file;

// 重载的输出流类，同时输出到控制台和文件
class DualStream {
public:
    template<typename T>
    DualStream& operator<<(const T& data) {
        std::cout << data;
        if (g_log_file.is_open()) {
            g_log_file << data;
        }
        return *this;
    }

    // 支持 std::endl 等操作符
    DualStream& operator<<(std::ostream& (*manip)(std::ostream&)) {
        std::cout << manip;
        if (g_log_file.is_open()) {
            g_log_file << manip;
        }
        return *this;
    }

    // 支持 std::setw, std::setprecision 等操作符
    DualStream& operator<<(std::ios_base& (*manip)(std::ios_base&)) {
        std::cout << manip;
        if (g_log_file.is_open()) {
            g_log_file << manip;
        }
        return *this;
    }
};

DualStream dout;  // 全局双输出流对象

// 单个文件的推理结果
struct InferenceResult {
    std::string filename;
    bool success;
    int num_faces;
    int num_coedges;
    int num_edges;
    int inference_time_ms;
    std::map<int, int> class_distribution;  // class_id -> count
    float max_confidence;
    float avg_confidence;
    float min_confidence;
    std::vector<float> losses;  // 每个面的loss值
    std::vector<std::vector<float>> logits;  // 每个面的softmax概率 [num_faces, num_classes]
};

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
        std::cerr << "[Error] Failed to read directory: " << e.what() << std::endl;
    }
    std::sort(step_files.begin(), step_files.end());
    return step_files;
}

// 对单个 STEP 文件进行推理
InferenceResult run_inference(const std::string& step_file, std::shared_ptr<BRepNetImpl> model) {
    InferenceResult result;
    result.filename = fs::path(step_file).filename().string();
    result.success = false;
    result.max_confidence = 0.0f;
    result.avg_confidence = 0.0f;
    result.min_confidence = 1.0f;

    dout << "\n" << std::string(70, '=') << std::endl;
    dout << "处理文件: " << result.filename << std::endl;
    dout << std::string(70, '=') << std::endl;

    // 1. 数据预处理
    BRepPipeline pipeline;
    if (!pipeline.process(step_file)) {
        std::cerr << "[错误] 处理失败: " << result.filename << std::endl;
        return result;
    }

    result.num_coedges = (int)pipeline.coedges.size();
    result.num_faces = (int)pipeline.unique_faces.Extent();
    result.num_edges = (int)pipeline.unique_edges.Extent();

    dout << "[拓扑] " << result.num_coedges << " 个共边, "
              << result.num_faces << " 个面, "
              << result.num_edges << " 个边" << std::endl;

    // 2. 转换数据格式
    auto coedges = BRepNetAdapter::extract_coedges(pipeline, model->surf_enc, model->curve_enc);
    auto faces = BRepNetAdapter::extract_faces(pipeline);
    auto edges = BRepNetAdapter::extract_edges(pipeline);

    // 3. 推理
    auto start = std::chrono::high_resolution_clock::now();
    breptorch::Tensor logits = model->forward(coedges, faces, edges);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    result.inference_time_ms = (int)duration.count();

    // 4. 计算 Softmax 概率并获取预测结果
    breptorch::Tensor probs = breptorch::softmax(logits, 1);
    int num_faces = (int)logits.size(0);
    int num_classes = (int)logits.size(1);

    dout << "[推理] 耗时: " << result.inference_time_ms << " ms" << std::endl;
    dout << "[输出] Logits 形状: [" << num_faces << ", " << num_classes << "]" << std::endl;

    // 输出每个面的详细预测结果
    dout << "\n[预测结果]" << std::endl;
    float confidence_sum = 0.0f;
    float loss_sum = 0.0f;

    for (int f = 0; f < num_faces; ++f) {
        float max_prob = -1e9f;
        int pred_class = 0;

        // 保存该面的所有类别概率
        std::vector<float> face_probs;
        for (int c = 0; c < num_classes; ++c) {
            float p = probs.at({f, c});
            face_probs.push_back(p);
            if (p > max_prob) {
                max_prob = p;
                pred_class = c;
            }
        }

        // 保存到结果中
        result.logits.push_back(face_probs);

        // 计算该面的损失
        float face_loss = -std::log(max_prob + 1e-10f);

        // 保存loss到结果中
        result.losses.push_back(face_loss);

        // 输出该面的预测结果
        dout << "  Face " << std::setw(3) << f << ": Class " << std::setw(2) << pred_class
                  << " (置信度: " << std::fixed << std::setprecision(6) << max_prob
                  << ", loss: " << std::setprecision(6) << face_loss << ")" << std::endl;

        result.class_distribution[pred_class]++;
        confidence_sum += max_prob;
        loss_sum += face_loss;
        result.max_confidence = std::max(result.max_confidence, max_prob);
        result.min_confidence = std::min(result.min_confidence, max_prob);
    }

    result.avg_confidence = confidence_sum / num_faces;
    float avg_loss = loss_sum / num_faces;

    // 输出汇总信息
    dout << "\n[汇总] 平均置信度: " << std::fixed << std::setprecision(6) << result.avg_confidence
              << ", 平均 Loss: " << std::setprecision(6) << avg_loss << std::endl;

    result.success = true;
    return result;
}

int main() {
    SetConsoleOutputCP(65001);  // UTF-8 编码
    SetConsoleCP(65001);        // UTF-8 输入编码

    // 打开日志文件
    g_log_file.open("cpp_inference.txt", std::ios::out);
    if (!g_log_file.is_open()) {
        std::cerr << "无法创建输出文件: cpp_inference.txt" << std::endl;
        return -1;
    }

    // 写入UTF-8 BOM
    g_log_file << "\xEF\xBB\xBF";

    dout << "=== BRepNet 批量推理测试 ===" << std::endl;
    dout << "输出文件: cpp_inference.txt" << std::endl;

    // 打印浮点精度信息
    dout << "\n[系统信息]" << std::endl;
    dout << "  Float 精度: " << std::numeric_limits<float>::digits10 << " 位十进制" << std::endl;
    dout << "  Double 精度: " << std::numeric_limits<double>::digits10 << " 位十进制" << std::endl;
    dout << "  Float 测试值: " << std::fixed << std::setprecision(20) << 2.3932483196258545f << std::endl;
    dout << "  编译器: " <<
#ifdef _MSC_VER
        "MSVC " << _MSC_VER <<
#elif defined(__GNUC__)
        "GCC " << __GNUC__ << "." << __GNUC_MINOR__ <<
#else
        "Unknown" <<
#endif
        std::endl;

    // ========================================================================
    // 1. 加载模型（只加载一次）
    // ========================================================================
    std::string weights_file = "inference_data/state_dict.npz";
    std::string step_dir = "inference_data/step_files";

    dout << "\n[模型] 从以下位置加载权重: " << weights_file << std::endl;

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

    dout << "[模型] 权重加载成功!" << std::endl;

    // ========================================================================
    // 2. 获取所有 STEP 文件
    // ========================================================================
    dout << "\n[文件] 扫描目录: " << step_dir << std::endl;
    auto step_files = get_step_files(step_dir);

    if (step_files.empty()) {
        std::cerr << "[错误] 未找到 STEP 文件: " << step_dir << std::endl;
        return -1;
    }

    dout << "[文件] 找到 " << step_files.size() << " 个 STEP 文件" << std::endl;

    // ========================================================================
    // 3. 批量推理
    // ========================================================================
    std::vector<InferenceResult> all_results;
    auto total_start = std::chrono::high_resolution_clock::now();

    for (const auto& step_file : step_files) {
        InferenceResult result = run_inference(step_file, model);
        all_results.push_back(result);
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);

    // ========================================================================
    // 4. 批量推理完成
    // ========================================================================
    int success_count = 0, fail_count = 0;
    for (const auto& r : all_results) {
        if (r.success) success_count++;
        else fail_count++;
    }

    dout << "\n" << std::string(70, '=') << std::endl;
    dout << "批量推理完成: " << success_count << "/" << step_files.size()
              << " 个文件成功 (总耗时 " << total_duration.count() << " ms)" << std::endl;
    dout << std::string(70, '=') << std::endl;

    // ========================================================================
    // 5. 生成每个文件的loss和logits文件
    // ========================================================================
    dout << "\n[生成Loss和Logits文件]" << std::endl;

    // 创建cpp_loss文件夹
    std::string loss_dir = "cpp_loss";
    if (!fs::exists(loss_dir)) {
        fs::create_directory(loss_dir);
        dout << "  创建目录: " << loss_dir << std::endl;
    }

    // 创建cpp_logits文件夹
    std::string logits_dir = "cpp_logits";
    if (!fs::exists(logits_dir)) {
        fs::create_directory(logits_dir);
        dout << "  创建目录: " << logits_dir << std::endl;
    }

    for (size_t i = 0; i < all_results.size(); ++i) {
        const auto& result = all_results[i];
        if (!result.success) continue;

        fs::path step_path(step_files[i]);
        std::string base_name = step_path.stem().string();

        // ========== 生成loss文件 ==========
        std::string loss_filename = loss_dir + "/" + base_name + "_result.loss";
        std::ofstream loss_file(loss_filename, std::ios::out);
        if (!loss_file.is_open()) {
            std::cerr << "  无法创建loss文件: " << loss_filename << std::endl;
            continue;
        }

        // 写入UTF-8 BOM
        loss_file << "\xEF\xBB\xBF";

        // 写入文件名标识
        loss_file << "这是" << base_name << "的loss:" << std::endl;

        // 写入所有loss值，用空格分隔，使用科学计数法，精度20位
        loss_file << std::scientific << std::setprecision(20);
        for (size_t j = 0; j < result.losses.size(); ++j) {
            if (j > 0) loss_file << " ";
            loss_file << result.losses[j];
        }
        loss_file << std::endl;
        loss_file.close();

        dout << "  生成: " << loss_filename << " (" << result.losses.size() << " 个loss值)" << std::endl;

        // ========== 生成logits文件 ==========
        std::string logits_filename = logits_dir + "/" + base_name + "_result.logits";
        std::ofstream logits_file(logits_filename, std::ios::out);
        if (!logits_file.is_open()) {
            std::cerr << "  无法创建logits文件: " << logits_filename << std::endl;
            continue;
        }

        // 写入所有面的softmax概率，每行一个面，每行27个概率值
        logits_file << std::scientific << std::setprecision(20);
        for (size_t f = 0; f < result.logits.size(); ++f) {
            for (size_t c = 0; c < result.logits[f].size(); ++c) {
                if (c > 0) logits_file << " ";
                logits_file << result.logits[f][c];
            }
            logits_file << std::endl;
        }
        logits_file.close();

        dout << "  生成: " << logits_filename << " (" << result.logits.size() << " 个面 × "
             << (result.logits.empty() ? 0 : result.logits[0].size()) << " 个类别)" << std::endl;
    }

    // 关闭日志文件
    g_log_file.close();

    return (fail_count == 0) ? 0 : 1;
}
