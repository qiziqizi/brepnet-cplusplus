#define _CRT_SECURE_NO_WARNINGS // <--- 必须加在最第一行！
#define ENABLE_TEST  // <--- 加上这一行以启用测试主函数
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <filesystem>
#include <chrono>   // 用于计时
#include <windows.h> // Windows 系统 API
#include <psapi.h>   // 用于查询进程内存状态
#pragma comment(lib, "psapi.lib") // 链接库
namespace fs = std::filesystem;

#include "BRepTorch.h"
namespace torch = breptorch;

#include "BRepNet.h"
#include "BRepPipeline.h"
#include "InferenceEngine.h"
#include "SimpleLogger.h" 
//#include "BRepTest.h"


// 辅助：检查文件是否存在
void check_file(const std::string& path) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("文件不存在: " + path);
    }
    std::cout << " 文件存在: " << path << std::endl;
}


// 获取当前物理内存占用 (Working Set)
double get_current_memory_mb() {
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        // WorkingSetSize 是当前占用的物理内存 (Bytes)
        // 除以 1024*1024 转换为 MB
        return (double)pmc.WorkingSetSize / (1024.0 * 1024.0);
    }
    return 0.0;
}

#ifdef ENABLE_TEST
int main() {

    // 只要这行存在，所有的 cout/cerr 都会被记录，程序退出时自动保存
    Tools::AutoLogger _logger;
    try {
        auto start_total = std::chrono::high_resolution_clock::now();
        double mem_start = get_current_memory_mb();
        std::cout << "[Perf] 初始内存: " << mem_start << " MB" << std::endl;

		// 测试文件路径修改为相对路径
        fs::path base_dir = "test_data";
        std::string verify_path = (base_dir / "verification_data_0101.npz").string();
        std::string weights_path = (base_dir / "brepnet_weights_0101.npz").string();
        std::string step_path = (base_dir / "136322_81d84c1b_1.stp").string();
        // 使用 Tools::GetAbsPath 打印绝对路径，方便日志回溯
        std::cout << "[Config] Verify File : " << Tools::GetAbsPath(verify_path) << std::endl;
        std::cout << "[Config] Weights File: " << Tools::GetAbsPath(weights_path) << std::endl;
        std::cout << "[Config] STEP File   : " << Tools::GetAbsPath(step_path) << std::endl;


        check_file(verify_path);

        // 第一段数据预处理
        auto start_load = std::chrono::high_resolution_clock::now();

        BRepPipeline pipeline;
        pipeline.process(step_path);

        pipeline.load_stats(weights_path); // 加载均值和方差
        if (pipeline.has_stats)
            pipeline.standardize(); // 执行 (x - mean) / std

        // ==========================================
        // 阶段一：数据加载与预处理 (BRepPipeline)
        // ==========================================

        // 1. 加载数据
        cnpy::npz_t npz_data = cnpy::npz_load(verify_path);

        auto load_t = [&](std::string key) {
            if (!npz_data.count(key)) throw std::runtime_error("Missing: " + key);
            cnpy::NpyArray arr = npz_data[key];
            std::vector<int64_t> s(arr.shape.begin(), arr.shape.end());
            return torch::from_blob(arr.data<float>(), s, torch::kFloat32).clone();
            };
        auto load_long = [&](std::string key) {
            if (!npz_data.count(key)) throw std::runtime_error("Missing: " + key);
            cnpy::NpyArray arr = npz_data[key];
            std::vector<int64_t> s(arr.shape.begin(), arr.shape.end());
            if (arr.word_size == 8) return torch::from_blob(arr.data<long long>(), s, torch::kLong).clone();
            else return torch::from_blob(arr.data<int>(), s, torch::kInt).to(torch::kLong).clone();
            };

        // 验证局部坐标系变换 (LCS Math Check)
        //加载原始特征 加载原始索引
        /*
        //加载原始特征 (此时还没有 Padding) 
        pipeline.Xf = load_t("Xf");
        pipeline.Xe = load_t("Xe");
        pipeline.Xc = load_t("Xc");
        //加载原始索引 (0-based)
        pipeline.Kf = load_long("Kf");
        std::cout << " C++ Ke: " << pipeline.Ke << std::endl;
        pipeline.Ke = load_long("Ke");
        std::cout << " Python Ke: " << pipeline.Ke << std::endl;
        pipeline.Kc = load_long("Kc");
        pipeline.Ce = load_long("Ce");
        pipeline.Cf = load_long("Cf");*/

        // BRepPipeline.中 generate_tensors 将 int max_cpf 从 64改为 512，舍弃复杂的 Csf 列表，把 Cf 矩阵开大一点
        /*if (npz_data.count("num_big_faces")) {
            int num = *npz_data["num_big_faces"].data<int>();
            for (int i = 0; i < num; ++i) pipeline.Csf.push_back(load_long("Csf_" + std::to_string(i)));
        }*/

                    data[i] = v + 1;
                }
                // 否则 (通常是 padding index = limit)，归 0
                else {
                    data[i] = 0;
                }
            }
            };

        // 索引偏移 Kf 指向 Face，Ke 指向 Edge等
        shift_indices(pipeline.Kf, num_faces);
        shift_indices(pipeline.Ke, num_edges);
        shift_indices(pipeline.Kc, num_coedges);
        shift_indices(pipeline.Ce, num_coedges);
        shift_indices(pipeline.Cf, num_coedges);
        for (auto& t : pipeline.Csf) shift_indices(t, num_coedges);

        // 3. 给矩阵头部加 0 (Padding)
        // 索引已经 +1 腾出位置了，现在真正插入这一行

        // 特征补0
        auto pad_front = [](torch::Tensor& x) {
            auto pad = torch::zeros({ 1, x.size(1) }, x.options());
            x = torch::cat({ pad, x }, 0);
            };
        pad_front(pipeline.Xf);
        pad_front(pipeline.Xe);
        pad_front(pipeline.Xc);

        // 拓扑补0
        pad_front(pipeline.Kf);
        pad_front(pipeline.Ke);
        pad_front(pipeline.Kc);

        // 网格补齐，Grid必须和特征矩阵行数一致 (因为特征矩阵刚补了一行 0)
        auto align_grid = [](torch::Tensor& g, int64_t target_rows) {
            if (g.defined() && g.size(0) == target_rows - 1) {
                std::vector<int64_t> s = g.sizes(); 
                s[0] = 1;
                g = torch::cat({ torch::zeros(s, g.options()), g }, 0);
            }
            };

        // 要用Kf,而不是Xf
        align_grid(pipeline.FaceGridsLocal, pipeline.Kf.size(0));
        align_grid(pipeline.EdgeGridsLocal, pipeline.Xe.size(0));
        align_grid(pipeline.CoedgeGridsLocal, pipeline.Xc.size(0));

        //第一段数据预处理结束
        auto end_load = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> load_ms = end_load - start_load;
        std::cout << "[Perf] 数据预处理耗时: " << load_ms.count() << " ms" << std::endl;
        std::cout << "[Perf] 数据预处理后内存: " << get_current_memory_mb() << " MB" << std::endl;

        // ==============================================================================================================================
        // 第二阶段 & 第三阶段：模型初始化与推理
        // ==============================================================================================================================
        
        auto start_init = std::chrono::high_resolution_clock::now();
        // 1. 初始化引擎 (参数: 320, 120, 5, 8)
        InferenceEngine engine(320, 120, 5, 8);
        // 2. 加载权重
        engine.load_weights(weights_path);
        auto end_init = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> init_ms = end_init - start_init;
        std::cout << "[Perf] 模型初始化耗时: " << init_ms.count() << " ms" << std::endl;
        std::cout << "[Perf] 模型加载后内存: " << get_current_memory_mb() << " MB" << std::endl;
        
        auto start_infer = std::chrono::high_resolution_clock::now();
        // 3. 运行推理 (直接把处理好的 pipeline 扔进去)
        torch::Tensor logits = engine.predict(pipeline);
        auto end_infer = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> infer_ms = end_infer - start_infer;
        std::cout << "=== 推理成功! ===" << std::endl;
        std::cout << "[Perf] 推理计算耗时: " << infer_ms.count() << " ms" << std::endl;
        std::cout << "[Perf] 推理峰值内存 (近似): " << get_current_memory_mb() << " MB" << std::endl;

        // 对比结果
        if (npz_data.count("expected_output")) {
            torch::Tensor expected = load_t("expected_output");

            // 对齐切片 (跳过 C++ 的第 0 行)
            // Python 的 logits 通常不含 Padding (N行)
            // C++ 的 logits 含 Padding (N+1行)
            int64_t rows = std::min(logits.size(0) - 1, expected.size(0));
            torch::Tensor c_valid = logits.slice(0, 1, 1 + rows);
            torch::Tensor p_valid = expected.slice(0, 0, rows);

            std::cout << "\nC++ Logits (row 1):\n" << c_valid.slice(0, 0, 1) << std::endl; 
            std::cout << "Py  Logits (row 0):\n" << p_valid.slice(0, 0, 1) << std::endl;

            float err = (c_valid - p_valid).abs().sum().item<float>();
            std::cout << ">>> 最终误差: " << err << std::endl;

            if (err < 0.1) std::cout << "SUCCESS! 通过" << std::endl;
            else std::cout << "失败" << std::endl;
        }

        // ==========================================
        // 第四段 总结
        // ==========================================
        //
        auto end_total = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> total_ms = end_total - start_total;
        std::cout << "\n---------------- 性能报告 ----------------" << std::endl;
        std::cout << "总耗时: " << total_ms.count() << " ms" << std::endl;
        std::cout << "  - 数据预处理: " << load_ms.count() << " ms" << std::endl;
        std::cout << "  - 模型加载: " << init_ms.count() << " ms" << std::endl;
        std::cout << "  - 网络推理: " << infer_ms.count() << " ms" << std::endl;
        std::cout << "内存增长: " << (get_current_memory_mb() - mem_start) << " MB" << std::endl;
        std::cout << "------------------------------------------" << std::endl;

    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    std::cin.get();
    return 0;
}
#endif