#define _CRT_SECURE_NO_WARNINGS // <--- ����������һ�У�
#define ENABLE_TEST  // <--- ������һ�������ò���������
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <filesystem>
#include <chrono>   // ���ڼ�ʱ
#include <windows.h> // Windows ϵͳ API
#include <psapi.h>   // ���ڲ�ѯ�����ڴ�״̬
#pragma comment(lib, "psapi.lib") // ���ӿ�
namespace fs = std::filesystem;

#include "BRepTorch.h"
namespace torch = breptorch;

#include "BRepNet.h"
#include "BRepPipeline.h"
#include "InferenceEngine.h"
#include "SimpleLogger.h"
#include "VerificationLogger.h" 
//#include "BRepTest.h"


// ����������ļ��Ƿ����
void check_file(const std::string& path) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("�ļ�������: " + path);
    }
    std::cout << " �ļ�����: " << path << std::endl;
}


// ��ȡ��ǰ�����ڴ�ռ�� (Working Set)
double get_current_memory_mb() {
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        // WorkingSetSize �ǵ�ǰռ�õ������ڴ� (Bytes)
        // ���� 1024*1024 ת��Ϊ MB
        return (double)pmc.WorkingSetSize / (1024.0 * 1024.0);
    }
    return 0.0;
}

#ifdef ENABLE_TEST
int main() {

    // ֻҪ���д��ڣ����е� cout/cerr ���ᱻ��¼�������˳�ʱ�Զ�����
    Tools::AutoLogger _logger;
    try {
        auto start_total = std::chrono::high_resolution_clock::now();
        double mem_start = get_current_memory_mb();
        std::cout << "[Perf] ��ʼ�ڴ�: " << mem_start << " MB" << std::endl;

		// �����ļ�·���޸�Ϊ���·��
        fs::path base_dir = "test_data";
        std::string verify_path = (base_dir / "verification_data_0101.npz").string();
        std::string weights_path = (base_dir / "brepnet_weights_0101.npz").string();
        std::string step_path = (base_dir / "136322_81d84c1b_1.stp").string();
        // ʹ�� Tools::GetAbsPath ��ӡ����·����������־����
        std::cout << "[Config] Verify File : " << Tools::GetAbsPath(verify_path) << std::endl;
        std::cout << "[Config] Weights File: " << Tools::GetAbsPath(weights_path) << std::endl;
        std::cout << "[Config] STEP File   : " << Tools::GetAbsPath(step_path) << std::endl;


        check_file(verify_path);

        // ��һ������Ԥ����
        auto start_load = std::chrono::high_resolution_clock::now();

        BRepPipeline pipeline;
        pipeline.process(step_path);

        pipeline.load_stats(weights_path); // ���ؾ�ֵ�ͷ���
        if (pipeline.has_stats)
            pipeline.standardize(); // ִ�� (x - mean) / std

        // ==========================================
        // �׶�һ�����ݼ�����Ԥ���� (BRepPipeline)
        // ==========================================

        // 1. ��������
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

        // ��֤�ֲ�����ϵ�任 (LCS Math Check)
        //����ԭʼ���� ����ԭʼ����
        /*
        //����ԭʼ���� (��ʱ��û�� Padding) 
        pipeline.Xf = load_t("Xf");
        pipeline.Xe = load_t("Xe");
        pipeline.Xc = load_t("Xc");
        //����ԭʼ���� (0-based)
        pipeline.Kf = load_long("Kf");
        std::cout << " C++ Ke: " << pipeline.Ke << std::endl;
        pipeline.Ke = load_long("Ke");
        std::cout << " Python Ke: " << pipeline.Ke << std::endl;
        pipeline.Kc = load_long("Kc");
        pipeline.Ce = load_long("Ce");
        pipeline.Cf = load_long("Cf");*/

        // BRepPipeline.�� generate_tensors �� int max_cpf �� 64��Ϊ 512���������ӵ� Csf �б����� Cf ���󿪴�һ��
        /*if (npz_data.count("num_big_faces")) {
            int num = *npz_data["num_big_faces"].data<int>();
            for (int i = 0; i < num; ++i) pipeline.Csf.push_back(load_long("Csf_" + std::to_string(i)));
        }*/

                    data[i] = v + 1;
                }
                // ���� (ͨ���� padding index = limit)���� 0
                else {
                    data[i] = 0;
                }
            }
            };

        // ����ƫ�� Kf ָ�� Face��Ke ָ�� Edge��
        shift_indices(pipeline.Kf, num_faces);
        shift_indices(pipeline.Ke, num_edges);
        shift_indices(pipeline.Kc, num_coedges);
        shift_indices(pipeline.Ce, num_coedges);
        shift_indices(pipeline.Cf, num_coedges);
        for (auto& t : pipeline.Csf) shift_indices(t, num_coedges);

        // 3. ������ͷ���� 0 (Padding)
        // �����Ѿ� +1 �ڳ�λ���ˣ���������������һ��

        // ������0
        auto pad_front = [](torch::Tensor& x) {
            auto pad = torch::zeros({ 1, x.size(1) }, x.options());
            x = torch::cat({ pad, x }, 0);
            };
        pad_front(pipeline.Xf);
        pad_front(pipeline.Xe);
        pad_front(pipeline.Xc);

        // ���˲�0
        pad_front(pipeline.Kf);
        pad_front(pipeline.Ke);
        pad_front(pipeline.Kc);

        // �����룬Grid�����������������һ�� (��Ϊ��������ղ���һ�� 0)
        auto align_grid = [](torch::Tensor& g, int64_t target_rows) {
            if (g.defined() && g.size(0) == target_rows - 1) {
                std::vector<int64_t> s = g.sizes(); 
                s[0] = 1;
                g = torch::cat({ torch::zeros(s, g.options()), g }, 0);
            }
            };

        // Ҫ��Kf,������Xf
        align_grid(pipeline.FaceGridsLocal, pipeline.Kf.size(0));
        align_grid(pipeline.EdgeGridsLocal, pipeline.Xe.size(0));
        align_grid(pipeline.CoedgeGridsLocal, pipeline.Xc.size(0));

        //��һ������Ԥ��������
        auto end_load = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> load_ms = end_load - start_load;
        std::cout << "[Perf] ����Ԥ������ʱ: " << load_ms.count() << " ms" << std::endl;
        std::cout << "[Perf] ����Ԥ�������ڴ�: " << get_current_memory_mb() << " MB" << std::endl;

        // ==============================================================================================================================
        // �ڶ��׶� & �����׶Σ�ģ�ͳ�ʼ��������
        // ==============================================================================================================================
        
        auto start_init = std::chrono::high_resolution_clock::now();
        // 1. ��ʼ������ (����: 320, 120, 5, 8)
        InferenceEngine engine(320, 120, 5, 8);
        // 2. ����Ȩ��
        engine.load_weights(weights_path);
        auto end_init = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> init_ms = end_init - start_init;
        std::cout << "[Perf] ģ�ͳ�ʼ����ʱ: " << init_ms.count() << " ms" << std::endl;
        std::cout << "[Perf] ģ�ͼ��غ��ڴ�: " << get_current_memory_mb() << " MB" << std::endl;
        
        auto start_infer = std::chrono::high_resolution_clock::now();
        // 3. �������� (ֱ�ӰѴ����õ� pipeline �ӽ�ȥ)
        torch::Tensor logits = engine.predict(pipeline);
        auto end_infer = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> infer_ms = end_infer - start_infer;
        std::cout << "=== �����ɹ�! ===" << std::endl;
        std::cout << "[Perf] ���������ʱ: " << infer_ms.count() << " ms" << std::endl;
        std::cout << "[Perf] ������ֵ�ڴ� (����): " << get_current_memory_mb() << " MB" << std::endl;

        // �ԱȽ��
        if (npz_data.count("expected_output")) {
            torch::Tensor expected = load_t("expected_output");

            // ������Ƭ (���� C++ �ĵ� 0 ��)
            // Python �� logits ͨ������ Padding (N��)
            // C++ �� logits �� Padding (N+1��)
            int64_t rows = std::min(logits.size(0) - 1, expected.size(0));
            torch::Tensor c_valid = logits.slice(0, 1, 1 + rows);
            torch::Tensor p_valid = expected.slice(0, 0, rows);

            Verification::LogTensorSlice("CPP_Logits_Row1", c_valid, 0, 1, 0, c_valid.size(1));
            Verification::LogTensorSlice("Py_Logits_Row0", p_valid, 0, 1, 0, p_valid.size(1));

            float err = (c_valid - p_valid).abs().sum().item<float>();
            Verification::Log("Total_Error", err);

            if (err < 0.1) std::cout << "SUCCESS! ͨ��" << std::endl;
            else std::cout << "ʧ��" << std::endl;
        }

        // ==========================================
        // ���Ķ� �ܽ�
        // ==========================================
        //
        auto end_total = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> total_ms = end_total - start_total;
        std::cout << "\n---------------- ���ܱ��� ----------------" << std::endl;
        std::cout << "�ܺ�ʱ: " << total_ms.count() << " ms" << std::endl;
        std::cout << "  - ����Ԥ����: " << load_ms.count() << " ms" << std::endl;
        std::cout << "  - ģ�ͼ���: " << init_ms.count() << " ms" << std::endl;
        std::cout << "  - ��������: " << infer_ms.count() << " ms" << std::endl;
        std::cout << "�ڴ�����: " << (get_current_memory_mb() - mem_start) << " MB" << std::endl;
        std::cout << "------------------------------------------" << std::endl;

    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    std::cin.get();
    return 0;
}
#endif