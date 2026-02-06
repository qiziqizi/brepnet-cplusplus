#include "BRepNet.h"
#include <iostream>

int main() {
    // 创建Output MLP
    BRepNetMLP output_mlp(90, 30, 30, true);

    // 加载权重
    cnpy::npz_t npz = cnpy::npz_load("inference_data/state_dict.npz");
    auto params = output_mlp->named_parameters();

    for (auto& item : npz) {
        std::string key = item.first;
        if (key.find("output_layer.mlp.mlp") != std::string::npos) {
            // 提取子键名
            std::string subkey = key.substr(std::string("output_layer.mlp.").length());
            std::cout << "[Debug] Loading: " << key << " -> " << subkey << std::endl;

            if (params.find(subkey) != params.end()) {
                auto arr = item.second;
                std::vector<int64_t> shape(arr.shape.begin(), arr.shape.end());
                *params[subkey] = breptorch::from_blob(arr.data<float>(), shape, breptorch::kFloat32).clone();
                std::cout << "[Debug] Loaded " << subkey << " with shape: [";
                for (size_t i = 0; i < shape.size(); ++i) {
                    std::cout << shape[i];
                    if (i < shape.size() - 1) std::cout << ", ";
                }
                std::cout << "]" << std::endl;
            } else {
                std::cout << "[Warning] Key not found in params: " << subkey << std::endl;
            }
        }
    }

    // 验证权重
    std::cout << "\n[Debug] Verifying loaded weights..." << std::endl;
    if (params.find("mlp.linear_0.weight") != params.end()) {
        auto w = *params["mlp.linear_0.weight"];
        std::cout << "linear_0.weight[0, :10]: ";
        for (int i = 0; i < 10; ++i) {
            std::cout << w.at({0, i}) << " ";
        }
        std::cout << std::endl;
    }

    if (params.find("mlp.linear_1.weight") != params.end()) {
        auto w = *params["mlp.linear_1.weight"];
        std::cout << "linear_1.weight[0, :10]: ";
        for (int i = 0; i < 10; ++i) {
            std::cout << w.at({0, i}) << " ";
        }
        std::cout << std::endl;
    }

    // 测试前向传播
    std::cout << "\n[Debug] Testing forward pass..." << std::endl;
    std::vector<float> input_data = {
        3.7095279693603516, 1.3930604457855225, 0.8963956236839294, 0.0, 0.0, 0.0, 2.9675285816192627, 6.556656360626221, 5.645405292510986, 2.739002227783203,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 8.829337120056152, 7.12383508682251, 0.0, 0.0, 0.0, 1.2519614696502686, 6.247445106506348, 10.776010513305664, 0.9942263960838318,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        3.3867387771606445, 5.288075923919678, 6.41499662399292, 1.5830342769622803, 7.170553207397461, 0.0, 1.9200149774551392, 0.21108371019363403, 0.7127571105957031, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    };

    breptorch::Tensor input = breptorch::from_blob(input_data.data(), {1, 90}, breptorch::kFloat32).clone();
    breptorch::Tensor output = output_mlp->forward(input);

    std::cout << "Output shape: [" << output.size(0) << ", " << output.size(1) << "]" << std::endl;
    std::cout << "Output[:10]: ";
    for (int i = 0; i < 10; ++i) {
        std::cout << output.at({0, i}) << " ";
    }
    std::cout << std::endl;

    std::cout << "\nExpected (Python): 7.28031 -3.20024 -4.98247 -5.08260 -2.57899 0.70982 -10.69880 3.26065 -10.76078 -1.33437" << std::endl;

    return 0;
}
