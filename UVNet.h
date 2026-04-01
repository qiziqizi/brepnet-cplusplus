#pragma once
//#include <torch/torch.h>
#include "BRepTorch.h"
#include "DebugControl.h"
#include <map>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <fstream>

// ============================================================================
// 调试输出开关：注释掉此行可禁用所有UVNet调试输出
// ============================================================================
// #define UVNET_DEBUG_OUTPUT

#ifdef UVNET_DEBUG_OUTPUT
    #define UVNET_LOG(x) std::cout << x
#else
    #define UVNET_LOG(x) do {} while(0)
#endif

// BRepTorch already defines the breptorch namespace
using Tensor = breptorch::Tensor;
using namespace breptorch;

namespace breptorch {
namespace nn {

// Simple UV-Net encoder using Torch Tensor operations
// We don't manually implement conv2d, just use PyTorch C++ API
class UVNetSurfaceEncoderImpl : public breptorch::nn::Module {
private:
    // For simplicity, manually manage weights
    // Key is the parameter name from Python training, Value is Tensor
    std::map<std::string, breptorch::Tensor> params;
    std::map<std::string, breptorch::Tensor> buffers; // For BatchNorm running_mean/var

public:
    // Default constructor
    UVNetSurfaceEncoderImpl() = default;
    
    // Load weights from npz file
    void load_weights(const std::map<std::string, breptorch::Tensor>& weight_dict) {
        // std::cout << std::fixed << std::setprecision(10);  // 设置输出精度
        // std::cout << "[UVNetSurfaceEncoder] Loading " << weight_dict.size() << " weights..." << std::endl;

        for (auto const& [key, val] : weight_dict) {
            if (key.find("running") != std::string::npos) {
                buffers[key] = val;
            }
            else {
                params[key] = val;
            }

            // 打印权重信息（仅打印第一个卷积层的权重）
            if (key == "conv_layers.0.0.weight") {
                // std::cout << "[Debug Weight] " << key << " shape: [";
                for (size_t i = 0; i < val.sizes().size(); ++i) {
                    // std::cout << val.sizes()[i];
                    // if (i < val.sizes().size() - 1) std::cout << ", ";
                }
                // std::cout << "]" << std::endl;
                // std::cout << "  First 10 values: ";
                // 创建非const副本来访问data_ptr
                breptorch::Tensor val_copy = val.clone();
                for (int i = 0; i < std::min(10, (int)val.numel()); ++i) {
                    // std::cout << val_copy.data_ptr<float>()[i] << " ";
                }
                // std::cout << std::endl;
            }
        }
        // std::cout << "[UVNetSurfaceEncoder] Loaded " << params.size() << " params, " << buffers.size() << " buffers" << std::endl;
    }
    // Basic building block: Conv2d + BN + LeakyReLU
    breptorch::Tensor conv2d_block(breptorch::Tensor x, std::string prefix) {
        // 1. Conv2d
        std::string weight_key = prefix + ".0.weight";
        if (params.find(weight_key) == params.end()) {
            ERR_LOG << "[Error] Weight not found: " << weight_key << std::endl;
            return breptorch::Tensor();
        }

        // DEBUG: 输入信息
        // std::cout << "\n[DEBUG] " << prefix << " INPUT: shape=[" << x.size(0) << ", " << x.size(1) << ", "
        //           << x.size(2) << ", " << x.size(3) << "]" << std::endl;

        auto w = params[weight_key];
        auto conv_opts = breptorch::nn::functional::Conv2dFuncOptions()
            .stride(1)
            .padding(1);

        x = breptorch::nn::functional::conv2d(x, w, conv_opts);

        // DEBUG: Conv2d输出 - 检查完整张量而不是只检查前100个元素（已禁用）
        // float conv_max = -1e9f, conv_min = 1e9f;
        // const float* data = x.data_ptr<float>();
        // for (int64_t i = 0; i < x.numel(); ++i) {
        //     float val = data[i];
        //     conv_max = std::max(conv_max, val);
        //     conv_min = std::min(conv_min, val);
        // }
        // std::cout << "[DEBUG] " << prefix << " AFTER CONV2D: shape=[" << x.size(0) << ", " << x.size(1) << ", "
        //           << x.size(2) << ", " << x.size(3) << "] max=" << conv_max << " min=" << conv_min << std::endl;

        // 2. BatchNorm
        auto bn_mean = buffers[prefix + ".1.running_mean"];
        auto bn_var = buffers[prefix + ".1.running_var"];
        auto bn_w = params[prefix + ".1.weight"];
        auto bn_b = params[prefix + ".1.bias"];

        // DEBUG: BatchNorm参数
        // std::cout << "[DEBUG] " << prefix << " BN PARAMS: mean[0]=" << bn_mean.at({0})
        //           << " var[0]=" << bn_var.at({0}) << " weight[0]=" << bn_w.at({0})
        //           << " bias[0]=" << bn_b.at({0}) << std::endl;

        auto bn_opts = breptorch::nn::functional::BatchNormFuncOptions()
            .weight(bn_w)
            .bias(bn_b)
            .training(false)
            .momentum(0.1)
            .eps(1e-5);

        x = breptorch::nn::functional::batch_norm(x, bn_mean, bn_var, bn_opts);

        // DEBUG: BatchNorm输出 - 检查完整张量（已禁用）
        // float bn_max = -1e9f, bn_min = 1e9f;
        // data = x.data_ptr<float>();
        // for (int64_t i = 0; i < x.numel(); ++i) {
        //     float val = data[i];
        //     bn_max = std::max(bn_max, val);
        //     bn_min = std::min(bn_min, val);
        // }
        // std::cout << "[DEBUG] " << prefix << " AFTER BATCHNORM: max=" << bn_max << " min=" << bn_min << std::endl;

        // 3. LeakyReLU
        x = breptorch::leaky_relu(x, 0.01);

        // DEBUG: ReLU输出 - 检查完整张量（已禁用）
        // float relu_max = -1e9f, relu_min = 1e9f;
        // data = x.data_ptr<float>();
        // for (int64_t i = 0; i < x.numel(); ++i) {
        //     float val = data[i];
        //     relu_max = std::max(relu_max, val);
        //     relu_min = std::min(relu_min, val);
        // }
        // std::cout << "[DEBUG] " << prefix << " AFTER RELU: max=" << relu_max << " min=" << relu_min << std::endl;

        return x;
    }

    // Basic fully connected block: FC Block
    breptorch::Tensor fc_block(breptorch::Tensor x, std::string prefix) {
        // 1. Linear
        // functional::linear takes 3 params: (input, weight, bias)
        // If no bias, pass an empty Tensor, like {} or Tensor()

        auto w = params[prefix + ".0.weight"];

        // std::cout << "\n[DEBUG] " << prefix << " FC INPUT: shape=[" << x.size(0) << ", " << x.size(1) << "]" << std::endl;

        // Note: passing empty Tensor as bias here
        x = breptorch::nn::functional::linear(x, w, breptorch::Tensor());

        // DEBUG: Linear输出 - 检查完整张量（已禁用）
        // float linear_max = -1e9f, linear_min = 1e9f;
        // const float* data = x.data_ptr<float>();
        // for (int64_t i = 0; i < x.numel(); ++i) {
        //     float val = data[i];
        //     linear_max = std::max(linear_max, val);
        //     linear_min = std::min(linear_min, val);
        // }
        // std::cout << "[DEBUG] " << prefix << " AFTER LINEAR: shape=[" << x.size(0) << ", " << x.size(1) << "] max=" << linear_max << " min=" << linear_min << std::endl;

        // 2. BatchNorm1d
        // functional::batch_norm works for both 1D and 2D
        auto bn_mean = buffers[prefix + ".1.running_mean"];
        auto bn_var = buffers[prefix + ".1.running_var"];
        auto bn_w = params[prefix + ".1.weight"];
        auto bn_b = params[prefix + ".1.bias"];

        // DEBUG: BatchNorm参数
        // std::cout << "[DEBUG] " << prefix << " BN PARAMS: mean[0]=" << bn_mean.at({0})
        //           << " var[0]=" << bn_var.at({0}) << " weight[0]=" << bn_w.at({0})
        //           << " bias[0]=" << bn_b.at({0}) << std::endl;

        auto bn_opts = breptorch::nn::functional::BatchNormFuncOptions()
            .weight(bn_w)
            .bias(bn_b)
            .training(false)
            .momentum(0.1)
            .eps(1e-5);

        x = breptorch::nn::functional::batch_norm(x, bn_mean, bn_var, bn_opts);

        // DEBUG: BatchNorm输出 - 检查完整张量（已禁用）
        // float bn_max = -1e9f, bn_min = 1e9f;
        // data = x.data_ptr<float>();
        // for (int64_t i = 0; i < x.numel(); ++i) {
        //     float val = data[i];
        //     bn_max = std::max(bn_max, val);
        //     bn_min = std::min(bn_min, val);
        // }
        // std::cout << "[DEBUG] " << prefix << " AFTER BATCHNORM: max=" << bn_max << " min=" << bn_min << std::endl;

        // 3. LeakyReLU
        x = breptorch::leaky_relu(x, 0.01);

        // DEBUG: ReLU输出 - 检查完整张量（已禁用）
        // float relu_max = -1e9f, relu_min = 1e9f;
        // data = x.data_ptr<float>();
        // for (int64_t i = 0; i < x.numel(); ++i) {
        //     float val = data[i];
        //     relu_max = std::max(relu_max, val);
        //     relu_min = std::min(relu_min, val);
        // }
        // std::cout << "[DEBUG] " << prefix << " AFTER RELU: max=" << relu_max << " min=" << relu_min << std::endl;

        return x;
    }

    breptorch::Tensor forward(breptorch::Tensor x) {
        if (params.find("surface_encoder.conv1.0.weight") == params.end()) {
            ERR_LOG << "[Error] Weight not found: surface_encoder.conv1.0.weight" << std::endl;
            return breptorch::Tensor();
        }

        // std::cout << "\n=== SURFACE ENCODER FORWARD ===" << std::endl;
        // std::cout << "[INPUT] Shape: [" << x.size(0) << ", " << x.size(1) << ", " << x.size(2) << ", " << x.size(3) << "]" << std::endl;

        // Conv1: 9 -> 64
        x = conv2d_block(x, "surface_encoder.conv1");

        // Conv2: 64 -> 128
        x = conv2d_block(x, "surface_encoder.conv2");

        // std::cout << "[DEBUG] Before Global Pool: shape=[" << x.size(0) << ", " << x.size(1) << ", "
        //           << x.size(2) << ", " << x.size(3) << "]" << std::endl;

        // Global Pool: [N, C, H, W] -> [N, C, 1, 1]
        x = breptorch::adaptive_avg_pool2d(x, { 1, 1 });

        // std::cout << "[DEBUG] After Global Pool: shape=[" << x.size(0) << ", " << x.size(1) << ", "
        //           << x.size(2) << ", " << x.size(3) << "]" << std::endl;

        // 【调试】保存 GlobalPool 后的完整数据到文件（批量测试时已禁用）
        // {
        //     std::ofstream globalpool_file("cpp_feature_maps/uvnet_surface_globalpool_debug.txt");
        //     if (!globalpool_file.is_open()) {
        //         std::cerr << "[Error] Cannot open globalpool debug file" << std::endl;
        //     } else {
        //         globalpool_file << std::fixed << std::setprecision(10);
        //         const float* x_ptr = x.data_ptr<float>();
        //
        //         // 写出所有 faces 的 128 维数据
        //         for (int face_idx = 0; face_idx < x.size(0); ++face_idx) {
        //             for (int ch = 0; ch < x.size(1); ++ch) {
        //                 globalpool_file << x_ptr[face_idx * x.size(1) + ch];
        //                 if (ch < x.size(1) - 1) globalpool_file << " ";
        //             }
        //             globalpool_file << "\n";
        //         }
        //         globalpool_file.close();
        //         // std::cout << "[DEBUG] GlobalPool 后的完整数据已保存到 cpp_feature_maps/uvnet_surface_globalpool_debug.txt" << std::endl;
        //     }
        // }

        // Flatten: [N, C, 1, 1] -> [N, C]
        x = x.view({ x.size(0), -1 });

        // std::cout << "[DEBUG] After Flatten: shape=[" << x.size(0) << ", " << x.size(1) << "]" << std::endl;

        // FC: 128 -> 64
        x = fc_block(x, "surface_encoder.fc");

        // std::cout << "[OUTPUT] Final shape: [" << x.size(0) << ", " << x.size(1) << "]" << std::endl;
        // std::cout << "=== SURFACE ENCODER DONE ===" << std::endl;

        return x;
    }
};
TORCH_MODULE(UVNetSurfaceEncoder)


// Curve encoder in UVNet.h
class UVNetCurveEncoderImpl : public breptorch::nn::Module {
private:
    std::map<std::string, breptorch::Tensor> params;
    std::map<std::string, breptorch::Tensor> buffers;

public:
    // Default constructor
    UVNetCurveEncoderImpl() = default;
    
    void load_weights(const std::map<std::string, breptorch::Tensor>& weight_dict) {
        // std::cout << std::fixed << std::setprecision(10);  // 设置输出精度
        // std::cout << "[UVNetCurveEncoder] Loading " << weight_dict.size() << " weights..." << std::endl;

        for (auto const& [key, val] : weight_dict) {
            if (key.find("running") != std::string::npos) buffers[key] = val;
            else params[key] = val;

            // 打印权重信息（仅打印第一个卷积层的权重）
            if (key == "conv_layers.0.0.weight") {
                // std::cout << "[Debug Weight] " << key << " shape: [";
                for (size_t i = 0; i < val.sizes().size(); ++i) {
                    // std::cout << val.sizes()[i];
                    // if (i < val.sizes().size() - 1) std::cout << ", ";
                }
                // std::cout << "]" << std::endl;
                // std::cout << "  First 10 values: ";
                // 创建非const副本来访问data_ptr
                breptorch::Tensor val_copy = val.clone();
                for (int i = 0; i < std::min(10, (int)val.numel()); ++i) {
                    // std::cout << val_copy.data_ptr<float>()[i] << " ";
                }
                // std::cout << std::endl;
            }
        }
        // std::cout << "[UVNetCurveEncoder] Loaded " << params.size() << " params, " << buffers.size() << " buffers" << std::endl;
    }

    // 1D convolution block
    breptorch::Tensor conv1d_block(breptorch::Tensor x, std::string prefix) {
        // Conv1d(in, out, kernel=3, padding=1)
        auto w = params[prefix + ".0.weight"];

        // DEBUG: 输入信息
        // std::cout << "\n[DEBUG] " << prefix << " INPUT: shape=[" << x.size(0) << ", " << x.size(1) << ", "
        //           << x.size(2) << "]" << std::endl;

        x = breptorch::conv1d(x, w, {}, { 1 }, { 1 }, { 1 }, 1);

        // DEBUG: Conv1d输出 - 检查完整张量（已禁用）
        // float conv_max = -1e9f, conv_min = 1e9f;
        // const float* data = x.data_ptr<float>();
        // for (int64_t i = 0; i < x.numel(); ++i) {
        //     float val = data[i];
        //     conv_max = std::max(conv_max, val);
        //     conv_min = std::min(conv_min, val);
        // }
        // std::cout << "[DEBUG] " << prefix << " AFTER CONV1D: shape=[" << x.size(0) << ", " << x.size(1) << ", "
        //           << x.size(2) << "] max=" << conv_max << " min=" << conv_min << std::endl;

        // BatchNorm1d
        auto bn_mean = buffers[prefix + ".1.running_mean"];
        auto bn_var = buffers[prefix + ".1.running_var"];
        auto bn_w = params[prefix + ".1.weight"];
        auto bn_b = params[prefix + ".1.bias"];

        // DEBUG: BatchNorm参数
        // std::cout << "[DEBUG] " << prefix << " BN PARAMS: mean[0]=" << bn_mean.at({0})
        //           << " var[0]=" << bn_var.at({0}) << " weight[0]=" << bn_w.at({0})
        //           << " bias[0]=" << bn_b.at({0}) << std::endl;

        x = breptorch::batch_norm(x, bn_w, bn_b, bn_mean, bn_var, false, 0.1, 1e-5, true);

        // DEBUG: BatchNorm输出 - 检查完整张量（已禁用）
        // float bn_max = -1e9f, bn_min = 1e9f;
        // data = x.data_ptr<float>();
        // for (int64_t i = 0; i < x.numel(); ++i) {
        //     float val = data[i];
        //     bn_max = std::max(bn_max, val);
        //     bn_min = std::min(bn_min, val);
        // }
        // std::cout << "[DEBUG] " << prefix << " AFTER BATCHNORM: max=" << bn_max << " min=" << bn_min << std::endl;

        x = breptorch::leaky_relu(x, 0.01);

        // DEBUG: ReLU输出 - 检查完整张量（已禁用）
        // float relu_max = -1e9f, relu_min = 1e9f;
        // data = x.data_ptr<float>();
        // for (int64_t i = 0; i < x.numel(); ++i) {
        //     float val = data[i];
        //     relu_max = std::max(relu_max, val);
        //     relu_min = std::min(relu_min, val);
        // }
        // std::cout << "[DEBUG] " << prefix << " AFTER RELU: max=" << relu_max << " min=" << relu_min << std::endl;

        return x;
    }

    // FC block (same as Surface Encoder)
    breptorch::Tensor fc_block(breptorch::Tensor x, std::string prefix) {
        auto w = params[prefix + ".0.weight"];
        x = breptorch::linear(x, w, {});

        auto bn_mean = buffers[prefix + ".1.running_mean"];
        auto bn_var = buffers[prefix + ".1.running_var"];
        auto bn_w = params[prefix + ".1.weight"];
        auto bn_b = params[prefix + ".1.bias"];
        x = breptorch::batch_norm(x, bn_w, bn_b, bn_mean, bn_var, false, 0.1, 1e-5, true);

        x = breptorch::leaky_relu(x, 0.01);
        return x;
    }

    breptorch::Tensor forward(breptorch::Tensor x) {
        // x: [N, 13, 10]
        // Check if weights are loaded
        if (params.find("curve_encoder.conv1.0.weight") == params.end()) {
            ERR_LOG << "[Error] curve_encoder weights not loaded!" << std::endl;
            return breptorch::Tensor();
        }

        // std::cout << "\n=== CURVE ENCODER FORWARD ===" << std::endl;
        // std::cout << "[INPUT] Shape: [" << x.size(0) << ", " << x.size(1) << ", " << x.size(2) << "]" << std::endl;

        x = conv1d_block(x, "curve_encoder.conv1");
        x = conv1d_block(x, "curve_encoder.conv2");
        // Note: Check if conv3 exists
        if (params.find("curve_encoder.conv3.0.weight") != params.end()) {
            x = conv1d_block(x, "curve_encoder.conv3");
        }

        // std::cout << "[DEBUG] Before Global Pool: shape=[" << x.size(0) << ", " << x.size(1) << ", " << x.size(2) << "]" << std::endl;

        // Global Pool 1D: [N, C, L] -> [N, C, 1]
        x = breptorch::adaptive_avg_pool1d(x, { 1 });
        x = x.view({ x.size(0), -1 }); // Flatten

        // std::cout << "[DEBUG] After Flatten: shape=[" << x.size(0) << ", " << x.size(1) << "]" << std::endl;

        x = fc_block(x, "curve_encoder.fc");

        // std::cout << "[OUTPUT] Final shape: [" << x.size(0) << ", " << x.size(1) << "]" << std::endl;
        // std::cout << "=== CURVE ENCODER DONE ===" << std::endl;

        return x;
    }
};
TORCH_MODULE(UVNetCurveEncoder)

} // namespace nn
} // namespace breptorch