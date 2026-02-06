#pragma once
//#include <torch/torch.h>
#include "BRepTorch.h"
#include <map>
#include <vector>
#include <string>
#include <cmath>

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
    // Load weights from npz file
    void load_weights(const std::map<std::string, breptorch::Tensor>& weight_dict) {
        std::cout << "[UVNetSurfaceEncoder] Loading " << weight_dict.size() << " weights..." << std::endl;
        for (auto const& [key, val] : weight_dict) {
            if (key.find("running") != std::string::npos) {
                buffers[key] = val;
            }
            else {
                params[key] = val;
            }
        }
        std::cout << "[UVNetSurfaceEncoder] Loaded " << params.size() << " params, " << buffers.size() << " buffers" << std::endl;
    }
    // Basic building block: Conv2d + BN + LeakyReLU
    breptorch::Tensor conv2d_block(breptorch::Tensor x, std::string prefix) {
        // 1. Conv2d
        std::string weight_key = prefix + ".0.weight";
        if (params.find(weight_key) == params.end()) {
            std::cerr << "[Error] Weight not found: " << weight_key << std::endl;
            return breptorch::Tensor();
        }

        auto w = params[weight_key];
        auto conv_opts = breptorch::nn::functional::Conv2dFuncOptions()
            .stride(1)
            .padding(1);

        x = breptorch::nn::functional::conv2d(x, w, conv_opts);

        // 2. BatchNorm
        auto bn_mean = buffers[prefix + ".1.running_mean"];
        auto bn_var = buffers[prefix + ".1.running_var"];
        auto bn_w = params[prefix + ".1.weight"];
        auto bn_b = params[prefix + ".1.bias"];

        auto bn_opts = breptorch::nn::functional::BatchNormFuncOptions()
            .weight(bn_w)
            .bias(bn_b)
            .training(false)
            .momentum(0.1)
            .eps(1e-5);

        x = breptorch::nn::functional::batch_norm(x, bn_mean, bn_var, bn_opts);

        // 3. LeakyReLU
        x = breptorch::leaky_relu(x, 0.01);
        return x;
    }

    // Basic fully connected block: FC Block
    breptorch::Tensor fc_block(breptorch::Tensor x, std::string prefix) {
        // 1. Linear
        // functional::linear takes 3 params: (input, weight, bias)
        // If no bias, pass an empty Tensor, like {} or Tensor()

        auto w = params[prefix + ".0.weight"];

        // Note: passing empty Tensor as bias here
        x = breptorch::nn::functional::linear(x, w, breptorch::Tensor());

        // 2. BatchNorm1d
        // functional::batch_norm works for both 1D and 2D
        auto bn_mean = buffers[prefix + ".1.running_mean"];
        auto bn_var = buffers[prefix + ".1.running_var"];
        auto bn_w = params[prefix + ".1.weight"];
        auto bn_b = params[prefix + ".1.bias"];

        auto bn_opts = breptorch::nn::functional::BatchNormFuncOptions()
            .weight(bn_w)
            .bias(bn_b)
            .training(false)
            .momentum(0.1)
            .eps(1e-5);

        x = breptorch::nn::functional::batch_norm(x, bn_mean, bn_var, bn_opts);

        // 3. LeakyReLU
        x = breptorch::leaky_relu(x, 0.01);
        return x;
    }

    breptorch::Tensor forward(breptorch::Tensor x) {
        if (params.find("surface_encoder.conv1.0.weight") == params.end()) {
            std::cerr << "[Error] Weight not found: surface_encoder.conv1.0.weight" << std::endl;
            return breptorch::Tensor();
        }

        // Conv1: 9 -> 64
        x = conv2d_block(x, "surface_encoder.conv1");

        // Conv2: 64 -> 128
        x = conv2d_block(x, "surface_encoder.conv2");

        // Global Pool: [N, C, H, W] -> [N, C, 1, 1]
        x = breptorch::adaptive_avg_pool2d(x, { 1, 1 });

        // Flatten: [N, C, 1, 1] -> [N, C]
        x = x.view({ x.size(0), -1 });

        // FC: 128 -> 64
        x = fc_block(x, "surface_encoder.fc");

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
    void load_weights(const std::map<std::string, breptorch::Tensor>& weight_dict) {
        std::cout << "[UVNetCurveEncoder] Loading " << weight_dict.size() << " weights..." << std::endl;
        for (auto const& [key, val] : weight_dict) {
            if (key.find("running") != std::string::npos) buffers[key] = val;
            else params[key] = val;
        }
        std::cout << "[UVNetCurveEncoder] Loaded " << params.size() << " params, " << buffers.size() << " buffers" << std::endl;
    }

    // 1D convolution block
    breptorch::Tensor conv1d_block(breptorch::Tensor x, std::string prefix) {
        // Conv1d(in, out, kernel=3, padding=1)
        auto w = params[prefix + ".0.weight"];
        // C++ API: conv1d(input, weight, bias, stride, padding, dilation, groups)
        x = breptorch::conv1d(x, w, {}, { 1 }, { 1 }, { 1 }, 1);

        // BatchNorm1d
        auto bn_mean = buffers[prefix + ".1.running_mean"];
        auto bn_var = buffers[prefix + ".1.running_var"];
        auto bn_w = params[prefix + ".1.weight"];
        auto bn_b = params[prefix + ".1.bias"];
        x = breptorch::batch_norm(x, bn_w, bn_b, bn_mean, bn_var, false, 0.1, 1e-5, true);

        x = breptorch::leaky_relu(x, 0.01);
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
            std::cerr << "[Error] curve_encoder weights not loaded!" << std::endl;
            return breptorch::Tensor();
        }

        x = conv1d_block(x, "curve_encoder.conv1");
        x = conv1d_block(x, "curve_encoder.conv2");
        // Note: Check if conv3 exists
        if (params.find("curve_encoder.conv3.0.weight") != params.end()) {
            x = conv1d_block(x, "curve_encoder.conv3");
        }

        // Global Pool 1D: [N, C, L] -> [N, C, 1]
        x = breptorch::adaptive_avg_pool1d(x, { 1 });
        x = x.view({ x.size(0), -1 }); // Flatten

        x = fc_block(x, "curve_encoder.fc");
        return x;
    }
};
TORCH_MODULE(UVNetCurveEncoder)

} // namespace nn
} // namespace breptorch
