#pragma once
#include "BRepTorch.h"
#include "DebugControl.h"
#include <map>
#include <vector>
#include <string>
#include <cmath>

using Tensor = breptorch::Tensor;
using namespace breptorch;

namespace breptorch {
namespace nn {

// UV-Net Surface Encoder
class UVNetSurfaceEncoderImpl : public breptorch::nn::Module {
private:
    std::map<std::string, breptorch::Tensor> params;
    std::map<std::string, breptorch::Tensor> buffers;

public:
    UVNetSurfaceEncoderImpl() = default;

    void load_weights(const std::map<std::string, breptorch::Tensor>& weight_dict) {
        for (auto const& [key, val] : weight_dict) {
            std::string stored_key = key;
            breptorch::Tensor stored_val = val;

            // SymmetricConv2d: expand weight_quarter [out,in,3,3] -> [out,in,5,5]
            if (key.find("weight_quarter") != std::string::npos) {
                breptorch::Tensor w = val.clone();
                breptorch::Tensor right = breptorch::flip(w.slice(3, 0, w.size(3) - 1), {3});
                breptorch::Tensor w_h = breptorch::cat({w, right}, 3);
                breptorch::Tensor bottom = breptorch::flip(w_h.slice(2, 0, w_h.size(2) - 1), {2});
                breptorch::Tensor w_full = breptorch::cat({w_h, bottom}, 2);
                stored_val = w_full;
                stored_key = key.substr(0, key.find("weight_quarter")) + "weight";
            }

            if (stored_key.find("running") != std::string::npos) {
                buffers[stored_key] = stored_val;
            } else {
                params[stored_key] = stored_val;
            }
        }
    }

    // Conv2d + BatchNorm + LeakyReLU
    breptorch::Tensor conv2d_block(breptorch::Tensor x, std::string prefix) {
        std::string weight_key = prefix + ".0.weight";
        if (params.find(weight_key) == params.end()) {
            ERR_LOG << "[Error] Weight not found: " << weight_key << std::endl;
            return breptorch::Tensor();
        }

        auto w = params[weight_key];
        auto conv_opts = breptorch::nn::functional::Conv2dFuncOptions()
            .stride(1)
            .padding(2);

        x = breptorch::nn::functional::conv2d(x, w, conv_opts);

        // BatchNorm
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

        // LeakyReLU
        x = breptorch::leaky_relu(x, 0.01);

        return x;
    }

    // FC + BatchNorm1d + LeakyReLU
    breptorch::Tensor fc_block(breptorch::Tensor x, std::string prefix) {
        auto w = params[prefix + ".0.weight"];
        x = breptorch::nn::functional::linear(x, w, breptorch::Tensor());

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

        x = breptorch::leaky_relu(x, 0.01);

        return x;
    }

    breptorch::Tensor forward(breptorch::Tensor x) {
        if (params.find("surface_encoder.conv1.0.weight") == params.end()) {
            ERR_LOG << "[Error] Weight not found: surface_encoder.conv1.0.weight" << std::endl;
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


// UV-Net Curve Encoder
class UVNetCurveEncoderImpl : public breptorch::nn::Module {
private:
    std::map<std::string, breptorch::Tensor> params;
    std::map<std::string, breptorch::Tensor> buffers;

public:
    UVNetCurveEncoderImpl() = default;

    void load_weights(const std::map<std::string, breptorch::Tensor>& weight_dict) {
        for (auto const& [key, val] : weight_dict) {
            std::string stored_key = key;
            breptorch::Tensor stored_val = val;

            // SymmetricConv1d: expand weight_half [out,in,3] -> [out,in,5]
            if (key.find("weight_half") != std::string::npos) {
                breptorch::Tensor w = val.clone();
                breptorch::Tensor right = breptorch::flip(w.slice(2, 0, w.size(2) - 1), {2});
                breptorch::Tensor w_full = breptorch::cat({w, right}, 2);
                stored_val = w_full;
                stored_key = key.substr(0, key.find("weight_half")) + "weight";
            }

            if (stored_key.find("running") != std::string::npos) buffers[stored_key] = stored_val;
            else params[stored_key] = stored_val;
        }
    }

    // Conv1d + BatchNorm1d + LeakyReLU
    breptorch::Tensor conv1d_block(breptorch::Tensor x, std::string prefix) {
        auto w = params[prefix + ".0.weight"];
        x = breptorch::conv1d(x, w, {}, { 1 }, { 2 }, { 1 }, 1);

        auto bn_mean = buffers[prefix + ".1.running_mean"];
        auto bn_var = buffers[prefix + ".1.running_var"];
        auto bn_w = params[prefix + ".1.weight"];
        auto bn_b = params[prefix + ".1.bias"];

        x = breptorch::batch_norm(x, bn_w, bn_b, bn_mean, bn_var, false, 0.1, 1e-5, true);
        x = breptorch::leaky_relu(x, 0.01);

        return x;
    }

    // FC + BatchNorm1d + LeakyReLU
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
        // x: [N, 13, 20]
        if (params.find("curve_encoder.conv1.0.weight") == params.end()) {
            ERR_LOG << "[Error] curve_encoder weights not loaded!" << std::endl;
            return breptorch::Tensor();
        }

        x = conv1d_block(x, "curve_encoder.conv1");
        x = conv1d_block(x, "curve_encoder.conv2");
        if (params.find("curve_encoder.conv3.0.weight") != params.end()) {
            x = conv1d_block(x, "curve_encoder.conv3");
        }

        // Global Pool 1D: [N, C, L] -> [N, C, 1]
        x = breptorch::adaptive_avg_pool1d(x, { 1 });
        x = x.view({ x.size(0), -1 });

        x = fc_block(x, "curve_encoder.fc");

        return x;
    }
};
TORCH_MODULE(UVNetCurveEncoder)

} // namespace nn
} // namespace breptorch
