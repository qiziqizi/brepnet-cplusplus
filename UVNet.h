#pragma once
#include <torch/torch.h>
#include <map>
#include <vector>
#include <string>
#include <cmath>

// 一个简单的 UV-Net 推理器，基于 Torch Tensor 操作
// 这样你就不用手写 conv2d 了，直接用 PyTorch C++ API
class UVNetSurfaceEncoderImpl : public torch::nn::Module {
private:
    // 为了灵活性，我们在这里手动管理权重
    // Key 是 Python 导出时的名字，Value 是 Tensor
    std::map<std::string, torch::Tensor> params;
    std::map<std::string, torch::Tensor> buffers; // 存 BatchNorm 的 running_mean/var

public:
    // 从 npz 加载权重
    void load_weights(const std::map<std::string, torch::Tensor>& weight_dict) {
        for (auto const& [key, val] : weight_dict) {
            if (key.find("running") != std::string::npos) {
                buffers[key] = val;
            }
            else {
                params[key] = val;
            }
        }
    }
    // 辅助函数：模拟 Conv2d + BN + LeakyReLU
    torch::Tensor conv2d_block(torch::Tensor x, std::string prefix) {
        // 1. Conv2d
        // 修正：LibTorch functional::conv2d 接受 3 个参数: (input, weight, options)
        // Bias 需要在 options 中设置。因为 Python 端 bias=False，所以这里不设置 bias。

        auto w = params[prefix + ".0.weight"];

        auto conv_opts = torch::nn::functional::Conv2dFuncOptions()
            .stride(1)
            .padding(1);
        // .bias(tensor)  <-- 如果有偏置，在这里设置

        x = torch::nn::functional::conv2d(x, w, conv_opts);

        // 2. BatchNorm
        // 保持不变，使用 Options 传递参数
        auto bn_mean = buffers[prefix + ".1.running_mean"];
        auto bn_var = buffers[prefix + ".1.running_var"];
        auto bn_w = params[prefix + ".1.weight"];
        auto bn_b = params[prefix + ".1.bias"];

        auto bn_opts = torch::nn::functional::BatchNormFuncOptions()
            .weight(bn_w)
            .bias(bn_b)
            .training(false)
            .momentum(0.1)
            .eps(1e-5);

        x = torch::nn::functional::batch_norm(x, bn_mean, bn_var, bn_opts);

        // 3. LeakyReLU
        x = torch::leaky_relu(x, 0.01);
        return x;
    }

    // 辅助函数：全连接层 FC Block
    torch::Tensor fc_block(torch::Tensor x, std::string prefix) {
        // 1. Linear
        // torch::nn::functional::linear 接受 3 个参数: (input, weight, bias)
        // 如果没有 bias，可以传一个空的 Tensor，即 {} 或 torch::Tensor()

        auto w = params[prefix + ".0.weight"];

        // 修正：这里传入空 Tensor 作为 bias
        x = torch::nn::functional::linear(x, w, torch::Tensor());

        // 2. BatchNorm1d
        // functional::batch_norm 对 1D 和 2D 都可以使用
        auto bn_mean = buffers[prefix + ".1.running_mean"];
        auto bn_var = buffers[prefix + ".1.running_var"];
        auto bn_w = params[prefix + ".1.weight"];
        auto bn_b = params[prefix + ".1.bias"];

        auto bn_opts = torch::nn::functional::BatchNormFuncOptions()
            .weight(bn_w)
            .bias(bn_b)
            .training(false)
            .momentum(0.1)
            .eps(1e-5);

        x = torch::nn::functional::batch_norm(x, bn_mean, bn_var, bn_opts);

        // 3. LeakyReLU
        x = torch::leaky_relu(x, 0.01);
        return x;
    }

    torch::Tensor forward(torch::Tensor x) {
        // Layer 1
        x = conv2d_block(x, "surface_encoder.conv1");
        // Layer 2
        x = conv2d_block(x, "surface_encoder.conv2");
        // Layer 3
        x = conv2d_block(x, "surface_encoder.conv3");

        // Global Pool: [N, C, H, W] -> [N, C, 1, 1]
        x = torch::adaptive_avg_pool2d(x, { 1, 1 });

        // Flatten: [N, C, 1, 1] -> [N, C]
        x = x.view({ x.size(0), -1 });

        // FC
        x = fc_block(x, "surface_encoder.fc");

        return x;
    }
};
TORCH_MODULE(UVNetSurfaceEncoder);


// 在 UVNet.h 中新增
class UVNetCurveEncoderImpl : public torch::nn::Module {
private:
    std::map<std::string, torch::Tensor> params;
    std::map<std::string, torch::Tensor> buffers;

public:
    void load_weights(const std::map<std::string, torch::Tensor>& weight_dict) {
        for (auto const& [key, val] : weight_dict) {
            if (key.find("running") != std::string::npos) buffers[key] = val;
            else params[key] = val;
        }
    }

    // 1D 卷积块
    torch::Tensor conv1d_block(torch::Tensor x, std::string prefix) {
        // Conv1d(in, out, kernel=3, padding=1)
        auto w = params[prefix + ".0.weight"];
        // C++ API: conv1d(input, weight, bias, stride, padding, dilation, groups)
        x = torch::conv1d(x, w, {}, { 1 }, { 1 }, { 1 }, 1);

        // BatchNorm1d
        auto bn_mean = buffers[prefix + ".1.running_mean"];
        auto bn_var = buffers[prefix + ".1.running_var"];
        auto bn_w = params[prefix + ".1.weight"];
        auto bn_b = params[prefix + ".1.bias"];
        x = torch::batch_norm(x, bn_w, bn_b, bn_mean, bn_var, false, 0.1, 1e-5, true);

        x = torch::leaky_relu(x, 0.01);
        return x;
    }

    // FC 块 (和 Surface Encoder 一样)
    torch::Tensor fc_block(torch::Tensor x, std::string prefix) {
        auto w = params[prefix + ".0.weight"];
        x = torch::linear(x, w, {});

        auto bn_mean = buffers[prefix + ".1.running_mean"];
        auto bn_var = buffers[prefix + ".1.running_var"];
        auto bn_w = params[prefix + ".1.weight"];
        auto bn_b = params[prefix + ".1.bias"];
        x = torch::batch_norm(x, bn_w, bn_b, bn_mean, bn_var, false, 0.1, 1e-5, true);

        x = torch::leaky_relu(x, 0.01);
        return x;
    }

    torch::Tensor forward(torch::Tensor x) {
        // x: [N, 12, 10]
        x = conv1d_block(x, "curve_encoder.conv1");
        x = conv1d_block(x, "curve_encoder.conv2");
        x = conv1d_block(x, "curve_encoder.conv3");

        // Global Pool 1D: [N, C, L] -> [N, C, 1]
        x = torch::adaptive_avg_pool1d(x, { 1 });
        x = x.view({ x.size(0), -1 }); // Flatten

        x = fc_block(x, "curve_encoder.fc");
        return x;
    }
};
TORCH_MODULE(UVNetCurveEncoder);