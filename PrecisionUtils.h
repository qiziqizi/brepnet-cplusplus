#pragma once

#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <stdexcept>

namespace breptorch {

// ---------------------------------------------------------------------------
// bf16 → float （纯位运算，无依赖）
// bf16 就是 float 的高 16 位：指数 8 位 + 尾数 7 位 + 符号位
// 直接左移 16 位即得到 float（截断模式，与 numpy 默认一致）
// ---------------------------------------------------------------------------
inline float bf16_to_float(uint16_t bf16_val) {
    uint32_t f = static_cast<uint32_t>(bf16_val) << 16;
    float result;
    std::memcpy(&result, &f, sizeof(result));
    return result;
}

// ---------------------------------------------------------------------------
// fp16 → float （纯位运算）
// fp16: 符号 1 位 + 指数 5 位 + 尾数 10 位
// float: 符号 1 位 + 指数 8 位 + 尾数 23 位
// ---------------------------------------------------------------------------
inline float fp16_to_float(uint16_t fp16_val) {
    uint32_t sign = (static_cast<uint32_t>(fp16_val) >> 15) & 1;
    uint32_t exp  = (static_cast<uint32_t>(fp16_val) >> 10) & 0x1F;
    uint32_t mant = static_cast<uint32_t>(fp16_val) & 0x3FF;

    uint32_t f32;

    if (exp == 0) {
        if (mant == 0) {
            // Zero
            f32 = sign << 31;
        } else {
            // Denormalized: 0.xxxx * 2^(-14) → 规范化
            // 将尾数左移直到最高位为 1，同时调整指数
            exp = 127 - 15 + 1;  // 113
            while ((mant & 0x400) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FF;  // 去掉隐含的 1
            f32 = (sign << 31) | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        // Inf or NaN
        f32 = (sign << 31) | (0xFF << 23) | (mant << 13);
    } else {
        // Normalized
        exp = exp - 15 + 127;  // 指数偏移转换
        f32 = (sign << 31) | (exp << 23) | (mant << 13);
    }

    float result;
    std::memcpy(&result, &f32, sizeof(result));
    return result;
}

// ---------------------------------------------------------------------------
// 权重精度枚举
// ---------------------------------------------------------------------------
enum class WeightPrecision {
    FP32,   // float32
    FP16,   // float16
    BF16    // bfloat16
};

// ---------------------------------------------------------------------------
// 字符串 → WeightPrecision
// ---------------------------------------------------------------------------
inline WeightPrecision parse_precision(const std::string& s) {
    if (s == "fp32" || s == "FP32" || s == "float32") return WeightPrecision::FP32;
    if (s == "fp16" || s == "FP16" || s == "float16") return WeightPrecision::FP16;
    if (s == "bf16" || s == "BF16" || s == "bfloat16") return WeightPrecision::BF16;
    throw std::runtime_error("Unknown precision: " + s);
}

// ---------------------------------------------------------------------------
// 将任意精度的权重数据转换为 fp32 vector<float>
// 参数：
//   raw_data   — 原始数据指针（指向 NPZ 数组的起始位置）
//   num_vals   — 元素个数
//   precision  — 权重精度
// 返回：
//   转换后的 fp32 数据
// ---------------------------------------------------------------------------
inline std::vector<float> convert_weights_to_fp32(
    const void* raw_data,
    size_t num_vals,
    WeightPrecision precision)
{
    std::vector<float> result(num_vals);

    switch (precision) {
        case WeightPrecision::FP32: {
            const float* src = static_cast<const float*>(raw_data);
            std::memcpy(result.data(), src, num_vals * sizeof(float));
            break;
        }
        case WeightPrecision::FP16: {
            const uint16_t* src = static_cast<const uint16_t*>(raw_data);
            for (size_t i = 0; i < num_vals; ++i) {
                result[i] = fp16_to_float(src[i]);
            }
            break;
        }
        case WeightPrecision::BF16: {
            const uint16_t* src = static_cast<const uint16_t*>(raw_data);
            for (size_t i = 0; i < num_vals; ++i) {
                result[i] = bf16_to_float(src[i]);
            }
            break;
        }
    }

    return result;
}

// ---------------------------------------------------------------------------
// 从 cnpy::NpyArray 加载并转换为 fp32 Tensor
// 封装了精度判断和转换逻辑
// 注意：需要调用方 include "cnpy.h" 和 "BRepTorch.h"
// ---------------------------------------------------------------------------
// 使用示例：
//   auto arr = item.second;  // cnpy::NpyArray
//   std::vector<int64_t> shape(arr.shape.begin(), arr.shape.end());
//   Tensor t = load_weight_tensor(arr, shape, precision);
// ---------------------------------------------------------------------------

} // namespace breptorch
