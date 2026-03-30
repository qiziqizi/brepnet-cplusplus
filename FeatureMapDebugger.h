#ifndef FEATURE_MAP_DEBUGGER_H
#define FEATURE_MAP_DEBUGGER_H

#include "BRepTorch.h"
#include <string>
#include <fstream>
#include <iomanip>
#include <vector>
#include <map>

/**
 * Feature Map 调试器
 * 功能：在BRepNet的每一层输出中间结果，用于与Python端对比
 * 目的：定位class 9/24/25误差巨大的根源
 */
class FeatureMapDebugger {
public:
    FeatureMapDebugger(const std::string& output_dir = "debug_feature_maps")
        : output_dir_(output_dir), enabled_(true) {
        // 创建输出目录
        std::filesystem::create_directories(output_dir_);
    }

    // 启用/禁用调试
    void setEnabled(bool enabled) { enabled_ = enabled; }
    bool isEnabled() const { return enabled_; }

    // ========== 核心功能：保存每一层的feature map ==========

    /**
     * 保存张量到文本文件
     * 格式：每行一个样本，空格分隔各个维度的值
     */
    void saveTensor(const breptorch::Tensor& tensor,
                    const std::string& layer_name,
                    const std::string& file_name) {
        if (!enabled_) return;

        std::string filepath = output_dir_ + "/" + file_name + "_" + layer_name + ".txt";
        std::ofstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "[FeatureMapDebugger] 无法创建文件: " << filepath << std::endl;
            return;
        }

        // 写入UTF-8 BOM
        file << "\xEF\xBB\xBF";

        // 写入形状信息作为注释
        file << "# Shape: [";
        for (size_t i = 0; i < tensor.ndim(); ++i) {
            if (i > 0) file << ", ";
            file << tensor.size(i);
        }
        file << "]\n";

        // 设置高精度输出
        file << std::scientific << std::setprecision(20);

        // 处理不同维度的张量
        if (tensor.ndim() == 1) {
            // 1D: [N]
            for (int64_t i = 0; i < tensor.size(0); ++i) {
                file << tensor.at({i}) << "\n";
            }
        } else if (tensor.ndim() == 2) {
            // 2D: [N, D]
            for (int64_t i = 0; i < tensor.size(0); ++i) {
                for (int64_t j = 0; j < tensor.size(1); ++j) {
                    if (j > 0) file << " ";
                    file << tensor.at({i, j});
                }
                file << "\n";
            }
        } else if (tensor.ndim() == 3) {
            // 3D: [N, C, L] - 展平为2D
            for (int64_t i = 0; i < tensor.size(0); ++i) {
                for (int64_t j = 0; j < tensor.size(1); ++j) {
                    for (int64_t k = 0; k < tensor.size(2); ++k) {
                        if (j > 0 || k > 0) file << " ";
                        file << tensor.at({i, j, k});
                    }
                }
                file << "\n";
            }
        } else if (tensor.ndim() == 4) {
            // 4D: [N, C, H, W] - 展平为2D
            for (int64_t i = 0; i < tensor.size(0); ++i) {
                for (int64_t j = 0; j < tensor.size(1); ++j) {
                    for (int64_t k = 0; k < tensor.size(2); ++k) {
                        for (int64_t l = 0; l < tensor.size(3); ++l) {
                            if (j > 0 || k > 0 || l > 0) file << " ";
                            file << tensor.at({i, j, k, l});
                        }
                    }
                }
                file << "\n";
            }
        }

        file.close();
        std::cout << "[FeatureMapDebugger] 已保存: " << filepath << std::endl;
    }

    /**
     * 保存带标签的feature map（用于分析特定面或特定类别）
     */
    void saveTensorWithLabels(const breptorch::Tensor& tensor,
                              const std::string& layer_name,
                              const std::string& file_name,
                              const std::vector<int>& labels) {
        if (!enabled_) return;

        std::string filepath = output_dir_ + "/" + file_name + "_" + layer_name + "_labeled.txt";
        std::ofstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "[FeatureMapDebugger] 无法创建文件: " << filepath << std::endl;
            return;
        }

        file << "\xEF\xBB\xBF";
        file << std::scientific << std::setprecision(20);

        // 只处理2D张量
        if (tensor.ndim() == 2) {
            for (int64_t i = 0; i < tensor.size(0); ++i) {
                // 写入标签
                if (i < static_cast<int64_t>(labels.size())) {
                    file << "# Face " << i << " (Class " << labels[i] << "): ";
                }

                for (int64_t j = 0; j < tensor.size(1); ++j) {
                    if (j > 0) file << " ";
                    file << tensor.at({i, j});
                }
                file << "\n";
            }
        }

        file.close();
        std::cout << "[FeatureMapDebugger] 已保存（带标签）: " << filepath << std::endl;
    }

    /**
     * 保存统计信息（min, max, mean, std）
     */
    void saveTensorStats(const breptorch::Tensor& tensor,
                        const std::string& layer_name,
                        const std::string& file_name) {
        if (!enabled_) return;

        std::string filepath = output_dir_ + "/" + file_name + "_" + layer_name + "_stats.txt";
        std::ofstream file(filepath);
        if (!file.is_open()) return;

        file << "\xEF\xBB\xBF";
        file << "Layer: " << layer_name << "\n";
        file << "File: " << file_name << "\n";
        file << "Shape: [";
        for (size_t i = 0; i < tensor.ndim(); ++i) {
            if (i > 0) file << ", ";
            file << tensor.size(i);
        }
        file << "]\n\n";

        // 计算统计量
        int64_t total = 1;
        for (size_t i = 0; i < tensor.ndim(); ++i) {
            total *= tensor.size(i);
        }

        float min_val = 1e9f, max_val = -1e9f, sum = 0.0f;

        // 遍历所有元素（假设是float类型）
        auto data_ptr = tensor.data_ptr<float>();
        for (int64_t i = 0; i < total; ++i) {
            float val = data_ptr[i];
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
            sum += val;
        }

        float mean = sum / total;

        // 计算标准差
        float var_sum = 0.0f;
        for (int64_t i = 0; i < total; ++i) {
            float diff = data_ptr[i] - mean;
            var_sum += diff * diff;
        }
        float std_dev = std::sqrt(var_sum / total);

        file << std::scientific << std::setprecision(10);
        file << "Min: " << min_val << "\n";
        file << "Max: " << max_val << "\n";
        file << "Mean: " << mean << "\n";
        file << "Std: " << std_dev << "\n";

        file.close();
    }

    /**
     * 检查是否有异常值（NaN, Inf, 超大值）
     */
    bool checkAnomalies(const breptorch::Tensor& tensor,
                       const std::string& layer_name,
                       float threshold = 1e6f) {
        int64_t total = 1;
        for (size_t i = 0; i < tensor.ndim(); ++i) {
            total *= tensor.size(i);
        }

        auto data_ptr = tensor.data_ptr<float>();
        int nan_count = 0, inf_count = 0, large_count = 0;

        for (int64_t i = 0; i < total; ++i) {
            float val = data_ptr[i];
            if (std::isnan(val)) nan_count++;
            else if (std::isinf(val)) inf_count++;
            else if (std::abs(val) > threshold) large_count++;
        }

        if (nan_count > 0 || inf_count > 0 || large_count > 0) {
            std::cerr << "[警告] " << layer_name << " 包含异常值:\n";
            std::cerr << "  NaN: " << nan_count << "\n";
            std::cerr << "  Inf: " << inf_count << "\n";
            std::cerr << "  超大值(>" << threshold << "): " << large_count << "\n";
            return true;
        }
        return false;
    }

private:
    std::string output_dir_;
    bool enabled_;
};

#endif // FEATURE_MAP_DEBUGGER_H
