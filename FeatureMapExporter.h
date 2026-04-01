#ifndef FEATURE_MAP_EXPORTER_H

// 包含调试配置
#include "DebugControl.h"

#define FEATURE_MAP_EXPORTER_H

#include "BRepTorch.h"
#include <string>
#include <fstream>
#include <iomanip>
#include <filesystem>

namespace fs = std::filesystem;

/**
 * Feature Map 导出器
 * 功能：模仿 cpp_logits/cpp_probs 的方式，将中间层输出保存到文件
 * 目的：方便与Python端逐层对比，定位误差源头
 */
class FeatureMapExporter {
public:
    FeatureMapExporter(const std::string& base_dir = "cpp_feature_maps")
        : base_dir_(base_dir) {
        // 仅在 export 模式下创建目录
        if (EXPORT_ENABLED) {
            if (!fs::exists(base_dir_)) {
                fs::create_directory(base_dir_);
                DBG_LOG << "  Creating directory: " << base_dir_ << std::endl;
            }
        }
    }

    /**
     * 导出Tensor到指定子文件夹
     */
    void exportTensor(const breptorch::Tensor& tensor,
                     const std::string& layer_name,
                     const std::string& step_file_name) {
        if (!EXPORT_ENABLED) return;

        // 创建子文件夹
        std::string layer_dir = base_dir_ + "/" + layer_name;
        if (!fs::exists(layer_dir)) {
            fs::create_directory(layer_dir);
        }

        // 生成文件路径
        std::string filename = layer_dir + "/" + step_file_name + "_result.txt";
        std::ofstream file(filename, std::ios::out);

        if (!file.is_open()) {
            ERR_LOG << "  [错误] 无法创建文件: " << filename << std::endl;
            return;
        }

        // 写入UTF-8 BOM
        file << "\xEF\xBB\xBF";

        // 设置输出格式（科学计数法，精度20位）
        file << std::scientific << std::setprecision(20);

        // 获取维度数
        size_t ndim = tensor.sizes_.size();

        if (ndim == 2) {
            // 2D张量：[num_samples, num_features]
            int64_t num_samples = tensor.sizes_[0];
            int64_t num_features = tensor.sizes_[1];

            for (int64_t i = 0; i < num_samples; ++i) {
                for (int64_t j = 0; j < num_features; ++j) {
                    if (j > 0) file << " ";
                    file << tensor.at({i, j});
                }
                file << "\n";
            }
        } else if (ndim == 1) {
            // 1D张量：[num_samples]
            int64_t num_samples = tensor.sizes_[0];
            for (int64_t i = 0; i < num_samples; ++i) {
                if (i > 0) file << " ";
                file << tensor.at({i});
            }
            file << "\n";
        } else if (ndim == 3) {
            // 3D张量：[num_samples, C, L] -> 展平为2D
            int64_t num_samples = tensor.sizes_[0];
            int64_t C = tensor.sizes_[1];
            int64_t L = tensor.sizes_[2];

            for (int64_t i = 0; i < num_samples; ++i) {
                for (int64_t j = 0; j < C; ++j) {
                    for (int64_t k = 0; k < L; ++k) {
                        if (j > 0 || k > 0) file << " ";
                        file << tensor.at({i, j, k});
                    }
                }
                file << "\n";
            }
        } else if (ndim == 4) {
            // 4D张量：[num_samples, C, H, W] -> 展平为2D
            int64_t num_samples = tensor.sizes_[0];
            int64_t C = tensor.sizes_[1];
            int64_t H = tensor.sizes_[2];
            int64_t W = tensor.sizes_[3];

            for (int64_t i = 0; i < num_samples; ++i) {
                for (int64_t j = 0; j < C; ++j) {
                    for (int64_t k = 0; k < H; ++k) {
                        for (int64_t l = 0; l < W; ++l) {
                            if (j > 0 || k > 0 || l > 0) file << " ";
                            file << tensor.at({i, j, k, l});
                        }
                    }
                }
                file << "\n";
            }
        }

        file.close();
    }

    /**
     * 导出vector<vector<float>>（从coedges/faces/edges的状态）
     */
    void exportVectorData(const std::vector<std::vector<float>>& data,
                         const std::string& layer_name,
                         const std::string& step_file_name) {
        if (!EXPORT_ENABLED) return;

        // 创建子文件夹
        std::string layer_dir = base_dir_ + "/" + layer_name;
        if (!fs::exists(layer_dir)) {
            fs::create_directory(layer_dir);
        }

        // 生成文件路径
        std::string filename = layer_dir + "/" + step_file_name + "_result.txt";
        std::ofstream file(filename, std::ios::out);

        if (!file.is_open()) {
            ERR_LOG << "  [错误] 无法创建文件: " << filename << std::endl;
            return;
        }

        // 写入UTF-8 BOM
        file << "\xEF\xBB\xBF";

        // 设置输出格式
        file << std::scientific << std::setprecision(20);

        // 写入数据
        for (const auto& row : data) {
            for (size_t i = 0; i < row.size(); ++i) {
                if (i > 0) file << " ";
                file << row[i];
            }
            file << "\n";
        }

        file.close();
    }

    void printExportSummary() const {
        if (!EXPORT_ENABLED) return;

        DBG_LOG << "\n=== Feature Map Export Summary ===" << std::endl;
        DBG_LOG << "Base directory: " << base_dir_ << std::endl;
        DBG_LOG << "\nExported layers:" << std::endl;

        const std::vector<std::string> layers = {
            "uvnet_surface", "uvnet_curve",
            "layer0_input_concat", "layer0_mlp_output",
            "layer0_edge_pooling", "layer0_face_pooling",
            "layer1_input_concat", "layer1_mlp_output",
            "layer1_edge_pooling", "layer1_face_pooling",
            "output_layer_input_concat", "output_layer_mlp_output",
            "output_layer_face_embedding"
        };

        for (const auto& layer : layers) {
            std::string layer_dir = base_dir_ + "/" + layer;
            if (fs::exists(layer_dir)) {
                int file_count = 0;
                for (const auto& entry : fs::directory_iterator(layer_dir)) {
                    if (entry.is_regular_file()) file_count++;
                }
                DBG_LOG << "  OK " << layer << " (" << file_count << " files)" << std::endl;
            }
        }
    }

private:
    std::string base_dir_;
};

#endif // FEATURE_MAP_EXPORTER_H
