#include "FaceClassifier.h"
#include "BRepNet.h"
#include "BRepNetAdapter.h"
#include "BRepPipeline.h"
#include "cnpy.h"
#include <iostream>

FaceClassifier::FaceClassifier()
    : modelLoaded_(false) {
}

FaceClassifier::~FaceClassifier() {
}

bool FaceClassifier::loadModel(const std::string& weightsPath) {
    try {
        std::cout << "[FaceClassifier] 正在加载模型权重: " << weightsPath << std::endl;

        // 创建模型（27个类别）
        model_ = std::make_shared<BRepNetImpl>(27);
        
        // 检查 surf_enc 和 curve_enc 是否有效
        std::cout << "[FaceClassifier] surf_enc 有效: " << (model_->surf_enc ? "yes" : "NO!") << std::endl;
        std::cout << "[FaceClassifier] curve_enc 有效: " << (model_->curve_enc ? "yes" : "NO!") << std::endl;

        // 加载NPZ权重文件
        cnpy::npz_t npz = cnpy::npz_load(weightsPath);
        
        // 打印 npz 文件中的键
        std::cout << "[FaceClassifier] NPZ 文件中的键数量: " << npz.size() << std::endl;
        int surf_count = 0, curve_count = 0;
        for (auto& item : npz) {
            if (item.first.find("surface_encoder") != std::string::npos) surf_count++;
            if (item.first.find("curve_encoder") != std::string::npos) curve_count++;
        }
        std::cout << "[FaceClassifier] surface_encoder 权重数量: " << surf_count << std::endl;
        std::cout << "[FaceClassifier] curve_encoder 权重数量: " << curve_count << std::endl;

        // 加载 UV-Net 权重（Surface Encoder + Curve Encoder + V4 surf_enc2）
        std::map<std::string, breptorch::Tensor> surf_weights;
        std::map<std::string, breptorch::Tensor> curve_weights;
        std::map<std::string, breptorch::Tensor> surf2_weights;
        for (auto& item : npz) {
            auto arr = item.second;
            std::vector<int64_t> shape(arr.shape.begin(), arr.shape.end());
            breptorch::Tensor t = breptorch::from_blob(
                arr.data<float>(), shape, breptorch::kFloat32).clone();

            // V4: 必须区分 surface_encoder. 和 surface_encoder2.
            if (item.first.substr(0, 17) == "surface_encoder2.") {
                // 替换前缀: surface_encoder2. → surface_encoder.
                surf2_weights["surface_encoder." + item.first.substr(17)] = t;
            } else
            if (item.first.find("surface_encoder") != std::string::npos) {
                surf_weights[item.first] = t;
            }
            if (item.first.find("curve_encoder") != std::string::npos) {
                curve_weights[item.first] = t;
            }
        }
        std::cout << "[FaceClassifier] 加载 surface_encoder 权重: " << surf_weights.size() << std::endl;
        model_->surf_enc->load_weights(surf_weights);

        std::cout << "[FaceClassifier] 加载 curve_encoder 权重: " << curve_weights.size() << std::endl;
        model_->curve_enc->load_weights(curve_weights);

        std::cout << "[FaceClassifier] 加载 surface_encoder2 权重: " << surf2_weights.size() << std::endl;
        model_->surf_enc2->load_weights(surf2_weights);

        // 加载 BRepNet 权重
        auto params = model_->named_parameters();
        std::cout << "[FaceClassifier] 模型参数数量: " << params.size() << std::endl;
        
        int loaded_count = 0;
        for (auto& item : npz) {
            std::string key = item.first;

            // 转换参数名（Python风格 → C++风格）
            if (key.find("layers.0.mlp") != std::string::npos) {
                key = "layer_0.mlp" + key.substr(key.find(".mlp") + 4);
            } else if (key.find("layers.1.mlp") != std::string::npos) {
                key = "layer_1.mlp" + key.substr(key.find(".mlp") + 4);
            }

            if (params.find(key) != params.end()) {
                auto arr = item.second;
                std::vector<int64_t> shape(arr.shape.begin(), arr.shape.end());
                *params[key] = breptorch::from_blob(
                    arr.data<float>(), shape, breptorch::kFloat32).clone();
                loaded_count++;
            }
        }
        std::cout << "[FaceClassifier] 加载 BRepNet 权重: " << loaded_count << std::endl;

        modelLoaded_ = true;
        std::cout << "[FaceClassifier] 模型加载成功" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "[FaceClassifier] 模型加载失败: " << e.what() << std::endl;
        modelLoaded_ = false;
        return false;
    }
}

std::vector<int> FaceClassifier::predict(const std::string& stepFilePath) {
    std::vector<int> predictions;

    if (!modelLoaded_) {
        std::cerr << "[FaceClassifier] 模型未加载，无法预测" << std::endl;
        return predictions;
    }

    try {
        // 1. 使用BRepPipeline处理STEP文件
        BRepPipeline pipeline;
        if (!pipeline.process(stepFilePath)) {
            std::cerr << "[FaceClassifier] 处理STEP文件失败" << std::endl;
            return predictions;
        }

        int num_faces = pipeline.unique_faces.Extent();
        std::cout << "[FaceClassifier] 处理文件: " << stepFilePath
                  << " (" << num_faces << " 个面)" << std::endl;

        // 2. 转换数据格式
        auto coedges = BRepNetAdapter::extract_coedges(pipeline, model_->surf_enc, model_->curve_enc, model_->surf_enc2);
        auto faces = BRepNetAdapter::extract_faces(pipeline);

        // 3. 前向推理
        breptorch::Tensor logits = model_->forward(coedges, faces);

        // 4. 计算Softmax并获取预测类别
        breptorch::Tensor probs = breptorch::softmax(logits, 1);
        int num_classes = logits.size(1);

        for (int f = 0; f < num_faces; ++f) {
            float max_prob = -1e9f;
            int pred_class = 0;

            for (int c = 0; c < num_classes; ++c) {
                float p = probs.at({f, c});
                if (p > max_prob) {
                    max_prob = p;
                    pred_class = c;
                }
            }

            predictions.push_back(pred_class);
        }

        std::cout << "[FaceClassifier] 预测完成: " << predictions.size() << " 个面" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[FaceClassifier] 预测失败: " << e.what() << std::endl;
        predictions.clear();
    }

    return predictions;
}
