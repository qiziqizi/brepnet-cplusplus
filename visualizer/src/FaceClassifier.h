#ifndef FACECLASSIFIER_H
#define FACECLASSIFIER_H

#include <string>
#include <vector>
#include <memory>

#include "PrecisionUtils.h"

// 前向声明，避免包含复杂的BRepNet头文件
class BRepNetImpl;

/**
 * 面分类器：封装BRepNet推理逻辑
 * 功能：
 * 1. 加载模型权重
 * 2. 对单个STEP文件进行预测
 * 3. 返回每个面的类别ID
 */
class FaceClassifier {
public:
    FaceClassifier();
    ~FaceClassifier();

    // 加载模型权重
    // precision: 权重文件精度（FP32 / FP16 / BF16）
    bool loadModel(const std::string& weightsPath,
                   breptorch::WeightPrecision precision = breptorch::WeightPrecision::FP32);

    // 预测单个STEP文件
    // 返回：每个面的类别ID（索引i对应第i个面）
    std::vector<int> predict(const std::string& stepFilePath);

    // 检查模型是否已加载
    bool isModelLoaded() const { return modelLoaded_; }

private:
    std::shared_ptr<BRepNetImpl> model_;
    bool modelLoaded_;
};

#endif // FACECLASSIFIER_H
