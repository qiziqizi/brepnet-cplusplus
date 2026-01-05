#pragma once
#include <string>
#include <torch/torch.h>
#include "BRepNet.h"       
#include "BRepPipeline.h"  // 需要用到 pipeline里的数据

class InferenceEngine {
public:
    // 构造函数：可以在这里传入网络参数
    InferenceEngine(int f_in, int e_in, int c_in, int emb_dim);

    // 加载权重
    void load_weights(const std::string& weights_path);

    // 执行推理
    // 输入：处理好的 pipeline 对象
    // 输出：Logits 张量
    torch::Tensor predict(BRepPipeline& pipeline);

private:
    // 将网络作为成员变量持有
    // 使用 std::shared_ptr 管理生命周期，或者是直接持有对象
    std::shared_ptr<BRepNetImpl> net;
};
