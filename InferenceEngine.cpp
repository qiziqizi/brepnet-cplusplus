#include "InferenceEngine.h"
#include <iostream>
namespace torch = breptorch;

InferenceEngine::InferenceEngine(int f_in, int e_in, int c_in, int emb_dim) {
    // 实例化网络，使用 make_shared
    net = std::make_shared<BRepNetImpl>(f_in, e_in, c_in, emb_dim);
    // 默认设置为评估模式
    net->eval();
}

void InferenceEngine::load_weights(const std::string& weights_path) {
    if (!net) throw std::runtime_error("Network not initialized!");

    std::cout << "[Inference] Loading weights from: " << weights_path << std::endl;
    net->load_mlp_weights(weights_path);
    net->load_uvnet_weights(weights_path);
    std::cout << "[Inference] Weights loaded successfully." << std::endl;
}

torch::Tensor InferenceEngine::predict(BRepPipeline& pipeline) {
    if (!net) throw std::runtime_error("Network not initialized!");

    // 使用 NoGradGuard 关闭梯度计算，节省内存和时间
    torch::NoGradGuard no_grad;

    // 执行 Forward
    // 直接从 pipeline 中取数据，保持 test.cpp 的整洁
    torch::Tensor logits = net->forward(
        pipeline.Xf, pipeline.Xe, pipeline.Xc,
        pipeline.Kf, pipeline.Ke, pipeline.Kc,
        pipeline.Ce, pipeline.Cf, pipeline.Csf,
        pipeline.FaceGridsLocal, pipeline.EdgeGridsLocal, pipeline.CoedgeGridsLocal
    );

    return logits;
}
