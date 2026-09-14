//=========================================================================================
// acis_eval.cpp
// T4 M2：ACIS 端到端 BRepNet 前向（OCCT 无关）
//
// 读入 acis_pipeline.exe 转储的 network 输入（FaceGridsLocal / CoedgeGridsLocal + 拓扑），
// 用与 main_export_features.cpp 相同的模型加载与 forward 逻辑，输出各面 logits 与预测类别。
// 仅依赖 OCCT 无关的 BRepTorch / UVNet / cnpy。
//
// 用法: acis_eval.exe <inputs.bin> <out.logits> [weights.npz] [--precision fp32|fp16|bf16]
//
// 说明（面序）：
//   ACIS 面序 = STEP 原始面序 = OCCT 基线 logits 顺序，因此直接按 ACIS 面序输出即可逐面对比。
//=========================================================================================

#include "DebugControl.h"
#include "BRepNet.h"
#include "UVNet.h"
#include "PrecisionUtils.h"
#include "cnpy.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <map>
#include <sstream>
#include <stdexcept>

// 模型输出类别数（与权重文件 classification_layer 维度一致）
static constexpr int kNumClasses = 27;

// 与 main_export_features.cpp 一致的模型加载
static std::shared_ptr<BRepNetImpl> load_model(const cnpy::npz_t& npz,
                                               breptorch::WeightPrecision precision) {
    auto model = std::make_shared<BRepNetImpl>(kNumClasses);

    std::map<std::string, breptorch::Tensor> surf_weights, curve_weights, surf2_weights;
    for (auto& item : npz) {
        auto arr = item.second;
        std::vector<int64_t> shape(arr.shape.begin(), arr.shape.end());
        std::vector<float> fp32_data = breptorch::convert_weights_to_fp32(
            arr.data<uint8_t>(), arr.num_vals, precision);
        breptorch::Tensor t = breptorch::from_blob(
            fp32_data.data(), shape, breptorch::kFloat32).clone();

        if (item.first.substr(0, 17) == "surface_encoder2.") {
            surf2_weights["surface_encoder." + item.first.substr(17)] = t;
        } else if (item.first.find("surface_encoder.") != std::string::npos) {
            surf_weights[item.first] = t;
        }
        if (item.first.find("curve_encoder.") != std::string::npos) {
            curve_weights[item.first] = t;
        }
    }
    model->surf_enc->load_weights(surf_weights);
    model->curve_enc->load_weights(curve_weights);
    model->surf_enc2->load_weights(surf2_weights);

    auto params = model->named_parameters();
    for (auto& item : npz) {
        std::string key = item.first;
        if (key.find("layers.0.mlp") != std::string::npos) {
            key = "layer_0.mlp" + key.substr(key.find(".mlp") + 4);
        } else if (key.find("layers.1.mlp") != std::string::npos) {
            key = "layer_1.mlp" + key.substr(key.find(".mlp") + 4);
        }
        if (params.find(key) != params.end()) {
            auto arr = item.second;
            std::vector<int64_t> shape(arr.shape.begin(), arr.shape.end());
            std::vector<float> fp32_data = breptorch::convert_weights_to_fp32(
                arr.data<uint8_t>(), arr.num_vals, precision);
            *params[key] = breptorch::from_blob(
                fp32_data.data(), shape, breptorch::kFloat32).clone();
        }
    }
    return model;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("Usage: acis_eval.exe <inputs.bin> <out.logits> [weights.npz] [--precision fp32|fp16|bf16]\n");
        return 1;
    }
    const char* inputs_bin = argv[1];
    const char* out_logits = argv[2];
    std::string weights_file = "inference_data/state_dict_v4.npz";
    breptorch::WeightPrecision precision = breptorch::WeightPrecision::FP32;

    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--precision" && i + 1 < argc) {
            precision = breptorch::parse_precision(argv[++i]);
        } else if (a == "--weights" && i + 1 < argc) {
            weights_file = argv[++i];
        }
    }

    setvbuf(stdout, NULL, _IONBF, 0);   // 无缓冲，便于定位崩溃点
    try {
    // ---------- 读取转储的 network 输入 ----------
    FILE* bf = fopen(inputs_bin, "rb");
    if (!bf) { printf("[eval] ERROR: cannot open %s\n", inputs_bin); return 1; }
    int32_t magic = 0; size_t Nf = 0, Nc = 0;
    if (fread(&magic, sizeof(int32_t), 1, bf) != 1 || magic != 0x54344231) {
        printf("[eval] ERROR: bad magic in %s\n", inputs_bin); fclose(bf); return 1;
    }
    { int32_t tmp = 0; fread(&tmp, sizeof(int32_t), 1, bf); Nf = (size_t)tmp;
      fread(&tmp, sizeof(int32_t), 1, bf); Nc = (size_t)tmp; }

    std::vector<int32_t> fidx(Nc), midx(Nc), mfidx(Nc);
    for (size_t c = 0; c < Nc; ++c) {
        fread(&fidx[c], sizeof(int32_t), 1, bf);
        fread(&midx[c], sizeof(int32_t), 1, bf);
        fread(&mfidx[c], sizeof(int32_t), 1, bf);
    }
    const size_t G = 400;   // 20 * 20
    std::vector<float> coedge_local(Nc * 9 * G);
    std::vector<float> face_local(Nc * 2 * 9 * G);
    size_t rd1 = fread(coedge_local.data(), sizeof(float), Nc * 9 * G, bf);
    size_t rd2 = fread(face_local.data(), sizeof(float), Nc * 2 * 9 * G, bf);
    fclose(bf);
    if (rd1 != Nc * 9 * G || rd2 != Nc * 2 * 9 * G) {
        printf("[eval] ERROR: short read in %s (rd1=%zu/%zu rd2=%zu/%zu)\n",
               inputs_bin, rd1, Nc * 9 * G, rd2, Nc * 2 * 9 * G);
        return 1;
    }
    printf("[eval] inputs: Nf=%zu Nc=%zu\n", Nf, Nc);

    // ---------- 加载模型 ----------
    {
        using namespace std;
        if (!ifstream(weights_file.c_str()).good()) {
            // 尝试常见候选路径
            std::vector<std::string> cands = {"bin/inference_data/state_dict_v4.npz",
                                               "inference_data/state_dict.npz",
                                               "bin/inference_data/state_dict.npz"};
            for (const auto& p : cands) if (ifstream(p.c_str()).good()) { weights_file = p; break; }
        }
    }
    printf("[eval] loading weights: %s\n", weights_file.c_str());
    cnpy::npz_t npz = cnpy::npz_load(weights_file);
    printf("[eval] NPZ keys: %d\n", (int)npz.size());
    auto model = load_model(npz, precision);

    // ---------- 构建张量 + UVNet 编码 ----------
    const int64_t NU = 20, NV = 20;
    Tensor FaceLocal = breptorch::from_blob(face_local.data(),
        {(int64_t)Nc, 2, 9, NU, NV}, breptorch::kFloat32).clone();
    Tensor CoedgeLocal = breptorch::from_blob(coedge_local.data(),
        {(int64_t)Nc, 9, NU, NV}, breptorch::kFloat32).clone();

    Tensor all_face = FaceLocal.clone().view({(int64_t)Nc * 2, 9, NU, NV});
    Tensor Xf = model->surf_enc->forward(all_face.clone());      // [Nc*2, 64]
    Tensor Xc = model->surf_enc2->forward(CoedgeLocal.clone());  // [Nc, 64]

    // ---------- 组装 CoedgeData / FaceData ----------
    std::vector<CoedgeData> coedges;
    coedges.reserve(Nc);
    for (size_t c = 0; c < Nc; ++c) {
        CoedgeData ce;
        ce.coedge_id = (int)c;
        ce.parent_face_id = fidx[c];
        ce.mate_face_id = mfidx[c];
        ce.edge_id = (int)c;           // forward 未用到 edge_id，但保持自洽
        for (int i = 0; i < 64; ++i) ce.parent_face_features.push_back((float)Xf.at({(int64_t)(2 * c), i}));
        for (int i = 0; i < 64; ++i) ce.mate_face_features.push_back((float)Xf.at({(int64_t)(2 * c + 1), i}));
        for (int i = 0; i < 64; ++i) ce.edge_features.push_back((float)Xc.at({(int64_t)c, i}));
        coedges.push_back(ce);
    }
    std::vector<FaceData> faces;
    faces.reserve(Nf);
    for (size_t f = 0; f < Nf; ++f) {
        FaceData fd;
        fd.face_id = (int)f;
        for (size_t c = 0; c < Nc; ++c) if ((size_t)fidx[c] == f) fd.coedge_ids.push_back((int)c);
        faces.push_back(fd);
    }

    // ---------- 前向 ----------
    Tensor logits = model->forward(coedges, faces);   // [Nf, 27]

    // ---------- 输出 logits（ACIS 面序 = STEE 原始面序 = OCCT 基线序）----------
    FILE* lf = fopen(out_logits, "w");
    if (!lf) { printf("[eval] ERROR: cannot open %s\n", out_logits); return 1; }
    for (size_t f = 0; f < Nf; ++f) {
        for (int c = 0; c < kNumClasses; ++c) {
            float v = (float)logits.at({(int64_t)f, c});
            if (f > 0 || c > 0) fprintf(lf, " ");
            fprintf(lf, "%.17e", (double)v);
        }
        fprintf(lf, "\n");
    }
    fclose(lf);
    printf("[eval] wrote logits: %s (Nf=%zu)\n", out_logits, Nf);
    return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[eval] EXCEPTION(std): %s\n", e.what());
        return 2;
    } catch (...) {
        fprintf(stderr, "[eval] EXCEPTION(unknown)\n");
        return 2;
    }
}