#pragma once
//#include <torch/torch.h>
#include "BRepTorch.h"
#include <vector>
#include <string>
#include <iostream>
#include <tuple>
#include <memory>
#include "cnpy.h" 
#include "UVNet.h" 

using Tensor = breptorch::Tensor;
using namespace breptorch;
using namespace breptorch::nn;

// --- 辅助数学函数 ---

// 0. 安全检查函数 (新增)
inline void check_indices(const std::string& name, Tensor values, Tensor indices) {
    int64_t max_idx = indices.max().item<int64_t>();
    int64_t size = values.size(0);
    if (max_idx >= size) {
        std::cerr << "\n=========================================" << std::endl;
        std::cerr << "致命错误: 索引越界检测 (" << name << ")" << std::endl;
        std::cerr << "   矩阵总行数 (Size): " << size << " (有效索引 0 ~ " << size - 1 << ")" << std::endl;
        std::cerr << "   请求的索引 (Index): " << max_idx << std::endl;
        std::cerr << "=========================================\n" << std::endl;
        // 这一行会让程序暂停，让你看到上面的错误
        throw std::runtime_error("Index out of bounds in " + name);
    }
}

// 修改 build_matrix_Psi为local
inline Tensor build_matrix_Psi(Tensor Xf, Tensor Xe, Tensor Xc,
    Tensor Kf, Tensor Ke, Tensor Kc) {
    // Local 模式下:
    // Xf 已经是 [N_c, 128] (LeftFace + RightFace)
    // Xe, Xc 也是对齐到 Coedge 的
    // 只有 Edge 和 Coedge 的邻居需要查表 (根据 Python 代码 build_matrix_Psi_local)
    // Python 代码: Pet = Xe[Ke], Pct = Xc[Kc], Pt = Xf (直接赋值)

    Tensor Pet = Xe.index({ Ke });
    Tensor Pct = Xc.index({ Kc });

    // Xf 不需要 index select!
    Tensor Pt = Xf;

    Tensor Pe = breptorch::flatten(Pet, 1);
    Tensor Pc = breptorch::flatten(Pct, 1);

    // --- 诊断：检查 Psi 组成部分 ---
    static bool psi_diag_printed = false;
    if (!psi_diag_printed) {
        std::cout << "\n[Diagnostic] build_matrix_Psi Components:" << std::endl;
        std::cout << "  Pt (Face) Shape: " << Pt.sizes() << " Min: " << Pt.min().item<float>() << " Max: " << Pt.max().item<float>() << std::endl;
        std::cout << "  Pe (Edge) Shape: " << Pe.sizes() << " Min: " << Pe.min().item<float>() << " Max: " << Pe.max().item<float>() << std::endl;
        std::cout << "  Pc (Coedge) Shape: " << Pc.sizes() << " Min: " << Pc.min().item<float>() << " Max: " << Pc.max().item<float>() << std::endl;
        psi_diag_printed = true;
    }
    // -----------------------------

    return breptorch::cat({ Pt, Pe, Pc }, 1);
}

// 2. 边特征的最大池化 (带自动 Padding 修正)
inline Tensor find_max_feature_vectors_for_each_edge(Tensor Ze, Tensor Ce) {
    int64_t max_req = Ce.max().item<int64_t>();

    if (Ze.size(0) <= max_req) {
        int64_t diff = max_req - Ze.size(0) + 1;
        auto pad = breptorch::full({ diff, Ze.size(1) }, -1e9, Ze.options());
        Ze = breptorch::cat({ Ze, pad }, 0);
    }

    // check_indices("Pooling Edge (Ze/Ce)", Ze, Ce);

    // 设置第0行为负无穷
    //if (Ze.size(0) > 0) Ze.index_put_({ 0 }, -1e9);
    if (Ze.size(0) > 0) Ze.index_put_({ 0 }, 0);
    Tensor Zet = Ze.index({ Ce });
    Tensor He_raw = std::get<0>(breptorch::max(Zet, 1));

    Tensor padding = breptorch::zeros({ 1, He_raw.size(1) }, Ze.options());
    return breptorch::cat({ padding, He_raw }, 0);
}
 //3. 面特征的最大池化 (带自动 Padding 修正)
inline Tensor find_max_feature_vectors_for_each_face(Tensor Zf, Tensor Cf, const std::vector<Tensor>& Csf) {
    int64_t num_filters = Zf.size(1);

    // 1. 检查 Cf 的最大索引需求
    int64_t max_req = Cf.max().item<int64_t>();
    if (!Csf.empty()) {
        for (auto& c : Csf) max_req = std::max(max_req, c.max().item<int64_t>());
    }

    // 2. 如果 Zf 不够大，自动补 Padding
    if (Zf.size(0) <= max_req) {
        // std::cout << "  [Pooling] 补全 Zf: " << Zf.size(0) << " -> Req: " << max_req << std::endl;
        int64_t diff = max_req - Zf.size(0) + 1;
        // 用负无穷补，不影响 Max Pooling
        //auto pad = breptorch::full({ diff, num_filters }, -1e9, Zf.options()); 
        auto pad = breptorch::full({ diff, num_filters }, 0, Zf.options());
        Zf = breptorch::cat({ Zf, pad }, 0);
    }

    // 3. 这里的 index_put 是为了让第 0 行 (Padding) 不参与 Max Pooling
    // 确保 Zf 至少有 1 行
    if (Zf.size(0) > 0) {
        //Zf.index_put_({ 0 }, -1e9);
        Zf.index_put_({ 0 }, 0);
    }

    // check_indices("Pooling Face Small (Zf/Cf)", Zf, Cf);

    Tensor Zft = Zf.index({ Cf });
    Tensor Hf_small = std::get<0>(breptorch::max(Zft, 1));

    Tensor Hf_final;
    if (Csf.empty()) {
        Hf_final = Hf_small;
    }
    else {
        std::vector<Tensor> Hf_list;
        Hf_list.push_back(Hf_small);
        for (size_t i = 0; i < Csf.size(); ++i) {
            Tensor Zsingle = Zf.index({ Csf[i] });
            Tensor Hbig = std::get<0>(breptorch::max(Zsingle, 0));
            Hf_list.push_back(Hbig.reshape({ 1, num_filters }));
        }
        Hf_final = breptorch::cat(Hf_list, 0);
    }

    // 这一步输出给下一层用的 Padding 必须是 0
    Tensor padding_out = breptorch::zeros({ 1, Hf_final.size(1) }, Zf.options());
    return breptorch::cat({ padding_out, Hf_final }, 0);
}

// 新增：平均池化 (Mean Pooling)，但未调用
// Edge 平均池化
inline Tensor get_average_feature_vectors_for_each_edge(Tensor Ze, Tensor Ce) {
    // 1. 确保第 0 行是 0 (Padding)
    if (Ze.size(0) > 0) Ze[0].zero_();

    // 2. 查表 [N_e, 2, D]
    Tensor Zet = Ze.index({ Ce });

    // 3. 取平均
    Tensor He = breptorch::mean(Zet, 1); // dim=1

    // 4. 补回 Padding (保持格式一致)
    Tensor padding = breptorch::zeros({ 1, He.size(1) }, Ze.options());
    return breptorch::cat({ padding, He }, 0);
}
// Face 平均池化
inline Tensor get_average_feature_vectors_for_each_face(Tensor Zf, Tensor Cf, const std::vector<Tensor>& Csf) {
    int64_t num_filters = Zf.size(1);

    // 1. 确保第 0 行是 0 (Padding)
    // Mean Pooling 必须用 0 填充，不能用 -1e9
    if (Zf.size(0) > 0) Zf[0].zero_();

    // 2. 小面处理 [N_small, 64, D]
    Tensor Zft = Zf.index({ Cf });

    // 取平均
    Tensor Hf_small = breptorch::mean(Zft, 1);

    // 3. 大面处理
    Tensor Hf_final;
    if (Csf.empty()) {
        Hf_final = Hf_small;
    }
    else {
        std::vector<Tensor> Hf_list;
        Hf_list.push_back(Hf_small);
        for (const auto& indices : Csf) {
            Tensor Zsingle = Zf.index({ indices });
            // 大面直接对所有边取平均
            Tensor Hbig = breptorch::mean(Zsingle, 0);
            Hf_list.push_back(Hbig.reshape({ 1, num_filters }));
        }
        Hf_final = breptorch::cat(Hf_list, 0);
    }


    // 4. 补回 Padding
    Tensor padding_out = breptorch::zeros({ 1, Hf_final.size(1) }, Zf.options());
    return breptorch::cat({ padding_out, Hf_final }, 0);
}
 
/*--- 网络模块定义 ---*/

namespace breptorch {
namespace nn {

//1. 基础 MLP 模块
struct BRepNetMLPImpl : Module {
    SequentialPtr mlp;

    BRepNetMLPImpl(int in_size, int hidden, int out_size, bool is_final)
        : mlp(register_module("mlp", Sequential())) {

        // MLP 第一层 (Layer 0 of MLP)
        // 任何层的第一级都有 Bias 和 ReLU
        mlp->push_back("linear_0", Linear(LinearOptions(in_size, hidden).bias(true)));
        mlp->push_back("relu_0", ReLU());

        // MLP 第二层 (Layer 1 of MLP)
        if (is_final) { 
            // [Output Layer] 根据Python: use_bias=False, use_relu=False
            mlp->push_back("linear_1", Linear(LinearOptions(hidden, out_size).bias(false)));
            // mlp->push_back("relu_1", torch::nn::ReLU());
        }
        else {
            // [(Layer 0)] 根据Python: use_bias=True, 且有 ReLU()
            mlp->push_back("linear_1", Linear(LinearOptions(hidden, out_size).bias(true)));
            mlp->push_back("relu_1", ReLU());
        }
    }

    Tensor forward(Tensor x) { 
        return mlp->forward(x); 
       
    }
};
TORCH_MODULE(BRepNetMLP)


// 2. 通用层 (BRepNetLayer)
struct BRepNetLayerImpl : Module {
    BRepNetMLP mlp{ nullptr };
    int out_size;
    bool use_average_pooling = false;


    BRepNetLayerImpl(int in_s, int out_s) : out_size(out_s) {
        // 输出维度扩大3倍，因为要切分成 Face, Edge, Coedge 三部分
        mlp = register_module("mlp", BRepNetMLP(in_s, 3 * out_s, 3 * out_s, false));
    }

    std::tuple<Tensor, Tensor, Tensor> forward(Tensor Xf, Tensor Xe, Tensor Xc, Tensor Kf, Tensor Ke, Tensor Kc, Tensor Ce, Tensor Cf, const std::vector<Tensor>& Csf) {
        Tensor Psi = build_matrix_Psi(Xf, Xe, Xc, Kf, Ke, Kc);


        // 1. 诊断输入 Xf (Layer 0 Input)
        /*static bool printed_L0_in = false;
        if (!printed_L0_in) { // 只打印一次
            std::cout << "\n---------------- [C++ Layer 0 诊断] ----------------" << std::endl;
            // 打印第 1 行 (跳过 Padding)，前 5 列
            std::cout << "Cpp L0 Input Xf (Head): " << Xf.slice(0, 1, 2).slice(1, 0, 5) << std::endl;
            printed_L0_in = true;
        }*/
        // ========================================================
        // 拆解 MLP
        // ========================================================
        /*static bool debug_printed = false;
        if (!debug_printed) {
            std::cout << "\n---------------- [C++ Layer 0 深度诊断] ----------------" << std::endl;

            // 1. 获取 Linear 层
            // mlp 是一个 Sequential，第 0 个是 Linear，第 1 个是 ReLU
            auto linear = mlp->mlp->children()[0]; // dynamic cast not used in mock


            if (linear) {
                // 2. 运行 Linear
                Tensor z_lin = linear->forward(Psi);

                // 打印第 1 行 (跳过 Padding 0)，前 10 个
                // 注意：这里要对应 Python 的 [0]，因为 Python 无 Padding
                if (z_lin.size(0) > 1) {
                    std::cout << "Cpp L0 Linear (Head): " << z_lin[1].slice(0, 0, 10) << std::endl;
                }
                else {
                    std::cout << "Cpp L0 Linear (Head): " << z_lin[0].slice(0, 0, 10) << std::endl;
                }

                std::cout << "Cpp L0 Linear Mean:   " << z_lin.mean().item<float>() << std::endl;
                std::cout << "Cpp L0 Linear Max:    " << z_lin.max().item<float>() << std::endl;
            }
            else {
                std::cout << "? 无法获取 Linear 层！" << std::endl;
            }
            std::cout << "------------------------------------------------------\n" << std::endl;
            debug_printed = true;
        }*/
        
        Tensor Z = mlp->forward(Psi);
        
        // 2. 诊断 Psi
        /*static bool printed_L0_psi = false;
        if (!printed_L0_psi) {
            std::cout << "Cpp L0 Psi (Head):      " << Psi.slice(0, 1, 2).slice(1, 0, 5) << std::endl;
            // 假设 Face=128, Edge=64. Edge 大概在 128~133 附近
            std::cout << "Cpp L0 Psi (Middle/Edge): " << Psi.slice(0, 1, 2).slice(1, 128, 133) << std::endl;
            // 打印 Psi 的后半段 (Coedge 部分)，看看是否对齐
            // 假设总长 320，最后 5 位
            std::cout << "Cpp L0 Psi (Tail):      " << Psi.slice(0, 1, 2).slice(1, Psi.size(1) - 5, Psi.size(1)) << std::endl;
            printed_L0_psi = true;
        }*/

        // 切片操作 252维变成3*84维 
        Tensor Zc = Z.slice(1, 0, out_size);
        Tensor Ze = Z.slice(1, out_size, 2 * out_size);
        Tensor Zf = Z.slice(1, 2 * out_size, 3 * out_size);

        Tensor He, Hf;

        // 【分支选择】
        if (use_average_pooling) {
            He = get_average_feature_vectors_for_each_edge(Ze, Ce);
            Hf = get_average_feature_vectors_for_each_face(Zf, Cf, Csf);
        }
        else {
            He = find_max_feature_vectors_for_each_edge(Ze, Ce);
            Hf = find_max_feature_vectors_for_each_face(Zf, Cf, Csf);
        }

        // 3. 诊断输出 Hf
        /*static bool printed_L0_out = false;
        if (!printed_L0_out) {
            std::cout << "Cpp L0 Output Hf (Head):" << Hf.slice(0, 1, 2).slice(1, 0, 5) << std::endl;
            std::cout << "----------------------------------------------------" << std::endl;
            printed_L0_out = true;
        }*/


        // Zc 不需要池化，直接传下去
        return std::make_tuple(Hf, He, Zc);
    }
};
TORCH_MODULE(BRepNetLayer)

// 3. 输出层 
struct BRepNetFaceOutputLayerImpl : Module {
    BRepNetMLP mlp{ nullptr };

    BRepNetFaceOutputLayerImpl(int in_s, int out_s) {
        // final_layer = true
        mlp = register_module("mlp", BRepNetMLP(in_s, out_s, out_s, true));
    }

    Tensor forward(Tensor Xf, Tensor Xe, Tensor Xc, Tensor Kf, Tensor Ke, Tensor Kc, Tensor Ce, Tensor Cf, const std::vector<Tensor>& Csf) {

        // 2. 聚合面 边 共边特征 (Kernel Size = 2)
        Tensor Pft = Xf.index({ Kf }); // [N, 2, 64]
        Tensor Pet = Xe.index({ Ke }); // [N, 5, 64]
        Tensor Pct = Xc.index({ Kc });

        Tensor Pt = breptorch::flatten(Pft, 1); // [N, 128]
        Tensor Pe = breptorch::flatten(Pet, 1); // [N, 320]
        Tensor Pc = breptorch::flatten(Pct, 1); 
        // 5. 拼接 (包含 Coedge!)
        Tensor Psi = breptorch::cat({ Pt, Pe, Pc}, 1);

        Tensor Z = mlp->forward(Psi);

        // 【调试插入】打印 Z
        /*std::cout << "Cpp Final Z Shape: " << Z.sizes() << std::endl;
        std::cout << "Cpp Final Z [Head]:\n" << Z.slice(0, 1, 6).slice(1, 0, 5) << std::endl;
        std::cout << "--------------------------------------------------------------" << std::endl;*/

        // 6. 池化回Face
        Tensor embeds = find_max_feature_vectors_for_each_face(Z, Cf, Csf);
        // ========== 新增：打印Output Layer池化后的embeds ==========
        //std::cout << "[Output Layer] Pooled Embeds:\n" << embeds.slice(0, 1, 2).slice(1, 0, 5) << std::endl;
        //std::cout << "\n[关键验证] C++ OutputLayer池化后embeds:\n"<< embeds.slice(0, 0, 5).slice(1, 0, 5) << std::endl;
        return embeds;
    }
};
TORCH_MODULE(BRepNetFaceOutputLayer)

// 4. 主网络 (BRepNet) 
struct BRepNetImpl : Module {
    // GNN 核心层
    SequentialPtr layers{ nullptr };
    BRepNetFaceOutputLayer output_layer{ nullptr };
    LinearPtr classification_layer{ nullptr };

    // UV-Net 编码器模块
    UVNetSurfaceEncoder surf_enc{ nullptr }; // 用于面 (Face)
    UVNetCurveEncoder curve_enc{ nullptr };  // 用于边 (Edge) 和 共边 (Coedge)

    bool use_uvnet = false;

    // 构造函数
    BRepNetImpl(int input_dim, int hidden_dim, int kernel_size_sum, int num_classes) {
        // 1. Hidden Layers (Layers.0)
        // 注意：input_dim 必须等于 (手工特征 + Grid特征) 的总和
        layers = register_module("layers", Sequential());
        layers->push_back("0", BRepNetLayer(input_dim, hidden_dim));

        // 2. Output Layer
        output_layer = register_module("output_layer", BRepNetFaceOutputLayer(kernel_size_sum * hidden_dim, hidden_dim));

        // 3. Classification
        classification_layer = register_module("classification_layer", Linear(LinearOptions(hidden_dim, num_classes)));

        // 4. 初始化 UV-Net 模块
        surf_enc = register_module("surface_encoder", UVNetSurfaceEncoder(new UVNetSurfaceEncoderImpl()));
        curve_enc = register_module("curve_encoder", UVNetCurveEncoder(new UVNetCurveEncoderImpl()));
    }

    // 辅助函数：智能对齐并提取特征
    // 输入: Target(手工特征张量), Grid(网格张量), Encoder(使用的编码器), Name(用于日志)
    // 输出: 拼接后的新特征张量
    template<typename ModuleType>
    Tensor process_grid_feature(Tensor target_feat, Tensor grid, ModuleType& encoder, std::string name) {
        if (!grid.defined()) return target_feat;

        int64_t target_rows = target_feat.size(0);
        int64_t grid_rows = grid.size(0);
        Tensor input_grid = grid;

        // 1. 自动 Padding 对齐检查
        if (grid_rows == target_rows - 1) {
            // Grid 少一行 (Raw Data) -> 头部补 0
            // 获取 Grid 的维度: [1, C, H, W] or [1, C, L]
            std::vector<int64_t> pad_shape = grid.sizes().vec();
            pad_shape[0] = 1;
            auto padding = breptorch::zeros(pad_shape, grid.options());
            input_grid = breptorch::cat({ padding, grid }, 0);
        }
        else if (grid_rows != target_rows) {
            std::cerr << "[Error] " << name << " 维度严重不匹配! Target: " << target_rows << ", Grid: " << grid_rows << std::endl;
            throw std::runtime_error("Grid dimension mismatch in " + name);
        }

        // 2. 卷积提取特征
        // input_grid 包含了 Padding (第0行为0)，卷积后输出也是 0 (或 bias)，安全。
        Tensor grid_emb = encoder->forward(input_grid);

        // 3. 拼接: [N, 手工Dim] + [N, 64] -> [N, 手工Dim+64]
        return breptorch::cat({ target_feat, grid_emb }, 1);
    }

    // Forward 函数
    Tensor forward(Tensor Xf, Tensor Xe, Tensor Xc,
        Tensor Kf, Tensor Ke, Tensor Kc,
        Tensor Ce, Tensor Cf, const std::vector<Tensor>& Csf,
        Tensor FaceGridsLocal = Tensor(),
        Tensor EdgeGridsLocal = Tensor(),
        Tensor CoedgeGridsLocal = Tensor()) {

        // ----------------------------------------------------------------
        // 1. UV-Net 特征提取 (Local 模式)
        // ----------------------------------------------------------------
        if (use_uvnet) {

            // --- A. Face 处理 (最复杂) ---
            if (FaceGridsLocal.defined()) {
                // 输入: [N_c, 2, 9, 10, 10]
                // 目标: 变成 [N_c, 128] (即 64*2)

                int64_t Nc = FaceGridsLocal.size(0);

                // 1. Reshape 成 [N_c * 2, 9, 10, 10] 以便批量卷积
                Tensor input_f = FaceGridsLocal.view({ Nc * 2, 9, 10, 10 });

                // 2. 卷积 -> [N_c * 2, 64]
                Tensor out_f = surf_enc->forward(input_f);

                // 3. Reshape 回 [N_c, 128] (Concatenate Left & Right)
                // view(Nc, -1) 会自动把最后两维 (2, 64) 展平为 128
                Tensor uv_feat_f = out_f.view({ Nc, -1 });

                // 4. 这里的 Xf 不再是拼接，而是直接替换
                // 根据 Python: Pt = Xf (Xf 来自 surface_encoder)
                Xf = uv_feat_f;
            }
            // --- B. Edge 处理 ---
            if (EdgeGridsLocal.defined()) {
                // EdgeGridsLocal: [N_c, 13, 10]
                // 卷积 -> [N_c, 64]
                Tensor uv_feat_e = curve_enc->forward(EdgeGridsLocal);

                // 覆盖 Xe
                Xe = uv_feat_e;
            }
            // --- B. Coedge 处理 ---
            if (CoedgeGridsLocal.defined()) {
                // CoedgeGridsLocal: [N_c, 13, 10]
                Tensor uv_feat_c = curve_enc->forward(CoedgeGridsLocal);

                // Python 代码: Xe = curve_encoder(Ge), Xc = curve_encoder(Gc)
                // 在 Local 模式下，Ge 和 Gc 通常都是基于 Coedge Grid 变换来的
                // 这里简化：假设 Xe 和 Xc 都用这一套特征
                Xc = uv_feat_c;
            }
        }
        /*
        // ====================================================
        //  【诊断插入】检查特征是否存活
        // ====================================================
        std::cout << "\n---------------- [Feature Diagnosis] ----------------" << std::endl;
        std::cout << "Xf Max: " << Xf.max().item<float>() << " (Should > 0)" << std::endl;
        std::cout << "Xe Max: " << Xe.max().item<float>() << " (Should > 0)" << std::endl;
        std::cout << "Xc Max: " << Xc.max().item<float>() << " (Should > 0)" << std::endl;

        // 检查是否全 0
        if (std::abs(Xe.max().item<float>()) < 1e-6) std::cerr << "?? 警告: Xe (Edge) 特征全为 0！" << std::endl;
        if (std::abs(Xc.max().item<float>()) < 1e-6) std::cerr << "?? 警告: Xc (Coedge) 特征全为 0！" << std::endl;
        std::cout << "-----------------------------------------------------\n" << std::endl;*/


        // ----------------------------------------------------------------
        // 2. GNN 推理 (由于 build_matrix_Psi 改了，这里直接传)
        // ----------------------------------------------------------------

        auto modules = layers->children();
        auto layer0_base = modules[0];
        auto layer0 = std::dynamic_pointer_cast<BRepNetLayerImpl>(layer0_base);

        // 运行 Layer 0
        auto res = layer0->forward(Xf, Xe, Xc, Kf, Ke, Kc, Ce, Cf, Csf);

        Tensor next_Xf = std::get<0>(res);
        Tensor next_Xe = std::get<1>(res);
        Tensor next_Xc = std::get<2>(res);

        // 清洗 Padding (Mean Pooling 对 0 敏感，依然建议清洗)
        if (next_Xf.size(0) > 0) next_Xf[0].zero_();
        if (next_Xe.size(0) > 0) next_Xe[0].zero_();
        if (next_Xc.size(0) > 0) next_Xc[0].zero_();

        // 运行 Output Layer
        Tensor embeds = output_layer->forward(next_Xf, next_Xe, next_Xc, Kf, Ke, Kc, Ce, Cf, Csf);

        return classification_layer->forward(embeds);
    }

    // 【关键】加载 UV-Net 权重 (自动分流)
    void load_uvnet_weights(const std::string& npz_path) {
        std::cout << "[Debug] load_uvnet_weights start" << std::endl;
        cnpy::npz_t npz = cnpy::npz_load(npz_path);
        std::cout << "[Debug] npz loaded" << std::endl;
        std::map<std::string, Tensor> surf_dict;
        std::map<std::string, Tensor> curve_dict;

        bool found_any = false;

        for (auto& item : npz) {
            std::string name = item.first;

            // 筛选 Surface Encoder 权重
            if (name.find("surface_encoder") != std::string::npos) {
                // std::cout << "[Debug] Loading surface_encoder weight: " << name << std::endl;
                cnpy::NpyArray arr = item.second;
                std::vector<int64_t> shape; for (auto s : arr.shape) shape.push_back(s);
                // Check word size
                if (arr.word_size != 4) {
                    std::cerr << "[Warning] Skipping " << name << " due to word_size " << arr.word_size << std::endl;
                    continue;
                }
                surf_dict[name] = breptorch::from_blob(arr.data<float>(), shape, breptorch::kFloat32).clone();
                found_any = true;
            }
            // 筛选 Curve Encoder 权重
            else if (name.find("curve_encoder") != std::string::npos) {
                // std::cout << "[Debug] Loading curve_encoder weight: " << name << std::endl;
                cnpy::NpyArray arr = item.second;
                std::vector<int64_t> shape; for (auto s : arr.shape) shape.push_back(s);
                if (arr.word_size != 4) {
                    std::cerr << "[Warning] Skipping " << name << " due to word_size " << arr.word_size << std::endl;
                    continue;
                }
                curve_dict[name] = breptorch::from_blob(arr.data<float>(), shape, breptorch::kFloat32).clone();
                found_any = true;
            }
        }

        if (found_any) {
            std::cout << "[Debug] Loading weights into surf_enc..." << std::endl;
            if (surf_enc) surf_enc->load_weights(surf_dict);
            else std::cerr << "[Error] surf_enc is null!" << std::endl;
            
            std::cout << "[Debug] Loading weights into curve_enc..." << std::endl;
            if (curve_enc) curve_enc->load_weights(curve_dict);
            else std::cerr << "[Error] curve_enc is null!" << std::endl;

            use_uvnet = true;
            std::cout << " UV-Net Weights Loaded (Surface & Curve)!" << std::endl;
        }
        else {
            std::cerr << " Warning: Called load_uvnet_weights but no keys found in npz!" << std::endl;
        }
    }

    // 加载 MLP 权重 (保持不变)
    void load_mlp_weights(const std::string& path) {
        cnpy::npz_t npz = cnpy::npz_load(path);
        breptorch::NoGradGuard no_grad;

        for (auto& p : this->named_parameters()) {
            std::string name = p.first;
            // 跳过 UV-Net 的参数，因为它们通过 load_uvnet_weights 单独加载
            if (name.find("surface_encoder") != std::string::npos || name.find("curve_encoder") != std::string::npos) {
                continue;
            }

            if (npz.count(name)) {
                cnpy::NpyArray arr = npz[name];
                auto t = breptorch::from_blob(arr.data<float>(), p.second->sizes(), breptorch::kFloat32).clone();
                p.second->copy_(t);
                //test_step.cpp暂时取消
                //std::cout << "Loaded MLP Weight: " << name << std::endl;
            }
        }
        // 2. 【关键新增】加载 Buffers(Running Mean / Var)
        for (auto& p : this->named_buffers()) {
            std::string name = p.first;
            // 跳过 UV-Net
            if (name.find("surface_encoder") != std::string::npos || name.find("curve_encoder") != std::string::npos) {
                continue;
            }

            if (npz.count(name)) {
                cnpy::NpyArray arr = npz[name];
                auto t = breptorch::from_blob(arr.data<float>(), p.second->sizes(), breptorch::kFloat32).clone();
                p.second->copy_(t);
                std::cout << "Loaded Buffer: " << name << std::endl;
            }
            else {
                // 有些 buffer 比如 num_batches_tracked 不需要加载，可以忽略警告
                if (name.find("num_batches_tracked") == std::string::npos)
                    std::cerr << "Missing Buffer: " << name << std::endl;
            }
        }

    }
};
TORCH_MODULE(BRepNet)

} // namespace nn
} // namespace breptorch
