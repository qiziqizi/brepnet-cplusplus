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
#include "VerificationLogger.h" 

using Tensor = breptorch::Tensor;
using namespace breptorch;
using namespace breptorch::nn;

// --- ������ѧ���� ---

// 0. ��ȫ��麯�� (����)
inline void check_indices(const std::string& name, Tensor values, Tensor indices) {
    int64_t max_idx = indices.max().item<int64_t>();
    int64_t size = values.size(0);
    if (max_idx >= size) {
        std::cerr << "\n=========================================" << std::endl;
        std::cerr << "��������: ����Խ���� (" << name << ")" << std::endl;
        std::cerr << "   ���������� (Size): " << size << " (��Ч���� 0 ~ " << size - 1 << ")" << std::endl;
        std::cerr << "   ��������� (Index): " << max_idx << std::endl;
        std::cerr << "=========================================\n" << std::endl;
        // ��һ�л��ó�����ͣ�����㿴������Ĵ���
        throw std::runtime_error("Index out of bounds in " + name);
    }
}

// �޸� build_matrix_PsiΪlocal
inline Tensor build_matrix_Psi(Tensor Xf, Tensor Xe, Tensor Xc,
    Tensor Kf, Tensor Ke, Tensor Kc) {
    // Local ģʽ��:
    // Xf �Ѿ��� [N_c, 128] (LeftFace + RightFace)
    // Xe, Xc Ҳ�Ƕ��뵽 Coedge ��
    // ֻ�� Edge �� Coedge ���ھ���Ҫ��� (���� Python ���� build_matrix_Psi_local)
    // Python ����: Pet = Xe[Ke], Pct = Xc[Kc], Pt = Xf (ֱ�Ӹ�ֵ)

    Tensor Pet = Xe.index({ Ke });
    Tensor Pct = Xc.index({ Kc });

    // Xf ����Ҫ index select!
    Tensor Pt = Xf;

    Tensor Pe = breptorch::flatten(Pet, 1);
    Tensor Pc = breptorch::flatten(Pct, 1);

    // Verification: Log Psi components
    Verification::LogOnce("Psi_Pt_Shape", Pt.sizes());
    Verification::LogOnce("Psi_Pt_Range", std::string("Min: ") + std::to_string(Pt.min().item<float>()) + " Max: " + std::to_string(Pt.max().item<float>()));
    Verification::LogOnce("Psi_Pe_Shape", Pe.sizes());
    Verification::LogOnce("Psi_Pe_Range", std::string("Min: ") + std::to_string(Pe.min().item<float>()) + " Max: " + std::to_string(Pe.max().item<float>()));
    Verification::LogOnce("Psi_Pc_Shape", Pc.sizes());
    Verification::LogOnce("Psi_Pc_Range", std::string("Min: ") + std::to_string(Pc.min().item<float>()) + " Max: " + std::to_string(Pc.max().item<float>()));

    return breptorch::cat({ Pt, Pe, Pc }, 1);
}

// 2. �����������ػ� (���Զ� Padding ����)
inline Tensor find_max_feature_vectors_for_each_edge(Tensor Ze, Tensor Ce) {
    int64_t max_req = Ce.max().item<int64_t>();

    if (Ze.size(0) <= max_req) {
        int64_t diff = max_req - Ze.size(0) + 1;
        auto pad = breptorch::full({ diff, Ze.size(1) }, -1e9, Ze.options());
        Ze = breptorch::cat({ Ze, pad }, 0);
    }

    // check_indices("Pooling Edge (Ze/Ce)", Ze, Ce);

    // ���õ�0��Ϊ������
    //if (Ze.size(0) > 0) Ze.index_put_({ 0 }, -1e9);
    if (Ze.size(0) > 0) Ze.index_put_({ 0 }, 0);
    Tensor Zet = Ze.index({ Ce });
    Tensor He_raw = std::get<0>(breptorch::max(Zet, 1));

    Tensor padding = breptorch::zeros({ 1, He_raw.size(1) }, Ze.options());
    return breptorch::cat({ padding, He_raw }, 0);
}
 //3. �����������ػ� (���Զ� Padding ����)
inline Tensor find_max_feature_vectors_for_each_face(Tensor Zf, Tensor Cf, const std::vector<Tensor>& Csf) {
    int64_t num_filters = Zf.size(1);

    // 1. ��� Cf �������������
    int64_t max_req = Cf.max().item<int64_t>();
    if (!Csf.empty()) {
        for (auto& c : Csf) max_req = std::max(max_req, c.max().item<int64_t>());
    }

    // 2. ��� Zf �������Զ��� Padding
    if (Zf.size(0) <= max_req) {
        // std::cout << "  [Pooling] ��ȫ Zf: " << Zf.size(0) << " -> Req: " << max_req << std::endl;
        int64_t diff = max_req - Zf.size(0) + 1;
        // �ø��������Ӱ�� Max Pooling
        //auto pad = breptorch::full({ diff, num_filters }, -1e9, Zf.options()); 
        auto pad = breptorch::full({ diff, num_filters }, 0, Zf.options());
        Zf = breptorch::cat({ Zf, pad }, 0);
    }

    // 3. ����� index_put ��Ϊ���õ� 0 �� (Padding) ������ Max Pooling
    // ȷ�� Zf ������ 1 ��
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

    // ��һ���������һ���õ� Padding ������ 0
    Tensor padding_out = breptorch::zeros({ 1, Hf_final.size(1) }, Zf.options());
    return breptorch::cat({ padding_out, Hf_final }, 0);
}

 
/*--- ����ģ�鶨�� ---*/

namespace breptorch {
namespace nn {

//1. ���� MLP ģ��
struct BRepNetMLPImpl : Module {
    SequentialPtr mlp;

    BRepNetMLPImpl(int in_size, int hidden, int out_size, bool is_final)
        : mlp(register_module("mlp", Sequential())) {

        // MLP ��һ�� (Layer 0 of MLP)
        // �κβ�ĵ�һ������ Bias �� ReLU
        mlp->push_back("linear_0", Linear(LinearOptions(in_size, hidden).bias(true)));
        mlp->push_back("relu_0", ReLU());

        // MLP �ڶ��� (Layer 1 of MLP)
        if (is_final) { 
            // [Output Layer] ����Python: use_bias=False, use_relu=False
            mlp->push_back("linear_1", Linear(LinearOptions(hidden, out_size).bias(false)));
            // mlp->push_back("relu_1", torch::nn::ReLU());
        }
        else {
            // [(Layer 0)] ����Python: use_bias=True, ���� ReLU()
            mlp->push_back("linear_1", Linear(LinearOptions(hidden, out_size).bias(true)));
            mlp->push_back("relu_1", ReLU());
        }
    }

    Tensor forward(Tensor x) { 
        return mlp->forward(x); 
       
    }
};
TORCH_MODULE(BRepNetMLP)


// 2. ͨ�ò� (BRepNetLayer)
struct BRepNetLayerImpl : Module {
    BRepNetMLP mlp{ nullptr };
    int out_size;


    BRepNetLayerImpl(int in_s, int out_s) : out_size(out_s) {
        // ���ά������3������ΪҪ�зֳ� Face, Edge, Coedge ������
        mlp = register_module("mlp", BRepNetMLP(in_s, 3 * out_s, 3 * out_s, false));
    }

    std::tuple<Tensor, Tensor, Tensor> forward(Tensor Xf, Tensor Xe, Tensor Xc, Tensor Kf, Tensor Ke, Tensor Kc, Tensor Ce, Tensor Cf, const std::vector<Tensor>& Csf) {
        Tensor Psi = build_matrix_Psi(Xf, Xe, Xc, Kf, Ke, Kc);


        // 1. ������� Xf (Layer 0 Input)
        Tensor Z = mlp->forward(Psi);
        classification_layer = register_module("classification_layer", Linear(LinearOptions(hidden_dim, num_classes)));

        if (!grid.defined()) return target_feat;

        int64_t target_rows = target_feat.size(0);
        int64_t grid_rows = grid.size(0);
        Tensor input_grid = grid;

        // 1. �Զ� Padding ������
        if (grid_rows == target_rows - 1) {
            // Grid ��һ�� (Raw Data) -> ͷ���� 0
            // ��ȡ Grid ��ά��: [1, C, H, W] or [1, C, L]
            std::vector<int64_t> pad_shape = grid.sizes().vec();
            pad_shape[0] = 1;
            auto padding = breptorch::zeros(pad_shape, grid.options());
            input_grid = breptorch::cat({ padding, grid }, 0);
        }
        else if (grid_rows != target_rows) {
            std::cerr << "[Error] " << name << " ά�����ز�ƥ��! Target: " << target_rows << ", Grid: " << grid_rows << std::endl;
            throw std::runtime_error("Grid dimension mismatch in " + name);
        }


        // 3. ƴ��: [N, �ֹ�Dim] + [N, 64] -> [N, �ֹ�Dim+64]
        return breptorch::cat({ target_feat, grid_emb }, 1);
    }

    // Forward ����
    Tensor forward(Tensor Xf, Tensor Xe, Tensor Xc,
        Tensor Kf, Tensor Ke, Tensor Kc,
        Tensor Ce, Tensor Cf, const std::vector<Tensor>& Csf,
        Tensor FaceGridsLocal = Tensor(),
        Tensor EdgeGridsLocal = Tensor(),
        Tensor CoedgeGridsLocal = Tensor()) {

        // ----------------------------------------------------------------
        // 1. UV-Net ������ȡ (Local ģʽ)
        // ----------------------------------------------------------------
        if (use_uvnet) {

            // --- A. Face ���� (���) ---
            if (FaceGridsLocal.defined()) {
                // ����: [N_c, 2, 9, 10, 10]
                // Ŀ��: ��� [N_c, 128] (�� 64*2)

                int64_t Nc = FaceGridsLocal.size(0);

                // 1. Reshape �� [N_c * 2, 9, 10, 10] �Ա���������
                Tensor input_f = FaceGridsLocal.view({ Nc * 2, 9, 10, 10 });

                // 2. ���� -> [N_c * 2, 64]
                Tensor out_f = surf_enc->forward(input_f);

                // 3. Reshape �� [N_c, 128] (Concatenate Left & Right)
                // view(Nc, -1) ���Զ��������ά (2, 64) չƽΪ 128
                Tensor uv_feat_f = out_f.view({ Nc, -1 });

                // 4. ����� Xf ������ƴ�ӣ�����ֱ���滻
                // ���� Python: Pt = Xf (Xf ���� surface_encoder)
            if (EdgeGridsLocal.defined()) {
                // EdgeGridsLocal: [N_c, 13, 10]
                // ���� -> [N_c, 64]
                Tensor uv_feat_e = curve_enc->forward(EdgeGridsLocal);

                // ���� Xe
                Xe = uv_feat_e;
            }
            // --- B. Coedge ���� ---
            if (CoedgeGridsLocal.defined()) {
                // CoedgeGridsLocal: [N_c, 13, 10]
                Tensor uv_feat_c = curve_enc->forward(CoedgeGridsLocal);

                // Python ����: Xe = curve_encoder(Ge), Xc = curve_encoder(Gc)
                // �� Local ģʽ�£�Ge �� Gc ͨ�����ǻ��� Coedge Grid �任����
                // ����򻯣����� Xe �� Xc ������һ������
                Xc = uv_feat_c;
            }
        }


        // ----------------------------------------------------------------
        // 2. GNN ���� (���� build_matrix_Psi ���ˣ�����ֱ�Ӵ�)
        // ----------------------------------------------------------------

        auto modules = layers->children();
        auto layer0_base = modules[0];
        auto layer0 = std::dynamic_pointer_cast<BRepNetLayerImpl>(layer0_base);

        // ���� Layer 0
        auto res = layer0->forward(Xf, Xe, Xc, Kf, Ke, Kc, Ce, Cf, Csf);

        Tensor next_Xf = std::get<0>(res);
        Tensor next_Xe = std::get<1>(res);
        Tensor next_Xc = std::get<2>(res);

        // ��ϴ Padding (Mean Pooling �� 0 ���У���Ȼ������ϴ)
        if (next_Xf.size(0) > 0) next_Xf[0].zero_();
        if (next_Xe.size(0) > 0) next_Xe[0].zero_();
        if (next_Xc.size(0) > 0) next_Xc[0].zero_();

        // ���� Output Layer
        Tensor embeds = output_layer->forward(next_Xf, next_Xe, next_Xc, Kf, Ke, Kc, Ce, Cf, Csf);

        return classification_layer->forward(embeds);
    }

    // ���ؼ������� UV-Net Ȩ�� (�Զ�����)
    void load_uvnet_weights(const std::string& npz_path) {
        std::cout << "[Debug] load_uvnet_weights start" << std::endl;
        cnpy::npz_t npz = cnpy::npz_load(npz_path);
        std::cout << "[Debug] npz loaded" << std::endl;
        std::map<std::string, Tensor> surf_dict;
        std::map<std::string, Tensor> curve_dict;

        bool found_any = false;

        for (auto& item : npz) {
            std::string name = item.first;

            // ɸѡ Surface Encoder Ȩ��
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
            // ɸѡ Curve Encoder Ȩ��
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

    // ���� MLP Ȩ�� (���ֲ���)
    void load_mlp_weights(const std::string& path) {
        cnpy::npz_t npz = cnpy::npz_load(path);
        breptorch::NoGradGuard no_grad;

        for (auto& p : this->named_parameters()) {
            std::string name = p.first;
            // ���� UV-Net �Ĳ�������Ϊ����ͨ�� load_uvnet_weights ��������
            if (name.find("surface_encoder") != std::string::npos || name.find("curve_encoder") != std::string::npos) {
                continue;
            }

            if (npz.count(name)) {
                cnpy::NpyArray arr = npz[name];
                auto t = breptorch::from_blob(arr.data<float>(), p.second->sizes(), breptorch::kFloat32).clone();
                p.second->copy_(t);
                //test_step.cpp��ʱȡ��
                //std::cout << "Loaded MLP Weight: " << name << std::endl;
            }
        }
        // 2. ���ؼ����������� Buffers(Running Mean / Var)
        for (auto& p : this->named_buffers()) {
            std::string name = p.first;
            // ���� UV-Net
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
                // ��Щ buffer ���� num_batches_tracked ����Ҫ���أ����Ժ��Ծ���
                if (name.find("num_batches_tracked") == std::string::npos)
                    std::cerr << "Missing Buffer: " << name << std::endl;
            }
        }

    }
};
TORCH_MODULE(BRepNet)

} // namespace nn
} // namespace breptorch
