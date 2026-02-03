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

// --- Helper Math Functions ---

// Set row 0 to 0 (padding)
inline void check_indices(const std::string& name, Tensor values, Tensor indices) {
    int64_t max_idx = indices.max().item<int64_t>();
    int64_t size = values.size(0);
    if (max_idx >= size) {
        std::cerr << "\n=========================================" << std::endl;
        std::cerr << "ERROR: Index out of bounds (" << name << ")" << std::endl;
        std::cerr << "   Tensor size: " << size << " (valid range: 0 ~ " << size - 1 << ")" << std::endl;
        std::cerr << "   Max index: " << max_idx << std::endl;
        std::cerr << "=========================================\n" << std::endl;
        // Throw exception on first occurrence to debug
        throw std::runtime_error("Index out of bounds in " + name);
    }
}

// Modified build_matrix_Psi for local mode
inline Tensor build_matrix_Psi(Tensor Xf, Tensor Xe, Tensor Xc,
    Tensor Kf, Tensor Ke, Tensor Kc) {
    // Local mode:
    // Xf is already [N_c, 128] (LeftFace + RightFace)
    // Xe, Xc are also aligned to Coedge
    // Only Edge and Coedge need indexing (similar to Python's build_matrix_Psi_local)
    // Python equivalent: Pet = Xe[Ke], Pct = Xc[Kc], Pt = Xf (direct assignment)

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

// 2. Edge max pooling (with auto padding)
inline Tensor find_max_feature_vectors_for_each_edge(Tensor Ze, Tensor Ce) {
    int64_t max_req = Ce.max().item<int64_t>();

    if (Ze.size(0) <= max_req) {
        int64_t diff = max_req - Ze.size(0) + 1;
        auto pad = breptorch::full({ diff, Ze.size(1) }, -1e9, Ze.options());
        Ze = breptorch::cat({ Ze, pad }, 0);
    }


    // Set row 0 to 0 (padding)
    //if (Ze.size(0) > 0) Ze.index_put_({ 0 }, -1e9);
    if (Ze.size(0) > 0) Ze.index_put_({ 0 }, 0);
    Tensor Zet = Ze.index({ Ce });
    Tensor He_raw = std::get<0>(breptorch::max(Zet, 1));

    Tensor padding = breptorch::zeros({ 1, He_raw.size(1) }, Ze.options());
    return breptorch::cat({ padding, He_raw }, 0);
}
 //3. Face max pooling (with auto padding)
inline Tensor find_max_feature_vectors_for_each_face(Tensor Zf, Tensor Cf, const std::vector<Tensor>& Csf) {
    int64_t num_filters = Zf.size(1);

    // 1. Find max required index from Cf and Csf
    int64_t max_req = Cf.max().item<int64_t>();
    if (!Csf.empty()) {
        for (auto& c : Csf) max_req = std::max(max_req, c.max().item<int64_t>());
    }

    // 2. Edge max pooling (with auto padding)
    if (Zf.size(0) <= max_req) {
        // Set row 0 to 0 (padding)
        int64_t diff = max_req - Zf.size(0) + 1;
        // �ø��������Ӱ�� Max Pooling
        //auto pad = breptorch::full({ diff, num_filters }, -1e9, Zf.options()); 
        auto pad = breptorch::full({ diff, num_filters }, 0, Zf.options());
        Zf = breptorch::cat({ Zf, pad }, 0);
    }

    // Set row 0 to 0 (padding)
    // Ensure Zf has at least 1 row
    if (Zf.size(0) > 0) {
        //Zf.index_put_({ 0 }, -1e9);
        Zf.index_put_({ 0 }, 0);
    }


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

    // Set row 0 to 0 (padding)
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

        // Set row 0 to 0 (padding)
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
            // Set row 0 to 0 (padding)
            mlp->push_back("linear_1", Linear(LinearOptions(hidden, out_size).bias(true)));
            mlp->push_back("relu_1", ReLU());
        }
    }

    Tensor forward(Tensor x) { 
        return mlp->forward(x); 
       
    }
};
TORCH_MODULE(BRepNetMLP)


// 2. Edge max pooling (with auto padding)
struct BRepNetLayerImpl : Module {
    BRepNetMLP mlp{ nullptr };
    int out_size;


    BRepNetLayerImpl(int in_s, int out_s) : out_size(out_s) {
        // ���ά������3������ΪҪ�зֳ� Face, Edge, Coedge ������
        mlp = register_module("mlp", BRepNetMLP(in_s, 3 * out_s, 3 * out_s, false));
    }

    std::tuple<Tensor, Tensor, Tensor> forward(Tensor Xf, Tensor Xe, Tensor Xc, Tensor Kf, Tensor Ke, Tensor Kc, Tensor Ce, Tensor Cf, const std::vector<Tensor>& Csf) {
        Tensor Psi = build_matrix_Psi(Xf, Xe, Xc, Kf, Ke, Kc);

        Tensor Z = mlp->forward(Psi);

        // Slice into coedge, edge, face features
        Tensor Zc = Z.slice(1, 0, out_size);
        Tensor Ze = Z.slice(1, out_size, 2 * out_size);
        Tensor Zf = Z.slice(1, 2 * out_size, 3 * out_size);

        Tensor He, Hf;
        He = find_max_feature_vectors_for_each_edge(Ze, Ce);
        Hf = find_max_feature_vectors_for_each_face(Zf, Cf, Csf);

        // Zc doesn't need pooling, pass through directly
        return std::make_tuple(Hf, He, Zc);
    }
};
TORCH_MODULE(BRepNetLayer)

// 3. Main BRepNet Implementation
struct BRepNetImpl : Module {
    bool use_uvnet = false;
    UVNetSurfaceEncoder surf_enc{ nullptr };
    UVNetCurveEncoder curve_enc{ nullptr };
    SequentialPtr layers{ nullptr };
    BRepNetFaceOutputLayer output_layer{ nullptr };
    LinearPtr classification_layer{ nullptr };

    BRepNetImpl(int kernel_size_face, int kernel_size_edge, int num_layers, int num_classes) {
        // Initialize layers
        layers = register_module("layers", Sequential());

        // Add layers based on configuration
        // Layer 0
        layers->push_back("layer_0", BRepNetLayer(kernel_size_face * 128 + kernel_size_edge * 64 + kernel_size_edge * 64, 120));

        // Middle layers
        for (int i = 1; i < num_layers; ++i) {
            layers->push_back("layer_" + std::to_string(i), BRepNetLayer(120 * 3, 120));
        }

        // Output layer
        output_layer = register_module("output_layer", BRepNetFaceOutputLayer(kernel_size_face * 120 + kernel_size_edge * 120 + kernel_size_edge * 120, 120));

        // Classification layer
        classification_layer = register_module("classification_layer", Linear(LinearOptions(120, num_classes).bias(false)));
    }

    // Forward function
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

            // --- Helper Math Functions ---
            if (FaceGridsLocal.defined()) {
                // Set row 0 to 0 (padding)
                // Ŀ��: ��� [N_c, 128] (�� 64*2)

                int64_t Nc = FaceGridsLocal.size(0);

                // Set row 0 to 0 (padding)
                Tensor input_f = FaceGridsLocal.view({ Nc * 2, 9, 10, 10 });

                // 2. Edge max pooling (with auto padding)
                Tensor out_f = surf_enc->forward(input_f);

                // 3. Reshape �� [N_c, 128] (Concatenate Left & Right)
                // view(Nc, -1) ���Զ��������ά (2, 64) չƽΪ 128
                Tensor uv_feat_f = out_f.view({ Nc, -1 });

                // 4. Update Xf with UV features (direct replacement)
                // Python equivalent: Pt = Xf (Xf from surface_encoder)
                Xf = uv_feat_f;
            }
            if (EdgeGridsLocal.defined()) {
                // Set row 0 to 0 (padding)
                // ���� -> [N_c, 64]
                Tensor uv_feat_e = curve_enc->forward(EdgeGridsLocal);

                // ���� Xe
                Xe = uv_feat_e;
            }
            // --- Helper Math Functions ---
            if (CoedgeGridsLocal.defined()) {
                // Set row 0 to 0 (padding)
                Tensor uv_feat_c = curve_enc->forward(CoedgeGridsLocal);

                // Python ����: Xe = curve_encoder(Ge), Xc = curve_encoder(Gc)
                // �� Local ģʽ�£�Ge �� Gc ͨ�����ǻ��� Coedge Grid �任����
                // ����򻯣����� Xe �� Xc ������һ������
                Xc = uv_feat_c;
            }
        }


        // ----------------------------------------------------------------
        // 2. Edge max pooling (with auto padding)
        // ----------------------------------------------------------------

        auto modules = layers->children();
        auto layer0_base = modules[0];
        auto layer0 = std::dynamic_pointer_cast<BRepNetLayerImpl>(layer0_base);

        // Set row 0 to 0 (padding)
        auto res = layer0->forward(Xf, Xe, Xc, Kf, Ke, Kc, Ce, Cf, Csf);

        Tensor next_Xf = std::get<0>(res);
        Tensor next_Xe = std::get<1>(res);
        Tensor next_Xc = std::get<2>(res);

        // Set row 0 to 0 (padding)
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
        // 2. Edge max pooling (with auto padding)
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
