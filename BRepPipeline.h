#pragma once

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

// LibTorch
//#include <torch/torch.h>
#include "BRepTorch.h"
#include "cnpy.h"
#include "BRepUtils.h"
#include "DebugControl.h"

// OpenCascade 头文件
#include <STEPControl_Reader.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Shape.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Edge.hxx>
#include <TopoDS_Wire.hxx>
#include <TopExp.hxx>
#include <TopExp_Explorer.hxx>
#include <TopTools_IndexedMapOfShape.hxx>
#include <GProp_GProps.hxx>
#include <BRepGProp.hxx>
#include <BRep_Tool.hxx>
#include <GeomAbs_SurfaceType.hxx>
#include <GeomAbs_CurveType.hxx>
#include <BRepAdaptor_Surface.hxx>
#include <BRepAdaptor_Curve.hxx>
#include <GCPnts_AbscissaPoint.hxx>
#include <Geom_Surface.hxx>
#include <Geom_BSplineSurface.hxx>
#include <Geom_BSplineCurve.hxx>
#include <BRepTools_WireExplorer.hxx>
#include <Geom_BezierSurface.hxx>
#include <BRepBndLib.hxx>
#include <Bnd_Box.hxx>
#include <BRepBuilderAPI_Transform.hxx>
#include <gp_Trsf.hxx>
#include <gp_Vec.hxx>
#include <gp_Pnt2d.hxx>
#include <GeomAPI_ProjectPointOnSurf.hxx>
#include <BRepLProp_SLProps.hxx>
#include <Geom2d_Curve.hxx>

// extract_face_point_grids
#include <BRepTools.hxx>
#include <BRepTopAdaptor_FClass2d.hxx>
#include <gp_Pnt2d.hxx>
#include <Precision.hxx>

// extract_face_point_grids
#include <GCPnts_UniformAbscissa.hxx>
#include <GeomLProp_SLProps.hxx>

#include <BRepGProp_Face.hxx>

//namespace breptorch = ::torch; using Tensor = bpt::Tensor;
using namespace breptorch;

// Helper functions for Tensor slicing (to bypass BRepTorch limitations)
inline Tensor get_slice(const Tensor& t, int index) {
    std::vector<int64_t> new_sizes = t.sizes();
    if (new_sizes.empty()) return Tensor();
    new_sizes.erase(new_sizes.begin());

    int64_t block_size = 1;
    for (auto s : new_sizes) block_size *= s;

    Tensor out(new_sizes, t.dtype());
    if (t.dtype() == kFloat32) {
        const float* src = const_cast<Tensor&>(t).data_ptr<float>() + index * block_size;
        std::memcpy(out.data_ptr<float>(), src, block_size * sizeof(float));
    } else if (t.dtype() == kLong) {
        const int64_t* src = const_cast<Tensor&>(t).data_ptr<int64_t>() + index * block_size;
        std::memcpy(out.data_ptr<int64_t>(), src, block_size * sizeof(int64_t));
    }
    return out;
}

inline void set_slice(Tensor& t, int index, const Tensor& val) {
    int64_t block_size = val.numel();
    if (t.dtype() == kFloat32) {
        float* dst = t.data_ptr<float>() + index * block_size;
        const float* src = const_cast<Tensor&>(val).data_ptr<float>();
        std::memcpy(dst, src, block_size * sizeof(float));
    } else if (t.dtype() == kLong) {
        int64_t* dst = t.data_ptr<int64_t>() + index * block_size;
        const int64_t* src = const_cast<Tensor&>(val).data_ptr<int64_t>();
        std::memcpy(dst, src, block_size * sizeof(int64_t));
    }
}

struct CoedgeInfo {
    int id;
    int face_idx;
    int edge_idx;
    int next_idx;
    int prev_idx;
    int mate_idx;
    bool orientation;
};

// ===== 【方案三诊断系统】全局变量和函数声明 =====
FILE* g_diag_arc_length = nullptr;
FILE* g_diag_face_grid = nullptr;

void diagnose_face_19_23_26_grids(
    const std::vector<CoedgeInfo>& coedges,
    const Tensor& FaceGridsLocal,
    const Tensor& EdgeGridsLocal,
    int num_faces,
    int num_edges);

void init_diagnostics();
void close_diagnostics();
// ===== 诊断系统声明结束 =====

class BRepPipeline {
public:
    TopTools_IndexedMapOfShape unique_faces;
    TopTools_IndexedMapOfShape unique_edges;
    std::vector<CoedgeInfo> coedges;

    Tensor Xf, Xe, Xc;
    Tensor Kf, Ke, Kc;
    Tensor Ce, Cf;
    std::vector<Tensor> Csf;

    Tensor mean_f, std_f, mean_e, std_e, mean_c, std_c;
    bool has_stats = false;

    BRepPipeline() {}

    // ===== 【诊断清理】位置5：析构函数 =====
    ~BRepPipeline() {
        close_diagnostics();
        DBG_PRINTF("[DIAGNOSTIC] Diagnostic system closed\n");
    }
    // ===== 诊断清理完毕 =====


    // --- 主处理流程 ---
    bool process(const std::string& step_file_path) {
        // ===== 【诊断初始化】位置4 =====
        init_diagnostics();
        DBG_PRINTF("[DIAGNOSTIC] Diagnostic system initialized\n");
        // ===== 诊断初始化完毕 =====

        coedges.clear();
        unique_faces.Clear();
        unique_edges.Clear();

        // 2. 读取 STEP
        STEPControl_Reader reader;
        IFSelect_ReturnStatus status = reader.ReadFile(step_file_path.c_str());
        int num_roots = reader.NbRootsForTransfer();
        reader.TransferRoots();
        TopoDS_Shape original_shape = reader.OneShape();

        // FIXME: Disable scaling to match Python behavior (which uses original STEP coordinates)
        // TopoDS_Shape shape = BRepUtils::ScaleShape(original_shape);
        TopoDS_Shape shape = original_shape;

        // Build unique faces and edges using default traversal order (same as Python)
        // Python uses: TopologyUtils.TopologyExplorer(body, ignore_orientation=True)
        // C++ equivalent: TopExp_Explorer with default settings

        TopExp_Explorer faceExp(shape, TopAbs_FACE);
        for (; faceExp.More(); faceExp.Next()) {
            TopoDS_Face f = TopoDS::Face(faceExp.Current());
            unique_faces.Add(f);
        }

        TopExp_Explorer edgeExp(shape, TopAbs_EDGE);
        for (; edgeExp.More(); edgeExp.Next()) {
            TopoDS_Edge e = TopoDS::Edge(edgeExp.Current());
            unique_edges.Add(e);
        }

        build_topology();
        extract_features();
        generate_tensors();
        generate_local_grids();
        use_uvnet = true;
        return true;
    }

    void load_stats(const std::string& npz_path) {
        DBG_LOG << "[Debug] Loading stats from: " << npz_path << std::endl;
        try {
            cnpy::npz_t npz = cnpy::npz_load(npz_path);
            auto load_t = [&](const std::string& key) {
                if (!npz.count(key)) {
                    ERR_LOG << "Stats missing key: " << key << std::endl;
                    return breptorch::ones({ 1 }, breptorch::kFloat32);
                }
                cnpy::NpyArray arr = npz[key];
                std::vector<int64_t> shape;
                for (auto s : arr.shape) shape.push_back(s);
                return breptorch::from_blob(arr.data<float>(), shape, breptorch::kFloat32).clone();
                };
            mean_f = load_t("mean_f"); std_f = load_t("std_f");
            mean_e = load_t("mean_e"); std_e = load_t("std_e");
            mean_c = load_t("mean_c"); std_c = load_t("std_c");
            float eps = 1e-6;
            if (std_f.defined())
                std_f = breptorch::where(std_f < eps, breptorch::ones_like(std_f), std_f);
            if (std_e.defined())
                std_e = breptorch::where(std_e < eps, breptorch::ones_like(std_e), std_e);
            if (std_c.defined())
                std_c = breptorch::where(std_c < eps, breptorch::ones_like(std_c), std_c);
            has_stats = true;
        }
        catch (const std::exception& e) {
            ERR_LOG << "Load stats failed: " << e.what() << std::endl;
            has_stats = false;
        }
    }

    // 标准化
    void standardize() {
        if (!has_stats) {
            DBG_LOG << "[Warn] No stats loaded, skipping standardization." << std::endl;
            return;
        }
        DBG_LOG << "[Debug] Executing standardization..." << std::endl;
        if (Xf.size(0) > 1) Xf.sub_(mean_f).div_(std_f);
        if (Xe.size(0) > 1) Xe.sub_(mean_e).div_(std_e);
        if (Xc.size(0) > 1) Xc.sub_(mean_c).div_(std_c);
    }

    Tensor FaceGridsGlobal; // 存储提取的全局 Grid 数据 [N, 9, 40, 40]
    std::vector<std::array<float, 3>> coedge_origins_; // 每条 coedge 的中点坐标（退化边存 {-2000,-2000,-2000}）
    Tensor EdgeGridsGlobal;
    Tensor CoedgeGridsGlobal;
    bool use_uvnet = false;

    // 从外部加载已有 Grid 数据 (从 Python 生成的 npz)
    void load_grids_from_npz(const std::string& npz_path) {
        try {
            cnpy::npz_t npz = cnpy::npz_load(npz_path);

            // 1. Face
            if (npz.count("face_point_grids")) {
                cnpy::NpyArray arr = npz["face_point_grids"];
                std::vector<int64_t> s; for (auto d : arr.shape) s.push_back(d);
                FaceGridsGlobal = breptorch::from_blob(arr.data<float>(), s, breptorch::kFloat32).clone();
                use_uvnet = true;
                DBG_LOG << "Loaded Face Grids: " << FaceGridsGlobal.sizes() << std::endl;
            }

            // 2. Edge
            if (npz.count("edge_point_grids")) {
                cnpy::NpyArray arr = npz["edge_point_grids"];
                std::vector<int64_t> s; for (auto d : arr.shape) s.push_back(d);
                EdgeGridsGlobal = breptorch::from_blob(arr.data<float>(), s, breptorch::kFloat32).clone();
                DBG_LOG << "Loaded Edge Grids: " << EdgeGridsGlobal.sizes() << std::endl;
            }

            // 3. Coedge
            if (npz.count("coedge_point_grids")) {
                cnpy::NpyArray arr = npz["coedge_point_grids"];
                std::vector<int64_t> s; for (auto d : arr.shape) s.push_back(d);
                CoedgeGridsGlobal = breptorch::from_blob(arr.data<float>(), s, breptorch::kFloat32).clone();
                DBG_LOG << "Loaded Coedge Grids: " << CoedgeGridsGlobal.sizes() << std::endl;
            }
        }
        catch (const std::exception& e) {
            ERR_LOG << "Failed to load grids from npz: " << e.what() << std::endl;
        }
    }

    // 存储局部坐标系下的数据
    Tensor FaceGridsLocal;   // [N_c, 2, 9, 20, 20]
    Tensor EdgeGridsLocal;    // 实际没用到
    Tensor CoedgeGridsLocal;  // 实际没用到


private:
    void build_topology() {
        coedges.clear();
        std::map<int, std::vector<int>> edge_to_coedge_map;

        for (int f_idx = 1; f_idx <= unique_faces.Extent(); ++f_idx) {
            const TopoDS_Face& face = TopoDS::Face(unique_faces.FindKey(f_idx));

            // 调试：统计 Face 0 的 Wires 和 Edges
            if (f_idx == 1 && DebugControl::instance().shouldDebug()) {
                int wire_count = 0;
                TopExp_Explorer wireCounter(face, TopAbs_WIRE);
                for (; wireCounter.More(); wireCounter.Next()) wire_count++;
                DBG_LOG << "[Debug Topology] Face 0 has " << wire_count << " wires" << std::endl;
            }

            TopExp_Explorer wireExp(face, TopAbs_WIRE);
            int wire_idx = 0;
            for (; wireExp.More(); wireExp.Next()) {
                const TopoDS_Wire& wire = TopoDS::Wire(wireExp.Current());
                int first_coedge = -1;
                int prev_coedge = -1;

                int edge_count_in_wire = 0;
                BRepTools_WireExplorer edgeExp(wire);
                for (; edgeExp.More(); edgeExp.Next()) {
                    const TopoDS_Edge& edge = edgeExp.Current();
                    int e_idx = unique_edges.FindIndex(edge);

                    CoedgeInfo c;
                    c.id = (int)coedges.size();
                    c.face_idx = f_idx - 1;
                    c.edge_idx = e_idx - 1;
                    c.orientation = (edge.Orientation() == TopAbs_FORWARD);
                    c.next_idx = -1;
                    c.prev_idx = -1;
                    c.mate_idx = -1;

                    coedges.push_back(c);
                    edge_to_coedge_map[e_idx].push_back(c.id);
                    edge_count_in_wire++;

                    if (prev_coedge != -1) {
                        coedges[prev_coedge].next_idx = c.id;
                        coedges[c.id].prev_idx = prev_coedge;
                    }
                    else {
                        first_coedge = c.id;
                    }
                    prev_coedge = c.id;
                }

                // 调试：打印 Face 0 每个 Wire 的边数
                if (f_idx == 1 && DebugControl::instance().shouldDebug()) {
                    DBG_LOG << "[Debug Topology] Face 0, Wire " << wire_idx << " has " << edge_count_in_wire << " edges" << std::endl;
                }
                wire_idx++;

                if (prev_coedge != -1 && first_coedge != -1) {
                    coedges[prev_coedge].next_idx = first_coedge;
                    coedges[first_coedge].prev_idx = prev_coedge;
                }
            }
        }
        for (auto& entry : edge_to_coedge_map) {
            if (entry.second.size() == 2) {
                coedges[entry.second[0]].mate_idx = entry.second[1];
                coedges[entry.second[1]].mate_idx = entry.second[0];
            }
            else {
                for (int id : entry.second) coedges[id].mate_idx = id;
            }
        }
    }

    int walk(int start, const std::vector<int>&cmds) {
        int curr = start;
        for (int cmd : cmds) {
            if (curr < 0 || curr >= coedges.size()) return -1;
            const auto& c = coedges[curr];
            if (cmd == 1) curr = c.mate_idx;
            else if (cmd == 2) curr = c.next_idx;
            else if (cmd == 3) curr = c.prev_idx;

            if (curr == -1) return 0;
        }
        return curr;
    }

    void extract_features() {
        int num_f = unique_faces.Extent();
        int num_e = unique_edges.Extent();
        int num_c = coedges.size();
        Xf = breptorch::zeros({ num_f, 7 });
        Xe = breptorch::zeros({ num_e, 10 });
        Xc = breptorch::zeros({ num_c, 1 });
        auto Xc_a = Xc.accessor<float, 2>();
        for (int i = 0; i < num_c; ++i) {
            if (!coedges[i].orientation) Xc_a[i][0] = 1;
        }

        DBG_LOG << "[Info] Simplified Feature Extraction (No Global Geom Stats)." << std::endl;
    }

    void generate_tensors() {
        int num_f = unique_faces.Extent();
        int num_e = unique_edges.Extent();
        int num_c = coedges.size();

        std::vector<int64_t> kf, ke, kc;
        std::vector<std::vector<int>> fw = { {},{1} }, ew = { {} }, cw = { {},{1} };
        // --- 生成 Kf ---
        for (const auto& c : coedges) {
            for (auto& rule : fw) {
                int t = walk(c.id, rule);
                kf.push_back(t == -1 ? num_f : coedges[t].face_idx);
            }
        }
        Kf = breptorch::from_blob(kf.data(), { num_c, (long long)fw.size() }, breptorch::kLong).clone();

        // --- 生成 Ke ---
        for (const auto& c : coedges) {
            for (auto& rule : ew) {
                int t = walk(c.id, rule);
                ke.push_back(t == -1 ? num_e : coedges[t].edge_idx);
            }
        }
        Ke = breptorch::from_blob(ke.data(), { num_c, (long long)ew.size() }, breptorch::kLong).clone();

        // --- 生成 Kc ---
        for (const auto& c : coedges) {
            for (auto& rule : cw) {
                int t = walk(c.id, rule);
                kc.push_back(t == -1 ? num_c : t);
            }
        }
        Kc = breptorch::from_blob(kc.data(), { num_c, (long long)cw.size() }, breptorch::kLong).clone();

        // --- Pooling Ce ---
        std::vector<int64_t> ce(num_e * 2, num_c);
        std::vector<int> ec(num_e, 0);
        for (const auto& c : coedges) {
            if (ec[c.edge_idx] < 2) ce[c.edge_idx * 2 + ec[c.edge_idx]++] = c.id;
        }
        Ce = breptorch::from_blob(ce.data(), { num_e, 2 }, breptorch::kLong).clone();

        // --- Pooling Cf (按照 Python 的方式：small faces + big faces) ---
        int max_cpf = 30;

        // 1. 统计每个 face 的 coedge 数量
        std::vector<int> fc(num_f, 0);
        for (const auto& c : coedges) {
            fc[c.face_idx]++;
        }

        // 2. 分离 small faces 和 big faces
        std::vector<int> small_face_indices;
        std::vector<int> big_face_indices;
        for (int f = 0; f < num_f; ++f) {
            if (fc[f] <= max_cpf) {
                small_face_indices.push_back(f);
            } else {
                big_face_indices.push_back(f);
            }
        }

        DBG_LOG << "[Debug BRepPipeline] Small faces: " << small_face_indices.size()
                << ", Big faces: " << big_face_indices.size() << std::endl;

        // 3. 构建 face_permutation (small faces 在前，big faces 在后)
        std::vector<int> face_permutation;
        face_permutation.insert(face_permutation.end(), small_face_indices.begin(), small_face_indices.end());
        face_permutation.insert(face_permutation.end(), big_face_indices.begin(), big_face_indices.end());

        // 4. 构建 Cf (只包含 small faces)
        int num_small_faces = small_face_indices.size();
        std::vector<int64_t> cf(num_small_faces * max_cpf, num_c);

        // 为每个 face 收集 coedges
        std::vector<std::vector<int>> face_to_coedges(num_f);
        for (const auto& c : coedges) {
            face_to_coedges[c.face_idx].push_back(c.id);
        }

        // 填充 Cf (按照 face_permutation 的顺序)
        for (int i = 0; i < num_small_faces; ++i) {
            int original_face_idx = small_face_indices[i];
            const auto& coedge_list = face_to_coedges[original_face_idx];
            for (size_t j = 0; j < coedge_list.size() && j < max_cpf; ++j) {
                cf[i * max_cpf + j] = coedge_list[j];
            }
        }

        // 调试：打印 Cf[0] (第一个 small face)
        if (DebugControl::instance().shouldDebug()) {
            DBG_LOG << "[Debug BRepPipeline] Cf[0] corresponds to original Face " << small_face_indices[0] << std::endl;
            DBG_LOG << "[Debug BRepPipeline] Cf[0] has " << face_to_coedges[small_face_indices[0]].size() << " coedges" << std::endl;
            std::ostringstream oss;
            oss << "[Debug BRepPipeline] Cf[0] coedge IDs (first 30): ";
            for (int i = 0; i < std::min(30, (int)face_to_coedges[small_face_indices[0]].size()); ++i) {
                oss << face_to_coedges[small_face_indices[0]][i] << " ";
            }
            DBG_LOG << oss.str() << std::endl;
        }

        Cf = breptorch::from_blob(cf.data(), { num_small_faces, max_cpf }, breptorch::kLong).clone();

        // 5. 构建 Csf (big faces 的 coedges)
        Csf.clear();
        for (int big_face_idx : big_face_indices) {
            const auto& coedge_list = face_to_coedges[big_face_idx];
            std::vector<int64_t> coedge_tensor_data(coedge_list.begin(), coedge_list.end());
            Tensor coedge_tensor = breptorch::from_blob(coedge_tensor_data.data(),
                                                        {(int64_t)coedge_tensor_data.size()},
                                                        breptorch::kLong).clone();
            Csf.push_back(coedge_tensor);
        }

        DBG_LOG << "[Debug BRepPipeline] Csf has " << Csf.size() << " big faces" << std::endl;
    }


    // 对应 python 的 extract_face_point_grid
    // =========================================================================

    // BRepPipeline.h: generate_global_face_grid()
    Tensor generate_global_face_grid(const TopoDS_Face& face) {
        int num_u = 40;
        int num_v = 40;

        // Shape: [9, 40, 40]
        Tensor grid = breptorch::zeros({ 9, num_u, num_v }, breptorch::kFloat32);

        static int debug_face_count = 0;
        bool debug_first_face = (debug_face_count == 0);
        bool debug_first_three = (debug_face_count < 3);

        int64_t stride_c = num_u * num_v;
        int64_t stride_h = num_v;

        Standard_Real umin, umax, vmin, vmax;
        BRepTools::UVBounds(face, umin, umax, vmin, vmax);
        BRepAdaptor_Surface surf(face);
        BRepTopAdaptor_FClass2d classifier(face, 1e-9);

        // IMPORTANT: UV sampling direction depends on face orientation
        bool is_reversed = (face.Orientation() == TopAbs_REVERSED);
        bool u_reverse = is_reversed;
        bool v_reverse = false;

        float* data = grid.data_ptr<float>();

        for (int i = 0; i < num_u; ++i) {
            for (int j = 0; j < num_v; ++j) {
                double u = BRepUtils::GetParamStrict(i, num_u, umin, umax, u_reverse);
                double v = BRepUtils::GetParamStrict(j, num_v, vmin, vmax, v_reverse);

                gp_Pnt p;
                gp_Vec d1u, d1v;
                surf.D1(u, v, p, d1u, d1v);

                // 使用 GeomLProp_SLProps 计算法线（与Python一致）
                Handle(Geom_Surface) geom_surf = BRep_Tool::Surface(face);
                GeomLProp_SLProps props(geom_surf, u, v, 1, 1e-9);

                gp_Vec n;
                if (props.IsNormalDefined()) {
                    gp_Dir normal = props.Normal();
                    n = gp_Vec(normal.XYZ());
                } else {
                    n = gp_Vec(0, 0, 0);
                }

                if (face.Orientation() == TopAbs_REVERSED) {
                    n.Reverse();
                }

                gp_Pnt2d p2d(u, v);
                TopAbs_State state = classifier.Perform(p2d);
                float mask_val = (state == TopAbs_IN) ? 1.0f : 0.0f;

                int64_t idx = i * stride_h + j;

                data[0 * stride_c + idx] = (float)p.X();
                data[1 * stride_c + idx] = (float)p.Y();
                data[2 * stride_c + idx] = (float)p.Z();

                data[3 * stride_c + idx] = (float)n.X();
                data[4 * stride_c + idx] = (float)n.Y();
                data[5 * stride_c + idx] = (float)n.Z();

                data[6 * stride_c + idx] = mask_val;

                data[7 * stride_c + idx] = (float)u;
                data[8 * stride_c + idx] = (float)v;
            }
        }

        return grid;
    }

    Tensor generate_global_coedge_grid(int coedge_idx) {
        DBG_CERR << "[DEBUG ArcLength] generate_global_coedge_grid() called for coedge " << coedge_idx << std::endl;
        const CoedgeInfo& c_info = coedges[coedge_idx];

        // 1. 获取几何实体
        TopoDS_Face face_left = TopoDS::Face(unique_faces.FindKey(c_info.face_idx + 1));
        TopoDS_Edge edge = TopoDS::Edge(unique_edges.FindKey(c_info.edge_idx + 1));

        // 获取 Mate 面 (Right Face)
        TopoDS_Face face_right;
        bool has_mate = (c_info.mate_idx != -1);
        if (has_mate) {
            int mate_face_idx = coedges[c_info.mate_idx].face_idx;
            face_right = TopoDS::Face(unique_faces.FindKey(mate_face_idx + 1));
        }

        // 2. 准备 Tensor [13, 20]
        int num_u = 20;
        Tensor grid = breptorch::zeros({13, num_u}, breptorch::kFloat32);

        // ✅ 【关键修复】检查curve是否为NULL
        double u0_check, u1_check;
        Handle(Geom_Curve) curve_check = BRep_Tool::Curve(edge, u0_check, u1_check);
        if (curve_check.IsNull()) {
            return grid;
        }

        // 3. 创建curve adaptor
        BRepAdaptor_Curve curve_adaptor(edge);
        double first = curve_adaptor.FirstParameter();
        double last = curve_adaptor.LastParameter();
        double len = last - first;

        // 4. 弧长参数化（与Python一致）
        DBG_CERR << "[DEBUG ArcLength] Coedge " << coedge_idx << ": Using NEW arc-length parameterization (100 samples, line distance)" << std::endl;

        // 步骤1：在边上采样100个点
        std::vector<gp_Pnt> sample_points;
        std::vector<double> sample_params;
        for (int j = 0; j < 100; ++j) {
            double u = first + (len * j) / 99.0;
            gp_Pnt p = curve_adaptor.Value(u);
            sample_points.push_back(p);
            sample_params.push_back(u);
        }

        // 步骤2：计算直线距离近似弧长
        std::vector<double> lengths;
        double total_length = 0;
        for (int j = 1; j < 100; ++j) {
            double length = sample_points[j].Distance(sample_points[j-1]);
            lengths.push_back(length);
            total_length += length;
        }

        // 步骤3：计算累积弧长比例
        std::vector<double> arc_length_fraction;
        arc_length_fraction.push_back(0.0);
        double cumulative_length = 0;
        for (int j = 0; j < lengths.size(); ++j) {
            cumulative_length += lengths[j];
            arc_length_fraction.push_back(cumulative_length / total_length);
        }

        // 步骤4：对于每个目标采样点，使用线性插值计算参数
        std::vector<double> target_params;
        for (int i = 0; i < num_u; ++i) {
            double desired_arc_length_fraction = i / (double)(num_u - 1);
            int arc_length_index = 0;
            while (arc_length_fraction[arc_length_index] < desired_arc_length_fraction) {
                arc_length_index++;
                if (arc_length_index >= (int)arc_length_fraction.size() - 1) break;
            }
            double frac_low, frac_high, u_low, u_high;
            if (arc_length_index == 0) {
                u_low = sample_params[0];
                frac_low = arc_length_fraction[0];
            } else {
                u_low = sample_params[arc_length_index - 1];
                frac_low = arc_length_fraction[arc_length_index - 1];
            }
            u_high = sample_params[arc_length_index];
            frac_high = arc_length_fraction[arc_length_index];
            double d_frac = frac_high - frac_low;
            double param;
            if (d_frac <= 0.0) {
                param = u_low;
            } else {
                double t = (desired_arc_length_fraction - frac_low) / d_frac;
                param = u_low + t * (u_high - u_low);
            }
            target_params.push_back(param);
        }

        // 5. 获取pcurve（用于获取UV参数，与Python一致）
        Handle(Geom2d_Curve) left_pcurve;
        double left_u0, left_u1;
        left_pcurve = BRep_Tool::CurveOnSurface(edge, face_left, left_u0, left_u1);

        Handle(Geom2d_Curve) right_pcurve;
        double right_u0, right_u1;
        if (has_mate) {
            right_pcurve = BRep_Tool::CurveOnSurface(edge, face_right, right_u0, right_u1);
        }

        // 6. 循环采样
        float* data = grid.data_ptr<float>();
        int64_t stride_c = num_u;

        for (int i = 0; i < num_u; ++i) {
            double param = target_params[i];

            gp_Pnt p;
            gp_Vec tangent;
            curve_adaptor.D1(param, p, tangent);

            // 归一化切线向量
            if (tangent.Magnitude() > 1e-7) tangent.Normalize();

            // 处理切线方向 (Orientation)
            if (!c_info.orientation) {
                tangent.Reverse();
            }

            // 计算左面法线
            gp_Vec n_left;
            if (!left_pcurve.IsNull()) {
                gp_Pnt2d uv_left_2d;
                left_pcurve->D0(param, uv_left_2d);
                double u_left = uv_left_2d.X();
                double v_left = uv_left_2d.Y();

                // 【步骤3】使用 GeomLProp_SLProps + BRep_Tool::Surface（与Python一致）
                TopLoc_Location loc_left;
                Handle(Geom_Surface) geom_surf_left = BRep_Tool::Surface(face_left, loc_left);
                GeomLProp_SLProps props_left(geom_surf_left, u_left, v_left, 1, 1e-9);

                if (props_left.IsNormalDefined()) {
                    n_left = gp_Vec(props_left.Normal());
                    if (!loc_left.IsIdentity()) {
                        n_left.Transform(loc_left.Transformation());
                    }

                    if (face_left.Orientation() == TopAbs_REVERSED) n_left.Reverse();
                    if (n_left.Magnitude() > 1e-7) n_left.Normalize();
                }
            }
            // Python在IsNormalDefined()为false时返回零向量，不做fallback

            // 计算右面法线
            gp_Vec n_right;
            if (has_mate && !right_pcurve.IsNull()) {
                gp_Pnt2d uv_right_2d;
                right_pcurve->D0(param, uv_right_2d);
                double u_right = uv_right_2d.X();
                double v_right = uv_right_2d.Y();

                // 【步骤3】使用 GeomLProp_SLProps + BRep_Tool::Surface（与Python一致）
                TopLoc_Location loc_right;
                Handle(Geom_Surface) geom_surf_right = BRep_Tool::Surface(face_right, loc_right);
                GeomLProp_SLProps props_right(geom_surf_right, u_right, v_right, 1, 1e-9);

                if (props_right.IsNormalDefined()) {
                    n_right = gp_Vec(props_right.Normal());
                    if (!loc_right.IsIdentity()) {
                        n_right.Transform(loc_right.Transformation());
                    }


                    if (face_right.Orientation() == TopAbs_REVERSED) n_right.Reverse();
                    if (n_right.Magnitude() > 1e-7) n_right.Normalize();
                }
            }
            // Python在IsNormalDefined()为false时返回零向量，不做fallback

            // 写入 Tensor
            // Points (0-2)
            data[0 * stride_c + i] = (float)p.X();
            data[1 * stride_c + i] = (float)p.Y();
            data[2 * stride_c + i] = (float)p.Z();

            // Tangents (3-5)
            data[3 * stride_c + i] = (float)tangent.X();
            data[4 * stride_c + i] = (float)tangent.Y();
            data[5 * stride_c + i] = (float)tangent.Z();

            // Left Normals (6-8)
            data[6 * stride_c + i] = (float)n_left.X();
            data[7 * stride_c + i] = (float)n_left.Y();
            data[8 * stride_c + i] = (float)n_left.Z();

            // Right Normals (9-11)
            data[9 * stride_c + i] = (float)n_right.X();
            data[10 * stride_c + i] = (float)n_right.Y();
            data[11 * stride_c + i] = (float)n_right.Z();

            // 第12通道（u参数）对应Python的u_params
            data[12 * stride_c + i] = (float)param;
        }

        // 如果 orientation 是 false（REVERSED），则需要 flip dim 1
        if (!c_info.orientation) {
            grid = breptorch::flip(grid, {1});
        }

        return grid;
    }

    double compute_arc_length_midpoint(Handle(Geom_Curve) curve, double u0, double u1) {
        // 与 Python ArcLengthParamFinder(num_arc_length_samples=100) 保持一致
        // Python 源码: D:\occwl\src\occwl\geometry\arc_length_param_finder.py 第12行默认值
        double param_span = u1 - u0;
        int num_samples = 100;

        std::vector<gp_Pnt> sampled_points;
        std::vector<double> arc_length_vals;

        gp_Pnt p_first = curve->Value(u0);
        sampled_points.push_back(p_first);
        arc_length_vals.push_back(0.0);

        for (int i = 1; i < num_samples; ++i) {
            double u_sample = u0 + (u1 - u0) * i / (num_samples - 1);
            gp_Pnt p_sample = curve->Value(u_sample);
            sampled_points.push_back(p_sample);
            double dist = sampled_points[i - 1].Distance(p_sample);
            arc_length_vals.push_back(arc_length_vals.back() + dist);
        }

        double total_length = arc_length_vals.back();
        double mid_length = total_length / 2.0;

        int idx_left = 0;
        for (int i = 0; i < (int)arc_length_vals.size() - 1; ++i) {
            if (arc_length_vals[i] <= mid_length && mid_length <= arc_length_vals[i + 1]) {
                idx_left = i;
                break;
            }
        }

        if (idx_left >= (int)arc_length_vals.size() - 1) {
            idx_left = (int)arc_length_vals.size() - 2;
        }

        double denom = arc_length_vals[idx_left + 1] - arc_length_vals[idx_left];
        double ratio = (denom > 1e-10) ? (mid_length - arc_length_vals[idx_left]) / denom : 0.5;
        ratio = std::max(0.0, std::min(1.0, ratio));

        double u_left = u0 + (u1 - u0) * idx_left / (num_samples - 1);
        double u_right = u0 + (u1 - u0) * (idx_left + 1) / (num_samples - 1);
        double u_arc_mid = u_left + (u_right - u_left) * ratio;

        // ===== 【诊断输出】Arc-Length vs Parameter中点对比 =====
        if (EXPORT_ENABLED) {
            double u_param_mid = (u0 + u1) / 2.0;
            double diff = std::abs(u_arc_mid - u_param_mid);
            double diff_percent = (param_span > 1e-10) ? (diff / param_span) * 100.0 : 0;

            if (!g_diag_arc_length && diff_percent > 1.0) {
                g_diag_arc_length = fopen("arc_length_diagnosis.txt", "w");
                if (g_diag_arc_length) {
                    fprintf(g_diag_arc_length, "Arc-Length vs Parameter Midpoint Tracking\n");
                    fprintf(g_diag_arc_length, "span u0 u1 param_mid arc_mid diff diff%% samples\n\n");
                }
            }

            if (g_diag_arc_length && diff_percent > 1.0) {
                fprintf(g_diag_arc_length,
                        "%.4f %.6f %.6f %.6f %.6f %.6f %.2f%% %d\n",
                        param_span, u0, u1, u_param_mid, u_arc_mid, diff, diff_percent, num_samples);
                fflush(g_diag_arc_length);
            }
        }
        // ===== 诊断输出完毕 =====

        return u_arc_mid;
    }

    Tensor compute_coedge_lcs(int coedge_idx) {
        const CoedgeInfo& c_info = coedges[coedge_idx];
        TopoDS_Edge edge = TopoDS::Edge(unique_edges.FindKey(c_info.edge_idx + 1));
        TopoDS_Face face = TopoDS::Face(unique_faces.FindKey(c_info.face_idx + 1));

        // 1. 获取中点、切线、法线
        double u0, u1;
        Handle(Geom_Curve) curve = BRep_Tool::Curve(edge, u0, u1);

        // 检查曲线是否有效（退化边可能没有曲线）
        if (curve.IsNull()) {
            coedge_origins_[coedge_idx] = {-2000.0f, -2000.0f, -2000.0f};
            Tensor lcs_inv = breptorch::eye(4);
            return lcs_inv;
        }

        double u_mid = compute_arc_length_midpoint(curve, u0, u1);

        gp_Pnt p;
        gp_Vec tangent;

        curve->D1(u_mid, p, tangent);
        coedge_origins_[coedge_idx] = {(float)p.X(), (float)p.Y(), (float)p.Z()};

        // 【步骤2】使用 pcurve + GeomLProp_SLProps 计算法线（与Python一致）
        gp_Vec normal;

        bool normal_found = false;

        // 方法1: 通过 pcurve 获取 UV，再用 GeomLProp_SLProps 计算法线
        Handle(Geom2d_Curve) pcurve_lcs;
        double pc_u0, pc_u1;
        pcurve_lcs = BRep_Tool::CurveOnSurface(edge, face, pc_u0, pc_u1);
        if (!pcurve_lcs.IsNull()) {
            gp_Pnt2d uv_mid;
            pcurve_lcs->D0(u_mid, uv_mid);
            double uu = uv_mid.X();
            double vv = uv_mid.Y();

            TopLoc_Location loc;
            Handle(Geom_Surface) geom_surf = BRep_Tool::Surface(face, loc);
            GeomLProp_SLProps props(geom_surf, uu, vv, 1, 1e-9);
            if (props.IsNormalDefined()) {
                gp_Dir n_dir = props.Normal();
                normal = gp_Vec(n_dir);
                // 将法线从几何表面坐标系变换到全局坐标系
                if (!loc.IsIdentity()) {
                    normal.Transform(loc.Transformation());
                }
                // orientation 翻转（与 Python face.normal(uv) 一致）
                if (face.Orientation() == TopAbs_REVERSED) {
                    normal.Reverse();
                }
                normal_found = true;
            }
        }

        // 方法2: fallback - 使用 GeomAPI_ProjectPointOnSurf
        if (!normal_found) {
            TopLoc_Location loc2;
            Handle(Geom_Surface) geom_surf2 = BRep_Tool::Surface(face, loc2);
            GeomAPI_ProjectPointOnSurf proj(p, geom_surf2);
            if (proj.NbPoints() > 0) {
                double uu2, vv2;
                proj.LowerDistanceParameters(uu2, vv2);
                GeomLProp_SLProps props2(geom_surf2, uu2, vv2, 1, 1e-9);
                if (props2.IsNormalDefined()) {
                    gp_Dir n_dir2 = props2.Normal();
                    normal = gp_Vec(n_dir2);
                    if (!loc2.IsIdentity()) {
                        normal.Transform(loc2.Transformation());
                    }
                    if (face.Orientation() == TopAbs_REVERSED) {
                        normal.Reverse();
                    }
                    normal_found = true;
                }
            }
        }

        // 方法3: 最终 fallback - 使用 BRepUtils::GetNormalAtPoint
        if (!normal_found) {
            normal = BRepUtils::GetNormalAtPoint(face, p);
        }

        gp_Vec t_vec = tangent;
        gp_Vec n_vec = normal;

        if (!c_info.orientation) {
            t_vec.Reverse();
        }

        float p_arr[3] = { (float)p.X(), (float)p.Y(), (float)p.Z() };
        float t_arr[3] = { (float)t_vec.X(), (float)t_vec.Y(), (float)t_vec.Z() };
        float n_arr[3] = { (float)n_vec.X(), (float)n_vec.Y(), (float)n_vec.Z() };

        // 【步骤6】W 轴归一化（与Python numpy.linalg.norm一致，不加epsilon）
        float w_norm = sqrt(n_arr[0] * n_arr[0] + n_arr[1] * n_arr[1] + n_arr[2] * n_arr[2]);
        if (w_norm < 1e-10f) w_norm = 1e-10f;  // 仅防除零

        float w_vec[3] = {
            n_arr[0] / w_norm,
            n_arr[1] / w_norm,
            n_arr[2] / w_norm
        };

        // 2. V 轴 (切线投影到垂直于 W 轴的平面)
        float dot_tw = t_arr[0] * w_vec[0] + t_arr[1] * w_vec[1] + t_arr[2] * w_vec[2];
        float v_vec[3] = {
            t_arr[0] - dot_tw * w_vec[0],
            t_arr[1] - dot_tw * w_vec[1],
            t_arr[2] - dot_tw * w_vec[2]
        };

        // V 轴归一化
        float v_norm = sqrt(v_vec[0] * v_vec[0] + v_vec[1] * v_vec[1] + v_vec[2] * v_vec[2]);

        if (v_norm < 1e-6f) {
            float temp[3] = { 1.0f, 0.0f, 0.0f };
            if (fabs(w_vec[0]) > 0.9f) {
                temp[0] = 0.0f;
                temp[1] = 1.0f;
                temp[2] = 0.0f;
            }

            float dot_temp_w = temp[0] * w_vec[0] + temp[1] * w_vec[1] + temp[2] * w_vec[2];
            v_vec[0] = temp[0] - dot_temp_w * w_vec[0];
            v_vec[1] = temp[1] - dot_temp_w * w_vec[1];
            v_vec[2] = temp[2] - dot_temp_w * w_vec[2];

            v_norm = sqrt(v_vec[0] * v_vec[0] + v_vec[1] * v_vec[1] + v_vec[2] * v_vec[2]);
        }

        // 【步骤6】V 轴归一化（与Python numpy.linalg.norm一致，不加epsilon）
        if (v_norm < 1e-10f) v_norm = 1e-10f;  // 仅防除零
        v_vec[0] /= v_norm;
        v_vec[1] /= v_norm;
        v_vec[2] /= v_norm;

        // 3. U 轴 (V × W)
        float u_vec[3] = {
            v_vec[1] * w_vec[2] - v_vec[2] * w_vec[1],
            v_vec[2] * w_vec[0] - v_vec[0] * w_vec[2],
            v_vec[0] * w_vec[1] - v_vec[1] * w_vec[0]
        };


        // 4. 组装矩阵 (手动赋值)
        Tensor mat = breptorch::eye(4);
        float* mat_ptr = mat.data_ptr<float>();

        // 旋转部分 (前3x3) - 列主序
        mat_ptr[0 * 4 + 0] = u_vec[0];  mat_ptr[0 * 4 + 1] = v_vec[0];  mat_ptr[0 * 4 + 2] = w_vec[0];
        mat_ptr[1 * 4 + 0] = u_vec[1];  mat_ptr[1 * 4 + 1] = v_vec[1];  mat_ptr[1 * 4 + 2] = w_vec[1];
        mat_ptr[2 * 4 + 0] = u_vec[2];  mat_ptr[2 * 4 + 1] = v_vec[2];  mat_ptr[2 * 4 + 2] = w_vec[2];

        // 平移部分 (第4列)
        mat_ptr[0 * 4 + 3] = p_arr[0];
        mat_ptr[1 * 4 + 3] = p_arr[1];
        mat_ptr[2 * 4 + 3] = p_arr[2];

        return mat;
    }

    // 对应 Python 的 transform_face_point_grid_to_local
    Tensor transform_grid_to_local(Tensor grid, Tensor lcs_inv, bool is_face) {
        Tensor new_grid = grid.clone();
        float* data = new_grid.data_ptr<float>();

        // grid shape: [C, H, W] or [C, L]
        int C = (int)grid.size(0);
        int N = (int)(grid.numel() / C);

        // lcs_inv: [4, 4]
        float* mat = lcs_inv.data_ptr<float>();

        for (int i = 0; i < N; ++i) {
            // Points: channels 0, 1, 2
            float x = data[0 * N + i];
            float y = data[1 * N + i];
            float z = data[2 * N + i];

            // Apply affine transform: P' = M * P
            float x_new = mat[0]*x + mat[1]*y + mat[2]*z + mat[3];
            float y_new = mat[4]*x + mat[5]*y + mat[6]*z + mat[7];
            float z_new = mat[8]*x + mat[9]*y + mat[10]*z + mat[11];

            data[0 * N + i] = x_new;
            data[1 * N + i] = y_new;
            data[2 * N + i] = z_new;

            // Vectors
            if (is_face) {
                // Normals: channels 3, 4, 5
                float nx = data[3 * N + i];
                float ny = data[4 * N + i];
                float nz = data[5 * N + i];

                // Apply rotation only
                float nx_new = mat[0]*nx + mat[1]*ny + mat[2]*nz;
                float ny_new = mat[4]*nx + mat[5]*ny + mat[6]*nz;
                float nz_new = mat[8]*nx + mat[9]*ny + mat[10]*nz;

                data[3 * N + i] = nx_new;
                data[4 * N + i] = ny_new;
                data[5 * N + i] = nz_new;
            } else {
                // Coedge:
                // Tangent: 3,4,5
                // LeftN: 6,7,8
                // RightN: 9,10,11

                for (int k = 0; k < 3; ++k) { // 3 vectors
                    int base_c = 3 + k * 3;
                    float vx = data[(base_c + 0) * N + i];
                    float vy = data[(base_c + 1) * N + i];
                    float vz = data[(base_c + 2) * N + i];

                    float vx_new = mat[0]*vx + mat[1]*vy + mat[2]*vz;
                    float vy_new = mat[4]*vx + mat[5]*vy + mat[6]*vz;
                    float vz_new = mat[8]*vx + mat[9]*vy + mat[10]*vz;

                    data[(base_c + 0) * N + i] = vx_new;
                    data[(base_c + 1) * N + i] = vy_new;
                    data[(base_c + 2) * N + i] = vz_new;
                }
            }
        }

        return new_grid;
    }


    // 生成全局Face网格
    void generate_global_face_grids() {
        if (unique_faces.Extent() == 0) return;

        std::vector<Tensor> grids_list;
        int num_faces = unique_faces.Extent();

        for (int i = 1; i <= num_faces; ++i) {
            const TopoDS_Face& face = TopoDS::Face(unique_faces.FindKey(i));
            Tensor single_grid = generate_global_face_grid(face);
            grids_list.push_back(single_grid.clone());
        }

        if (!grids_list.empty()) {
            this->FaceGridsGlobal = breptorch::stack(grids_list);
        }
    }

    // 计算所有Coedge的LCS变换矩阵
    void compute_all_lcs_matrices(std::vector<Tensor>& lcs_invs) {
        int num_c = coedges.size();
        lcs_invs.clear();
        lcs_invs.reserve(num_c);
        coedge_origins_.resize(num_c);

        for (int i = 0; i < num_c; ++i) {
            Tensor mat = compute_coedge_lcs(i);

            // ===== 【LCS调试】输出Face 19的coedges（87, 88, 89）的LCS矩阵 =====
            if (DebugControl::instance().shouldDebug() && (i == 87 || i == 88 || i == 89)) {
                DBG_LOG << "\n[DEBUG LCS] Coedge " << i << " (Face 19):" << std::endl;
                DBG_LOG << "  Forward LCS Matrix:" << std::endl;
                float* m = const_cast<Tensor&>(mat).data_ptr<float>();
                for (int r = 0; r < 4; r++) {
                    std::ostringstream row;
                    row << std::fixed << std::setprecision(6);
                    row << "    [" << std::setw(12) << m[r * 4]
                        << ", " << std::setw(12) << m[r * 4 + 1]
                        << ", " << std::setw(12) << m[r * 4 + 2]
                        << ", " << std::setw(12) << m[r * 4 + 3] << "]";
                    DBG_LOG << row.str() << std::endl;
                }
                {
                    std::ostringstream oss;
                    oss << std::fixed << std::setprecision(6);
                    oss << "  Origin: [" << std::setw(12) << m[3] << ", " << std::setw(12) << m[7] << ", " << std::setw(12) << m[11] << "]";
                    DBG_LOG << oss.str() << std::endl;
                }
                {
                    std::ostringstream oss;
                    oss << std::fixed << std::setprecision(6);
                    oss << "  u_vec:  [" << std::setw(12) << m[0] << ", " << std::setw(12) << m[4] << ", " << std::setw(12) << m[8] << "]";
                    DBG_LOG << oss.str() << std::endl;
                }
                {
                    std::ostringstream oss;
                    oss << std::fixed << std::setprecision(6);
                    oss << "  v_vec:  [" << std::setw(12) << m[1] << ", " << std::setw(12) << m[5] << ", " << std::setw(12) << m[9] << "]";
                    DBG_LOG << oss.str() << std::endl;
                }
                {
                    std::ostringstream oss;
                    oss << std::fixed << std::setprecision(6);
                    oss << "  w_vec:  [" << std::setw(12) << m[2] << ", " << std::setw(12) << m[6] << ", " << std::setw(12) << m[10] << "]";
                    DBG_LOG << oss.str() << std::endl;
                }
                {
                    float origin_norm = std::sqrt(m[3]*m[3] + m[7]*m[7] + m[11]*m[11]);
                    DBG_LOG << "  LCS Origin Norm: " << std::fixed << std::setprecision(6) << origin_norm << std::endl;
                }
            }
            // ===== LCS调试结束 =====

            if (std::abs(breptorch::det(mat)) < 1e-10) {
                mat = breptorch::eye(4);
            }

            Tensor mat_inv = breptorch::inverse(mat);

            // ===== 【LCS调试】输出逆矩阵 =====
            if (DebugControl::instance().shouldDebug() && (i == 87 || i == 88 || i == 89)) {
                DBG_LOG << "  Inverse LCS Matrix:" << std::endl;
                float* m_inv = const_cast<Tensor&>(mat_inv).data_ptr<float>();
                for (int r = 0; r < 4; r++) {
                    std::ostringstream row;
                    row << std::fixed << std::setprecision(6);
                    row << "    [" << std::setw(12) << m_inv[r * 4]
                        << ", " << std::setw(12) << m_inv[r * 4 + 1]
                        << ", " << std::setw(12) << m_inv[r * 4 + 2]
                        << ", " << std::setw(12) << m_inv[r * 4 + 3] << "]";
                    DBG_LOG << row.str() << std::endl;
                }
            }
            // ===== LCS调试结束 =====

            lcs_invs.push_back(mat_inv);
        }
    }

    // 生成Coedge局部网格
    void generate_coedge_local_grids(const std::vector<Tensor>& lcs_invs) {
        int num_c = coedges.size();
        std::vector<Tensor> c_list;
        c_list.reserve(num_c);

        DBG_CERR << "[DEBUG] generate_coedge_local_grids() called, processing " << num_c << " coedges" << std::endl;
        for (int i = 0; i < num_c; ++i) {
            DBG_CERR << "[DEBUG] Processing coedge " << i << "..." << std::endl;
            Tensor g_global = generate_global_coedge_grid(i);
            Tensor g_local = transform_grid_to_local(g_global, lcs_invs[i], false);
            c_list.push_back(g_local);
        }

        CoedgeGridsLocal = breptorch::stack(c_list);
    }

    // ---- Face Grid 裁剪辅助结构（与 Python new/ 版本对应）----
    struct CropRange { int row_min, row_max, col_min, col_max; };

    // 在 40×40 全局 face grid 中找离 target_point 最近的格点，
    // 以该点为中心计算 20×20 裁剪范围（精确复刻 Python new/ 逻辑）。
    // 退化边（target_point[0] <= -1000）返回左上角 {0,20,0,20}。
    CropRange compute_crop_range(Tensor face_grid_global,
                                  const std::array<float, 3>& target_point)
    {
        const int grid_h = 40, grid_w = 40, samplesize = 20;

        if (target_point[0] <= -1000.0f) {
            return {0, samplesize, 0, samplesize};
        }

        const float* data = face_grid_global.data_ptr<float>();
        const int N = grid_h * grid_w;
        float min_dist_sq = std::numeric_limits<float>::max();
        int best_row = 0, best_col = 0;
        for (int i = 0; i < grid_h; ++i) {
            for (int j = 0; j < grid_w; ++j) {
                int idx = i * grid_w + j;
                float dx = data[0 * N + idx] - target_point[0];
                float dy = data[1 * N + idx] - target_point[1];
                float dz = data[2 * N + idx] - target_point[2];
                float d2 = dx*dx + dy*dy + dz*dz;
                if (d2 < min_dist_sq) { min_dist_sq = d2; best_row = i; best_col = j; }
            }
        }

        // 与 Python 逐维裁剪逻辑完全一致
        auto calc_range = [&](int pos, int size) -> std::pair<int,int> {
            int d0 = pos;
            int d1 = size - pos - 1;
            int min_d = std::min(d0, d1);
            bool closer_to_start = (d0 <= d1);
            int rmin, rmax;
            if (min_d >= samplesize / 2 - 1) {
                rmin = min_d - samplesize / 2 + 1;
                rmax = rmin + samplesize - 1;
            } else {
                if (closer_to_start) {
                    rmin = 0; rmax = samplesize - 1;
                } else {
                    rmax = size - 1; rmin = rmax - samplesize + 1;
                }
            }
            return {rmin, rmax + 1};  // rmax+1 对应 Python 切片上界
        };

        auto [row_min, row_max] = calc_range(best_row, grid_h);
        auto [col_min, col_max] = calc_range(best_col, grid_w);
        return {row_min, row_max, col_min, col_max};
    }

    // 从 [9, 40, 40] tensor 中裁剪出 [9, 20, 20]
    Tensor crop_face_grid(Tensor t, const CropRange& cr, int samplesize) {
        Tensor out = breptorch::zeros({ 9, samplesize, samplesize }, breptorch::kFloat32);
        const float* src = t.data_ptr<float>();
        float* dst = out.data_ptr<float>();
        const int grid_w = 40;
        const int out_w = samplesize;
        for (int c = 0; c < 9; ++c) {
            for (int r = cr.row_min; r < cr.row_max; ++r) {
                for (int col = cr.col_min; col < cr.col_max; ++col) {
                    int src_idx = c * grid_w * grid_w + r * grid_w + col;
                    int dst_r = r - cr.row_min;
                    int dst_c = col - cr.col_min;
                    int dst_idx = c * out_w * out_w + dst_r * out_w + dst_c;
                    dst[dst_idx] = src[src_idx];
                }
            }
        }
        return out;
    }

    // 生成Face局部网格（以 Coedge 中点为中心裁剪 20×20，与 Python new/ 版本一致）
    void generate_face_local_grids(std::vector<Tensor>& lcs_invs) {
        int num_c = coedges.size();
        std::vector<Tensor> f_list;
        f_list.reserve(num_c);

        const int samplesize = 20;

        for (int i = 0; i < num_c; ++i) {
            Tensor pair = breptorch::zeros({ 2, 9, samplesize, samplesize }, breptorch::kFloat32);

            // Left Face (parent face of coedge i)
            int f_idx = coedges[i].face_idx;
            if (FaceGridsGlobal.defined() && f_idx < FaceGridsGlobal.size(0)) {
                bool is_degenerate = (coedge_origins_[i][0] <= -1000.0f);
                if (!is_degenerate) {
                    Tensor global_grid = get_slice(FaceGridsGlobal, f_idx);  // [9, 40, 40]
                    CropRange cr = compute_crop_range(global_grid, coedge_origins_[i]);
                    Tensor t = transform_grid_to_local(global_grid, lcs_invs[i], true);
                    set_slice(pair, 0, crop_face_grid(t, cr, samplesize));
                }
                // 退化时 pair[0] 保持全零
            }

            // Right Face (mate face)
            int mate_idx = coedges[i].mate_idx;
            if (mate_idx != -1) {
                int mf_idx = coedges[mate_idx].face_idx;
                if (FaceGridsGlobal.defined() && mf_idx < FaceGridsGlobal.size(0)) {
                    bool is_degenerate_m = (coedge_origins_[mate_idx][0] <= -1000.0f);
                    if (!is_degenerate_m) {
                        Tensor global_grid_m = get_slice(FaceGridsGlobal, mf_idx);
                        CropRange cr_m = compute_crop_range(global_grid_m, coedge_origins_[mate_idx]);
                        Tensor tm = transform_grid_to_local(global_grid_m, lcs_invs[mate_idx], true);
                        set_slice(pair, 1, crop_face_grid(tm, cr_m, samplesize));
                    }
                }
            }

            f_list.push_back(pair);
        }

        FaceGridsLocal = breptorch::stack(f_list);
    }

    // 生成Edge局部坐标系网格（修正版：使用Python的"左coedge"选择逻辑）
    void generate_edge_local_grids() {
        int num_e = unique_edges.Extent();
        DBG_LOG << "[DEBUG EdgeGrids] generate_edge_local_grids() called, num_e=" << num_e
                << ", CoedgeGridsLocal.defined()=" << CoedgeGridsLocal.defined() << std::endl;
        if (num_e == 0 || !CoedgeGridsLocal.defined()) {
            DBG_LOG << "[DEBUG EdgeGrids] Early return due to num_e==0 or CoedgeGridsLocal not defined" << std::endl;
            return;
        }

        DBG_LOG << "[DEBUG EdgeGrids] generate_edge_local_grids() starting..." << std::endl;
        DBG_LOG << "[DEBUG EdgeGrids] num_edges = " << num_e << std::endl;

        // 步骤1：按Edge索引收集Coedges
        std::vector<std::vector<int>> coedges_of_edges(num_e);
        for (const auto& c : coedges) {
            int eid = c.edge_idx;
            if (eid >= 0 && eid < num_e) {
                coedges_of_edges[eid].push_back(c.id);
            }
        }

        // 步骤2：处理特殊情况（球面等只有1个coedge的edge）
        for (int i = 0; i < num_e; ++i) {
            if (coedges_of_edges[i].size() == 1) {
                coedges_of_edges[i].push_back(coedges_of_edges[i][0]);
            }
        }

        // 步骤3：应用Python逻辑选择"左coedge"
        std::vector<Tensor> e_list;
        e_list.reserve(num_e);

        static bool first_print = true;

        for (int edge_idx = 0; edge_idx < num_e; ++edge_idx) {
            int selected_coedge = -1;

            const auto& edge_coedges = coedges_of_edges[edge_idx];

            if (edge_coedges.size() >= 2) {
                int first_coedge_id = edge_coedges[0];
                int second_coedge_id = edge_coedges[1];

                if (!coedges[second_coedge_id].orientation) {
                    selected_coedge = first_coedge_id;
                } else {
                    selected_coedge = second_coedge_id;
                }

                // 调试输出（仅前10条edge）
                if (DebugControl::instance().shouldDebug() && first_print && edge_idx < 10) {
                    DBG_PRINTF("[DEBUG EdgeGrids] Edge %d: "
                            "first=%d(orient=%s), second=%d(orient=%s), selected=%d\n",
                            edge_idx,
                            first_coedge_id, coedges[first_coedge_id].orientation ? "FORWARD" : "REVERSED",
                            second_coedge_id, coedges[second_coedge_id].orientation ? "FORWARD" : "REVERSED",
                            selected_coedge);
                }
            } else if (edge_coedges.size() == 1) {
                selected_coedge = edge_coedges[0];
            }

            if (selected_coedge != -1 && selected_coedge < CoedgeGridsLocal.size(0)) {
                e_list.push_back(get_slice(CoedgeGridsLocal, selected_coedge));
            } else {
                e_list.push_back(breptorch::zeros({ 13, 20 }, CoedgeGridsLocal.options()));
            }
        }

        first_print = false;

        EdgeGridsLocal = breptorch::stack(e_list);

        DBG_LOG << "[DEBUG EdgeGrids] Generated EdgeGridsLocal with shape "
                << EdgeGridsLocal.sizes() << std::endl;
    }

    // 主入口：生成所有局部网格
    void generate_local_grids() {
        if (unique_faces.Extent() == 0) return;

        // 1. 生成全局Face网格
        generate_global_face_grids();

        int num_c = coedges.size();
        if (num_c == 0) return;

        DBG_LOG << "Generating local coordinate system features (LCS Transformation)..." << std::endl;
        DBG_CERR << "[DEBUG] generate_local_grids() called for file with " << coedges.size() << " coedges" << std::endl;

        // 2. 计算LCS变换矩阵
        std::vector<Tensor> lcs_invs;
        compute_all_lcs_matrices(lcs_invs);

        // 3. 生成各类LocalGrids
        generate_coedge_local_grids(lcs_invs);
        generate_face_local_grids(lcs_invs);
        generate_edge_local_grids();

        // 4. 诊断调用（仅在 export 模式下）
        if (EXPORT_ENABLED) {
            DBG_PRINTF("[DIAGNOSTIC] Calling face 19/23/26 grid diagnosis...\n");
            diagnose_face_19_23_26_grids(
                coedges,
                FaceGridsLocal,
                EdgeGridsLocal,
                unique_faces.Size(),
                unique_edges.Size()
            );
            DBG_PRINTF("[DIAGNOSTIC] Face grid diagnosis complete\n");
        }
    }
};

// ===== 【方案三诊断函数实现】 =====

void init_diagnostics() {
    if (!EXPORT_ENABLED) return;
    if (g_diag_face_grid) fclose(g_diag_face_grid);
    if (g_diag_arc_length) fclose(g_diag_arc_length);
    g_diag_face_grid = nullptr;
    g_diag_arc_length = nullptr;
    DBG_PRINTF("[DIAGNOSTIC] Diagnostics initialized\n");
}

void close_diagnostics() {
    if (g_diag_face_grid) {
        fprintf(g_diag_face_grid, "\n========================================\n");
        fprintf(g_diag_face_grid, "End of Grid Diagnosis\n");
        fprintf(g_diag_face_grid, "========================================\n");
        fclose(g_diag_face_grid);
        g_diag_face_grid = nullptr;
        DBG_PRINTF("[DIAGNOSTIC] Grid diagnosis saved to coedge_grid_diagnosis.txt\n");
    }
    if (g_diag_arc_length) {
        fprintf(g_diag_arc_length, "\n========================================\n");
        fprintf(g_diag_arc_length, "End of Arc-Length Diagnosis\n");
        fprintf(g_diag_arc_length, "========================================\n");
        fclose(g_diag_arc_length);
        g_diag_arc_length = nullptr;
        DBG_PRINTF("[DIAGNOSTIC] Arc-length diagnosis saved to arc_length_diagnosis.txt\n");
    }
}

// 【诊断函数1】Face 19/23/26的Grid数据和拓扑诊断
void diagnose_face_19_23_26_grids(
    const std::vector<CoedgeInfo>& coedges,
    const Tensor& FaceGridsLocal,
    const Tensor& EdgeGridsLocal,
    int num_faces,
    int num_edges) {

    if (!EXPORT_ENABLED) return;

    if (!g_diag_face_grid) {
        g_diag_face_grid = fopen("coedge_grid_diagnosis.txt", "w");
        if (!g_diag_face_grid) {
            ERR_LOG << "[ERROR] Cannot create coedge_grid_diagnosis.txt" << std::endl;
            return;
        }
    }

    fprintf(g_diag_face_grid, "========================================\n");
    fprintf(g_diag_face_grid, "Face 19, 23, 26 Grid Diagnosis\n");
    fprintf(g_diag_face_grid, "Total coedges in model: %zu\n", coedges.size());
    fprintf(g_diag_face_grid, "========================================\n\n");

    int target_faces[] = {19, 23, 26};
    int num_targets = 3;

    for (int t = 0; t < num_targets; t++) {
        int face_idx = target_faces[t];

        fprintf(g_diag_face_grid, "\n========================================\n");
        fprintf(g_diag_face_grid, "FACE %d ANALYSIS\n", face_idx);
        fprintf(g_diag_face_grid, "========================================\n\n");

        // 找出该face的所有coedges
        std::vector<int> face_coedges;
        for (size_t i = 0; i < coedges.size(); i++) {
            if (coedges[i].face_idx == face_idx) {
                face_coedges.push_back(i);
            }
        }

        fprintf(g_diag_face_grid, "【Coedge Information】\n");
        fprintf(g_diag_face_grid, "Total coedges for this face: %zu\n", face_coedges.size());

        for (size_t i = 0; i < face_coedges.size(); i++) {
            int coedge_idx = face_coedges[i];
            int edge_idx = coedges[coedge_idx].edge_idx;
            int mate_idx = coedges[coedge_idx].mate_idx;

            fprintf(g_diag_face_grid, "  [%zu] Coedge %d: edge_idx=%d, mate_idx=%d\n",
                    i, coedge_idx, edge_idx, mate_idx);
        }
        fprintf(g_diag_face_grid, "\n");

        // Grid数据统计
        fprintf(g_diag_face_grid, "【Grid Data Statistics】\n");

        for (size_t i = 0; i < face_coedges.size(); i++) {
            int coedge_idx = face_coedges[i];
            int edge_idx = coedges[coedge_idx].edge_idx;

            fprintf(g_diag_face_grid, "  Coedge %d (edge %d):\n", coedge_idx, edge_idx);

            // Face Grids (2个面)
            for (int f = 0; f < 2; f++) {
                if (coedge_idx >= (int)FaceGridsLocal.sizes()[0]) {
                    fprintf(g_diag_face_grid, "    Face Grid %d: COEDGE OUT OF RANGE\n", f + 1);
                    continue;
                }

                Tensor coedge_grids = get_slice(FaceGridsLocal, coedge_idx);
                if (f >= (int)coedge_grids.sizes()[0]) {
                    fprintf(g_diag_face_grid, "    Face Grid %d: FACE INDEX OUT OF RANGE\n", f + 1);
                    continue;
                }

                Tensor grid = get_slice(coedge_grids, f);

                if (grid.numel() == 0) {
                    fprintf(g_diag_face_grid, "    Face Grid %d: EMPTY\n", f + 1);
                    continue;
                }

                float* grid_ptr = grid.data_ptr<float>();
                int total = grid.numel();

                double sum = 0, min_v = 1e9, max_v = -1e9;
                for (int j = 0; j < total; j++) {
                    float val = grid_ptr[j];
                    sum += val;
                    if (val < min_v) min_v = val;
                    if (val > max_v) max_v = val;
                }
                double mean = sum / total;

                fprintf(g_diag_face_grid, "    Face Grid %d: Mean=%.6f, Min=%.6f, Max=%.6f, Range=%.6f\n",
                        f + 1, mean, min_v, max_v, max_v - min_v);
            }

            // Edge Grid
            if (edge_idx < (int)EdgeGridsLocal.sizes()[0]) {
                Tensor grid = get_slice(EdgeGridsLocal, edge_idx);

                if (grid.numel() > 0) {
                    float* grid_ptr = grid.data_ptr<float>();
                    int total = grid.numel();

                    double sum = 0, min_v = 1e9, max_v = -1e9;
                    for (int j = 0; j < total; j++) {
                        float val = grid_ptr[j];
                        sum += val;
                        if (val < min_v) min_v = val;
                        if (val > max_v) max_v = val;
                    }
                    double mean = sum / total;

                    fprintf(g_diag_face_grid, "    Edge Grid: Mean=%.6f, Min=%.6f, Max=%.6f, Range=%.6f\n",
                            mean, min_v, max_v, max_v - min_v);
                }
            }
        }
        fprintf(g_diag_face_grid, "\n");
    }

    fflush(g_diag_face_grid);
}

// ===== 诊断函数实现结束 =====