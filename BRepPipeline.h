#pragma once

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <algorithm> // ���� std::sort
#include <cmath>

// LibTorch
//#include <torch/torch.h>
#include "BRepTorch.h"
#include "cnpy.h"
#include "BRepUtils.h"

// OpenCascade ͷ�ļ�
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
#include <GeomAPI_ProjectPointOnSurf.hxx>
#include <BRepLProp_SLProps.hxx>

// extract_face_point_grids ���
#include <BRepTools.hxx>
#include <BRepTopAdaptor_FClass2d.hxx>    
#include <gp_Pnt2d.hxx>
#include <Precision.hxx>

// extract_face_point_grids ���
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


    // --- ������� ---
    bool process(const std::string& step_file_path) {
        coedges.clear();
        unique_faces.Clear();
        unique_edges.Clear();

        // 2. ��ȡ STEP
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
        std::cout << "[Debug] Loading stats from: " << npz_path << std::endl;
        try {
            cnpy::npz_t npz = cnpy::npz_load(npz_path);
            auto load_t = [&](const std::string& key) {
                if (!npz.count(key)) {
                    std::cerr << "Stats missing key: " << key << std::endl;
                    // ����һ���� tensor �����׳��쳣����ֹ�����
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
            std::cerr << " Load stats failed: " << e.what() << std::endl;
            has_stats = false; 
        }
    }

    // ��׼��
    void standardize() {
        if (!has_stats) {
            std::cout << "[Warn] No stats loaded, skipping standardization." << std::endl;
            return;
        }
        std::cout << "[Debug] Executing standardization..." << std::endl;
        if (Xf.size(0) > 1) Xf.sub_(mean_f).div_(std_f);
        if (Xe.size(0) > 1) Xe.sub_(mean_e).div_(std_e);
        if (Xc.size(0) > 1) Xc.sub_(mean_c).div_(std_c);
    }
    // �� BRepPipeline ��� public �������ӣ�

    Tensor FaceGridsGlobal; // �洢��ȡ������ Grid ���� [N, 7, 10, 10]
    Tensor EdgeGridsGlobal;
    Tensor CoedgeGridsGlobal;
    bool use_uvnet = false;

    // �������������� Grid ���� (�� Python ������ npz)
    void load_grids_from_npz(const std::string& npz_path) {
        try {
            cnpy::npz_t npz = cnpy::npz_load(npz_path);

            // 1. Face
            if (npz.count("face_point_grids")) {
                cnpy::NpyArray arr = npz["face_point_grids"];
                std::vector<int64_t> s; for (auto d : arr.shape) s.push_back(d);
                FaceGridsGlobal = breptorch::from_blob(arr.data<float>(), s, breptorch::kFloat32).clone();
                // û�в� Padding ���ⲿ���߼��� Forward ��������ȫ������ֻ�������
                use_uvnet = true;
                std::cout << "Loaded Face Grids: " << FaceGridsGlobal.sizes() << std::endl;
                //std::cout << FaceGridsGlobal[0][0] << std::endl;
            }

            // 2. Edge
            if (npz.count("edge_point_grids")) {
                cnpy::NpyArray arr = npz["edge_point_grids"];
                std::vector<int64_t> s; for (auto d : arr.shape) s.push_back(d);
                EdgeGridsGlobal = breptorch::from_blob(arr.data<float>(), s, breptorch::kFloat32).clone();
                std::cout << "Loaded Edge Grids: " << EdgeGridsGlobal.sizes() << std::endl;
                //std::cout << EdgeGridsGlobal[0] << std::endl;
            }

            // 3. Coedge
            if (npz.count("coedge_point_grids")) {
                cnpy::NpyArray arr = npz["coedge_point_grids"];
                std::vector<int64_t> s; for (auto d : arr.shape) s.push_back(d);
                CoedgeGridsGlobal = breptorch::from_blob(arr.data<float>(), s, breptorch::kFloat32).clone();
                std::cout << "Loaded Coedge Grids: " << CoedgeGridsGlobal.sizes() << std::endl;
            }
        }
        catch (const std::exception& e) {
                std::cerr << "Failed to load grids from npz: " << e.what() << std::endl;
        }
    }

    // �洢�ֲ�����ϵ�µ�����
    Tensor FaceGridsLocal;   // [N_c, 2, 9, 10, 10]
    Tensor EdgeGridsLocal;    //ʵ��û����
    Tensor CoedgeGridsLocal;  //ʵ��û����


private:
    void build_topology() {
        coedges.clear();
        std::map<int, std::vector<int>> edge_to_coedge_map;

        for (int f_idx = 1; f_idx <= unique_faces.Extent(); ++f_idx) {
            const TopoDS_Face& face = TopoDS::Face(unique_faces.FindKey(f_idx));

            // 调试：统计 Face 0 的 Wires 和 Edges
            if (f_idx == 1) {
                int wire_count = 0;
                TopExp_Explorer wireCounter(face, TopAbs_WIRE);
                for (; wireCounter.More(); wireCounter.Next()) wire_count++;
                std::cout << "[Debug Topology] Face 0 has " << wire_count << " wires" << std::endl;
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
                if (f_idx == 1) {
                    std::cout << "[Debug Topology] Face 0, Wire " << wire_idx << " has " << edge_count_in_wire << " edges" << std::endl;
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
        // 1. ��ʼ�� Face �������� (Xf)
        // ��ʦָʾ������Ҫȫ�ּ����������������ռλ��ȫ��Ϊ 0 �� 1
        // ά�ȱ��� 7 ��Ϊ�˼������е�Ȩ���ļ��ṹ
        Xf = breptorch::zeros({ num_f, 7 });

        // 2. ��ʼ�� Edge �������� (Xe)
        // ά�ȱ��� 10 �Լ���Ȩ��
        Xe = breptorch::zeros({ num_e, 10 });
        // 3. ��ʼ�� Coedge �������� (Xc)
        Xc = breptorch::zeros({ num_c, 1 });
        auto Xc_a = Xc.accessor<float, 2>();
        for (int i = 0; i < num_c; ++i) {
            // ����������һ�����Ϣ����Ϊ���������˽ṹ��һ����
            if (!coedges[i].orientation) Xc_a[i][0] = 1;
        }

        std::cout << "[Info] Simplified Feature Extraction (No Global Geom Stats)." << std::endl;
    }

    void generate_tensors() {
        // ��ȡ����
        int num_f = unique_faces.Extent();
        int num_e = unique_edges.Extent();
        int num_c = coedges.size();

        std::vector<int64_t> kf, ke, kc;
        // winged_edge.json��Ϊ simple_edge.json
        //std::vector<std::vector<int>> fw = { {},{1} }, ew = { {},{2},{3},{1,2},{1,3} }, cw = { {},{1},{2},{3},{1,2},{1,3} };
        std::vector<std::vector<int>> fw = { {},{1} }, ew = { {} }, cw = { {},{1} };
        // --- ���� Kf ---
        // Kf�ò�����
        // ���޸� 1�� ��Ҫ push_back(0) ��
        for (const auto& c : coedges) {
            for (auto& rule : fw) {
                int t = walk(c.id, rule);
                // ���޸� 2�� ��� t==-1 (���ھ�)���� num_f (��ΪԽ��ֵ)
                // ��� t!=-1���� coedges[t].face_idx (0-based)
                kf.push_back(t == -1 ? num_f : coedges[t].face_idx);
            }
        }
        // ��״��Ϊ {num_c, ...}
        Kf = breptorch::from_blob(kf.data(), { num_c, (long long)fw.size() }, breptorch::kLong).clone();

        // --- ���� Ke ---
        // ��Ҫ push_back(0)
        for (const auto& c : coedges) {
            for (auto& rule : ew) {
                int t = walk(c.id, rule);
                // ���ھӴ� num_e�����ھӴ� edge_idx
                ke.push_back(t == -1 ? num_e : coedges[t].edge_idx);
            }
        }
        Ke = breptorch::from_blob(ke.data(), { num_c, (long long)ew.size() }, breptorch::kLong).clone();

        // --- ���� Kc ---
        // ��Ҫ push_back(0)
        for (const auto& c : coedges) {
            for (auto& rule : cw) {
                int t = walk(c.id, rule);
                // ���ھӴ� num_c�����ھӴ� t (����ID��������0-based)
                kc.push_back(t == -1 ? num_c : t);
            }
        }
        Kc = breptorch::from_blob(kc.data(), { num_c, (long long)cw.size() }, breptorch::kLong).clone();

        // --- Pooling Ce ---
        std::vector<int64_t> ce(num_e * 2, num_c); // ��ʼ��Ϊ num_c (��Чֵ)
        std::vector<int> ec(num_e, 0);
        for (const auto& c : coedges) {
            // �� c.id (0-based)
            if (ec[c.edge_idx] < 2) ce[c.edge_idx * 2 + ec[c.edge_idx]++] = c.id;
        }
        Ce = breptorch::from_blob(ce.data(), { num_e, 2 }, breptorch::kLong).clone();

        // --- Pooling Cf (按照 Python 的方式：small faces + big faces) ---
        int max_cpf = 30;  // Python 的 max_coedges_per_face

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

        std::cout << "[Debug BRepPipeline] Small faces: " << small_face_indices.size()
                  << ", Big faces: " << big_face_indices.size() << std::endl;

        // 3. 构建 face_permutation (small faces 在前，big faces 在后)
        std::vector<int> face_permutation;
        face_permutation.insert(face_permutation.end(), small_face_indices.begin(), small_face_indices.end());
        face_permutation.insert(face_permutation.end(), big_face_indices.begin(), big_face_indices.end());

        // 4. 构建 Cf (只包含 small faces)
        int num_small_faces = small_face_indices.size();
        std::vector<int64_t> cf(num_small_faces * max_cpf, num_c); // 初始化为 num_c (无效值)

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
        std::cout << "[Debug BRepPipeline] Cf[0] corresponds to original Face " << small_face_indices[0] << std::endl;
        std::cout << "[Debug BRepPipeline] Cf[0] has " << face_to_coedges[small_face_indices[0]].size() << " coedges" << std::endl;
        std::cout << "[Debug BRepPipeline] Cf[0] coedge IDs (first 30): ";
        for (int i = 0; i < std::min(30, (int)face_to_coedges[small_face_indices[0]].size()); ++i) {
            std::cout << face_to_coedges[small_face_indices[0]][i] << " ";
        }
        std::cout << std::endl;

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

        std::cout << "[Debug BRepPipeline] Csf has " << Csf.size() << " big faces" << std::endl;
    }


    // ��Ӧ python �� extract_face_point_grid
    // =========================================================================

    // BRepPipeline.h: generate_global_face_grid()
    Tensor generate_global_face_grid(const TopoDS_Face& face) {
        int num_u = 10;
        int num_v = 10;

        // Shape: [9, 10, 10]
        Tensor grid = breptorch::zeros({ 9, num_u, num_v }, breptorch::kFloat32);

        static int debug_face_count = 0;
        bool debug_first_face = (debug_face_count == 0);
        bool debug_first_three = (debug_face_count < 3);

        int64_t stride_c = num_u * num_v;
        int64_t stride_h = num_v;

        Standard_Real umin, umax, vmin, vmax;
        BRepTools::UVBounds(face, umin, umax, vmin, vmax);
        BRepAdaptor_Surface surf(face);
        BRepTopAdaptor_FClass2d classifier(face, 0.0);

        // IMPORTANT: UV sampling direction depends on face orientation
        // Based on testing with Python output:
        // REVERSED faces: U from max to min, V from min to max
        // FORWARD faces: U from min to max, V from min to max (NOT max to min!)
        bool is_reversed = (face.Orientation() == TopAbs_REVERSED);
        bool u_reverse = is_reversed;   // REVERSED: max->min, FORWARD: min->max
        bool v_reverse = false;         // Both: min->max

        float* data = grid.data_ptr<float>();

        for (int i = 0; i < num_u; ++i) {
            for (int j = 0; j < num_v; ++j) {
                double u = BRepUtils::GetParamStrict(i, num_u, umin, umax, u_reverse);
                double v = BRepUtils::GetParamStrict(j, num_v, vmin, vmax, v_reverse);

                gp_Pnt p;
                gp_Vec d1u, d1v;
                surf.D1(u, v, p, d1u, d1v);

                gp_Vec n = d1u ^ d1v;
                if (n.Magnitude() > Precision::Confusion()) {
                    n.Normalize();
                }
                else {
                    n = gp_Vec(0, 0, 0);
                }

                if (face.Orientation() == TopAbs_REVERSED) {
                    n.Reverse();
                }

                gp_Pnt2d p2d(u, v);
                TopAbs_State state = classifier.Perform(p2d);
                float mask_val = (state == TopAbs_IN) ? 1.0f : 0.0f;

                bool is_on_border = (i == 0 || i == num_u - 1 || j == 0 || j == num_v - 1);
                if (is_on_border) {
                    mask_val = 0.0f;
                }

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

        // NOTE: Python does not flip REVERSED faces, so we don't either
        // This was causing data corruption issues

        return grid;
    }

    Tensor generate_global_coedge_grid(int coedge_idx) {
        const CoedgeInfo& c_info = coedges[coedge_idx];
        
        // 1. ��ȡ����ʵ��
        TopoDS_Face face_left = TopoDS::Face(unique_faces.FindKey(c_info.face_idx + 1));
        TopoDS_Edge edge = TopoDS::Edge(unique_edges.FindKey(c_info.edge_idx + 1));
        
        // ��ȡ Mate �� (Right Face)
        TopoDS_Face face_right;
        bool has_mate = (c_info.mate_idx != -1);
        if (has_mate) {
            int mate_face_idx = coedges[c_info.mate_idx].face_idx;
            face_right = TopoDS::Face(unique_faces.FindKey(mate_face_idx + 1));
        }

        // 2. ׼�� Tensor [12, 10] (������Ҫת��Ϊ [12, 10] ? Python �� [12, 10]���� PyTorch Conv1d ����ͨ���� [Batch, Channel, Length])
        // Python extract_coedge_point_grid ���ص��� np.transpose(single_grid, (1,0)) 
        // single_grid �� [10, 12]��ת�ú��� [12, 10]��
        // ����ֱ������ [12, 10]��
        int num_u = 10;
        //Tensor grid = torch::zeros({12, num_u}, torch::kFloat32);
        Tensor grid = breptorch::zeros({13, num_u}, breptorch::kFloat32);
        // auto accessor = grid.accessor<float, 2>();

        // 3. ����������
        BRepAdaptor_Curve curve_adaptor(edge);
        double first = curve_adaptor.FirstParameter();
        double last = curve_adaptor.LastParameter();
        double len = last - first;

        // 4. �Ȼ������� (Uniform Abscissa) - ģ�� Python use_arclength_params=True
        // �������̫�̻��˻������˵���������
        bool use_uniform = true;
        GCPnts_UniformAbscissa uniform_sampler;
        try {
            uniform_sampler.Initialize(curve_adaptor, num_u, -1); // -1 tol
            if (!uniform_sampler.IsDone()) use_uniform = false;
        } catch(...) { use_uniform = false; }

        // 5. ѭ������
        float* data = grid.data_ptr<float>();
        int64_t stride_c = num_u;


        for (int i = 0; i < num_u; ++i) {

            double param;
            if (use_uniform && uniform_sampler.NbPoints() >= num_u) {
                // GCPnts ���ɵ��ǵ�����������1��ʼ
                // ������Ҫ����ӳ��һ�£����߼򵥾��ȷֲ�����
                // Ϊ�˼����Ƚ������������ü򵥵Ĳ����ռ���Ȳ�����
                // ����ϸ�׷�󾫶ȣ������� GCPnts_UniformAbscissa �� Parameter(i+1)
                // ��Ҫע�� GCPnts �ĵ������ܲ���ȫ���� num_u
                 param = uniform_sampler.Parameter(i + 1);
            } else {
                // ���ˣ������ռ���Ȳ���
                param = first + (len * i) / (double)(num_u - 1);
            }

            gp_Pnt p;
            gp_Vec tangent;
            curve_adaptor.D1(param, p, tangent);
            
            // ��һ��������
            if (tangent.Magnitude() > 1e-7) tangent.Normalize();

            // �������߷��� (Orientation)
            // ��� Coedge �� Reversed��˵�������� Edge �ķ�������
            // ���ġ���������Ӧ��ȡ��
            if (!c_info.orientation) { // orientation == false means REVERSED
                tangent.Reverse();
            }
            // ע�⣺Python����������� Reversed�������б���Ҳ�Ƿ�����
            // ͨ�� Grid �ǰ�����˳���ģ��� Tangent ����� Coedge ���������
            // ��� Python ������ coedge_data �ƺ�������������⡣
            // �������Ǽ��� Grid ���ǰ����� U �����棬�� Tangent ���� Coedge��

            // ���㷨��
            gp_Vec n_left = BRepUtils::GetNormalAtPoint(face_left, p);
            gp_Vec n_right = (has_mate) ? BRepUtils::GetNormalAtPoint(face_right, p) : gp_Vec(0,0,0);
            // ��������Edge �ı�ԵĨ��

            // ���� Tensor
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

            // �޸ģ����������12ͨ������ӦPython��u_params��
            data[12 * stride_c + i] = (float)param; 
        }
        
        // ��� Coedge �Ƿ���ģ�Python �� EdgeDataExtractor ���ܻ�ѵ�����Ҳ��ת
        // (��: grid[][0] ���յ㣬grid[][9] �����)
        // Ϊ�˺� Python ����һ�£���� orientation �� false��������Ҫ flip dim 1
        if (!c_info.orientation) {
            grid = breptorch::flip(grid, {1});
        }

        return grid;
    }

    // �޸� compute_coedge_lcs (ʹ�þ�ȷ�е�)

    
    Tensor compute_coedge_lcs(int coedge_idx) {
        const CoedgeInfo& c_info = coedges[coedge_idx];
        TopoDS_Edge edge = TopoDS::Edge(unique_edges.FindKey(c_info.edge_idx + 1));
        TopoDS_Face face = TopoDS::Face(unique_faces.FindKey(c_info.face_idx + 1));

        // 1. ��ȡ���е㡢���ߡ�����
        double u0, u1;
        Handle(Geom_Curve) curve = BRep_Tool::Curve(edge, u0, u1);
        double u_mid = (u0 + u1) / 2.0;
        gp_Pnt p;
        gp_Vec tangent;
        curve->D1(u_mid, p, tangent);

        gp_Vec normal;
        BRepGProp_Face face_prop(face);
        gp_Pnt p_face;
        gp_Vec du, dv;
        Standard_Real uu, vv;
        GeomAPI_ProjectPointOnSurf proj(p, BRep_Tool::Surface(face));
        proj.LowerDistanceParameters(uu, vv);
        face_prop.Normal(uu, vv, p_face, normal);


        // �ж��Ƿ���
        gp_Vec t_vec = tangent;
        gp_Vec n_vec = normal;
        if (!c_info.orientation) {
            t_vec.Reverse();
        }

        // ת������������ֶ�����
        float p_arr[3] = { (float)p.X(), (float)p.Y(), (float)p.Z() };
        float t_arr[3] = { (float)t_vec.X(), (float)t_vec.Y(), (float)t_vec.Z() };
        float n_arr[3] = { (float)n_vec.X(), (float)n_vec.Y(), (float)n_vec.Z() };

        // 1. W �� (��������һ��)
        float w_norm = sqrt(n_arr[0] * n_arr[0] + n_arr[1] * n_arr[1] + n_arr[2] * n_arr[2]) + 1e-7f;
        float w_vec[3] = {
            n_arr[0] / w_norm,
            n_arr[1] / w_norm,
            n_arr[2] / w_norm
        };

        // 2. V �� (������ͶӰ����ֱ�� W ��ƽ��)
        float dot_tw = t_arr[0] * w_vec[0] + t_arr[1] * w_vec[1] + t_arr[2] * w_vec[2];
        float v_vec[3] = {
            t_arr[0] - dot_tw * w_vec[0],
            t_arr[1] - dot_tw * w_vec[1],
            t_arr[2] - dot_tw * w_vec[2]
        };

        // V ���һ��
        float v_norm = sqrt(v_vec[0] * v_vec[0] + v_vec[1] * v_vec[1] + v_vec[2] * v_vec[2]);

        // ��� V �᳤��̫С��˵�����߼���ƽ���ڷ��ߣ���Ҫ��һ����������
        if (v_norm < 1e-6f) {
            // ѡ��һ����ƽ���� W ������
            float temp[3] = { 1.0f, 0.0f, 0.0f };
            if (fabs(w_vec[0]) > 0.9f) {
                temp[0] = 0.0f;
                temp[1] = 1.0f;
                temp[2] = 0.0f;
            }

            // ͶӰ����ֱ�� W ��ƽ��
            float dot_temp_w = temp[0] * w_vec[0] + temp[1] * w_vec[1] + temp[2] * w_vec[2];
            v_vec[0] = temp[0] - dot_temp_w * w_vec[0];
            v_vec[1] = temp[1] - dot_temp_w * w_vec[1];
            v_vec[2] = temp[2] - dot_temp_w * w_vec[2];

            v_norm = sqrt(v_vec[0] * v_vec[0] + v_vec[1] * v_vec[1] + v_vec[2] * v_vec[2]);
        }

        v_vec[0] /= (v_norm + 1e-7f);
        v_vec[1] /= (v_norm + 1e-7f);
        v_vec[2] /= (v_norm + 1e-7f);

        // 3. U �� (V �� W)
        float u_vec[3] = {
            v_vec[1] * w_vec[2] - v_vec[2] * w_vec[1],
            v_vec[2] * w_vec[0] - v_vec[0] * w_vec[2],
            v_vec[0] * w_vec[1] - v_vec[1] * w_vec[0]
        };


        // 4. ��װ���� (�ֶ����)
        Tensor mat = breptorch::eye(4);
        float* mat_ptr = mat.data_ptr<float>();

        // �����ת���� (ǰ3x3) - ������
        mat_ptr[0 * 4 + 0] = u_vec[0];  mat_ptr[0 * 4 + 1] = v_vec[0];  mat_ptr[0 * 4 + 2] = w_vec[0];
        mat_ptr[1 * 4 + 0] = u_vec[1];  mat_ptr[1 * 4 + 1] = v_vec[1];  mat_ptr[1 * 4 + 2] = w_vec[1];
        mat_ptr[2 * 4 + 0] = u_vec[2];  mat_ptr[2 * 4 + 1] = v_vec[2];  mat_ptr[2 * 4 + 2] = w_vec[2];

        // ���ƽ�Ʋ��� (��4��)
        mat_ptr[0 * 4 + 3] = p_arr[0];
        mat_ptr[1 * 4 + 3] = p_arr[1];
        mat_ptr[2 * 4 + 3] = p_arr[2];

        return mat;
    }

    // ��Ӧ Python �� transform_face_point_grid_to_local
    // �� Grid (ȫ��) �任�� Local
    // grid: [Channels, H, W] �� [Channels, L]
    // lcs_inv: [4, 4] �����
    // is_face: true Ϊ FaceGrid (9ch), false Ϊ CoedgeGrid (13ch)
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
    

    // ����ȫ��Face����
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

    // ��������Coedge��LCS�任����
    void compute_all_lcs_matrices(std::vector<Tensor>& lcs_invs) {
        int num_c = coedges.size();
        lcs_invs.clear();
        lcs_invs.reserve(num_c);

        for (int i = 0; i < num_c; ++i) {
            Tensor mat = compute_coedge_lcs(i);
            //// ? ���ԣ���ӡǰ3������������
            //if (i < 3) {
            //    std::cout << "\n[Debug] Coedge " << i << ":\n";
            //    std::cout << "  Forward mat (det=" << breptorch::det(mat) << "):\n";
            //    float* m = const_cast<Tensor&>(mat).data_ptr<float>();
            //    for (int r = 0; r < 4; r++) {
            //        printf("    [%8.4f, %8.4f, %8.4f, %8.4f]\n",
            //            m[r * 4], m[r * 4 + 1], m[r * 4 + 2], m[r * 4 + 3]);
            //    }
            //}

            if (std::abs(breptorch::det(mat)) < 1e-6) {
                mat = breptorch::eye(4);
            }

            Tensor mat_inv = breptorch::inverse(mat);

            //// ? ���ԣ���ӡ�����
            //if (i < 3) {
            //    std::cout << "  Inverse mat:\n";
            //    float* m_inv = const_cast<Tensor&>(mat_inv).data_ptr<float>();
            //    for (int r = 0; r < 4; r++) {
            //        printf("    [%8.4f, %8.4f, %8.4f, %8.4f]\n",
            //            m_inv[r * 4], m_inv[r * 4 + 1], m_inv[r * 4 + 2], m_inv[r * 4 + 3]);
            //    }
            //}
            //if (std::abs(breptorch::det(mat)) < 1e-6) {
            //    mat = breptorch::eye(4);
            //}
            lcs_invs.push_back(breptorch::inverse(mat));
        }
    }

    // ����Coedge�ֲ�����
    void generate_coedge_local_grids(const std::vector<Tensor>& lcs_invs) {
        int num_c = coedges.size();
        std::vector<Tensor> c_list;
        c_list.reserve(num_c);

        // �����ɵ�CoedgeGridsGlobal����û��Ҫ����
        for (int i = 0; i < num_c; ++i) {
            Tensor g_global = generate_global_coedge_grid(i);
            Tensor g_local = transform_grid_to_local(g_global, lcs_invs[i], false);
            c_list.push_back(g_local);
        }

        CoedgeGridsLocal = breptorch::stack(c_list);
    }

    // ����Face�ֲ�����
    void generate_face_local_grids(std::vector<Tensor>& lcs_invs) {
        int num_c = coedges.size();
        std::vector<Tensor> f_list;
        f_list.reserve(num_c);

        for (int i = 0; i < num_c; ++i) {
            Tensor pair = breptorch::zeros({ 2, 9, 10, 10 }, breptorch::kFloat32);

            // Left Face
            int f_idx = coedges[i].face_idx;
            if (FaceGridsGlobal.defined() && f_idx < FaceGridsGlobal.size(0)) {
                Tensor global_grid = get_slice(FaceGridsGlobal, f_idx);
                Tensor t = transform_grid_to_local(global_grid, lcs_invs[i], true);
                set_slice(pair, 0, t);
            }

            // Right Face (Mate)
            int mate_idx = coedges[i].mate_idx;
            if (mate_idx != -1) {
                int mf_idx = coedges[mate_idx].face_idx;
                if (FaceGridsGlobal.defined() && mf_idx < FaceGridsGlobal.size(0)) {
                    Tensor t = transform_grid_to_local(get_slice(FaceGridsGlobal, mf_idx), lcs_invs[mate_idx], true);
                    set_slice(pair, 1, t);
                }
            }

            f_list.push_back(pair);
        }

        FaceGridsLocal = breptorch::stack(f_list);
    }

    // ����Edge�ֲ�����
    void generate_edge_local_grids() {
        int num_e = unique_edges.Extent();
        if (num_e == 0 || !CoedgeGridsLocal.defined()) return;

        // ����Edge��Coedgeӳ���
        std::vector<int> edge_representatives(num_e, -1);
        for (const auto& c : coedges) {
            int eid = c.edge_idx;
            if (eid >= 0 && eid < num_e) {
                if (edge_representatives[eid] == -1 || c.orientation == true) {
                    edge_representatives[eid] = c.id;
                }
            }
        }

        // ����Edge����
        std::vector<Tensor> e_list;
        e_list.reserve(num_e);
        for (int i = 0; i < num_e; ++i) {
            int cid = edge_representatives[i];
            if (cid != -1) {
                e_list.push_back(get_slice(CoedgeGridsLocal, cid));
            }
            else {
                e_list.push_back(breptorch::zeros({ 13, 10 }, CoedgeGridsLocal.options()));
            }
        }

        EdgeGridsLocal = breptorch::stack(e_list);
    }

    // ���������������оֲ�����
    // ֮ǰ�ṹ���ң����޸�
    void generate_local_grids() {
        if (unique_faces.Extent() == 0) return;

        // 1. ����ȫ��Face����
        generate_global_face_grids();

        int num_c = coedges.size();
        if (num_c == 0) return;

        std::cout << "Generating local coordinate system features (LCS Transformation)..." << std::endl;

        // 2. ����LCS�任����
        std::vector<Tensor> lcs_invs;
        compute_all_lcs_matrices(lcs_invs);

        // 3. ��������LocalGrids
        generate_coedge_local_grids(lcs_invs);
        generate_face_local_grids(lcs_invs);
        generate_edge_local_grids();

        // 4. ���ͳ����Ϣ
        //std::cout << " Local Features Generated." << std::endl;
        //if (FaceGridsLocal.defined()) std::cout << "   Face: " << FaceGridsLocal.sizes() << std::endl;
        //if (CoedgeGridsLocal.defined()) std::cout << "   Coedge: " << CoedgeGridsLocal.sizes() << std::endl;
        //if (EdgeGridsLocal.defined()) std::cout << "   Edge: " << EdgeGridsLocal.sizes() << std::endl;
        //std::cout << "\n[Debug] FaceGridsGlobal Summary:\n";
        //for (int f = 0; f < std::min(3, (int)FaceGridsGlobal.size(0)); f++) {
        //    std::cout << "  Face[" << f << "] point[0,5,5]=" << FaceGridsGlobal.at({ f, 0, 5, 5 })
        //        << " normal[3,5,5]=" << FaceGridsGlobal.at({ f, 3, 5, 5 }) << "\n";
        //}

    }
};
