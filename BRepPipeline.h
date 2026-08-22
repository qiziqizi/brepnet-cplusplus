#pragma once

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <array>
#include <cmath>

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

class BRepPipeline {
public:
    TopTools_IndexedMapOfShape unique_faces;
    TopTools_IndexedMapOfShape unique_edges;
    std::vector<CoedgeInfo> coedges;



    BRepPipeline() {}

    ~BRepPipeline() {}


    // --- 主处理流程 ---
    bool process(const std::string& step_file_path) {

        coedges.clear();
        unique_faces.Clear();
        unique_edges.Clear();

        // 2. 读取 STEP
        STEPControl_Reader reader;
        IFSelect_ReturnStatus status = reader.ReadFile(step_file_path.c_str());
        int num_roots = reader.NbRootsForTransfer();
        reader.TransferRoots();
        TopoDS_Shape original_shape = reader.OneShape();

        // 使用原始 STEP 坐标（与 Python 行为一致，不做缩放）
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
        generate_local_grids();
        return true;
    }

    Tensor FaceGridsGlobal; // 存储提取的全局 Grid 数据 [N, 9, 20, 20]
    std::vector<std::array<double, 3>> coedge_origins_; // 每条 coedge 的中点坐标（退化边存 {-2000,-2000,-2000}），double 精度与 Python float64 一致
    std::vector<bool> bool_array_;

    // 存储局部坐标系下的数据
    Tensor FaceGridsLocal;   // [N_c, 2, 9, 20, 20]
    Tensor CoedgeGridsLocal;  // [N_c, 9, 20, 20] mate_relative_face_grids，由 surface_encoder2 编码


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

    // 对应 python 的 extract_face_point_grid
    // =========================================================================

    // BRepPipeline.h: generate_global_face_grid()
    Tensor generate_global_face_grid(const TopoDS_Face& face) {
        int num_u = 20;
        int num_v = 20;

        // Shape: [9, num_u, num_v]
        Tensor grid = breptorch::zeros({ 9, num_u, num_v }, breptorch::kFloat32);

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

                // 使用 GeomLProp_SLProps 计算法线（与Python face.normal() 一致）
                // 必须带 location：BRep_Tool::Surface(face, loc) 返回局部坐标系表面，
                // 法线需经 loc.Transformation() 变换到全局坐标系（Python 端同样处理）
                TopLoc_Location loc;
                Handle(Geom_Surface) geom_surf = BRep_Tool::Surface(face, loc);
                GeomLProp_SLProps props(geom_surf, u, v, 1, 1e-9);

                gp_Vec n;
                if (props.IsNormalDefined()) {
                    gp_Dir normal = props.Normal();
                    n = gp_Vec(normal.XYZ());
                    if (!loc.IsIdentity()) {
                        n.Transform(loc.Transformation());
                    }
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
        // 与 Python has_curve() 判定一致：退化边（如球面极点）无有效3D曲线，
        // 返回全零矩阵使 compute_all_lcs_matrices 中 det<1e-3 判定为无效，
        // bool_array_=false，对应网格被清零（Python 端同样清零）
        if (BRep_Tool::Degenerated(edge) || curve.IsNull()) {
            coedge_origins_[coedge_idx] = {-2000.0, -2000.0, -2000.0};
            return breptorch::zeros({4, 4});
        }

        double u_mid = compute_arc_length_midpoint(curve, u0, u1);

        gp_Pnt p;
        gp_Vec tangent;

        curve->D1(u_mid, p, tangent);
        coedge_origins_[coedge_idx] = {p.X(), p.Y(), p.Z()};

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

        // V4: u=normal, v=tangent(unprojected), w=cross(u,v)
        float u_norm = sqrt(n_arr[0]*n_arr[0] + n_arr[1]*n_arr[1] + n_arr[2]*n_arr[2]);
        if (u_norm < 1e-10f) u_norm = 1e-10f;
        float u_vec[3] = { n_arr[0]/u_norm, n_arr[1]/u_norm, n_arr[2]/u_norm };

        float v_norm = sqrt(t_arr[0]*t_arr[0] + t_arr[1]*t_arr[1] + t_arr[2]*t_arr[2]);
        if (v_norm < 1e-10f) v_norm = 1e-10f;
        float v_vec[3] = { t_arr[0]/v_norm, t_arr[1]/v_norm, t_arr[2]/v_norm };

        // Check if tangent is parallel to normal (projection fails)
        float cross_check = sqrt(
            (u_vec[1]*v_vec[2] - u_vec[2]*v_vec[1]) * (u_vec[1]*v_vec[2] - u_vec[2]*v_vec[1]) +
            (u_vec[2]*v_vec[0] - u_vec[0]*v_vec[2]) * (u_vec[2]*v_vec[0] - u_vec[0]*v_vec[2]) +
            (u_vec[0]*v_vec[1] - u_vec[1]*v_vec[0]) * (u_vec[0]*v_vec[1] - u_vec[1]*v_vec[0])
        );
        if (cross_check < 1e-6f) {
            // Return zero matrix (singular)
            return breptorch::zeros({4, 4});
        }

        // w = cross(u, v)
        float w_vec[3] = {
            u_vec[1] * v_vec[2] - u_vec[2] * v_vec[1],
            u_vec[2] * v_vec[0] - u_vec[0] * v_vec[2],
            u_vec[0] * v_vec[1] - u_vec[1] * v_vec[0]
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

            if (std::abs(breptorch::det(mat)) < 1e-3) {
                mat = breptorch::eye(4);
                bool_array_.push_back(false);
            } else {
                bool_array_.push_back(true);
            }

            Tensor mat_inv = breptorch::inverse(mat);

            lcs_invs.push_back(mat_inv);
        }
    }

    // 生成Coedge局部网格
    void generate_coedge_local_grids(const std::vector<Tensor>& lcs_invs) {
        int num_c = coedges.size();
        std::vector<Tensor> c_list;
        c_list.reserve(num_c);

        // V4: CoedgeGridsLocal = mate_relative_face_grids [9, 20, 20]
        // mate face's global grid transformed to CURRENT coedge's LCS, × bool_array
        for (int i = 0; i < num_c; ++i) {
            Tensor mate_grid = breptorch::zeros({ 9, 20, 20 }, breptorch::kFloat32);

            int mate_idx = coedges[i].mate_idx;
            if (mate_idx != -1) {
                int mf_idx = coedges[mate_idx].face_idx;
                if (FaceGridsGlobal.defined() && mf_idx < FaceGridsGlobal.size(0)) {
                    bool is_degenerate_m = (coedge_origins_[mate_idx][0] <= -1000.0);
                    bool is_valid = bool_array_.size() > (size_t)i ? bool_array_[i] : true;
                    if (!is_degenerate_m && is_valid) {
                        Tensor global_grid_m = get_slice(FaceGridsGlobal, mf_idx);  // [9, 20, 20]
                        // Transform with CURRENT coedge's LCS (not mate's)
                        mate_grid = transform_grid_to_local(global_grid_m, lcs_invs[i], true);
                    }
                }
            }
            c_list.push_back(mate_grid);
        }

        CoedgeGridsLocal = breptorch::stack(c_list);
    }

    // 生成Face局部网格
    void generate_face_local_grids(std::vector<Tensor>& lcs_invs) {
        int num_c = coedges.size();
        std::vector<Tensor> f_list;
        f_list.reserve(num_c);

        const int samplesize = 20;

        for (int i = 0; i < num_c; ++i) {
            Tensor pair = breptorch::zeros({ 2, 9, samplesize, samplesize }, breptorch::kFloat32);

            // V4: no crop (already 20×20), with bool_array
            bool is_valid = bool_array_.size() > (size_t)i ? bool_array_[i] : true;

            // Left Face (parent face of coedge i) - use lcs_i, no crop
            int f_idx = coedges[i].face_idx;
            if (FaceGridsGlobal.defined() && f_idx < FaceGridsGlobal.size(0) && is_valid) {
                Tensor global_grid = get_slice(FaceGridsGlobal, f_idx);  // [9, 20, 20]
                Tensor t = transform_grid_to_local(global_grid, lcs_invs[i], true);
                set_slice(pair, 0, t);
            }

            // Right Face (mate face) - use lcs_mate, no crop
            int mate_idx = coedges[i].mate_idx;
            if (mate_idx != -1) {
                int mf_idx = coedges[mate_idx].face_idx;
                if (FaceGridsGlobal.defined() && mf_idx < FaceGridsGlobal.size(0) && is_valid) {
                    Tensor global_grid_m = get_slice(FaceGridsGlobal, mf_idx);  // [9, 20, 20]
                    Tensor tm = transform_grid_to_local(global_grid_m, lcs_invs[mate_idx], true);
                    set_slice(pair, 1, tm);
                }
            }

            f_list.push_back(pair);
        }

        FaceGridsLocal = breptorch::stack(f_list);
    }

    // 主入口：生成所有局部网格
    void generate_local_grids() {
        if (unique_faces.Extent() == 0) return;

        // 1. 生成全局Face网格
        generate_global_face_grids();

        int num_c = coedges.size();
        if (num_c == 0) return;

        // 2. 计算LCS变换矩阵
        std::vector<Tensor> lcs_invs;
        compute_all_lcs_matrices(lcs_invs);

        // 3. 生成各类LocalGrids
        generate_coedge_local_grids(lcs_invs);
        generate_face_local_grids(lcs_invs);

    }
};

