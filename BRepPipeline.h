#pragma once

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <fstream>
#include <filesystem>

// LibTorch
//#include <torch/torch.h>
#include "BRepTorch.h"
#include "cnpy.h"
#include "BRepUtils.h"
#include "DebugControl.h"
#include "VersionConfig.h"

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

    Tensor FaceGridsGlobal; // 存储提取的全局 Grid 数据 [N, 9, 40, 40]
    std::vector<std::vector<std::array<double, 3>>> face_grid_coords64_; // [N, 40*40] 每条 face 的 double 精度 xyz 坐标（仅用于 crop 搜索，与 Python float64 一致）
    std::vector<std::array<float, 3>> coedge_origins_; // 每条 coedge 的中点坐标（退化边存 {-2000,-2000,-2000}）
#if BREPNET_VERSION == 4
    std::vector<bool> bool_array_;
#endif

    // 存储局部坐标系下的数据
    Tensor FaceGridsLocal;   // [N_c, 2, 9, 20, 20]
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

    // 对应 python 的 extract_face_point_grid
    // =========================================================================

    // BRepPipeline.h: generate_global_face_grid()
    Tensor generate_global_face_grid(const TopoDS_Face& face) {
#if BREPNET_VERSION == 4
        int num_u = 20;
        int num_v = 20;
#else
        int num_u = 40;
        int num_v = 40;
#endif

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

#if BREPNET_VERSION == 4
        // V4: 20 samples (curve data, but not used for CoedgeGridsLocal in V4)
        int num_u = 20;
#else
        // V123: 40 samples
        int num_u = 40;
#endif
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

#if BREPNET_VERSION == 4
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
#else
        // V123: w=normal, v=projected_tangent, u=cross(v,w)
        float w_norm = sqrt(n_arr[0] * n_arr[0] + n_arr[1] * n_arr[1] + n_arr[2] * n_arr[2]);
        if (w_norm < 1e-10f) w_norm = 1e-10f;

        float w_vec[3] = {
            n_arr[0] / w_norm,
            n_arr[1] / w_norm,
            n_arr[2] / w_norm
        };

        float dot_tw = t_arr[0] * w_vec[0] + t_arr[1] * w_vec[1] + t_arr[2] * w_vec[2];
        float v_vec[3] = {
            t_arr[0] - dot_tw * w_vec[0],
            t_arr[1] - dot_tw * w_vec[1],
            t_arr[2] - dot_tw * w_vec[2]
        };

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

        if (v_norm < 1e-10f) v_norm = 1e-10f;
        v_vec[0] /= v_norm;
        v_vec[1] /= v_norm;
        v_vec[2] /= v_norm;

        float u_vec[3] = {
            v_vec[1] * w_vec[2] - v_vec[2] * w_vec[1],
            v_vec[2] * w_vec[0] - v_vec[0] * w_vec[2],
            v_vec[0] * w_vec[1] - v_vec[1] * w_vec[0]
        };
#endif


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
        face_grid_coords64_.clear();
        face_grid_coords64_.resize(num_faces);

        for (int i = 1; i <= num_faces; ++i) {
            const TopoDS_Face& face = TopoDS::Face(unique_faces.FindKey(i));
            Tensor single_grid = generate_global_face_grid(face);
            grids_list.push_back(single_grid.clone());
        }

        if (!grids_list.empty()) {
            this->FaceGridsGlobal = breptorch::stack(grids_list);
        }
    }

    // 生成每条 face 的 double 精度 xyz 坐标（[num_u*num_v, 3]），用于 crop 搜索
    // 注意：必须与 generate_global_face_grid 中相同的采样顺序 (i,j) -> u,v
    void generate_face_grid_coords64() {
        if (unique_faces.Extent() == 0) return;
        int num_faces = unique_faces.Extent();
        face_grid_coords64_.clear();
        face_grid_coords64_.resize(num_faces);

        for (int i = 1; i <= num_faces; ++i) {
            const TopoDS_Face& face = TopoDS::Face(unique_faces.FindKey(i));
            face_grid_coords64_[i-1] = generate_face_grid_coords64_single(face);
        }
    }

    std::vector<std::array<double, 3>> generate_face_grid_coords64_single(const TopoDS_Face& face) {
        int num_u = 40, num_v = 40;
        std::vector<std::array<double, 3>> coords;
        coords.reserve(num_u * num_v);

        Standard_Real umin, umax, vmin, vmax;
        BRepTools::UVBounds(face, umin, umax, vmin, vmax);
        BRepAdaptor_Surface surf(face);

        bool is_reversed = (face.Orientation() == TopAbs_REVERSED);
        bool u_reverse = is_reversed;
        bool v_reverse = false;

        for (int i = 0; i < num_u; ++i) {
            for (int j = 0; j < num_v; ++j) {
                double u = BRepUtils::GetParamStrict(i, num_u, umin, umax, u_reverse);
                double v = BRepUtils::GetParamStrict(j, num_v, vmin, vmax, v_reverse);
                gp_Pnt p;
                gp_Vec d1u, d1v;
                surf.D1(u, v, p, d1u, d1v);
                coords.push_back({p.X(), p.Y(), p.Z()});
            }
        }
        return coords;
    }

    // 计算所有Coedge的LCS变换矩阵
    void compute_all_lcs_matrices(std::vector<Tensor>& lcs_invs) {
        int num_c = coedges.size();
        lcs_invs.clear();
        lcs_invs.reserve(num_c);
        coedge_origins_.resize(num_c);

        for (int i = 0; i < num_c; ++i) {
            Tensor mat = compute_coedge_lcs(i);

#if BREPNET_VERSION == 4
            if (std::abs(breptorch::det(mat)) < 1e-3) {
                mat = breptorch::eye(4);
                bool_array_.push_back(false);
            } else {
                bool_array_.push_back(true);
            }
#else
            if (std::abs(breptorch::det(mat)) < 1e-10) {
                mat = breptorch::eye(4);
            }
#endif

            Tensor mat_inv = breptorch::inverse(mat);

            lcs_invs.push_back(mat_inv);
        }
    }

    // 生成Coedge局部网格
    void generate_coedge_local_grids(const std::vector<Tensor>& lcs_invs) {
        int num_c = coedges.size();
        std::vector<Tensor> c_list;
        c_list.reserve(num_c);

#if BREPNET_VERSION == 4
        // V4: CoedgeGridsLocal = mate_relative_face_grids [9, 20, 20]
        // mate face's global grid transformed to CURRENT coedge's LCS, × bool_array
        for (int i = 0; i < num_c; ++i) {
            Tensor mate_grid = breptorch::zeros({ 9, 20, 20 }, breptorch::kFloat32);

            int mate_idx = coedges[i].mate_idx;
            if (mate_idx != -1) {
                int mf_idx = coedges[mate_idx].face_idx;
                if (FaceGridsGlobal.defined() && mf_idx < FaceGridsGlobal.size(0)) {
                    bool is_degenerate_m = (coedge_origins_[mate_idx][0] <= -1000.0f);
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
#else
        // V123: CoedgeGridsLocal = curve data [13, 40]
        for (int i = 0; i < num_c; ++i) {
            Tensor g_global = generate_global_coedge_grid(i);
            Tensor g_local = transform_grid_to_local(g_global, lcs_invs[i], false);
            c_list.push_back(g_local);
        }
#endif

        CoedgeGridsLocal = breptorch::stack(c_list);
    }

    // ---- Face Grid 裁剪辅助结构（与 Python new/ 版本对应）----
    struct CropRange { int row_min, row_max, col_min, col_max; int best_row = 0, best_col = 0; };

    // 在 40×40 全局 face grid 中找离 target_point 最近的格点，
    // 以该点为中心计算 20×20 裁剪范围（精确复刻 Python new/ 逻辑）。
    // 使用 double 精度坐标（face_grid_coords64_）搜索，与 Python float64 行为一致。
    // 退化边（target_point[0] <= -1000）返回左上角 {0,20,0,20}。
    CropRange compute_crop_range(int face_idx,
                                  const std::array<float, 3>& target_point)
    {
        const int grid_h = 40, grid_w = 40, samplesize = 20;

        if (target_point[0] <= -1000.0f) {
            return {0, samplesize, 0, samplesize};
        }

        // 从 double 精度坐标中搜索（行优先，与 Python points_flat 的 reshape 顺序一致）
        const auto& coords = face_grid_coords64_[face_idx];
        double min_dist_sq = std::numeric_limits<double>::max();
        int best_row = 0, best_col = 0;
        for (int i = 0; i < grid_h; ++i) {
            for (int j = 0; j < grid_w; ++j) {
                int idx = i * grid_w + j;
                double dx = coords[idx][0] - (double)target_point[0];
                double dy = coords[idx][1] - (double)target_point[1];
                double dz = coords[idx][2] - (double)target_point[2];
                double d2 = dx*dx + dy*dy + dz*dz;
                // 复刻 np.argmin：严格小于（保留第一个出现的全局最小值）
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
        return {row_min, row_max, col_min, col_max, best_row, best_col};
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

    // 生成Face局部网格
    void generate_face_local_grids(std::vector<Tensor>& lcs_invs) {
        int num_c = coedges.size();
        std::vector<Tensor> f_list;
        f_list.reserve(num_c);

        const int samplesize = 20;

        // ---- 调试导出：裁剪窗口 + 原点 + LCS旋转矩阵（仅 --debug/--export 时开启）----
        std::ofstream crop_debug;
        if (EXPORT_ENABLED) {
            std::filesystem::create_directories("cpp_facecrop_debug");
            std::string fname = "cpp_facecrop_debug/" + DebugControl::instance().current_file + "_facecrop.txt";
            crop_debug.open(fname);
            if (crop_debug.is_open()) {
                crop_debug << "\xEF\xBB\xBF";
                crop_debug << std::scientific << std::setprecision(20);
                crop_debug << "coedge_idx slot origin_x origin_y origin_z best_row best_col "
                              "row_min row_max col_min col_max R00 R01 R02 R10 R11 R12 R20 R21 R22\n";
            }
        }
        // 写一行裁剪调试记录。lcs_inv 为 inverse-LCS(global->local)，取其 3x3 旋转部分（实际施加的旋转）。
        auto write_crop_row = [&](std::ofstream& os, int ci, int slot,
                                  const std::array<float, 3>& origin,
                                  const CropRange& cr,
                                  Tensor& lcs_inv) {
            if (!os.is_open()) return;
            const float* m = lcs_inv.data_ptr<float>();
            os << ci << " " << slot << " "
               << origin[0] << " " << origin[1] << " " << origin[2] << " "
               << cr.best_row << " " << cr.best_col << " "
               << cr.row_min << " " << cr.row_max << " " << cr.col_min << " " << cr.col_max << " "
               << m[0] << " " << m[1] << " " << m[2] << " "
               << m[4] << " " << m[5] << " " << m[6] << " "
               << m[8] << " " << m[9] << " " << m[10] << "\n";
        };

        for (int i = 0; i < num_c; ++i) {
            Tensor pair = breptorch::zeros({ 2, 9, samplesize, samplesize }, breptorch::kFloat32);

#if BREPNET_VERSION == 4
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
#else
            // V123: crop from 40×40 to 20×20
            // Left Face (parent face of coedge i)
            int f_idx = coedges[i].face_idx;
            if (FaceGridsGlobal.defined() && f_idx < FaceGridsGlobal.size(0)) {
                bool is_degenerate = (coedge_origins_[i][0] <= -1000.0f);
                if (!is_degenerate) {
                    Tensor global_grid = get_slice(FaceGridsGlobal, f_idx);  // [9, 40, 40]
                    CropRange cr = compute_crop_range(f_idx, coedge_origins_[i]);
                    Tensor t = transform_grid_to_local(global_grid, lcs_invs[i], true);
                    set_slice(pair, 0, crop_face_grid(t, cr, samplesize));
                    write_crop_row(crop_debug, i, 0, coedge_origins_[i], cr, lcs_invs[i]);
                }
            }

            // Right Face (mate face)
            int mate_idx = coedges[i].mate_idx;
            if (mate_idx != -1) {
                int mf_idx = coedges[mate_idx].face_idx;
                if (FaceGridsGlobal.defined() && mf_idx < FaceGridsGlobal.size(0)) {
                    bool is_degenerate_m = (coedge_origins_[mate_idx][0] <= -1000.0f);
                    if (!is_degenerate_m) {
                        Tensor global_grid_m = get_slice(FaceGridsGlobal, mf_idx);
                        CropRange cr_m = compute_crop_range(mf_idx, coedge_origins_[mate_idx]);
                        Tensor tm = transform_grid_to_local(global_grid_m, lcs_invs[mate_idx], true);
                        set_slice(pair, 1, crop_face_grid(tm, cr_m, samplesize));
                        write_crop_row(crop_debug, i, 1, coedge_origins_[mate_idx], cr_m, lcs_invs[mate_idx]);
                    }
                }
            }
#endif

            f_list.push_back(pair);
        }

        if (crop_debug.is_open()) crop_debug.close();
        FaceGridsLocal = breptorch::stack(f_list);
    }

    // 主入口：生成所有局部网格
    void generate_local_grids() {
        if (unique_faces.Extent() == 0) return;

        // 1. 生成全局Face网格
        generate_global_face_grids();
        // 1.1 生成 double 精度坐标（用于 crop 搜索，与 Python float64 一致）
        generate_face_grid_coords64();

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

