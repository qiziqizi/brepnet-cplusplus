#pragma once

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <algorithm> // 用于 std::sort
#include <cmath>

// LibTorch
//#include <torch/torch.h>
#include "BRepTorch.h"
#include "cnpy.h"

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
#include <GeomAPI_ProjectPointOnSurf.hxx>
#include <BRepLProp_SLProps.hxx>

// extract_face_point_grids 相关
#include <BRepTools.hxx>
#include <BRepTopAdaptor_FClass2d.hxx>    
#include <gp_Pnt2d.hxx>
#include <Precision.hxx>

// extract_face_point_grids 相关
#include <GCPnts_UniformAbscissa.hxx>
#include <GeomLProp_SLProps.hxx>

//using namespace torch;
using namespace breptorch;

namespace BRepUtils {
	// --- 数学辅助函数 ---
    // 
    // 辅助：计算投影
    Tensor ProjectVector(Tensor vec, Tensor target_plane_normal) {
        // vec: [3], normal: [3]
        // v_proj = v - (v . n) * n
        float dp = dot(vec, target_plane_normal);
        Tensor delta = dp * target_plane_normal;
        Tensor res = vec - delta;
        float len = norm(res);
        if (len < 1e-7) return Tensor(); // 失败
        return res / len; // 归一化
    }

    // 辅助：找任意正交向量
    Tensor AnyOrthogonalTensor(Tensor vec) {
        Tensor res = ProjectVector(tensor({ 1.0, 0.0, 0.0 }, vec.options()), vec);
        if (res.defined()) return res;
        res = ProjectVector(tensor({ 0.0, 1.0, 0.0 }, vec.options()), vec);
        if (res.defined()) return res;
        return ProjectVector(tensor({ 0.0, 0.0, 1.0 }, vec.options()), vec);
    }

    // 辅助：严格的线性插值，保证首尾精确命中边界
    double GetParamStrict(int index, int total, double min_val, double max_val) {
        if (index == 0) return min_val;
        if (index == total - 1) return max_val;
        return min_val + (max_val - min_val) * (double)index / (double)(total - 1);
    }
	// --- 几何辅助函数 ---

    // --- 辅助：计算几何属性 ---
    double GetFaceArea(const TopoDS_Shape& face) {
        GProp_GProps props;
        BRepGProp::SurfaceProperties(face, props);
        return props.Mass();
    }
    double GetEdgeLength(const TopoDS_Shape& edge) {
        GProp_GProps props;
        BRepGProp::LinearProperties(edge, props);
        return props.Mass();
    }
    TopoDS_Shape ScaleShape(const TopoDS_Shape& s) {
        Bnd_Box box;
        BRepBndLib::Add(s, box);
        if (box.IsVoid()) return s;
        double xmin, ymin, zmin, xmax, ymax, zmax;
        box.Get(xmin, ymin, zmin, xmax, ymax, zmax);
        double max_dim = std::max({ xmax - xmin, ymax - ymin, zmax - zmin });
        if (max_dim < 1e-7) return s;

        gp_Trsf t;
        t.SetTranslation(gp_Pnt((xmin + xmax) * 0.5, (ymin + ymax) * 0.5, (zmin + zmax) * 0.5), gp_Pnt(0, 0, 0));
        gp_Trsf sc;
        sc.SetScale(gp_Pnt(0, 0, 0), 2.0 / max_dim);
        return BRepBuilderAPI_Transform(s, sc * t, Standard_True).Shape();
    }

    // 计算曲面上某点的法线
    gp_Vec GetNormalAtPoint(const TopoDS_Face& face, const gp_Pnt& p) {
        BRepAdaptor_Surface surf(face);
        // 将 3D 点投影回 UV 空间 (这对计算精确法线很重要)
        // 这里简化处理：假设点在面上，使用 GeomAPI_ProjectPointOnSurf
        // 为了性能，工业级实现通常会利用 Edge 的 pcurve，这里用投影作为通用解法
        GeomAPI_ProjectPointOnSurf proj(p, BRep_Tool::Surface(face));
        if (proj.NbPoints() > 0) {
            double u, v;
            proj.LowerDistanceParameters(u, v);

            BRepLProp_SLProps props(surf, u, v, 1, 1e-6);
            if (props.IsNormalDefined()) {
                gp_Vec n = props.Normal();
                if (face.Orientation() == TopAbs_REVERSED) n.Reverse();
                return n;
            }
        }
        return gp_Vec(0, 0, 0); // 失
    }

    gp_Vec GetNormalAtFace(const TopoDS_Face& face, const gp_Pnt& p) {
        Handle(Geom_Surface) surf = BRep_Tool::Surface(face);
        GeomAPI_ProjectPointOnSurf proj(p, surf);
        if (proj.NbPoints() < 1) return gp_Vec(0, 0, 1);
        double u, v;
        proj.LowerDistanceParameters(u, v);
        BRepLProp_SLProps props(BRepAdaptor_Surface(face), u, v, 1, 1e-6);
        if (props.IsNormalDefined()) {
            gp_Vec n = props.Normal();
            if (face.Orientation() == TopAbs_REVERSED) n.Reverse();
            return n;
        }
        return gp_Vec(0, 0, 1);
    }

    int CalcConvexity(const TopoDS_Edge& edge, int f1_idx, int f2_idx, const TopTools_IndexedMapOfShape& unique_faces) {
        if (f1_idx == f2_idx) return 2;
        TopoDS_Face f1 = TopoDS::Face(unique_faces.FindKey(f1_idx + 1));
        TopoDS_Face f2 = TopoDS::Face(unique_faces.FindKey(f2_idx + 1));

        double first, last;
        Handle(Geom_Curve) c3d = BRep_Tool::Curve(edge, first, last);
        if (c3d.IsNull()) return 2;

        // 取中点附近，防止特殊点
        double p_val = first + (last - first) * 0.43;
        gp_Pnt p; gp_Vec tang;
        c3d->D1(p_val, p, tang);

        gp_Vec n1 = GetNormalAtFace(f1, p);
        gp_Vec n2 = GetNormalAtFace(f2, p);

        if (n1.Angle(n2) < 0.087 || (M_PI - n1.Angle(n2)) < 0.087) return 2; // Smooth

        // 凸凹判断
        gp_Vec cross = n1 ^ n2;

        bool f1_fwd = false;
        TopExp_Explorer ex(f1, TopAbs_EDGE);
        for (; ex.More(); ex.Next()) {
            if (ex.Current().IsSame(edge)) {
                if (ex.Current().Orientation() == TopAbs_FORWARD) f1_fwd = true;
                break;
            }
        }
        if (!f1_fwd) tang.Reverse();

        if (cross * tang > 0) return 1; // Convex
        return 0; // Concave
    }
}
