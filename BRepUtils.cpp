#include "BRepUtils.h"

// OpenCascade 算法实现所需的头文件
// 这些头文件比较重，放在 .cpp 可以加快其他文件的编译速度
#include <BRep_Tool.hxx>
#include <BRepAdaptor_Surface.hxx>
#include <GeomAPI_ProjectPointOnSurf.hxx>
#include <BRepLProp_SLProps.hxx>
#include <Geom_Surface.hxx>
#include <TopLoc_Location.hxx>

using namespace breptorch;

namespace BRepUtils {
    // --- 数学工具函数 ---

    // 获取更严格的参数插值，保证首尾精确落在边界
    // IMPORTANT: Python samples U from max to min, but V from min to max
    // U parameter (reverse=true): index=0 -> max_val, index=total-1 -> min_val
    // V parameter (reverse=false): index=0 -> min_val, index=total-1 -> max_val
    double GetParamStrict(int index, int total, double min_val, double max_val, bool reverse) {
        if (reverse) {
            // Sample from max to min (for U parameter)
            if (index == 0) return max_val;
            if (index == total - 1) return min_val;
            return max_val - (max_val - min_val) * (double)index / (double)(total - 1);
        } else {
            // Sample from min to max (for V parameter)
            if (index == 0) return min_val;
            if (index == total - 1) return max_val;
            return min_val + (max_val - min_val) * (double)index / (double)(total - 1);
        }
    }

    // --- 几何辅助函数 ---

    // 获取面上某点的法线
    gp_Vec GetNormalAtPoint(const TopoDS_Face& face, const gp_Pnt& p) {
        BRepAdaptor_Surface surf(face);
        // 将 3D 点投影到 UV 空间 (对计算精确法线很重要)
        // 这里简化处理，对于任意面上，使用 GeomAPI_ProjectPointOnSurf
        // 为了性能，工业级实现通常利用 Edge 的 pcurve，这里用投影作为通用解法
        // 注意：投影必须使用带 location 的表面，并将点变换到局部坐标系，
        // 否则对带 location 的面会投影到错误的 (u,v)
        TopLoc_Location loc;
        Handle(Geom_Surface) geom_surf = BRep_Tool::Surface(face, loc);
        gp_Pnt p_local = p;
        if (!loc.IsIdentity()) {
            p_local.Transform(loc.Transformation().Inverted());
        }
        GeomAPI_ProjectPointOnSurf proj(p_local, geom_surf);
        if (proj.NbPoints() > 0) {
            double u, v;
            proj.LowerDistanceParameters(u, v);

            BRepLProp_SLProps props(surf, u, v, 1, 1e-6);
            if (props.IsNormalDefined()) {
                gp_Vec n = props.Normal();
                if (face.Orientation() == TopAbs_REVERSED) n.Reverse();
                if (n.Magnitude() > 1e-7) {
                    n.Normalize();
                }

                return n;
            }
        }
        return gp_Vec(0, 0, 0); // 失败
    }
}
