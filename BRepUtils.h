#pragma once
#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <cmath>
//#include <torch/torch.h>
#include "BRepTorch.h"
#include "cnpy.h"

// OpenCascade 必要的最小头文件
// (只包含声明需要的头文件，减少编译依赖)
#include <TopoDS.hxx>
#include <TopoDS_Shape.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Edge.hxx>
#include <gp_Vec.hxx>
#include <gp_Pnt.hxx>
#include <TopTools_IndexedMapOfShape.hxx>

//using namespace torch;
using namespace breptorch;

namespace BRepUtils {
    // --- 数学工具函数 ---

    // 向量到平面上的投影
    Tensor ProjectVector(Tensor vec, Tensor target_plane_normal);

    // 求任意正交向量
    Tensor AnyOrthogonalTensor(Tensor vec);

    // Get parameter with strict boundary matching
    // reverse=true: sample from max to min (for U parameter)
    // reverse=false: sample from min to max (for V parameter)
    double GetParamStrict(int index, int total, double min_val, double max_val, bool reverse = true);

    // --- 面和边的计算函数 ---

    // 计算面的面积
    double GetFaceArea(const TopoDS_Shape& face);

    // 计算边的长度
    double GetEdgeLength(const TopoDS_Shape& edge);

    // 缩放 Shape 到归一化尺寸
    TopoDS_Shape ScaleShape(const TopoDS_Shape& s);

    // 获取面上某点的法向量 (通过 UV 投影)
    gp_Vec GetNormalAtPoint(const TopoDS_Face& face, const gp_Pnt& p);

    // 获取面上某点的法向量 (通过 Geom Surface)
    gp_Vec GetNormalAtFace(const TopoDS_Face& face, const gp_Pnt& p);

    // 计算边的凹凸性 (0: 凹, 1: 凸, 2: 平滑/光顺)
    int CalcConvexity(const TopoDS_Edge& edge, int f1_idx, int f2_idx, const TopTools_IndexedMapOfShape& unique_faces);
}
