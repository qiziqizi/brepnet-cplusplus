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
#include <gp_Vec.hxx>
#include <gp_Pnt.hxx>

//using namespace torch;
using namespace breptorch;

namespace BRepUtils {
    // --- 数学工具函数 ---

    // Get parameter with strict boundary matching
    // reverse=true: sample from max to min (for U parameter)
    // reverse=false: sample from min to max (for V parameter)
    double GetParamStrict(int index, int total, double min_val, double max_val, bool reverse = true);

    // --- 面和边的计算函数 ---

    // 获取面上某点的法向量 (通过 UV 投影)
    gp_Vec GetNormalAtPoint(const TopoDS_Face& face, const gp_Pnt& p);
}
