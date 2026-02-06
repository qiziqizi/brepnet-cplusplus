#pragma once
#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <cmath>
//#include <torch/torch.h>
#include "BRepTorch.h"
#include "cnpy.h"

// OpenCascade ������������ͷ�ļ�
// (ֻ����������Ҫ��ͷ�ļ������ٱ�������)
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
    // --- ��ѧ���ߺ��� ---

    // ������ƽ���ϵ�ͶӰ
    Tensor ProjectVector(Tensor vec, Tensor target_plane_normal);

    // ����������������
    Tensor AnyOrthogonalTensor(Tensor vec);

    // Get parameter with strict boundary matching
    // reverse=true: sample from max to min (for U parameter)
    // reverse=false: sample from min to max (for V parameter)
    double GetParamStrict(int index, int total, double min_val, double max_val, bool reverse = true);

    // --- �����������㺯�� ---

    // ����������
    double GetFaceArea(const TopoDS_Shape& face);

    // ����ߵĳ���
    double GetEdgeLength(const TopoDS_Shape& edge);

    // ���� Shape ����һ���ߴ�
    TopoDS_Shape ScaleShape(const TopoDS_Shape& s);

    // ��ȡ�������ϵķ����� (ͨ�� UV ͶӰ)
    gp_Vec GetNormalAtPoint(const TopoDS_Face& face, const gp_Pnt& p);

    // ��ȡ�������ϵķ����� (ͨ�� Geom Surface)
    gp_Vec GetNormalAtFace(const TopoDS_Face& face, const gp_Pnt& p);

    // ����ߵ�͹�� (0: ��, 1: ͹, 2: ƽ��/����)
    int CalcConvexity(const TopoDS_Edge& edge, int f1_idx, int f2_idx, const TopTools_IndexedMapOfShape& unique_faces);
}
