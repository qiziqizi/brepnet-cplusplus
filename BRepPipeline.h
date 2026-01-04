#pragma once

#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <algorithm> // 用于 std::sort
#include <cmath>

// LibTorch
#include <torch/torch.h>
#include "cnpy.h"
#include "BRepUtils.h"

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

using namespace torch;

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


    // --- 核心入口 ---
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
        TopoDS_Shape shape = BRepUtils::ScaleShape(original_shape);

        // ---------------- 面排序逻辑优化 ----------------
        // 使用 pair 存储 <面积, Face>，避免在 sort 里重复计算
        std::vector<std::pair<double, TopoDS_Face>> face_props;
        TopExp_Explorer faceExp(shape, TopAbs_FACE);

        int f_count = 0;
        for (; faceExp.More(); faceExp.Next()) {
            TopoDS_Face f = TopoDS::Face(faceExp.Current());
            double area = 0.0;

            // 【安全防护】加 try-catch 防止 OCCT 在计算几何属性时崩溃
            try {
                GProp_GProps props;
                BRepGProp::SurfaceProperties(f, props);
                area = props.Mass();
                // 处理极小值或负值（防止 NaN 导致排序崩溃）
                if (area < 1e-9) area = 0.0;
            }
            catch (...) {
                std::cerr << "  [Warning] Face " << f_count << " 面积计算失败，设为 0" << std::endl;
                area = 0.0;
            }
            face_props.push_back({ area, f });
            f_count++;
        }

        // 存入 unique_faces
        for (const auto& pair : face_props) {
            unique_faces.Add(pair.second);
        }
        //std::cout << "[Debug] 面排序完成。" << std::endl;

        // ---------------- 边排序逻辑优化 ----------------
        std::vector<std::pair<double, TopoDS_Edge>> edge_props;
        TopExp_Explorer edgeExp(shape, TopAbs_EDGE);

        int e_count = 0;
        for (; edgeExp.More(); edgeExp.Next()) {
            TopoDS_Edge e = TopoDS::Edge(edgeExp.Current());
            double length = 0.0;

            // 【安全防护】加 try-catch
            try {
                GProp_GProps props;
                BRepGProp::LinearProperties(e, props);
                length = props.Mass();
                if (length < 1e-9) length = 0.0;
            }
            catch (...) {
                std::cerr << " [Warning] Edge " << e_count << " 长度计算失败，设为 0" << std::endl;
                length = 0.0;
            }
            edge_props.push_back({ length, e });
            e_count++;
        }


        // 存入 unique_edges
        for (const auto& pair : edge_props) {
            unique_edges.Add(pair.second);
        }
        //std::cout << "[Debug] 边排序完成。" << std::endl;


        build_topology();
        extract_features();
        generate_tensors();
        generate_global_face_grids(); // c++这边生成的 uv打点
        //generate_curve_grids(); // (Coedge + Edge)
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
                    // 返回一个空 tensor 或者抛出异常，防止后面崩
                    return torch::ones({ 1 }, torch::kFloat32);
                }
                cnpy::NpyArray arr = npz[key];
                std::vector<int64_t> shape;
                for (auto s : arr.shape) shape.push_back(s);
                return torch::from_blob(arr.data<float>(), shape, torch::kFloat32).clone();
                };
            mean_f = load_t("mean_f"); std_f = load_t("std_f");
            mean_e = load_t("mean_e"); std_e = load_t("std_e");
            mean_c = load_t("mean_c"); std_c = load_t("std_c");
            float eps = 1e-6;
            if (std_f.defined())
                std_f = torch::where(std_f < eps, torch::ones_like(std_f), std_f);
            if (std_e.defined())
                std_e = torch::where(std_e < eps, torch::ones_like(std_e), std_e);
            if (std_c.defined())
                std_c = torch::where(std_c < eps, torch::ones_like(std_c), std_c);
            has_stats = true;
        }
        catch (const std::exception& e) {
            std::cerr << " Load stats failed: " << e.what() << std::endl;
            has_stats = false; 
        }
    }

    // 标准化
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
    // 在 BRepPipeline 类的 public 部分增加：

    Tensor FaceGridsGlobal; // 存储读取进来的 Grid 数据 [N, 7, 10, 10]
    Tensor EdgeGridsGlobal;
    Tensor CoedgeGridsGlobal;
    bool use_uvnet = false;

    // 新增函数：加载 Grid 数据 (从 Python 导出的 npz)
    void load_grids_from_npz(const std::string& npz_path) {
        try {
            cnpy::npz_t npz = cnpy::npz_load(npz_path);

            // 1. Face
            if (npz.count("face_point_grids")) {
                cnpy::NpyArray arr = npz["face_point_grids"];
                std::vector<int64_t> s; for (auto d : arr.shape) s.push_back(d);
                FaceGridsGlobal = torch::from_blob(arr.data<float>(), s, torch::kFloat32).clone();
                // 没有补 Padding ，这部分逻辑在 Forward 里做更安全，这里只负责加载
                use_uvnet = true;
                std::cout << "Loaded Face Grids: " << FaceGridsGlobal.sizes() << std::endl;
                //std::cout << FaceGridsGlobal[0][0] << std::endl;
            }

            // 2. Edge
            if (npz.count("edge_point_grids")) {
                cnpy::NpyArray arr = npz["edge_point_grids"];
                std::vector<int64_t> s; for (auto d : arr.shape) s.push_back(d);
                EdgeGridsGlobal = torch::from_blob(arr.data<float>(), s, torch::kFloat32).clone();
                std::cout << "Loaded Edge Grids: " << EdgeGridsGlobal.sizes() << std::endl;
                //std::cout << EdgeGridsGlobal[0] << std::endl;
            }

            // 3. Coedge
            if (npz.count("coedge_point_grids")) {
                cnpy::NpyArray arr = npz["coedge_point_grids"];
                std::vector<int64_t> s; for (auto d : arr.shape) s.push_back(d);
                CoedgeGridsGlobal = torch::from_blob(arr.data<float>(), s, torch::kFloat32).clone();
                std::cout << "Loaded Coedge Grids: " << CoedgeGridsGlobal.sizes() << std::endl;
            }
        }
        catch (const std::exception& e) {
                std::cerr << "Failed to load grids from npz: " << e.what() << std::endl;
        }
    }

    // 存储局部坐标系下的特征
    Tensor FaceGridsLocal;   // [N_c, 2, 9, 10, 10]
    Tensor EdgeGridsLocal;
    Tensor CoedgeGridsLocal; // [N_c, 13, 10]

    // 生成局部特征主函数 (整合了 Face, Coedge 和 Edge)
    void generate_local_grids() {
        int num_c = coedges.size();
        if (num_c == 0) return;

        std::cout << "正在生成局部坐标系特征 (LCS Transformation)..." << std::endl;

        // ---------------------------------------------------------
        // 1. 计算所有 LCS 及其逆矩阵
        // ---------------------------------------------------------
        std::vector<Tensor> lcs_invs;
        lcs_invs.reserve(num_c);
        for (int i = 0; i < num_c; ++i) {
            Tensor mat = compute_coedge_lcs(i); // 使用 compute_coedge_lcs (注意你代码里可能叫 ComputeCoedgeLCS，请统一名字)
            // 检查奇异性
            if (torch::det(mat).abs().item<float>() < 1e-6) {
                mat = torch::eye(4);
            }
            lcs_invs.push_back(torch::inverse(mat));
        }

        // ---------------------------------------------------------
        // 2. Coedge Local Grids [N_c, 13, 10]
        // ---------------------------------------------------------
        std::vector<Tensor> c_list;
        c_list.reserve(num_c);
        for (int i = 0; i < num_c; ++i) {
            Tensor g_global = generate_global_coedge_grid(i);
            Tensor g_local = transform_grid_to_local(g_global, lcs_invs[i], false); // false = is_coedge
            c_list.push_back(g_local);
        }
        CoedgeGridsLocal = torch::stack(c_list);

        // ---------------------------------------------------------
        // 3. Face Local Grids [N_c, 2, 9, 10, 10]
        // ---------------------------------------------------------
        std::vector<Tensor> f_list;
        f_list.reserve(num_c);
        for (int i = 0; i < num_c; ++i) {
            Tensor pair = torch::zeros({ 2, 9, 10, 10 }, torch::kFloat32);

            // A. Self Face
            int f_idx = coedges[i].face_idx;
            // 假设 FaceGridsGlobal 是 Raw 模式 (N行)，直接用 f_idx
            if (FaceGridsGlobal.defined() && f_idx < FaceGridsGlobal.size(0)) {
                pair[0] = transform_grid_to_local(FaceGridsGlobal[f_idx], lcs_invs[i], true); // true = is_face
            }

            // B. Mate Face
            int mate_idx = coedges[i].mate_idx;
            if (mate_idx != -1) {
                int mf_idx = coedges[mate_idx].face_idx;
                if (FaceGridsGlobal.defined() && mf_idx < FaceGridsGlobal.size(0)) {
                    // 使用 Mate 的 LCS 变换 Mate 的面
                    pair[1] = transform_grid_to_local(FaceGridsGlobal[mf_idx], lcs_invs[mate_idx], true);
                }
            }
            f_list.push_back(pair);
        }
        FaceGridsLocal = torch::stack(f_list);

        // ---------------------------------------------------------
        // 4. 【整合】Edge Local Grids [N_e, 13, 10]
        // ---------------------------------------------------------
        int num_e = unique_edges.Extent();
        if (num_e > 0 && CoedgeGridsLocal.defined()) {
            // 优化：使用查找表代替嵌套循环，O(Nc) 复杂度
            // edge_representatives[edge_idx] = best_coedge_id
            std::vector<int> edge_representatives(num_e, -1);

            for (const auto& c : coedges) {
                int eid = c.edge_idx;
                if (eid >= 0 && eid < num_e) {
                    // 策略：如果还没找到，就先填这个；如果找到了Forward的，就覆盖
                    if (edge_representatives[eid] == -1 || c.orientation == true) {
                        edge_representatives[eid] = c.id;
                    }
                }
            }

            std::vector<Tensor> e_list;
            e_list.reserve(num_e);
            for (int i = 0; i < num_e; ++i) {
                int cid = edge_representatives[i];
                if (cid != -1) {
                    // 直接拷贝 Coedge 的局部 Grid
                    e_list.push_back(CoedgeGridsLocal[cid]);
                }
                else {
                    // 防御性：理论上不应发生
                    e_list.push_back(torch::zeros({ 13, 10 }, CoedgeGridsLocal.options()));
                }
            }
            EdgeGridsLocal = torch::stack(e_list);
        }

        std::cout << " Local Features Generated." << std::endl;
        if (FaceGridsLocal.defined()) std::cout << "   Face: " << FaceGridsLocal.sizes() << std::endl;
        if (CoedgeGridsLocal.defined()) std::cout << "   Coedge: " << CoedgeGridsLocal.sizes() << std::endl;
        if (EdgeGridsLocal.defined()) std::cout << "   Edge: " << EdgeGridsLocal.sizes() << std::endl;
    }

private:
    void build_topology() {
        coedges.clear();
        std::map<int, std::vector<int>> edge_to_coedge_map;

        for (int f_idx = 1; f_idx <= unique_faces.Extent(); ++f_idx) {
            const TopoDS_Face& face = TopoDS::Face(unique_faces.FindKey(f_idx));
            TopExp_Explorer wireExp(face, TopAbs_WIRE);
            for (; wireExp.More(); wireExp.Next()) {
                const TopoDS_Wire& wire = TopoDS::Wire(wireExp.Current());
                int first_coedge = -1;
                int prev_coedge = -1;

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

                    if (prev_coedge != -1) {
                        coedges[prev_coedge].next_idx = c.id;
                        coedges[c.id].prev_idx = prev_coedge;
                    }
                    else {
                        first_coedge = c.id;
                    }
                    prev_coedge = c.id;
                }
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

    // 生成 Xf，Xe, Xc (Raw)
    void extract_features() {
        int num_f = unique_faces.Extent();
        int num_e = unique_edges.Extent();
        int num_c = coedges.size();

        // 1. 构建 Edge -> Faces 映射
        std::vector<std::vector<int>> edge_owner_faces(num_e);
        for (int i = 0; i < num_f; ++i) {
            const TopoDS_Face& f = TopoDS::Face(unique_faces.FindKey(i + 1));
            TopExp_Explorer ex(f, TopAbs_EDGE);
            for (; ex.More(); ex.Next()) {
                int e_idx = unique_edges.FindIndex(ex.Current());
                // 【安全检查】OpenCascade 索引从1开始，0表示没找到
                if (e_idx > 0 && e_idx <= num_e) {
                    edge_owner_faces[e_idx - 1].push_back(i);
                }
            }
        }

        // std::cout << "  [Feat] 正在提取 Face 特征..." << std::endl;
        // 2. 提取 Face 特征
        Xf = torch::zeros({ num_f, 7 });
        auto Xf_a = Xf.accessor<float, 2>();
        for (int i = 0; i < num_f; ++i) {
            try {
                int row = i;
                const TopoDS_Face& f = TopoDS::Face(unique_faces.FindKey(i + 1));
                BRepAdaptor_Surface s(f);
                GeomAbs_SurfaceType t = s.GetType();

                if (t == GeomAbs_Plane) Xf_a[row][0] = 1;
                else if (t == GeomAbs_Cylinder) Xf_a[row][1] = 1;
                else if (t == GeomAbs_Cone) Xf_a[row][2] = 1;
                else if (t == GeomAbs_Sphere) Xf_a[row][3] = 1;
                else if (t == GeomAbs_Torus) Xf_a[row][4] = 1;

                Xf_a[row][5] = (float)BRepUtils::GetFaceArea(f);

                if (t == GeomAbs_BSplineSurface || t == GeomAbs_BezierSurface) Xf_a[row][6] = 1;
            }
            catch (...) {
                std::cerr << "  Face " << i << " 特征提取失败，跳过。" << std::endl;
            }
        }

        // 3. 提取 Edge 特征
        Xe = torch::zeros({ num_e, 10 });
        auto Xe_a = Xe.accessor<float, 2>();
        for (int i = 0; i < num_e; ++i) {
            try {
                int row = i;
                const TopoDS_Edge& e = TopoDS::Edge(unique_edges.FindKey(i + 1));

                // 默认初始化: 平滑，非闭合
                Xe_a[row][0] = 0; Xe_a[row][1] = 0; Xe_a[row][2] = 1;

                // 计算凸凹性 (CalcConvexity 是高危函数)
                if (edge_owner_faces[i].size() == 2) {
                    // 【安全包裹】
                    try {
                        int cvx = BRepUtils::CalcConvexity(e, edge_owner_faces[i][0], edge_owner_faces[i][1], this->unique_faces);
                        if (cvx == 0) { Xe_a[row][0] = 1; Xe_a[row][2] = 0; } // Concave
                        else if (cvx == 1) { Xe_a[row][1] = 1; Xe_a[row][2] = 0; } // Convex
                    }
                    catch (...) {
                        // 如果几何计算崩溃，保持默认(Smooth)
                    }
                }

                Xe_a[row][3] = (float)BRepUtils::GetEdgeLength(e);

                BRepAdaptor_Curve c(e);
                GeomAbs_CurveType t = c.GetType();
                if (t == GeomAbs_Circle) Xe_a[row][4] = 1;
                else if (t == GeomAbs_Ellipse) Xe_a[row][6] = 1;
                else if (t == GeomAbs_Line) Xe_a[row][9] = 1;
                else Xe_a[row][7] = 1; // Spline/Other

                if (BRep_Tool::IsClosed(e)) Xe_a[row][5] = 1;

            }
            catch (...) {
                std::cerr << "  Edge " << i << " 特征提取失败，跳过。" << std::endl;
            }
        }

        // 4. 提取 Coedge 特征
        Xc = torch::zeros({ num_c, 1 });
        auto Xc_a = Xc.accessor<float, 2>();
        for (int i = 0; i < num_c; ++i) {
            if (!coedges[i].orientation) Xc_a[i][0] = 1;
        }
    }

    void generate_tensors() {
        // 获取数量
        int num_f = unique_faces.Extent();
        int num_e = unique_edges.Extent();
        int num_c = coedges.size();

        std::vector<int64_t> kf, ke, kc;
        // winged_edge.json改为 simple_edge.json
        //std::vector<std::vector<int>> fw = { {},{1} }, ew = { {},{2},{3},{1,2},{1,3} }, cw = { {},{1},{2},{3},{1,2},{1,3} };
        std::vector<std::vector<int>> fw = { {},{1} }, ew = { {} }, cw = { {},{1} };
        // --- 生成 Kf ---
        // 【修改 1】 不要 push_back(0) 了
        for (const auto& c : coedges) {
            for (auto& rule : fw) {
                int t = walk(c.id, rule);
                // 【修改 2】 如果 t==-1 (无邻居)，存 num_f (作为越界值)
                // 如果 t!=-1，存 coedges[t].face_idx (0-based)
                kf.push_back(t == -1 ? num_f : coedges[t].face_idx);
            }
        }
        // 形状改为 {num_c, ...}
        Kf = torch::from_blob(kf.data(), { num_c, (long long)fw.size() }, torch::kLong).clone();

        // --- 生成 Ke ---
        // 不要 push_back(0)
        for (const auto& c : coedges) {
            for (auto& rule : ew) {
                int t = walk(c.id, rule);
                // 无邻居存 num_e，有邻居存 edge_idx
                ke.push_back(t == -1 ? num_e : coedges[t].edge_idx);
            }
        }
        Ke = torch::from_blob(ke.data(), { num_c, (long long)ew.size() }, torch::kLong).clone();

        // --- 生成 Kc ---
        // 不要 push_back(0)
        for (const auto& c : coedges) {
            for (auto& rule : cw) {
                int t = walk(c.id, rule);
                // 无邻居存 num_c，有邻居存 t (共边ID本身就是0-based)
                kc.push_back(t == -1 ? num_c : t);
            }
        }
        Kc = torch::from_blob(kc.data(), { num_c, (long long)cw.size() }, torch::kLong).clone();

        // --- Pooling Ce ---
        std::vector<int64_t> ce(num_e * 2, num_c); // 初始化为 num_c (无效值)
        std::vector<int> ec(num_e, 0);
        for (const auto& c : coedges) {
            // 存 c.id (0-based)
            if (ec[c.edge_idx] < 2) ce[c.edge_idx * 2 + ec[c.edge_idx]++] = c.id;
        }
        Ce = torch::from_blob(ce.data(), { num_e, 2 }, torch::kLong).clone();

        // --- Pooling Cf ---
        // 修改！！！
        int max_cpf = 512;
        std::vector<int64_t> cf(num_f * max_cpf, num_c); // 初始化为 num_c (无效值)
        std::vector<int> fc(num_f, 0);
        for (const auto& c : coedges) {
            // 存 c.id (0-based)
            if (fc[c.face_idx] < max_cpf) cf[c.face_idx * max_cpf + fc[c.face_idx]++] = c.id;
        }
        Cf = torch::from_blob(cf.data(), { num_f, max_cpf }, torch::kLong).clone();

        Csf.clear();
    }


    // 对应 python 中 extract_face_point_grid
    // =========================================================================

    Tensor generate_global_face_grid(const TopoDS_Face& face) {
        int num_u = 10;
        int num_v = 10;

        // 1. 准备 Tensor (7通道: x,y,z, nx,ny,nz, mask)
        //// 形状: [7, 10, 10]
        // Tensor grid = torch::zeros({ 7, num_u, num_v }, torch::kFloat32);

        // 形状: [9, 10, 10]
        Tensor grid = torch::zeros({ 9, num_u, num_v }, torch::kFloat32);

        // 获取 Tensor 的访问器，用于快速写入数据 (比 .index_put 快得多)
        auto accessor = grid.accessor<float, 3>();

        // 2. 获取 UV 边界
        Standard_Real umin, umax, vmin, vmax;
        BRepTools::UVBounds(face, umin, umax, vmin, vmax);
        // 3. 几何适配器 (用于计算点和法线)
        BRepAdaptor_Surface surf(face);

        // 4. 点分类器 (用于计算 Mask - 判断点是否在面的裁剪轮廓内)
        // 传入 face 初始化，用于后续判断点是否在面上
        BRepTopAdaptor_FClass2d  classifier(face, 0.0);
        //BRepTopAdaptor_FClass2d  classifier(face, Precision::PConfusion());
        // 5. 双重循环采样
        for (int i = 0; i < num_u; ++i) {
            for (int j = 0; j < num_v; ++j) {
                double u = BRepUtils::GetParamStrict(i, num_u, umin, umax);
                double v = BRepUtils::GetParamStrict(j, num_v, vmin, vmax);

                // A. 计算几何信息 (点坐标 + 一阶导数用于算法线)
                gp_Pnt p;
                gp_Vec d1u, d1v;
                surf.D1(u, v, p, d1u, d1v);

                // B. 计算法线 (叉乘)
                gp_Vec n = d1u ^ d1v;
                // 归一化法线
                if (n.Magnitude() > Precision::Confusion()) {
                    n.Normalize();
                }
                else {
                    n = gp_Vec(0, 0, 0); // 奇异点处理
                }

                // 【关键】处理面的朝向 (Orientation)
                // 如果面是反向的 (REVERSED)，OpenCascade 的几何法线是反的，需要乘回修正
                if (face.Orientation() == TopAbs_REVERSED) {
                    n.Reverse();
                }

                // C. 计算 Mask (是否在面内)
                gp_Pnt2d p2d(u, v);
                TopAbs_State state = classifier.Perform(p2d);
                //TopAbs_State state = classifier.Perform(p2d, 1e-5);
                //TopAbs_State state = classifier.Perform(p2d, 1e-7);
                // IN: 在内部, ON: 在边界上 -> 都算 1
                //float mask_val = (state == TopAbs_IN || state == TopAbs_ON) ? 1.0f : 0.0f;
                
                float mask_val = (state == TopAbs_IN) ? 1.0f : 0.0f;

                bool is_on_border = (i == 0 || i == num_u - 1 || j == 0 || j == num_v - 1);
                if (is_on_border) {
                    mask_val = 0.0f; // 在源头把 Mask 修正为 0
                }
                // D. 填入 Tensor (注意顺序：C, H, W)
                // 通道 0-2: 坐标 xyz
                accessor[0][i][j] = (float)p.X();
                accessor[1][i][j] = (float)p.Y();
                accessor[2][i][j] = (float)p.Z();

                // 通道 3-5: 法线 nx, ny, nz
                accessor[3][i][j] = (float)n.X();
                accessor[4][i][j] = (float)n.Y();
                accessor[5][i][j] = (float)n.Z();

                // 通道 6: Mask
                accessor[6][i][j] = mask_val;

                accessor[7][i][j] = (float)u;
                accessor[8][i][j] = (float)v;
            }
        }
        // 必须加在 generate_global_face_grid 里

        //加上这行代码，误差从5.多到1.0左右
        if (face.Orientation() == TopAbs_REVERSED) {
            // std::cout << "  [Info] Face is Reversed, flipping Grid U-axis..." << std::endl;
            grid = torch::flip(grid, { 1 }); // 翻转 U 轴 (即第 1 维, 0是Channel)
        }

        return grid;
    }

    // 对应 python 中 extract_face_point_grids
    void generate_global_face_grids() {
        if (unique_faces.Extent() == 0) return;

        std::vector<Tensor> grids_list;

        // 遍历所有唯一面 (注意 unique_faces 从 1 开始索引)
        int num_faces = unique_faces.Extent();
        for (int i = 1; i <= num_faces; ++i) {
            const TopoDS_Face& face = TopoDS::Face(unique_faces.FindKey(i));

            // 采样单个面
            Tensor single_grid = generate_global_face_grid(face);

            // 放入列表
            grids_list.push_back(single_grid);
        }

        // 堆叠成一个大 Tensor: [N, 7, 10, 10]
        // 此时是没有 Padding 的 (N行)
        if (!grids_list.empty()) {
            this->FaceGridsGlobal = torch::stack(grids_list);
        }
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

        // 2. 准备 Tensor [12, 10] (最终需要转置为 [12, 10] ? Python 是 [12, 10]，但 PyTorch Conv1d 输入通常是 [Batch, Channel, Length])
        // Python extract_coedge_point_grid 返回的是 np.transpose(single_grid, (1,0)) 
        // single_grid 是 [10, 12]。转置后是 [12, 10]。
        // 我们直接生成 [12, 10]。
        int num_u = 10;
        //Tensor grid = torch::zeros({12, num_u}, torch::kFloat32);
        Tensor grid = torch::zeros({13, num_u}, torch::kFloat32);
        auto accessor = grid.accessor<float, 2>();

        // 3. 曲线适配器
        BRepAdaptor_Curve curve_adaptor(edge);
        double first = curve_adaptor.FirstParameter();
        double last = curve_adaptor.LastParameter();
        double len = last - first;

        // 4. 等弧长采样 (Uniform Abscissa) - 模仿 Python use_arclength_params=True
        // 如果曲线太短或退化，回退到参数采样
        bool use_uniform = true;
        GCPnts_UniformAbscissa uniform_sampler;
        try {
            uniform_sampler.Initialize(curve_adaptor, num_u, -1); // -1 tol
            if (!uniform_sampler.IsDone()) use_uniform = false;
        } catch(...) { use_uniform = false; }

        // 5. 循环采样
        for (int i = 0; i < num_u; ++i) {
            double param;
            if (use_uniform && uniform_sampler.NbPoints() >= num_u) {
                // GCPnts 生成的是点数，索引从1开始
                // 我们需要重新映射一下，或者简单均匀分布参数
                // 为了简化且稳健，这里我们用简单的参数空间均匀采样，
                // 如果严格追求精度，可以用 GCPnts_UniformAbscissa 的 Parameter(i+1)
                // 但要注意 GCPnts 的点数可能不完全等于 num_u
                 param = uniform_sampler.Parameter(i + 1);
            } else {
                // 回退：参数空间均匀采样
                param = first + (len * i) / (double)(num_u - 1);
            }

            gp_Pnt p;
            gp_Vec tangent;
            curve_adaptor.D1(param, p, tangent);
            
            // 归一化切向量
            if (tangent.Magnitude() > 1e-7) tangent.Normalize();

            // 处理共边方向 (Orientation)
            // 如果 Coedge 是 Reversed，说明它沿着 Edge 的反方向走
            // 它的“切向量”应该取反
            if (!c_info.orientation) { // orientation == false means REVERSED
                tangent.Reverse();
            }
            // 注意：Python代码里如果是 Reversed，点序列本身也是反的吗？
            // 通常 Grid 是按几何顺序存的，但 Tangent 会根据 Coedge 方向调整。
            // 你的 Python 代码里 coedge_data 似乎处理了这个问题。
            // 这里我们假设 Grid 还是按几何 U 增量存，但 Tangent 跟随 Coedge。

            // 计算法线
            gp_Vec n_left = BRepUtils::GetNormalAtPoint(face_left, p);
            gp_Vec n_right = (has_mate) ? BRepUtils::GetNormalAtPoint(face_right, p) : gp_Vec(0,0,0);

            // 填入 Tensor
            // Points (0-2)
            accessor[0][i] = (float)p.X();
            accessor[1][i] = (float)p.Y();
            accessor[2][i] = (float)p.Z();
            
            // Tangents (3-5)
            accessor[3][i] = (float)tangent.X();
            accessor[4][i] = (float)tangent.Y();
            accessor[5][i] = (float)tangent.Z();

            // Left Normals (6-8)
            accessor[6][i] = (float)n_left.X();
            accessor[7][i] = (float)n_left.Y();
            accessor[8][i] = (float)n_left.Z();

            // Right Normals (9-11)
            accessor[9][i] = (float)n_right.X();
            accessor[10][i] = (float)n_right.Y();
            accessor[11][i] = (float)n_right.Z();

            // 修改：填充新增第12通道（对应Python的u_params）
            accessor[12][i] = (float)param; // 将曲线参数u值填入第12通道（0索引）
        }
        
        // 如果 Coedge 是反向的，Python 的 EdgeDataExtractor 可能会把点序列也翻转
        // (即: grid[][0] 是终点，grid[][9] 是起点)
        // 为了和 Python 保持一致，如果 orientation 是 false，我们需要 flip dim 1
        if (!c_info.orientation) {
            grid = torch::flip(grid, {1});
        }

        return grid;
    }

    // 修改 compute_coedge_lcs (使用精确中点)
    Tensor compute_coedge_lcs(int coedge_idx) {
        const CoedgeInfo& c_info = coedges[coedge_idx];
        TopoDS_Edge edge = TopoDS::Edge(unique_edges.FindKey(c_info.edge_idx + 1));
        TopoDS_Face face = TopoDS::Face(unique_faces.FindKey(c_info.face_idx + 1));

        // 1. 使用曲线适配器找精确中点
        BRepAdaptor_Curve curve(edge);

        // 处理等弧长参数 (Uniform Abscissa)
        // 我们需要找到曲线总长的 0.5 位置
        GCPnts_UniformAbscissa sampler;
        double mid_u = 0.0;

        // 尝试初始化采样器 (3个点: 起点, 中点, 终点)
        try {
            sampler.Initialize(curve, 3, -1); // 3 points -> index 1 is middle
            if (sampler.IsDone() && sampler.NbPoints() >= 2) {
                // index 1 (从1开始计数，所以是参数2)
                mid_u = sampler.Parameter(2);
            }
            else {
                // 回退到参数中点
                mid_u = (curve.FirstParameter() + curve.LastParameter()) * 0.5;
            }
        }
        catch (...) {
            mid_u = (curve.FirstParameter() + curve.LastParameter()) * 0.5;
        }

        // 2. 计算原点 P 和 切线 T
        gp_Pnt p;
        gp_Vec t_vec;
        curve.D1(mid_u, p, t_vec);

        // 归一化切线
        if (t_vec.Magnitude() > 1e-7) t_vec.Normalize();

        // 处理反向
        if (!c_info.orientation) t_vec.Reverse();

        // 3. 计算左面法线 N (W轴)
        gp_Vec n_vec = BRepUtils::GetNormalAtPoint(face, p);

        // 转换成 Tensor
        Tensor origin = torch::tensor({ (float)p.X(), (float)p.Y(), (float)p.Z() });
        Tensor tangent = torch::tensor({ (float)t_vec.X(), (float)t_vec.Y(), (float)t_vec.Z() });
        Tensor left_n = torch::tensor({ (float)n_vec.X(), (float)n_vec.Y(), (float)n_vec.Z() });

        // --- 以下逻辑保持不变 ---

        // W 轴
        Tensor w_vec = left_n / (torch::norm(left_n) + 1e-7);

        // V 轴 (Project Tangent to Normal Plane)
        Tensor v_vec = BRepUtils::ProjectVector(tangent, w_vec);
        if (!v_vec.defined()) {
            v_vec = BRepUtils::AnyOrthogonalTensor(w_vec);
        }
        else {
            v_vec = v_vec / (torch::norm(v_vec) + 1e-7);
        }

        // U 轴 (Cross)
        Tensor u_vec = torch::cross(v_vec, w_vec);

        // 4. 组装矩阵
        Tensor mat = torch::eye(4);
        mat.slice(0, 0, 3).slice(1, 0, 1).copy_(u_vec.view({ 3, 1 }));
        mat.slice(0, 0, 3).slice(1, 1, 2).copy_(v_vec.view({ 3, 1 }));
        mat.slice(0, 0, 3).slice(1, 2, 3).copy_(w_vec.view({ 3, 1 }));
        mat.slice(0, 0, 3).slice(1, 3, 4).copy_(origin.view({ 3, 1 }));
        return mat;
    }
    // 对应 Python 的 transform_face_point_grid_to_local
    // 将 Grid (全局) 变换到 Local
    // grid: [Channels, H, W] 或 [Channels, L]
    // lcs_inv: [4, 4] 逆矩阵
    // is_face: true 为 FaceGrid (9ch), false 为 CoedgeGrid (13ch)
    Tensor transform_grid_to_local(Tensor grid, Tensor lcs_inv, bool is_face) {
        // 1. 提取几何部分 (XYZ: 0-2, Normal: 3-5)
        // grid 形状可能是 [9, 10, 10] 或 [13, 10]

        long C = grid.size(0);
        long N_points = 1;
        for (int i = 1; i < grid.dim(); ++i) N_points *= grid.size(i);

        // 展平为 [C, N_points]
        Tensor flat = grid.view({ C, N_points });

        Tensor points = flat.slice(0, 0, 3); // [3, N]
        Tensor normals = flat.slice(0, 3, 6); // [3, N]

        // 2. 变换点 (Points)
        // 构造齐次坐标 [4, N]
        Tensor ones = torch::ones({ 1, N_points }, points.options());
        Tensor points_h = torch::cat({ points, ones }, 0);

        // 矩阵乘法: [4, 4] x [4, N] -> [4, N]
        Tensor points_local_h = torch::matmul(lcs_inv, points_h);
        Tensor points_local = points_local_h.slice(0, 0, 3); // 取前3行

        // 3. 变换法线 (Normals)
        // 法线只受旋转影响，不受平移影响。使用 LCS 的左上角 3x3
        Tensor rot_mat = lcs_inv.slice(0, 0, 3).slice(1, 0, 3); // [3, 3]
        Tensor normals_local = torch::matmul(rot_mat, normals);

        // 4. 如果是 Coedge，还有 Tangent (Channel 3-5 is Tangent, 6-8 LeftN, 9-11 RightN)
        // 上面的代码假设 3-5 是法线。
        // 对于 Face (9ch): 0-2 pts, 3-5 norm, 6 mask, 7-8 uv. -> 符合逻辑
        // 对于 Coedge (13ch): 0-2 pts, 3-5 tan, 6-8 Ln, 9-11 Rn, 12 u.

        Tensor new_flat = flat.clone();

        // 填回 Points
        new_flat.slice(0, 0, 3).copy_(points_local);

        if (is_face) {
            // Face: 3-5 是 Normal
            new_flat.slice(0, 3, 6).copy_(normals_local);
        }
        else {
            // Coedge: 需要变换 3组向量 (Tan, LeftN, RightN)
            Tensor tan = flat.slice(0, 3, 6);
            Tensor ln = flat.slice(0, 6, 9);
            Tensor rn = flat.slice(0, 9, 12);

            new_flat.slice(0, 3, 6).copy_(torch::matmul(rot_mat, tan));
            new_flat.slice(0, 6, 9).copy_(torch::matmul(rot_mat, ln));
            new_flat.slice(0, 9, 12).copy_(torch::matmul(rot_mat, rn));
        }

        // 5. 恢复形状
        return new_flat.view(grid.sizes());
    }
    
};