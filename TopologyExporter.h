/**
 * TopologyExporter.h
 * 
 * 导出拓扑信息用于与 Python 端对比
 * 功能：
 * 1. 导出 coedge 基本信息 (id, face, mate, edge)
 * 2. 导出 face -> coedges 映射
 * 3. 导出 mate 数组
 * 
 * 用法：
 *   // 在需要的地方添加以下代码：
 *   TopologyExporter exporter("cpp_topology");
 *   exporter.export_topology(pipeline, step_file_name);
 */

#pragma once
#ifndef TOPOLOGY_EXPORTER_H
#define TOPOLOGY_EXPORTER_H

#include "BRepPipeline.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <map>
#include <set>

namespace fs = std::filesystem;

class TopologyExporter {
public:
    TopologyExporter(const std::string& base_dir = "cpp_topology")
        : base_dir_(base_dir) {
        if (!fs::exists(base_dir_)) {
            fs::create_directories(base_dir_);
        }
    }

    /**
     * 导出拓扑信息
     */
    void export_topology(BRepPipeline& pipeline, const std::string& file_name) {
        std::string output_dir = base_dir_ + "/" + file_name;
        if (!fs::exists(output_dir)) {
            fs::create_directories(output_dir);
        }

        std::cout << "\n[Topology Export] Exporting to " << output_dir << std::endl;

        // 1. 导出共边详细信息
        export_coedge_info(pipeline, output_dir + "/" + file_name + "_coedge_info.csv");

        // 2. 导出 face -> coedges 映射
        export_face_coedges_mapping(pipeline, output_dir + "/" + file_name + "_face_coedges.csv");

        // 3. 导出 mate 数组
        export_mate_array(pipeline, output_dir + "/" + file_name + "_mate_array.csv");

        // 4. 导出拓扑摘要
        export_summary(pipeline, output_dir + "/" + file_name + "_summary.txt");

        std::cout << "[Topology Export] Done!" << std::endl;
    }

private:
    std::string base_dir_;

    /**
     * 导出共边详细信息
     * 格式: coedge_id,parent_face_id,mate_coedge_id,mate_face_id,edge_id
     */
    void export_coedge_info(BRepPipeline& pipeline, const std::string& path) {
        std::ofstream file(path);
        file << std::setprecision(15);

        file << "coedge_id,parent_face_id,mate_coedge_id,mate_face_id,edge_id,orientation\n";

        int num_coedges = (int)pipeline.coedges.size();
        for (int i = 0; i < num_coedges; ++i) {
            const auto& coedge = pipeline.coedges[i];
            int mate_coedge_id = coedge.mate_idx;
            int mate_face_id = (mate_coedge_id >= 0 && mate_coedge_id < num_coedges) 
                ? pipeline.coedges[mate_coedge_id].face_idx 
                : -1;

            file << i << ","
                 << coedge.face_idx << ","
                 << mate_coedge_id << ","
                 << mate_face_id << ","
                 << coedge.edge_idx << ","
                 << (coedge.orientation ? "true" : "false") << "\n";
        }

        file.close();
        std::cout << "  [OK] Coedge info: " << path << std::endl;
    }

    /**
     * 导出 face -> coedges 映射
     */
    void export_face_coedges_mapping(BRepPipeline& pipeline, const std::string& path) {
        int num_faces = pipeline.unique_faces.Extent();
        int num_coedges = (int)pipeline.coedges.size();

        // 构建 face -> coedges 映射
        std::map<int, std::vector<int>> face_to_coedges;
        for (int face_id = 0; face_id < num_faces; ++face_id) {
            face_to_coedges[face_id] = {};
        }
        for (int i = 0; i < num_coedges; ++i) {
            int face_id = pipeline.coedges[i].face_idx;
            if (face_id >= 0 && face_id < num_faces) {
                face_to_coedges[face_id].push_back(i);
            }
        }

        // 导出 CSV
        std::ofstream file(path);
        file << std::setprecision(15);

        file << "face_id,num_coedges,coedge_list\n";

        for (int face_id = 0; face_id < num_faces; ++face_id) {
            const auto& coedges = face_to_coedges[face_id];
            file << face_id << "," << coedges.size() << ",\"[";
            for (size_t i = 0; i < coedges.size(); ++i) {
                if (i > 0) file << ",";
                file << coedges[i];
            }
            file << "]\"\n";
        }

        file.close();
        std::cout << "  [OK] Face-Coedges mapping: " << path << std::endl;
    }

    /**
     * 导出 mate 数组
     */
    void export_mate_array(BRepPipeline& pipeline, const std::string& path) {
        std::ofstream file(path);
        file << std::setprecision(15);

        file << "coedge_id,mate_coedge_id\n";

        int num_coedges = (int)pipeline.coedges.size();
        for (int i = 0; i < num_coedges; ++i) {
            file << i << "," << pipeline.coedges[i].mate_idx << "\n";
        }

        file.close();
        std::cout << "  [OK] Mate array: " << path << std::endl;
    }

    /**
     * 导出拓扑摘要
     */
    void export_summary(BRepPipeline& pipeline, const std::string& path) {
        std::ofstream file(path);

        int num_faces = pipeline.unique_faces.Extent();
        int num_coedges = (int)pipeline.coedges.size();
        int num_edges = pipeline.unique_edges.Extent();

        file << "C++ 拓扑数据摘要\n";
        file << "=" << std::string(60, '=') << "\n\n";
        file << "面总数: " << num_faces << "\n";
        file << "Coedge总数: " << num_coedges << "\n";
        file << "Edge总数: " << num_edges << "\n\n";

        // 构建 face -> coedges 映射
        std::map<int, std::vector<int>> face_to_coedges;
        for (int face_id = 0; face_id < num_faces; ++face_id) {
            face_to_coedges[face_id] = {};
        }
        for (int i = 0; i < num_coedges; ++i) {
            int face_id = pipeline.coedges[i].face_idx;
            if (face_id >= 0 && face_id < num_faces) {
                face_to_coedges[face_id].push_back(i);
            }
        }

        // 共边数统计
        std::vector<int> coedge_counts;
        for (const auto& pair : face_to_coedges) {
            coedge_counts.push_back((int)pair.second.size());
        }

        file << "每个面的共边数统计:\n";
        file << "  最小: " << *std::min_element(coedge_counts.begin(), coedge_counts.end()) << "\n";
        file << "  最大: " << *std::max_element(coedge_counts.begin(), coedge_counts.end()) << "\n";
        file << "  平均: " << std::fixed << std::setprecision(2) 
             << (double)std::accumulate(coedge_counts.begin(), coedge_counts.end(), 0) / coedge_counts.size() << "\n\n";

        // 大面列表 (>30 coedges)
        std::vector<int> big_faces;
        for (const auto& pair : face_to_coedges) {
            if ((int)pair.second.size() > 30) {
                big_faces.push_back(pair.first);
            }
        }
        std::sort(big_faces.begin(), big_faces.end());

        file << "大面 (>30 coedges): " << big_faces.size() << " 个\n";
        if (!big_faces.empty()) {
            file << "  Face IDs: ";
            for (size_t i = 0; i < std::min((size_t)10, big_faces.size()); ++i) {
                if (i > 0) file << ", ";
                file << big_faces[i];
            }
            if (big_faces.size() > 10) {
                file << ", ...";
            }
            file << "\n";
        }

        // 列出每个大面的详细信息
        if (!big_faces.empty()) {
            file << "\n大面详细信息:\n";
            for (int face_id : big_faces) {
                const auto& coedges = face_to_coedges[face_id];
                file << "  Face " << face_id << ": " << coedges.size() << " coedges [";
                for (size_t i = 0; i < std::min((size_t)5, coedges.size()); ++i) {
                    if (i > 0) file << ", ";
                    file << coedges[i];
                }
                if (coedges.size() > 5) {
                    file << ", ...";
                }
                file << "]\n";
            }
        }

        file.close();
        std::cout << "  [OK] Summary: " << path << std::endl;
    }
};

#endif // TOPOLOGY_EXPORTER_H
