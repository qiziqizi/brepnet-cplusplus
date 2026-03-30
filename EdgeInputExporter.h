#ifndef EDGE_INPUT_EXPORTER_H
#define EDGE_INPUT_EXPORTER_H

#include "BRepPipeline.h"
#include <fstream>
#include <iomanip>

/**
 * Edge Input Diagnostic Exporter
 * Purpose: Export raw edge grid input to diagnose why Edge 38 has large error
 */
class EdgeInputExporter {
public:
    static void export_edge_grids(BRepPipeline& pipeline, const std::string& step_file_name) {
        std::string output_dir = "cpp_edge_grid_debug";

        if (!std::filesystem::exists(output_dir)) {
            std::filesystem::create_directory(output_dir);
        }

        // Export EdgeGridsLocal tensor
        std::string filename = output_dir + "/" + step_file_name + "_edge_grids_local.txt";
        std::ofstream file(filename, std::ios::out);

        if (!file.is_open()) {
            std::cerr << "[Error] Cannot create file: " << filename << std::endl;
            return;
        }

        // UTF-8 BOM
        file << "\xEF\xBB\xBF";
        file << std::scientific << std::setprecision(20);

        auto& edge_grids = pipeline.EdgeGridsLocal;

        if (!edge_grids.defined()) {
            std::cerr << "[Error] EdgeGridsLocal not defined" << std::endl;
            return;
        }

        int num_edges = edge_grids.sizes_[0];
        int num_channels = edge_grids.sizes_[1];  // Should be 13
        int num_points = edge_grids.sizes_[2];    // Should be 10

        std::cout << "[EdgeInputExporter] Exporting " << num_edges << " edges" << std::endl;
        std::cout << "  Shape: [" << num_edges << ", " << num_channels << ", " << num_points << "]" << std::endl;

        // Export each edge as one row (flattened)
        for (int e = 0; e < num_edges; ++e) {
            for (int c = 0; c < num_channels; ++c) {
                for (int p = 0; p < num_points; ++p) {
                    if (c > 0 || p > 0) file << " ";
                    file << edge_grids.at({e, c, p});
                }
            }
            file << "\n";
        }

        file.close();
        std::cout << "[EdgeInputExporter] Saved to: " << filename << std::endl;
    }

    static void export_edge_representatives(BRepPipeline& pipeline, const std::string& step_file_name) {
        std::string output_dir = "cpp_edge_grid_debug";

        if (!std::filesystem::exists(output_dir)) {
            std::filesystem::create_directory(output_dir);
        }

        std::string filename = output_dir + "/" + step_file_name + "_edge_representatives.txt";
        std::ofstream file(filename, std::ios::out);

        if (!file.is_open()) {
            std::cerr << "[Error] Cannot create file: " << filename << std::endl;
            return;
        }

        // UTF-8 BOM
        file << "\xEF\xBB\xBF";

        int num_edges = pipeline.unique_edges.Extent();

        // Reconstruct edge_representatives logic from BRepPipeline.h:993-1001
        std::vector<int> edge_representatives(num_edges, -1);
        for (const auto& c : pipeline.coedges) {
            int eid = c.edge_idx;
            if (eid >= 0 && eid < num_edges) {
                if (edge_representatives[eid] == -1 || c.orientation == true) {
                    edge_representatives[eid] = c.id;
                }
            }
        }

        file << "# Edge Representative Coedges\n";
        file << "# Format: edge_id coedge_id orientation\n";

        for (int e = 0; e < num_edges; ++e) {
            int cid = edge_representatives[e];
            bool orientation = (cid >= 0 && cid < (int)pipeline.coedges.size()) ?
                               pipeline.coedges[cid].orientation : false;

            file << e << " " << cid << " " << (orientation ? "true" : "false") << "\n";
        }

        file.close();
        std::cout << "[EdgeInputExporter] Saved edge representatives to: " << filename << std::endl;

        // Print Edge 38 specifically
        if (38 < num_edges) {
            int cid = edge_representatives[38];
            std::cout << "\n[DEBUG] Edge 38 uses coedge " << cid;
            if (cid >= 0 && cid < (int)pipeline.coedges.size()) {
                std::cout << " (orientation=" << (pipeline.coedges[cid].orientation ? "true" : "false") << ")";
            }
            std::cout << std::endl;
        }
    }

    static void export_all_coedge_grids(BRepPipeline& pipeline, const std::string& step_file_name) {
        std::string output_dir = "cpp_edge_grid_debug";

        if (!std::filesystem::exists(output_dir)) {
            std::filesystem::create_directory(output_dir);
        }

        std::string filename = output_dir + "/" + step_file_name + "_all_coedge_grids.txt";
        std::ofstream file(filename, std::ios::out);

        if (!file.is_open()) {
            std::cerr << "[Error] Cannot create file: " << filename << std::endl;
            return;
        }

        // UTF-8 BOM
        file << "\xEF\xBB\xBF";
        file << std::scientific << std::setprecision(20);

        auto& coedge_grids = pipeline.CoedgeGridsLocal;

        if (!coedge_grids.defined()) {
            std::cerr << "[Error] CoedgeGridsLocal not defined" << std::endl;
            return;
        }

        int num_coedges = coedge_grids.sizes_[0];
        int num_channels = coedge_grids.sizes_[1];
        int num_points = coedge_grids.sizes_[2];

        std::cout << "[EdgeInputExporter] Exporting all " << num_coedges << " coedge grids" << std::endl;

        // Export each coedge as one row
        for (int c = 0; c < num_coedges; ++c) {
            for (int ch = 0; ch < num_channels; ++ch) {
                for (int p = 0; p < num_points; ++p) {
                    if (ch > 0 || p > 0) file << " ";
                    file << coedge_grids.at({c, ch, p});
                }
            }
            file << "\n";
        }

        file.close();
        std::cout << "[EdgeInputExporter] Saved all coedge grids to: " << filename << std::endl;
    }
};

#endif // EDGE_INPUT_EXPORTER_H
