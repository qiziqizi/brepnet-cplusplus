#pragma once
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <iostream>

class DebugControl {
public:
    bool debug_mode = false;
    bool export_mode = false;
    std::vector<std::string> targets;
    std::string current_file;

    static DebugControl& instance() {
        static DebugControl inst;
        return inst;
    }

    void parse(int argc, char* argv[]) {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--debug") {
                debug_mode = true;
            } else if (arg == "--export") {
                export_mode = true;
            } else if (arg == "--target" && i + 1 < argc) {
                ++i;
                std::string target_str = argv[i];
                std::stringstream ss(target_str);
                std::string item;
                while (std::getline(ss, item, ',')) {
                    if (!item.empty()) targets.push_back(item);
                }
            }
        }
        if (debug_mode) export_mode = true;
        if (debug_mode || export_mode) {
            std::cout << "=== DEBUG CONTROL ===" << std::endl;
            std::cout << "  debug_mode: " << (debug_mode ? "ON" : "OFF") << std::endl;
            std::cout << "  export_mode: " << (export_mode ? "ON" : "OFF") << std::endl;
            if (!targets.empty()) {
                std::cout << "  targets: ";
                for (size_t i = 0; i < targets.size(); ++i) {
                    if (i > 0) std::cout << ", ";
                    std::cout << targets[i];
                }
                std::cout << std::endl;
            } else {
                std::cout << "  targets: ALL FILES" << std::endl;
            }
            std::cout << "=====================" << std::endl;
        }
    }

    void setCurrentFile(const std::string& filename) {
        current_file = filename;
    }

    bool shouldDebug() const {
        if (!debug_mode) return false;
        if (targets.empty()) return true;
        return std::find(targets.begin(), targets.end(), current_file) != targets.end();
    }

    bool shouldExport() const {
        if (!export_mode) return false;
        if (targets.empty()) return true;
        return std::find(targets.begin(), targets.end(), current_file) != targets.end();
    }

    static bool alwaysOn() { return true; }

private:
    DebugControl() = default;
};

// ============================================================================
// 便捷宏
// ============================================================================

// DBG_LOG: 受调试开关控制的终端输出
// 用法: DBG_LOG << "[Layer 0] ..." << std::endl;
#define DBG_LOG if (DebugControl::instance().shouldDebug()) std::cout

// DBG_CERR: 受调试开关控制的stderr输出
#define DBG_CERR if (DebugControl::instance().shouldDebug()) std::cerr

// DBG_PRINTF: 受调试开关控制的printf
// 用法: DBG_PRINTF("[DEBUG] value=%d\n", val);
#define DBG_PRINTF(...) do { if (DebugControl::instance().shouldDebug()) printf(__VA_ARGS__); } while(0)

// DBG_FPRINTF: 受调试开关控制的fprintf
// 用法: DBG_FPRINTF(stderr, "[DEBUG] value=%d\n", val);
#define DBG_FPRINTF(fp, ...) do { if (DebugControl::instance().shouldDebug()) fprintf(fp, __VA_ARGS__); } while(0)

// EXPORT_ENABLED: 判断是否应该导出中间文件
#define EXPORT_ENABLED (DebugControl::instance().shouldExport())

// ERR_LOG: 错误输出，始终开启（不受调试开关控制）
// 用于 "[Error] Processing failed" 等关键错误
#define ERR_LOG std::cerr

// INFO_LOG: 进度信息，始终开启
// 用于 "step0001 -> [✓] F:12 E:18" 等进度信息
#define INFO_LOG std::cout
