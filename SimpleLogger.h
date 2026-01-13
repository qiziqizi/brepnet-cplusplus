#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <filesystem>
#include <chrono>
#include <ctime>
#include <iomanip>

// Windows API (用于编码转换和文件夹操作)
#ifdef _WIN32
#include <direct.h>
#include <windows.h>
#else
#include <sys/stat.h>
#endif

namespace Tools {

    class AutoLogger {
        // 内部类：双缓冲流（同时输出到控制台和内存）
        class TeeBuf : public std::streambuf {
            std::streambuf* sb1;
            std::streambuf* sb2;
        public:
            TeeBuf(std::streambuf* s1, std::streambuf* s2) : sb1(s1), sb2(s2) {}

            virtual int overflow(int c) override {
                if (c == EOF) return !EOF;
                int r1 = sb1->sputc(c);
                int r2 = sb2->sputc(c);
                return (r1 == EOF || r2 == EOF) ? EOF : c;
            }

            virtual int sync() override {
                int r1 = sb1->pubsync();
                int r2 = sb2->pubsync();
                return (r1 == 0 && r2 == 0) ? 0 : -1;
            }
        };

        std::string log_dir;
        std::string log_file;
        std::stringstream buffer;
        std::streambuf* old_cout_buf;
        std::streambuf* old_cerr_buf;
        TeeBuf* tee_cout;
        TeeBuf* tee_cerr;

#ifdef _WIN32
        // GBK 转 UTF-8（Windows专用）
        std::string GBKToUTF8(const std::string& gbk) {
            if (gbk.empty()) return "";

            // 1. GBK (CP_ACP) → UTF-16 (Wide Char)
            int wlen = MultiByteToWideChar(CP_ACP, 0, gbk.c_str(), -1, nullptr, 0);
            if (wlen <= 0) return gbk;

            std::wstring wstr(wlen, 0);
            MultiByteToWideChar(CP_ACP, 0, gbk.c_str(), -1, &wstr[0], wlen);

            // 2. UTF-16 → UTF-8
            int utf8len = WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), -1, nullptr, 0, nullptr, nullptr);
            if (utf8len <= 0) return gbk;

            std::string utf8(utf8len - 1, 0);  // 减1去掉null terminator
            WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), -1, &utf8[0], utf8len, nullptr, nullptr);

            return utf8;
        }
#endif

    public:
        AutoLogger(const std::string& directory = "logs",
            const std::string& filename = "execution_history.log")
            : log_dir(directory), log_file(filename) {

            // ★ 不修改控制台编码，保持 GBK（删除了之前的 SetConsoleOutputCP）

            // 1. 创建日志目录
            std::filesystem::path dir_path(log_dir);
            if (!std::filesystem::exists(dir_path)) {
                std::filesystem::create_directories(dir_path);
            }

            // 2. 劫持 cout（双缓冲：同时输出到控制台和 buffer）
            old_cout_buf = std::cout.rdbuf();
            tee_cout = new TeeBuf(old_cout_buf, buffer.rdbuf());
            std::cout.rdbuf(tee_cout);

            // 3. 劫持 cerr（让错误信息也进日志）
            old_cerr_buf = std::cerr.rdbuf();
            tee_cerr = new TeeBuf(old_cerr_buf, buffer.rdbuf());
            std::cerr.rdbuf(tee_cerr);
        }

        ~AutoLogger() {
            // 4. 恢复劫持（必须在写文件前恢复，否则输出会递归）
            std::cout.rdbuf(old_cout_buf);
            std::cerr.rdbuf(old_cerr_buf);
            delete tee_cout;
            delete tee_cerr;

            // 5. 写入文件
            try {
                std::filesystem::path full_path = std::filesystem::path(log_dir) / log_file;

                // 获取缓冲区内容（GBK 编码）
                std::string log_content = buffer.str();

                // 以二进制模式打开文件（追加）
                std::ofstream file(full_path, std::ios::binary | std::ios::app);
                if (file.is_open()) {
                    // 如果是新文件，写入 UTF-8 BOM
                    file.seekp(0, std::ios::end);
                    if (file.tellp() == 0) {
                        const char bom[] = "\xEF\xBB\xBF";  // UTF-8 BOM
                        file.write(bom, 3);
                    }

                    // 生成时间戳
                    auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
                    struct tm local_tm;
#ifdef _WIN32
                    localtime_s(&local_tm, &now);
#else
                    localtime_r(&now, &local_tm);
#endif

                    // 构建头部信息
                    std::ostringstream header;
                    header << "\n============================================================\n";
                    header << "[RUN RECORD] " << std::put_time(&local_tm, "%Y-%m-%d %H:%M:%S") << "\n";
                    header << "============================================================\n";

                    std::string header_str = header.str();

#ifdef _WIN32
                    // ★ 只在写文件时转换为 UTF-8
                    header_str = GBKToUTF8(header_str);
                    log_content = GBKToUTF8(log_content);
#endif

                    // 写入文件
                    file << header_str;
                    file << log_content;

                    // 恢复原始的 cout 后打印提示
                    std::cout << "\n[System] Log saved to " << full_path.string() << std::endl;
                }
            }
            catch (const std::exception& e) {
                std::cerr << "[System] Failed to save log file: " << e.what() << std::endl;
            }
            catch (...) {
                std::cerr << "[System] Failed to save log file (unknown error)." << std::endl;
            }
        }
    };

    // 辅助工具：获取绝对路径
    inline std::string GetAbsPath(const std::string& path) {
        try {
            if (std::filesystem::exists(path))
                return std::filesystem::absolute(path).string();
            return path + " (Not Found)";
        }
        catch (...) {
            return path;
        }
    }

} // namespace Tools
