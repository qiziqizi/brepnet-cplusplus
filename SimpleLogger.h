#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <filesystem>
#include <chrono>
#include <ctime>

// 如果需要 Windows API (例如创建文件夹)，在这里引用
#ifdef _WIN32
#include <direct.h>
#include <windows.h>
#else
#include <sys/stat.h>
#endif

namespace Tools {

    class AutoLogger {
        // 内部类：双向流缓冲区
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
        std::streambuf* old_cerr_buf; // 建议连 cerr 也一起劫持
        TeeBuf* tee_cout;
        TeeBuf* tee_cerr;

    public:
        AutoLogger(const std::string& directory = "logs", const std::string& filename = "execution_history.log")
            : log_dir(directory), log_file(filename) {

            // 1. 创建目录
            std::filesystem::path dir_path(log_dir);
            if (!std::filesystem::exists(dir_path)) {
                std::filesystem::create_directories(dir_path);
            }

            // 2. 劫持 cout
            old_cout_buf = std::cout.rdbuf();
            tee_cout = new TeeBuf(old_cout_buf, buffer.rdbuf());
            std::cout.rdbuf(tee_cout);

            // 3. 劫持 cerr (让错误信息也进日志)
            old_cerr_buf = std::cerr.rdbuf();
            tee_cerr = new TeeBuf(old_cerr_buf, buffer.rdbuf());
            std::cerr.rdbuf(tee_cerr);
        }

        ~AutoLogger() {
            // 4. 还原现场 (必须在写文件前还原，否则可能死锁或奔溃)
            std::cout.rdbuf(old_cout_buf);
            std::cerr.rdbuf(old_cerr_buf);
            delete tee_cout;
            delete tee_cerr;

            // 5. 写入文件
            try {
                // 拼接完整路径
                std::filesystem::path full_path = std::filesystem::path(log_dir) / log_file;

                std::ofstream file(full_path, std::ios::app);
                if (file.is_open()) {
                    auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
                    struct tm local_tm;
#ifdef _WIN32
                    localtime_s(&local_tm, &now);
#else
                    localtime_r(&now, &local_tm);
#endif

                    file << "\n============================================================\n";
                    file << "[RUN RECORD] " << std::put_time(&local_tm, "%Y-%m-%d %H:%M:%S") << "\n";
                    file << "============================================================\n";

                    file << buffer.str();

                    // 这里用原来的 cout 打印提示，因为上面的劫持已经取消了
                    std::cout << "\n[System] Log saved to " << full_path.string() << std::endl;
                }
            }
            catch (...) {
                std::cerr << "[System] Failed to save log file." << std::endl;
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
        catch (...) { return path; }
    }

}
