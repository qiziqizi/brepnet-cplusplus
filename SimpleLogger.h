#pragma once

// Windows API 宏定义（必须在包含 Windows.h 之前）
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <filesystem>
#include <chrono>
#include <ctime>
#include <iomanip>

// Windows API (用于编码转换，文件操作等)
#ifdef _WIN32
#include <direct.h>
#include <Windows.h>
#else
#include <sys/stat.h>
#endif

namespace Tools {

    class AutoLogger {
        // �ڲ��ࣺ˫��������ͬʱ���������̨���ڴ棩
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
        // GBK ת UTF-8��Windowsר�ã�
        std::string GBKToUTF8(const std::string& gbk) {
            if (gbk.empty()) return "";

            // 1. GBK (CP_ACP) �� UTF-16 (Wide Char)
            int wlen = MultiByteToWideChar(CP_ACP, 0, gbk.c_str(), -1, nullptr, 0);
            if (wlen <= 0) return gbk;

            std::wstring wstr(wlen, 0);
            MultiByteToWideChar(CP_ACP, 0, gbk.c_str(), -1, &wstr[0], wlen);

            // 2. UTF-16 �� UTF-8
            int utf8len = WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), -1, nullptr, 0, nullptr, nullptr);
            if (utf8len <= 0) return gbk;

            std::string utf8(utf8len - 1, 0);  // ��1ȥ��null terminator
            WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), -1, &utf8[0], utf8len, nullptr, nullptr);

            return utf8;
        }
#endif

    public:
        AutoLogger(const std::string& directory = "logs",
            const std::string& filename = "execution_history.log")
            : log_dir(directory), log_file(filename) {

            // �� ���޸Ŀ���̨���룬���� GBK��ɾ����֮ǰ�� SetConsoleOutputCP��

            // 1. ������־Ŀ¼
            std::filesystem::path dir_path(log_dir);
            if (!std::filesystem::exists(dir_path)) {
                std::filesystem::create_directories(dir_path);
            }

            // 2. �ٳ� cout��˫���壺ͬʱ���������̨�� buffer��
            old_cout_buf = std::cout.rdbuf();
            tee_cout = new TeeBuf(old_cout_buf, buffer.rdbuf());
            std::cout.rdbuf(tee_cout);

            // 3. �ٳ� cerr���ô�����ϢҲ����־��
            old_cerr_buf = std::cerr.rdbuf();
            tee_cerr = new TeeBuf(old_cerr_buf, buffer.rdbuf());
            std::cerr.rdbuf(tee_cerr);
        }

        ~AutoLogger() {
            // 4. �ָ��ٳ֣�������д�ļ�ǰ�ָ������������ݹ飩
            std::cout.rdbuf(old_cout_buf);
            std::cerr.rdbuf(old_cerr_buf);
            delete tee_cout;
            delete tee_cerr;

            // 5. д���ļ�
            try {
                std::filesystem::path full_path = std::filesystem::path(log_dir) / log_file;

                // ��ȡ���������ݣ�GBK ���룩
                std::string log_content = buffer.str();

                // �Զ�����ģʽ���ļ���׷�ӣ�
                std::ofstream file(full_path, std::ios::binary | std::ios::app);
                if (file.is_open()) {
                    // ��������ļ���д�� UTF-8 BOM
                    file.seekp(0, std::ios::end);
                    if (file.tellp() == 0) {
                        const char bom[] = "\xEF\xBB\xBF";  // UTF-8 BOM
                        file.write(bom, 3);
                    }

                    // ����ʱ���
                    auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
                    struct tm local_tm;
#ifdef _WIN32
                    localtime_s(&local_tm, &now);
#else
                    localtime_r(&now, &local_tm);
#endif

                    // ����ͷ����Ϣ
                    std::ostringstream header;
                    header << "\n============================================================\n";
                    header << "[RUN RECORD] " << std::put_time(&local_tm, "%Y-%m-%d %H:%M:%S") << "\n";
                    header << "============================================================\n";

                    std::string header_str = header.str();

#ifdef _WIN32
                    // �� ֻ��д�ļ�ʱת��Ϊ UTF-8
                    header_str = GBKToUTF8(header_str);
                    log_content = GBKToUTF8(log_content);
#endif

                    // д���ļ�
                    file << header_str;
                    file << log_content;

                    // �ָ�ԭʼ�� cout ���ӡ��ʾ
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

    // �������ߣ���ȡ����·��
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
