#ifndef OUTPUT_LOGGER_H
#define OUTPUT_LOGGER_H

#include <iostream>
#include <fstream>
#include <streambuf>
#include <string>

/**
 * 输出重定向类
 * 功能：同时将输出写到控制台和文件
 * 用法：在main函数开始时创建 OutputLogger logger("output.txt");
 */
class TeeBuf : public std::streambuf {
private:
    std::streambuf* console_buf;
    std::streambuf* file_buf;

public:
    TeeBuf(std::streambuf* console, std::streambuf* file)
        : console_buf(console), file_buf(file) {}

protected:
    virtual int overflow(int c) override {
        if (c == EOF) {
            return !EOF;
        } else {
            int const r1 = console_buf->sputc(c);
            int const r2 = file_buf->sputc(c);
            return (r1 == EOF || r2 == EOF) ? EOF : c;
        }
    }

    virtual int sync() override {
        int r1 = console_buf->pubsync();
        int r2 = file_buf->pubsync();
        return (r1 == 0 && r2 == 0) ? 0 : -1;
    }
};

class OutputLogger {
private:
    std::ofstream file;
    TeeBuf* tee_buf;
    std::streambuf* old_cout_buf;
    std::streambuf* old_cerr_buf;

public:
    OutputLogger(const std::string& filename) {
        // 打开日志文件
        file.open(filename, std::ios::out);

        if (file.is_open()) {
            // 保存原始的cout buffer
            old_cout_buf = std::cout.rdbuf();
            old_cerr_buf = std::cerr.rdbuf();

            // 创建tee buffer（同时写到控制台和文件）
            tee_buf = new TeeBuf(old_cout_buf, file.rdbuf());

            // 重定向cout和cerr到tee buffer
            std::cout.rdbuf(tee_buf);
            std::cerr.rdbuf(tee_buf);

            std::cout << "=== Output logging started ===" << std::endl;
            std::cout << "Log file: " << filename << std::endl;
            std::cout << "================================" << std::endl;
            std::cout << std::endl;
        } else {
            std::cerr << "Failed to open log file: " << filename << std::endl;
        }
    }

    ~OutputLogger() {
        if (file.is_open()) {
            std::cout << std::endl;
            std::cout << "================================" << std::endl;
            std::cout << "=== Output logging ended ===" << std::endl;

            // 恢复原始的cout buffer
            std::cout.rdbuf(old_cout_buf);
            std::cerr.rdbuf(old_cerr_buf);

            delete tee_buf;
            file.close();
        }
    }
};

#endif // OUTPUT_LOGGER_H
