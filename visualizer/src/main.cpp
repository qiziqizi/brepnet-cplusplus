#include "MainWindow.h"
#include "PrecisionUtils.h"
#include "BRepTorch.h"
#include <QApplication>
#include <iostream>

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);

    // 设置应用程序信息
    app.setApplicationName("BRepNet Visualizer");
    app.setOrganizationName("BRepNet");
    app.setApplicationVersion("1.0");

    std::cout << "启动 BRepNet 可视化工具..." << std::endl;
    const bool c_avx2 = breptorch::cpu_supports_avx2_fma();
    std::cout << "[CPU] AVX2+FMA: " << (c_avx2 ? "supported (using AVX2+FMA path)"
                                               : "not supported (using SSE2 fallback)") << std::endl;

    // 解析 --precision 参数（与批量工具 brepnet.exe 一致，默认 fp32）
    breptorch::WeightPrecision precision = breptorch::WeightPrecision::FP32;
    for (int i = 1; i < argc - 1; ++i) {
        if (std::string(argv[i]) == "--precision") {
            try {
                precision = breptorch::parse_precision(argv[i + 1]);
                std::cout << "权重精度: " << argv[i + 1] << std::endl;
            } catch (...) {
                std::cerr << "无效的精度参数: " << argv[i + 1] << " (使用默认 fp32)" << std::endl;
            }
            break;
        }
    }

    // 创建主窗口
    MainWindow mainWindow(precision);
    mainWindow.show();

    return app.exec();
}