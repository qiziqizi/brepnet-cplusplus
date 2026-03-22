#include "MainWindow.h"
#include <QApplication>
#include <iostream>

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);

    // 设置应用程序信息
    app.setApplicationName("BRepNet Visualizer");
    app.setOrganizationName("BRepNet");
    app.setApplicationVersion("1.0");

    std::cout << "启动 BRepNet 可视化工具..." << std::endl;

    // 创建主窗口
    MainWindow mainWindow;
    mainWindow.show();

    return app.exec();
}
