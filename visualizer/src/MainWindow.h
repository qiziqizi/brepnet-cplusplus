#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QSplitter>
#include <QPushButton>
#include <QLabel>
#include <QTextEdit>
#include <QGroupBox>
#include <QVBoxLayout>
#include <QScrollArea>
#include <memory>

#include "OCCTViewer.h"
#include "StepLoader.h"
#include "ColorMapper.h"
#include "FaceClassifier.h"

/**
 * 主窗口
 * 功能：
 * 1. 组织整体UI布局（3D视图 + 控制面板）
 * 2. 协调各模块交互
 * 3. 管理工作流程（加载 → 显示 → 预测 → 着色）
 */
class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow();

private slots:
    // 按钮响应槽
    void onLoadFile();
    void onRunPrediction();
    void onExportResults();
    void onLoadLabels();

    // 面选择响应槽
    void onFaceSelected(int faceIndex);

private:
    void setupUI();
    void setupConnections();
    void updateModelInfo();
    void updatePredictionResults();
    void displayStatistics();
    void updateComparisonResults();
    std::vector<int> loadLabelsFromFile(const QString& filePath);

    // UI组件
    QSplitter* mainSplitter_;
    OCCTViewer* viewer_;

    // 控制面板组件
    QPushButton* btnLoadFile_;
    QPushButton* btnRunPrediction_;
    QPushButton* btnExportResults_;
    QPushButton* btnLoadLabels_;

    QLabel* lblFileName_;
    QLabel* lblNumFaces_;
    QLabel* lblStatus_;
    QLabel* lblSelectedFace_;
    QLabel* lblAccuracy_;

    QTextEdit* txtStatistics_;

    // 数据模块
    std::unique_ptr<StepLoader> loader_;
    std::unique_ptr<ColorMapper> colorMapper_;
    std::unique_ptr<FaceClassifier> classifier_;

    // 当前状态
    QString currentFilePath_;
    std::vector<int> predictions_;
    std::vector<int> groundTruthLabels_;
    std::vector<int> errorFaceIndices_;
    bool modelLoaded_;
};

#endif // MAINWINDOW_H
