#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QSplitter>
#include <QPushButton>
#include <QLabel>
#include <QTextEdit>
#include <QGroupBox>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QScrollArea>
#include <memory>

#include "OCCTViewer.h"
#include "StepLoader.h"
#include "ColorMapper.h"
#include "FaceClassifier.h"

/**
 * 工作模式枚举
 */
enum class WorkMode {
    None,           // 未操作
    Prediction,     // 预测模式
    ManualLabeling  // 人工标注模式
};

/**
 * 主窗口
 * 功能：
 * 1. 组织整体UI布局（3D视图 + 控制面板）
 * 2. 协调各模块交互
 * 3. 管理工作流程（加载 → 显示 → 预测/标注 → 着色）
 */
class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow();

private slots:
    // 文件操作
    void onLoadFile();

    // 预测操作
    void onRunPrediction();
    void onLoadPredictionLabels();
    void onExportPrediction();

    // 人工标注
    void onLoadManualLabels();
    void onModifyFaceClass();
    void onExportManualLabels();

    // 重置
    void onReset();

    // 面选择/悬停响应槽
    void onFaceSelected(int faceIndex);
    void onFaceHovered(int faceIndex, int mouseX, int mouseY);

private:
    void setupUI();
    void setupConnections();
    void setWorkMode(WorkMode mode);
    void updateModelInfo();
    void updatePredictionResults();
    void updateManualLabelingResults();
    void updateComparisonResults();
    std::vector<int> loadLabelsFromFile(const QString& filePath);
    void refreshLegendLayout();

protected:
    void resizeEvent(QResizeEvent* event) override;

    // UI组件
    QSplitter* mainSplitter_;
    OCCTViewer* viewer_;

    // 文件操作区
    QPushButton* btnLoadFile_;

    // 预测操作区
    QPushButton* btnRunPrediction_;
    QPushButton* btnLoadPredictionLabels_;
    QPushButton* btnExportPrediction_;
    QLabel* lblPredictionAccuracy_;

    // 人工标注区
    QPushButton* btnLoadManualLabels_;
    QPushButton* btnModifyFaceClass_;
    QPushButton* btnExportManualLabels_;

    // 重置区
    QPushButton* btnReset_;

    // 模型信息区
    QLabel* lblFileName_;
    QLabel* lblNumFaces_;
    QLabel* lblCurrentMode_;
    QLabel* lblSelectedFace_;

    // 悬停信息区
    QLabel* lblHoveredFace_;
    QLabel* lblHoverEdgeCount_;
    QLabel* lblHoverEdgeTypes_;

    QTextEdit* txtStatistics_;

    // 颜色图例组件
    QScrollArea* legendScroll_;
    QGridLayout* legendGrid_;
    std::vector<QWidget*> legendItems_;

    // 数据模块
    std::unique_ptr<StepLoader> loader_;
    std::unique_ptr<ColorMapper> colorMapper_;
    std::unique_ptr<FaceClassifier> classifier_;

    // 当前状态
    WorkMode currentMode_;
    QString currentFilePath_;
    std::vector<int> predictions_;
    std::vector<int> groundTruthLabels_;
    std::vector<int> manualLabels_;
    std::vector<int> errorFaceIndices_;
    bool modelLoaded_;
};

#endif // MAINWINDOW_H
