#include "MainWindow.h"
#include <QFileDialog>
#include <QMessageBox>
#include <QProgressDialog>
#include <QTextStream>
#include <QFile>
#include <QApplication>
#include <QMenuBar>
#include <QStatusBar>
#include <QDir>
#include <QGridLayout>
#include <QScrollArea>
#include <QResizeEvent>
#include <QTimer>
#include <Quantity_Color.hxx>
#include <algorithm>
#include <map>

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
    , modelLoaded_(false) {

    // 调试输出：显示当前工作目录
    std::cout << "[调试] 当前工作目录: " << QDir::currentPath().toStdString() << std::endl;

    // 创建数据模块
    loader_ = std::make_unique<StepLoader>();
    colorMapper_ = std::make_unique<ColorMapper>();
    classifier_ = std::make_unique<FaceClassifier>();

    // 设置UI
    setupUI();
    setupConnections();

    // 自动加载模型 - 基于exe所在目录查找
    QString exeDir = QCoreApplication::applicationDirPath();
    std::cout << "[调试] exe所在目录: " << exeDir.toStdString() << std::endl;
    
    QString weightsPath;
    QStringList possiblePaths = {
        exeDir + "/inference_data/state_dict.npz",
        exeDir + "/../../../inference_data/state_dict.npz"
    };
    
    for (const QString& path : possiblePaths) {
        QFileInfo info(path);
        std::cout << "[调试] 检查路径: " << path.toStdString() 
                  << " -> 存在: " << (info.exists() ? "是" : "否") << std::endl;
        if (info.exists()) {
            weightsPath = path;
            break;
        }
    }
    
    if (!weightsPath.isEmpty()) {
        std::cout << "[MainWindow] 尝试加载模型: " << weightsPath.toStdString() << std::endl;
        if (classifier_->loadModel(weightsPath.toStdString())) {
            modelLoaded_ = true;
            lblStatus_->setText("状态: 模型已加载");
            std::cout << "[MainWindow] 模型加载成功" << std::endl;
        } else {
            lblStatus_->setText("状态: 模型加载失败");
            std::cerr << "[MainWindow] 模型加载失败" << std::endl;
        }
    } else {
        lblStatus_->setText("状态: 未找到模型权重文件");
        std::cerr << "[MainWindow] 未找到模型权重文件" << std::endl;
    }
}

MainWindow::~MainWindow() {
}

void MainWindow::resizeEvent(QResizeEvent* event) {
    QMainWindow::resizeEvent(event);
    refreshLegendLayout();
}

void MainWindow::refreshLegendLayout() {
    if (!legendGrid_ || legendItems_.empty()) {
        return;
    }

    // 清空现有布局
    QLayoutItem* item;
    while ((item = legendGrid_->takeAt(0)) != nullptr) {
        if (!item->widget()) delete item;
    }

    // 每个图例项的固定宽度：234px（20正方形 + 4间距 + 210文字）
    int itemWidth = 234;
    int itemHeight = 24;
    int spacing = 4;
    int availableWidth = legendScroll_->viewport()->width();
    if (availableWidth < 234) availableWidth = 234;  // 至少显示 1 列

    // 计算可以放多少列
    int cols = std::max(1, availableWidth / itemWidth);

    // 计算总行数
    int rows = (27 + cols - 1) / cols;  // 向上取整

    // 添加图例到 grid
    for (int i = 0; i < 27; ++i) {
        int row = i / cols;
        int col = i % cols;
        legendGrid_->addWidget(legendItems_[i], row, col, Qt::AlignTop | Qt::AlignLeft);
    }

    // 计算并设置 legendContent 的实际高度
    int totalHeight = rows * itemHeight + (rows - 1) * spacing;
    legendScroll_->widget()->setFixedHeight(totalHeight);
}

void MainWindow::setupUI() {
    setWindowTitle("BRepNet 可视化工具");
    resize(1400, 900);

    // 创建中心部件
    QWidget* centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);

    // 创建分割器（左右布局）
    mainSplitter_ = new QSplitter(Qt::Horizontal, centralWidget);

    // ========== 左侧：3D视图 ==========
    viewer_ = new OCCTViewer(mainSplitter_);
    mainSplitter_->addWidget(viewer_);

    // ========== 右侧：控制面板 ==========
    QWidget* controlPanel = new QWidget(mainSplitter_);
    QVBoxLayout* panelLayout = new QVBoxLayout(controlPanel);

    // === 区域1: 文件操作 ===
    QGroupBox* fileGroup = new QGroupBox("文件操作", controlPanel);
    QVBoxLayout* fileLayout = new QVBoxLayout(fileGroup);

    btnLoadFile_ = new QPushButton("加载STEP文件", fileGroup);
    fileLayout->addWidget(btnLoadFile_);

    panelLayout->addWidget(fileGroup);

    // === 区域2: 预测操作 ===
    QGroupBox* predGroup = new QGroupBox("预测操作", controlPanel);
    QVBoxLayout* predLayout = new QVBoxLayout(predGroup);

    btnRunPrediction_ = new QPushButton("运行预测", predGroup);
    btnRunPrediction_->setEnabled(false);
    predLayout->addWidget(btnRunPrediction_);

    btnExportResults_ = new QPushButton("导出结果", predGroup);
    btnExportResults_->setEnabled(false);
    predLayout->addWidget(btnExportResults_);

    panelLayout->addWidget(predGroup);

    // === 区域3: 对比验证 ===
    QGroupBox* compareGroup = new QGroupBox("对比验证", controlPanel);
    QVBoxLayout* compareLayout = new QVBoxLayout(compareGroup);

    btnLoadLabels_ = new QPushButton("导入真实标签", compareGroup);
    btnLoadLabels_->setEnabled(false);
    compareLayout->addWidget(btnLoadLabels_);

    lblAccuracy_ = new QLabel("准确率: --", compareGroup);
    compareLayout->addWidget(lblAccuracy_);

    panelLayout->addWidget(compareGroup);

    // === 区域4: 模型信息 ===
    QGroupBox* infoGroup = new QGroupBox("模型信息", controlPanel);
    QVBoxLayout* infoLayout = new QVBoxLayout(infoGroup);

    lblFileName_ = new QLabel("文件: 未加载", infoGroup);
    lblNumFaces_ = new QLabel("面数: 0", infoGroup);
    lblStatus_ = new QLabel("状态: 未加载模型", infoGroup);
    lblSelectedFace_ = new QLabel("选中面: 无", infoGroup);

    infoLayout->addWidget(lblFileName_);
    infoLayout->addWidget(lblNumFaces_);
    infoLayout->addWidget(lblStatus_);
    infoLayout->addWidget(lblSelectedFace_);

    panelLayout->addWidget(infoGroup);

    // === 区域5: 预测结果 ===
    QGroupBox* resultGroup = new QGroupBox("预测结果", controlPanel);
    QVBoxLayout* resultLayout = new QVBoxLayout(resultGroup);

    txtStatistics_ = new QTextEdit(resultGroup);
    txtStatistics_->setReadOnly(true);
    txtStatistics_->setPlainText("等待预测...");
    resultLayout->addWidget(txtStatistics_);

    panelLayout->addWidget(resultGroup);

    // === 区域6: 颜色图例 ===
    QGroupBox* legendGroup = new QGroupBox("颜色图例", controlPanel);
    legendGroup->setStyleSheet("QGroupBox { padding-top: 15px; margin: 0px; } QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0px 5px; }");
    QVBoxLayout* legendOuterLayout = new QVBoxLayout(legendGroup);
    legendOuterLayout->setContentsMargins(0, 0, 0, 0);
    legendOuterLayout->setSpacing(0);

    legendScroll_ = new QScrollArea(legendGroup);
    legendScroll_->setWidgetResizable(true);
    legendScroll_->setFrameShape(QFrame::NoFrame);
    legendScroll_->setStyleSheet("QScrollArea { margin: 0px; padding: 0px; border: none; background: transparent; }");
    legendScroll_->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    legendScroll_->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);

    QWidget* legendContent = new QWidget();
    legendContent->setStyleSheet("QWidget { margin: 0px; padding: 0px; background: transparent; }");
    legendGrid_ = new QGridLayout(legendContent);
    legendGrid_->setSpacing(4);
    legendGrid_->setContentsMargins(4, 0, 0, 0);  // 左边距4px

    // 创建所有图例项（每个项目包含正方形+文字，固定宽度）
    for (int i = 0; i < 27; ++i) {
        // 创建图例项容器
        QWidget* itemWidget = new QWidget();
        QHBoxLayout* itemLayout = new QHBoxLayout(itemWidget);
        itemLayout->setContentsMargins(0, 0, 0, 0);
        itemLayout->setSpacing(4);

        // 正方形
        QLabel* colorSwatch = new QLabel();
        colorSwatch->setFixedSize(20, 20);
        Quantity_Color qc = colorMapper_->getColor(i);
        int r = static_cast<int>(qc.Red() * 255);
        int g = static_cast<int>(qc.Green() * 255);
        int b = static_cast<int>(qc.Blue() * 255);
        colorSwatch->setStyleSheet(
            QString("background-color: rgb(%1,%2,%3); border: 1px solid #888;")
                .arg(r).arg(g).arg(b));

        // 文字
        QLabel* nameLabel = new QLabel(
            QString("%1: %2").arg(i).arg(QString::fromStdString(colorMapper_->getClassName(i))));
        nameLabel->setStyleSheet("font-size: 13px;");
        nameLabel->setFixedWidth(210);

        itemLayout->addWidget(colorSwatch);
        itemLayout->addWidget(nameLabel);
        itemLayout->addStretch();

        // 设置图例项的固定尺寸：宽度234px，高度24px（固定）
        itemWidget->setFixedSize(234, 24);
        legendItems_.push_back(itemWidget);
    }

    legendScroll_->setWidget(legendContent);
    legendScroll_->viewport()->setContentsMargins(0, 0, 0, 0);
    legendOuterLayout->addWidget(legendScroll_);

    panelLayout->addWidget(legendGroup, 1);

    mainSplitter_->addWidget(controlPanel);

    // 延迟初始化图例布局，确保窗口已完成布局
    QTimer::singleShot(100, this, &MainWindow::refreshLegendLayout);

    // 设置分割比例（70% vs 30%）
    mainSplitter_->setStretchFactor(0, 7);
    mainSplitter_->setStretchFactor(1, 3);

    // 设置主布局
    QHBoxLayout* mainLayout = new QHBoxLayout(centralWidget);
    mainLayout->addWidget(mainSplitter_);
    mainLayout->setContentsMargins(0, 0, 0, 0);

    // 创建菜单栏
    QMenu* fileMenu = menuBar()->addMenu("文件");
    fileMenu->addAction("打开STEP文件", this, &MainWindow::onLoadFile);
    fileMenu->addSeparator();
    fileMenu->addAction("退出", qApp, &QApplication::quit);

    // 创建状态栏
    statusBar()->showMessage("就绪");
}

void MainWindow::setupConnections() {
    connect(btnLoadFile_, &QPushButton::clicked, this, &MainWindow::onLoadFile);
    connect(btnRunPrediction_, &QPushButton::clicked, this, &MainWindow::onRunPrediction);
    connect(btnExportResults_, &QPushButton::clicked, this, &MainWindow::onExportResults);
    connect(btnLoadLabels_, &QPushButton::clicked, this, &MainWindow::onLoadLabels);
    connect(viewer_, &OCCTViewer::faceSelected, this, &MainWindow::onFaceSelected);
}

void MainWindow::onLoadFile() {
    // 基于exe所在目录构建STEP文件默认路径
    QString exeDir = QCoreApplication::applicationDirPath();
    QString defaultPath = exeDir + "/../../../inference_data/step_files";
    
    QString fileName = QFileDialog::getOpenFileName(
        this,
        "加载STEP文件",
        defaultPath,
        "STEP Files (*.step *.stp);;All Files (*)");

    if (fileName.isEmpty()) {
        return;
    }

    // 加载文件
    statusBar()->showMessage("正在加载文件...");
    QApplication::setOverrideCursor(Qt::WaitCursor);

    if (!loader_->loadFile(fileName.toStdString())) {
        QApplication::restoreOverrideCursor();
        QMessageBox::critical(this, "错误", "无法加载STEP文件");
        statusBar()->showMessage("加载失败");
        return;
    }

    // 显示面
    viewer_->displayFaces(loader_->getFaces());

    QApplication::restoreOverrideCursor();

    // 更新状态
    currentFilePath_ = fileName;
    predictions_.clear();
    groundTruthLabels_.clear();
    errorFaceIndices_.clear();
    lblAccuracy_->setText("准确率: --");
    txtStatistics_->setPlainText("等待预测...");
    lblSelectedFace_->setText("选中面: --");
    updateModelInfo();

    // 调试输出：显示模型加载状态
    std::cout << "[调试] onLoadFile: modelLoaded_ = " << modelLoaded_ << std::endl;
    btnRunPrediction_->setEnabled(modelLoaded_);
    btnLoadLabels_->setEnabled(true);
    btnExportResults_->setEnabled(false);

    statusBar()->showMessage("加载成功: " + QString::fromStdString(loader_->getFileName()));
}

void MainWindow::onRunPrediction() {
    if (!modelLoaded_) {
        QMessageBox::warning(this, "警告", "模型未加载");
        return;
    }

    if (currentFilePath_.isEmpty()) {
        QMessageBox::warning(this, "警告", "请先加载STEP文件");
        return;
    }

    // 显示进度对话框
    QProgressDialog progress("正在运行预测...", "取消", 0, 0, this);
    progress.setWindowModality(Qt::WindowModal);
    progress.show();
    QApplication::processEvents();

    statusBar()->showMessage("正在预测...");

    // 运行预测
    predictions_ = classifier_->predict(currentFilePath_.toStdString());

    progress.close();

    if (predictions_.empty()) {
        QMessageBox::critical(this, "错误", "预测失败");
        statusBar()->showMessage("预测失败");
        return;
    }

    // 验证结果数量
    if (predictions_.size() != static_cast<size_t>(loader_->getNumFaces())) {
        QMessageBox::critical(this, "错误",
            QString("预测结果数量(%1)与面数量(%2)不匹配")
            .arg(predictions_.size())
            .arg(loader_->getNumFaces()));
        statusBar()->showMessage("预测错误");
        return;
    }

    // 更新颜色
    std::vector<Quantity_Color> colors;
    for (int classId : predictions_) {
        colors.push_back(colorMapper_->getColor(classId));
    }
    viewer_->updateAllFaceColors(colors);

    // 更新统计信息
    updatePredictionResults();

    // 更新状态显示
    updateModelInfo();

    // 如果已有真实标签，自动进行对比
    if (!groundTruthLabels_.empty()) {
        updateComparisonResults();
    }

    btnExportResults_->setEnabled(true);
    statusBar()->showMessage("预测完成");

    QMessageBox::information(this, "成功", "预测完成！模型已按类别着色。");
}

void MainWindow::onExportResults() {
    if (predictions_.empty()) {
        QMessageBox::warning(this, "警告", "没有可导出的结果");
        return;
    }

    QString fileName = QFileDialog::getSaveFileName(
        this,
        "导出预测结果",
        "prediction_results.txt",
        "Text Files (*.txt);;All Files (*)");

    if (fileName.isEmpty()) {
        return;
    }

    QFile file(fileName);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QMessageBox::critical(this, "错误", "无法创建文件");
        return;
    }

    QTextStream out(&file);
    out.setCodec("UTF-8");

    // 写入文件头
    out << "BRepNet 预测结果\n";
    out << "================\n\n";
    out << "文件: " << currentFilePath_ << "\n";
    out << "面数: " << loader_->getNumFaces() << "\n\n";

    // 写入每个面的预测结果
    out << "面索引\t类别ID\t类别名称\n";
    out << "------\t------\t--------\n";
    for (size_t i = 0; i < predictions_.size(); ++i) {
        int classId = predictions_[i];
        out << i << "\t" << classId << "\t"
            << QString::fromStdString(colorMapper_->getClassName(classId)) << "\n";
    }

    // 写入统计信息
    out << "\n\n类别分布统计\n";
    out << "============\n\n";

    std::map<int, int> distribution;
    for (int classId : predictions_) {
        distribution[classId]++;
    }

    for (const auto& pair : distribution) {
        out << QString::fromStdString(colorMapper_->getClassName(pair.first))
            << ": " << pair.second << "\n";
    }

    file.close();
    statusBar()->showMessage("结果已导出: " + fileName);
    QMessageBox::information(this, "成功", "结果已成功导出");
}

void MainWindow::onFaceSelected(int faceIndex) {
    if (faceIndex < 0 || faceIndex >= loader_->getNumFaces()) {
        return;
    }

    QString info = QString("选中面: #%1").arg(faceIndex);

    if (!predictions_.empty() && faceIndex < static_cast<int>(predictions_.size())) {
        int predClassId = predictions_[faceIndex];
        QString predClassName = QString::fromStdString(colorMapper_->getClassName(predClassId));
        info += QString(" | 预测: %1(%2)").arg(predClassName).arg(predClassId);

        // 如果有真实标签，显示对比
        if (!groundTruthLabels_.empty() && faceIndex < static_cast<int>(groundTruthLabels_.size())) {
            int trueClassId = groundTruthLabels_[faceIndex];
            QString trueClassName = QString::fromStdString(colorMapper_->getClassName(trueClassId));
            info += QString(" | 真实: %1(%2)").arg(trueClassName).arg(trueClassId);

            if (predClassId == trueClassId) {
                info += QString::fromUtf8(" ✓");
            } else {
                info += QString::fromUtf8(" ✗");
            }
        }
    }

    lblSelectedFace_->setText(info);
}

void MainWindow::updateModelInfo() {
    lblFileName_->setText("文件: " + QString::fromStdString(loader_->getFileName()));
    lblNumFaces_->setText(QString("面数: %1").arg(loader_->getNumFaces()));

    if (!predictions_.empty()) {
        lblStatus_->setText("状态: 已预测");
    } else {
        lblStatus_->setText("状态: 已加载，未预测");
    }
}

void MainWindow::updatePredictionResults() {
    if (predictions_.empty()) {
        txtStatistics_->setPlainText("等待预测...");
        return;
    }

    // 统计每个类别的数量
    std::map<int, int> distribution;
    for (int classId : predictions_) {
        distribution[classId]++;
    }

    // 生成统计报告
    QString report = "类别分布统计\n";
    report += "============\n\n";

    for (const auto& pair : distribution) {
        QString className = QString::fromStdString(colorMapper_->getClassName(pair.first));
        report += QString("%1: %2\n").arg(className, -25).arg(pair.second);
    }

    report += QString("\n总计: %1 个面\n").arg(predictions_.size());

    txtStatistics_->setPlainText(report);
}

void MainWindow::onLoadLabels() {
    QString fileName = QFileDialog::getOpenFileName(
        this,
        "导入真实标签文件",
        "inference_data/step_files",
        "Label Files (*.labels *.txt);;All Files (*)");

    if (fileName.isEmpty()) {
        return;
    }

    groundTruthLabels_ = loadLabelsFromFile(fileName);

    if (groundTruthLabels_.empty()) {
        QMessageBox::critical(this, "错误", "无法解析标签文件");
        return;
    }

    // 验证标签数量
    if (static_cast<int>(groundTruthLabels_.size()) != loader_->getNumFaces()) {
        QMessageBox::warning(this, "警告",
            QString("标签数量(%1)与面数量(%2)不匹配")
            .arg(groundTruthLabels_.size())
            .arg(loader_->getNumFaces()));
        groundTruthLabels_.clear();
        return;
    }

    statusBar()->showMessage("标签已加载: " + fileName);

    // 如果已有预测结果，自动进行对比
    if (!predictions_.empty()) {
        updateComparisonResults();
    } else {
        lblAccuracy_->setText(QString("准确率: -- (已加载%1个标签)").arg(groundTruthLabels_.size()));
    }
}

std::vector<int> MainWindow::loadLabelsFromFile(const QString& filePath) {
    std::vector<int> labels;
    QFile file(filePath);

    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        return labels;
    }

    QTextStream in(&file);
    while (!in.atEnd()) {
        QString line = in.readLine().trimmed();

        // 跳过空行和注释
        if (line.isEmpty() || line.startsWith('#')) {
            continue;
        }

        bool ok;
        int classId = line.toInt(&ok);
        if (ok && classId >= 0 && classId < 27) {
            labels.push_back(classId);
        } else {
            // 遇到无效数据，返回空
            labels.clear();
            break;
        }
    }

    file.close();
    return labels;
}

void MainWindow::updateComparisonResults() {
    if (predictions_.empty() || groundTruthLabels_.empty()) {
        return;
    }

    if (predictions_.size() != groundTruthLabels_.size()) {
        lblAccuracy_->setText("准确率: 数量不匹配");
        return;
    }

    // 计算正确率和错误面列表
    int correct = 0;
    errorFaceIndices_.clear();

    for (size_t i = 0; i < predictions_.size(); ++i) {
        if (predictions_[i] == groundTruthLabels_[i]) {
            correct++;
        } else {
            errorFaceIndices_.push_back(static_cast<int>(i));
        }
    }

    int total = static_cast<int>(predictions_.size());
    double accuracy = 100.0 * correct / total;

    lblAccuracy_->setText(QString("准确率: %1% (%2/%3 正确)")
        .arg(accuracy, 0, 'f', 1)
        .arg(correct)
        .arg(total));

    // 高亮显示错误面
    viewer_->clearErrorHighlights();
    if (!errorFaceIndices_.empty()) {
        viewer_->highlightErrorFaces(errorFaceIndices_, true);
    }

    // 更新统计信息，添加错误面列表
    QString currentText = txtStatistics_->toPlainText();
    currentText += "\n\n============\n对比验证结果\n============\n";
    currentText += QString("准确率: %1%\n").arg(accuracy, 0, 'f', 1);
    currentText += QString("正确: %1 | 错误: %2\n").arg(correct).arg(errorFaceIndices_.size());

    if (!errorFaceIndices_.empty()) {
        currentText += "\n错误面列表:\n";
        for (int idx : errorFaceIndices_) {
            int pred = predictions_[idx];
            int truth = groundTruthLabels_[idx];
            QString predName = QString::fromStdString(colorMapper_->getClassName(pred));
            QString truthName = QString::fromStdString(colorMapper_->getClassName(truth));
            currentText += QString("#%1: 预测=%2(%3), 真实=%4(%5)\n")
                .arg(idx).arg(predName).arg(pred).arg(truthName).arg(truth);
        }
    }

    txtStatistics_->setPlainText(currentText);

    statusBar()->showMessage(QString("对比完成: 准确率 %1%").arg(accuracy, 0, 'f', 1));
}
