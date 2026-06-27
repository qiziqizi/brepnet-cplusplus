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
#include <QInputDialog>
#include <QDialog>
#include <QDialogButtonBox>
#include <QFileInfo>
#include <QStringList>
#include <Quantity_Color.hxx>
#include <gp_Pnt.hxx>
#include <TopExp_Explorer.hxx>
#include <TopoDS_Edge.hxx>
#include <TopoDS.hxx>
#include <BRepAdaptor_Curve.hxx>
#include <BRep_Tool.hxx>
#include <QPixmap>
#include <QPainter>
#include <algorithm>
#include <map>

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
    , currentMode_(WorkMode::None)
    , modelLoaded_(false)
    , prevHoveredGlobalEdgeId_(-1) {

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
            std::cout << "[MainWindow] 模型加载成功" << std::endl;
        } else {
            std::cerr << "[MainWindow] 模型加载失败" << std::endl;
        }
    } else {
        std::cerr << "[MainWindow] 未找到模型权重文件" << std::endl;
    }

    // 初始化模式
    setWorkMode(WorkMode::None);
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
    int numClasses = colorMapper_->getNumClasses();
    int rows = (numClasses + cols - 1) / cols;  // 向上取整

    // 添加图例到 grid
    for (int i = 0; i < numClasses; ++i) {
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

    // === 顶部2×2网格：文件操作+预测操作 / 重置+人工标注 ===
    QGridLayout* topGrid = new QGridLayout();
    topGrid->setSpacing(4);
    topGrid->setColumnStretch(0, 1);
    topGrid->setColumnStretch(1, 3);

    // 第0行：文件操作 | 预测操作
    // 文件操作
    QGroupBox* fileGroup = new QGroupBox("文件操作", controlPanel);
    QHBoxLayout* fileLayout = new QHBoxLayout(fileGroup);
    btnLoadFile_ = new QPushButton("加载STEP文件", fileGroup);
    fileLayout->addWidget(btnLoadFile_);
    topGrid->addWidget(fileGroup, 0, 0);

    // 预测操作
    QGroupBox* predGroup = new QGroupBox("预测操作", controlPanel);
    QHBoxLayout* predLayout = new QHBoxLayout(predGroup);
    btnRunPrediction_ = new QPushButton("运行预测", predGroup);
    btnRunPrediction_->setEnabled(false);
    predLayout->addWidget(btnRunPrediction_);
    btnLoadPredictionLabels_ = new QPushButton("导入真实标签(对比)", predGroup);
    btnLoadPredictionLabels_->setEnabled(false);
    predLayout->addWidget(btnLoadPredictionLabels_);
    btnExportPrediction_ = new QPushButton("导出结果", predGroup);
    btnExportPrediction_->setEnabled(false);
    predLayout->addWidget(btnExportPrediction_);
    lblPredictionAccuracy_ = new QLabel("准确率: --", predGroup);
    predLayout->addWidget(lblPredictionAccuracy_);
    topGrid->addWidget(predGroup, 0, 1);

    // 第1行：重置 | 人工标注
    // 重置
    QGroupBox* resetGroup = new QGroupBox("重置", controlPanel);
    QHBoxLayout* resetLayout = new QHBoxLayout(resetGroup);
    btnReset_ = new QPushButton("清除所有操作", resetGroup);
    btnReset_->setEnabled(false);
    resetLayout->addWidget(btnReset_);
    topGrid->addWidget(resetGroup, 1, 0);

    // 人工标注
    QGroupBox* manualGroup = new QGroupBox("人工标注", controlPanel);
    QHBoxLayout* manualLayout = new QHBoxLayout(manualGroup);
    btnLoadManualLabels_ = new QPushButton("导入标签并上色", manualGroup);
    btnLoadManualLabels_->setEnabled(false);
    manualLayout->addWidget(btnLoadManualLabels_);
    btnModifyFaceClass_ = new QPushButton("修改当前面类别", manualGroup);
    btnModifyFaceClass_->setEnabled(false);
    manualLayout->addWidget(btnModifyFaceClass_);
    btnExportManualLabels_ = new QPushButton("导出标注结果", manualGroup);
    btnExportManualLabels_->setEnabled(false);
    manualLayout->addWidget(btnExportManualLabels_);
    topGrid->addWidget(manualGroup, 1, 1);

    panelLayout->addLayout(topGrid);

    // === 区域5: 模型信息 ===
    QGroupBox* infoGroup = new QGroupBox("模型信息", controlPanel);
    QVBoxLayout* infoLayout = new QVBoxLayout(infoGroup);

    lblFileName_ = new QLabel("文件: 未加载", infoGroup);
    lblNumFaces_ = new QLabel("面数: 0", infoGroup);
    lblCurrentMode_ = new QLabel("当前模式: 未操作", infoGroup);
    lblSelectedFace_ = new QLabel("选中面: 无", infoGroup);

    infoLayout->addWidget(lblFileName_);
    infoLayout->addWidget(lblNumFaces_);
    infoLayout->addWidget(lblCurrentMode_);
    infoLayout->addWidget(lblSelectedFace_);

    panelLayout->addWidget(infoGroup);

    // === 区域5b: 悬停信息（面 + 边左右并排） ===
    hoverGroup_ = new QGroupBox("悬停信息", controlPanel);
    QHBoxLayout* hoverRowLayout = new QHBoxLayout(hoverGroup_);

    // 左列：面信息
    QVBoxLayout* hoverLeftCol = new QVBoxLayout();
    lblHoverFaceIndex_ = new QLabel("面索引: --", hoverGroup_);
    lblHoverEdgeCount_ = new QLabel("Edge 数: --", hoverGroup_);
    lblHoverFaceEdgeIds_ = new QLabel("Edge ID: --", hoverGroup_);
    lblHoverFaceEdgeIds_->setWordWrap(true);
    hoverLeftCol->addWidget(lblHoverFaceIndex_);
    hoverLeftCol->addWidget(lblHoverEdgeCount_);
    hoverLeftCol->addWidget(lblHoverFaceEdgeIds_);

    // 右列：边信息
    QVBoxLayout* hoverRightCol = new QVBoxLayout();
    lblHoverEdgeId_ = new QLabel("Edge ID: --", hoverGroup_);
    lblHoverEdgeType_ = new QLabel("类型: --", hoverGroup_);
    hoverRightCol->addWidget(lblHoverEdgeId_);
    hoverRightCol->addWidget(lblHoverEdgeType_);

    hoverRowLayout->addLayout(hoverLeftCol, 1);

    // 分隔线
    QFrame* separator = new QFrame(hoverGroup_);
    separator->setFrameShape(QFrame::VLine);
    separator->setFrameShadow(QFrame::Sunken);
    hoverRowLayout->addWidget(separator);

    hoverRowLayout->addLayout(hoverRightCol, 1);
    panelLayout->addWidget(hoverGroup_);

    // === 区域6: 结果统计 ===
    QGroupBox* resultGroup = new QGroupBox("结果统计", controlPanel);
    QVBoxLayout* resultLayout = new QVBoxLayout(resultGroup);

    txtStatistics_ = new QTextEdit(resultGroup);
    txtStatistics_->setReadOnly(true);
    txtStatistics_->setPlainText("等待操作...");
    resultLayout->addWidget(txtStatistics_);

    panelLayout->addWidget(resultGroup);

    // === 区域7: 颜色图例 ===
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
    for (int i = 0; i < colorMapper_->getNumClasses(); ++i) {
        // 创建图例项容器
        QWidget* itemWidget = new QWidget();
        QHBoxLayout* itemLayout = new QHBoxLayout(itemWidget);
        itemLayout->setContentsMargins(0, 0, 0, 0);
        itemLayout->setSpacing(4);

        // 正方形
        QLabel* colorSwatch = new QLabel();
        colorSwatch->setFixedSize(20, 20);

        // other 显示红→绿（0°~120°）区间条纹，表示该面为彩色系中的某一种
        if (i == 3) {
            QPixmap pixmap(20, 20);
            pixmap.fill(Qt::transparent);
            {
                QPainter painter(&pixmap);
                int numStripes = 7;
                for (int s = 0; s < numStripes; ++s) {
                    double hue = 0.0 + s * (120.0 / 6.0);  // 0, 20, 40, 60, 80, 100, 120
                    Quantity_Color stripColor(hue, 0.80, 0.65, Quantity_TOC_HLS);
                    QColor qc(
                        static_cast<int>(stripColor.Red() * 255),
                        static_cast<int>(stripColor.Green() * 255),
                        static_cast<int>(stripColor.Blue() * 255));
                    int sw = (s < numStripes - 1) ? 20 / numStripes
                                                  : 20 - s * (20 / numStripes);
                    painter.fillRect(s * (20 / numStripes), 0, sw, 20, qc);
                }
                painter.setPen(QColor(0x88, 0x88, 0x88));
                painter.drawRect(0, 0, 19, 19);
            }
            colorSwatch->setPixmap(pixmap);
        } else {
            Quantity_Color qc = colorMapper_->getColor(i);
            int r = static_cast<int>(qc.Red() * 255);
            int g = static_cast<int>(qc.Green() * 255);
            int b = static_cast<int>(qc.Blue() * 255);
            colorSwatch->setStyleSheet(
                QString("background-color: rgb(%1,%2,%3); border: 1px solid #888;")
                    .arg(r).arg(g).arg(b));
        }

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

    // 左右1:1等分（窗口resize后仍保持比例）
    mainSplitter_->setStretchFactor(0, 1);
    mainSplitter_->setStretchFactor(1, 1);
    QTimer::singleShot(0, [this]() {
        int w = mainSplitter_->width();
        if (w > 0) mainSplitter_->setSizes({w / 2, w / 2});
    });

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
    // 文件操作
    connect(btnLoadFile_, &QPushButton::clicked, this, &MainWindow::onLoadFile);

    // 预测操作
    connect(btnRunPrediction_, &QPushButton::clicked, this, &MainWindow::onRunPrediction);
    connect(btnLoadPredictionLabels_, &QPushButton::clicked, this, &MainWindow::onLoadPredictionLabels);
    connect(btnExportPrediction_, &QPushButton::clicked, this, &MainWindow::onExportPrediction);

    // 人工标注
    connect(btnLoadManualLabels_, &QPushButton::clicked, this, &MainWindow::onLoadManualLabels);
    connect(btnModifyFaceClass_, &QPushButton::clicked, this, &MainWindow::onModifyFaceClass);
    connect(btnExportManualLabels_, &QPushButton::clicked, this, &MainWindow::onExportManualLabels);

    // 重置
    connect(btnReset_, &QPushButton::clicked, this, &MainWindow::onReset);

    // 面选择
    connect(viewer_, &OCCTViewer::faceSelected, this, &MainWindow::onFaceSelected);
    connect(viewer_, &OCCTViewer::faceHovered, this, &MainWindow::onFaceHovered);
}

void MainWindow::setWorkMode(WorkMode mode) {
    currentMode_ = mode;

    // 根据模式启用/禁用按钮
    bool isPredMode = (mode == WorkMode::Prediction);
    bool isManualMode = (mode == WorkMode::ManualLabeling);
    bool isNone = (mode == WorkMode::None);
    bool hasFile = !currentFilePath_.isEmpty();

    // 预测操作区
    btnRunPrediction_->setEnabled(!isPredMode && modelLoaded_ && hasFile);
    btnLoadPredictionLabels_->setEnabled(isPredMode);
    btnExportPrediction_->setEnabled(isPredMode);

    // 人工标注区
    btnLoadManualLabels_->setEnabled((isNone || isManualMode) && hasFile);
    btnModifyFaceClass_->setEnabled(isManualMode);
    btnExportManualLabels_->setEnabled(isManualMode);

    // 重置按钮
    btnReset_->setEnabled(!isNone && hasFile);

    // 更新状态显示
    QString modeText;
    switch (mode) {
        case WorkMode::None:
            modeText = "未操作";
            break;
        case WorkMode::Prediction:
            modeText = "预测模式";
            break;
        case WorkMode::ManualLabeling:
            modeText = "标注模式";
            break;
    }
    lblCurrentMode_->setText("当前模式: " + modeText);
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

    // 显示面（灰色）
    viewer_->displayFaces(loader_->getFaces());

    QApplication::restoreOverrideCursor();

    // 更新状态
    currentFilePath_ = fileName;
    predictions_.clear();
    groundTruthLabels_.clear();
    errorFaceIndices_.clear();
    lblPredictionAccuracy_->setText("准确率: --");
    lblSelectedFace_->setText("选中面: --");
    lblHoverFaceIndex_->setText("面索引: --");
    lblHoverEdgeCount_->setText("Edge 数: --");
    lblHoverFaceEdgeIds_->setText("Edge ID: --");
    lblHoverEdgeId_->setText("Edge ID: --");
    lblHoverEdgeType_->setText("类型: --");
    prevHoveredGlobalEdgeId_ = -1;
    updateModelInfo();

    // 构建全局 Edge ID 映射（TShape hash → 全局 ID）
    edgeGlobalIdMap_.clear();
    {
        int globalId = 0;
        for (int fi = 0; fi < loader_->getNumFaces(); ++fi) {
            const TopoDS_Face& f = loader_->getFaces()[fi];
            for (TopExp_Explorer exp(f, TopAbs_EDGE); exp.More(); exp.Next()) {
                const TopoDS_Edge& e = TopoDS::Edge(exp.Current());
                const TopoDS_TShape* key = e.TShape().operator->();
                if (edgeGlobalIdMap_.find(key) == edgeGlobalIdMap_.end()) {
                    edgeGlobalIdMap_[key] = globalId++;
                }
            }
        }
    }

    // 给每个面分配暖色（全部视为 other），便于区分相邻面
    int numFaces = loader_->getNumFaces();
    manualLabels_.assign(numFaces, 3);
    warmOtherColors_ = ColorMapper::generateOtherColors(numFaces);
    viewer_->updateAllFaceColors(warmOtherColors_);

    setWorkMode(WorkMode::ManualLabeling);
    updateManualLabelingResults();

    statusBar()->showMessage("加载成功: " + QString::fromStdString(loader_->getFileName())
        + QString("（%1 个面以区分色显示）").arg(numFaces));
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

    // 切换到预测模式
    setWorkMode(WorkMode::Prediction);

    // 更新统计信息
    updatePredictionResults();

    statusBar()->showMessage("预测完成");
    QMessageBox::information(this, "成功", "预测完成！模型已按类别着色。");
}

void MainWindow::onLoadPredictionLabels() {
    QString fileName = QFileDialog::getOpenFileName(
        this,
        "导入真实标签文件(对比)",
        "inference_data/step_files",
        "Segmentation Files (*.seg);;All Files (*)");

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

    // 自动进行对比
    updateComparisonResults();
}

void MainWindow::onExportPrediction() {
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

    // 根据当前模式显示不同信息
    if (currentMode_ == WorkMode::Prediction && !predictions_.empty()) {
        int predClassId = predictions_[faceIndex];
        QString predClassName = QString::fromStdString(colorMapper_->getClassName(predClassId));
        info += QString(" | 预测: %1(%2)").arg(predClassName).arg(predClassId);

        // 如果有对比标签
        if (!groundTruthLabels_.empty()) {
            int trueClassId = groundTruthLabels_[faceIndex];
            QString trueClassName = QString::fromStdString(colorMapper_->getClassName(trueClassId));
            info += QString(" | 真实: %1(%2)").arg(trueClassName).arg(trueClassId);
            info += (predClassId == trueClassId) ? QString::fromUtf8(" ✓") : QString::fromUtf8(" ✗");
        }
    }
    else if (currentMode_ == WorkMode::ManualLabeling && !manualLabels_.empty()) {
        int labelClassId = manualLabels_[faceIndex];
        QString labelClassName = QString::fromStdString(colorMapper_->getClassName(labelClassId));
        info += QString(" | 标注: %1(%2)").arg(labelClassName).arg(labelClassId);
    }

    lblSelectedFace_->setText(info);
}

void MainWindow::onFaceHovered(int faceIndex, int mouseX, int mouseY) {
    if (faceIndex < 0 || !loader_ || faceIndex >= loader_->getNumFaces()) {
        lblHoverFaceIndex_->setText("面索引: --");
        lblHoverEdgeCount_->setText("Edge 数: --");
        lblHoverFaceEdgeIds_->setText("Edge ID: --");
        lblHoverEdgeId_->setText("Edge ID: --");
        lblHoverEdgeType_->setText("类型: --");
        prevHoveredGlobalEdgeId_ = -1;
        return;
    }

    // ========== 1. 悬停面：面索引 + Edge 总数 + Edge ID 列表 ==========
    const TopoDS_Face& face = loader_->getFaces()[faceIndex];
    lblHoverFaceIndex_->setText(QString("面索引: #%1").arg(faceIndex));

    QStringList faceEdgeIdStr;
    int edgeCount = 0;
    for (TopExp_Explorer exp(face, TopAbs_EDGE); exp.More(); exp.Next()) {
        const TopoDS_Edge& edge = TopoDS::Edge(exp.Current());
        if (BRep_Tool::Degenerated(edge)) continue;
        int gid = -1;
        auto it = edgeGlobalIdMap_.find(edge.TShape().operator->());
        if (it != edgeGlobalIdMap_.end()) gid = it->second;
        faceEdgeIdStr << QString::number(gid);
        edgeCount++;
    }
    lblHoverEdgeCount_->setText(QString("Edge 数: %1").arg(edgeCount));
    lblHoverFaceEdgeIds_->setText("Edge ID: " + (faceEdgeIdStr.isEmpty()
        ? QString("--") : faceEdgeIdStr.join(", ")));

    // ========== 2. 屏幕投影找出当前鼠标下最近 Edge ==========
    const Handle(V3d_View)& view = viewer_->getView();
    if (view.IsNull()) return;

    // 检测阈值 + 滞回阈值（离开展示的边时需要更远距离）
    const double enterThreshold = 12.0;
    const double leaveThreshold = 22.0;
    double useThreshold = (prevHoveredGlobalEdgeId_ >= 0) ? leaveThreshold : enterThreshold;

    int bestGlobalEdgeId = -1;
    double bestDist = useThreshold;
    BRepAdaptor_Curve bestCurve;

    for (TopExp_Explorer exp(face, TopAbs_EDGE); exp.More(); exp.Next()) {
        const TopoDS_Edge& edge = TopoDS::Edge(exp.Current());
        if (BRep_Tool::Degenerated(edge)) continue;

        int gid = -1;
        auto it = edgeGlobalIdMap_.find(edge.TShape().operator->());
        if (it != edgeGlobalIdMap_.end()) gid = it->second;

        BRepAdaptor_Curve curveAdaptor(edge);
        Standard_Real tFirst = curveAdaptor.FirstParameter();
        Standard_Real tLast  = curveAdaptor.LastParameter();
        const int numSamples = 9;
        for (int s = 0; s < numSamples; ++s) {
            Standard_Real t = tFirst + (tLast - tFirst) * s / (numSamples - 1);
            gp_Pnt pt3d = curveAdaptor.Value(t);
            Standard_Integer sx, sy;
            view->Convert(pt3d.X(), pt3d.Y(), pt3d.Z(), sx, sy);
            double dx = sx - mouseX;
            double dy = sy - mouseY;
            double dist = sqrt(dx * dx + dy * dy);
            if (dist < bestDist) {
                bestDist = dist;
                bestGlobalEdgeId = gid;
                bestCurve = curveAdaptor;
            }
        }
    }

    // ========== 3. 悬停边：显示全局 Edge ID + 类型 ==========
    if (bestGlobalEdgeId >= 0) {
        QString edgeTypeStr;
        switch (bestCurve.GetType()) {
            case GeomAbs_Line:       edgeTypeStr = "直线";  break;
            case GeomAbs_Circle:     edgeTypeStr = "圆弧";  break;
            case GeomAbs_Ellipse:    edgeTypeStr = "椭圆";  break;
            case GeomAbs_BSplineCurve: edgeTypeStr = "B样条"; break;
            default:                 edgeTypeStr = "其他";  break;
        }
        lblHoverEdgeId_->setText(QString("Edge ID: #%1").arg(bestGlobalEdgeId));
        lblHoverEdgeType_->setText(QString("类型: %1").arg(edgeTypeStr));
        prevHoveredGlobalEdgeId_ = bestGlobalEdgeId;
    } else {
        lblHoverEdgeId_->setText("Edge ID: --");
        lblHoverEdgeType_->setText("类型: --");
        prevHoveredGlobalEdgeId_ = -1;
    }
}

void MainWindow::updateModelInfo() {
    lblFileName_->setText("文件: " + QString::fromStdString(loader_->getFileName()));
    lblNumFaces_->setText(QString("面数: %1").arg(loader_->getNumFaces()));
}

void MainWindow::updateManualLabelingResults() {
    if (manualLabels_.empty()) {
        txtStatistics_->setPlainText("等待标注...");
        return;
    }

    // 统计每个类别的数量
    std::map<int, int> distribution;
    for (int classId : manualLabels_) {
        distribution[classId]++;
    }

    // 生成统计报告
    QString report = "标注类别分布\n";
    report += "============\n\n";

    for (const auto& pair : distribution) {
        QString className = QString::fromStdString(colorMapper_->getClassName(pair.first));
        report += QString("%1: %2\n").arg(className, -25).arg(pair.second);
    }

    report += QString("\n总计: %1 个面\n").arg(manualLabels_.size());

    txtStatistics_->setPlainText(report);
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
    QString report = "预测类别分布\n";
    report += "============\n\n";

    for (const auto& pair : distribution) {
        QString className = QString::fromStdString(colorMapper_->getClassName(pair.first));
        report += QString("%1: %2\n").arg(className, -25).arg(pair.second);
    }

    report += QString("\n总计: %1 个面\n").arg(predictions_.size());

    txtStatistics_->setPlainText(report);
}

void MainWindow::onLoadManualLabels() {
    QString fileName = QFileDialog::getOpenFileName(
        this,
        "导入标签文件",
        "inference_data/step_files",
        "Segmentation Files (*.seg);;All Files (*)");

    if (fileName.isEmpty()) {
        return;
    }

    manualLabels_ = loadLabelsFromFile(fileName);

    if (manualLabels_.empty()) {
        QMessageBox::critical(this, "错误", "无法解析标签文件");
        return;
    }

    // 验证标签数量
    if (static_cast<int>(manualLabels_.size()) != loader_->getNumFaces()) {
        QMessageBox::warning(this, "警告",
            QString("标签数量(%1)与面数量(%2)不匹配")
            .arg(manualLabels_.size())
            .arg(loader_->getNumFaces()));
        manualLabels_.clear();
        return;
    }

    // 旧标签中的 4(unlabeled) → 3(other)
    for (int& classId : manualLabels_) {
        if (classId == 4) classId = 3;
    }

    // 生成暖色系颜色，给 other(3) 的面使用
    int numFaces = loader_->getNumFaces();
    warmOtherColors_ = ColorMapper::generateOtherColors(numFaces);

    // 上色：other 用暖色系该面的颜色，其余用固定类颜色
    std::vector<Quantity_Color> colors;
    for (int i = 0; i < numFaces; ++i) {
        int classId = manualLabels_[i];
        if (classId == 3) {
            colors.push_back(warmOtherColors_[i]);
        } else {
            colors.push_back(colorMapper_->getColor(classId));
        }
    }
    viewer_->updateAllFaceColors(colors);

    // 切换到标注模式
    setWorkMode(WorkMode::ManualLabeling);

    // 更新统计
    updateManualLabelingResults();

    statusBar()->showMessage("标签已加载并上色: " + fileName);
    QMessageBox::information(this, "成功", "标签已加载并按类别着色。");
}

void MainWindow::onModifyFaceClass() {
    // 检查是否有选中的面
    int selectedFace = viewer_->getSelectedFaceIndex();
    if (selectedFace < 0) {
        QMessageBox::warning(this, "警告", "请先点击选中一个面");
        return;
    }

    int currentClass = manualLabels_[selectedFace];
    QString currentClassName = QString::fromStdString(colorMapper_->getClassName(currentClass));

    // 自定义对话框：5 个彩色按钮
    QDialog dlg(this);
    dlg.setWindowTitle("修改面类别");
    QVBoxLayout* dlgLayout = new QVBoxLayout(&dlg);

    QLabel* infoLbl = new QLabel(
        QString("当前面 #%1\n当前类别: %2 (%3)\n\n请选择新类别:")
            .arg(selectedFace).arg(currentClass).arg(currentClassName), &dlg);
    dlgLayout->addWidget(infoLbl);

    int newClass = -1;
    int numClasses = colorMapper_->getNumClasses();
    for (int i = 0; i < numClasses; ++i) {
        Quantity_Color qc = colorMapper_->getColor(i);
        int r = static_cast<int>(qc.Red() * 255);
        int g = static_cast<int>(qc.Green() * 255);
        int b = static_cast<int>(qc.Blue() * 255);
        // 根据背景亮度选择文字颜色（深色背景白字、浅色背景黑字）
        double luma = 0.299 * qc.Red() + 0.587 * qc.Green() + 0.114 * qc.Blue();
        QString textColor = (luma > 0.55) ? "black" : "white";

        QPushButton* btn = new QPushButton(
            QString("%1 - %2").arg(i).arg(QString::fromStdString(colorMapper_->getClassName(i))),
            &dlg);
        btn->setStyleSheet(
            QString("QPushButton { background-color: rgb(%1,%2,%3); color: %4; "
                    "font-size: 14px; font-weight: bold; padding: 10px; border: 1px solid #444; border-radius: 4px; }"
                    "QPushButton:hover { border: 2px solid #000; }")
                .arg(r).arg(g).arg(b).arg(textColor));
        btn->setMinimumHeight(40);
        connect(btn, &QPushButton::clicked, &dlg, [&dlg, &newClass, i]() {
            newClass = i;
            dlg.accept();
        });
        dlgLayout->addWidget(btn);
    }

    QPushButton* cancelBtn = new QPushButton("取消", &dlg);
    connect(cancelBtn, &QPushButton::clicked, &dlg, &QDialog::reject);
    dlgLayout->addWidget(cancelBtn);

    if (dlg.exec() != QDialog::Accepted || newClass < 0) return;

    // 更新标签
    manualLabels_[selectedFace] = newClass;

    // 更新颜色：other 用暖色系中该面的颜色，其余用固定类颜色
    Quantity_Color newColor;
    if (newClass == 3 && selectedFace >= 0 && selectedFace < (int)warmOtherColors_.size()) {
        newColor = warmOtherColors_[selectedFace];
    } else {
        newColor = colorMapper_->getColor(newClass);
    }
    viewer_->updateSingleFaceColor(selectedFace, newColor);

    // 更新选中面信息
    onFaceSelected(selectedFace);

    // 更新统计
    updateManualLabelingResults();

    QString newClassName = QString::fromStdString(colorMapper_->getClassName(newClass));
    statusBar()->showMessage(
        QString("面 #%1 类别已修改: %2(%3) → %4(%5)")
        .arg(selectedFace)
        .arg(currentClass).arg(currentClassName)
        .arg(newClass).arg(newClassName));
}

void MainWindow::onExportManualLabels() {
    if (manualLabels_.empty()) {
        QMessageBox::warning(this, "警告", "没有可导出的标注结果");
        return;
    }

    // 默认保存在 STEP 文件同目录，文件名为 STEP 文件名（.seg 扩展名）
    // 如已存在则追加序号 _1、_2 ... 避免覆盖
    QFileInfo stepInfo(currentFilePath_);
    QString baseDir = stepInfo.absolutePath();
    QString baseName = stepInfo.completeBaseName();
    QString fileName = baseDir + "/" + baseName + ".seg";
    int suffix = 1;
    while (QFileInfo::exists(fileName)) {
        fileName = QString("%1/%2_%3.seg").arg(baseDir, baseName).arg(suffix);
        ++suffix;
    }

    QFile file(fileName);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QMessageBox::critical(this, "错误", "无法创建文件");
        return;
    }

    QTextStream out(&file);
    out.setCodec("UTF-8");

    // 直接写入标签（每行一个，无文件头注释）
    for (int classId : manualLabels_) {
        out << classId << "\n";
    }

    file.close();
    statusBar()->showMessage("标注结果已导出: " + fileName);
    QMessageBox::information(this, "成功", "标注结果已导出至:\n" + fileName);
}

void MainWindow::onReset() {
    // 确认对话框
    QMessageBox::StandardButton reply = QMessageBox::question(
        this,
        "确认重置",
        "确定要清除所有操作吗？\n这将清除预测结果和标注数据，但不会关闭STEP文件。",
        QMessageBox::Yes | QMessageBox::No);

    if (reply != QMessageBox::Yes) return;

    // 清除数据
    predictions_.clear();
    groundTruthLabels_.clear();
    errorFaceIndices_.clear();

    // 回到全 other 状态（与刚加载时一致）
    int numFaces = loader_ ? loader_->getNumFaces() : 0;
    if (numFaces > 0) {
        manualLabels_.assign(numFaces, 3);
        warmOtherColors_ = ColorMapper::generateOtherColors(numFaces);
        viewer_->resetAllFaceColors();
        viewer_->updateAllFaceColors(warmOtherColors_);
    } else {
        manualLabels_.clear();
        warmOtherColors_.clear();
    }
    viewer_->clearErrorHighlights();

    // 回到标注模式
    setWorkMode(WorkMode::ManualLabeling);

    // 重置UI
    lblPredictionAccuracy_->setText("准确率: --");
    lblSelectedFace_->setText("选中面: --");
    lblHoverFaceIndex_->setText("面索引: --");
    lblHoverEdgeCount_->setText("Edge 数: --");
    lblHoverFaceEdgeIds_->setText("Edge ID: --");
    lblHoverEdgeId_->setText("Edge ID: --");
    lblHoverEdgeType_->setText("类型: --");
    prevHoveredGlobalEdgeId_ = -1;
    txtStatistics_->setPlainText("等待操作...");

    statusBar()->showMessage("已重置");
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
        if (ok && classId >= 0 && classId < 5) {
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
        lblPredictionAccuracy_->setText("准确率: 数量不匹配");
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

    lblPredictionAccuracy_->setText(QString("准确率: %1% (%2/%3 正确)")
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
