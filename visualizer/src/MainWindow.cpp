#include "MainWindow.h"
#include "VersionConfig.h"
#include <QFileDialog>
#include <QMessageBox>
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
#include <QShortcut>
#include <QKeyEvent>
#include <QAbstractItemView>
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
#include <QBrush>
#include <QFont>
#include <QColor>
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
        exeDir + "/../../../inference_data/state_dict.npz",
        exeDir + "/../../../../inference_data/state_dict.npz",
        exeDir + "/inference_data/state_dict_v4.npz",
        exeDir + "/inference_data/state_dict_v123.npz",
        exeDir + "/../../../inference_data/state_dict_v4.npz",
        exeDir + "/../../../inference_data/state_dict_v123.npz",
        exeDir + "/../../../../inference_data/state_dict_v4.npz",
        exeDir + "/../../../../inference_data/state_dict_v123.npz",
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
#if BREPNET_VERSION == 4
    setWindowTitle("BRepNet 可视化工具 - v4");
#elif BREPNET_VERSION == 123
    setWindowTitle("BRepNet 可视化工具 - v123");
#else
    setWindowTitle("BRepNet 可视化工具");
#endif
    resize(1400, 900);

    // 创建中心部件
    QWidget* centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);

    // 创建分割器（左右布局）
    mainSplitter_ = new QSplitter(Qt::Horizontal, centralWidget);

    // ========== 最左侧：文件浏览侧边栏（可拖拽，不随窗口伸缩）==========
    fileSidebar_ = new QWidget(mainSplitter_);
    fileSidebar_->setMinimumWidth(180);
    fileSidebar_->setMaximumWidth(400);
    QVBoxLayout* sidebarLayout = new QVBoxLayout(fileSidebar_);
    sidebarLayout->setContentsMargins(2, 2, 2, 2);

    btnSelectDir_ = new QPushButton("选择目录", fileSidebar_);
    sidebarLayout->addWidget(btnSelectDir_);

    fileListWidget_ = new QListWidget(fileSidebar_);
    sidebarLayout->addWidget(fileListWidget_, 1);

    // 导航按钮行
    QHBoxLayout* navLayout = new QHBoxLayout();
    btnPrevFile_ = new QPushButton("← 上一个", fileSidebar_);
    btnNextFile_ = new QPushButton("下一个 →", fileSidebar_);
    navLayout->addWidget(btnPrevFile_);
    navLayout->addWidget(btnNextFile_);
    sidebarLayout->addLayout(navLayout);

    btnLoadSelected_ = new QPushButton("加载选中文件", fileSidebar_);
    sidebarLayout->addWidget(btnLoadSelected_);

    lblFileIndex_ = new QLabel("未加载", fileSidebar_);
    lblFileIndex_->setAlignment(Qt::AlignCenter);
    lblFileIndex_->setStyleSheet("font-weight: bold; padding: 4px;");
    sidebarLayout->addWidget(lblFileIndex_);

    chkAutoLoadLabels_ = new QCheckBox("自动导入标签", fileSidebar_);
    chkAutoLoadLabels_->setToolTip("勾选后，加载STEP文件时自动导入同名的.seg标签文件");
    sidebarLayout->addWidget(chkAutoLoadLabels_);

    mainSplitter_->addWidget(fileSidebar_);

    // ========== 中间：3D视图 ==========
    viewer_ = new OCCTViewer(mainSplitter_);
    mainSplitter_->addWidget(viewer_);

    // ========== 右侧：控制面板 ==========
    QWidget* controlPanel = new QWidget(mainSplitter_);
    QVBoxLayout* panelLayout = new QVBoxLayout(controlPanel);

    // === 顶部布局 ===
    // 第一行：文件操作 | 重置 | 导出
    // 第二行：预测操作 | 人工标注
    QGridLayout* topGrid = new QGridLayout();
    topGrid->setSpacing(4);
    topGrid->setColumnStretch(0, 1);
    topGrid->setColumnStretch(1, 1);
    topGrid->setColumnStretch(2, 1);

    // 第0行：文件操作 | 重置 | 导出
    QGroupBox* fileGroup = new QGroupBox("文件操作", controlPanel);
    QHBoxLayout* fileLayout = new QHBoxLayout(fileGroup);
    btnLoadFile_ = new QPushButton("加载STEP文件", fileGroup);
    fileLayout->addWidget(btnLoadFile_);
    topGrid->addWidget(fileGroup, 0, 0);

    QGroupBox* resetGroup = new QGroupBox("重置", controlPanel);
    QHBoxLayout* resetLayout = new QHBoxLayout(resetGroup);
    btnReset_ = new QPushButton("清除所有操作", resetGroup);
    btnReset_->setEnabled(false);
    resetLayout->addWidget(btnReset_);
    topGrid->addWidget(resetGroup, 0, 1);

    QGroupBox* exportGroup = new QGroupBox("导出", controlPanel);
    QHBoxLayout* exportLayout = new QHBoxLayout(exportGroup);
    btnExportResults_ = new QPushButton("导出结果", exportGroup);
    btnExportResults_->setEnabled(false);
    exportLayout->addWidget(btnExportResults_);
    topGrid->addWidget(exportGroup, 0, 2);

    // 第1行：预测操作（跨3列）
    QGroupBox* predGroup = new QGroupBox("预测操作", controlPanel);
    QHBoxLayout* predLayout = new QHBoxLayout(predGroup);
    btnRunPrediction_ = new QPushButton("运行预测", predGroup);
    btnRunPrediction_->setEnabled(false);
    predLayout->addWidget(btnRunPrediction_);
    btnLoadPredictionLabels_ = new QPushButton("导入真实标签(对比)", predGroup);
    btnLoadPredictionLabels_->setEnabled(false);
    predLayout->addWidget(btnLoadPredictionLabels_);
    lblPredictionAccuracy_ = new QLabel("准确率: --", predGroup);
    predLayout->addWidget(lblPredictionAccuracy_);
    topGrid->addWidget(predGroup, 1, 0, 1, 3);

    // 第2行：人工标注（跨3列）
    QGroupBox* manualGroup = new QGroupBox("人工标注", controlPanel);
    QHBoxLayout* manualLayout = new QHBoxLayout(manualGroup);
    btnLoadManualLabels_ = new QPushButton("手动导入标签", manualGroup);
    btnLoadManualLabels_->setEnabled(false);
    manualLayout->addWidget(btnLoadManualLabels_);
    btnLoadAutoLabels_ = new QPushButton("自动导入标签", manualGroup);
    btnLoadAutoLabels_->setEnabled(false);
    manualLayout->addWidget(btnLoadAutoLabels_);
    btnModifyFaceClass_ = new QPushButton("修改当前面类别", manualGroup);
    btnModifyFaceClass_->setEnabled(false);
    manualLayout->addWidget(btnModifyFaceClass_);
    topGrid->addWidget(manualGroup, 2, 0, 1, 3);

    panelLayout->addLayout(topGrid);

    // === 区域5: 模型信息 ===
    QGroupBox* infoGroup = new QGroupBox("模型信息", controlPanel);
    QVBoxLayout* infoLayout = new QVBoxLayout(infoGroup);

    lblFileName_ = new QLabel("文件: 未加载", infoGroup);
    lblNumFaces_ = new QLabel("面数: 0", infoGroup);
    lblSelectedFace_ = new QLabel("当前面: 无", infoGroup);

    infoLayout->addWidget(lblFileName_);
    infoLayout->addWidget(lblNumFaces_);
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
    lblHoverCoedgeId_ = new QLabel("Coedge ID(Face): --", hoverGroup_);
    hoverRightCol->addWidget(lblHoverEdgeId_);
    hoverRightCol->addWidget(lblHoverEdgeType_);
    hoverRightCol->addWidget(lblHoverCoedgeId_);

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

    // 三栏布局：侧边栏(固定宽220) | 3D视图(伸缩) | 控制面板(伸缩)
    mainSplitter_->setStretchFactor(0, 0);  // 侧边栏不伸缩
    mainSplitter_->setStretchFactor(1, 1);  // 视图伸缩
    mainSplitter_->setStretchFactor(2, 1);  // 控制面板伸缩
    QTimer::singleShot(0, [this]() {
        int w = mainSplitter_->width();
        int sidebarW = 220;
        if (w > sidebarW) mainSplitter_->setSizes({sidebarW, (w - sidebarW) / 2, (w - sidebarW) / 2});
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

    // 文件浏览侧边栏
    connect(btnSelectDir_, &QPushButton::clicked, this, &MainWindow::onSelectDir);
    connect(btnLoadSelected_, &QPushButton::clicked, this, &MainWindow::onLoadSelected);
    connect(btnPrevFile_, &QPushButton::clicked, this, &MainWindow::onPrevFile);
    connect(btnNextFile_, &QPushButton::clicked, this, &MainWindow::onNextFile);
    connect(fileListWidget_, &QListWidget::itemDoubleClicked, this, &MainWindow::onLoadSelected);
    connect(fileListWidget_, &QListWidget::itemClicked, this, &MainWindow::onFileListItemClicked);

    // 预测操作
    connect(btnRunPrediction_, &QPushButton::clicked, this, &MainWindow::onRunPrediction);
    connect(btnLoadPredictionLabels_, &QPushButton::clicked, this, &MainWindow::onLoadPredictionLabels);

    // 人工标注
    connect(btnLoadManualLabels_, &QPushButton::clicked, this, &MainWindow::onLoadManualLabels);
    connect(btnLoadAutoLabels_, &QPushButton::clicked, this, &MainWindow::onLoadAutoLabels);
    connect(btnModifyFaceClass_, &QPushButton::clicked, this, &MainWindow::onModifyFaceClass);

    // 导出
    connect(btnExportResults_, &QPushButton::clicked, this, &MainWindow::onExportResults);

    // 重置
    connect(btnReset_, &QPushButton::clicked, this, &MainWindow::onReset);

    // 面选择
    connect(viewer_, &OCCTViewer::faceSelected, this, &MainWindow::onFaceSelected);
    connect(viewer_, &OCCTViewer::faceHovered, this, &MainWindow::onFaceHovered);
    connect(viewer_, &OCCTViewer::faceSelectionChanged, this, &MainWindow::onFaceSelectionChanged);

    connect(viewer_, &OCCTViewer::faceModifyRequested, this, &MainWindow::onModifyFaceClass);
}

void MainWindow::setWorkMode(WorkMode mode) {
    currentMode_ = mode;

    // 按钮状态：只根据"有文件"和"模型已加载"决定，不互斥
    bool hasFile = !currentFilePath_.isEmpty();

    // 预测操作区
    btnRunPrediction_->setEnabled(modelLoaded_ && hasFile);
    btnLoadPredictionLabels_->setEnabled(hasFile && !predictions_.empty());

    // 人工标注区 (随时可用)
    btnLoadManualLabels_->setEnabled(hasFile);
    btnLoadAutoLabels_->setEnabled(hasFile);
    btnModifyFaceClass_->setEnabled(hasFile);

    // 重置按钮
    btnReset_->setEnabled(hasFile);

    // 导出按钮 (有预测或标注结果时可导出)
    btnExportResults_->setEnabled(
        (!predictions_.empty() || !manualLabels_.empty()) && hasFile);
}

void MainWindow::onLoadFile() {
    // 基于exe所在目录构建STEP文件默认路径
    QString exeDir = QCoreApplication::applicationDirPath();
    QString defaultPath = exeDir + "/../../../inference_data/step_files";
    if (!currentDir_.isEmpty()) {
        defaultPath = currentDir_;
    }

    QString fileName = QFileDialog::getOpenFileName(
        this,
        "加载STEP文件",
        defaultPath,
        "STEP Files (*.step *.stp);;All Files (*)");

    if (fileName.isEmpty()) {
        return;
    }

    loadStepFile(fileName);
}

void MainWindow::loadStepFile(const QString& fileName) {
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
    viewer_->displayFaces(loader_->getFaces(), loader_->getShape());

    QApplication::restoreOverrideCursor();

    // 更新状态
    currentFilePath_ = fileName;
    predictions_.clear();
    groundTruthLabels_.clear();
    errorFaceIndices_.clear();
    lblPredictionAccuracy_->setText("准确率: --");
    lblSelectedFace_->setText("当前面: --");
    lblHoverFaceIndex_->setText("面索引: --");
    lblHoverEdgeCount_->setText("Edge 数: --");
    lblHoverFaceEdgeIds_->setText("Edge ID: --");
    lblHoverEdgeId_->setText("Edge ID: --");
    lblHoverEdgeType_->setText("类型: --");
    lblHoverCoedgeId_->setText("Coedge ID(Face): --");
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

    // 构建 coedge 映射
    faceEdgeCoedge_.clear();
    edgeToCoedges_.clear();
    faceCoedgeOffset_.clear();
    {
        int numFaces = loader_->getNumFaces();
        faceEdgeCoedge_.resize(numFaces);
        faceCoedgeOffset_.resize(numFaces, 0);
        int globalCoedgeCounter = 0;
        for (int fi = 0; fi < numFaces; ++fi) {
            faceCoedgeOffset_[fi] = globalCoedgeCounter;
            const TopoDS_Face& f = loader_->getFaces()[fi];
            int coedgeIdx = 0;
            for (TopExp_Explorer exp(f, TopAbs_EDGE); exp.More(); exp.Next()) {
                const TopoDS_Edge& e = TopoDS::Edge(exp.Current());
                // 包含退化边，保证 coedge 全局编号与 Python npz 一致
                const TopoDS_TShape* key = e.TShape().operator->();
                auto it = edgeGlobalIdMap_.find(key);
                if (it != edgeGlobalIdMap_.end()) {
                    int gid = it->second;
                    faceEdgeCoedge_[fi].push_back(gid);
                    edgeToCoedges_[gid].emplace_back(fi, coedgeIdx);
                    ++globalCoedgeCounter;
                }
                ++coedgeIdx;
            }
        }
    }

    // 给每个面分配暖色（全部视为 other），便于区分相邻面
    int numFaces = loader_->getNumFaces();
    manualLabels_.assign(numFaces, 3);
    warmOtherColors_ = ColorMapper::generateOtherColors(numFaces);
    viewer_->updateAllFaceColors(warmOtherColors_);

    setWorkMode(WorkMode::ManualLabeling);
    updateStatistics();

    // 同步侧边栏选中状态
    refreshFileListSelection();

    statusBar()->showMessage("加载成功: " + QString::fromStdString(loader_->getFileName())
        + QString("（%1 个面以区分色显示）").arg(numFaces));

    // 自动导入同名 .seg 标签文件
    if (chkAutoLoadLabels_ && chkAutoLoadLabels_->isChecked()) {
        QFileInfo stepInfo(fileName);
        QString segPath = stepInfo.absolutePath() + "/" + stepInfo.completeBaseName() + ".seg";
        if (QFileInfo::exists(segPath)) {
            loadAndApplyLabels(segPath, true);
            statusBar()->showMessage("加载成功: " + QString::fromStdString(loader_->getFileName())
                + QString("（已自动导入标签: %1）").arg(stepInfo.completeBaseName() + ".seg"));
        }
    }
}

void MainWindow::onSelectDir() {
    QString dir = QFileDialog::getExistingDirectory(
        this, "选择STEP文件所在目录",
        currentDir_.isEmpty() ? QCoreApplication::applicationDirPath() : currentDir_);

    if (dir.isEmpty()) return;

    currentDir_ = dir;

    // 扫描目录下所有 .step / .stp 文件（不区分大小写）
    QDir d(dir);
    QStringList filters;
    filters << "*.step" << "*.stp" << "*.STEP" << "*.STP";
    QFileInfoList files = d.entryInfoList(filters, QDir::Files, QDir::Name);

    stepFiles_.clear();
    fileListWidget_->clear();
    int hasSegCount = 0;
    for (const QFileInfo& fi : files) {
        QListWidgetItem* item = new QListWidgetItem(fi.fileName());
        item->setToolTip(fi.absoluteFilePath());

        // 检查同目录下是否存在同名的 .seg 文件
        QString segPath = fi.absolutePath() + "/" + fi.completeBaseName() + ".seg";
        if (QFileInfo::exists(segPath)) {
            // 绿色高亮表示已有标注文件
            QBrush greenBrush(QColor(0, 140, 0));
            item->setForeground(greenBrush);
            QFont f = item->font();
            f.setBold(true);
            item->setFont(f);
            item->setToolTip(fi.absoluteFilePath() + "\n[已有标注: " + segPath + "]");
            ++hasSegCount;
        }

        fileListWidget_->addItem(item);
        stepFiles_.append(fi.absoluteFilePath());
    }

    lblFileIndex_->setText(QString("共 %1 个文件 (%2 个已标注)")
        .arg(stepFiles_.size()).arg(hasSegCount));

    // 如果当前已加载文件在列表中，高亮选中
    refreshFileListSelection();
}

void MainWindow::onLoadSelected() {
    int row = fileListWidget_->currentRow();
    if (row < 0 || row >= stepFiles_.size()) {
        QMessageBox::warning(this, "提示", "请先在列表中选择一个文件");
        return;
    }
    loadStepFile(stepFiles_[row]);
}

void MainWindow::onFileListItemClicked(QListWidgetItem* item) {
    // 单击只选中，不加载（双击才加载）
    // 更新索引显示
    int row = fileListWidget_->currentRow();
    if (row >= 0 && row < stepFiles_.size()) {
        QFileInfo fi(stepFiles_[row]);
        lblFileIndex_->setText(QString("%1 / %2\n%3")
            .arg(row + 1).arg(stepFiles_.size()).arg(fi.fileName()));
    }
}

void MainWindow::onPrevFile() {
    if (stepFiles_.isEmpty()) {
        QMessageBox::information(this, "提示", "请先选择目录");
        return;
    }

    int row = fileListWidget_->currentRow();
    // 如果当前已加载文件在列表中，从当前位置向前；否则从头开始
    int currentRow = -1;
    if (!currentFilePath_.isEmpty()) {
        for (int i = 0; i < stepFiles_.size(); ++i) {
            if (QFileInfo(stepFiles_[i]) == QFileInfo(currentFilePath_)) {
                currentRow = i;
                break;
            }
        }
    }
    if (currentRow < 0) currentRow = row;
    if (currentRow < 0) currentRow = 0;

    int prevRow = (currentRow - 1 + stepFiles_.size()) % stepFiles_.size();
    loadStepFile(stepFiles_[prevRow]);
}

void MainWindow::onNextFile() {
    if (stepFiles_.isEmpty()) {
        QMessageBox::information(this, "提示", "请先选择目录");
        return;
    }

    int currentRow = -1;
    if (!currentFilePath_.isEmpty()) {
        for (int i = 0; i < stepFiles_.size(); ++i) {
            if (QFileInfo(stepFiles_[i]) == QFileInfo(currentFilePath_)) {
                currentRow = i;
                break;
            }
        }
    }
    if (currentRow < 0) currentRow = fileListWidget_->currentRow();
    if (currentRow < 0) currentRow = -1; // 从第一个开始

    int nextRow = (currentRow + 1) % stepFiles_.size();
    loadStepFile(stepFiles_[nextRow]);
}

void MainWindow::refreshFileListSelection() {
    if (stepFiles_.isEmpty() || currentFilePath_.isEmpty()) return;

    for (int i = 0; i < stepFiles_.size(); ++i) {
        if (QFileInfo(stepFiles_[i]) == QFileInfo(currentFilePath_)) {
            fileListWidget_->setCurrentRow(i);
            fileListWidget_->scrollToItem(fileListWidget_->item(i), QAbstractItemView::PositionAtCenter);
            lblFileIndex_->setText(QString("%1 / %2\n%3")
                .arg(i + 1).arg(stepFiles_.size())
                .arg(QFileInfo(stepFiles_[i]).fileName()));
            return;
        }
    }
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

    statusBar()->showMessage("正在预测...");

    // 弹出"正在预测"提示窗口（模态，无按钮无进度条）
    QMessageBox* waitMsg = new QMessageBox(QMessageBox::NoIcon,
        "请稍候", "正在运行预测，请等待...", QMessageBox::NoButton, this);
    waitMsg->setStandardButtons(QMessageBox::NoButton);
    waitMsg->setModal(true);
    waitMsg->show();
    QApplication::processEvents();

    // 运行预测 (返回 27 类结果)
    auto predictions_27 = classifier_->predict(currentFilePath_.toStdString());

    // 关闭提示窗口
    waitMsg->close();
    delete waitMsg;
    QApplication::processEvents();

    if (predictions_27.empty()) {
        QMessageBox::critical(this, "错误", "预测失败");
        statusBar()->showMessage("预测失败");
        return;
    }

    // 验证结果数量
    if (predictions_27.size() != static_cast<size_t>(loader_->getNumFaces())) {
        QMessageBox::critical(this, "错误",
            QString("预测结果数量(%1)与面数量(%2)不匹配")
            .arg(predictions_27.size())
            .arg(loader_->getNumFaces()));
        statusBar()->showMessage("预测错误");
        return;
    }

    // 27 类 → 4 类映射
    // 映射规则: 0=chamfer, 23=round, 1/12=hole, 其余=other(3)
    predictions_.clear();
    predictions_.reserve(predictions_27.size());
    for (int cls27 : predictions_27) {
        int cls4;
        switch (cls27) {
            case 0:  cls4 = 0; break;  // chamfer
            case 23: cls4 = 1; break;  // round
            case 1:
            case 12: cls4 = 2; break;  // hole
            default: cls4 = 3; break;  // other
        }
        predictions_.push_back(cls4);
    }

    // 更新颜色: other(3) 用暖色系区分每个面，其余用固定类颜色
    int numFaces = static_cast<int>(predictions_.size());
    warmOtherColors_ = ColorMapper::generateOtherColors(numFaces);
    std::vector<Quantity_Color> colors;
    for (int i = 0; i < numFaces; ++i) {
        int classId = predictions_[i];
        if (classId == 3) {
            colors.push_back(warmOtherColors_[i]);
        } else {
            colors.push_back(colorMapper_->getColor(classId));
        }
    }
    viewer_->updateAllFaceColors(colors);

    // 同步预测结果到 manualLabels_，使其成为当前工作状态
    manualLabels_ = predictions_;

    // 更新按钮状态
    setWorkMode(WorkMode::Prediction);

    // 更新统计信息
    updateStatistics();

    statusBar()->showMessage("预测完成");
    QApplication::processEvents();
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

void MainWindow::onExportResults() {
    // 优先导出人工标注，其次导出预测结果
    bool hasManual = !manualLabels_.empty();
    bool hasPrediction = !predictions_.empty();

    if (!hasManual && !hasPrediction) {
        QMessageBox::warning(this, "警告", "没有可导出的结果");
        return;
    }

    // 默认保存在 STEP 文件同目录，文件名为 STEP 文件名（.seg 扩展名）
    QFileInfo stepInfo(currentFilePath_);
    QString baseDir = stepInfo.absolutePath();
    QString baseName = stepInfo.completeBaseName();

    QString fileName = QString("%1/%2.seg").arg(baseDir, baseName);

    // 文件已存在时弹出替换确认
    if (QFileInfo::exists(fileName)) {
        QMessageBox::StandardButton reply = QMessageBox::question(
            this, "文件已存在",
            QString("文件已存在:\n%1\n\n是否替换？").arg(fileName),
            QMessageBox::Yes | QMessageBox::No);
        if (reply != QMessageBox::Yes) return;
    }

    QFile file(fileName);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QMessageBox::critical(this, "错误", "无法创建文件");
        return;
    }

    QTextStream out(&file);
    out.setCodec("UTF-8");

    // 直接写入标签（每行一个，无文件头注释）
    const std::vector<int>& labels = hasManual ? manualLabels_ : predictions_;
    for (int classId : labels) {
        out << classId << "\n";
    }

    file.close();
    QString source = hasManual ? "人工标注" : "预测结果";
    statusBar()->showMessage(source + "已导出: " + fileName);
    QMessageBox::information(this, "成功", source + "已导出至:\n" + fileName);

    // 刷新侧边栏：将当前文件对应的列表项标记为已标注（绿色粗体）
    if (!stepFiles_.isEmpty() && !currentFilePath_.isEmpty()) {
        for (int i = 0; i < stepFiles_.size(); ++i) {
            if (QFileInfo(stepFiles_[i]) == QFileInfo(currentFilePath_)) {
                QListWidgetItem* item = fileListWidget_->item(i);
                if (item) {
                    QBrush greenBrush(QColor(0, 140, 0));
                    item->setForeground(greenBrush);
                    QFont f = item->font();
                    f.setBold(true);
                    item->setFont(f);
                    item->setToolTip(stepFiles_[i] + "\n[已有标注: " + fileName + "]");
                }
                break;
            }
        }
        // 更新计数
        int hasSegCount = 0;
        for (int i = 0; i < stepFiles_.size(); ++i) {
            QFileInfo fi(stepFiles_[i]);
            QString segPath = fi.absolutePath() + "/" + fi.completeBaseName() + ".seg";
            if (QFileInfo::exists(segPath)) ++hasSegCount;
        }
        lblFileIndex_->setText(QString("共 %1 个文件 (%2 个已标注)")
            .arg(stepFiles_.size()).arg(hasSegCount));
    }
}

void MainWindow::onFaceSelected(int faceIndex) {
    if (faceIndex < 0 || faceIndex >= loader_->getNumFaces()) {
        return;
    }

    // 修改面类别后刷新模型信息（由 onModifyFaceClass 调用）
    QString info = QString("当前面: #%1").arg(faceIndex);

    if (!manualLabels_.empty() && faceIndex < static_cast<int>(manualLabels_.size())) {
        int classId = manualLabels_[faceIndex];
        QString className = QString::fromStdString(colorMapper_->getClassName(classId));
        info += QString(" | 类别: %1(%2)").arg(className).arg(classId);
    }

    if (!groundTruthLabels_.empty() && faceIndex < static_cast<int>(groundTruthLabels_.size())) {
        int trueClassId = groundTruthLabels_[faceIndex];
        QString trueClassName = QString::fromStdString(colorMapper_->getClassName(trueClassId));
        info += QString(" | 真实: %1(%2)").arg(trueClassName).arg(trueClassId);
        int currentClassId = (faceIndex < static_cast<int>(manualLabels_.size())) ? manualLabels_[faceIndex] : -1;
        info += (currentClassId == trueClassId) ? QString::fromUtf8(" ✓") : QString::fromUtf8(" ✗");
    }

    lblSelectedFace_->setText(info);
}

void MainWindow::onFaceHovered(int faceIndex, int mouseX, int mouseY) {
    if (faceIndex < 0 || !loader_ || faceIndex >= loader_->getNumFaces()) {
        // 没有模型时清空
        if (!loader_) {
            lblHoverFaceIndex_->setText("面索引: --");
            lblHoverEdgeCount_->setText("Edge 数: --");
            lblHoverFaceEdgeIds_->setText("Edge ID(coedge): --");
            lblHoverEdgeId_->setText("Edge ID: --");
            lblHoverEdgeType_->setText("类型: --");
            lblHoverCoedgeId_->setText("Coedge ID(Face): --");
            prevHoveredGlobalEdgeId_ = -1;
            return;
        }
        // 鼠标离开视图区域：保留上次信息不清空，仅清除边高亮
        prevHoveredGlobalEdgeId_ = -1;
        return;
    }

    // ========== 1. 悬停面：面索引 + Edge 总数 + Edge ID(coedge)列表 ==========
    // 不清空 Edge 信息！等采样完成后决定：检测到新 Edge 就更新，否则保留旧信息或清空

    lblHoverFaceIndex_->setText(QString("面索引: #%1").arg(faceIndex));

    // 更新模型信息中的"当前面"信息（基于悬停面）
    QString faceInfo = QString("当前面: #%1").arg(faceIndex);
    if (!manualLabels_.empty() && faceIndex < static_cast<int>(manualLabels_.size())) {
        int classId = manualLabels_[faceIndex];
        QString className = QString::fromStdString(colorMapper_->getClassName(classId));
        faceInfo += QString(" | 类别: %1(%2)").arg(className).arg(classId);
    }
    if (!groundTruthLabels_.empty() && faceIndex < static_cast<int>(groundTruthLabels_.size())) {
        int trueClassId = groundTruthLabels_[faceIndex];
        QString trueClassName = QString::fromStdString(colorMapper_->getClassName(trueClassId));
        faceInfo += QString(" | 真实: %1(%2)").arg(trueClassName).arg(trueClassId);
        int currentClassId = (faceIndex < static_cast<int>(manualLabels_.size())) ? manualLabels_[faceIndex] : -1;
        faceInfo += (currentClassId == trueClassId) ? QString::fromUtf8(" ✓") : QString::fromUtf8(" ✗");
    }
    lblSelectedFace_->setText(faceInfo);

    const auto& coedgeList = faceEdgeCoedge_[faceIndex];
    lblHoverEdgeCount_->setText(QString("Edge 数: %1").arg(coedgeList.size()));

    QStringList faceEdgeCoedgeStr;
    for (size_t ci = 0; ci < coedgeList.size(); ++ci) {
        int gid = coedgeList[ci];
        // 全局 coedge 索引（0-based，与 Python npz 一致）
        int thisGlobalCoedge = faceCoedgeOffset_[faceIndex] + static_cast<int>(ci);
        // 查找该 edge 的另一个 coedge 的全局索引
        int otherGlobalCoedge = -1;
        auto cit = edgeToCoedges_.find(gid);
        if (cit != edgeToCoedges_.end()) {
            for (const auto& p : cit->second) {
                if (p.first != faceIndex) {
                    otherGlobalCoedge = faceCoedgeOffset_[p.first] + p.second;
                    break;
                }
            }
        }
        if (otherGlobalCoedge >= 0) {
            faceEdgeCoedgeStr << QString("%1(%2,%3)").arg(gid).arg(thisGlobalCoedge).arg(otherGlobalCoedge);
        } else {
            faceEdgeCoedgeStr << QString("%1(%2)").arg(gid).arg(thisGlobalCoedge);
        }
    }
    lblHoverFaceEdgeIds_->setText("Edge ID(coedge): " + (faceEdgeCoedgeStr.isEmpty()
        ? QString("--") : faceEdgeCoedgeStr.join(", ")));

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

    const TopoDS_Face& face = loader_->getFaces()[faceIndex];
    for (TopExp_Explorer exp(face, TopAbs_EDGE); exp.More(); exp.Next()) {
        const TopoDS_Edge& edge = TopoDS::Edge(exp.Current());
        if (BRep_Tool::Degenerated(edge)) continue;

        int gid = -1;
        auto it = edgeGlobalIdMap_.find(edge.TShape().operator->());
        if (it != edgeGlobalIdMap_.end()) gid = it->second;

        BRepAdaptor_Curve curveAdaptor(edge);
        Standard_Real tFirst = curveAdaptor.FirstParameter();
        Standard_Real tLast  = curveAdaptor.LastParameter();

        // 估算曲线在屏幕上的像素长度，决定采样数（保证间隔 ≤6px，最多120点）
        const int ESTIMATE_POINTS = 10;
        int numSamples = 5;  // 最少 5 点
        {
            Standard_Real tPrev = tFirst;
            Standard_Integer sxPrev, syPrev;
            gp_Pnt ptPrev = curveAdaptor.Value(tPrev);
            view->Convert(ptPrev.X(), ptPrev.Y(), ptPrev.Z(), sxPrev, syPrev);
            double screenLen = 0.0;
            for (int si = 1; si <= ESTIMATE_POINTS; ++si) {
                double t = tFirst + (tLast - tFirst) * si / ESTIMATE_POINTS;
                gp_Pnt pt = curveAdaptor.Value(t);
                Standard_Integer sx, sy;
                view->Convert(pt.X(), pt.Y(), pt.Z(), sx, sy);
                screenLen += sqrt((double)((sx - sxPrev) * (sx - sxPrev) + (sy - syPrev) * (sy - syPrev)));
                tPrev = t; sxPrev = sx; syPrev = sy;
            }
            numSamples = std::max(5, std::min(120, (int)(screenLen / 6.0 + 0.5)));
        }
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

    // ========== 3. 悬停边：显示全局 Edge ID + 类型 + Coedge 归属 ==========
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

        // Coedge 归属信息（全局索引）
        auto cit = edgeToCoedges_.find(bestGlobalEdgeId);
        if (cit != edgeToCoedges_.end()) {
            QStringList coedgeParts;
            for (const auto& p : cit->second) {
                // 全局 coedge 索引（0-based，与 Python npz 一致）
                int globalCoedge = faceCoedgeOffset_[p.first] + p.second;
                coedgeParts << QString("%1(%2)").arg(globalCoedge).arg(p.first);
            }
            lblHoverCoedgeId_->setText("Coedge ID(Face): " + coedgeParts.join(", "));
        } else {
            lblHoverCoedgeId_->setText("Coedge ID(Face): --");
        }

        prevHoveredGlobalEdgeId_ = bestGlobalEdgeId;
    }
    // 未检测到 Edge → 保留上次 Edge 信息，不主动清空
    //（仅在加载新模型 / 重置时由外部清零）
}

void MainWindow::updateModelInfo() {
    lblFileName_->setText("文件: " + QString::fromStdString(loader_->getFileName()));
    lblNumFaces_->setText(QString("面数: %1").arg(loader_->getNumFaces()));
}

void MainWindow::updateStatistics() {
    // manualLabels_ 始终代表当前工作状态（预测后同步、修改后更新）
    if (manualLabels_.empty()) {
        txtStatistics_->setPlainText("等待操作...");
        return;
    }

    // 统计每个类别的面ID及数量
    std::map<int, std::vector<int>> classToFaces;
    for (int i = 0; i < static_cast<int>(manualLabels_.size()); ++i) {
        classToFaces[manualLabels_[i]].push_back(i);
    }

    // 生成统计报告
    QString report = "类别分布\n";
    report += "============\n";

    for (const auto& pair : classToFaces) {
        QString className = QString::fromStdString(colorMapper_->getClassName(pair.first));
        report += QString("%1(%2): ").arg(className).arg(pair.second.size());
        QStringList faceIds;
        for (int fid : pair.second) {
            faceIds << QString::number(fid);
        }
        report += faceIds.join(", ") + "\n";
    }

    report += QString("总计: %1个面").arg(manualLabels_.size());

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

    loadAndApplyLabels(fileName);
}

void MainWindow::onLoadAutoLabels() {
    if (currentFilePath_.isEmpty()) return;

    QFileInfo stepInfo(currentFilePath_);
    QString baseDir = stepInfo.absolutePath();
    QString baseName = stepInfo.completeBaseName();
    QString segPath = QString("%1/%2.seg").arg(baseDir, baseName);

    if (!QFileInfo::exists(segPath)) {
        QMessageBox::warning(this, "未找到标签文件",
            QString("在STEP文件所在目录未找到同名的.seg文件:\n%1").arg(segPath));
        return;
    }

    loadAndApplyLabels(segPath);
}

void MainWindow::loadAndApplyLabels(const QString& fileName, bool silent) {
    manualLabels_ = loadLabelsFromFile(fileName);

    if (manualLabels_.empty()) {
        if (!silent) QMessageBox::critical(this, "错误", "无法解析标签文件");
        return;
    }

    // 验证标签数量
    if (static_cast<int>(manualLabels_.size()) != loader_->getNumFaces()) {
        if (!silent) QMessageBox::warning(this, "警告",
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
    updateStatistics();

    statusBar()->showMessage("标签已加载并上色: " + fileName);
    if (!silent) QMessageBox::information(this, "成功", "标签已加载并按类别着色。");
}

void MainWindow::onFaceSelectionChanged(int faceIndex) {
    const QSet<int>& selected = viewer_->getMultiSelectedFaces();
    if (selected.isEmpty()) {
        statusBar()->showMessage("多选已清空", 2000);
    } else {
        QStringList faceIds;
        for (int fid : selected) {
            faceIds << QString::number(fid);
        }
        statusBar()->showMessage(
            QString("已选中 %1 个面: %2 (右键单击批量修改)")
            .arg(selected.size()).arg(faceIds.join(", ")), 5000);
    }
}

void MainWindow::onModifyFaceClass() {
    // 获取多选集合或单选面
    QSet<int> selectedFaces = viewer_->getMultiSelectedFaces();
    int selectedFace = viewer_->getSelectedFaceIndex();

    // 如果没有多选，使用单选面
    bool isBatch = !selectedFaces.isEmpty();
    if (!isBatch) {
        if (selectedFace < 0) {
            QMessageBox::warning(this, "警告", "请先右键点击选中一个面");
            return;
        }
        selectedFaces.insert(selectedFace);
    }

    // 构建提示信息
    QString title = isBatch ? QString("批量修改 %1 个面的类别").arg(selectedFaces.size()) : "修改面类别";
    QString infoText;
    if (isBatch) {
        QStringList faceIds;
        for (int fid : selectedFaces) {
            faceIds << QString::number(fid);
        }
        infoText = QString("选中面: %1\n\n请选择新类别:").arg(faceIds.join(", "));
    } else {
        int currentClass = manualLabels_[selectedFace];
        QString currentClassName = QString::fromStdString(colorMapper_->getClassName(currentClass));
        infoText = QString("当前面 #%1\n当前类别: %2 (%3)\n\n请选择新类别:")
            .arg(selectedFace).arg(currentClass).arg(currentClassName);
    }

    // 自定义对话框：4 个彩色按钮
    QDialog dlg(this);
    dlg.setWindowTitle(title);
    QVBoxLayout* dlgLayout = new QVBoxLayout(&dlg);

    QLabel* infoLbl = new QLabel(infoText, &dlg);
    dlgLayout->addWidget(infoLbl);

    int newClass = -1;
    int numClasses = colorMapper_->getNumClasses();
    for (int i = 0; i < numClasses; ++i) {
        Quantity_Color qc = colorMapper_->getColor(i);
        int r = static_cast<int>(qc.Red() * 255);
        int g = static_cast<int>(qc.Green() * 255);
        int b = static_cast<int>(qc.Blue() * 255);
        double luma = 0.299 * qc.Red() + 0.587 * qc.Green() + 0.114 * qc.Blue();
        QString textColor = (luma > 0.55) ? "black" : "white";

        QPushButton* btn = new QPushButton(
            QString("%1 - %2  [按 %1 键]").arg(i).arg(QString::fromStdString(colorMapper_->getClassName(i))),
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

        // 键盘快捷键: 数字键 0-3 和小键盘 0-3
        auto* scMain = new QShortcut(QKeySequence(Qt::Key_0 + i), &dlg);
        auto* scNumpad = new QShortcut(QKeySequence(Qt::KeypadModifier | (Qt::Key_0 + i)), &dlg);
        connect(scMain, &QShortcut::activated, &dlg, [&dlg, &newClass, i]() {
            newClass = i;
            dlg.accept();
        });
        connect(scNumpad, &QShortcut::activated, &dlg, [&dlg, &newClass, i]() {
            newClass = i;
            dlg.accept();
        });
    }

    QPushButton* cancelBtn = new QPushButton("取消", &dlg);
    connect(cancelBtn, &QPushButton::clicked, &dlg, &QDialog::reject);
    dlgLayout->addWidget(cancelBtn);

    if (dlg.exec() != QDialog::Accepted || newClass < 0) return;

    // 批量更新标签和颜色
    for (int faceIdx : selectedFaces) {
        manualLabels_[faceIdx] = newClass;

        Quantity_Color newColor;
        if (newClass == 3 && faceIdx >= 0 && faceIdx < (int)warmOtherColors_.size()) {
            newColor = warmOtherColors_[faceIdx];
        } else {
            newColor = colorMapper_->getColor(newClass);
        }
        viewer_->updateSingleFaceColor(faceIdx, newColor);
    }

    // 清除多选集合
    viewer_->clearMultiSelection();

    // 更新面信息
    if (selectedFace >= 0) {
        onFaceSelected(selectedFace);
    }

    // 更新统计
    updateStatistics();

    QString newClassName = QString::fromStdString(colorMapper_->getClassName(newClass));
    if (isBatch) {
        statusBar()->showMessage(
            QString("已批量修改 %1 个面的类别为 %2(%3)")
            .arg(selectedFaces.size()).arg(newClassName).arg(newClass));
    } else {
        statusBar()->showMessage(
            QString("面 #%1 类别已修改为 %2(%3)")
            .arg(selectedFace).arg(newClassName).arg(newClass));
    }
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
    viewer_->clearMultiSelection();

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
    lblSelectedFace_->setText("当前面: --");
    lblHoverFaceIndex_->setText("面索引: --");
    lblHoverEdgeCount_->setText("Edge 数: --");
    lblHoverFaceEdgeIds_->setText("Edge ID: --");
    lblHoverEdgeId_->setText("Edge ID: --");
    lblHoverEdgeType_->setText("类型: --");
    lblHoverCoedgeId_->setText("Coedge ID(Face): --");
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
