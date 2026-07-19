#include "OCCTViewer.h"
#include <QMouseEvent>
#include <QWheelEvent>
#include <iostream>

#include <Aspect_DisplayConnection.hxx>
#include <OpenGl_GraphicDriver.hxx>
#include <TopExp_Explorer.hxx>
#include <Prs3d_Drawer.hxx>
#include <Graphic3d_MaterialAspect.hxx>
#include <Prs3d_LineAspect.hxx>
#include <Prs3d_IsoAspect.hxx>
#include <IntCurvesFace_Intersector.hxx>
#include <BRep_Tool.hxx>
#include <Precision.hxx>
#include <BRep_Builder.hxx>
#include <TopoDS_Compound.hxx>
#include <TopoDS_Edge.hxx>

OCCTViewer::OCCTViewer(QWidget* parent)
    : QWidget(parent)
    , currentMode_(None)
    , selectedFaceIndex_(-1)
    , previousSelectedFaceIndex_(-1)
    , hoveredFaceIndex_(-1) {

    setAttribute(Qt::WA_PaintOnScreen, true);
    setAttribute(Qt::WA_NoSystemBackground, true);
    setAttribute(Qt::WA_NativeWindow, true);
    setAutoFillBackground(false);
    setFocusPolicy(Qt::StrongFocus);
    setMouseTracking(true);

    winId();  // Force native window creation
    initOCCT();
}

OCCTViewer::~OCCTViewer() {
    if (!context_.IsNull()) {
        clearAll();
    }
}

void OCCTViewer::initOCCT() {
    Handle(Aspect_DisplayConnection) aDispConnection = new Aspect_DisplayConnection();
    Handle(OpenGl_GraphicDriver) aGraphicDriver = new OpenGl_GraphicDriver(aDispConnection);

    viewer_ = new V3d_Viewer(aGraphicDriver);
    viewer_->SetDefaultLights();
    viewer_->SetLightOn();

    view_ = viewer_->CreateView();

#ifdef _WIN32
    hWnd_ = new WNT_Window((Aspect_Handle)winId());
    view_->SetWindow(hWnd_);
    if (!hWnd_->IsMapped()) hWnd_->Map();
#endif

    context_ = new AIS_InteractiveContext(viewer_);
    view_->SetBackgroundColor(Quantity_NOC_WHITE);
    view_->SetProj(V3d_XposYnegZpos);
    view_->TriedronDisplay(Aspect_TOTP_LEFT_LOWER, Quantity_NOC_GOLD, 0.08);
    view_->MustBeResized();
}

void OCCTViewer::paintEvent(QPaintEvent*) {
    if (!view_.IsNull()) view_->Redraw();
}

void OCCTViewer::resizeEvent(QResizeEvent*) {
    if (!view_.IsNull()) view_->MustBeResized();
}

void OCCTViewer::displayFaces(const std::vector<TopoDS_Face>& faces, const TopoDS_Shape& fullShape) {
    if (context_.IsNull()) return;
    clearAll();

    fullShape_ = fullShape;

    // 单个 AIS_ColoredShape：整个模型一个三角化，边界天然连续
    coloredShape_ = new AIS_ColoredShape(fullShape);
    coloredShape_->SetDisplayMode(AIS_Shaded);
    coloredShape_->SetMaterial(Graphic3d_NOM_PLASTIC);
    // 开启 FaceBoundaryDraw：来自单一三角化，共享 edge 只画一次
    coloredShape_->Attributes()->SetFaceBoundaryDraw(true);
    // 过滤 seam edge（缝合边）：圆柱/球面等周期面的接缝在几何上 C1 连续，
    // 设为 GeomAbs_C1 后只画 C0 及以下连续性的真实棱边，圆柱面竖线消失。
    coloredShape_->Attributes()->SetFaceBoundaryUpperContinuity(GeomAbs_C1);
    coloredShape_->Attributes()->SetIsoOnTriangulation(Standard_False);
    coloredShape_->Attributes()->UIsoAspect()->SetNumber(0);
    coloredShape_->Attributes()->VIsoAspect()->SetNumber(0);
    coloredShape_->Attributes()->SetDeviationAngle(1.0);
    coloredShape_->Attributes()->SetDeviationCoefficient(0.001);
    // 边界线颜色：中灰，细线
    Handle(Prs3d_LineAspect) boundaryAspect = new Prs3d_LineAspect(
        Quantity_NOC_GRAY50, Aspect_TOL_SOLID, 1.0);
    coloredShape_->Attributes()->SetFaceBoundaryAspect(boundaryAspect);

    // 为每个面设置初始颜色（灰色）
    Quantity_Color grayColor(0.7, 0.7, 0.7, Quantity_TOC_RGB);
    for (size_t i = 0; i < faces.size(); ++i) {
        coloredShape_->SetCustomColor(faces[i], grayColor);
        faceColors_[static_cast<int>(i)] = grayColor;
    }

    // Display 使用默认显示模式（shaded），默认选择模式（mode 0，整模型）
    context_->Display(coloredShape_, Standard_True);
    // 先禁用所有已激活的选择模式（mode 0 含整形状/顶点敏感实体，可能还有边选择 mode 等），
    // 避免鼠标悬停边/顶点时出现 OCCT 内建的蓝色高亮（圆圈或线条）。
    context_->Deactivate(coloredShape_);
    // 只激活面级选择模式（mode 1），使悬停/点击仅检测到具体面
    context_->Activate(coloredShape_, 1);

    // 创建悬停面边线高亮叠加层（与透明化配合，黑色细线显示该面轮廓）
    if (hoverEdges_.IsNull()) {
        hoverEdges_ = new AIS_Shape(TopoDS_Shape());
        hoverEdges_->SetDisplayMode(AIS_WireFrame); // 线框模式，只显示边
        hoverEdges_->SetColor(Quantity_Color(0.0, 0.0, 0.0, Quantity_TOC_RGB));
        hoverEdges_->SetWidth(1.5);
    }

    context_->UpdateCurrentViewer();
    fitAll();
}

void OCCTViewer::updateFaceColor(int faceIndex, const Quantity_Color& color) {
    if (coloredShape_.IsNull() || fullShape_.IsNull()) return;
    applyFaceColor(faceIndex, color);
    context_->Redisplay(coloredShape_, Standard_False);
    context_->UpdateCurrentViewer();
}

void OCCTViewer::updateAllFaceColors(const std::vector<Quantity_Color>& colors) {
    if (coloredShape_.IsNull() || fullShape_.IsNull()) return;
    if (colors.size() != faceColors_.size()) return;

    TopExp_Explorer exp(fullShape_, TopAbs_FACE);
    int idx = 0;
    for (; exp.More(); exp.Next(), ++idx) {
        faceColors_[idx] = colors[idx];
        coloredShape_->SetCustomColor(exp.Current(), colors[idx]);
    }

    context_->Redisplay(coloredShape_, Standard_False);
    context_->UpdateCurrentViewer();
}

void OCCTViewer::updateSingleFaceColor(int faceIndex, const Quantity_Color& color) {
    if (coloredShape_.IsNull() || fullShape_.IsNull()) return;
    applyFaceColor(faceIndex, color);
    context_->Redisplay(coloredShape_, Standard_False);
    context_->UpdateCurrentViewer();
}

void OCCTViewer::resetAllFaceColors() {
    if (context_.IsNull() || coloredShape_.IsNull() || fullShape_.IsNull()) return;

    Quantity_Color grayColor(0.7, 0.7, 0.7, Quantity_TOC_RGB);
    TopExp_Explorer exp(fullShape_, TopAbs_FACE);
    int idx = 0;
    for (; exp.More(); exp.Next(), ++idx) {
        faceColors_[idx] = grayColor;
        coloredShape_->SetCustomColor(exp.Current(), grayColor);
    }

    // 清除选中状态
    previousSelectedFaceIndex_ = -1;
    selectedFaceIndex_ = -1;
    savedFaceColors_.clear();

    context_->Redisplay(coloredShape_, Standard_False);
    context_->UpdateCurrentViewer();
}

void OCCTViewer::highlightErrorFaces(const std::vector<int>& errorIndices, bool highlight) {
    if (context_.IsNull() || coloredShape_.IsNull() || fullShape_.IsNull()) return;

    if (highlight) {
        // 保存原色并设为淡红
        Quantity_Color errorColor(1.0, 0.6, 0.6, Quantity_TOC_RGB);
        for (int idx : errorIndices) {
            auto it = faceColors_.find(idx);
            if (it != faceColors_.end()) {
                savedFaceColors_[idx] = it->second;  // 保存原色
                applyFaceColor(idx, errorColor);
            }
        }
    } else {
        // 恢复原色
        for (int idx : errorIndices) {
            auto savedIt = savedFaceColors_.find(idx);
            if (savedIt != savedFaceColors_.end()) {
                applyFaceColor(idx, savedIt->second);
                savedFaceColors_.erase(savedIt);
            }
        }
    }

    context_->Redisplay(coloredShape_, Standard_False);
    context_->UpdateCurrentViewer();
}

void OCCTViewer::clearErrorHighlights() {
    if (context_.IsNull() || coloredShape_.IsNull() || fullShape_.IsNull()) return;

    // 恢复所有保存的原色
    for (auto& pair : savedFaceColors_) {
        applyFaceColor(pair.first, pair.second);
    }
    savedFaceColors_.clear();

    context_->Redisplay(coloredShape_, Standard_False);
    context_->UpdateCurrentViewer();
}

void OCCTViewer::clearAll() {
    if (context_.IsNull()) return;
    if (!hoverEdges_.IsNull()) {
        context_->Erase(hoverEdges_, Standard_False);
        hoverEdges_.Nullify();
    }
    if (!multiSelectEdges_.IsNull()) {
        context_->Erase(multiSelectEdges_, Standard_False);
        multiSelectEdges_.Nullify();
    }
    multiSelectedFaces_.clear();
    if (!coloredShape_.IsNull()) {
        context_->Remove(coloredShape_, Standard_False);
        coloredShape_.Nullify();
    }
    fullShape_.Nullify();
    faceColors_.clear();
    savedFaceColors_.clear();
    previousSelectedFaceIndex_ = -1;
    selectedFaceIndex_ = -1;
    hoveredFaceIndex_ = -1;
    context_->UpdateCurrentViewer();
}

void OCCTViewer::fitAll() {
    if (view_.IsNull()) return;
    view_->FitAll();
    view_->ZFitAll();
    view_->Redraw();
}

void OCCTViewer::mousePressEvent(QMouseEvent* event) {
    lastMousePos_ = event->pos();
    double dpr = devicePixelRatio();
    QPoint devPos = event->pos() * dpr;

    if (event->button() == Qt::LeftButton) {
        // 左键：旋转
        currentMode_ = Rotate;
        view_->StartRotation(devPos.x(), devPos.y());
    } else if (event->button() == Qt::RightButton) {
        // 右键单击：修改当前面类别（或 Ctrl+右键多选）
        if (context_.IsNull() || view_.IsNull()) return;
        context_->MoveTo(devPos.x(), devPos.y(), view_, Standard_True);
        int pickedFace = pickFaceAtPos(devPos.x(), devPos.y());

        if (event->modifiers() & Qt::ControlModifier) {
            // Ctrl+右键：添加/移除多选集合，点击空白则清空所有
            if (pickedFace >= 0) {
                if (multiSelectedFaces_.contains(pickedFace)) {
                    multiSelectedFaces_.remove(pickedFace);
                } else {
                    multiSelectedFaces_.insert(pickedFace);
                }
                updateMultiSelectHighlight();
                emit faceSelectionChanged(pickedFace);
            } else {
                // 点击空白 → 清空多选
                multiSelectedFaces_.clear();
                updateMultiSelectHighlight();
                emit faceSelectionChanged(-1);
            }
        } else {
            // 普通右键：修改面类别
            if (pickedFace >= 0) {
                selectedFaceIndex_ = pickedFace;
                previousSelectedFaceIndex_ = pickedFace;
                emit faceModifyRequested(pickedFace);
            }
        }
        currentMode_ = None;
    } else if (event->button() == Qt::MiddleButton) {
        // 中键：平移
        currentMode_ = Pan;
    }
}

void OCCTViewer::mouseDoubleClickEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton) {
        fitAll();  // 左键双击：重置视图
    }
}

void OCCTViewer::mouseMoveEvent(QMouseEvent* event) {
    QPoint pos = event->pos();
    double dpr = devicePixelRatio();
    QPoint devPos(pos.x() * dpr, pos.y() * dpr);

    if (currentMode_ == None) {
        // 无按键按下 → 悬停检测（用 3px 阈值限频）
        if (!view_.IsNull() && (pos - lastMousePos_).manhattanLength() > 3) {
            lastMousePos_ = pos;
            context_->MoveTo(devPos.x(), devPos.y(), view_, Standard_True);
            checkHoveredFace();
            // 屏蔽 OCCT 内建线框高亮，悬停视觉反馈完全由 hoverHighlight_ 透明叠加层负责
            context_->ClearDetected(Standard_True);
        }
        return;
    }

    if (view_.IsNull()) return;

    if (currentMode_ == Rotate) {
        view_->Rotation(devPos.x(), devPos.y());
    } else if (currentMode_ == Pan) {
        view_->Pan(devPos.x() - lastMousePos_.x() * dpr,
                   lastMousePos_.y() * dpr - devPos.y());
    }
    lastMousePos_ = pos;
    view_->Redraw();
}

void OCCTViewer::mouseReleaseEvent(QMouseEvent* event) {
    currentMode_ = None;
}

void OCCTViewer::wheelEvent(QWheelEvent* event) {
    if (view_.IsNull()) return;
    view_->SetZoom(event->angleDelta().y() > 0 ? 1.1 : 0.9);
    view_->Redraw();
}

void OCCTViewer::checkHoveredFace() {
    if (context_.IsNull()) return;

    double dpr = devicePixelRatio();
    Standard_Integer devX = lastMousePos_.x() * dpr;
    Standard_Integer devY = lastMousePos_.y() * dpr;

    int pickedFace = pickFaceAtPos(devX, devY);

    // 更新悬停状态：让悬停面本身变透明（保留原色），透过它能看到模型内部结构
    if (pickedFace >= 0 && pickedFace != hoveredFaceIndex_) {
        // 恢复上一个悬停面的透明度
        if (hoveredFaceIndex_ >= 0) {
            setFaceTransparency(hoveredFaceIndex_, 0.0);
        }
        // 让新悬停面变透明（0.5 = 50% 透明，既能看到内部又不至于太透）
        setFaceTransparency(pickedFace, 0.5);
        context_->Redisplay(coloredShape_, Standard_True);
        // 高亮显示该面的所有边线（红色粗线叠加层）
        TopoDS_Compound edgesCompound = buildFaceEdgesCompound(pickedFace);
        if (!edgesCompound.IsNull() && !hoverEdges_.IsNull()) {
            hoverEdges_->Set(edgesCompound);
            hoverEdges_->SetColor(Quantity_Color(0.0, 0.0, 0.0, Quantity_TOC_RGB)); // 黑色
            hoverEdges_->SetWidth(1.5);
            if (context_->IsDisplayed(hoverEdges_)) {
                context_->Redisplay(hoverEdges_, Standard_True);
            } else {
                context_->Display(hoverEdges_, Standard_False);
            }
        }
        context_->UpdateCurrentViewer();
        hoveredFaceIndex_ = pickedFace;
        emit faceHovered(pickedFace, lastMousePos_.x(), lastMousePos_.y());
        return;
    }

    // 同一个面，无需更新
    if (pickedFace == hoveredFaceIndex_) {
        return;
    }

    // 没有检测到任何面 → 恢复透明度
    restoreHoveredFace();
}

void OCCTViewer::restoreHoveredFace() {
    if (hoveredFaceIndex_ >= 0) {
        setFaceTransparency(hoveredFaceIndex_, 0.0);
        context_->Redisplay(coloredShape_, Standard_True);
        // 隐藏边线高亮叠加层
        if (!hoverEdges_.IsNull() && context_->IsDisplayed(hoverEdges_)) {
            context_->Erase(hoverEdges_, Standard_False);
        }
        context_->UpdateCurrentViewer();
        hoveredFaceIndex_ = -1;
        emit faceHovered(-1, 0, 0);
    }
}

void OCCTViewer::updateMultiSelectHighlight() {
    if (context_.IsNull()) return;

    // 初始化叠加层
    if (multiSelectEdges_.IsNull()) {
        multiSelectEdges_ = new AIS_Shape(TopoDS_Shape());
        multiSelectEdges_->SetDisplayMode(AIS_WireFrame);
        multiSelectEdges_->SetColor(Quantity_Color(0.0, 0.0, 0.0, Quantity_TOC_RGB)); // 纯黑
        multiSelectEdges_->SetWidth(2.5);
    }

    if (multiSelectedFaces_.isEmpty()) {
        // 清空高亮
        if (context_->IsDisplayed(multiSelectEdges_)) {
            context_->Erase(multiSelectEdges_, Standard_False);
            context_->UpdateCurrentViewer();
        }
        return;
    }

    // 构建所有多选面的边线 compound
    TopoDS_Compound compound;
    BRep_Builder builder;
    builder.MakeCompound(compound);
    bool hasEdge = false;

    for (int faceIdx : multiSelectedFaces_) {
        TopoDS_Compound faceEdges = buildFaceEdgesCompound(faceIdx);
        for (TopExp_Explorer exp(faceEdges, TopAbs_EDGE); exp.More(); exp.Next()) {
            builder.Add(compound, TopoDS::Edge(exp.Current()));
            hasEdge = true;
        }
    }

    if (!hasEdge) {
        if (context_->IsDisplayed(multiSelectEdges_)) {
            context_->Erase(multiSelectEdges_, Standard_False);
            context_->UpdateCurrentViewer();
        }
        return;
    }

    multiSelectEdges_->Set(compound);
    multiSelectEdges_->SetColor(Quantity_Color(0.0, 0.0, 0.0, Quantity_TOC_RGB));
    multiSelectEdges_->SetWidth(2.5);

    if (context_->IsDisplayed(multiSelectEdges_)) {
        context_->Redisplay(multiSelectEdges_, Standard_True);
    } else {
        context_->Display(multiSelectEdges_, Standard_False);
    }
    context_->UpdateCurrentViewer();
}

void OCCTViewer::setFaceTransparency(int faceIndex, Standard_Real transparency) {
    if (coloredShape_.IsNull() || fullShape_.IsNull()) return;
    TopExp_Explorer exp(fullShape_, TopAbs_FACE);
    int idx = 0;
    for (; exp.More(); exp.Next(), ++idx) {
        if (idx == faceIndex) {
            // AIS_ColoredShape 支持按子形状设置透明度，悬停面本身变透明
            coloredShape_->SetCustomTransparency(exp.Current(), transparency);
            break;
        }
    }
}

TopoDS_Compound OCCTViewer::buildFaceEdgesCompound(int faceIndex) const {
    TopoDS_Compound compound;
    if (fullShape_.IsNull() || faceIndex < 0) return compound;

    // 先定位到指定 face
    TopExp_Explorer faceExp(fullShape_, TopAbs_FACE);
    int idx = 0;
    for (; faceExp.More(); faceExp.Next(), ++idx) {
        if (idx == faceIndex) break;
    }
    if (!faceExp.More()) return compound; // 未找到

    // 收集该 face 的所有 edge 到 compound
    // 过滤 seam edge：用 BRep_Tool::IsClosed(edge, face) 检测。
    // seam edge（圆柱/球面等周期面的缝合边）在 OCCT 中 IsClosed 返回 true，
    // 这比 TShape 去重更可靠——seam edge 的两条拷贝方向不同导致 TShape 不同，
    // 但 IsClosed 能正确识别它们的几何闭合特性。
    BRep_Builder builder;
    builder.MakeCompound(compound);
    bool hasEdge = false;
    const TopoDS_Face& face = TopoDS::Face(faceExp.Current());
    for (TopExp_Explorer edgeExp(face, TopAbs_EDGE); edgeExp.More(); edgeExp.Next()) {
        const TopoDS_Edge& edge = TopoDS::Edge(edgeExp.Current());
        // 跳过 seam edge（周期面的缝合边）
        if (BRep_Tool::IsClosed(edge, face)) {
            continue;
        }
        builder.Add(compound, edge);
        hasEdge = true;
    }

    if (!hasEdge) {
        // 空的 compound，返回空（调用方会判 IsNull）
        return TopoDS_Compound();
    }
    return compound;
}

int OCCTViewer::pickFaceByRay(int mouseX, int mouseY) {
    if (view_.IsNull() || fullShape_.IsNull()) return -1;

    // 将 2D 鼠标位置转换为 3D 射线
    Standard_Real X, Y, Z, DX, DY, DZ;
    view_->ConvertWithProj(mouseX, mouseY, X, Y, Z, DX, DY, DZ);

    gp_Pnt origin(X, Y, Z);
    gp_Dir direction(DX, DY, DZ);
    gp_Lin ray(origin, direction);

    // 遍历所有面，找出与射线最近的面
    TopExp_Explorer exp(fullShape_, TopAbs_FACE);
    int idx = 0;
    int closestFace = -1;
    Standard_Real minDist = std::numeric_limits<double>::max();
    const Standard_Real tolerance = Precision::Confusion();

    for (; exp.More(); exp.Next(), ++idx) {
        const TopoDS_Face& face = TopoDS::Face(exp.Current());

        // IntCurvesFace_Intersector 直接处理 FACE（含裁剪），比 GeomAPI_IntCS 更鲁棒
        IntCurvesFace_Intersector intersector(face, tolerance, Standard_True, Standard_True);
        intersector.Perform(ray, -RealLast(), +RealLast());

        if (intersector.IsDone() && intersector.NbPnt() > 0) {
            for (int pi = 1; pi <= intersector.NbPnt(); ++pi) {
                // State() 返回 TopAbs_IN（内部）或 TopAbs_ON（边界上）
                TopAbs_State ptState = intersector.State(pi);
                if (ptState == TopAbs_OUT) continue;

                gp_Pnt hitPoint = intersector.Pnt(pi);
                gp_Vec diff(origin, hitPoint);
                Standard_Real projDist = diff.Dot(gp_Vec(direction));
                if (projDist <= 0) continue;

                if (projDist < minDist) {
                    minDist = projDist;
                    closestFace = idx;
                }
            }
        }
    }

    return closestFace;
}

int OCCTViewer::pickFaceAtPos(int devX, int devY) {
    if (context_.IsNull() || view_.IsNull()) return -1;

    int pickedFace = -1;

    // 方法一：通过 MainSelector 遍历所有检测到的实体
    const Handle(StdSelect_ViewerSelector3d)& aSelector = context_->MainSelector();
    if (!aSelector.IsNull()) {
        for (Standard_Integer i = 1; i <= aSelector->NbPicked(); i++) {
            Handle(SelectMgr_EntityOwner) anOwner = aSelector->Picked(i);
            Handle(StdSelect_BRepOwner) aBRepOwner =
                Handle(StdSelect_BRepOwner)::DownCast(anOwner);
            if (!aBRepOwner.IsNull()) {
                TopoDS_Shape aShape = aBRepOwner->Shape();
                if (!aShape.IsNull()) {
                    int idx = findFaceIndex(aShape);
                    if (idx >= 0) {
                        pickedFace = idx;
                        break;
                    }
                }
            }
        }
    }

    // 方法二：几何求交
    if (pickedFace < 0) {
        pickedFace = pickFaceByRay(devX, devY);
    }

    return pickedFace;
}

void OCCTViewer::handleSelection() {
    if (context_.IsNull() || view_.IsNull()) return;

    double dpr = devicePixelRatio();
    Standard_Integer devX = lastMousePos_.x() * dpr;
    Standard_Integer devY = lastMousePos_.y() * dpr;

    context_->MoveTo(devX, devY, view_, Standard_False);
    context_->Select(Standard_True);

    int pickedFace = pickFaceAtPos(devX, devY);

    if (pickedFace >= 0) {
        if (pickedFace == previousSelectedFaceIndex_) {
            emit faceSelected(pickedFace);
            return;
        }
        previousSelectedFaceIndex_ = pickedFace;
        selectedFaceIndex_ = pickedFace;
        context_->UpdateCurrentViewer();
        emit faceSelected(pickedFace);
    }
}

int OCCTViewer::findFaceIndex(const TopoDS_Shape& subShape) const {
    if (fullShape_.IsNull()) return -1;
    TopExp_Explorer exp(fullShape_, TopAbs_FACE);
    int idx = 0;
    for (; exp.More(); exp.Next(), ++idx) {
        if (exp.Current().IsSame(subShape)) {
            return idx;
        }
    }
    return -1;
}

void OCCTViewer::applyFaceColor(int faceIndex, const Quantity_Color& color) {
    if (coloredShape_.IsNull() || fullShape_.IsNull()) return;
    faceColors_[faceIndex] = color;
    setCustomFaceColor(faceIndex, color);
}

void OCCTViewer::setCustomFaceColor(int faceIndex, const Quantity_Color& color) {
    if (coloredShape_.IsNull() || fullShape_.IsNull()) return;
    TopExp_Explorer exp(fullShape_, TopAbs_FACE);
    int idx = 0;
    for (; exp.More(); exp.Next(), ++idx) {
        if (idx == faceIndex) {
            coloredShape_->SetCustomColor(exp.Current(), color);
            break;
        }
    }
}
