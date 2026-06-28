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
#include <GeomAPI_IntCS.hxx>
#include <BRep_Tool.hxx>
#include <BRepTools.hxx>
#include <Geom_Line.hxx>
#include <Geom_Surface.hxx>

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
    // 额外激活面级选择模式（mode 1），使悬停/点击可检测到具体面
    context_->Activate(coloredShape_, 1);
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
    if (!coloredShape_.IsNull()) {
        context_->Remove(coloredShape_, Standard_False);
        coloredShape_.Nullify();
    }
    fullShape_.Nullify();
    faceColors_.clear();
    savedFaceColors_.clear();
    previousSelectedFaceIndex_ = -1;
    selectedFaceIndex_ = -1;
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
    
    // 检查是否左右键同时按下（平移模式）
    Qt::MouseButtons buttons = event->buttons();
    if ((buttons & Qt::LeftButton) && (buttons & Qt::RightButton)) {
        currentMode_ = Pan;
        return;
    }

    if (event->button() == Qt::LeftButton) {
        currentMode_ = None;
        handleSelection();
    } else if (event->button() == Qt::RightButton) {
        currentMode_ = Rotate;
        view_->StartRotation(lastMousePos_.x(), lastMousePos_.y());
    } else if (event->button() == Qt::MiddleButton) {
        currentMode_ = Pan;
    }
}

void OCCTViewer::mouseDoubleClickEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton || event->button() == Qt::MiddleButton) {
        fitAll();  // Double-click to reset view
    }
}

void OCCTViewer::mouseMoveEvent(QMouseEvent* event) {
    QPoint pos = event->pos();

    if (currentMode_ == None) {
        // 无按键按下 → 悬停检测（用 3px 阈值限频）
        if (!view_.IsNull() && (pos - lastMousePos_).manhattanLength() > 3) {
            lastMousePos_ = pos;
            context_->MoveTo(pos.x(), pos.y(), view_, Standard_True);
            checkHoveredFace();
        }
        return;
    }

    if (view_.IsNull()) return;

    if (currentMode_ == Rotate) {
        view_->Rotation(pos.x(), pos.y());
    } else if (currentMode_ == Pan) {
        view_->Pan(pos.x() - lastMousePos_.x(), lastMousePos_.y() - pos.y());
    }
    lastMousePos_ = pos;
    view_->Redraw();
}

void OCCTViewer::mouseReleaseEvent(QMouseEvent* event) {
    // 检查是否还保持有其他按键
    Qt::MouseButtons buttons = event->buttons();
    if ((buttons & Qt::LeftButton) && (buttons & Qt::RightButton)) {
        // 仍然左右键同时按下，保持平移模式
        return;
    } else if (buttons & Qt::RightButton) {
        // 只按住右键，切换到旋转模式
        currentMode_ = Rotate;
        view_->StartRotation(lastMousePos_.x(), lastMousePos_.y());
    } else if (buttons & Qt::LeftButton) {
        // 只按住左键，切换到选择模式
        currentMode_ = None;
    } else {
        // 没有按键按下
        currentMode_ = None;
    }
}

void OCCTViewer::wheelEvent(QWheelEvent* event) {
    if (view_.IsNull()) return;
    view_->SetZoom(event->angleDelta().y() > 0 ? 1.1 : 0.9);
    view_->Redraw();
}

void OCCTViewer::checkHoveredFace() {
    if (context_.IsNull()) return;

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

    // 方法二：如果选取器没找到面 → 用几何求交（光线投射）
    if (pickedFace < 0) {
        pickedFace = pickFaceByRay(lastMousePos_.x(), lastMousePos_.y());
    }

    // 更新悬停状态
    if (pickedFace >= 0 && pickedFace != hoveredFaceIndex_) {
        // 恢复上一个悬停面
        if (hoveredFaceIndex_ >= 0) {
            setCustomFaceColor(hoveredFaceIndex_, hoverSavedColor_);
        }
        // 高亮新面
        auto it = faceColors_.find(pickedFace);
        if (it != faceColors_.end()) {
            hoverSavedColor_ = it->second;
            setCustomFaceColor(pickedFace, Quantity_NOC_LIGHTBLUE);
            hoveredFaceIndex_ = pickedFace;
            context_->RecomputePrsOnly(coloredShape_, Standard_False, Standard_False);
            context_->UpdateCurrentViewer();
            emit faceHovered(pickedFace, lastMousePos_.x(), lastMousePos_.y());
        }
        return;
    }

    // 同一个面，无需更新
    if (pickedFace == hoveredFaceIndex_) {
        return;
    }

    // 没有检测到任何面 → 清空悬停
    restoreHoveredFace();
}

void OCCTViewer::restoreHoveredFace() {
    if (hoveredFaceIndex_ >= 0) {
        setCustomFaceColor(hoveredFaceIndex_, hoverSavedColor_);
        hoveredFaceIndex_ = -1;
        context_->RecomputePrsOnly(coloredShape_, Standard_False, Standard_False);
        context_->UpdateCurrentViewer();
        emit faceHovered(-1, 0, 0);
    }
}

int OCCTViewer::pickFaceByRay(int mouseX, int mouseY) {
    if (view_.IsNull() || fullShape_.IsNull()) return -1;

    // 将 2D 鼠标位置转换为 3D 射线
    Standard_Real X, Y, Z, DX, DY, DZ;
    view_->ConvertWithProj(mouseX, mouseY, X, Y, Z, DX, DY, DZ);

    gp_Pnt origin(X, Y, Z);
    gp_Dir direction(DX, DY, DZ);

    // 创建射线几何线（在所有面之外创建一次）
    Handle(Geom_Line) geomLine = new Geom_Line(origin, direction);

    // 遍历所有面，找出与射线最近的面
    TopExp_Explorer exp(fullShape_, TopAbs_FACE);
    int idx = 0;
    int closestFace = -1;
    Standard_Real minDist = std::numeric_limits<double>::max();

    for (; exp.More(); exp.Next(), ++idx) {
        const TopoDS_Face& face = TopoDS::Face(exp.Current());

        // 获取面的几何曲面
        Handle(Geom_Surface) surface = BRep_Tool::Surface(face);
        if (surface.IsNull()) continue;

        // 计算射线与面的交点
        GeomAPI_IntCS intCS(geomLine, surface);
        if (intCS.IsDone() && intCS.NbPoints() > 0) {
            for (int pi = 1; pi <= intCS.NbPoints(); ++pi) {
                gp_Pnt hitPoint = intCS.Point(pi);
                // 检查交点是否在面前方（沿射线方向为正）
                gp_Vec diff(origin, hitPoint);
                Standard_Real projDist = diff.Dot(gp_Vec(direction));
                if (projDist <= 0) continue;

                // 检查交点是否在面的参数域内（带容差）
                Standard_Real u, v, w;
                intCS.Parameters(pi, u, v, w);
                Standard_Real uMin, uMax, vMin, vMax;
                BRepTools::UVBounds(face, uMin, uMax, vMin, vMax);
                const Standard_Real uvTol = 1e-4;
                if (u < uMin - uvTol || u > uMax + uvTol ||
                    v < vMin - uvTol || v > vMax + uvTol) continue;

                if (projDist < minDist) {
                    minDist = projDist;
                    closestFace = idx;
                }
            }
        }
    }

    return closestFace;
}

void OCCTViewer::handleSelection() {
    if (context_.IsNull() || view_.IsNull()) return;

    context_->MoveTo(lastMousePos_.x(), lastMousePos_.y(), view_, Standard_False);
    context_->Select(Standard_True);

    int pickedFace = -1;

    // 方法一：遍历检测到的实体
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
        pickedFace = pickFaceByRay(lastMousePos_.x(), lastMousePos_.y());
    }

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
