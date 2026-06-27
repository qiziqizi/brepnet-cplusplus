#include "OCCTViewer.h"
#include <QMouseEvent>
#include <QWheelEvent>
#include <iostream>

#include <Aspect_DisplayConnection.hxx>
#include <OpenGl_GraphicDriver.hxx>
#include <AIS_Shape.hxx>
#include <Prs3d_Drawer.hxx>
#include <Graphic3d_MaterialAspect.hxx>
#include <Prs3d_LineAspect.hxx>
#include <Prs3d_IsoAspect.hxx>

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
        for (auto& pair : faceObjects_) {
            context_->Remove(pair.second, Standard_False);
        }
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

void OCCTViewer::displayFaces(const std::vector<TopoDS_Face>& faces) {
    if (context_.IsNull()) return;
    clearAll();

    Quantity_Color grayColor(0.7, 0.7, 0.7, Quantity_TOC_RGB);
    for (size_t i = 0; i < faces.size(); ++i) {
        Handle(AIS_Shape) aisShape = new AIS_Shape(faces[i]);
        aisShape->SetColor(Quantity_NOC_GRAY70);
        aisShape->SetMaterial(Graphic3d_NOM_PLASTIC);
        aisShape->SetDisplayMode(AIS_Shaded);
        aisShape->Attributes()->SetFaceBoundaryDraw(true);
        aisShape->Attributes()->SetIsoOnTriangulation(Standard_False);
        aisShape->Attributes()->UIsoAspect()->SetNumber(0);
        aisShape->Attributes()->VIsoAspect()->SetNumber(0);
        context_->Display(aisShape, Standard_False);
        faceObjects_[static_cast<int>(i)] = aisShape;
        faceColors_[static_cast<int>(i)] = grayColor;
    }

    context_->UpdateCurrentViewer();
    fitAll();
}

void OCCTViewer::updateFaceColor(int faceIndex, const Quantity_Color& color) {
    auto it = faceObjects_.find(faceIndex);
    if (it != faceObjects_.end()) {
        faceColors_[faceIndex] = color;
        context_->SetColor(it->second, color, Standard_True);
    }
}

void OCCTViewer::updateAllFaceColors(const std::vector<Quantity_Color>& colors) {
    if (colors.size() != faceObjects_.size()) return;

    for (size_t i = 0; i < colors.size(); ++i) {
        int idx = static_cast<int>(i);
        auto it = faceObjects_.find(idx);
        if (it != faceObjects_.end()) {
            faceColors_[idx] = colors[i];
            context_->SetColor(it->second, colors[i], Standard_False);
        }
    }

    // 如果当前有选中的面，重新应用透明度（保持其半透明效果）
    if (previousSelectedFaceIndex_ >= 0) {
        auto selIt = faceObjects_.find(previousSelectedFaceIndex_);
        if (selIt != faceObjects_.end()) {
            context_->SetTransparency(selIt->second, 0.3, Standard_False);
        }
    }

    context_->UpdateCurrentViewer();
}

void OCCTViewer::updateSingleFaceColor(int faceIndex, const Quantity_Color& color) {
    auto it = faceObjects_.find(faceIndex);
    if (it != faceObjects_.end()) {
        faceColors_[faceIndex] = color;
        context_->SetColor(it->second, color, Standard_False);
        // 此面恰好被选中时，需重新应用透明度
        if (faceIndex == previousSelectedFaceIndex_) {
            context_->SetTransparency(it->second, 0.3, Standard_True);
        } else {
            context_->UpdateCurrentViewer();
        }
    }
}

void OCCTViewer::resetAllFaceColors() {
    if (context_.IsNull()) return;

    Quantity_Color grayColor(0.7, 0.7, 0.7, Quantity_TOC_RGB);
    for (auto& pair : faceObjects_) {
        faceColors_[pair.first] = grayColor;
        context_->SetColor(pair.second, grayColor, Standard_False);
        context_->UnsetTransparency(pair.second, Standard_False);
    }

    // 清除选中状态
    previousSelectedFaceIndex_ = -1;
    selectedFaceIndex_ = -1;
    context_->UpdateCurrentViewer();
}

void OCCTViewer::highlightErrorFaces(const std::vector<int>& errorIndices, bool highlight) {
    if (context_.IsNull()) return;

    Quantity_Color errorBoundaryColor(1.0, 0.0, 0.0, Quantity_TOC_RGB);  // Red

    for (int idx : errorIndices) {
        auto it = faceObjects_.find(idx);
        if (it != faceObjects_.end()) {
            Handle(Prs3d_Drawer) drawer = it->second->Attributes();
            if (highlight) {
                drawer->SetFaceBoundaryDraw(true);
                Handle(Prs3d_LineAspect) lineAspect = new Prs3d_LineAspect(
                    errorBoundaryColor, Aspect_TOL_SOLID, 3.0);
                drawer->SetFaceBoundaryAspect(lineAspect);
            } else {
                drawer->SetFaceBoundaryDraw(true);
                Handle(Prs3d_LineAspect) lineAspect = new Prs3d_LineAspect(
                    Quantity_NOC_BLACK, Aspect_TOL_SOLID, 1.0);
                drawer->SetFaceBoundaryAspect(lineAspect);
            }
            context_->Redisplay(it->second, Standard_False);
        }
    }
    context_->UpdateCurrentViewer();
}

void OCCTViewer::clearErrorHighlights() {
    if (context_.IsNull()) return;

    for (auto& pair : faceObjects_) {
        Handle(Prs3d_Drawer) drawer = pair.second->Attributes();
        drawer->SetFaceBoundaryDraw(true);
        Handle(Prs3d_LineAspect) lineAspect = new Prs3d_LineAspect(
            Quantity_NOC_BLACK, Aspect_TOL_SOLID, 1.0);
        drawer->SetFaceBoundaryAspect(lineAspect);
        context_->Redisplay(pair.second, Standard_False);
    }
    context_->UpdateCurrentViewer();
}

void OCCTViewer::clearAll() {
    if (context_.IsNull()) return;
    for (auto& pair : faceObjects_) {
        context_->Remove(pair.second, Standard_False);
    }
    faceObjects_.clear();
    faceColors_.clear();
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

    if (context_->HasDetected()) {
        Handle(AIS_InteractiveObject) obj = context_->DetectedInteractive();
        for (auto& pair : faceObjects_) {
            if (pair.second == obj) {
                if (pair.first != hoveredFaceIndex_) {
                    hoveredFaceIndex_ = pair.first;
                    emit faceHovered(pair.first, lastMousePos_.x(), lastMousePos_.y());
                }
                return;
            }
        }
    }

    // 没有检测到物体 → 清空悬停
    if (hoveredFaceIndex_ != -1) {
        hoveredFaceIndex_ = -1;
        emit faceHovered(-1, 0, 0);
    }
}

void OCCTViewer::handleSelection() {
    if (context_.IsNull() || view_.IsNull()) return;

    context_->MoveTo(lastMousePos_.x(), lastMousePos_.y(), view_, Standard_False);
    context_->Select(Standard_True);

    if (context_->HasDetected()) {
        Handle(AIS_InteractiveObject) obj = context_->DetectedInteractive();
        for (auto& pair : faceObjects_) {
            if (pair.second == obj) {
                int newIndex = pair.first;

                // 点击同一个面 → 不做任何变化（与 Python 行为一致）
                if (newIndex == previousSelectedFaceIndex_) {
                    emit faceSelected(newIndex);
                    return;
                }

                // 1. 恢复上一个选中面：移除透明度（颜色保持不变）
                if (previousSelectedFaceIndex_ >= 0) {
                    auto prevIt = faceObjects_.find(previousSelectedFaceIndex_);
                    if (prevIt != faceObjects_.end()) {
                        context_->UnsetTransparency(prevIt->second, Standard_False);
                    }
                }

                // 2. 设置新选中面：半透明（颜色保留）
                context_->SetTransparency(pair.second, 0.3, Standard_False);
                previousSelectedFaceIndex_ = newIndex;
                selectedFaceIndex_ = newIndex;

                context_->UpdateCurrentViewer();
                emit faceSelected(newIndex);
                return;
            }
        }
    }
}
