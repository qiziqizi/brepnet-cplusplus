#ifndef OCCTVIEWER_H
#define OCCTVIEWER_H

#include <QWidget>
#include <map>
#include <vector>

#include <AIS_InteractiveContext.hxx>
#include <V3d_View.hxx>
#include <V3d_Viewer.hxx>
#include <AIS_Shape.hxx>
#include <TopoDS_Face.hxx>
#include <Quantity_Color.hxx>

#ifdef _WIN32
#include <WNT_Window.hxx>
#endif

class OCCTViewer : public QWidget {
    Q_OBJECT

public:
    explicit OCCTViewer(QWidget* parent = nullptr);
    ~OCCTViewer();

    void displayFaces(const std::vector<TopoDS_Face>& faces, const TopoDS_Shape& fullShape);
    void updateFaceColor(int faceIndex, const Quantity_Color& color);
    void updateAllFaceColors(const std::vector<Quantity_Color>& colors);
    void updateSingleFaceColor(int faceIndex, const Quantity_Color& color);
    void resetAllFaceColors();
    void highlightErrorFaces(const std::vector<int>& errorIndices, bool highlight = true);
    void clearErrorHighlights();
    void clearAll();
    void fitAll();
    int getNumFaces() const { return static_cast<int>(faceObjects_.size()); }
    int getSelectedFaceIndex() const { return selectedFaceIndex_; }
    const Handle(V3d_View)& getView() const { return view_; }

signals:
    void faceSelected(int faceIndex);
    void faceHovered(int faceIndex, int mouseX, int mouseY);  // faceIndex=-1 = 离开

protected:
    void paintEvent(QPaintEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseDoubleClickEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;
    QPaintEngine* paintEngine() const override { return nullptr; }

private:
    void initOCCT();
    void handleSelection();
    void checkHoveredFace();

    Handle(V3d_Viewer) viewer_;
    Handle(V3d_View) view_;
    Handle(AIS_InteractiveContext) context_;
#ifdef _WIN32
    Handle(WNT_Window) hWnd_;
#endif

    std::map<int, Handle(AIS_Shape)> faceObjects_;
    std::map<int, Quantity_Color> faceColors_;     // 每个面当前颜色
    Handle(AIS_Shape) wireframeShape_;              // 整个模型的线框叠加层

    enum MouseMode { None, Rotate, Pan };
    MouseMode currentMode_;
    QPoint lastMousePos_;
    int selectedFaceIndex_;
    int previousSelectedFaceIndex_;                 // 上一次选中的面索引
    int hoveredFaceIndex_;                          // -1 = 未悬停
};

#endif
