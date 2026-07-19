#ifndef OCCTVIEWER_H
#define OCCTVIEWER_H

#include <QWidget>
#include <map>
#include <vector>
#include <QSet>

#include <AIS_InteractiveContext.hxx>
#include <StdSelect_BRepOwner.hxx>
#include <StdSelect_ViewerSelector3d.hxx>
#include <V3d_View.hxx>
#include <V3d_Viewer.hxx>
#include <AIS_Shape.hxx>
#include <AIS_ColoredShape.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Shape.hxx>
#include <TopoDS.hxx>
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
    int getNumFaces() const { return static_cast<int>(faceColors_.size()); }
    int getSelectedFaceIndex() const { return selectedFaceIndex_; }
    const QSet<int>& getMultiSelectedFaces() const { return multiSelectedFaces_; }
    void clearMultiSelection() { multiSelectedFaces_.clear(); updateMultiSelectHighlight(); }
    const Handle(V3d_View)& getView() const { return view_; }

signals:
    void faceSelected(int faceIndex);
    void faceHovered(int faceIndex, int mouseX, int mouseY);  // faceIndex=-1 = 离开
    void faceModifyRequested(int faceIndex);                  // 右键单击请求修改
    void faceSelectionChanged(int faceIndex);                 // Ctrl+右键多选变化

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
    int findFaceIndex(const TopoDS_Shape& subShape) const;
    void applyFaceColor(int faceIndex, const Quantity_Color& color);
    void setCustomFaceColor(int faceIndex, const Quantity_Color& color); // 仅设颜色，不更新 faceColors_
    void setFaceTransparency(int faceIndex, Standard_Real transparency); // 按面设置透明度（悬停用）
    void restoreHoveredFace();
    void updateMultiSelectHighlight();                   // 更新多选面边线高亮
    int pickFaceByRay(int mouseX, int mouseY);                           // 几何光线求交
    int pickFaceAtPos(int devX, int devY);                               // 综合拾取（选择器+光线），不改选中状态
    TopoDS_Compound buildFaceEdgesCompound(int faceIndex) const;         // 提取面的所有边组成 compound

    Handle(V3d_Viewer) viewer_;
    Handle(V3d_View) view_;
    Handle(AIS_InteractiveContext) context_;
#ifdef _WIN32
    Handle(WNT_Window) hWnd_;
#endif

    Handle(AIS_ColoredShape) coloredShape_;          // 整模型，内部按面着色
    Handle(AIS_Shape) hoverEdges_;                   // 悬停面边线高亮叠加层
    Handle(AIS_Shape) multiSelectEdges_;             // 多选面边线高亮叠加层
    TopoDS_Shape fullShape_;                         // 原始完整形状，用于子面遍历
    std::map<int, Quantity_Color> faceColors_;       // 每个面当前颜色
    std::map<int, Quantity_Color> savedFaceColors_;  // 高亮前保存的面颜色

    enum MouseMode { None, Rotate, Pan };
    MouseMode currentMode_;
    QPoint lastMousePos_;
    int selectedFaceIndex_;
    int previousSelectedFaceIndex_;                 // 上一次选中的面索引
    int hoveredFaceIndex_;                          // -1 = 未悬停
    QSet<int> multiSelectedFaces_;                  // Ctrl+右键多选集合
};

#endif // OCCTVIEWER_H
