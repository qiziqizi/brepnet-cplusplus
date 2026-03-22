#ifndef STEPLOADER_H
#define STEPLOADER_H

#include <string>
#include <vector>
#include <TopoDS_Shape.hxx>
#include <TopoDS_Face.hxx>

/**
 * STEP文件加载器
 * 功能：
 * 1. 加载STEP文件
 * 2. 提取所有拓扑面
 * 3. 确保面的遍历顺序一致性（与预测模块保持一致）
 */
class StepLoader {
public:
    StepLoader();
    ~StepLoader();

    // 加载STEP文件
    bool loadFile(const std::string& filePath);

    // 获取加载的形状
    TopoDS_Shape getShape() const { return shape_; }

    // 获取所有面的有序列表
    const std::vector<TopoDS_Face>& getFaces() const { return faces_; }

    // 获取面的数量
    int getNumFaces() const { return static_cast<int>(faces_.size()); }

    // 获取当前文件名
    std::string getFileName() const { return fileName_; }

    // 清除当前数据
    void clear();

private:
    void extractFaces();

    TopoDS_Shape shape_;
    std::vector<TopoDS_Face> faces_;
    std::string fileName_;
};

#endif // STEPLOADER_H
