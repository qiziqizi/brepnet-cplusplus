#ifndef COLORMAPPER_H
#define COLORMAPPER_H

#include <map>
#include <string>
#include <vector>
#include <Quantity_Color.hxx>

/**
 * 颜色映射器：管理27个面类别的颜色映射
 * 提供类别ID到OCCT颜色的转换功能
 */
class ColorMapper {
public:
    ColorMapper();

    // 获取指定类别的颜色
    Quantity_Color getColor(int classId) const;

    // 获取默认颜色（未分类/灰色）
    Quantity_Color getDefaultColor() const;

    // 获取类别名称
    std::string getClassName(int classId) const;

    // 获取所有类别名称
    const std::vector<std::string>& getAllClassNames() const;

    // 生成 N 个视觉上互不相同的颜色（用于推理前的面区分显示）
    // 使用黄金角（Golden Angle ≈ 137.508°）色相分布 + 亮度/饱和度微变，
    // 确保任意连续面的色相都接近互补色，杜绝"相邻面颜色一致"的问题
    static std::vector<Quantity_Color> generateDistinctColors(int count);

    // 生成 N 个互不相同的颜色（用于 other 面）
    // 覆盖红色→青绿（0°~180°），每面色相不同
    static std::vector<Quantity_Color> generateOtherColors(int count);

    // 获取类别数量
    int getNumClasses() const { return 4; }

private:
    void initializeColors();
    void initializeClassNames();

    std::map<int, Quantity_Color> colorMap_;
    std::vector<std::string> classNames_;
    Quantity_Color defaultColor_;
};

#endif // COLORMAPPER_H
