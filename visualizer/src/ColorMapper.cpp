#include "ColorMapper.h"
#include <Quantity_Color.hxx>

ColorMapper::ColorMapper() {
    initializeClassNames();
    initializeColors();
    defaultColor_ = Quantity_Color(0.7, 0.7, 0.7, Quantity_TOC_RGB);  // 灰色
}

void ColorMapper::initializeClassNames() {
    // 4个类别名称
    classNames_ = {
        "chamfer",  // 0 倒角
        "round",    // 1 圆角
        "hole",     // 2 孔
        "other"     // 3 其他
    };
}

void ColorMapper::initializeColors() {
    // 定义4种视觉上差异化大的颜色（RGB值）
    std::vector<std::tuple<float, float, float>> colors = {
        {0.90f, 0.15f, 0.15f},  // 0: chamfer (倒角) - 红
        {0.15f, 0.45f, 0.95f},  // 1: round   (圆角) - 蓝
        {0.20f, 0.75f, 0.30f},  // 2: hole    (孔)   - 绿
        {0.75f, 0.75f, 0.75f}   // 3: other   (其他) - 浅灰
    };

    for (int i = 0; i < static_cast<int>(colors.size()); ++i) {
        auto [r, g, b] = colors[i];
        colorMap_[i] = Quantity_Color(r, g, b, Quantity_TOC_RGB);
    }
}

Quantity_Color ColorMapper::getColor(int classId) const {
    auto it = colorMap_.find(classId);
    if (it != colorMap_.end()) {
        return it->second;
    }
    return defaultColor_;
}

Quantity_Color ColorMapper::getDefaultColor() const {
    return defaultColor_;
}

std::string ColorMapper::getClassName(int classId) const {
    if (classId >= 0 && classId < static_cast<int>(classNames_.size())) {
        return classNames_[classId];
    }
    return "unknown";
}

const std::vector<std::string>& ColorMapper::getAllClassNames() const {
    return classNames_;
}
