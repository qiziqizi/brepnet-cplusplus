#include "ColorMapper.h"
#include <Quantity_Color.hxx>

ColorMapper::ColorMapper() {
    initializeClassNames();
    initializeColors();
    defaultColor_ = Quantity_Color(0.7, 0.7, 0.7, Quantity_TOC_RGB);  // 灰色
}

void ColorMapper::initializeClassNames() {
    // 27个类别名称（根据BRepNet论文）
    classNames_ = {
        "plane",           // 0
        "cylinder",        // 1
        "cone",            // 2
        "sphere",          // 3
        "torus",           // 4
        "revolution",      // 5
        "extrusion",       // 6
        "other",           // 7
        "blend",           // 8
        "chamfer",         // 9
        "fillet",          // 10
        "hole",            // 11
        "pocket",          // 12
        "protrusion",      // 13
        "depression",      // 14
        "imprint",         // 15
        "island",          // 16
        "through_hole",    // 17
        "blind_hole",      // 18
        "rectangular_pocket", // 19
        "rectangular_protrusion", // 20
        "circular_pocket", // 21
        "circular_protrusion", // 22
        "thread",          // 23
        "draft",           // 24
        "split",           // 25
        "unknown"          // 26
    };
}

void ColorMapper::initializeColors() {
    // 定义27种视觉上易区分的颜色（RGB值）
    // 色相跨度大，便于区分
    std::vector<std::tuple<float, float, float>> colors = {
        {1.0f, 0.0f, 0.0f},    // 0: 红色
        {0.0f, 1.0f, 0.0f},    // 1: 绿色
        {0.0f, 0.0f, 1.0f},    // 2: 蓝色
        {1.0f, 1.0f, 0.0f},    // 3: 黄色
        {1.0f, 0.0f, 1.0f},    // 4: 品红
        {0.0f, 1.0f, 1.0f},    // 5: 青色
        {1.0f, 0.5f, 0.0f},    // 6: 橙色
        {0.5f, 0.0f, 1.0f},    // 7: 紫色
        {0.0f, 0.5f, 0.0f},    // 8: 深绿
        {0.5f, 0.5f, 0.0f},    // 9: 橄榄绿
        {0.0f, 0.5f, 0.5f},    // 10: 深青
        {0.5f, 0.0f, 0.0f},    // 11: 深红
        {0.5f, 0.25f, 0.0f},   // 12: 棕色
        {0.75f, 0.75f, 0.0f},  // 13: 亮黄
        {0.0f, 0.75f, 0.75f},  // 14: 亮青
        {0.75f, 0.0f, 0.75f},  // 15: 亮紫
        {0.25f, 0.5f, 0.0f},   // 16: 黄绿
        {0.0f, 0.25f, 0.5f},   // 17: 蓝绿
        {0.5f, 0.0f, 0.25f},   // 18: 酒红
        {0.75f, 0.5f, 0.25f},  // 19: 土黄
        {0.25f, 0.75f, 0.5f},  // 20: 春绿
        {0.5f, 0.25f, 0.75f},  // 21: 紫罗兰
        {0.9f, 0.6f, 0.0f},    // 22: 金色
        {0.0f, 0.6f, 0.9f},    // 23: 天蓝
        {0.9f, 0.0f, 0.6f},    // 24: 玫瑰红
        {0.3f, 0.3f, 0.3f},    // 25: 深灰
        {0.6f, 0.4f, 0.2f}     // 26: 棕褐
    };

    for (int i = 0; i < 27; ++i) {
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
