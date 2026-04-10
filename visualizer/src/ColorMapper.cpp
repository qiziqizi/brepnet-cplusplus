#include "ColorMapper.h"
#include <Quantity_Color.hxx>

ColorMapper::ColorMapper() {
    initializeClassNames();
    initializeColors();
    defaultColor_ = Quantity_Color(0.7, 0.7, 0.7, Quantity_TOC_RGB);  // 灰色
}

void ColorMapper::initializeClassNames() {
    // 27个类别名称（与 segment_names.json / MFCAD 数据集一致）
    classNames_ = {
        "chamfer",                    // 0
        "through_hole",               // 1
        "triangular_passage",         // 2
        "rectangular_passage",        // 3
        "6sides_passage",             // 4
        "triangular_through_slot",    // 5
        "rectangular_through_slot",   // 6
        "circular_through_slot",      // 7
        "rectangular_through_step",   // 8
        "2sides_through_step",        // 9
        "slanted_through_step",       // 10
        "Oring",                      // 11
        "blind_hole",                 // 12
        "triangular_pocket",          // 13
        "rectangular_pocket",         // 14
        "6sides_pocket",              // 15
        "circular_end_pocket",        // 16
        "rectangular_blind_slot",     // 17
        "v_circular_end_blind_slot",  // 18
        "h_circular_end_blind_slot",  // 19
        "triangular_blind_step",      // 20
        "circular_blind_step",        // 21
        "rectangular_blind_step",     // 22
        "round",                      // 23
        "plane",                      // 24
        "cylinder",                   // 25
        "cone"                        // 26
    };
}

void ColorMapper::initializeColors() {
    // 定义27种视觉上易区分的颜色（RGB值）
    // 色相跨度大，便于区分
    std::vector<std::tuple<float, float, float>> colors = {
        {0.184f, 0.310f, 0.310f},  // 0:  chamfer - 暗青
        {0.545f, 0.271f, 0.075f},  // 1:  through_hole - 鞍棕
        {0.502f, 0.502f, 0.0f},    // 2:  triangular_passage - 橄榄
        {0.282f, 0.239f, 0.545f},  // 3:  rectangular_passage - 暗紫
        {0.0f, 0.502f, 0.0f},      // 4:  6sides_passage - 绿
        {0.737f, 0.561f, 0.561f},  // 5:  triangular_through_slot - 玫瑰棕
        {0.604f, 0.804f, 0.196f},  // 6:  rectangular_through_slot - 黄绿
        {0.0f, 0.0f, 0.545f},      // 7:  circular_through_slot - 深蓝
        {0.561f, 0.737f, 0.561f},  // 8:  rectangular_through_step - 暗海绿
        {0.502f, 0.0f, 0.502f},    // 9:  2sides_through_step - 紫
        {0.690f, 0.188f, 0.376f},  // 10: slanted_through_step - 栗红
        {1.0f, 0.0f, 0.0f},        // 11: Oring - 红
        {1.0f, 0.549f, 0.0f},      // 12: blind_hole - 暗橙
        {1.0f, 1.0f, 0.0f},        // 13: triangular_pocket - 黄
        {0.498f, 1.0f, 0.0f},      // 14: rectangular_pocket - 草绿
        {0.0f, 0.980f, 0.604f},    // 15: 6sides_pocket - 春绿
        {0.863f, 0.078f, 0.235f},  // 16: circular_end_pocket - 绯红
        {0.0f, 1.0f, 1.0f},        // 17: rectangular_blind_slot - 青
        {0.0f, 0.0f, 1.0f},        // 18: v_circular_end_blind_slot - 蓝
        {1.0f, 0.0f, 1.0f},        // 19: h_circular_end_blind_slot - 品红
        {0.941f, 0.902f, 0.549f},  // 20: triangular_blind_step - 卡其
        {0.529f, 0.808f, 0.922f},  // 21: circular_blind_step - 天蓝
        {0.392f, 0.584f, 0.929f},  // 22: rectangular_blind_step - 矢车菊蓝
        {1.0f, 0.078f, 0.576f},    // 23: round - 深粉
        {0.482f, 0.408f, 0.933f},  // 24: plane - 中紫
        {1.0f, 0.627f, 0.478f},    // 25: cylinder - 浅鲑
        {0.933f, 0.510f, 0.933f}   // 26: cone - 紫罗兰
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
