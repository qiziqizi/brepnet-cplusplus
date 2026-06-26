#include "ColorMapper.h"
#include <cmath>
#include <Quantity_Color.hxx>

ColorMapper::ColorMapper() {
    initializeClassNames();
    initializeColors();
    defaultColor_ = Quantity_Color(0.7, 0.7, 0.7, Quantity_TOC_RGB);  // 灰色
}

void ColorMapper::initializeClassNames() {
    // 5个类别名称
    classNames_ = {
        "chamfer",    // 0 倒角
        "round",      // 1 圆角
        "hole",       // 2 孔
        "other",      // 3 其他
        "unlabeled"   // 4 未标注
    };
}

void ColorMapper::initializeColors() {
    // 定义5种视觉上差异化大的颜色（RGB值）
    std::vector<std::tuple<float, float, float>> colors = {
        {0.90f, 0.15f, 0.15f},  // 0: chamfer   (倒角) - 红
        {0.15f, 0.45f, 0.95f},  // 1: round     (圆角) - 蓝
        {0.45f, 0.12f, 0.60f},  // 2: hole      (孔)   - 深紫（与 other 的玫红拉开亮度）
        {0.85f, 0.25f, 0.55f},  // 3: other     (其他) - 玫红（与 unlabeled 色域拉开距离）
        {0.75f, 0.75f, 0.75f}   // 4: unlabeled (未标注) - 浅灰
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

std::vector<Quantity_Color> ColorMapper::generateDistinctColors(int count) {
    std::vector<Quantity_Color> colors;
    if (count <= 0) return colors;

    colors.reserve(count);
    // 限定在橙黄→纯青（30°~180°），避开四类色相（红0°/玫红340°/蓝220°/紫280°），
    // 150° 跨度覆盖橙/黄/绿/青，不含蓝色成分。
    const double goldenAngle = 137.508;
    const double hueStart = 30.0;
    const double hueRange = 150.0;
    for (int i = 0; i < count; ++i) {
        double hue = hueStart + std::fmod(i * goldenAngle, hueRange);
        double lightness  = 0.72 + 0.10 * std::sin(i * 1.3);
        double saturation = 0.82 + 0.12 * std::cos(i * 0.7);
        colors.emplace_back(hue, lightness, saturation, Quantity_TOC_HLS);
    }
    return colors;
}
