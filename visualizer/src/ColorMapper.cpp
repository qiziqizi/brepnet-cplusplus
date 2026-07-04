#include "ColorMapper.h"
#include <cmath>
#include <Quantity_Color.hxx>

ColorMapper::ColorMapper() {
    initializeClassNames();
    initializeColors();
    defaultColor_ = Quantity_Color(0.7, 0.7, 0.7, Quantity_TOC_RGB);  // 灰色
}

void ColorMapper::initializeClassNames() {
    // 4个类别名称
    classNames_ = {
        "chamfer",    // 0 倒角
        "round",      // 1 圆角
        "hole",       // 2 孔
        "other",      // 3 其他（暖色系随面变化）
    };
}

void ColorMapper::initializeColors() {
    // 4类别颜色：前3冷色系，other用彩色fallback（实际着色时每个面单独生成红→青绿色）
    std::vector<std::tuple<float, float, float>> colors = {
        {0.20f, 0.70f, 0.75f},  // 0: chamfer (倒角) - 青/cyan (冷)
        {0.15f, 0.35f, 0.85f},  // 1: round   (圆角) - 蓝 (冷)
        {0.45f, 0.15f, 0.65f},  // 2: hole    (孔)   - 紫 (冷)
        {0.85f, 0.50f, 0.15f},  // 3: other   (其他) - 暖橙 fallback
    };

    colorMap_.clear();
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
    // 限定在橙→绿（30°~130°），暖色占60%，与偏冷的四类标注色形成对比。
    // 降低饱和度、提高明度，呈现浅淡暖调，符合导师建议。
    const double goldenAngle = 137.508;
    const double hueStart = 30.0;
    const double hueRange = 100.0;
    for (int i = 0; i < count; ++i) {
        // 先在 [0,360) 上用黄金角生成，再线性映射到 [0,hueRange]。
        // 黄金角/360 = 1/φ² = 0.381966（最无理数），保证相邻索引色相差最大化且无周期性。
        // 若直接 fmod(goldenAngle, hueRange)，有效步长 137.508 mod 100 = 37.508 ≈ 3/8×100，
        // 退化为 ~8 周期的低质量分布。
        double hue = hueStart + std::fmod(i * goldenAngle, 360.0) * hueRange / 360.0;
        double lightness  = 0.82 + 0.06 * std::sin(i * 1.3);
        double saturation = 0.65 + 0.08 * std::cos(i * 0.7);
        colors.emplace_back(hue, lightness, saturation, Quantity_TOC_HLS);
    }
    return colors;
}

std::vector<Quantity_Color> ColorMapper::generateOtherColors(int count) {
    std::vector<Quantity_Color> colors;
    if (count <= 0) return colors;

    colors.reserve(count);
    // 红色→黄绿（0°~80°），覆盖暖色到黄绿，不进入纯绿色，每面不同
    const double goldenAngle = 137.508;
    const double hueStart = 0.0;
    const double hueRange = 80.0;
    for (int i = 0; i < count; ++i) {
        // 先在 [0,360) 上用黄金角生成，再线性映射到 [0,hueRange]。
        // 黄金角/360 = 1/φ² = 0.381966（最无理数），保证相邻索引色相差最大化且无周期性。
        // 若直接 fmod(goldenAngle, hueRange)，有效步长 137.508 mod 80 = 57.508 ≈ 5/7×80，
        // 退化为 ~7 周期的低质量分布（相邻色相差交替为 57.5°/22.5°）。
        // 修复后有效步长 = 137.508×80/360 = 30.557°，相邻色相差恒为 30.6° 或 49.4°。
        double hue = hueStart + std::fmod(i * goldenAngle, 360.0) * hueRange / 360.0;
        double lightness  = 0.78 + 0.06 * std::sin(i * 1.3);
        double saturation = 0.65 + 0.08 * std::cos(i * 0.7);
        colors.emplace_back(hue, lightness, saturation, Quantity_TOC_HLS);
    }
    return colors;
}
