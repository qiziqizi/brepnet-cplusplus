# BRepNet C++ 使用指南

> 本文档整合了批量推理工具（CLI）和可视化工具（Qt GUI）的完整使用说明。

---

## 1. 概述

本项目提供两个工具：

| 工具 | 入口 | 用途 |
|------|------|------|
| **批量推理工具** | `main_export_features.cpp` | 命令行批量处理 STEP 文件，输出 logits 和分类结果 |
| **可视化工具** | `visualizer/` (Qt + OpenCascade) | 加载单个 STEP 文件，3D 显示预测着色，对比真实标签 |

两个工具共享相同的推理引擎和面序号体系，预测结果完全一致。

---

## 2. 批量推理工具

### 2.1 运行模式总览

| 模式 | 命令参数 | 终端输出 | 输出文件 | 适用场景 |
|------|---------|---------|---------|---------|
| **静默批量** | _(留空)_ | 仅文件名+结果 | logits + results | 28000文件批量跑 |
| **调试单文件** | `--debug --target 文件名` | 该文件全量调试 | logits + results + 中间层 | 定点调试 |
| **调试全部** | `--debug` | 所有文件全量调试 | logits + results + 中间层 | 全量检查 |
| **仅导出中间文件** | `--export` | 仅文件名+结果 | logits + results + 中间层 | 需要中间数据但不刷屏 |
| **导出指定文件** | `--export --target 文件名` | 仅文件名+结果 | logits + results + 中间层 | 只导出某文件中间数据 |

### 2.2 Visual Studio 配置

1. **解决方案资源管理器** → 右键项目 → **属性**
2. **配置属性** → **调试** → **命令参数**
3. 填入对应参数字符串，直接 F5 运行

> 修改命令参数**不需要重新编译**。

### 2.3 各场景详细操作

#### 场景 A：批量跑（日常生产）

**命令参数**：_(留空)_

```
=== BRepNet Inference Tool ===
[Model] Weights loaded successfully!
[Files] Found 28548 STEP files

[1/28548] step0001 -> [✓] F:12 E:18
[2/28548] step0002 -> [✓] F:8 E:12
...
All 28548 files completed!
```

- 不输出 `[DEBUG]`、`[Layer 0]` 等调试信息
- 不创建 `cpp_feature_maps/`、`cpp_uv_grids/` 中间文件
- 创建 `cpp_logits/*.logits` 和 `cpp_results/*.results`

#### 场景 B：调试单个问题文件

**命令参数**：`--debug --target step3268`

- 只对 `step3268` 输出全量调试信息和中间文件
- 其他文件保持静默

#### 场景 C：调试所有文件

**命令参数**：`--debug`

> ⚠️ 会产生大量终端输出和磁盘文件，仅建议少量文件（< 100）时使用。

#### 场景 D：只导出中间文件

**命令参数**：`--export`

终端简洁输出，但为每个文件创建中间文件。

#### 场景 E：只导出某个文件的中间数据

**命令参数**：`--export --target step3268`

### 2.4 参数组合速查表

| 命令参数 | 终端调试 | 始终生成 | 条件生成 | 诊断文件 |
|---------|---------|---------|---------|---------|
| _(空)_ | ❌ | ✅ logits + results | ❌ | ❌ |
| `--debug` | ✅ 全部 | ✅ logits + results | ✅ 全部中间层 | ✅ |
| `--debug --target X` | ✅ 仅X | ✅ logits + results | ✅ 仅X中间层 | ✅ 仅X |
| `--export` | ❌ | ✅ logits + results | ✅ 全部中间层 | ✅ |
| `--export --target X` | ❌ | ✅ logits + results | ✅ 仅X中间层 | ✅ 仅X |

- **始终生成** = `cpp_logits/`、`cpp_results/`
- **条件生成** = `cpp_feature_maps/`、`cpp_uv_grids/`
- **诊断文件** = `arc_length_diagnosis.txt`、`coedge_grid_diagnosis.txt` 等

### 2.5 target 参数说明

`--target` 后跟 STEP 文件的 stem 名称（不含路径和 `.step` 扩展名），大小写敏感。

```
# 文件路径: inference_data/step_files/20240116_231044_0_result.step
--debug --target 20240116_231044_0_result

# 多个 target 用逗号分隔（无空格）
--debug --target step3268,step5001,step9999
```

---

## 3. 可视化工具

### 3.1 功能概述

基于 Qt + OpenCascade 的桌面应用，支持：

1. 加载 STEP 文件并 3D 显示
2. 运行 BRepNet 推理，按预测类别给每个面着色（27 种颜色）
3. 导入真实标签对比，计算准确率并高亮错误面
4. 点击任意面查看序号、预测类别、真实类别

### 3.2 界面布局

窗口分左右两栏（约 70%/30%）：

| 区域 | 内容 |
|------|------|
| **左侧** | 3D 视图（OCCT 渲染） |
| **右侧** | 控制面板 |

控制面板从上到下：

| 区域 | 控件 | 说明 |
|------|------|------|
| 文件操作 | "加载STEP文件" | 打开 `.step`/`.stp` |
| 预测操作 | "运行预测" / "导出结果" | 执行推理 / 导出文本 |
| 对比验证 | "导入真实标签" / 准确率 | 加载 labels 并对比 |
| 模型信息 | 文件名/面数/状态/选中面 | 状态一览 |
| 预测结果 | 文本区 | 类别分布 + 错误面列表 |
| 颜色图例 | 可滚动列表 | 27 类别颜色方块 |

### 3.3 鼠标操作

| 操作 | 功能 |
|------|------|
| 左键单击 | 选中面（显示序号和类别） |
| 右键拖动 | 旋转视图 |
| 中键拖动 / 左右键同时拖动 | 平移 |
| 滚轮 | 缩放 |
| 双击 | 重置视图（Fit All） |

### 3.4 基本工作流程

```
① 加载STEP文件 → ② 运行预测 → ③ 导入真实标签 → ④ 查看对比结果
```

1. **启动程序**：自动搜索并加载 `inference_data/state_dict.npz`
2. **加载STEP文件**：所有面以灰色显示
3. **运行预测**：每个面按类别着色，右侧显示统计
4. **导入标签对比**（可选）：显示准确率，错误面用红色粗边框高亮
5. **点击查看**：
   ```
   选中面: #5 | 预测: through_hole(1) | 真实: through_hole(1) ✓
   选中面: #12 | 预测: chamfer(0) | 真实: slanted_through_step(10) ✗
   ```

### 3.5 导出预测结果

点击"导出结果"保存为 `.txt`：

```
BRepNet 预测结果
================
文件: D:/path/to/part001.step
面数: 6

面索引  类别ID  类别名称
------  ------  --------
0       24      plane
1       25      cylinder
...

类别分布统计
============
plane: 2
cylinder: 1
```

### 3.6 颜色对照表

| ID | 类别名 | 颜色 | RGB |
|----|--------|------|-----|
| 0 | chamfer | 暗青 | (0.184, 0.310, 0.310) |
| 1 | through_hole | 鞍棕 | (0.545, 0.271, 0.075) |
| 2 | triangular_passage | 橄榄 | (0.502, 0.502, 0.0) |
| 3 | rectangular_passage | 暗紫 | (0.282, 0.239, 0.545) |
| 4 | 6sides_passage | 绿 | (0.0, 0.502, 0.0) |
| 5 | triangular_through_slot | 玫瑰棕 | (0.737, 0.561, 0.561) |
| 6 | rectangular_through_slot | 黄绿 | (0.604, 0.804, 0.196) |
| 7 | circular_through_slot | 深蓝 | (0.0, 0.0, 0.545) |
| 8 | rectangular_through_step | 暗海绿 | (0.561, 0.737, 0.561) |
| 9 | 2sides_through_step | 紫 | (0.502, 0.0, 0.502) |
| 10 | slanted_through_step | 栗红 | (0.690, 0.188, 0.376) |
| 11 | Oring | 红 | (1.0, 0.0, 0.0) |
| 12 | blind_hole | 暗橙 | (1.0, 0.549, 0.0) |
| 13 | triangular_pocket | 黄 | (1.0, 1.0, 0.0) |
| 14 | rectangular_pocket | 草绿 | (0.498, 1.0, 0.0) |
| 15 | 6sides_pocket | 春绿 | (0.0, 0.980, 0.604) |
| 16 | circular_end_pocket | 绯红 | (0.863, 0.078, 0.235) |
| 17 | rectangular_blind_slot | 青 | (0.0, 1.0, 1.0) |
| 18 | v_circular_end_blind_slot | 蓝 | (0.0, 0.0, 1.0) |
| 19 | h_circular_end_blind_slot | 品红 | (1.0, 0.0, 1.0) |
| 20 | triangular_blind_step | 卡其 | (0.941, 0.902, 0.549) |
| 21 | circular_blind_step | 天蓝 | (0.529, 0.808, 0.922) |
| 22 | rectangular_blind_step | 矢车菊蓝 | (0.392, 0.584, 0.929) |
| 23 | round | 深粉 | (1.0, 0.078, 0.576) |
| 24 | plane | 中紫 | (0.482, 0.408, 0.933) |
| 25 | cylinder | 浅鲑 | (1.0, 0.627, 0.478) |
| 26 | cone | 紫罗兰 | (0.933, 0.510, 0.933) |

未运行预测时显示为灰色 `(0.7, 0.7, 0.7)`。

---

## 4. 面序号与标签系统

> 本节内容适用于批量推理和可视化工具，两者共享相同的面序号体系。

### 4.1 面序号规则

面序号（#0, #1, #2, ...）由 OpenCascade 的 `TopExp_Explorer` 遍历顺序确定：

1. `STEPControl_Reader` 读取 STEP 文件得到 `TopoDS_Shape`
2. `TopExp_Explorer(shape, TopAbs_FACE)` 深度优先遍历所有面
3. 第 1 个发现的面 → **#0**，第 2 个 → **#1**，以此类推
4. **同一 STEP 文件，任何平台、任何次数运行，顺序完全一致**

面序号**不等于** CAD 软件中的特征编号或建模顺序。修改零件后重新导出 STEP，原有面序号可能变化。

### 4.2 确认面序号的方法

**方法 1（推荐）：可视化工具**

1. 加载目标 STEP 文件
2. 左键点击面，信息栏显示 `选中面: #N`
3. 逐面记录序号和类别，写入 labels 文件

**方法 2：Python + OCC 脚本**

```python
from OCP.STEPControl import STEPControl_Reader
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE
from OCP.TopoDS import topods
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere, GeomAbs_Torus

reader = STEPControl_Reader()
reader.ReadFile("your_file.step")
reader.TransferRoots()
shape = reader.OneShape()

surface_type_names = {
    GeomAbs_Plane: "Plane", GeomAbs_Cylinder: "Cylinder",
    GeomAbs_Cone: "Cone", GeomAbs_Sphere: "Sphere", GeomAbs_Torus: "Torus",
}

exp = TopExp_Explorer(shape, TopAbs_FACE)
face_id = 0
while exp.More():
    face = topods.Face(exp.Current())
    adaptor = BRepAdaptor_Surface(face)
    stype = adaptor.GetType()
    print(f"Face #{face_id}: {surface_type_names.get(stype, 'Other')}")
    face_id += 1
    exp.Next()
print(f"\nTotal: {face_id} faces")
```

**方法 3：先预测再修正**

运行预测后在可视化工具中逐面检查，对错误面记录序号和正确类别。

### 4.3 三模块面顺序一致性

| 模块 | 遍历代码 | 容器 |
|------|---------|------|
| 可视化显示 | `StepLoader.cpp:65` | `vector<TopoDS_Face>` |
| 可视化预测 | `BRepPipeline.h:173` | `TopTools_IndexedMapOfShape` |
| 批量推理 | `BRepPipeline.h:173` | `TopTools_IndexedMapOfShape` |

三者都用 `TopExp_Explorer(shape, TopAbs_FACE)` 遍历，正常 STEP 文件结果完全一致。批量推理的 `.logits` 文件第 N 行（0-indexed）对应面 #N。

### 4.4 Labels 文件格式

**格式要求**：

| 项目 | 要求 |
|------|------|
| 扩展名 | `.labels` 或 `.txt` |
| 每行内容 | 整数 **0 ~ 26** |
| 注释 | `#` 开头的行跳过 |
| 空行 | 自动跳过 |
| 行数 | **必须等于 STEP 文件的面数** |
| 对应关系 | 第 K 个有效行对应面 #K |

**示例**：

```
# part001.step 的真实标签（6个面）
24
25
24
23
1
1
```

**27 个类别 ID 对照表**：

| ID | 类别名 | ID | 类别名 |
|----|--------|----|--------|
| 0 | chamfer | 14 | rectangular_pocket |
| 1 | through_hole | 15 | 6sides_pocket |
| 2 | triangular_passage | 16 | circular_end_pocket |
| 3 | rectangular_passage | 17 | rectangular_blind_slot |
| 4 | 6sides_passage | 18 | v_circular_end_blind_slot |
| 5 | triangular_through_slot | 19 | h_circular_end_blind_slot |
| 6 | rectangular_through_slot | 20 | triangular_blind_step |
| 7 | circular_through_slot | 21 | circular_blind_step |
| 8 | rectangular_through_step | 22 | rectangular_blind_step |
| 9 | 2sides_through_step | 23 | round |
| 10 | slanted_through_step | 24 | plane |
| 11 | Oring | 25 | cylinder |
| 12 | blind_hole | 26 | cone |
| 13 | triangular_pocket | | |

### 4.5 面序号常见问题

**Q: 不同 CAD 软件导出的 STEP 文件，面顺序一样吗？**
A: 不一定。必须基于实际使用的 STEP 文件确认。

**Q: 修改 STEP 文件后，原来的 labels 还能用吗？**
A: 不能。任何修改都可能改变面的数量或顺序，需重新构建 labels。

**Q: 面序号会因操作系统或编译器不同而变化吗？**
A: 不会。遍历顺序由 STEP 文件内容唯一决定。

**Q: 为什么有些面看起来一样但序号不同？**
A: 几何相似但拓扑上独立的面（如多个孔的内壁）有不同序号。

---

## 5. 输出文件说明

### 5.1 始终生成（所有模式）

| 目录 | 内容 |
|------|------|
| `cpp_logits/*.logits` | 原始 logits（softmax 前），每行 27 个浮点数，行号 = Face ID |
| `cpp_results/*.results` | 分类结果：预测类别、置信度、Top 3 |

### 5.2 调试/导出模式才生成

| 目录/文件 | 内容 |
|----------|------|
| `cpp_feature_maps/` | 各层中间特征 |
| `cpp_uv_grids/` | UV Grid 原始数据 |
| `arc_length_diagnosis.txt` | 弧长诊断（项目根目录） |
| `coedge_grid_diagnosis.txt` | Coedge Grid 诊断（项目根目录） |

### 5.3 可视化工具导出

点击"导出结果"生成 `.txt`，包含每面的类别 ID、类别名称和统计分布。

---

## 6. 典型工作流程

```
第1步：批量运行
  命令参数：（空）→ 跑完所有文件

第2步：发现问题
  → 用对比脚本发现 step3268 误差很大

第3步：定点调试
  命令参数：--debug --target step3268
  → F5 运行，查看终端和 cpp_feature_maps/ 中间文件
  → 与 Python 端逐层对比

第4步：可视化验证（可选）
  → 启动可视化工具加载同一 STEP 文件
  → 运行预测，逐面点击查看分类结果

第5步：修复后验证
  命令参数：--debug --target step3268
  → 确认误差消除

第6步：恢复批量
  命令参数：（清空）→ 批量重跑
```

> 整个过程中无需修改代码、注释/取消注释、重新编译。只需切换 VS 命令参数。

---

## 7. 注意事项

### 批量推理相关

- **修改命令参数不需要重新编译**，直接 F5
- **Debug/Release 配置分别设置**，切换配置时检查命令参数
- **`--target` 大小写敏感**，精确匹配 stem 名称
- **静默模式零性能损失**，调试代码被条件跳过

### 可视化工具相关

- **推理路径一致性**：可视化工具调用 `BRepNet::forward()`，批量推理手动逐层执行，两者逻辑已同步
- **面数校验**：运行预测和导入标签时均校验面数，不匹配会拒绝操作
- **模型权重路径**：自动搜索 `{exe目录}/inference_data/state_dict.npz` 和 `{exe目录}/../../../inference_data/state_dict.npz`，找不到则"运行预测"按钮禁用

---

## 8. 故障排查

| 问题 | 排查方向 |
|------|---------|
| 命令参数不生效 | 检查 Debug/Release 配置是否正确，检查属性页顶部的下拉框 |
| 调试输出没出现 | 确认 `--target` 文件名拼写正确（大小写敏感），确认文件存在于 `step_files/` |
| 中间文件在哪里 | `cpp_feature_maps/`、`cpp_uv_grids/`、`cpp_logits/`、项目根目录诊断文件 |
| 可视化工具预测按钮禁用 | 确保 `inference_data/state_dict.npz` 在可执行文件的相对路径下 |
| 标签导入失败 | 检查标签行数是否等于面数，每行是否为 0-26 的整数 |
