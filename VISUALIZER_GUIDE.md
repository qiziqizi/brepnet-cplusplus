# BRepNet 可视化工具使用说明

---

## 1. 功能概述

可视化工具是一个基于 Qt + OpenCascade 的桌面应用，用于：

1. 加载 STEP 文件并 3D 显示所有面
2. 运行 BRepNet 推理，按预测类别给每个面着色
3. 导入人工标注的真实标签（labels 文件），与预测结果对比，计算准确率并高亮错误面
4. 点击任意面查看其序号、预测类别、真实类别

---

## 2. 界面布局

启动后窗口分为左右两栏（约 70%/30%）：

| 区域 | 内容 |
|------|------|
| **左侧** | 3D 视图（OCCT 渲染） |
| **右侧** | 控制面板，从上到下分为 5 个区域 |

### 控制面板区域

| 区域 | 按钮/控件 | 说明 |
|------|----------|------|
| 文件操作 | "加载STEP文件" | 打开 `.step`/`.stp` 文件 |
| 预测操作 | "运行预测" / "导出结果" | 执行推理 / 导出预测到文本文件 |
| 对比验证 | "导入真实标签" / 准确率显示 | 加载 labels 文件并自动对比 |
| 模型信息 | 文件名 / 面数 / 状态 / 选中面 | 当前状态一览 |
| 预测结果 | 文本区 | 类别分布统计 + 错误面列表 |

### 鼠标操作

| 操作 | 功能 |
|------|------|
| 左键单击 | 选中面（显示序号和类别信息） |
| 右键拖动 | 旋转视图 |
| 中键拖动 / 左右键同时拖动 | 平移 |
| 滚轮 | 缩放 |
| 双击 | 重置视图（Fit All） |

---

## 3. 基本工作流程

```
① 加载STEP文件 → ② 运行预测 → ③ 导入真实标签 → ④ 查看对比结果
```

### 步骤详解

1. **启动程序**：自动在以下路径查找并加载模型权重文件：
   - `{exe目录}/inference_data/state_dict.npz`
   - `{exe目录}/../../../inference_data/state_dict.npz`
   - 加载成功后状态栏显示"模型已加载"

2. **加载STEP文件**：点击"加载STEP文件"，选择 `.step` 或 `.stp` 文件。加载后所有面以灰色显示在 3D 视图中。

3. **运行预测**：点击"运行预测"，程序对每个面运行 BRepNet 推理。完成后：
   - 每个面按预测类别着色（27 种颜色）
   - 右侧显示类别分布统计

4. **导入标签对比**（可选）：点击"导入真实标签"，选择 `.labels` 或 `.txt` 文件。程序自动：
   - 逐面对比预测 vs 真实标签
   - 显示准确率
   - 错误面用红色粗边框高亮
   - 错误面列表显示在统计区

5. **点击查看面信息**：左键点击任意面，信息栏显示：
   ```
   选中面: #5 | 预测: cylinder(1) | 真实: cylinder(1) ✓
   选中面: #12 | 预测: plane(0) | 真实: fillet(10) ✗
   ```

---

## 4. 面的序号规则（重要 — 构建 Labels 必读）

### 4.1 面序号是怎么产生的

本项目中面的序号（#0, #1, #2, ...）由 **OpenCascade 的 `TopExp_Explorer` 遍历顺序** 唯一确定。这不是随机的，也不是按 CAD 建模顺序排列的，而是由 STEP 文件中存储的 B-Rep 拓扑结构决定的**确定性深度优先遍历**。

具体流程：

1. `STEPControl_Reader` 读取 STEP 文件，得到一个复合形体 `TopoDS_Shape`
2. `TopExp_Explorer(shape, TopAbs_FACE)` 对这个形体做深度优先遍历，依次发现所有拓扑面
3. 第 1 个被发现的面 → **#0**，第 2 个 → **#1**，以此类推
4. **同一个 STEP 文件，无论运行多少次、在哪台机器上，遍历顺序完全一致**

```cpp
// 本项目三个模块（可视化显示、可视化预测、批量推理）都使用这个遍历：
for (TopExp_Explorer exp(shape, TopAbs_FACE); exp.More(); exp.Next()) {
    TopoDS_Face face = TopoDS::Face(exp.Current());
    // 第 0 次循环 → 面 #0，第 1 次循环 → 面 #1，...
}
```

### 4.2 面序号与 CAD 特征的关系

面的序号**不等于** CAD 软件中的特征编号或建模顺序。举例说明：

- 一个简单的长方体有 6 个面，`TopExp_Explorer` 可能按以下顺序遍历：
  - #0 = 顶面、#1 = 底面、#2 = 前面、#3 = 后面、#4 = 左面、#5 = 右面
- 但也可能是另一种顺序 — 取决于 STEP 文件内部的拓扑存储结构
- 如果在 CAD 软件中对零件做了修改（如增加倒角、打孔），重新导出 STEP 后，**原有面的序号可能会发生变化**

### 4.3 如何确认每个面的序号（构建 Labels 的方法）

#### 方法 1（推荐）：使用本项目的可视化工具

这是最直观的方式：

1. 启动可视化工具，加载目标 STEP 文件
2. 所有面以灰色显示在 3D 视图中
3. **左键点击任意面**，右侧信息栏显示：`选中面: #N`
4. 旋转、缩放模型，逐个点击所有面，记录每个面的序号和对应的类别
5. 按序号从 #0 开始，依次写入 labels 文件

具体操作示例：

```
点击顶面    → 信息栏显示 "选中面: #0"  → 判断为 plane   → labels 第 1 行写 0
点击侧面    → 信息栏显示 "选中面: #1"  → 判断为 cylinder → labels 第 2 行写 1
点击圆角面  → 信息栏显示 "选中面: #2"  → 判断为 fillet   → labels 第 3 行写 10
...
```

#### 方法 2：使用 Python + OCC 脚本

如果不方便使用可视化工具，可以编写 Python 脚本。安装 `pythonocc-core` 后：

```python
from OCP.STEPControl import STEPControl_Reader
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE
from OCP.TopoDS import topods
from OCP.BRep import BRep_Tool
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
    type_name = surface_type_names.get(stype, "Other")
    print(f"Face #{face_id}: {type_name}")
    face_id += 1
    exp.Next()

print(f"\nTotal: {face_id} faces")
```

输出示例：
```
Face #0: Plane
Face #1: Cylinder
Face #2: Plane
Face #3: Cylinder
Face #4: Plane
Face #5: Plane
Total: 6 faces
```

> **注意**：Python 的 `TopExp_Explorer` 与 C++ 完全一致（底层是同一个 OpenCascade 库），遍历顺序保证相同。

#### 方法 3：先运行预测，再修正

1. 运行批量推理或可视化工具的"运行预测"，得到每个面的预测类别
2. 在可视化工具中，点击每个面查看预测结果是否正确
3. 对于预测错误的面，记录其序号和正确类别
4. 构建完整的 labels 文件

### 4.4 三个模块的面顺序一致性保证

| 模块 | 遍历代码位置 | 容器类型 | 面序号来源 |
|------|-------------|----------|-----------|
| 可视化显示 | `StepLoader.cpp:65` | `vector<TopoDS_Face>` | vector 下标 |
| 可视化预测 | `BRepPipeline.h:173` | `TopTools_IndexedMapOfShape` | Map 索引 |
| 批量推理 | `BRepPipeline.h:173`（同上） | `TopTools_IndexedMapOfShape` | Map 索引 |

三者都使用 `TopExp_Explorer(shape, TopAbs_FACE)` 遍历。StepLoader 用 `vector`（不去重），BRepPipeline 用 `IndexedMapOfShape`（自动去重）。对于正常的 STEP 文件（没有重复面），两者结果完全一致。

**批量推理输出的 `.logits` 文件中，第 N 行（0-indexed）对应面 #N，与可视化工具中的面序号一致。**

### 4.5 常见问题

**Q: 同一个零件，在不同 CAD 软件中导出的 STEP 文件，面顺序会一样吗？**
A: **不一定**。不同 CAD 软件（SolidWorks、CATIA、NX 等）导出 STEP 时，内部拓扑结构的存储顺序可能不同。必须基于**实际要使用的那个 STEP 文件**来确认面顺序。

**Q: 修改了 STEP 文件后重新导出，原来的 labels 还能用吗？**
A: **不能直接用**。任何对零件的修改（加特征、删特征、修改参数）都可能导致面的数量或顺序变化。必须重新确认面顺序并重新构建 labels。

**Q: 面的序号会因为操作系统或编译器不同而变化吗？**
A: **不会**。面的遍历顺序由 STEP 文件内容唯一决定，与操作系统、编译器无关。只要使用 OpenCascade 的 `TopExp_Explorer`，任何平台上的结果都一致。

**Q: 为什么有些面看起来一样但序号不同？**
A: 一个复杂零件可能有多个几何相似但拓扑上独立的面（如多个相同的孔的内壁）。它们在拓扑结构中是不同的面，因此有不同的序号。

---

## 5. Labels 文件格式（手动构建标签）

### 格式规范

```
# 这是注释行，会被跳过
# 文件名: part001.step
# 每行一个整数，表示对应面的真实类别 ID（0-26）
# 第 1 个非空非注释行 → 面 #0 的标签
# 第 2 个非空非注释行 → 面 #1 的标签
# ...依此类推
0
1
8
8
0
10
13
```

### 详细要求

| 项目 | 要求 |
|------|------|
| 文件扩展名 | `.labels` 或 `.txt`（文件选择对话框默认筛选这两种） |
| 编码 | 纯文本（UTF-8 或 ANSI 均可） |
| 每行内容 | 一个整数，范围 **0 ~ 26**（共 27 个类别） |
| 注释 | 以 `#` 开头的行会被跳过 |
| 空行 | 自动跳过 |
| 标签数量 | **必须恰好等于 STEP 文件的面数**，否则拒绝加载并提示"标签数量(M)与面数量(N)不匹配" |
| 对应关系 | 第 K 个有效行（跳过空行和注释后从 0 开始计数）对应面 #K |
| 错误处理 | 如果任何一行不是合法整数或超出 0-26 范围，**整个文件被拒绝**（返回空） |

### 27 个类别 ID 对照表

| ID | 类别名 | ID | 类别名 |
|----|--------|----|--------|
| 0 | plane | 14 | depression |
| 1 | cylinder | 15 | imprint |
| 2 | cone | 16 | island |
| 3 | sphere | 17 | through_hole |
| 4 | torus | 18 | blind_hole |
| 5 | revolution | 19 | rectangular_pocket |
| 6 | extrusion | 20 | rectangular_protrusion |
| 7 | other | 21 | circular_pocket |
| 8 | blend | 22 | circular_protrusion |
| 9 | chamfer | 23 | thread |
| 10 | fillet | 24 | draft |
| 11 | hole | 25 | split |
| 12 | pocket | 26 | unknown |
| 13 | protrusion | | |

### 构建 Labels 文件的步骤

1. **在可视化工具中加载 STEP 文件**，记下面数（显示在"模型信息"区域）
2. **逐面确认类别**：点击每个面查看其序号 `#N`，根据 CAD 设计意图判断该面属于哪个类别
3. **按序号顺序写入 labels 文件**：从面 #0 开始，每行写一个类别 ID
4. **验证**：labels 文件的有效行数必须等于面数

### 示例

假设一个 STEP 文件有 6 个面：

```
# part001.step 的真实标签
# 面数: 6
#
# 面#0: 顶面，平面
0
# 面#1: 侧面，圆柱面
1
# 面#2: 底面，平面
0
# 面#3: 圆角过渡面
10
# 面#4: 通孔内壁
17
# 面#5: 通孔内壁
17
```

可以省略注释，最简形式：
```
0
1
0
10
17
17
```

---

## 6. 导出预测结果

点击"导出结果"后，保存为 `.txt` 文件，格式如下：

```
BRepNet 预测结果
================

文件: D:/path/to/part001.step
面数: 6

面索引	类别ID	类别名称
------	------	--------
0	0	plane
1	1	cylinder
2	0	plane
3	10	fillet
4	17	through_hole
5	17	through_hole


类别分布统计
============

plane: 2
cylinder: 1
fillet: 1
through_hole: 2
```

---

## 7. 颜色对照表

预测完成后每个面按类别着色。27 种颜色如下：

| ID | 类别名 | 颜色 | RGB |
|----|--------|------|-----|
| 0 | plane | 红色 | (1.0, 0.0, 0.0) |
| 1 | cylinder | 绿色 | (0.0, 1.0, 0.0) |
| 2 | cone | 蓝色 | (0.0, 0.0, 1.0) |
| 3 | sphere | 黄色 | (1.0, 1.0, 0.0) |
| 4 | torus | 品红 | (1.0, 0.0, 1.0) |
| 5 | revolution | 青色 | (0.0, 1.0, 1.0) |
| 6 | extrusion | 橙色 | (1.0, 0.5, 0.0) |
| 7 | other | 紫色 | (0.5, 0.0, 1.0) |
| 8 | blend | 深绿 | (0.0, 0.5, 0.0) |
| 9 | chamfer | 橄榄绿 | (0.5, 0.5, 0.0) |
| 10 | fillet | 深青 | (0.0, 0.5, 0.5) |
| 11 | hole | 深红 | (0.5, 0.0, 0.0) |
| 12 | pocket | 棕色 | (0.5, 0.25, 0.0) |
| 13 | protrusion | 亮黄 | (0.75, 0.75, 0.0) |
| 14 | depression | 亮青 | (0.0, 0.75, 0.75) |
| 15 | imprint | 亮紫 | (0.75, 0.0, 0.75) |
| 16 | island | 黄绿 | (0.25, 0.5, 0.0) |
| 17 | through_hole | 蓝绿 | (0.0, 0.25, 0.5) |
| 18 | blind_hole | 酒红 | (0.5, 0.0, 0.25) |
| 19 | rectangular_pocket | 土黄 | (0.75, 0.5, 0.25) |
| 20 | rectangular_protrusion | 春绿 | (0.25, 0.75, 0.5) |
| 21 | circular_pocket | 紫罗兰 | (0.5, 0.25, 0.75) |
| 22 | circular_protrusion | 金色 | (0.9, 0.6, 0.0) |
| 23 | thread | 天蓝 | (0.0, 0.6, 0.9) |
| 24 | draft | 玫瑰红 | (0.9, 0.0, 0.6) |
| 25 | split | 深灰 | (0.3, 0.3, 0.3) |
| 26 | unknown | 棕褐 | (0.6, 0.4, 0.2) |

未分类面（未运行预测时）显示为灰色 `(0.7, 0.7, 0.7)`。

---

## 8. 注意事项

### 8.1 推理路径

可视化工具内部调用 `BRepNet::forward()`（`BRepNet.h` 中定义），批量推理工具 `main_export_features.cpp` 使用手动逐层执行的推理路径。两者的推理逻辑已同步一致（MaxPooling 初始化、大小面阈值、小面 ReLU），预测结果应完全相同。

### 8.2 面数一致性校验

程序会在以下两处校验面数：
- **运行预测后**：检查预测结果数量是否等于 StepLoader 的面数
- **导入标签时**：检查标签数量是否等于 StepLoader 的面数

如果不匹配，操作会被拒绝并弹窗提示。

### 8.3 模型权重路径

程序启动时自动搜索 `state_dict.npz`。如果找不到，"运行预测"按钮将保持禁用状态。确保 `inference_data/state_dict.npz` 位于可执行文件的相对路径下。
