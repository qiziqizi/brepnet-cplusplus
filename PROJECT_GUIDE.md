# BRepNet C++ 项目完整指南

> **目标读者**：对本项目零基础的 AI Agent 或新开发者。  

---

## 1. 项目定位与背景

### 1.1 一句话总结

本项目是 **BRepNet 神经网络的 C++ 纯推理引擎**，读取 CAD STEP 文件，提取 B-Rep（边界表示）拓扑与几何特征，运行预训练的 BRepNet + UVNet 模型，输出每个面（Face）的 27 类加工特征（Manufacturing Feature）分类结果（logits）。

### 1.2 与 Python 端的关系

| 项目 | 路径（参考） | 功能 |
|------|-------------|------|
| **Python BRepNet** | `D:\BRepNet_MFTRe` | 训练 + 推理，使用 PyTorch + occwl 库 |
| **本项目（C++）** | `d:\brepnet-cplusplus` | **仅推理**，不训练。从 Python 端导出的 `.npz` 权重文件加载参数 |

关键约束：C++ 端的输出 logits 必须与 Python 端在数值上尽可能一致（目标：绝对误差 < 0.01）。python项目是基准文件，本项目是模仿python项目生成的。

### 1.3 核心依赖

| 依赖 | 用途 |
|------|------|
| **OpenCascade (OCCT)** | 读取 STEP 文件、解析 B-Rep 拓扑、计算几何属性（法线/切线/曲率等） |
| **cnpy** | 加载 `.npz` 格式的模型权重文件 |
| **BRepTorch**（自实现） | 替代 LibTorch 的轻量级张量库（详见 §4.1） |
| **Qt**（仅 visualizer） | 可视化工具的 GUI 框架 |

> **重要**：本项目 **没有使用 LibTorch / PyTorch C++**。所有张量运算（matmul、conv2d、batch_norm、softmax 等）都由 `BRepTorch.h` 手动实现。

---

## 2. 项目结构总览

```
d:\brepnet-cplusplus\
│
├── main_export_features.cpp   ← 主入口：批量推理 + 中间层导出
│
├── BRepPipeline.h             ← 核心：STEP 读取 → 拓扑构建 → Grid 生成 → LCS 变换
├── BRepNet.h                  ← 网络定义：MLP 层 + forward 逻辑 + 数据结构
├── BRepNetAdapter.h           ← 适配器：Pipeline 数据 → BRepNet 数据格式
├── UVNet.h                    ← UVNet 编码器：Surface(Conv2d) + Curve(Conv1d)
├── BRepTorch.h                ← 自实现张量库：Tensor + 所有算子
├── BRepUtils.h / .cpp         ← 几何工具函数（法线投影/面积/缩放等）
│
├── DebugConfig.h              ← 全局调试开关（已弃用，被 DebugControl.h 取代）
├── DebugControl.h             ← 运行时调试控制（命令行参数驱动，无需重编译）
├── FeatureMapExporter.h       ← 中间层特征导出工具
├── FeatureMapDebugger.h       ← 中间层特征调试（张量统计/导出）
├── OutputLogger.h             ← 终端输出同时写入日志文件
├── EdgeInputExporter.h        ← Edge 网格调试导出（已弃用）
├── TopologyExporter.h         ← 拓扑信息导出（已弃用）
│
├── inference_data/            ← 运行时数据目录
│   ├── state_dict.npz         ← Python 导出的模型权重
│   └── step_files/            ← 待推理的 STEP 文件
│
├── cpp_logits/                ← 输出：原始 logits（softmax 前，始终生成）
├── cpp_results/               ← 输出：分类预测结果（每个 Face 的类别、置信度、Top 3，始终生成）
├── cpp_uv_grids/              ← 输出：UV Grid 原始数据（调试用）
├── cpp_feature_maps/          ← 输出：每层中间特征（调试用）
│
├── visualizer/                ← Qt 可视化工具（独立子项目）
│   └── src/
│       ├── main.cpp           ← Qt 应用入口
│       ├── MainWindow.h/.cpp  ← 主窗口
│       ├── OCCTViewer.h/.cpp  ← OCCT 3D 视图
│       ├── StepLoader.h/.cpp  ← STEP 加载
│       ├── FaceClassifier.h/.cpp  ← 调用推理引擎分类
│       └── ColorMapper.h/.cpp ← 分类结果着色
│
└── BRepNetVisualizer/         ← Visual Studio 解决方案（visualizer 的构建项目）
```

---

## 3. 端到端数据流（最重要）

理解本项目的关键是理解 **一个 STEP 文件从输入到 logits 输出的完整数据流**。

### 3.1 总流程图

```
STEP 文件
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  BRepPipeline::process()                            │
│                                                     │
│  ① STEP 读取 → TopoDS_Shape                        │
│  ② build_topology() → CoedgeInfo[] 拓扑图           │
│  ③ extract_features() → Xf, Xe, Xc（占位特征）     │
│  ④ generate_tensors() → Kf, Ke, Kc, Cf, Ce, Csf    │
│  ⑤ generate_local_grids():                          │
│     a. generate_global_face_grids()   → [F, 9, 10, 10]  │
│     b. compute_all_lcs_matrices()     → LCS_inv[C]  │
│     c. generate_coedge_local_grids()  → [C, 13, 10] │
│     d. generate_face_local_grids()    → [C, 2, 9, 10, 10] │
│     e. generate_edge_local_grids()    → [E, 13, 10] │
└─────────────────────────────────────────────────────┘
    │
    ▼  FaceGridsLocal, EdgeGridsLocal, CoedgeGridsLocal
┌─────────────────────────────────────────────────────┐
│  BRepNetAdapter::extract_coedges/faces/edges()      │
│                                                     │
│  ⑥ UVNet Surface Encoder:                           │
│     FaceGridsLocal → [C*2, 9, 10, 10]              │
│     → Conv2d(9→64) → Conv2d(64→128)                │
│     → GlobalAvgPool → FC(128→64)                    │
│     → parent_face_features[64], mate_face_features[64] │
│                                                     │
│  ⑦ UVNet Curve Encoder:                             │
│     EdgeGridsLocal → [E, 13, 10]                    │
│     → Conv1d(13→64) → Conv1d(64→128)               │
│     → GlobalAvgPool → FC(128→64)                    │
│     → edge_features[64]                             │
│                                                     │
│  → CoedgeData[], FaceData[], EdgeData[]             │
└─────────────────────────────────────────────────────┘
    │
    ▼  每个 Coedge 拥有: parent_face[64] + mate_face[64] + edge[64]
┌─────────────────────────────────────────────────────┐
│  BRepNet Forward (手动逐层执行)                       │
│                                                     │
│  ⑧ Layer 0 (一阶邻居):                              │
│     MLP(192→60→60) → edge_state[30] + face_state[30] │
│     → Face MaxPool → face.layer0_state[30]          │
│     → Edge MaxPool → edge.layer0_state[30]          │
│                                                     │
│  ⑨ Layer 1 (二阶邻居):                              │
│     输入: face.layer0[30] + mate_face.layer0[30]     │
│           + edge.layer0[30] = 90                     │
│     MLP(90→60→60) → edge_state[30] + face_state[30] │
│     → Face MaxPool → face.layer1_state[30]          │
│     → Edge MaxPool → edge.layer1_state[30]          │
│                                                     │
│  ⑩ Output Layer (三阶邻居):                         │
│     MLP(90→30→30) → face_state[30]（无 edge）       │
│     → Face MaxPool → face.output_state[30]          │
│                                                     │
│  ⑪ Classification:                                  │
│     Linear(30→27) → logits[F, 27]                   │
│     → softmax → probs[F, 27]                        │
└─────────────────────────────────────────────────────┘
    │
    ├─ cpp_logits/<filename>.logits    （原始 logits）
    │
    └─ cpp_results/<filename>.results  （预测结果统计）
```

### 3.2 关键维度速查

| 符号 | 含义 | 典型值 |
|------|------|--------|
| F | 唯一面数 (unique_faces) | 10~50 |
| E | 唯一边数 (unique_edges) | 20~100 |
| C | Coedge 数量 | 40~200 |
| 27 | 加工特征类别数 | 固定 |
| 64 | UVNet 输出特征维度 | 固定 |
| 30 | BRepNet 内部状态维度 | 固定 |
| 9 | Face Grid 通道数 (xyz + normal_xyz + mask + uv) | 固定 |
| 13 | Coedge Grid 通道数 (xyz + tangent + left_normal + right_normal + u_param) | 固定 |
| 10×10 | Face Grid 空间分辨率 | 固定 |
| 10 | Coedge/Edge Grid 沿弧长的采样点数 | 固定 |

---

## 4. 核心文件详解

### 4.1 `BRepTorch.h` — 自实现张量库（1485 行）

**这是本项目最独特的部分。** 完全替代了 LibTorch，实现了推理所需的全部张量操作。

#### 架构设计

```
Storage (shared_ptr)      ← 实际数据持有者
    ├── dataf_: vector<float>    ← float32 数据
    └── datal_: vector<int64_t>  ← int64 数据

Tensor                     ← 引用语义（shared_ptr 指向 Storage）
    ├── storage_: shared_ptr<Storage>
    ├── sizes_: vector<int64_t>  ← 形状
    ├── dtype_: DType            ← kFloat32 / kInt / kLong
    └── storage_offset_          ← 偏移（目前未使用）
```

#### 关键设计决策

- **引用语义**：`Tensor a = b` 共享底层数据（类似 PyTorch）。修改 `a` 会影响 `b`。
- **`view()` 是浅拷贝**：共享 `storage_`，只改 `sizes_`。
- **`clone()` 是深拷贝**：新建 `Storage`，复制全部数据。
- **必须 `clone()` 的场景**：
  - `from_blob()` 后立即 clone（因为原指针可能失效）
  - 进入 `forward()` 前 clone（forward 可能修改输入）
  - 循环中复用的 tensor 必须 clone

#### 实现的算子列表

| 类别 | 算子 |
|------|------|
| 创建 | `zeros`, `ones`, `eye`, `full`, `from_blob`, `tensor` |
| 形变 | `view`, `reshape`, `flatten`, `slice`, `clone`, `flip` |
| 数学 | `+`, `-`, `*`, `/`, `abs`, `sum`, `max`, `min`, `mean` |
| 线代 | `matmul`（Kahan 求和）, `det`（4×4）, `inverse`（4×4） |
| NN | `conv2d`, `conv1d`, `batch_norm`, `linear`, `leaky_relu`, `softmax`, `adaptive_avg_pool1d/2d`, `dropout` |
| 工具 | `cat`, `stack`, `where`, `cross`, `dot`, `norm`, `index` |

> **matmul 使用 Kahan 求和算法**（双精度补偿），以减少浮点累积误差。

### 4.2 `BRepPipeline.h` — 几何+拓扑处理核心（1513 行）

这是项目中最复杂的文件，负责从 STEP 几何体提取 BRepNet 所需的全部输入数据。

#### 主要流程（`process()` 函数）

```
process(step_file)
    ├── STEPControl_Reader 读取 STEP
    ├── TopExp_Explorer 遍历 Face/Edge → unique_faces, unique_edges
    ├── build_topology()      → coedges[] (id, face_idx, edge_idx, mate_idx, orientation, next/prev)
    ├── extract_features()    → Xf[F,7], Xe[E,10], Xc[C,1]（占位，UVNet 会替代）
    ├── generate_tensors()    → Kf, Ke, Kc（邻接关系）, Cf, Ce, Csf（Pooling 索引）
    └── generate_local_grids() → FaceGridsLocal, CoedgeGridsLocal, EdgeGridsLocal
```

#### 核心数据结构

```cpp
struct CoedgeInfo {
    int id;          // Coedge 全局编号
    int face_idx;    // 所属 Face 的编号（0-based）
    int edge_idx;    // 对应 Edge 的编号（0-based）
    int next_idx;    // 同 Wire 中的下一条 Coedge
    int prev_idx;    // 同 Wire 中的前一条 Coedge
    int mate_idx;    //    一 Edge 对面的 Coedge（共享边）
    bool orientation; // true=FORWARD, false=REVERSED（相对于 Edge 的方向）
};
```

#### 拓扑构建 (`build_topology()`)

遍历所有 Face → 遍历每个 Face 的 Wire → 遍历 Wire 中的 Edge，建立：
- `next`/`prev` 关系（环形链表，同一 Wire 内相邻）
- `mate` 关系（共享同一 Edge 的两个 Coedge 互为 mate）
- 对只有一个 Coedge 的 Edge（如球面的缝合线），`mate = self`

#### Grid 生成

##### Face Grid（`generate_global_face_grid`）
- 输入：一个 `TopoDS_Face`
- 输出：`[9, 10, 10]` 张量
- 9 个通道：`[x, y, z, nx, ny, nz, mask, u, v]`
- UV 采样策略：
  - REVERSED 面：U 从 max→min，V 从 min→max
  - FORWARD 面：U 从 min→max，V 从 min→max
- 法线计算：`GeomLProp_SLProps(surf, u, v, 1, 1e-9)` + REVERSED 时翻转
- Mask：使用 `BRepTopAdaptor_FClass2d` 判断点是否在面内

##### Coedge Grid（`generate_global_coedge_grid`）
- 输入：coedge 索引
- 输出：`[13, 10]` 张量
- 13 个通道：`[x, y, z, tx, ty, tz, nL_x, nL_y, nL_z, nR_x, nR_y, nR_z, u_param]`
- **弧长参数化**：100 个均匀参数采样点 → 弦长累积近似弧长 → 线性插值得到 10 个等弧长采样点
- 法线计算：pcurve → UV → `BRepLProp_SLProps(face, u, v, 1, 1e-6)` + REVERSED 翻转
- 退化边（曲线为 Null）：返回全零 Grid
- REVERSED coedge：最后 `flip(grid, {1})` 翻转采样点顺序

##### LCS（局部坐标系，`compute_coedge_lcs`）
- 用途：将全局 Grid 变换到每个 Coedge 的局部坐标系
- 构建方式：
  1. 在边的**弧长中点**计算 3D 点 `p` 和切线 `t`
  2. 在 `p` 投影到所属面上，计算面法线 `n`
  3. `w_vec = normalize(n)`（法线方向）
  4. `v_vec = normalize(t - dot(t, w) * w)`（切线在法线平面的投影）
  5. `u_vec = cross(v, w)`
  6. 组装 4×4 变换矩阵 `[u|v|w|p]`
  7. 计算逆矩阵 `LCS_inv` 用于变换
- **行列式检查**：若 `|det(M)| < 1e-6`，替换为单位矩阵

##### 变换 (`transform_grid_to_local`)
- 点通道（0-2）：仿射变换 `P' = M_inv * P`
- 向量通道（法线/切线）：仅旋转 `V' = R_inv * V`

##### Edge Grid（`generate_edge_local_grids`）
- 每条 Edge 从其两个 Coedge 中选择"左 coedge"：
  - 若 second_coedge 是 REVERSED → 选 first_coedge
  - 否则 → 选 second_coedge
- Edge Grid = 选中 Coedge 的 CoedgeGridsLocal

##### Face Local Grid（`generate_face_local_grids`）
- 每个 Coedge 生成一对 `[2, 9, 10, 10]`：
  - `[0]`：左面（parent face）的 FaceGrid，用该 coedge 的 LCS_inv 变换
  - `[1]`：右面（mate face）的 FaceGrid，用 **mate coedge** 的 LCS_inv 变换

#### Pooling 索引（`generate_tensors`）

| 张量 | 形状 | 说明 |
|------|------|------|
| Kf | [C, 2] | 每个 Coedge 对应的 Face 邻居（self, mate） |
| Ke | [C, 1] | 每个 Coedge 对应的 Edge |
| Kc | [C, 2] | 每个 Coedge 的 Coedge 邻居（self, mate） |
| Ce | [E, 2] | 每条 Edge 的两个 Coedge |
| Cf | [small_F, 30] | 小面（≤30 coedge）的 Pooling 索引 |
| Csf | [big_F 个 Tensor] | 大面（>30 coedge）的 Coedge 列表 |

> **Small Face / Big Face 分离**是 Python 端的设计：小面用固定大小矩阵 + zero padding，大面用变长列表。这会影响 Output Layer 的 MaxPooling 初始化值。

### 4.3 `BRepNet.h` — 网络结构定义（622 行）

#### 数据结构

```cpp
struct CoedgeData {
    int coedge_id, parent_face_id, mate_face_id, edge_id;
    vector<float> parent_face_features;  // 64 维（UVNet 输出）
    vector<float> mate_face_features;    // 64 维
    vector<float> edge_features;         // 64 维
    vector<float> layer0_face_state;     // 30 维（Layer 0 MLP 输出）
    vector<float> layer0_edge_state;     // 30 维
    vector<float> layer1_face_state;     // 30 维（Layer 1 MLP 输出）
    vector<float> layer1_edge_state;     // 30 维
    vector<float> output_face_state;     // 30 维（Output Layer MLP 输出）
};

struct FaceData {
    int face_id;
    vector<int> coedge_ids;
    vector<float> layer0_state;   // 30 维（MaxPool 结果）
    vector<float> layer1_state;   // 30 维
    vector<float> output_state;   // 30 维（最终 embedding）
};

struct EdgeData {
    int edge_id;
    vector<int> coedge_ids;
    vector<float> layer0_state;   // 30 维
    vector<float> layer1_state;   // 30 维
};
```

#### MLP 结构（`BRepNetMLPImpl`）

```
Linear(in, hidden) → Dropout(0.3) → ReLU
→ Linear(hidden, out) → [Dropout(0.3) → ReLU]  // final_layer=false 时有
```

> 推理模式下 Dropout 等效于 identity（直接传递）。

#### 网络结构（`BRepNetImpl`）

```
BRepNet:
    ├── surface_encoder: UVNetSurfaceEncoder (Conv2d)
    ├── curve_encoder:   UVNetCurveEncoder   (Conv1d)
    ├── layer0_mlp:      MLP(192→60→60, final=false)
    ├── layer1_mlp:      MLP(90→60→60, final=false)
    ├── output_mlp:      MLP(90→30→30, final=true)    ← 无尾部ReLU
    └── classification_layer: Linear(30→27)
```

#### Forward 逻辑（消息传递）

BRepNet 的核心思想是 **基于 B-Rep 拓扑的消息传递**：

1. **每个 Coedge 看到三个实体**：parent_face + mate_face + edge
2. 拼接三者特征 → MLP → 分别更新 face_state 和 edge_state
3. **MaxPooling 聚合**：每个 Face/Edge 从其所有 Coedge 的状态中取 max
4. 重复 3 层（一阶→二阶→三阶邻居），每层扩展感受野
5. 最终 Face embedding → Linear → 27 类 logits

#### Output Layer MaxPooling 的特殊处理

```
Small Face (≤30 coedge): 初始化为 0.0 → max(0, coedge_states) = ReLU 效果
Big Face (>30 coedge):   初始化为 -inf → 保留所有负值
```

这是 Python 端 Cf 矩阵 zero-padding 引起的隐式行为，C++ 端必须精确复现。

### 4.4 `UVNet.h` — UV-Net 编码器（435 行）

#### Surface Encoder

```
输入: [N, 9, 10, 10]
→ Conv2d(9→64, k=3, p=1) + BN + LeakyReLU(0.01)
→ Conv2d(64→128, k=3, p=1) + BN + LeakyReLU(0.01)
→ AdaptiveAvgPool2d(1,1) → Flatten
→ Linear(128→64, no bias) + BN + LeakyReLU(0.01)
输出: [N, 64]
```

#### Curve Encoder

```
输入: [N, 13, 10]
→ Conv1d(13→64, k=3, p=1) + BN + LeakyReLU(0.01)
→ Conv1d(64→128, k=3, p=1) + BN + LeakyReLU(0.01)
→ [Conv1d(128→128) 如果权重存在]
→ AdaptiveAvgPool1d(1) → Flatten
→ Linear(128→64, no bias) + BN + LeakyReLU(0.01)
输出: [N, 64]
```

#### 权重管理

- 使用 `std::map<string, Tensor>` 手动管理（非 PyTorch 的 `register_parameter`）
- BatchNorm 的 `running_mean`/`running_var` 存储在 `buffers` 中
- 权重名称格式：`surface_encoder.conv1.0.weight`、`curve_encoder.fc.1.bias` 等

### 4.5 `BRepNetAdapter.h` — 数据桥接（159 行）

负责将 `BRepPipeline` 的原始数据转换为 `BRepNet` 需要的 `CoedgeData`/`FaceData`/`EdgeData` 格式：

1. `extract_coedges()`：
   - 将 `FaceGridsLocal[C, 2, 9, 10, 10]` reshape 为 `[C*2, 9, 10, 10]`
   - 批量送入 Surface Encoder → `[C*2, 64]` → reshape `[C, 128]`
   - 前 64 维 = parent_face，后 64 维 = mate_face
   - Edge 特征：`EdgeGridsLocal[E, 13, 10]` → Curve Encoder → `[E, 64]`
   - 按 coedge 的 `edge_idx` 查找对应 Edge 的特征

2. `extract_faces()`：按 `face_idx` 收集每个面的 coedge 列表
3. `extract_edges()`：按 `edge_idx` 收集每条边的 coedge 列表

### 4.6 `main_export_features.cpp` — 主入口（1255 行）

#### 执行流程

```
main()
├── 创建 FeatureMapExporter("cpp_feature_maps")
├── 加载模型权重 (state_dict.npz)
│   ├── surface_encoder 权重 → surf_enc
│   ├── curve_encoder 权重 → curve_enc
│   └── BRepNet MLP + classification 权重 → model
├── 扫描 step_files/ 目录
└── 对每个 STEP 文件:
    └── run_inference_with_export()
        ├── BRepPipeline::process()         // 几何+拓扑
        ├── BRepNetAdapter::extract_*()     // UVNet 特征
        ├── 手动执行 Layer 0 → MaxPool      // 不调用 BRepNet::forward()
        ├── 手动执行 Layer 1 → MaxPool
        ├── 手动执行 Output Layer → MaxPool
        ├── classification_layer → logits
        ├── 导出中间层到 cpp_feature_maps/
        ├── 导出 logits 到 cpp_logits/
        └── 显式内存清理（swap释放）
```

> **注意**：`main_export_features.cpp` 中的 forward 是**手动逐层执行**的（没有调用 `BRepNet::forward()`），因为需要在每层之后导出中间结果。两者的推理逻辑已同步一致（MaxPooling 初始化、大小面阈值、小面 ReLU），预测结果相同。`BRepNet::forward()` 被可视化工具（`FaceClassifier`）调用。

#### 权重名称映射

Python 端：`layers.0.mlp.mlp.linear_0.weight` → C++ 端：`layer_0.mlp.mlp.linear_0.weight`

```
layers.0.mlp → layer_0.mlp
layers.1.mlp → layer_1.mlp
```

#### 输出顺序

logits 按**原始 Face ID 顺序**导出（与 Python 一致），而非推理时的内部排列顺序。推理内部使用 `face_permutation`（small faces 在前，big faces 在后）。

### 4.7 `cpp_results` 预测结果导出（新增）

在 `main_export_features.cpp` 的 logits 导出后，自动生成预测结果文件。

#### 文件位置
`cpp_results/<filename>.results`

#### 生成时机
- **所有推理模式都生成**（不受 `--debug` 等参数限制）
- 与 `cpp_logits/<filename>.logits` 同时生成
- 在 logits 导出之后，内存清理之前

#### 用途
- 快速诊断分类结果的准确性
- 识别置信度低的 Face，用于深入调试
- 与 Python 端的 `print_prediction_distribution()` 输出对应

#### 包含信息
- **预测类别**：argmax(softmax(logits))，即最可能的加工特征（0-26）
- **置信度**：max(softmax(logits))，即最大概率值，衡量预测可靠性
- **Top 3 类别及其概率**：前三高的类别和对应概率，便于了解备选预测

#### 实现位置
`main_export_features.cpp` 第 884-957 行

#### 关键特性
- ✅ 按原始 Face 顺序输出（与 cpp_logits 一致）
- ✅ Top 3 通过排序得到（最高概率优先）
- ✅ 置信度采用普通浮点格式（易读）
- ✅ Top 3 概率采用科学计数法（高精度）
- ✅ 显式内存清理（probs_original_order 等向量）
- ✅ 详见：`cpp_results_输出说明.md`

### 4.8 辅助文件

| 文件 | 行数 | 职责 |
|------|------|------|
| `DebugConfig.h` | 32 | `ENABLE_DEBUG_OUTPUT` 宏（0/1），控制 `DEBUG_COUT` |
| `OutputLogger.h` | 90 | `TeeBuf` 实现 stdout 同时写控制台+文件 |
| `FeatureMapExporter.h` | 190 | 将 Tensor/vector 按层名导出为 `.txt` 文件 |
| `BRepUtils.h/.cpp` | 57+impl | `GetParamStrict()`（UV 采样）、`GetNormalAtPoint()`（法线）等 |

---

## 5. 关键算法与实现细节

### 5.1 弧长参数化

Python（occwl）和 C++ 都使用相同算法：
1. 在边上均匀取 100 个参数点
2. 相邻点间的欧氏距离（弦长）累积近似弧长
3. 累积弧长归一化为 [0,1] 分数
4. 对目标分数（0/9, 1/9, ..., 9/9），在累积弧长表中线性插值得到参数值

> **已知差异**：`compute_arc_length_midpoint()` 中 C++ 根据参数范围条件性使用 100/200 个采样点（`param_span < 3.0 ? 100 : 200`），Python 固定使用 100 个。

### 5.2 Face Orientation 处理

OCCT 中 Face 有 FORWARD/REVERSED 两种 orientation：
- **UV 采样方向**：REVERSED 面的 U 从 max→min
- **法线翻转**：REVERSED 面法线取反（`if (face.Orientation() == TopAbs_REVERSED) normal.Reverse()`）
- 这两处都必须与 Python 端一致

### 5.3 Coedge Orientation 处理

- `orientation == true (FORWARD)`：切线方向与 Edge 参数化方向一致
- `orientation == false (REVERSED)`：
  - 切线取反（`tangent.Reverse()`）
  - Grid 数据翻转（`flip(grid, {1})`）

### 5.4 Small/Big Face 的 MaxPooling 差异

这是 Python 端的一个微妙行为：
- **Small faces**（≤30 coedge）：使用 Cf 张量 + zero padding → MaxPool 时隐式与 0 取 max → 等价于 ReLU
- **Big faces**（>30 coedge）：使用 Csf 变长列表 → 无 padding → 保留负值

C++ 端通过 `init_value = is_small_face ? 0.0f : -1e9f` 精确复现。

### 5.5 matmul 精度

`BRepTorch.h` 中 `matmul` 使用 **Kahan 求和算法**（双精度补偿），将每次累积的舍入误差记录并补偿。这对 BRepNet 的多层 MLP 累积误差控制很关键。

---

## 6. 构建与运行

### 6.1 前置条件

- Windows 系统
- Visual Studio（C++17）
- OpenCascade 已安装并配置 include/lib 路径
- cnpy 库已编译
- （可选）Qt 5/6（仅 visualizer 需要）

### 6.2 输入数据准备

```
inference_data/
├── state_dict.npz       ← Python 端导出：torch.save(model.state_dict(), ...)  → 转 npz
└── step_files/
    ├── part001.step
    ├── part002.step
    └── ...
```

### 6.3 运行

编译 `main_export_features.cpp` 后运行。程序会：
1. 加载 `state_dict.npz` 中的所有权重
2. 扫描 `step_files/` 下所有 `.step` / `.stp` 文件
3. 对每个文件执行推理
4. 跳过已处理的文件（检查 `cpp_logits/` 下是否已有对应 `.logits` 文件）
5. 输出到 `cpp_logits/`、`cpp_uv_grids/`、`cpp_feature_maps/`

### 6.4 输出格式

**`cpp_logits/<name>.logits`**：
- 每行 27 个浮点数（科学计数法，20 位精度），空格分隔
- 行号 = 原始 Face ID（0-based）
- 值 = 未归一化的分类 logits

**`cpp_results/<name>.results`**：
- 文件头：三行注释
  - `# filename: <file_stem>.step` — 原始文件名
  - `# topology: C coedges, F faces, E edges` — 拓扑信息
  - `# format: face_id predicted_class confidence top3_classes` — 列说明
- 数据行：每行对应一个 Face，包含 5 列
  - `face_id`：Face 原始编号（face_0, face_1, ...）
  - `predicted_class`：预测的加工特征类别（0-26）
  - `confidence`：最大概率值（6 位精度，普通浮点格式）
  - `top3_classes`：Top 3 类别及其概率（科学计数法）
    - 格式：`class_id:probability class_id:probability class_id:probability`
    - 例：`5:9.532009e-01 8:3.214500e-02 12:9.283000e-03`
- 示例：
  ```
  # filename: 20240116_231044_0_result.step
  # topology: 190 coedges, 28 faces, 55 edges
  # format: face_id predicted_class confidence top3_classes
  face_0  5  0.953201  5:9.532009e-01 8:3.214500e-02 12:9.283000e-03
  face_1  8  0.872401  8:8.724013e-01 5:8.723401e-02 3:1.230400e-02
  face_2  16  0.941523  16:9.415234e-01 5:3.215600e-02 8:1.523400e-02
  ```

---

## 7. 调试与验证体系

### 7.1 与 Python 端对比

本项目的核心质量指标是 **C++ logits 与 Python logits 的一致性**。对比方法：

1. Python 端同样导出 logits 到文件
2. 逐 Face 逐 Class 计算绝对误差
3. 目标：所有 Face 所有 Class 的误差 < 0.01

### 7.2 中间层导出

`FeatureMapExporter` 在每层之后导出中间结果：

```
cpp_feature_maps/
├── uvnet_surface/        ← [C*2, 64] UVNet 面特征
├── uvnet_curve/          ← [E, 64] UVNet 边特征
├── layer0_input_concat/  ← [C, 192] Layer 0 MLP 输入
├── layer0_mlp_output/    ← [C, 60] Layer 0 MLP 输出
├── layer0_face_pooling/  ← [F, 30] Layer 0 Face MaxPool 结果
├── layer0_edge_pooling/  ← [E, 30] Layer 0 Edge MaxPool 结果
├── layer1_input_concat/  ← [C, 90]
├── layer1_mlp_output/    ← [C, 60]
├── layer1_face_pooling/  ← [F, 30]
├── layer1_edge_pooling/  ← [E, 30]
├── output_layer_input_concat/   ← [C, 90]
├── output_layer_mlp_output/     ← [C, 30]
└── output_layer_face_embedding/ ← [F, 30] 最终 Face 嵌入
```

### 7.3 调试开关

- `DebugControl.h`：**运行时调试控制系统**（已取代 `DebugConfig.h`），通过命令行参数 `--debug-*` 控制各模块调试输出，无需重新编译。支持的开关包括 `--debug-topology`、`--debug-uvnet`、`--debug-pipeline` 等
- `DebugConfig.h`：旧版编译期开关（`ENABLE_DEBUG_OUTPUT` 宏），已弃用但仍保留
- `UVNet.h`：`UVNET_DEBUG_OUTPUT` 宏控制 UVNet 层调试
- `main_export_features.cpp` 中有大量注释掉的调试代码块，需要时可取消注释

### 7.4 诊断系统

`BRepPipeline.h` 底部有诊断函数：
- `diagnose_face_19_23_26_grids()`：针对特定 Face 的 Grid 数据诊断
- 弧长中点诊断：输出到 `arc_length_diagnosis.txt`
- Coedge Grid 诊断：输出到 `coedge_grid_diagnosis.txt`

---

## 8. 已知问题与注意事项

### 8.1 内存管理

- 批量处理 28,000+ 文件时，每个文件处理完后必须显式释放临时数据（使用 `vector<>().swap()` 技巧）
- 每 100 个文件检查一次进程内存使用量


---

## 9. Visualizer 子项目

位于 `visualizer/` 目录，是一个独立的 Qt 桌面应用：

| 文件 | 职责 |
|------|------|
| `MainWindow` | 主窗口，整合 3D 视图和控件 |
| `OCCTViewer` | 基于 OCCT 的 3D 渲染视图 |
| `StepLoader` | 加载 STEP 文件到 OCCT Shape |
| `FaceClassifier` | 调用推理引擎对每个 Face 分类 |
| `ColorMapper` | 根据分类结果给 Face 着色 |

构建需要额外配置 Qt 和 OCCT 的集成。

---

## 10. 快速导航索引

### "我想了解 X 的实现"

| 你想了解... | 去看... | 行号 |
|------------|---------|------|
| STEP 文件怎么读取 | `BRepPipeline.h` `process()` | 147-190 |
| 拓扑怎么构建 | `BRepPipeline.h` `build_topology()` | 288-360 |
| Face Grid 怎么采样 | `BRepPipeline.h` `generate_global_face_grid()` | 531-614 |
| Coedge Grid 怎么生成 | `BRepPipeline.h` `generate_global_coedge_grid()` | 616-837 |
| 弧长参数化怎么做 | `BRepPipeline.h` `generate_global_coedge_grid()` 内部 | 657-720 |
| LCS 怎么计算 | `BRepPipeline.h` `compute_coedge_lcs()` | 912-1020 |
| Grid 怎么变换到局部坐标 | `BRepPipeline.h` `transform_grid_to_local()` | 1027-1092 |
| Edge Grid 怎么选左 coedge | `BRepPipeline.h` `generate_edge_local_grids()` | 1219-1306 |
| UVNet 怎么编码面 | `UVNet.h` `UVNetSurfaceEncoderImpl::forward()` | 222-280 |
| UVNet 怎么编码边 | `UVNet.h` `UVNetCurveEncoderImpl::forward()` | 398-430 |
| 特征怎么从 Pipeline 转到 BRepNet | `BRepNetAdapter.h` `extract_coedges()` | 9-104 |
| BRepNet MLP 结构 | `BRepNet.h` `BRepNetMLPImpl` | 23-61 |
| BRepNet forward 逻辑 | `BRepNet.h` `BRepNetImpl::forward()` | 173-619 |
| 权重怎么加载 | `main_export_features.cpp` | 1104-1198 |
| logits 怎么导出 | `main_export_features.cpp` | 1001-1032 |
| 张量库 matmul 实现 | `BRepTorch.h` `matmul()` | 575-608 |
| 张量库 conv2d 实现 | `BRepTorch.h` nn namespace | ~1050+ |
| Small/Big Face 分离 | `BRepPipeline.h` `generate_tensors()` | 454-523 |
| Output Layer MaxPool 特殊逻辑 | `main_export_features.cpp` | 908-948 |

---

## 11. 术语表

| 术语 | 含义 |
|------|------|
| **B-Rep** | Boundary Representation，边界表示法，用面/边/顶点描述实体 |
| **STEP** | 标准 CAD 交换格式（ISO 10303） |
| **Face** | B-Rep 中的一个面（可以是平面、柱面、球面等） |
| **Edge** | B-Rep 中的一条边（两个面的交线） |
| **Coedge** | 半边（Half-edge），一条 Edge 在一个 Face 上的使用实例。每条 Edge 通常有 2 个 Coedge |
| **Wire** | 面的边界环，由有序的 Coedge 组成 |
| **Mate** | 共享同一 Edge 的另一个 Coedge |
| **Orientation** | Coedge 相对于 Edge 参数化方向的一致性（FORWARD/REVERSED） |
| **LCS** | Local Coordinate System，局部坐标系 |
| **UVNet** | UV-Net，用卷积网络将 UV 采样的曲面/曲线 Grid 编码为特征向量 |
| **BRepNet** | 基于 B-Rep 拓扑的消息传递网络 |
| **MaxPooling** | 从多个 Coedge 的状态中取逐维最大值 |
| **pcurve** | 参数曲线，Edge 在某个 Face 的 UV 空间中的 2D 表示 |
| **occwl** | Python 端使用的 OCCT 封装库（提供 EdgeDataExtractor 等高级 API） |
| **cnpy** | C++ 库，用于读写 NumPy 的 `.npy`/`.npz` 文件 |
| **OCCT** | OpenCascade Technology，开源 CAD 内核 |
