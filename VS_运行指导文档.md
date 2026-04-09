# Visual Studio 运行指导文档

## 概述

本文档说明在改造调试系统后，如何在 Visual Studio 中配置和运行 BRepNet C++ 推理程序。

改造后，程序通过**命令行参数**控制调试行为，你**永远不需要注释/取消注释代码**，也**不需要重新编译**来切换模式。

---

## 一、运行模式总览

| 模式 | 命令参数 | 终端输出 | 输出文件 | 适用场景 |
|------|---------|---------|---------|---------|
| **静默批量模式** | _(留空)_ | 仅文件名+结果 | ✅ logits + results | 28000文件批量跑 |
| **调试单文件模式** | `--debug --target 文件名` | 该文件全量调试 | ✅ logits + results + 中间层 | 发现问题文件后定点调试 |
| **调试全部模式** | `--debug` | 所有文件全量调试 | ✅ logits + results + 中间层 | 对比验证/全量检查 |
| **仅导出中间文件** | `--export` | 仅文件名+结果 | ✅ logits + results + 中间层 | 需要中间数据但不要终端刷屏 |
| **导出指定文件** | `--export --target 文件名` | 仅文件名+结果 | ✅ logits + results + 中间层 | 只导出某个文件的中间数据 |

---

## 二、Visual Studio 配置步骤

### 步骤 1：打开项目属性

1. 在**解决方案资源管理器**中，右键点击你的项目（不是解决方案）
2. 选择 **属性(Properties)**

### 步骤 2：找到命令参数设置

1. 左侧展开 **配置属性(Configuration Properties)**
2. 点击 **调试(Debugging)**
3. 找到右侧的 **命令参数(Command Arguments)** 一栏

### 步骤 3：填写命令参数

根据你的需求，在 **命令参数** 栏中填入对应的参数字符串：

---

## 三、各场景详细操作

### 场景 A：批量跑 28000 个文件（日常生产）

**命令参数**：_(留空，什么都不填)_

```
（空白）
```

**效果**：
```
=== BRepNet Inference Tool ===
[Model] Weights loaded successfully!
[Files] Found 28548 STEP files

[1/28548] step0001 -> [✓] F:12 E:18
[2/28548] step0002 -> [✓] F:8 E:12
[3/28548] step0003 [SKIPPED]
...
[28548/28548] step28548 -> [✓] F:22 E:35

All 28548 files completed!
Total time: 4523.5 seconds
```

- ❌ 不输出任何 `[DEBUG]`、`[Layer 0]`、`[DIAGNOSTIC]` 信息
- ❌ 不创建 `cpp_feature_maps/`、`cpp_uv_grids/` 中间文件
- ❌ 不创建 `arc_length_diagnosis.txt`、`coedge_grid_diagnosis.txt`
- ✅ 创建最终结果
  - `cpp_logits/*.logits`：原始 logits（softmax 前）
  - `cpp_results/*.results`：分类预测结果（类别、置信度、Top 3）

---

### 场景 B：调试单个问题文件

**命令参数**：
```
--debug --target step3268
```

> 注意：`step3268` 是文件名（不含扩展名 `.step`），即 `step_path.stem()` 的值。

**效果**：
```
=== BRepNet Inference Tool === [DEBUG MODE: target=step3268]
[Model] Weights loaded successfully!

[1/28548] step0001 -> [✓] F:12 E:18          ← 静默，不输出调试
[2/28548] step0002 -> [✓] F:8 E:12           ← 静默
...
[3268/28548] step3268                          ← 匹配！触发调试
================================================================================
Forward Propagation Started
================================================================================
[Input Data]
  Coedges: 190, Faces: 28, Edges: 55

[Layer 0 - First Order Neighbors (MLP-G)]
  [DEBUG Layer 0] Coedge 0 (Face 0)
    face_state (first 5): 0.1234 0.5678 ...
...（完整调试信息）

[Export] Intermediate files saved to:
  cpp_logits/step3268.logits
  cpp_results/step3268.results
  cpp_feature_maps/step3268_*
  cpp_uv_grids/step3268_*
  arc_length_diagnosis.txt
  coedge_grid_diagnosis.txt

step3268 -> [✓] F:28 E:55

[3269/28548] step3269 -> [✓] F:15 E:22       ← 又回到静默
...
```

- ✅ 只对 `step3268` 输出全量调试信息
- ✅ 只对 `step3268` 导出中间文件
- 其他文件保持静默，不影响批量速度

---

### 场景 C：调试所有文件

**命令参数**：
```
--debug
```

**效果**：所有文件都输出完整调试信息 + 导出中间文件。

> ⚠️ 警告：这会产生大量终端输出和磁盘文件，仅建议在少量文件（< 100）时使用。

---

### 场景 D：只导出中间文件，不要终端调试输出

**命令参数**：
```
--export
```

**效果**：
- 终端输出与批量模式相同（简洁）
- 但会为每个文件创建中间文件到 `cpp_feature_maps/`、`cpp_uv_grids/` 等

---

### 场景 E：只导出某个文件的中间数据

**命令参数**：
```
--export --target step3268
```

**效果**：
- 终端输出简洁
- 只对 `step3268` 导出中间文件

---

## 四、参数组合速查表

| 命令参数 | 终端调试 | 永远生成 | 条件生成 | 诊断文件 |
|---------|---------|---------|---------|---------|
| _(空)_ | ❌ | ✅ logits + results | ❌ | ❌ |
| `--debug` | ✅ 全部 | ✅ logits + results | ✅ 全部中间层 | ✅ |
| `--debug --target X` | ✅ 仅X | ✅ logits + results | ✅ 仅X中间层 | ✅ 仅X |
| `--export` | ❌ | ✅ logits + results | ✅ 全部中间层 | ✅ |
| `--export --target X` | ❌ | ✅ logits + results | ✅ 仅X中间层 | ✅ 仅X |

**"终端调试"** = `[DEBUG]`、`[Layer 0]`、MLP输入输出、Pooling详情等  
**"永远生成"** = `cpp_logits/`、`cpp_results/`（所有模式都输出）  
**"条件生成"** = `cpp_feature_maps/`、`cpp_uv_grids/`（调试/导出模式才输出）  
**"诊断文件"** = `arc_length_diagnosis.txt`、`coedge_grid_diagnosis.txt`、`cpp_feature_maps/layer0_mlp_all_coedges_stats.txt` 等

---

## 五、target 参数说明

`--target` 后面跟的是 **STEP 文件的 stem 名称**（不含路径和扩展名）。

例如：
- 文件路径为 `inference_data/step_files/20240116_231044_0_result.step`
- 则 target 值为 `20240116_231044_0_result`

```
--debug --target 20240116_231044_0_result
```

可以指定多个 target（用逗号分隔，无空格）：
```
--debug --target step3268,step5001,step9999
```

---

## 六、典型工作流程

### 工作流：发现并调试问题文件

```
第1步：批量运行
  命令参数：（空）
  → 跑完28000文件，查看 cpp_logits/ 结果

第2步：发现问题
  → 用你的对比脚本发现 step3268 误差很大

第3步：定点调试
  命令参数：--debug --target step3268
  → 按 F5 运行
  → 查看终端输出中各层的数值
  → 查看 cpp_feature_maps/ 中的中间文件
  → 与 Python 端逐层对比，定位误差源

第4步：修复后验证
  命令参数：--debug --target step3268
  → 再次运行，确认误差消除

第5步：恢复批量
  命令参数：（清空）
  → 批量重跑
```

> **整个过程中你没有修改过任何一行代码，没有注释/取消注释，没有重新编译。**
> 只是在 VS 的命令参数栏中复制粘贴不同的字符串。

---

## 七、注意事项

1. **修改命令参数不需要重新编译**：只需在属性页改参数，直接 F5。
2. **Debug/Release 配置分别设置**：命令参数是按配置（Debug/Release）分开存储的，切换配置时请检查。
3. **`--target` 匹配规则**：精确匹配 stem 名称，大小写敏感。
4. **批量模式性能**：静默模式下所有调试代码被条件跳过（`if (false)`），编译器会优化掉，**零性能损失**。
5. **输出目录**：
   - 最终结果（始终生成）
     - `cpp_logits/` — 原始 logits（softmax 前）
     - `cpp_results/` — 分类预测结果（类别、置信度、Top 3）
   - 中间文件（调试/导出模式）
     - `cpp_feature_maps/` — 各层中间特征
     - `cpp_uv_grids/` — UV Grid 原始数据
   - 诊断文件（调试/导出模式）
     - `arc_length_diagnosis.txt`、`coedge_grid_diagnosis.txt`（项目根目录）

---

## 八、故障排查

### Q: 命令参数不生效？
- 检查你是否修改了正确的配置（Debug vs Release）
- 检查属性页顶部的"配置"和"平台"下拉框

### Q: 调试输出没有出现？
- 确认 `--target` 后面的文件名拼写正确（区分大小写）
- 确认该文件确实存在于 `inference_data/step_files/` 目录下

### Q: 中间文件在哪里？
- `cpp_feature_maps/` — 各层中间特征
- `cpp_uv_grids/` — UV Grid 数据
- `cpp_logits/` — 最终 logits（始终生成）
- 项目根目录下 — `arc_length_diagnosis.txt` 和 `coedge_grid_diagnosis.txt`
