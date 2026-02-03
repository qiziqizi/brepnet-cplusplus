# Visual Studio 编译和测试指南

## 📋 前置要求

### 必需的依赖
- ✅ Visual Studio 2022（v143 工具集）
- ✅ Windows 10 SDK
- ✅ OpenCascade（通过 vcpkg 安装在 `D:\vcpkg`）
- ✅ LibTorch（安装在 `D:\libtorch`）
- ✅ CUDA 12.4（安装在 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4`）
- ✅ cnpy 库（在 `D:\cnpy`）

### 验证依赖路径
在编译前，请确认以下路径存在：
```
D:\vcpkg\installed\x64-windows\include
D:\vcpkg\installed\x64-windows\include\opencascade
D:\libtorch\include
D:\cnpy
```

---

## 🔧 编译步骤

### 方法 1: 使用 Visual Studio GUI（推荐）

#### 1. 打开项目
1. 双击打开 `D:\brepnet-cplusplus\brepnet\brepnet.sln`
2. Visual Studio 会自动加载项目

#### 2. 选择配置
在工具栏选择：
- **配置**: `Release`（推荐）或 `Debug`
- **平台**: `x64`

#### 3. 配置验证输出（可选）

**启用验证输出**（用于调试）:
1. 右键点击项目 → **属性**
2. 导航到：**C/C++** → **预处理器** → **预处理器定义**
3. 添加：`ENABLE_VERIFICATION=1`
4. 点击 **应用** 和 **确定**

**禁用验证输出**（发布版本，默认）:
- 不需要额外配置，默认 `ENABLE_VERIFICATION=0`

#### 4. 编译项目
- 按 `Ctrl+Shift+B` 或
- 菜单：**生成** → **生成解决方案**

#### 5. 检查编译结果
编译成功后，可执行文件位于：
```
D:\brepnet-cplusplus\brepnet\x64\Release\brepnet.exe
```
或（Debug 模式）：
```
D:\brepnet-cplusplus\brepnet\x64\Debug\brepnet.exe
```

---

### 方法 2: 使用命令行（高级）

#### 1. 打开 Developer Command Prompt
- 开始菜单 → **Visual Studio 2022** → **Developer Command Prompt for VS 2022**

#### 2. 导航到项目目录
```cmd
cd /d D:\brepnet-cplusplus\brepnet
```

#### 3. 编译（Release 版本）
```cmd
msbuild brepnet.sln /p:Configuration=Release /p:Platform=x64 /m
```

#### 4. 编译（Debug 版本，启用验证）
```cmd
msbuild brepnet.sln /p:Configuration=Debug /p:Platform=x64 /p:DefineConstants="ENABLE_VERIFICATION=1" /m
```

---

## 🧪 测试步骤

### 1. 准备测试数据

确认以下文件存在：
```
D:\brepnet-cplusplus\verification_data_0101.npz
D:\brepnet-cplusplus\brepnet_weights_0101.npz
D:\brepnet-cplusplus\s2.0.0\breps\step\136322_81d84c1b_1.stp
```

如果文件不存在，请从原始数据源复制。

### 2. 运行测试

#### 方法 A: 在 Visual Studio 中运行
1. 按 `F5`（调试运行）或 `Ctrl+F5`（不调试运行）
2. 程序会自动运行并显示输出

#### 方法 B: 命令行运行
```cmd
cd /d D:\brepnet-cplusplus
brepnet\x64\Release\brepnet.exe
```

### 3. 验证输出

#### 成功标准
程序应该输出：
```
[Config] Verify File : D:\brepnet-cplusplus\verification_data_0101.npz
[Config] Weights File: D:\brepnet-cplusplus\brepnet_weights_0101.npz
[Config] STEP File   : D:\brepnet-cplusplus\s2.0.0\breps\step\136322_81d84c1b_1.stp

[Perf] 数据预处理耗时: XXX ms
[Perf] 模型初始化耗时: XXX ms
=== 推理成功! ===

SUCCESS! 通过验证
```

#### 关键验证点
- ✅ **编译无错误**: 0 errors
- ✅ **输出误差 < 0.1**: 查看 "Total_Error" 或最终误差值
- ✅ **无崩溃**: 程序正常退出

#### 如果启用了 ENABLE_VERIFICATION
你会看到额外的调试输出：
```
[Verify:Psi_Pt_Shape] ...
[Verify:Psi_Pe_Range] ...
[Verify:CPP_Logits_Row1] ...
[Verify:Total_Error] 0.05
```

---

## 🐛 常见问题和解决方案

### 问题 1: 找不到 OpenCascade 头文件
**错误**: `fatal error C1083: Cannot open include file: 'TopoDS_Shape.hxx'`

**解决方案**:
1. 检查 vcpkg 安装：
   ```cmd
   vcpkg list | findstr opencascade
   ```
2. 如果未安装，运行：
   ```cmd
   vcpkg install opencascade:x64-windows
   ```
3. 更新项目包含路径（已在 .vcxproj 中配置）

### 问题 2: 找不到 LibTorch 库
**错误**: `LINK : fatal error LNK1181: cannot open input file 'torch.lib'`

**解决方案**:
1. 确认 LibTorch 安装在 `D:\libtorch`
2. 如果路径不同，更新 .vcxproj 中的库路径
3. 确保下载的是 **Release** 版本的 LibTorch（带 CUDA 支持）

### 问题 3: CUDA 相关错误
**错误**: `cannot open file 'cudart.lib'`

**解决方案**:
1. 确认 CUDA 12.4 已安装
2. 检查环境变量 `CUDA_PATH` 是否设置
3. 如果使用不同版本的 CUDA，更新 .vcxproj 中的路径

### 问题 4: 编译时内存不足
**错误**: `fatal error C1060: compiler is out of heap space`

**解决方案**:
1. 关闭其他应用程序
2. 使用 `/m:1` 参数限制并行编译：
   ```cmd
   msbuild brepnet.sln /p:Configuration=Release /p:Platform=x64 /m:1
   ```

### 问题 5: 运行时找不到 DLL
**错误**: `The code execution cannot proceed because XXX.dll was not found`

**解决方案**:
1. 将以下目录添加到系统 PATH：
   ```
   D:\vcpkg\installed\x64-windows\bin
   D:\libtorch\lib
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin
   ```
2. 或者将所需 DLL 复制到可执行文件目录

### 问题 6: 验证失败（误差 > 0.1）
**可能原因**:
- 权重文件版本不匹配
- 数据预处理逻辑错误
- 浮点精度问题

**调试步骤**:
1. 启用 `ENABLE_VERIFICATION=1` 重新编译
2. 检查中间输出值
3. 对比 Python 版本的输出

---

## 📊 性能基准

### 预期性能（Release 模式，RTX 3090）
- **数据预处理**: ~500-1000 ms
- **模型初始化**: ~2000-3000 ms
- **推理时间**: ~100-300 ms
- **总内存占用**: ~2-4 GB

### 性能优化建议
1. 使用 **Release** 配置（比 Debug 快 5-10 倍）
2. 禁用 `ENABLE_VERIFICATION`（减少 I/O 开销）
3. 确保使用 GPU 加速（CUDA）

---

## 🔄 重新编译后的验证清单

每次修改代码后，请执行以下检查：

- [ ] **编译成功**: 0 errors, 0 warnings（或仅有可忽略的警告）
- [ ] **运行成功**: 程序正常退出，无崩溃
- [ ] **输出验证**: 误差 < 0.1
- [ ] **内存检查**: 无明显内存泄漏（可使用 Visual Studio 的诊断工具）
- [ ] **性能检查**: 推理时间在合理范围内

---

## 📝 提交前检查

在提交代码到 Git 之前：

1. **清理构建产物**:
   ```cmd
   git clean -fdx brepnet/
   ```

2. **确保 .gitignore 正确**:
   - 不要提交 `brepnet/x64/` 目录
   - 不要提交 `.vs/` 目录
   - 不要提交 `*.user` 文件

3. **验证分支状态**:
   ```cmd
   git status
   git log --oneline -5
   ```

---

## 🚀 合并到 dev 分支

测试通过后，执行以下步骤：

```bash
# 1. 确保当前分支是 refactor/inference-only
git branch

# 2. 提交所有更改（如果有）
git add .
git commit -m "chore: 更新项目文件以包含 VerificationLogger.h"

# 3. 切换到 dev 分支
git checkout dev

# 4. 合并重构分支
git merge refactor/inference-only

# 5. 解决冲突（如果有）
# 编辑冲突文件，然后：
git add <resolved-files>
git commit

# 6. 推送到远程
git push origin dev

# 7. 删除本地重构分支（可选）
git branch -d refactor/inference-only
```

---

## 📞 获取帮助

如果遇到问题：
1. 检查本文档的"常见问题"部分
2. 查看 `REFACTORING_SUMMARY.md` 了解重构详情
3. 查看 Git 提交历史了解具体更改
4. 在 GitHub Issues 中报告问题

---

## 📚 相关文档

- `REFACTORING_SUMMARY.md` - 重构总结
- `README.md` - 项目概述
- `refactoring_report.pdf` - 详细重构报告
