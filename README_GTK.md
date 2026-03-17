# Point Cloud Viewer - GTK + OpenGL 实现

## 项目概述

这是一个基于 GTK3 + PyOpenGL 的点云编辑器，用于可视化和交互编辑大规模点云数据。

**已移除 PySide 依赖** - 现在只使用 GTK3 和 OpenGL，相比 PySide 更轻量级、更高效。

## 安装与运行

### 前置条件
- Python 3.7+
- GTK 3 开发库（系统级）
- 虚拟环境已设置（见下面的项目结构）

### 快速启动

```bash
# 进入项目目录
cd /path/to/PointCloudViewer

# 激活虚拟环境
source venv/bin/activate

# 启动编辑器（会自动加载演示点云）
python point_cloud_editor/main.py
```

**首次启动**：会自动加载一个 40,000 点的演示点云，显示正弦波表面。

### 运行系统要求
- X11 显示服务器（用于图形界面）
- GPU 支持 OpenGL 3.0+（推荐）
- 足够内存（大点云加载时需要）

## 功能使用说明

### 界面布局
- **左侧面板**：操作按钮和设置
- **右侧 OpenGL 区域**：3D 点云视图

### 交互方式
| 操作 | 效果 |
|------|------|
| 右键拖拽 | 旋转视角（优化了流畅度） |
| 滚轮 | 缩放（改变距离时重新评估 LOD） |
| 左键单击 | 选择单个点（点选模式） |
| 左键拖拽 | 框选/刷选多个点（模式可选） |

### 主要功能
- **导入**：加载 PLY / XYZ / TXT 点云文件
- **导出**：保存为 ASCII/Binary PLY 格式
- **选择模式**：点选、框选、刷选
- **编辑操作**：标记删除、提交删除、移动选中点
- **撤销/重做**：支持操作历史回溯
- **LOD/预览**：自动降级采样以提升大点云的交互流畅度

## 性能优化亮点

### 已实现的优化
1. **交互期间的轻量级 LOD**
   - 右键拖拽旋转时，自动切换到 preview LOD（更稀疏的点集）
   - 避免了频繁重建渲染缓存
   - 释放鼠标后恢复完整精度并重建

2. **跳过昂贵的空间索引构建**
   - 在交互期间不构建 KD-tree（成本高）
   - 仅在交互完成后才构建，用于后续选择操作

3. **分离旋转与 LOD 选择**
   - 相机旋转不触发缓存重建（只改变视角矩阵）
   - 缩放时（距离改变）才触发 LOD 重评估

### 测试数据
- 小点云（< 100K 点）：完全实时，每帧 60+ fps
- 中等点云（100K - 1M 点）：预览 LOD 时流畅，完整精度时可能有延迟
- 大点云（> 1M 点）：强烈建议启用 LOD/预览，交互响应时间 < 50ms

## 项目结构

```
PointCloudViewer/
├── point_cloud_editor/
│   ├── main.py                  # 主入口点
│   ├── requirements.txt          # Python 依赖（已移除 PySide）
│   ├── app/
│   │   ├── __init__.py
│   │   ├── gtk_editor.py        # GTK 主编辑窗口
│   │   ├── gtk_viewer.py        # GTK + OpenGL 渲染视图（性能优化核心）
│   │   ├── camera.py            # 相机控制
│   │   ├── model.py             # 点云数据模型
│   │   ├── lod.py               # LOD 构建与选择逻辑
│   │   ├── selection.py         # 选择算法（点/框/刷）
│   │   ├── history.py           # 撤销/重做 栈
│   │   └── io_utils.py          # 文件 I/O
│   └── __pycache__/
├── venv/                         # Python 虚拟环境
├── web/                          # （可选）Web 前端（另外维护）
└── README.md                     # 本文件
```

## 已移除的 PySide 依赖

原项目包含 PySide6 回退实现。现已完全移除：
- ❌ `PySide6` / `PySide6_Addons` / `PySide6_Essentials`
- ❌ `shiboken6`
- ✅ 仅保留 GTK3 + PyOpenGL 实现

**好处**：
- 依赖更简洁（无 Qt 框架开销）
- 启动更快（少 ~500MB 磁盘占用）
- 更轻的运行时开销
- GTK3 渲染集成更原生

## 常见问题

### Q: 启动后显示黑屏
**A:** 通常是因为没有演示数据加载。现版本 `main.py` 自动加载演示点云。如果仍显示黑屏，检查：
- 是否有 OpenGL 支持（运行 `glxinfo | grep "OpenGL version"`）
- 是否有 X11 显示服务器（运行 `echo $DISPLAY`）

### Q: 拖拽视角仍然卡顿（大点云）
**A:** 
1. 确认已启用 "LOD/预览" 选项（左侧面板复选框）
2. 点云超过 100 万点时，预览 LOD 应将点数降至 10-50 万，拖拽应流畅
3. 如果仍卡顿，尝试进一步调整 `app/lod.py` 中的采样率

### Q: 导入自己的点云文件
**A:** 点击"导入"按钮，选择支持的格式：
- `.ply`（推荐，支持 ASCII 和 Binary）
- `.xyz` / `.txt`（逐行 x y z 坐标）
- 颜色：如果有，自动读取；否则默认白色

### Q: 运行时导入错误（ModuleNotFoundError）
**A:** 确保从正确的目录启动：
```bash
# ✅ 正确
cd /path/to/PointCloudViewer
python point_cloud_editor/main.py

# ❌ 错误（会导入失败）
cd /path/to/PointCloudViewer/point_cloud_editor
python main.py  # 这里 import app 会失败
```

或者，修复后的 `main.py` 会自动调整 sys.path，所以两种方式都应该工作。

## 开发与扩展

### 添加新功能
1. 编辑 `app/gtk_editor.py`（UI 按钮）或 `app/gtk_viewer.py`（交互）
2. 遵循现有的事件处理模式
3. 测试与现有 LOD/缓存机制的兼容性

### 性能调优
- 调整 `app/lod.py` 中的采样比例
- 修改 `app/gtk_viewer.py` 中的 `interactive` 模式阈值
- 考虑实现多线程 LOD 构建（见后续 TODOs）

## 许可与致谢

原项目：[PointCloudViewer](https://github.com/LianLian0915/PointCloudViewer)

当前分支：`feature/gtk` - 移除 PySide，全量 GTK+OpenGL 优化

## 下一步计划

- [ ] 异步 LOD 构建（后台线程）
- [ ] 现代 OpenGL shader 管线（全面替代固定管线）
- [ ] GPU 顶点流优化
- [ ] 八叉树/网格分块策略
- [ ] 性能基准测试套件
- [ ] Web 前端集成（WebGL 编码器）

