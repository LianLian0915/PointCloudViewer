# PointCloudViewer

GTK 3 + ModernGL 点云编辑查看器，支持 PLY / XYZ / TXT 点云导入、选择、移动、软删除、导出和大点云预览渲染。

## 特性

- GTK 3 桌面界面
- ModernGL / OpenGL 点云渲染
- 大点云交互预览 LOD
- PLY、XYZ、TXT 导入
- ASCII PLY / XYZ 和 Binary PLY 导出
- 点选、框选、刷选、移动、删除、撤销/重做

## 环境要求

- Python 3.10+
- GTK 3 和 PyGObject 系统库
- OpenGL 3.2+ 环境

Ubuntu / Debian 系统先安装 GTK 和 OpenGL 相关系统依赖：

```bash
sudo apt update
sudo apt install -y \
  python3-gi \
  python3-gi-cairo \
  gir1.2-gtk-3.0 \
  libgl1-mesa-dri \
  libgl1-mesa-glx
```

## 安装 Python 依赖

建议使用带系统包可见性的虚拟环境，因为 `gi` / GTK 通常由系统包提供：

```bash
python3 -m venv venv --system-site-packages
source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` 只保留项目核心运行依赖：

- `numpy`
- `moderngl`
- `PyOpenGL`
- `PyOpenGL-accelerate`
- `plyfile`

可选依赖：

- `scipy`：用于 benchmark 中的 KD-tree 性能测试
- `Pillow`：用于 `test_render_to_image.py` 保存截图

需要时单独安装：

```bash
pip install scipy Pillow
```

## 运行 GTK 编辑器

```bash
source venv/bin/activate
python point_cloud_editor/main.py
```

启动后可以点击“导入”加载 `.ply`、`.xyz` 或 `.txt` 点云文件。

## 测试和调试

基础导入测试：

```bash
python test_run.py
```

性能基准测试：

```bash
python benchmark.py
```

渲染截图测试需要额外安装 `Pillow`：

```bash
pip install Pillow
python test_render_to_image.py
```

## Web 查看器

仓库中还包含一个 Three.js Web 点云查看器，依赖由 `web/package.json` 管理，不包含在根目录 Python `requirements.txt` 中。

```bash
cd web
npm install
npm run dev
```
