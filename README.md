# PointCloudViewer - GTK + ModernGL 版本

![Status](https://img.shields.io/badge/status-✅%20Production%20Ready-green)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![GTK](https://img.shields.io/badge/GTK-3.0-blue)
![OpenGL](https://img.shields.io/badge/OpenGL-4.1%2B-blue)

现代化的点云编辑和查看工具，使用 **GTK 3.0** 界面和 **ModernGL** 渲染管线。

## 🚀 最新特性

- ✅ **GTK 界面**: 替代了 PySide6，更轻量级
- ✅ **ModernGL 渲染**: 支持 OpenGL Core Profile
- ✅ **三级 LOD 系统**: 优化大点云的拖动流畅度
- ✅ **CPU 端 MVP 计算**: 规避固定功能管道限制
- ✅ **40000+ 点实时渲染**: 已验证

## 📋 需求

```bash
python3 -m venv venv --system-site-packages