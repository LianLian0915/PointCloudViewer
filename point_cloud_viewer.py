#!/usr/bin/env python3
"""
Optimized Point Cloud Editor
PySide6 6.10.2 + PyOpenGL 3.1.10

Goals:
- Stable in VM / Mesa / llvmpipe
- Smooth viewing for ~2M points by rendering a downsampled preview
- VBO-based rendering (upload only when data changes)
- Import: XYZ / TXT / ASCII/Binary PLY
- Export: XYZ / TXT / ASCII PLY
- Select / delete / move points
- Reset view / point size / preview tuning

Install:
    pip install PySide6==6.10.2 PyOpenGL==3.1.10 numpy plyfile
Optional:
    pip install scipy
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from plyfile import PlyData
from PySide6.QtCore import Qt
from PySide6.QtGui import QMouseEvent, QSurfaceFormat, QWheelEvent
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from OpenGL.GL import *

try:
    from scipy.spatial import cKDTree  # type: ignore
except Exception:  # pragma: no cover
    cKDTree = None


# -------------------------
# Model
# -------------------------


class PointCloudModel:
    def __init__(self) -> None:
        self.positions = np.zeros((0, 3), dtype=np.float32)
        self.colors = np.zeros((0, 3), dtype=np.float32)
        self.selected = np.zeros((0,), dtype=bool)
        self.current_path: str = ""

    @property
    def count(self) -> int:
        return int(self.positions.shape[0])

    def clear(self) -> None:
        self.positions = np.zeros((0, 3), dtype=np.float32)
        self.colors = np.zeros((0, 3), dtype=np.float32)
        self.selected = np.zeros((0,), dtype=bool)
        self.current_path = ""

    def set_data(self, pos: np.ndarray, col: np.ndarray, path: str = "") -> None:
        pos = np.asarray(pos, dtype=np.float32).reshape(-1, 3)
        col = np.asarray(col, dtype=np.float32).reshape(-1, 3)
        if pos.shape[0] != col.shape[0]:
            raise ValueError("positions 和 colors 数量不一致")
        valid = np.isfinite(pos).all(axis=1)
        self.positions = np.ascontiguousarray(pos[valid], dtype=np.float32)
        self.colors = np.ascontiguousarray(np.clip(col[valid], 0.0, 1.0), dtype=np.float32)
        self.selected = np.zeros((self.positions.shape[0],), dtype=bool)
        self.current_path = path

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count == 0:
            z = np.zeros((3,), dtype=np.float32)
            return z, z
        return self.positions.min(axis=0), self.positions.max(axis=0)

    def clear_selection(self) -> None:
        if self.selected.size:
            self.selected[:] = False

    def selected_count(self) -> int:
        return int(np.count_nonzero(self.selected))

    def select_raw_index(self, idx: int) -> None:
        self.clear_selection()
        if 0 <= idx < self.count:
            self.selected[idx] = True

    def delete_selected(self) -> int:
        if self.count == 0:
            return 0
        keep = ~self.selected
        removed = int(np.count_nonzero(self.selected))
        self.positions = np.ascontiguousarray(self.positions[keep], dtype=np.float32)
        self.colors = np.ascontiguousarray(self.colors[keep], dtype=np.float32)
        self.selected = np.zeros((self.positions.shape[0],), dtype=bool)
        return removed

    def move_selected(self, dx: float, dy: float, dz: float) -> int:
        moved = self.selected_count()
        if moved > 0:
            self.positions[self.selected] += np.array([dx, dy, dz], dtype=np.float32)
            self.positions = np.ascontiguousarray(self.positions, dtype=np.float32)
        return moved


# -------------------------
# IO
# -------------------------


class PointCloudIO:
    @staticmethod
    def load_xyz(path: str) -> tuple[np.ndarray, np.ndarray]:
        pts, cols = [], []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                vals = line.split()
                if len(vals) < 3:
                    continue
                x, y, z = map(float, vals[:3])
                pts.append([x, y, z])
                if len(vals) >= 6:
                    r, g, b = map(float, vals[3:6])
                    if max(r, g, b) > 1.0:
                        r /= 255.0
                        g /= 255.0
                        b /= 255.0
                    cols.append([r, g, b])
                else:
                    cols.append([0.2, 0.8, 1.0])
        if not pts:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
        return np.asarray(pts, dtype=np.float32), np.asarray(cols, dtype=np.float32)

    @staticmethod
    def load_ply(path: str) -> tuple[np.ndarray, np.ndarray]:
        ply = PlyData.read(path)
        if "vertex" not in ply:
            raise ValueError("PLY 文件中没有 vertex")
        vertex = ply["vertex"]
        names = set(vertex.data.dtype.names or [])
        if not {"x", "y", "z"}.issubset(names):
            raise ValueError("PLY 顶点缺少 x/y/z 字段")

        pos = np.column_stack([
            np.asarray(vertex["x"], dtype=np.float32),
            np.asarray(vertex["y"], dtype=np.float32),
            np.asarray(vertex["z"], dtype=np.float32),
        ])
        valid = np.isfinite(pos).all(axis=1)
        pos = pos[valid]

        if {"red", "green", "blue"}.issubset(names):
            col = np.column_stack([
                np.asarray(vertex["red"], dtype=np.float32),
                np.asarray(vertex["green"], dtype=np.float32),
                np.asarray(vertex["blue"], dtype=np.float32),
            ])
            col = col[valid]
            if col.size and float(col.max()) > 1.0:
                col /= 255.0
        else:
            col = np.ones_like(pos, dtype=np.float32) * np.array([0.2, 0.8, 1.0], dtype=np.float32)

        return np.ascontiguousarray(pos, dtype=np.float32), np.ascontiguousarray(np.clip(col, 0.0, 1.0), dtype=np.float32)

    @staticmethod
    def save_xyz(path: str, positions: np.ndarray, colors: np.ndarray) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for p, c in zip(positions, colors):
                rgb = np.clip(np.round(c * 255.0), 0, 255).astype(np.int32)
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {rgb[0]} {rgb[1]} {rgb[2]}\n")

    @staticmethod
    def save_ply(path: str, positions: np.ndarray, colors: np.ndarray) -> None:
        rgb = np.clip(np.round(colors * 255.0), 0, 255).astype(np.uint8)
        with open(path, "w", encoding="utf-8") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {positions.shape[0]}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for p, c in zip(positions, rgb):
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")

    @staticmethod
    def load(path: str) -> tuple[np.ndarray, np.ndarray]:
        ext = Path(path).suffix.lower()
        if ext in {".xyz", ".txt"}:
            return PointCloudIO.load_xyz(path)
        if ext == ".ply":
            return PointCloudIO.load_ply(path)
        raise ValueError("Unsupported format")

    @staticmethod
    def save(path: str, positions: np.ndarray, colors: np.ndarray) -> None:
        ext = Path(path).suffix.lower()
        if ext in {".xyz", ".txt"}:
            PointCloudIO.save_xyz(path, positions, colors)
            return
        if ext == ".ply":
            PointCloudIO.save_ply(path, positions, colors)
            return
        raise ValueError("Unsupported format")


# -------------------------
# Camera
# -------------------------


@dataclass
class Camera:
    yaw: float = 0.0
    pitch: float = 0.0
    dist: float = 3.0
    center: np.ndarray = field(default_factory=lambda: np.zeros((3,), dtype=np.float32))


# -------------------------
# Viewer
# -------------------------


class GLViewer(QOpenGLWidget):
    def __init__(self, model: PointCloudModel, status_callback) -> None:
        super().__init__()
        self.model = model
        self.status_callback = status_callback
        self.camera = Camera()
        self.point_size = 3.0
        self.last_pos = None
        self.setMinimumSize(640, 480)
        self.setMouseTracking(True)

        # render cache
        self.preview_voxel = 0.003
        self.max_preview_points = 250_000
        self.use_preview = True
        self.render_indices = np.zeros((0,), dtype=np.int32)
        self.render_positions = np.zeros((0, 3), dtype=np.float32)
        self.render_colors = np.zeros((0, 3), dtype=np.float32)
        self.render_tree = None

        # vbo
        self.pos_vbo = None
        self.col_vbo = None
        self.gpu_dirty = True
        self.cache_dirty = True

    # ----- OpenGL init -----
    def initializeGL(self) -> None:
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_POINT_SMOOTH)

        self.pos_vbo = glGenBuffers(1)
        self.col_vbo = glGenBuffers(1)

        version = glGetString(GL_VERSION)
        renderer = glGetString(GL_RENDERER)
        v = version.decode("utf-8", errors="ignore") if version else "unknown"
        r = renderer.decode("utf-8", errors="ignore") if renderer else "unknown"
        self.status_callback(f"OpenGL: {v} | Renderer: {r}")

    def cleanupGL(self) -> None:
        self.makeCurrent()
        try:
            if self.pos_vbo:
                glDeleteBuffers(1, [self.pos_vbo])
                self.pos_vbo = None
            if self.col_vbo:
                glDeleteBuffers(1, [self.col_vbo])
                self.col_vbo = None
        finally:
            self.doneCurrent()

    def closeEvent(self, event) -> None:  # noqa: N802
        try:
            self.cleanupGL()
        except Exception:
            pass
        super().closeEvent(event)

    # ----- data preparation -----
    def mark_model_dirty(self) -> None:
        self.cache_dirty = True
        self.gpu_dirty = True
        self.update()

    def set_preview_settings(self, use_preview: bool, voxel: float, max_points: int) -> None:
        self.use_preview = use_preview
        self.preview_voxel = max(1e-6, float(voxel))
        self.max_preview_points = max(1000, int(max_points))
        self.mark_model_dirty()

    def rebuild_render_cache(self) -> None:
        if self.model.count == 0:
            self.render_indices = np.zeros((0,), dtype=np.int32)
            self.render_positions = np.zeros((0, 3), dtype=np.float32)
            self.render_colors = np.zeros((0, 3), dtype=np.float32)
            self.render_tree = None
            self.cache_dirty = False
            return

        pos = self.model.positions
        idx = np.arange(self.model.count, dtype=np.int32)

        if self.use_preview and self.model.count > self.max_preview_points:
            scaled = np.floor(pos / self.preview_voxel).astype(np.int64)
            _, unique_idx = np.unique(scaled, axis=0, return_index=True)
            idx = np.sort(unique_idx.astype(np.int32))
            if idx.shape[0] > self.max_preview_points:
                step = max(1, idx.shape[0] // self.max_preview_points)
                idx = idx[::step][: self.max_preview_points]

        self.render_indices = np.ascontiguousarray(idx, dtype=np.int32)
        self.render_positions = np.ascontiguousarray(self.model.positions[self.render_indices], dtype=np.float32)
        self.render_colors = np.ascontiguousarray(self.model.colors[self.render_indices], dtype=np.float32)

        if cKDTree is not None and self.render_positions.shape[0] > 0:
            try:
                self.render_tree = cKDTree(self.render_positions)
            except Exception:
                self.render_tree = None
        else:
            self.render_tree = None

        self.cache_dirty = False
        self.gpu_dirty = True

    def upload_gpu_buffers(self) -> None:
        if self.cache_dirty:
            self.rebuild_render_cache()

        colors = self.render_colors.copy()
        if self.model.selected_count() > 0 and self.render_indices.size > 0:
            sel_mask = self.model.selected[self.render_indices]
            colors[sel_mask] = np.array([1.0, 0.2, 0.1], dtype=np.float32)
        colors = np.ascontiguousarray(colors, dtype=np.float32)

        glBindBuffer(GL_ARRAY_BUFFER, self.pos_vbo)
        glBufferData(GL_ARRAY_BUFFER, self.render_positions.nbytes, self.render_positions, GL_STATIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, self.col_vbo)
        glBufferData(GL_ARRAY_BUFFER, colors.nbytes, colors, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        self.gpu_dirty = False

    # ----- camera -----
    def reset_view(self) -> None:
        self.camera.yaw = 0.0
        self.camera.pitch = 0.0
        self.fit_camera_to_model()
        self.update()

    def fit_camera_to_model(self) -> None:
        if self.model.count == 0:
            self.camera.center = np.zeros((3,), dtype=np.float32)
            self.camera.dist = 3.0
            return
        mn, mx = self.model.bounds()
        self.camera.center = ((mn + mx) * 0.5).astype(np.float32)
        extent = (mx - mn).astype(np.float32)
        radius = max(0.5, float(np.linalg.norm(extent)) * 0.5)
        self.camera.dist = max(2.0, radius * 3.0)

    def setup_projection(self) -> None:
        w = max(1, self.width())
        h = max(1, self.height())
        aspect = w / h
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        znear = 0.01
        zfar = 100000.0
        fovy = 45.0
        top = znear * math.tan(math.radians(fovy) * 0.5)
        bottom = -top
        right = top * aspect
        left = -right
        glFrustum(left, right, bottom, top, znear, zfar)

    def setup_modelview(self) -> None:
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -float(self.camera.dist))
        glRotatef(math.degrees(self.camera.pitch), 1.0, 0.0, 0.0)
        glRotatef(math.degrees(self.camera.yaw), 0.0, 1.0, 0.0)
        glTranslatef(-float(self.camera.center[0]), -float(self.camera.center[1]), -float(self.camera.center[2]))

    # ----- rendering -----
    def paintGL(self) -> None:
        glViewport(0, 0, self.width(), self.height())
        glClearColor(0.1, 0.1, 0.12, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if self.model.count == 0:
            return
        if self.gpu_dirty or self.cache_dirty:
            self.upload_gpu_buffers()

        self.setup_projection()
        self.setup_modelview()

        glPointSize(float(self.point_size))
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)

        glBindBuffer(GL_ARRAY_BUFFER, self.pos_vbo)
        glVertexPointer(3, GL_FLOAT, 0, None)
        glBindBuffer(GL_ARRAY_BUFFER, self.col_vbo)
        glColorPointer(3, GL_FLOAT, 0, None)

        glDrawArrays(GL_POINTS, 0, int(self.render_positions.shape[0]))

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)

    # ----- interaction -----
    def mousePressEvent(self, e: QMouseEvent) -> None:
        self.last_pos = e.position()
        if e.button() == Qt.LeftButton and self.model.count > 0:
            self.pick_point(e.position().x(), e.position().y())

    def mouseMoveEvent(self, e: QMouseEvent) -> None:
        if self.last_pos is None:
            return
        if e.buttons() & Qt.RightButton:
            dx = e.position().x() - self.last_pos.x()
            dy = e.position().y() - self.last_pos.y()
            self.camera.yaw += dx * 0.01
            self.camera.pitch += dy * 0.01
            self.camera.pitch = max(-1.5, min(1.5, self.camera.pitch))
            self.update()
        self.last_pos = e.position()

    def mouseReleaseEvent(self, e: QMouseEvent) -> None:
        self.last_pos = None
        super().mouseReleaseEvent(e)

    def wheelEvent(self, e: QWheelEvent) -> None:
        self.camera.dist += -e.angleDelta().y() * 0.001
        self.camera.dist = max(0.2, min(100000.0, self.camera.dist))
        self.update()

    # ----- picking -----
    def project_render_points(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.render_positions.shape[0] == 0:
            z = np.zeros((0,), dtype=np.float32)
            return z, z, z
        pts = self.render_positions - self.camera.center[None, :]
        cp = math.cos(self.camera.pitch)
        sp = math.sin(self.camera.pitch)
        cy = math.cos(self.camera.yaw)
        sy = math.sin(self.camera.yaw)

        x1 = cy * pts[:, 0] + sy * pts[:, 2]
        y1 = pts[:, 1]
        z1 = -sy * pts[:, 0] + cy * pts[:, 2]

        x2 = x1
        y2 = cp * y1 - sp * z1
        z2 = sp * y1 + cp * z1 - self.camera.dist

        valid = z2 < -1e-5
        aspect = max(1.0, self.width() / max(1, self.height()))
        f = 1.0 / math.tan(math.radians(45.0) * 0.5)
        ndc_x = np.zeros_like(x2)
        ndc_y = np.zeros_like(y2)
        depth = np.full_like(z2, np.inf)
        ndc_x[valid] = (x2[valid] * f / aspect) / (-z2[valid])
        ndc_y[valid] = (y2[valid] * f) / (-z2[valid])
        depth[valid] = -z2[valid]
        sx = (ndc_x * 0.5 + 0.5) * self.width()
        sy_scr = (1.0 - (ndc_y * 0.5 + 0.5)) * self.height()
        return sx, sy_scr, depth

    def pick_point(self, x: float, y: float) -> None:
        if self.cache_dirty:
            self.rebuild_render_cache()
        sx, sy, depth = self.project_render_points()
        if sx.size == 0:
            return
        d = np.sqrt((sx - x) ** 2 + (sy - y) ** 2)
        threshold = max(8.0, self.point_size + 4.0)
        candidates = np.where(d < threshold)[0]
        if candidates.size == 0:
            self.model.clear_selection()
            self.status_callback("未命中点")
        else:
            best_render = int(candidates[np.argmin(depth[candidates])])
            raw_idx = int(self.render_indices[best_render])
            self.model.select_raw_index(raw_idx)
            p = self.model.positions[raw_idx]
            self.status_callback(f"已选中点 #{raw_idx}: ({p[0]:.5f}, {p[1]:.5f}, {p[2]:.5f})")
        self.gpu_dirty = True
        self.update()


# -------------------------
# Main Window
# -------------------------


class Editor(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Optimized Point Cloud Editor (PySide6 + PyOpenGL)")
        self.model = PointCloudModel()

        root = QWidget()
        layout = QHBoxLayout(root)
        self.viewer = GLViewer(self.model, self.set_status)

        side = QVBoxLayout()
        title = QLabel("点云编辑查看器")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")

        load_btn = QPushButton("导入")
        save_btn = QPushButton("导出")
        demo_btn = QPushButton("生成示例点云")
        delete_btn = QPushButton("删除选中点")
        clear_sel_btn = QPushButton("清除选择")
        reset_btn = QPushButton("重置视角")

        load_btn.clicked.connect(self.load)
        save_btn.clicked.connect(self.save)
        demo_btn.clicked.connect(self.demo)
        delete_btn.clicked.connect(self.delete_selected)
        clear_sel_btn.clicked.connect(self.clear_selection)
        reset_btn.clicked.connect(self.reset_view)

        self.point_size_spin = QSpinBox()
        self.point_size_spin.setRange(1, 20)
        self.point_size_spin.setValue(3)
        self.point_size_spin.valueChanged.connect(self.on_point_size_changed)

        self.use_preview_check = QCheckBox("启用预览降采样")
        self.use_preview_check.setChecked(True)
        self.use_preview_check.toggled.connect(self.on_preview_settings_changed)

        self.voxel_spin = QDoubleSpinBox()
        self.voxel_spin.setRange(0.0001, 10.0)
        self.voxel_spin.setDecimals(4)
        self.voxel_spin.setSingleStep(0.001)
        self.voxel_spin.setValue(0.003)
        self.voxel_spin.valueChanged.connect(self.on_preview_settings_changed)

        self.max_points_spin = QSpinBox()
        self.max_points_spin.setRange(10_000, 2_000_000)
        self.max_points_spin.setSingleStep(10_000)
        self.max_points_spin.setValue(250_000)
        self.max_points_spin.valueChanged.connect(self.on_preview_settings_changed)

        self.move_x = QDoubleSpinBox()
        self.move_y = QDoubleSpinBox()
        self.move_z = QDoubleSpinBox()
        for spin in (self.move_x, self.move_y, self.move_z):
            spin.setRange(-1000.0, 1000.0)
            spin.setDecimals(4)
            spin.setSingleStep(0.01)
        move_btn = QPushButton("移动选中点")
        move_btn.clicked.connect(self.move_selected)

        self.status = QLabel("就绪")
        self.status.setWordWrap(True)

        side.addWidget(title)
        side.addWidget(load_btn)
        side.addWidget(save_btn)
        side.addWidget(demo_btn)
        side.addWidget(delete_btn)
        side.addWidget(clear_sel_btn)
        side.addWidget(reset_btn)
        side.addWidget(QLabel("点大小"))
        side.addWidget(self.point_size_spin)
        side.addWidget(self.use_preview_check)
        side.addWidget(QLabel("降采样体素"))
        side.addWidget(self.voxel_spin)
        side.addWidget(QLabel("最大预览点数"))
        side.addWidget(self.max_points_spin)
        side.addWidget(QLabel("dx"))
        side.addWidget(self.move_x)
        side.addWidget(QLabel("dy"))
        side.addWidget(self.move_y)
        side.addWidget(QLabel("dz"))
        side.addWidget(self.move_z)
        side.addWidget(move_btn)
        side.addStretch(1)
        side.addWidget(QLabel("操作说明：\n左键点选\n右键拖拽旋转\n滚轮缩放\n200万点建议开启预览降采样"))
        side.addWidget(self.status)

        layout.addLayout(side, 0)
        layout.addWidget(self.viewer, 1)
        self.setCentralWidget(root)

    def set_status(self, text: str) -> None:
        self.status.setText(text)

    def on_preview_settings_changed(self) -> None:
        self.viewer.set_preview_settings(
            self.use_preview_check.isChecked(),
            self.voxel_spin.value(),
            self.max_points_spin.value(),
        )
        if self.model.count > 0:
            self.set_status(
                f"预览设置已更新：体素={self.voxel_spin.value():.4f}，最大点数={self.max_points_spin.value()}"
            )

    def on_point_size_changed(self, value: int) -> None:
        self.viewer.point_size = float(value)
        self.viewer.update()

    def load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open", "", "PointCloud (*.ply *.xyz *.txt)")
        if not path:
            return
        try:
            pos, col = PointCloudIO.load(path)
            if pos.shape[0] == 0:
                raise ValueError("点云为空")
            self.model.set_data(pos, col, path)
            self.viewer.camera.yaw = 0.0
            self.viewer.camera.pitch = 0.0
            self.viewer.fit_camera_to_model()
            self.viewer.mark_model_dirty()
            mn, mx = self.model.bounds()
            self.set_status(
                f"已导入: {Path(path).name}，原始点数: {self.model.count}，范围: "
                f"x[{mn[0]:.3f},{mx[0]:.3f}] y[{mn[1]:.3f},{mx[1]:.3f}] z[{mn[2]:.3f},{mx[2]:.3f}]"
            )
        except Exception as e:
            QMessageBox.critical(self, "导入失败", str(e))
            self.set_status(f"导入失败: {e}")

    def save(self) -> None:
        if self.model.count == 0:
            self.set_status("没有可导出的点云")
            return
        default_name = str(Path(self.model.current_path).with_suffix(".ply") if self.model.current_path else "point_cloud.ply")
        path, _ = QFileDialog.getSaveFileName(self, "Save", default_name, "PointCloud (*.ply *.xyz *.txt)")
        if not path:
            return
        try:
            PointCloudIO.save(path, self.model.positions, self.model.colors)
            self.set_status(f"已导出: {Path(path).name}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))
            self.set_status(f"导出失败: {e}")

    def demo(self) -> None:
        x = np.linspace(-0.5, 0.5, 200, dtype=np.float32)
        z = np.linspace(-0.5, 0.5, 200, dtype=np.float32)
        gx, gz = np.meshgrid(x, z)
        gy = np.sin((gx + gz) * 10.0) * 0.05
        pos = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.float32)
        col = np.ones_like(pos, dtype=np.float32) * np.array([0.2, 0.8, 1.0], dtype=np.float32)
        self.model.set_data(pos, col)
        self.viewer.fit_camera_to_model()
        self.viewer.mark_model_dirty()
        self.set_status(f"已生成示例点云，原始点数: {self.model.count}")

    def delete_selected(self) -> None:
        removed = self.model.delete_selected()
        self.viewer.fit_camera_to_model()
        self.viewer.mark_model_dirty()
        self.set_status(f"已删除点数: {removed}")

    def clear_selection(self) -> None:
        self.model.clear_selection()
        self.viewer.gpu_dirty = True
        self.viewer.update()
        self.set_status("已清除选择")

    def move_selected(self) -> None:
        moved = self.model.move_selected(self.move_x.value(), self.move_y.value(), self.move_z.value())
        self.viewer.mark_model_dirty()
        self.set_status(f"已移动点数: {moved}")

    def reset_view(self) -> None:
        self.viewer.reset_view()
        self.set_status("已重置视角")


def main() -> None:
    fmt = QSurfaceFormat()
    fmt.setRenderableType(QSurfaceFormat.OpenGL)
    fmt.setVersion(2, 1)
    fmt.setProfile(QSurfaceFormat.NoProfile)
    fmt.setDepthBufferSize(24)
    fmt.setStencilBufferSize(8)
    fmt.setSwapBehavior(QSurfaceFormat.DoubleBuffer)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    w = Editor()
    w.resize(1360, 860)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
