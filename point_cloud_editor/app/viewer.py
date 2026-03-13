from __future__ import annotations

import math
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QMouseEvent, QWheelEvent
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.GL import *

from app.camera import Camera
from app.model import PointCloudModel

try:
    from scipy.spatial import cKDTree  # type: ignore
except Exception:
    cKDTree = None


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

        self.preview_voxel = 0.003
        self.max_preview_points = 250_000
        self.use_preview = True
        self.render_indices = np.zeros((0,), dtype=np.int32)
        self.render_positions = np.zeros((0, 3), dtype=np.float32)
        self.render_colors = np.zeros((0, 3), dtype=np.float32)
        self.render_tree = None

        self.pos_vbo = None
        self.col_vbo = None
        self.gpu_dirty = True
        self.cache_dirty = True

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

    def closeEvent(self, event) -> None:
        try:
            self.cleanupGL()
        except Exception:
            pass
        super().closeEvent(event)

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
