from __future__ import annotations

from pathlib import Path
import numpy as np
from PySide6.QtWidgets import (
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

from app.io_utils import PointCloudIO
from app.model import PointCloudModel
from app.viewer import GLViewer


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
