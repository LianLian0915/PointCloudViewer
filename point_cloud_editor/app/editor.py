from __future__ import annotations

from pathlib import Path
import numpy as np
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.history import HistoryStack
from app.io_utils import PointCloudIO
from app.model import PointCloudModel
from app.viewer import GLViewer


class Editor(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Enhanced Point Cloud Editor (PySide6 + PyOpenGL)")
        self.model = PointCloudModel()
        self.history = HistoryStack(max_size=50)

        root = QWidget()
        layout = QHBoxLayout(root)
        self.viewer = GLViewer(self.model, self.set_status)

        side = QVBoxLayout()
        title = QLabel("点云编辑查看器")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")

        load_btn = QPushButton("导入")
        save_ascii_btn = QPushButton("导出 ASCII PLY/XYZ")
        save_bin_btn = QPushButton("导出 Binary PLY")
        demo_btn = QPushButton("生成示例点云")
        mark_delete_btn = QPushButton("标记删除选中点")
        commit_delete_btn = QPushButton("提交删除")
        clear_sel_btn = QPushButton("清除选择")
        reset_btn = QPushButton("重置视角")
        undo_btn = QPushButton("Undo")
        redo_btn = QPushButton("Redo")

        load_btn.clicked.connect(self.load)
        save_ascii_btn.clicked.connect(self.save_ascii)
        save_bin_btn.clicked.connect(self.save_binary)
        demo_btn.clicked.connect(self.demo)
        mark_delete_btn.clicked.connect(self.mark_delete_selected)
        commit_delete_btn.clicked.connect(self.commit_delete)
        clear_sel_btn.clicked.connect(self.clear_selection)
        reset_btn.clicked.connect(self.reset_view)
        undo_btn.clicked.connect(self.undo)
        redo_btn.clicked.connect(self.redo)

        self.point_size_spin = QSpinBox()
        self.point_size_spin.setRange(1, 20)
        self.point_size_spin.setValue(3)
        self.point_size_spin.valueChanged.connect(self.on_point_size_changed)

        self.use_preview_check = QCheckBox("启用 LOD/预览")
        self.use_preview_check.setChecked(True)
        self.use_preview_check.toggled.connect(self.on_preview_settings_changed)

        self.move_x = QDoubleSpinBox()
        self.move_y = QDoubleSpinBox()
        self.move_z = QDoubleSpinBox()
        for spin in (self.move_x, self.move_y, self.move_z):
            spin.setRange(-1000.0, 1000.0)
            spin.setDecimals(4)
            spin.setSingleStep(0.01)
        move_btn = QPushButton("移动选中点")
        move_btn.clicked.connect(self.move_selected)

        point_mode = QRadioButton("点选")
        box_mode = QRadioButton("框选")
        brush_mode = QRadioButton("刷选")
        point_mode.setChecked(True)
        mode_group = QButtonGroup(self)
        for i, btn in enumerate((point_mode, box_mode, brush_mode)):
            mode_group.addButton(btn, i)
        mode_group.idClicked.connect(self.on_mode_changed)

        self.status = QLabel("就绪")
        self.status.setWordWrap(True)

        side.addWidget(title)
        side.addWidget(load_btn)
        side.addWidget(save_ascii_btn)
        side.addWidget(save_bin_btn)
        side.addWidget(demo_btn)
        side.addWidget(mark_delete_btn)
        side.addWidget(commit_delete_btn)
        side.addWidget(undo_btn)
        side.addWidget(redo_btn)
        side.addWidget(clear_sel_btn)
        side.addWidget(reset_btn)
        side.addWidget(QLabel("选择模式"))
        side.addWidget(point_mode)
        side.addWidget(box_mode)
        side.addWidget(brush_mode)
        side.addWidget(QLabel("点大小"))
        side.addWidget(self.point_size_spin)
        side.addWidget(self.use_preview_check)
        side.addWidget(QLabel("dx"))
        side.addWidget(self.move_x)
        side.addWidget(QLabel("dy"))
        side.addWidget(self.move_y)
        side.addWidget(QLabel("dz"))
        side.addWidget(self.move_z)
        side.addWidget(move_btn)
        side.addStretch(1)
        side.addWidget(QLabel("操作说明：\n左键按当前模式选择\n右键拖拽旋转\n滚轮缩放\n删除默认只打标记，点击提交删除再压缩数组"))
        side.addWidget(self.status)

        layout.addLayout(side, 0)
        layout.addWidget(self.viewer, 1)
        self.setCentralWidget(root)

    def set_status(self, text: str) -> None:
        self.status.setText(text)

    # command-based history: use push_command when making changes

    def on_mode_changed(self, idx: int) -> None:
        self.viewer.set_selection_mode(["point", "box", "brush"][idx])
        self.set_status(f"选择模式: {['点选', '框选', '刷选'][idx]}")

    def on_preview_settings_changed(self) -> None:
        self.viewer.set_preview_enabled(self.use_preview_check.isChecked())
        self.set_status(f"LOD/预览: {'开启' if self.use_preview_check.isChecked() else '关闭'}")

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
                f"已导入: {Path(path).name}，原始点数: {self.model.count}，存活点数: {self.model.alive_count}，范围: "
                f"x[{mn[0]:.3f},{mx[0]:.3f}] y[{mn[1]:.3f},{mx[1]:.3f}] z[{mn[2]:.3f},{mx[2]:.3f}]"
            )
        except Exception as e:
            QMessageBox.critical(self, "导入失败", str(e))
            self.set_status(f"导入失败: {e}")

    def save_ascii(self) -> None:
        if self.model.alive_count == 0:
            self.set_status("没有可导出的点云")
            return
        default_name = str(Path(self.model.current_path).with_suffix(".ply") if self.model.current_path else "point_cloud.ply")
        path, _ = QFileDialog.getSaveFileName(self, "Save ASCII", default_name, "PointCloud (*.ply *.xyz *.txt)")
        if not path:
            return
        try:
            PointCloudIO.save_ascii(path, self.model.positions[~self.model.deleted_mask], self.model.colors[~self.model.deleted_mask])
            self.set_status(f"已导出: {Path(path).name}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))
            self.set_status(f"导出失败: {e}")

    def save_binary(self) -> None:
        if self.model.alive_count == 0:
            self.set_status("没有可导出的点云")
            return
        default_name = str(Path(self.model.current_path).with_suffix(".ply") if self.model.current_path else "point_cloud_binary.ply")
        path, _ = QFileDialog.getSaveFileName(self, "Save Binary PLY", default_name, "PLY (*.ply)")
        if not path:
            return
        try:
            PointCloudIO.save_ply_binary(path, self.model.positions[~self.model.deleted_mask], self.model.colors[~self.model.deleted_mask])
            self.set_status(f"已导出 Binary PLY: {Path(path).name}")
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

    def mark_delete_selected(self) -> None:
        removed, indices = self.model.mark_selected_deleted()
        if removed > 0:
            # record command so it can be undone (restore deleted indices)
            self.history.push_command({"type": "mark_delete", "indices": indices.copy()})
            self.viewer.fit_camera_to_model()
            # notify viewer to remove soft-deleted from render cache
            try:
                self.viewer.update_after_soft_delete()
            except Exception:
                self.viewer.mark_model_dirty()
        self.set_status(f"已标记删除点数: {removed}，当前存活点数: {self.model.alive_count}")

    def commit_delete(self) -> None:
        removed, old_pos, old_col, old_deleted = self.model.compact_deleted()
        if removed > 0:
            # save a restore_state command so commit can be undone
            self.history.push_command({
                "type": "restore_state",
                "positions": old_pos,
                "colors": old_col,
                "deleted_mask": old_deleted,
            })
            self.viewer.fit_camera_to_model()
            self.viewer.mark_model_dirty()
        self.set_status(f"已提交删除，实际移除点数: {removed}，当前点数: {self.model.count}")

    def clear_selection(self) -> None:
        self.model.clear_selection()
        self.viewer.gpu_dirty = True
        self.viewer.update()
        self.set_status("已清除选择")

    def move_selected(self) -> None:
        dx = self.move_x.value()
        dy = self.move_y.value()
        dz = self.move_z.value()
        moved, indices = self.model.move_selected(dx, dy, dz)
        if moved > 0:
            self.history.push_command({
                "type": "move",
                "indices": indices.copy(),
                "delta": np.array([dx, dy, dz], dtype=np.float32),
            })
            self.viewer.mark_model_dirty()
        self.set_status(f"已移动点数: {moved}")

    def reset_view(self) -> None:
        self.viewer.reset_view()
        self.set_status("已重置视角")

    def undo(self) -> None:
        if self.history.undo(self.model):
            self.viewer.fit_camera_to_model()
            self.viewer.mark_model_dirty()
            self.set_status("Undo 成功")
        else:
            self.set_status("没有可撤销操作")

    def redo(self) -> None:
        if self.history.redo(self.model):
            self.viewer.fit_camera_to_model()
            self.viewer.mark_model_dirty()
            self.set_status("Redo 成功")
        else:
            self.set_status("没有可重做操作")
