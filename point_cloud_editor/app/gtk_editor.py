from __future__ import annotations

import numpy as np
import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, Pango

from app.history import HistoryStack
from app.io_utils import PointCloudIO
from app.model import PointCloudModel
from app.gtk_viewer import GLViewer


class Editor(Gtk.Window):
    SIDE_PANEL_WIDTH = 260

    def __init__(self) -> None:
        super().__init__()
        self.set_title("Point Cloud Editor (GTK3 + ModernGL)")
        self.model = PointCloudModel()
        self.history = HistoryStack(max_size=50)

        root = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        self.add(root)

        self.viewer = GLViewer(self.model, self.set_status)

        side = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        side.set_size_request(self.SIDE_PANEL_WIDTH, -1)
        side.set_hexpand(False)
        title = Gtk.Label(label="点云编辑查看器")
        title.set_markup("<span size='14000'><b>点云编辑查看器</b></span>")
        title.set_xalign(0)

        load_btn = Gtk.Button(label="导入")
        save_ascii_btn = Gtk.Button(label="导出 ASCII PLY/XYZ")
        save_bin_btn = Gtk.Button(label="导出 Binary PLY")
        demo_btn = Gtk.Button(label="生成示例点云")
        mark_delete_btn = Gtk.Button(label="标记删除选中点")
        commit_delete_btn = Gtk.Button(label="提交删除")
        clear_sel_btn = Gtk.Button(label="清除选择")
        reset_btn = Gtk.Button(label="重置视角")
        undo_btn = Gtk.Button(label="Undo")
        redo_btn = Gtk.Button(label="Redo")

        load_btn.connect("clicked", lambda w: self.load())
        save_ascii_btn.connect("clicked", lambda w: self.save_ascii())
        save_bin_btn.connect("clicked", lambda w: self.save_binary())
        demo_btn.connect("clicked", lambda w: self.demo())
        mark_delete_btn.connect("clicked", lambda w: self.mark_delete_selected())
        commit_delete_btn.connect("clicked", lambda w: self.commit_delete())
        clear_sel_btn.connect("clicked", lambda w: self.clear_selection())
        reset_btn.connect("clicked", lambda w: self.reset_view())
        undo_btn.connect("clicked", lambda w: self.undo())
        redo_btn.connect("clicked", lambda w: self.redo())

        adj = Gtk.Adjustment(value=3, lower=1, upper=20, step_increment=1)
        self.point_size_spin = Gtk.SpinButton(adjustment=adj, climb_rate=1, digits=0)
        self.point_size_spin.connect("value-changed", lambda w: self.on_point_size_changed(w.get_value_as_int()))

        self.use_preview_check = Gtk.CheckButton(label="启用 LOD/预览")
        self.use_preview_check.set_active(True)
        self.use_preview_check.connect("toggled", lambda w: self.on_preview_settings_changed())

        move_adj = Gtk.Adjustment(value=0.0, lower=-1000.0, upper=1000.0, step_increment=0.01)
        self.move_x = Gtk.SpinButton(adjustment=move_adj, climb_rate=0.1, digits=4)
        self.move_y = Gtk.SpinButton(adjustment=Gtk.Adjustment(value=0.0, lower=-1000.0, upper=1000.0, step_increment=0.01), climb_rate=0.1, digits=4)
        self.move_z = Gtk.SpinButton(adjustment=Gtk.Adjustment(value=0.0, lower=-1000.0, upper=1000.0, step_increment=0.01), climb_rate=0.1, digits=4)
        move_btn = Gtk.Button(label="移动选中点")
        move_btn.connect("clicked", lambda w: self.move_selected())

        # radio buttons
        point_mode = Gtk.RadioButton.new_with_label_from_widget(None, "点选")
        box_mode = Gtk.RadioButton.new_with_label_from_widget(point_mode, "框选")
        brush_mode = Gtk.RadioButton.new_with_label_from_widget(point_mode, "刷选")
        point_mode.connect("toggled", lambda w: self.on_mode_changed(0) if w.get_active() else None)
        box_mode.connect("toggled", lambda w: self.on_mode_changed(1) if w.get_active() else None)
        brush_mode.connect("toggled", lambda w: self.on_mode_changed(2) if w.get_active() else None)

        self.status = Gtk.Label(label="就绪")
        self.status.set_line_wrap(True)
        self.status.set_line_wrap_mode(Pango.WrapMode.CHAR)
        self.status.set_max_width_chars(24)
        self.status.set_xalign(0)
        self.status.set_selectable(True)

        help_label = Gtk.Label(
            label="操作说明：\n左键按当前模式选择\n右键拖拽旋转\n滚轮缩放\n删除默认只打标记，点击提交删除再压缩数组"
        )
        help_label.set_line_wrap(True)
        help_label.set_line_wrap_mode(Pango.WrapMode.CHAR)
        help_label.set_max_width_chars(24)
        help_label.set_xalign(0)

        side.pack_start(title, False, False, 0)
        for btn in (load_btn, save_ascii_btn, save_bin_btn, demo_btn, mark_delete_btn, commit_delete_btn, undo_btn, redo_btn, clear_sel_btn, reset_btn):
            side.pack_start(btn, False, False, 0)
        side.pack_start(Gtk.Label(label="选择模式"), False, False, 0)
        side.pack_start(point_mode, False, False, 0)
        side.pack_start(box_mode, False, False, 0)
        side.pack_start(brush_mode, False, False, 0)
        side.pack_start(Gtk.Label(label="点大小"), False, False, 0)
        side.pack_start(self.point_size_spin, False, False, 0)
        side.pack_start(self.use_preview_check, False, False, 0)
        side.pack_start(Gtk.Label(label="dx"), False, False, 0)
        side.pack_start(self.move_x, False, False, 0)
        side.pack_start(Gtk.Label(label="dy"), False, False, 0)
        side.pack_start(self.move_y, False, False, 0)
        side.pack_start(Gtk.Label(label="dz"), False, False, 0)
        side.pack_start(self.move_z, False, False, 0)
        side.pack_start(move_btn, False, False, 0)
        side.pack_start(Gtk.Box(), True, True, 0)  # spacer
        side.pack_start(help_label, False, False, 0)
        side.pack_start(self.status, False, False, 0)

        root.pack_start(side, False, False, 6)
        root.pack_start(self.viewer, True, True, 6)

    def set_status(self, text: str) -> None:
        self.status.set_text(text)

    def on_mode_changed(self, idx: int) -> None:
        self.viewer.set_selection_mode(["point", "box", "brush"][idx])
        self.set_status(f"选择模式: {['点选','框选','刷选'][idx]}")

    def on_preview_settings_changed(self) -> None:
        self.viewer.set_preview_enabled(self.use_preview_check.get_active())
        self.set_status(f"LOD/预览: {'开启' if self.use_preview_check.get_active() else '关闭'}")

    def on_point_size_changed(self, value: int) -> None:
        self.viewer.point_size = float(value)
        self.viewer.queue_draw()

    def load(self) -> None:
        dialog = Gtk.FileChooserDialog(title="Open", parent=self, action=Gtk.FileChooserAction.OPEN)
        dialog.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_OPEN, Gtk.ResponseType.OK)
        filter_p = Gtk.FileFilter()
        filter_p.set_name("PointCloud")
        filter_p.add_pattern("*.ply")
        filter_p.add_pattern("*.xyz")
        filter_p.add_pattern("*.txt")
        dialog.add_filter(filter_p)
        response = dialog.run()
        path = None
        if response == Gtk.ResponseType.OK:
            path = dialog.get_filename()
        dialog.destroy()
        if not path:
            return
        try:
            pos, col = PointCloudIO.load(path)
            if pos.shape[0] == 0:
                raise ValueError("点云为空")
            self.model.set_data(pos, col, path, validated=True)
            self.viewer.camera.yaw = 0.0
            self.viewer.camera.pitch = 0.0
            self.viewer.fit_camera_to_model()
            self.viewer.mark_model_dirty()
            mn, mx = self.model.bounds()
            self.set_status(
                f"已导入: {path.split('/')[-1]}\n"
                f"原始点数: {self.model.count}\n"
                f"存活点数: {self.model.alive_count}\n"
                f"范围: x[{mn[0]:.3f},{mx[0]:.3f}] "
                f"y[{mn[1]:.3f},{mx[1]:.3f}] "
                f"z[{mn[2]:.3f},{mx[2]:.3f}]"
            )
        except Exception as e:
            dlg = Gtk.MessageDialog(parent=self, flags=0, message_type=Gtk.MessageType.ERROR, buttons=Gtk.ButtonsType.OK, text="导入失败")
            dlg.format_secondary_text(str(e))
            dlg.run()
            dlg.destroy()
            self.set_status(f"导入失败: {e}")

    def save_ascii(self) -> None:
        if self.model.alive_count == 0:
            self.set_status("没有可导出的点云")
            return
        default_name = (self.model.current_path or "point_cloud.ply").rsplit('/', 1)[-1]
        dialog = Gtk.FileChooserDialog(title="Save ASCII", parent=self, action=Gtk.FileChooserAction.SAVE)
        dialog.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_SAVE, Gtk.ResponseType.OK)
        dialog.set_current_name(default_name)
        response = dialog.run()
        path = None
        if response == Gtk.ResponseType.OK:
            path = dialog.get_filename()
        dialog.destroy()
        if not path:
            return
        try:
            PointCloudIO.save_ascii(path, self.model.positions[~self.model.deleted_mask], self.model.colors[~self.model.deleted_mask])
            self.set_status(f"已导出: {path.split('/')[-1]}")
        except Exception as e:
            dlg = Gtk.MessageDialog(parent=self, flags=0, message_type=Gtk.MessageType.ERROR, buttons=Gtk.ButtonsType.OK, text="导出失败")
            dlg.format_secondary_text(str(e))
            dlg.run()
            dlg.destroy()
            self.set_status(f"导出失败: {e}")

    def save_binary(self) -> None:
        if self.model.alive_count == 0:
            self.set_status("没有可导出的点云")
            return
        default_name = (self.model.current_path or "point_cloud_binary.ply").rsplit('/', 1)[-1]
        dialog = Gtk.FileChooserDialog(title="Save Binary PLY", parent=self, action=Gtk.FileChooserAction.SAVE)
        dialog.add_buttons(Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL, Gtk.STOCK_SAVE, Gtk.ResponseType.OK)
        dialog.set_current_name(default_name)
        response = dialog.run()
        path = None
        if response == Gtk.ResponseType.OK:
            path = dialog.get_filename()
        dialog.destroy()
        if not path:
            return
        try:
            PointCloudIO.save_ply_binary(path, self.model.positions[~self.model.deleted_mask], self.model.colors[~self.model.deleted_mask])
            self.set_status(f"已导出 Binary PLY: {path.split('/')[-1]}")
        except Exception as e:
            dlg = Gtk.MessageDialog(parent=self, flags=0, message_type=Gtk.MessageType.ERROR, buttons=Gtk.ButtonsType.OK, text="导出失败")
            dlg.format_secondary_text(str(e))
            dlg.run()
            dlg.destroy()
            self.set_status(f"导出失败: {e}")

    def demo(self) -> None:
        print("[Editor.demo] Loading demo point cloud")
        x = np.linspace(-0.5, 0.5, 200, dtype=np.float32)
        z = np.linspace(-0.5, 0.5, 200, dtype=np.float32)
        gx, gz = np.meshgrid(x, z)
        gy = np.sin((gx + gz) * 10.0) * 0.05
        pos = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.float32)
        col = np.ones_like(pos, dtype=np.float32) * np.array([0.2, 0.8, 1.0], dtype=np.float32)
        print(f"[Editor.demo] Setting data: {pos.shape[0]} points")
        self.model.set_data(pos, col, validated=True)
        print(f"[Editor.demo] Model alive_count: {self.model.alive_count}")
        self.viewer.fit_camera_to_model()
        print(f"[Editor.demo] Camera: center={self.viewer.camera.center}, dist={self.viewer.camera.dist}")
        self.viewer.mark_model_dirty()
        print("[Editor.demo] Model marked dirty, requesting redraw")
        self.viewer.queue_draw()  # explicitly request a redraw
        self.set_status(f"已生成示例点云，原始点数: {self.model.count}")

    def mark_delete_selected(self) -> None:
        removed, indices = self.model.mark_selected_deleted()
        if removed > 0:
            self.history.push_command({"type": "mark_delete", "indices": indices.copy()})
            self.viewer.fit_camera_to_model()
            try:
                self.viewer.update_after_soft_delete()
            except Exception:
                self.viewer.mark_model_dirty()
        self.set_status(f"已标记删除点数: {removed}，当前存活点数: {self.model.alive_count}")

    def commit_delete(self) -> None:
        removed, old_pos, old_col, old_deleted = self.model.compact_deleted()
        if removed > 0:
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
        self.viewer.queue_draw()
        self.set_status("已清除选择")

    def move_selected(self) -> None:
        dx = self.move_x.get_value()
        dy = self.move_y.get_value()
        dz = self.move_z.get_value()
        moved, indices = self.model.move_selected(dx, dy, dz)
        if moved > 0:
            self.history.push_command({
                "type": "move",
                "indices": indices.copy(),
                "delta": np.array([dx, dy, dz], dtype=np.float32),
            })
            self.viewer.update_moved_points(indices)
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
