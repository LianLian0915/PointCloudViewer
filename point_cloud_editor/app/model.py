from __future__ import annotations

import numpy as np


class PointCloudModel:
    def __init__(self) -> None:
        self.positions = np.zeros((0, 3), dtype=np.float32)
        self.colors = np.zeros((0, 3), dtype=np.float32)
        self.selected = np.zeros((0,), dtype=bool)
        self.deleted_mask = np.zeros((0,), dtype=bool)
        self.current_path: str = ""

    @property
    def count(self) -> int:
        return int(self.positions.shape[0])

    @property
    def alive_count(self) -> int:
        if self.deleted_mask.size == 0:
            return self.count
        return int(np.count_nonzero(~self.deleted_mask))

    def clear(self) -> None:
        self.positions = np.zeros((0, 3), dtype=np.float32)
        self.colors = np.zeros((0, 3), dtype=np.float32)
        self.selected = np.zeros((0,), dtype=bool)
        self.deleted_mask = np.zeros((0,), dtype=bool)
        self.current_path = ""

    def restore_state(self, pos: np.ndarray, col: np.ndarray, deleted_mask: np.ndarray) -> None:
        self.positions = np.ascontiguousarray(pos, dtype=np.float32)
        self.colors = np.ascontiguousarray(col, dtype=np.float32)
        self.deleted_mask = np.ascontiguousarray(deleted_mask, dtype=bool)
        self.selected = np.zeros((self.positions.shape[0],), dtype=bool)

    def set_data(self, pos: np.ndarray, col: np.ndarray, path: str = "", validated: bool = False) -> None:
        pos = np.asarray(pos, dtype=np.float32).reshape(-1, 3)
        col = np.asarray(col, dtype=np.float32).reshape(-1, 3)
        if pos.shape[0] != col.shape[0]:
            raise ValueError("positions 和 colors 数量不一致")
        if validated:
            self.positions = np.ascontiguousarray(pos, dtype=np.float32)
            self.colors = np.ascontiguousarray(col, dtype=np.float32)
        else:
            valid = np.isfinite(pos).all(axis=1)
            self.positions = np.ascontiguousarray(pos[valid], dtype=np.float32)
            self.colors = np.ascontiguousarray(np.clip(col[valid], 0.0, 1.0), dtype=np.float32)
        self.selected = np.zeros((self.positions.shape[0],), dtype=bool)
        self.deleted_mask = np.zeros((self.positions.shape[0],), dtype=bool)
        self.current_path = path

    def alive_indices(self) -> np.ndarray:
        if self.count == 0:
            return np.zeros((0,), dtype=np.int32)
        return np.where(~self.deleted_mask)[0].astype(np.int32)

    def alive_positions(self) -> np.ndarray:
        return np.ascontiguousarray(self.positions[~self.deleted_mask], dtype=np.float32)

    def alive_colors(self) -> np.ndarray:
        return np.ascontiguousarray(self.colors[~self.deleted_mask], dtype=np.float32)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        alive = ~self.deleted_mask
        if self.count == 0 or not np.any(alive):
            z = np.zeros((3,), dtype=np.float32)
            return z, z
        pts = self.positions[alive]
        return pts.min(axis=0), pts.max(axis=0)

    def clear_selection(self) -> None:
        if self.selected.size:
            self.selected[:] = False

    def selected_count(self) -> int:
        return int(np.count_nonzero(self.selected & ~self.deleted_mask))

    def select_raw_index(self, idx: int) -> None:
        self.clear_selection()
        if 0 <= idx < self.count and not self.deleted_mask[idx]:
            self.selected[idx] = True

    def select_raw_indices(self, indices: np.ndarray, additive: bool = False) -> int:
        if not additive:
            self.clear_selection()
        valid = indices[(indices >= 0) & (indices < self.count)]
        valid = valid[~self.deleted_mask[valid]]
        if valid.size > 0:
            self.selected[valid] = True
        return int(valid.size)

    def mark_selected_deleted(self) -> tuple[int, np.ndarray]:
        mask = self.selected & ~self.deleted_mask
        removed = int(np.count_nonzero(mask))
        indices = np.where(mask)[0].astype(np.int32)
        if removed > 0:
            self.deleted_mask[mask] = True
            self.selected[:] = False
        return removed, indices

    def move_selected(self, dx: float, dy: float, dz: float) -> tuple[int, np.ndarray]:
        mask = self.selected & ~self.deleted_mask
        moved = int(np.count_nonzero(mask))
        indices = np.where(mask)[0].astype(np.int32)
        if moved > 0:
            self.positions[mask] += np.array([dx, dy, dz], dtype=np.float32)
        return moved, indices

    def compact_deleted(self) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        if self.count == 0:
            return 0, self.positions.copy(), self.colors.copy(), self.deleted_mask.copy()
        keep = ~self.deleted_mask
        removed = int(np.count_nonzero(self.deleted_mask))
        old_pos = self.positions.copy()
        old_col = self.colors.copy()
        old_deleted = self.deleted_mask.copy()
        self.positions = np.ascontiguousarray(self.positions[keep], dtype=np.float32)
        self.colors = np.ascontiguousarray(self.colors[keep], dtype=np.float32)
        self.selected = np.zeros((self.positions.shape[0],), dtype=bool)
        self.deleted_mask = np.zeros((self.positions.shape[0],), dtype=bool)
        return removed, old_pos, old_col, old_deleted

    def apply_inverse_command(self, cmd: dict) -> dict | None:
        typ = cmd.get("type")
        if typ == "move":
            indices = np.asarray(cmd.get("indices", []), dtype=np.int64)
            delta = np.asarray(cmd.get("delta", [0.0, 0.0, 0.0]), dtype=np.float32)
            valid = indices[(indices >= 0) & (indices < self.count)]
            if valid.size:
                self.positions[valid] -= delta
            return {"type": "move", "indices": valid.astype(np.int32), "delta": -delta}
        if typ == "mark_delete":
            indices = np.asarray(cmd.get("indices", []), dtype=np.int64)
            valid = indices[(indices >= 0) & (indices < self.count)]
            if valid.size:
                self.deleted_mask[valid] = False
            return {"type": "unmark_delete", "indices": valid.astype(np.int32)}
        if typ == "unmark_delete":
            indices = np.asarray(cmd.get("indices", []), dtype=np.int64)
            valid = indices[(indices >= 0) & (indices < self.count)]
            if valid.size:
                self.deleted_mask[valid] = True
                self.selected[valid] = False
            return {"type": "mark_delete", "indices": valid.astype(np.int32)}
        if typ == "restore_state":
            old_pos = self.positions.copy()
            old_col = self.colors.copy()
            old_deleted = self.deleted_mask.copy()
            self.restore_state(cmd["positions"], cmd["colors"], cmd["deleted_mask"])
            return {
                "type": "restore_state",
                "positions": old_pos,
                "colors": old_col,
                "deleted_mask": old_deleted,
            }
        return None
