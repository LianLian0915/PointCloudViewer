from __future__ import annotations

import numpy as np


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
