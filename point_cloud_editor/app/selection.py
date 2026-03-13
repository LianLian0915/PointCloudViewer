from __future__ import annotations

import numpy as np


def box_select_indices(screen_x: np.ndarray, screen_y: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    xmin, xmax = min(x1, x2), max(x1, x2)
    ymin, ymax = min(y1, y2), max(y1, y2)
    mask = (
        (screen_x >= xmin) &
        (screen_x <= xmax) &
        (screen_y >= ymin) &
        (screen_y <= ymax)
    )
    return np.where(mask)[0]


def brush_select_indices(screen_x: np.ndarray, screen_y: np.ndarray, cx: float, cy: float, radius: float) -> np.ndarray:
    d2 = (screen_x - cx) ** 2 + (screen_y - cy) ** 2
    return np.where(d2 <= radius * radius)[0]
