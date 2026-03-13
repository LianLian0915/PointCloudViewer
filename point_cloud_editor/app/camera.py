from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Camera:
    yaw: float = 0.0
    pitch: float = 0.0
    dist: float = 3.0
    center: np.ndarray = field(default_factory=lambda: np.zeros((3,), dtype=np.float32))
