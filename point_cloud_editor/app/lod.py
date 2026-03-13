from __future__ import annotations

import numpy as np


class LODBuilder:
    @staticmethod
    def voxel_sample_indices(points: np.ndarray, voxel: float, max_points: int | None = None) -> np.ndarray:
        if points.shape[0] == 0:
            return np.zeros((0,), dtype=np.int32)
        if voxel <= 0:
            idx = np.arange(points.shape[0], dtype=np.int32)
        else:
            scaled = np.floor(points / voxel).astype(np.int64)
            _, idx = np.unique(scaled, axis=0, return_index=True)
            idx = np.sort(idx.astype(np.int32))
        if max_points is not None and idx.shape[0] > max_points:
            step = max(1, idx.shape[0] // max_points)
            idx = idx[::step][:max_points]
        return np.ascontiguousarray(idx, dtype=np.int32)

    @staticmethod
    def build_lod_levels(points: np.ndarray) -> dict[str, np.ndarray]:
        n = points.shape[0]
        levels: dict[str, np.ndarray] = {}
        levels["full"] = np.arange(n, dtype=np.int32)
        levels["medium"] = LODBuilder.voxel_sample_indices(points, voxel=0.003, max_points=min(500_000, n))
        levels["preview"] = LODBuilder.voxel_sample_indices(points, voxel=0.008, max_points=min(150_000, n))
        return levels

    @staticmethod
    def choose_level(levels: dict[str, np.ndarray], point_count: int, camera_dist: float) -> np.ndarray:
        if point_count <= 150_000:
            return levels["full"]
        if camera_dist > 8.0:
            return levels["preview"]
        if camera_dist > 3.5:
            return levels["medium"]
        return levels["full"]
