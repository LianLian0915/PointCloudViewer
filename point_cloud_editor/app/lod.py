from __future__ import annotations

import numpy as np


class LODBuilder:
    @staticmethod
    def stride_sample_indices(count: int, max_points: int) -> np.ndarray:
        if count <= 0:
            return np.zeros((0,), dtype=np.int32)
        if count <= max_points:
            return np.arange(count, dtype=np.int32)
        step = max(1, int(np.ceil(count / max_points)))
        return np.ascontiguousarray(np.arange(0, count, step, dtype=np.int32)[:max_points])

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
        return LODBuilder.build_lod_levels_for_count(points.shape[0])

    @staticmethod
    def build_lod_levels_for_count(n: int) -> dict[str, np.ndarray]:
        levels: dict[str, np.ndarray] = {}
        levels["full"] = np.arange(n, dtype=np.int32)
        # Large point clouds must build LOD immediately after import. A stride sample is
        # much cheaper than voxel hashing for multi-million point clouds and keeps
        # interaction responsive while preserving the original full-resolution data.
        levels["medium"] = LODBuilder.stride_sample_indices(n, min(500_000, n))
        levels["preview"] = LODBuilder.stride_sample_indices(n, min(150_000, n))
        return levels

    @staticmethod
    def choose_level(levels: dict[str, np.ndarray], point_count: int, camera_dist: float) -> np.ndarray:
        # Keep the complete cloud visible when the view is idle. Preview LOD is
        # selected explicitly by GLViewer during interactive camera drags.
        return levels["full"]
