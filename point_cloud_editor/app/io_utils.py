from __future__ import annotations

from pathlib import Path
import numpy as np
from plyfile import PlyData, PlyElement


class PointCloudIO:
    @staticmethod
    def load_xyz(path: str) -> tuple[np.ndarray, np.ndarray]:
        pts, cols = [], []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                vals = line.split()
                if len(vals) < 3:
                    continue
                x, y, z = map(float, vals[:3])
                pts.append([x, y, z])
                if len(vals) >= 6:
                    r, g, b = map(float, vals[3:6])
                    if max(r, g, b) > 1.0:
                        r /= 255.0
                        g /= 255.0
                        b /= 255.0
                    cols.append([r, g, b])
                else:
                    cols.append([0.2, 0.8, 1.0])
        if not pts:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
        return np.asarray(pts, dtype=np.float32), np.asarray(cols, dtype=np.float32)

    @staticmethod
    def load_ply(path: str) -> tuple[np.ndarray, np.ndarray]:
        ply = PlyData.read(path)
        if "vertex" not in ply:
            raise ValueError("PLY 文件中没有 vertex")
        vertex = ply["vertex"]
        names = set(vertex.data.dtype.names or [])
        if not {"x", "y", "z"}.issubset(names):
            raise ValueError("PLY 顶点缺少 x/y/z 字段")

        pos = np.column_stack([
            np.asarray(vertex["x"], dtype=np.float32),
            np.asarray(vertex["y"], dtype=np.float32),
            np.asarray(vertex["z"], dtype=np.float32),
        ])
        valid = np.isfinite(pos).all(axis=1)
        pos = pos[valid]

        if {"red", "green", "blue"}.issubset(names):
            col = np.column_stack([
                np.asarray(vertex["red"], dtype=np.float32),
                np.asarray(vertex["green"], dtype=np.float32),
                np.asarray(vertex["blue"], dtype=np.float32),
            ])
            col = col[valid]
            if col.size and float(col.max()) > 1.0:
                col /= 255.0
        else:
            col = np.ones_like(pos, dtype=np.float32) * np.array([0.2, 0.8, 1.0], dtype=np.float32)

        return np.ascontiguousarray(pos, dtype=np.float32), np.ascontiguousarray(np.clip(col, 0.0, 1.0), dtype=np.float32)

    @staticmethod
    def save_xyz(path: str, positions: np.ndarray, colors: np.ndarray) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for p, c in zip(positions, colors):
                rgb = np.clip(np.round(c * 255.0), 0, 255).astype(np.int32)
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {rgb[0]} {rgb[1]} {rgb[2]}\n")

    @staticmethod
    def save_ply_ascii(path: str, positions: np.ndarray, colors: np.ndarray) -> None:
        rgb = np.clip(np.round(colors * 255.0), 0, 255).astype(np.uint8)
        with open(path, "w", encoding="utf-8") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {positions.shape[0]}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for p, c in zip(positions, rgb):
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")

    @staticmethod
    def save_ply_binary(path: str, positions: np.ndarray, colors: np.ndarray) -> None:
        rgb = np.clip(np.round(colors * 255.0), 0, 255).astype(np.uint8)
        vertex = np.empty(
            positions.shape[0],
            dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
        )
        vertex["x"] = positions[:, 0]
        vertex["y"] = positions[:, 1]
        vertex["z"] = positions[:, 2]
        vertex["red"] = rgb[:, 0]
        vertex["green"] = rgb[:, 1]
        vertex["blue"] = rgb[:, 2]
        PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(path)

    @staticmethod
    def save_ascii(path: str, positions: np.ndarray, colors: np.ndarray) -> None:
        ext = Path(path).suffix.lower()
        if ext == ".ply":
            PointCloudIO.save_ply_ascii(path, positions, colors)
            return
        if ext in {".xyz", ".txt"}:
            PointCloudIO.save_xyz(path, positions, colors)
            return
        raise ValueError("Unsupported format for ASCII save")

    @staticmethod
    def load(path: str) -> tuple[np.ndarray, np.ndarray]:
        ext = Path(path).suffix.lower()
        if ext in {".xyz", ".txt"}:
            return PointCloudIO.load_xyz(path)
        if ext == ".ply":
            return PointCloudIO.load_ply(path)
        raise ValueError("Unsupported format")

    @staticmethod
    def save(path: str, positions: np.ndarray, colors: np.ndarray) -> None:
        ext = Path(path).suffix.lower()
        if ext in {".xyz", ".txt"}:
            PointCloudIO.save_xyz(path, positions, colors)
            return
        if ext == ".ply":
            PointCloudIO.save_ply(path, positions, colors)
            return
        raise ValueError("Unsupported format")
