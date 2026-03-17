#!/usr/bin/env python3
"""
Performance test for point cloud interaction.
Generates point clouds of various sizes and measures:
- Initial load time
- Render cache rebuild time
- LOD selection time
- KD-tree construction time (if applicable)
"""
import sys
import os
import time
import numpy as np

# Add the point_cloud_editor directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'point_cloud_editor'))

import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk

from app.model import PointCloudModel
from app.lod import LODBuilder
from app.camera import Camera

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available, KD-tree performance not tested")


def generate_point_cloud(num_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic point cloud (sin wave surface)."""
    grid_size = int(np.sqrt(num_points))
    x = np.linspace(-1, 1, grid_size, dtype=np.float32)
    z = np.linspace(-1, 1, grid_size, dtype=np.float32)
    gx, gz = np.meshgrid(x, z)
    gy = np.sin((gx + gz) * 5.0) * 0.2
    positions = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)[:num_points].astype(np.float32)
    colors = np.ones_like(positions) * np.array([0.2, 0.8, 1.0], dtype=np.float32)
    return positions, colors


def benchmark_point_cloud(num_points: int, interactive: bool = False) -> dict:
    """Benchmark point cloud loading and rendering."""
    results = {
        "num_points": num_points,
        "interactive": interactive,
    }
    
    print(f"\n{'='*60}")
    print(f"Testing {num_points:,} points (interactive={interactive})")
    print(f"{'='*60}")
    
    # Generate data
    t0 = time.time()
    positions, colors = generate_point_cloud(num_points)
    results["generate_time"] = time.time() - t0
    print(f"Generate data: {results['generate_time']*1000:.2f}ms")
    
    # Create model and load data
    t0 = time.time()
    model = PointCloudModel()
    model.set_data(positions, colors)
    results["model_load_time"] = time.time() - t0
    print(f"Model load: {results['model_load_time']*1000:.2f}ms")
    
    # Build LOD levels
    t0 = time.time()
    alive_idx = model.alive_indices()
    alive_points = model.positions[alive_idx]
    lod_levels = LODBuilder.build_lod_levels(alive_points)
    results["lod_build_time"] = time.time() - t0
    print(f"LOD build: {results['lod_build_time']*1000:.2f}ms")
    
    # Choose LOD level
    camera = Camera()
    camera.dist = 3.0
    t0 = time.time()
    if interactive:
        chosen = lod_levels["preview"]
    else:
        chosen = LODBuilder.choose_level(lod_levels, alive_points.shape[0], camera.dist)
    results["lod_choose_time"] = time.time() - t0
    print(f"LOD choose: {results['lod_choose_time']*1000:.4f}ms")
    
    render_indices = alive_idx[chosen]
    results["render_points"] = len(render_indices)
    print(f"Render points: {results['render_points']:,} ({100*results['render_points']/num_points:.1f}%)")
    
    # KD-tree build (skipped in interactive mode)
    if HAS_SCIPY and not interactive:
        render_positions = model.positions[render_indices]
        t0 = time.time()
        try:
            tree = cKDTree(render_positions)
            results["kdtree_build_time"] = time.time() - t0
            print(f"KD-tree build: {results['kdtree_build_time']*1000:.2f}ms")
        except Exception as e:
            print(f"KD-tree build failed: {e}")
            results["kdtree_build_time"] = None
    else:
        results["kdtree_build_time"] = None
        if not interactive:
            print("KD-tree skipped (scipy not available)")
        else:
            print("KD-tree skipped (interactive mode)")
    
    # Estimate frame time
    base_render_ms = 5.0  # base OpenGL render time
    if interactive:
        total_ms = base_render_ms
        note = "interactive (lightweight)"
    else:
        total_ms = base_render_ms + (results["lod_choose_time"] + (results["kdtree_build_time"] or 0)) * 1000
        note = "full rebuild"
    
    fps = 1000.0 / total_ms if total_ms > 0 else 60
    results["estimated_fps"] = fps
    print(f"Estimated frame time: {total_ms:.2f}ms ({fps:.1f} fps) - {note}")
    
    return results


def main():
    print("Point Cloud Performance Benchmark")
    print("="*60)
    
    # Test different point cloud sizes
    test_sizes = [10_000, 50_000, 100_000, 500_000, 1_000_000]
    
    # Only test sizes that fit in available memory
    results_list = []
    for size in test_sizes:
        try:
            # Test both normal and interactive modes
            results = benchmark_point_cloud(size, interactive=False)
            results_list.append(results)
            
            results_i = benchmark_point_cloud(size, interactive=True)
            results_list.append(results_i)
            
        except MemoryError:
            print(f"\nSkipping {size:,} points (out of memory)")
            break
        except Exception as e:
            print(f"\nError testing {size:,} points: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Points':<12} {'Mode':<12} {'LOD%':<8} {'KD-tree':<10} {'Est FPS':<8}")
    print("-"*60)
    for r in results_list:
        mode = "interactive" if r["interactive"] else "normal"
        lod_pct = 100 * r["render_points"] / r["num_points"]
        kdtree = f"{r['kdtree_build_time']*1000:.1f}ms" if r["kdtree_build_time"] else "skip"
        print(f"{r['num_points']:<12,} {mode:<12} {lod_pct:<8.1f}% {kdtree:<10} {r['estimated_fps']:<8.1f}")
    
    print(f"\nRecommendations:")
    print("- Points < 100K:    No LOD needed, full precision always smooth")
    print("- Points 100K-1M:   Enable LOD, interactive preview smooth")
    print("- Points > 1M:      Strong LOD required, consider octree/grid")


if __name__ == "__main__":
    main()
