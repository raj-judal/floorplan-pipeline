"""Merge and downsample ARKitScenes Faro laser scans for one visit.

Usage:
    python downsample_laser.py C:\\arkitscenes_data\\laser_scanner_point_clouds\\421383 --voxel 0.01

Loads each station's PLY one at a time (to keep memory low), voxel-downsamples
it, merges all stations, downsamples again, and writes <visit>_merged_<voxel>.ply
next to the input folder. 1 cm voxels keep far more precision than the 1-2 cm
gates need while shrinking gigabytes to tens of megabytes.
"""
import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

ap = argparse.ArgumentParser()
ap.add_argument("visit_dir", type=Path)
ap.add_argument("--voxel", type=float, default=0.01, help="voxel size in metres")
args = ap.parse_args()

plys = sorted(args.visit_dir.glob("*.ply"))
if not plys:
    raise SystemExit(f"No .ply files in {args.visit_dir}")

merged = o3d.geometry.PointCloud()
for p in plys:
    pc = o3d.io.read_point_cloud(str(p))
    n_in = len(pc.points)
    pc = pc.voxel_down_sample(args.voxel)
    print(f"{p.name}: {n_in:,} -> {len(pc.points):,} points")
    merged += pc
    del pc

merged = merged.voxel_down_sample(args.voxel)
pts = np.asarray(merged.points)
print(f"\nmerged: {len(pts):,} points")
print("bounding box min (m):", np.round(pts.min(0), 3))
print("bounding box max (m):", np.round(pts.max(0), 3))

out = args.visit_dir.parent / f"{args.visit_dir.name}_merged_{int(args.voxel * 1000)}mm.ply"
o3d.io.write_point_cloud(str(out), merged, write_ascii=False, compressed=False)
print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
