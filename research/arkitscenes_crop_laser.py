"""Crop the merged laser cloud to the captured room using benchmarks/arkitscenes_421383/laser_crop.json.

usage: python research/arkitscenes_crop_laser.py MERGED_10MM.ply OUT_CROP.ply
"""
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

cfg = json.load(open(Path(__file__).resolve().parents[1] / "benchmarks" / "arkitscenes_421383" / "laser_crop.json"))
pc = o3d.io.read_point_cloud(sys.argv[1])
P = np.asarray(pc.points)
m = (np.all((P[:, :2] > cfg["crop_xy_min"]) & (P[:, :2] < cfg["crop_xy_max"]), axis=1)
     & (P[:, 2] > cfg["z_min"]) & (P[:, 2] < cfg["z_max"]))
o3d.io.write_point_cloud(sys.argv[2], pc.select_by_index(np.nonzero(m)[0]))
print(f"kept {m.sum():,} of {len(P):,} points -> {sys.argv[2]}")
