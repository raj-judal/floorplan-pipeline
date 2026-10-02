"""Load any supported capture, fuse it, and print a quick summary.

Usage: python tools/inspect_capture.py PATH/TO/CAPTURE [--stride 10] [--save out.ply]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import open3d as o3d

from floorplan.geometry.fusion import fuse, height_histogram_peaks
from floorplan.io import load_capture

ap = argparse.ArgumentParser()
ap.add_argument("capture", type=Path)
ap.add_argument("--stride", type=int, default=10)
ap.add_argument("--save", type=Path)
a = ap.parse_args()

t0 = time.time()
cap = load_capture(a.capture)
print(f"{cap.name}: format={cap.source_format}, frames={len(cap.frames)}, up_axis={cap.up_axis}")
for note in cap.notes:
    print("  note:", note)
centres = np.array([f.T_wc[:3, 3] for f in cap.frames])
print("  camera travel span (m):", np.round(np.ptp(centres, axis=0), 2))

pc = fuse(cap, stride=a.stride)
pts = np.asarray(pc.points)
print(f"  fused {len(pts):,} points (stride {a.stride}, 1 cm voxels) in {time.time() - t0:.1f}s")
print("  strongest horizontal surfaces (height m, share of points):",
      [(round(h, 3), round(s, 3)) for h, s in height_histogram_peaks(pts, cap.up_axis, cap.up_sign)])
if a.save:
    o3d.io.write_point_cloud(str(a.save), pc)
    print("  saved", a.save)
