"""Gravity from image geometry vs the phone's real gravity (thin tiers).
usage: python research/thin_gravity_check.py STRAY_CAPTURE [STRAY_CAPTURE ...] [--frames 60]
LiDAR depth stands in for the depth model. The starting guess is the nearest
image axis to true up (what image-orientation metadata provides).
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from floorplan.io import load_capture
from floorplan.thin.geometry import estimate_up, points_and_normals
ap = argparse.ArgumentParser()
ap.add_argument("captures", nargs="+", type=Path)
ap.add_argument("--frames", type=int, default=60)
a = ap.parse_args()
AX = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]], float)
for c in a.captures:
    cap = load_capture(c)
    errs = []
    for i in np.linspace(0.05 * len(cap.frames), 0.95 * len(cap.frames), a.frames).astype(int):
        f = cap.frames[i]
        d = cap.read_depth(f)
        d[cap.read_confidence(f) < 2] = 0
        true_up = f.T_wc[:3, :3].T @ np.array([0, 1.0, 0])          # Stray world: +y up
        _, N = points_and_normals(d, f.K)
        up, _ = estimate_up(N, AX[np.argmax(AX @ true_up)])
        errs.append(np.degrees(np.arccos(np.clip(up @ true_up, -1, 1))))
    e = np.array(errs)
    print(f"{c.name}: median {np.median(e):.2f} deg, 90% within {np.percentile(e, 90):.2f} deg, "
          f"worst {e.max():.1f} deg, frames over 5 deg: {(e > 5).sum()} of {len(e)}")
