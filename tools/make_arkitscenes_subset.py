"""Make the ARKitScenes subset used by the benchmark (cross-platform).
usage: python tools/make_arkitscenes_subset.py RAW_CAPTURE_DIR OUT_DIR [--every 6]
Copies every n-th depth frame (sorted by timestamp in the file name), the
matching confidence maps, all intrinsics files and the .traj pose file.
Depth is 60 Hz and poses 10 Hz, so every 6th frame is about one per pose.
"""
import argparse
import shutil
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument("raw", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("--every", type=int, default=6)
a = ap.parse_args()
for sub in ("lowres_depth", "confidence", "lowres_wide_intrinsics"):
    (a.out / sub).mkdir(parents=True, exist_ok=True)
depth = sorted((a.raw / "lowres_depth").glob("*.png"), key=lambda p: float(p.stem.split("_")[-1]))
picked = depth[::a.every]
for p in picked:
    shutil.copy2(p, a.out / "lowres_depth" / p.name)
    c = a.raw / "confidence" / p.name
    if c.exists():
        shutil.copy2(c, a.out / "confidence" / p.name)
for p in (a.raw / "lowres_wide_intrinsics").glob("*.pincam"):
    shutil.copy2(p, a.out / "lowres_wide_intrinsics" / p.name)
for p in a.raw.glob("*.traj"):
    shutil.copy2(p, a.out / p.name)
print(f"{len(picked)} of {len(depth)} depth frames -> {a.out}")
