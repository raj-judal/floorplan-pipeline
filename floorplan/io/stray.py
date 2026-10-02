"""Loader for Stray Scanner exports (the format of the company sample data).

Folder layout: depth/NNNNNN.png, confidence/NNNNNN.png, rgb.mp4, odometry.csv,
camera_matrix.csv, imu.csv.

Verified facts (see docs/lab_notebook.md):
  - odometry.csv gives a camera-to-world pose per frame as position + quaternion
    (qx, qy, qz, qw) in the OpenCV camera convention; world +y is up.
  - Intrinsics vary per frame (autofocus) and are given at RGB resolution in
    odometry.csv. camera_matrix.csv holds a single frame's values, so it is
    NOT used.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.transform import Rotation

from .capture import Capture, FrameMeta

RGB_SIZE = (1920, 1440)   # Stray Scanner records RGB at this size; intrinsics refer to it


def is_stray(root: Path) -> bool:
    return (root / "odometry.csv").exists() and (root / "depth").is_dir()


def load_stray(root: str | Path) -> Capture:
    root = Path(root)
    odo = pd.read_csv(root / "odometry.csv", skipinitialspace=True)
    first_depth = sorted((root / "depth").glob("*.png"))[0]
    dw, dh = Image.open(first_depth).size
    sx, sy = dw / RGB_SIZE[0], dh / RGB_SIZE[1]
    if abs(sx - sy) > 1e-6:
        raise ValueError(f"depth size {dw}x{dh} does not match the 4:3 RGB size {RGB_SIZE}")

    rot = Rotation.from_quat(odo[["qx", "qy", "qz", "qw"]].to_numpy()).as_matrix()
    pos = odo[["x", "y", "z"]].to_numpy()
    frames = []
    for i, row in enumerate(odo.itertuples(index=False)):
        fid = int(row.frame)
        K = np.array([[row.fx * sx, 0, row.cx * sx],
                      [0, row.fy * sy, row.cy * sy],
                      [0, 0, 1.0]])
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = rot[i], pos[i]
        frames.append(FrameMeta(
            index=fid, timestamp=float(row.timestamp), K=K, T_wc=T,
            depth_path=root / "depth" / f"{fid:06d}.png",
            confidence_path=root / "confidence" / f"{fid:06d}.png",
            rgb_ref=f"{root / 'rgb.mp4'}#{fid}"))
    n_fx = odo["fx"].nunique()
    return Capture(name=root.name, source_format="stray", root=root, frames=frames,
                   up_axis=1, up_sign=1.0,
                   notes=[f"{len(frames)} frames, {n_fx} distinct focal lengths (per-frame intrinsics used)"])
