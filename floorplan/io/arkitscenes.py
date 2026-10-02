"""Loader for ARKitScenes raw captures (used for laser ground truth).

Folder layout: lowres_depth/<vid>_<ts>.png, confidence/<vid>_<ts>.png,
lowres_wide_intrinsics/<vid>_<ts>.pincam, lowres_wide.traj.

Verified facts (see docs/lab_notebook.md):
  - Each .traj line is: timestamp, rotation (axis-angle), translation, and is a
    WORLD-TO-CAMERA transform; ARKitScenes' own loader inverts it.
  - Poses are 10 Hz while depth is 60 Hz, so poses are interpolated to each
    depth timestamp (slerp rotation, linear translation) instead of dropping
    frames without an exact pose.
  - Camera convention is OpenCV; world +z is up.
  - .pincam is "w h fx fy cx cy" at the 256x192 depth resolution.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .capture import Capture, FrameMeta

MAX_POSE_GAP_S = 0.15   # do not interpolate across tracking gaps longer than this


def is_arkitscenes(root: Path) -> bool:
    return (root / "lowres_depth").is_dir() and any(root.glob("*.traj"))


def _ts(path: Path) -> float:
    return float(path.stem.split("_")[-1])


def interpolate_poses(traj: np.ndarray, query_ts: np.ndarray):
    """traj rows: t, rx, ry, rz, tx, ty, tz (world-to-camera).

    Returns (T_wc array Nx4x4, valid mask N). A query is valid when it lies
    between two poses that are at most MAX_POSE_GAP_S apart.
    """
    t = traj[:, 0]
    R_cw = Rotation.from_rotvec(traj[:, 1:4])
    R_wc = R_cw.inv()
    C = -np.einsum("nij,nj->ni", R_wc.as_matrix(), traj[:, 4:7])   # camera centres

    j = np.searchsorted(t, query_ts)            # t[j-1] <= q < t[j]
    inside = (j > 0) & (j < len(t))
    jj = np.clip(j, 1, len(t) - 1)
    gap = t[jj] - t[jj - 1]
    valid = inside & (gap <= MAX_POSE_GAP_S)
    exact = np.isin(query_ts, t)
    valid |= exact

    q = np.clip(query_ts, t[0], t[-1])
    R_q = Slerp(t, R_wc)(q).as_matrix()
    C_q = np.stack([np.interp(q, t, C[:, k]) for k in range(3)], axis=1)
    T = np.tile(np.eye(4), (len(q), 1, 1))
    T[:, :3, :3], T[:, :3, 3] = R_q, C_q
    return T, valid


def load_arkitscenes(root: str | Path) -> Capture:
    root = Path(root)
    traj = np.loadtxt(next(root.glob("*.traj")))
    depth_files = sorted((root / "lowres_depth").glob("*.png"), key=_ts)
    depth_ts = np.array([_ts(p) for p in depth_files])

    pincams = sorted((root / "lowres_wide_intrinsics").glob("*.pincam"), key=_ts)
    pin_ts = np.array([_ts(p) for p in pincams])

    T_all, valid = interpolate_poses(traj, depth_ts)
    frames = []
    for i, (p, ts) in enumerate(zip(depth_files, depth_ts)):
        if not valid[i]:
            continue
        w, h, fx, fy, cx, cy = np.loadtxt(pincams[int(np.argmin(np.abs(pin_ts - ts)))])
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        frames.append(FrameMeta(index=i, timestamp=float(ts), K=K, T_wc=T_all[i],
                                depth_path=p, confidence_path=root / "confidence" / p.name))
    dropped = len(depth_files) - len(frames)
    return Capture(name=root.name, source_format="arkitscenes", root=root, frames=frames,
                   up_axis=2, up_sign=1.0,
                   notes=[f"{len(frames)} of {len(depth_files)} depth frames posed by interpolation "
                          f"({dropped} outside tracked span or across gaps > {MAX_POSE_GAP_S}s)"])
