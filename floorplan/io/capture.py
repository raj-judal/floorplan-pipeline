"""Format-independent capture model.

Every loader (Stray Scanner, ARKitScenes, later video and photo) produces a
Capture: a list of FrameMeta with intrinsics and a camera-to-world pose per
depth frame. Pixel data is read lazily so a 10,000-frame scan never sits in
memory at once.

Conventions, fixed here and nowhere else:
  - Camera frame is OpenCV: x right, y down, z forward.
  - T_wc maps camera coordinates to world coordinates (camera-to-world).
  - K is expressed at the DEPTH image resolution.
  - Depth is metres, float32; 0 means no measurement.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image


@dataclass
class FrameMeta:
    index: int
    timestamp: float
    K: np.ndarray            # 3x3, depth resolution
    T_wc: np.ndarray         # 4x4 camera-to-world
    depth_path: Path
    confidence_path: Optional[Path] = None
    rgb_ref: Optional[str] = None   # image path, or "video.mp4#frame" reference


@dataclass
class Capture:
    name: str
    source_format: str                 # "stray" | "arkitscenes"
    root: Path
    frames: list[FrameMeta]
    up_axis: int                       # world axis pointing up (0, 1 or 2)
    up_sign: float = 1.0
    depth_scale: float = 0.001         # stored units to metres
    notes: list[str] = field(default_factory=list)

    def read_depth(self, f: FrameMeta) -> np.ndarray:
        return np.asarray(Image.open(f.depth_path), dtype=np.float32) * self.depth_scale

    def read_confidence(self, f: FrameMeta) -> Optional[np.ndarray]:
        if f.confidence_path is None or not f.confidence_path.exists():
            return None
        return np.asarray(Image.open(f.confidence_path), dtype=np.uint8)


def backproject(depth: np.ndarray, K: np.ndarray, T_wc: np.ndarray,
                mask: Optional[np.ndarray] = None,
                range_correction: Optional[Callable[[np.ndarray], np.ndarray]] = None):
    """Depth image to world points.

    Returns (world_points Nx3, ranges N, pixel_rows N, pixel_cols N).
    range_correction, if given, maps measured range (m) to an offset (m) that
    is ADDED along each viewing ray (used for the depth bias correction).
    """
    valid = depth > 0
    if mask is not None:
        valid &= mask
    v, u = np.nonzero(valid)
    z = depth[v, u]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    # +0.5: pixel centres, matching how the depth map samples the scene
    P = np.stack([(u + 0.5 - cx) / fx * z, (v + 0.5 - cy) / fy * z, z], axis=1)
    rng = np.linalg.norm(P, axis=1)
    if range_correction is not None:
        P *= (1.0 + range_correction(rng) / rng)[:, None]
        rng = np.linalg.norm(P, axis=1)
    W = P @ T_wc[:3, :3].T + T_wc[:3, 3]
    return W, rng, v, u
