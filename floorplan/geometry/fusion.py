"""Fuse depth frames into one world-frame point cloud."""
from __future__ import annotations

import numpy as np
import open3d as o3d

from ..io.capture import Capture, backproject


def fuse(capture: Capture, stride: int = 1, min_confidence: int = 2,
         min_range: float = 0.2, max_range: float = 5.0, voxel: float = 0.01,
         range_correction=None, flush_every: int = 50) -> o3d.geometry.PointCloud:
    """Back-project every `stride`-th frame and voxel-downsample as we go.

    Only pixels with confidence >= min_confidence are kept (the bias study
    showed low-confidence pixels carry a larger offset). Downsampling every
    `flush_every` frames keeps memory bounded on long scans.
    """
    cloud = o3d.geometry.PointCloud()
    buf = []
    for n, f in enumerate(capture.frames[::stride]):
        d = capture.read_depth(f)
        mask = (d > min_range) & (d < max_range)
        c = capture.read_confidence(f)
        if c is not None:
            mask &= c >= min_confidence
        W, _, _, _ = backproject(d, f.K, f.T_wc, mask, range_correction)
        buf.append(W)
        if (n + 1) % flush_every == 0:
            cloud += o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(buf)))
            cloud = cloud.voxel_down_sample(voxel)
            buf = []
    if buf:
        cloud += o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(buf)))
    return cloud.voxel_down_sample(voxel)


def height_histogram_peaks(points: np.ndarray, up_axis: int, up_sign: float = 1.0,
                           bin_m: float = 0.01, n: int = 4, min_sep_m: float = 0.15):
    """Strongest horizontal-surface heights along the up axis (quick diagnostic, not the measurement)."""
    h = up_sign * points[:, up_axis]
    counts, edges = np.histogram(h, bins=np.arange(h.min(), h.max() + bin_m, bin_m))
    peaks = []
    for i in np.argsort(counts)[::-1]:
        if all(abs(edges[i] - edges[j]) >= min_sep_m for j in peaks):
            peaks.append(i)
        if len(peaks) == n:
            break
    return sorted((float(edges[i] + bin_m / 2), float(counts[i] / counts.sum())) for i in peaks)
