"""Floor, ceiling and wall planes from a fused point cloud.

Pipeline:
  1. gravity_align: rotate so the capture's up axis becomes +z.
  2. horizontal_planes: floor = lowest well-supported horizontal surface,
     ceiling = highest one at least MIN_ROOM_HEIGHT above the floor, each
     refined by a robust least-squares plane fit (tilt allowed).
  3. manhattan_yaw + wall_planes: dominant wall direction from normal angles,
     then wall offsets as peaks of a 1D histogram along each axis.

Every surface reports its support (point count) and spatial coverage so the
caller can decide how much to trust it, rather than returning bare numbers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d

MIN_ROOM_HEIGHT = 1.9      # m, ceiling must be at least this far above the floor
COVER_CELL = 0.10          # m, grid cell for coverage estimates


@dataclass
class HPlane:
    """Horizontal plane z = a*x + b*y + c (z up)."""
    a: float
    b: float
    c: float
    n_inliers: int
    residual_std: float     # m
    cells: set              # occupied COVER_CELL grid cells

    def z_at(self, x, y):
        return self.a * x + self.b * y + self.c

    @property
    def tilt_deg(self) -> float:
        return float(np.degrees(np.arctan(np.hypot(self.a, self.b))))


@dataclass
class Wall:
    axis: int               # 0: wall normal along x', 1: along y' (yaw-aligned frame)
    offset: float           # m, coordinate of the wall along its axis
    n_inliers: int
    extent: tuple           # (min, max) along the other horizontal axis
    residual_std: float
    upper_share: float = 0.0  # share of inliers more than 1.8 m above the floor
    structural: bool = False  # reaches the upper zone: a wall, not furniture

    @property
    def length_supported(self) -> float:
        return self.extent[1] - self.extent[0]


def gravity_align(up_axis: int, up_sign: float = 1.0) -> np.ndarray:
    """Rotation taking world coordinates to a frame with +z up."""
    if up_axis == 2:
        R = np.eye(3)
    elif up_axis == 1:   # y up -> z up: (x, y, z) -> (x, -z, y)
        R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
    else:                # x up -> z up
        R = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], float)
    if up_sign < 0:
        R = np.diag([1, -1, -1]) @ R
    return R


def with_normals(pc: o3d.geometry.PointCloud, voxel: float = 0.02):
    d = pc.voxel_down_sample(voxel)
    d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3, max_nn=30))
    return np.asarray(d.points), np.asarray(d.normals)


def _cells(xy: np.ndarray) -> set:
    return set(map(tuple, np.floor(xy / COVER_CELL).astype(int)))


def _robust_hfit(P: np.ndarray, iters: int = 5):
    keep = np.ones(len(P), bool)
    for _ in range(iters):
        A = np.c_[P[keep, 0], P[keep, 1], np.ones(keep.sum())]
        coef, *_ = np.linalg.lstsq(A, P[keep, 2], rcond=None)
        r = P[:, 2] - (np.c_[P[:, 0], P[:, 1], np.ones(len(P))] @ coef)
        s = 1.4826 * np.median(np.abs(r[keep]))
        keep = np.abs(r) < max(2.5 * s, 0.004)
    return coef, keep, float(np.std(r[keep]))


def _peaks(values, weights=None, bin_m=0.01, min_share=0.01, min_sep=0.15):
    counts, edges = np.histogram(values, bins=np.arange(values.min(), values.max() + bin_m, bin_m),
                                 weights=weights)
    total = counts.sum()
    out = []
    for i in np.argsort(counts)[::-1]:
        if counts[i] < min_share * total:
            break
        c = edges[i] + bin_m / 2
        if all(abs(c - o) >= min_sep for o in out):
            out.append(c)
    return sorted(out)


def horizontal_planes(P: np.ndarray, N: np.ndarray, band: float = 0.03):
    """Return (floor HPlane, ceiling HPlane or None). P, N gravity-aligned (z up)."""
    horiz = np.abs(N[:, 2]) > 0.95
    z = P[horiz, 2]
    peaks = _peaks(z, min_share=0.02)
    if not peaks:
        raise ValueError("no horizontal surface found")

    def fit(z0):
        sel = horiz & (np.abs(P[:, 2] - z0) < band)
        coef, keep, s = _robust_hfit(P[sel])
        Q = P[sel][keep]
        return HPlane(*map(float, coef), int(keep.sum()), s, _cells(Q[:, :2]))

    floor = fit(peaks[0])
    ceil_cands = [p for p in peaks if p > peaks[0] + MIN_ROOM_HEIGHT]
    ceiling = fit(ceil_cands[-1]) if ceil_cands else None
    return floor, ceiling


def manhattan_yaw(N: np.ndarray, P: np.ndarray, floor: HPlane) -> float:
    """Dominant wall direction (radians) from vertical-surface normals, modulo 90 degrees."""
    vert = (np.abs(N[:, 2]) < 0.1) & (P[:, 2] > floor.z_at(P[:, 0], P[:, 1]) + 0.3)
    ang = np.mod(np.arctan2(N[vert, 1], N[vert, 0]), np.pi / 2)
    hist, edges = np.histogram(ang, bins=180, range=(0, np.pi / 2))
    i = int(np.argmax(hist))
    centre = edges[i] + (edges[1] - edges[0]) / 2
    # refine with a circular mean (period 90 degrees) of angles near the peak
    d = np.angle(np.exp(4j * (ang - centre))) / 4
    near = np.abs(d) < np.radians(3)
    return float(centre + np.mean(d[near]))


def yaw_matrix(yaw: float) -> np.ndarray:
    c, s = np.cos(-yaw), np.sin(-yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def wall_planes(P: np.ndarray, N: np.ndarray, floor: HPlane, ceiling: HPlane | None,
                min_inliers: int = 300, min_extent: float = 0.4, band: float = 0.02):
    """Walls in a yaw-aligned frame (walls parallel to x' or y'). Returns list[Wall]."""
    h = P[:, 2] - floor.z_at(P[:, 0], P[:, 1])
    top = (ceiling.z_at(P[:, 0], P[:, 1]) - floor.z_at(P[:, 0], P[:, 1]) - 0.15) if ceiling is not None else 2.6
    zone = (h > 0.3) & (h < top)
    walls = []
    for axis in (0, 1):
        sel = zone & (np.abs(N[:, axis]) > np.cos(np.radians(10)))
        x = P[sel, axis]
        if len(x) < min_inliers:
            continue
        for p in _peaks(x, min_share=0.005, min_sep=0.08):
            m = np.abs(x - p) < band
            if m.sum() < min_inliers:
                continue
            xs = x[m]
            off = float(np.median(xs))
            other = P[sel][m][:, 1 - axis]
            lo, hi = np.percentile(other, [1, 99])
            if hi - lo < min_extent:
                continue
            hs = h[sel][m]
            upper = float(np.mean(hs > 1.8))
            walls.append(Wall(axis, off, int(m.sum()), (float(lo), float(hi)),
                              float(1.4826 * np.median(np.abs(xs - off))), upper, upper >= 0.15))
    return walls
