"""Per-image geometry for the thin tiers (photos, video): no phone gravity, no poses.
Measured in docs/lab_notebook.md:
  - estimate_up: gravity from surface normals, median 0.5-0.7 deg error
    (LiDAR stand-in depth, 180 frames, 3 captures).
  - camera_height: floor below the camera from the image's own depth; used to
    fix the depth model's scale (corrected scale median 0.998 / 0.988 on two
    captures).
"""
from __future__ import annotations
import numpy as np
SNAP_TOL_DEG = 10
def points_and_normals(depth: np.ndarray, K: np.ndarray, k: int = 3):
    """Camera-frame points and unit normals for pixels whose neighbours k px away are valid.
    Normals use neighbours k pixels apart so pixel-level depth noise does not dominate."""
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    z = depth
    P = np.stack([(u + 0.5 - K[0, 2]) / K[0, 0] * z, (v + 0.5 - K[1, 2]) / K[1, 1] * z, z], -1)
    dx = P[k:-k, 2 * k:] - P[k:-k, :-2 * k]
    dy = P[2 * k:, k:-k] - P[:-2 * k, k:-k]
    n = np.cross(dx, dy)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    valid = ((z[k:-k, k:-k] > 0) & (z[k:-k, 2 * k:] > 0) & (z[k:-k, :-2 * k] > 0)
             & (z[2 * k:, k:-k] > 0) & (z[:-2 * k, k:-k] > 0))
    n = n[valid]
    n *= -np.sign(np.sum(n * P[k:-k, k:-k][valid], axis=1))[:, None]    # face the camera
    return P[k:-k, k:-k][valid], n
def _perp_basis(u):
    a = np.array([1.0, 0, 0]) if abs(u[0]) < 0.9 else np.array([0, 1.0, 0])
    b1 = np.cross(u, a)
    b1 /= np.linalg.norm(b1)
    return b1, np.cross(u, b1)
def estimate_up(normals: np.ndarray, coarse_up, max_from_coarse=60, n_hyp=400, seed=0):
    """Gravity 'up' (camera frame) from surface normals.
    RANSAC: a hypothesis is a normal within max_from_coarse deg of the metadata
    guess; it scores the normals parallel (floor/ceiling) or perpendicular (walls)
    to it. The winner is refined, then the room's three axes are built and the one
    closest to the metadata guess is returned: box rooms look the same under axis
    swaps, the phone's rough orientation does not. Returns (up, share explained)."""
    cu = np.asarray(coarse_up, float)
    cu /= np.linalg.norm(cu)
    if len(normals) < 500:
        return cu, 0.0
    rng = np.random.default_rng(seed)
    sub = normals[rng.choice(len(normals), min(len(normals), 20000), replace=False)]
    d = sub @ cu
    pool = sub[np.abs(d) > np.cos(np.radians(max_from_coarse))]
    if len(pool) < 50:
        return cu, 0.0
    pool = pool * np.sign(pool @ cu)[:, None]
    hyps = pool[rng.choice(len(pool), min(n_hyp, len(pool)), replace=False)]
    c_par, s_perp = np.cos(np.radians(SNAP_TOL_DEG)), np.sin(np.radians(SNAP_TOL_DEG))
    A = np.abs(sub @ hyps.T)
    u = hyps[int(np.argmax(((A > c_par) | (A < s_perp)).sum(0)))]
    for _ in range(3):
        a = sub @ u
        par = sub[np.abs(a) > c_par] * np.sign(a[np.abs(a) > c_par])[:, None]
        perp = sub[np.abs(a) < s_perp]
        M = perp.T @ perp if len(perp) > 50 else np.zeros((3, 3))
        v = par.sum(0) if len(par) else np.zeros(3)
        w, V = np.linalg.eigh(M - np.outer(v, v) / max(len(par), 1))
        u = V[:, 0] * np.sign(V[:, 0] @ u)
        u /= np.linalg.norm(u)
    a = sub @ u
    share = float(np.mean((np.abs(a) > c_par) | (np.abs(a) < s_perp)))
    # resolve the axis-swap ambiguity with the metadata guess
    perp = sub[np.abs(a) < s_perp]
    if len(perp) >= 100:
        b1, b2 = _perp_basis(u)
        h = perp - np.outer(perp @ u, u)
        ang = np.mod(np.arctan2(h @ b1, h @ b2), np.pi / 2)
        hist, e = np.histogram(ang, bins=90, range=(0, np.pi / 2))
        th = e[np.argmax(hist)] + (e[1] - e[0]) / 2
        w = np.cos(th) * b2 + np.sin(th) * b1
        axes = [u, w / np.linalg.norm(w), np.cross(u, w / np.linalg.norm(w))]
        u = max(axes, key=lambda x: abs(x @ cu))
    return u * np.sign(u @ cu), share
def camera_height(points: np.ndarray, normals: np.ndarray, up: np.ndarray, min_share=0.03):
    """Camera height above the floor in the depth's own units, or None if no floor is
    visible. The floor is the LOWEST upward-facing layer holding min_share of the
    points (not the busiest one, which can be a bed or tabletop)."""
    h = points @ up
    scale = np.median(np.linalg.norm(points, axis=1))
    hh = h[normals @ up > 0.9]
    if len(hh) < 0.03 * len(h):
        return None
    width = 0.02 * scale
    bins = np.arange(hh.min(), hh.max() + width, width)
    if len(bins) < 3:
        return None
    c, e = np.histogram(hh, bins=bins)
    share = (c + np.r_[c[1:], 0] + np.r_[0, c[:-1]]) / len(h)
    cand = np.nonzero(share >= min_share)[0]
    if len(cand) == 0:
        return None
    i = cand[0]
    floor = np.median(hh[(hh >= e[max(i - 1, 0)]) & (hh <= e[min(i + 2, len(e) - 1)])])
    if floor > -0.4 * scale:
        return None
    return float(-floor)
def gravity_frame(up: np.ndarray, heading_hint=np.array([0, 0, 1.0])):
    """Rotation taking camera coordinates to a frame with +z = up (yaw arbitrary but
    deterministic: x is the camera's forward direction projected horizontal)."""
    z = up / np.linalg.norm(up)
    x = heading_hint - (heading_hint @ z) * z
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0, 0]) - z[0] * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])          # rows: new axes expressed in camera coords
