"""Thin tiers (photos, video): measure each room from its own images.
Multi-image assembly was tried and dropped (lab notebook). Instead, every
image measures wall-to-wall distances on its own, and the capture protocol
supplies views that show both side walls (photo straight across the room from
the middle of each wall; one clip per room for video).
Per image: depth -> gravity (geometry.estimate_up) -> scale from camera height
-> walls axis-aligned (Manhattan yaw) -> outermost wall surface on each side,
using wall points 0.6-2.3 m up (outermost per side, so furniture is skipped) -> spans where both sides are
seen. Per room: spans cluster into the two room dimensions.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .geometry import camera_height, estimate_up, gravity_frame, points_and_normals
H_PRIOR = 1.40                 # m, protocol: phone held at chest height
H_PRIOR_SIGMA = 0.07           # m, holding height varies (1.31-1.55 m in the sample capture)
PLAUSIBLE = (0.9, 1.8)         # camera height in model units / H_PRIOR (tabletop rejection)
FRAME_SCALE_SIGMA = 0.146      # relative, per frame, after correction (held-out capture: 90% within 24%)
GEOMETRY_SIGMA = 0.22          # relative: a view measures only the part of the room it sees. Calibrated on
                               # rendered protocol views of the sample apartment (90% of errors within ~40%);
                               # without it only 7 of 13 truths fell inside the 90% intervals (lab notebook)
WALL_BAND = (0.6, 2.3)         # m above floor; the outermost wall per side is used, so furniture in front is skipped
MIN_WALL_POINTS = 200
MIN_EXTENT = 0.5               # m of wall seen along its length
@dataclass
class ImageMeasure:
    name: str
    scale: float | None        # depth multiplier from camera height, None if no usable floor
    up_share: float
    spans: list = field(default_factory=list)          # wall-to-wall distances seen in this image (m)
    one_sided: list = field(default_factory=list)      # distance camera -> single wall (m), fallback
    ceiling_above_camera: float | None = None           # m, after scaling
def _walls_one_side(P, N, axis, sign):
    """Outermost wall on one side of the camera along `axis` (sign -1: negative side).
    Walls face the camera, so a wall on the negative side has normal pointing +axis."""
    m = (N[:, axis] * -sign > 0.9) & (P[:, axis] * sign > 0.3)
    if m.sum() < MIN_WALL_POINTS:
        return None
    x = P[m, axis]
    other = P[m, 1 - axis]
    bins = np.arange(x.min(), x.max() + 0.02, 0.02)
    if len(bins) < 2:
        return None
    c, e = np.histogram(x, bins=bins)
    order = np.argsort(e[:-1] * sign)[::-1]                 # outermost first
    for i in order:
        if c[max(i - 1, 0):i + 2].sum() < MIN_WALL_POINTS:
            continue
        near = np.abs(x - (e[i] + 0.01)) < 0.03
        if np.ptp(other[near]) >= MIN_EXTENT:
            return float(np.median(x[near]))
    return None
def measure_image(name, depth, K, coarse_up, stride=2, scale=None):
    """One image -> ImageMeasure. `scale` overrides the camera-height scale (used for
    images whose own floor is not visible: the room's median scale is applied later)."""
    P, N = points_and_normals(depth[::stride, ::stride], K / np.array([[stride], [stride], [1]]))
    up, share = estimate_up(N, coarse_up)
    h = camera_height(P, N, up)
    own = H_PRIOR / h if (h is not None and PLAUSIBLE[0] <= h / H_PRIOR <= PLAUSIBLE[1]) else None
    m = ImageMeasure(name, own, share)
    s = scale if scale is not None else own
    if s is None:
        m._raw = (P, N, up)
        return m
    _measure_scaled(m, P, N, up, s)
    return m
def _measure_scaled(m, P, N, up, s):
    R = gravity_frame(up)
    P = (P @ R.T) * s
    N = N @ R.T
    P[:, 2] += H_PRIOR
    wall = np.abs(N[:, 2]) < 0.2
    if wall.sum() > 100:                                     # rotate so walls are axis-aligned
        ang = np.mod(np.arctan2(N[wall, 1], N[wall, 0]), np.pi / 2)
        hist, e = np.histogram(ang, bins=180, range=(0, np.pi / 2))
        th = e[np.argmax(hist)] + (e[1] - e[0]) / 2
        c, si = np.cos(-th), np.sin(-th)
        Rz = np.array([[c, -si, 0], [si, c, 0], [0, 0, 1.0]])
        P, N = P @ Rz.T, N @ Rz.T
    band = (P[:, 2] > WALL_BAND[0]) & (P[:, 2] < WALL_BAND[1]) & wall
    Pb, Nb = P[band], N[band]
    for axis in (0, 1):
        lo = _walls_one_side(Pb, Nb, axis, -1)
        hi = _walls_one_side(Pb, Nb, axis, +1)
        if lo is not None and hi is not None:
            m.spans.append(hi - lo)
        else:
            m.one_sided += [abs(v) for v in (lo, hi) if v is not None]
    ceil = (N[:, 2] < -0.9) & (P[:, 2] > H_PRIOR + 0.3)
    if ceil.sum() > 300:
        m.ceiling_above_camera = float(np.median(P[ceil, 2]) - H_PRIOR)
def finish_room(measures, raw_images):
    """Apply the room's median scale to images without their own, then combine.
    raw_images: {name: (P, N, up)} kept by measure_image for floorless images."""
    own = [m.scale for m in measures if m.scale is not None]
    if not own:
        return None
    s_room = float(np.median(own))
    for m in measures:
        if m.scale is None and hasattr(m, "_raw"):
            P, N, up = m._raw
            _measure_scaled(m, P, N, up, s_room)
            del m._raw
    return s_room
def combine_room(measures):
    """Two room dimensions (m) with 1-sigma, from all images' spans.
    Spans from different images measure either room dimension (each image has its
    own axes), so they are split into two groups by 1-D 2-means; a square room
    gives two equal groups, which is the right answer. Uncertainty: scale (median
    of n per-image scales) plus the camera-height prior, both relative."""
    spans = np.array([s for m in measures for s in m.spans])
    n_scale = sum(m.scale is not None for m in measures)
    rel = np.sqrt((1.25 * FRAME_SCALE_SIGMA / np.sqrt(max(n_scale, 1))) ** 2 + (H_PRIOR_SIGMA / H_PRIOR) ** 2
                  + GEOMETRY_SIGMA ** 2)
    dims = []
    if len(spans) >= 2:
        lo, hi = np.percentile(spans, 25), np.percentile(spans, 75)
        for _ in range(20):
            a = spans[np.abs(spans - lo) <= np.abs(spans - hi)]
            b = spans[np.abs(spans - lo) > np.abs(spans - hi)]
            lo, hi = (np.median(a), np.median(b) if len(b) else lo)
        groups = [g for g in (a, b) if len(g)]
        for g in groups:
            v = float(np.median(g))
            spread = 1.25 * np.std(g) / np.sqrt(len(g)) if len(g) > 1 else 0.0
            dims.append({"value": v, "sigma": float(np.hypot(rel * v, spread)), "n": int(len(g)), "method": "wall_to_wall_in_image"})
    elif len(spans) == 1:
        v = float(spans[0])
        dims.append({"value": v, "sigma": float(rel * v * 1.5), "n": 1, "method": "wall_to_wall_in_image"})
    ceils = [m.ceiling_above_camera for m in measures if m.ceiling_above_camera is not None]
    ceiling = None
    if ceils:
        v = H_PRIOR + float(np.median(ceils))
        ceiling = {"value": v, "sigma": float(np.hypot(rel * float(np.median(ceils)), H_PRIOR_SIGMA)), "n": len(ceils)}
    return {"dims": sorted(dims, key=lambda d: d["value"]), "ceiling": ceiling,
            "scale_frames": n_scale, "relative_scale_sigma": float(rel)}
