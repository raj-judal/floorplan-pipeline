"""Rooms from a gravity- and yaw-aligned point cloud (z up, walls parallel to x/y).

  1. Occupancy grids: floor, furniture (anything 4 cm to 1.6 m above the floor)
     and structural wall evidence (vertical surfaces 1.5 to 2.3 m up, above
     most furniture).
  2. Interior = observed floor or furniture, closed and hole-filled, minus walls.
  3. Room seeds = interior cells more than DOOR_HALF_WIDTH from any boundary;
     door-sized passages vanish, so each room becomes its own component.
     Seeds are grown back over the interior by watershed.
  4. Each room mask becomes a rectilinear polygon, and every edge is snapped to
     the measured wall face nearest the room (sub-cell precision from the
     points, not the 3 cm grid).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed

RES = 0.03               # m per grid cell
DOOR_HALF_WIDTH = 0.42   # m; passages narrower than ~0.84 m separate rooms
MIN_ROOM_AREA = 1.0      # m2
MIN_EDGE = 0.25          # m; shorter polygon edges are merged away
SNAP_IN, SNAP_OUT = 0.10, 0.15   # m search window for a wall face, inward / outward of the mask edge


@dataclass
class Grid:
    x0: float
    y0: float
    shape: tuple
    floor_z: float
    wall_band: tuple = (1.5, 2.3)

    def to_cell(self, xy):
        return ((xy[:, 1] - self.y0) / RES).astype(int), ((xy[:, 0] - self.x0) / RES).astype(int)

    def to_world(self, rc):
        return np.c_[self.x0 + (rc[:, 1] + 0.5) * RES, self.y0 + (rc[:, 0] + 0.5) * RES]


@dataclass
class Edge:
    axis: int               # 0: edge at constant x (wall normal along x), 1: constant y
    coord: float            # snapped wall-face coordinate (m)
    span: tuple             # (start, end) along the edge direction (m)
    n_support: int = 0      # wall points supporting the snapped face
    face_std: float = 0.0   # m, spread of those points
    snapped: bool = False


@dataclass
class Room:
    label: int
    mask: np.ndarray
    area_m2: float
    polygon: np.ndarray = None          # Kx2 vertices, counter-clockwise, metres
    edges: list = field(default_factory=list)
    entered: bool = True                # did the camera path pass through it


def floor_height(P, N):
    hz = np.abs(N[:, 2]) > 0.95
    h, e = np.histogram(P[hz, 2], bins=np.arange(P[:, 2].min(), P[:, 2].max() + 0.01, 0.01))
    return float(e[np.argmax(h)] + 0.005)


def build_grids(P, N, floor_z):
    x0, y0 = P[:, 0].min() - 0.5, P[:, 1].min() - 0.5
    shape = (int((P[:, 1].max() - y0 + 0.5) / RES) + 1, int((P[:, 0].max() - x0 + 0.5) / RES) + 1)
    g = Grid(x0, y0, shape, floor_z)
    H = P[:, 2] - g.floor_z

    def inside(xy):
        r, c = g.to_cell(np.atleast_2d(xy))
        ok = (r >= 0) & (r < mask.shape[0]) & (c >= 0) & (c < mask.shape[1])
        return bool(ok[0] and mask[r[0], c[0]])


    def count(mask):
        out = np.zeros(shape, np.int32)
        r, c = g.to_cell(P[mask])
        np.add.at(out, (r, c), 1)
        return out

    # Floor is found per 0.3 m tile as the lowest horizontal surface within
    # -15/+10 cm of the global floor. One global band misses sunken bathrooms
    # and floors shifted by vertical drift (seen in the sample single-room scan,
    # where floor heights spread over 4 cm).
    cand = (np.abs(N[:, 2]) > 0.95) & (H > -0.15) & (H < 0.10)
    tiles = np.floor(P[cand, :2] / 0.3).astype(np.int64)
    key = tiles[:, 0] * 100003 + tiles[:, 1]
    order = np.lexsort((P[cand, 2], key))
    k_sorted, z_sorted = key[order], P[cand, 2][order]
    starts = np.r_[0, np.nonzero(np.diff(k_sorted))[0] + 1]
    counts_t = np.diff(np.r_[starts, len(k_sorted)])
    low = z_sorted[starts + (counts_t * 0.05).astype(int)]           # 5th percentile per tile
    low_per_point = np.repeat(low, counts_t)
    is_floor = np.zeros(len(P), bool)
    idx = np.nonzero(cand)[0][order]
    is_floor[idx] = z_sorted < low_per_point + 0.03
    floor = count(is_floor)
    furn = count((H > 0.04) & (H < 1.6))
    # Wall evidence comes from vertical surfaces above most furniture. The band
    # adapts to how high the capture reached: a floor-focused scan (camera
    # pointed down) has little data above 2 m, so the band slides down.
    top = min(2.3, float(np.percentile(H, 99)) - 0.05)
    bottom = max(0.9, top - 0.8)
    wall = count((np.abs(N[:, 2]) < 0.2) & (H > bottom) & (H < top))
    g.wall_band = (bottom, top)
    return g, floor, furn, wall


def segment_rooms(floor, furn, wall_count, cams_rc=None, free=None, min_free=2):
    wall = ndi.binary_dilation(wall_count >= 2, iterations=1)
    observed = (floor > 0) | (furn > 0)
    if free is not None:
        observed |= free >= min_free
    interior = ndi.binary_closing(observed, iterations=3) & ~wall
    interior = ndi.binary_fill_holes(interior) & ~wall
    interior = ndi.binary_opening(interior, iterations=2)
    lab, n = ndi.label(interior)
    sizes = ndi.sum(interior, lab, range(1, n + 1)) * RES * RES
    interior = np.isin(lab, 1 + np.nonzero(sizes > MIN_ROOM_AREA)[0])

    dist = ndi.distance_transform_edt(interior) * RES
    seeds, ns = ndi.label(dist > DOOR_HALF_WIDTH)
    ss = ndi.sum(seeds > 0, seeds, range(1, ns + 1)) * RES * RES
    seeds = np.where(np.isin(seeds, 1 + np.nonzero(ss > 0.6)[0]), seeds, 0)
    labels = watershed(-dist, markers=seeds, mask=interior)

    rooms = []
    for k, i in enumerate(sorted(set(np.unique(labels)) - {0}), start=1):
        m = labels == i
        area = m.sum() * RES * RES
        if area < MIN_ROOM_AREA:
            continue
        entered = True
        if cams_rc is not None:
            r, c = cams_rc
            ok = (r >= 0) & (r < m.shape[0]) & (c >= 0) & (c < m.shape[1])
            entered = bool(m[r[ok], c[ok]].any())
        rooms.append(Room(k, m, float(area), entered=entered))
    return rooms, wall


def _rectilinear(mask, g: Grid):
    """Mask -> list of axis-aligned edges (in metres) around its outer boundary."""
    m = ndi.binary_closing(mask, iterations=2).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea)
    approx = cv2.approxPolyDP(c, epsilon=0.08 / RES, closed=True)[:, 0, :]   # (col, row)
    pts = g.to_world(approx[:, ::-1].astype(float))
    # each segment becomes horizontal or vertical; merge consecutive same-axis segments
    segs = []
    for a, b in zip(pts, np.roll(pts, -1, axis=0)):
        d = b - a
        axis = 0 if abs(d[1]) >= abs(d[0]) else 1          # mostly-vertical segment: constant x
        coord = (a[0] + b[0]) / 2 if axis == 0 else (a[1] + b[1]) / 2
        length = abs(d[1]) if axis == 0 else abs(d[0])
        if segs and segs[-1][0] == axis:
            pa, pc, pl = segs[-1]
            segs[-1] = (axis, (pc * pl + coord * length) / max(pl + length, 1e-9), pl + length)
        else:
            segs.append((axis, coord, length))
    if len(segs) > 1 and segs[0][0] == segs[-1][0]:
        a0, c0, l0 = segs.pop()
        a1, c1, l1 = segs[0]
        segs[0] = (a1, (c0 * l0 + c1 * l1) / (l0 + l1), l0 + l1)
    # drop short segments (merge them into neighbours) until edges alternate axis
    changed = True
    while changed and len(segs) > 4:
        changed = False
        for i, s in enumerate(segs):
            if s[2] < MIN_EDGE:
                segs.pop(i)
                j = i % len(segs)
                k = (i - 1) % len(segs)
                if segs[j][0] == segs[k][0] and j != k:
                    a, cj, lj = segs[j]
                    _, ck, lk = segs[k]
                    merged = (a, (cj * lj + ck * lk) / (lj + lk), lj + lk)
                    segs[k] = merged
                    segs.pop(j)
                changed = True
                break
    return [Edge(axis=a, coord=c, span=(0, 0)) for a, c, _ in segs]


def _vertices(edges):
    """Intersect consecutive alternating-axis edges into polygon vertices."""
    V = []
    for e, f in zip(edges, edges[1:] + edges[:1]):
        V.append([e.coord, f.coord] if e.axis == 0 else [f.coord, e.coord])
    return np.array(V)


def snap_edges(edges, verts, P, N, g: Grid, mask):
    """Move each edge onto the nearest measured wall face (sub-centimetre)."""
    H = P[:, 2] - g.floor_z

    def inside(xy):
        r, c = g.to_cell(np.atleast_2d(xy))
        ok = (r >= 0) & (r < mask.shape[0]) & (c >= 0) & (c < mask.shape[1])
        return bool(ok[0] and mask[r[0], c[0]])

    structural = (np.abs(N[:, 2]) < 0.2) & (H > 0.3) & (H < 2.3)
    n = len(edges)
    for i, e in enumerate(edges):
        a, b = verts[i - 1], verts[i]               # edge i runs between vertex i-1 and i
        along = 1 - e.axis
        lo, hi = sorted([a[along], b[along]])
        e.span = (lo, hi)
        mid = np.zeros(2)
        mid[e.axis], mid[along] = e.coord, (lo + hi) / 2
        probe = mid.copy()
        probe[e.axis] += 0.12
        inward = 1.0 if inside(probe) else -1.0
        sel = structural & (np.abs(N[:, e.axis]) > 0.95) \
            & (P[:, along] > lo + 0.1) & (P[:, along] < hi - 0.1)
        x = P[sel, e.axis]
        rel = (x - e.coord) * -inward               # positive = outward from the room
        w = (rel > -SNAP_IN) & (rel < SNAP_OUT)
        if w.sum() < 50:
            continue
        hist, edg = np.histogram(rel[w], bins=np.arange(-SNAP_IN, SNAP_OUT + 0.01, 0.01))
        peak = edg[np.argmax(hist)] + 0.005
        near = np.abs(rel[w] - peak) < 0.02
        face = float(np.median(rel[w][near]))
        e.coord = e.coord + face * -inward
        e.n_support = int(near.sum())
        e.face_std = float(1.4826 * np.median(np.abs(rel[w][near] - face)))
        e.snapped = True
    return edges


def extract_room_polygons(rooms, g: Grid, P, N):
    for room in rooms:
        edges = _rectilinear(room.mask, g)
        if len(edges) < 4 or len(edges) % 2:
            continue
        if any(e.axis == f.axis for e, f in zip(edges, edges[1:] + edges[:1])):
            continue   # could not reduce to alternating edges; left without a polygon
        verts = _vertices(edges)
        edges = snap_edges(edges, verts, P, N, g, room.mask)
        room.edges = edges
        room.polygon = _vertices(edges)
    return rooms


def free_space(capture, R_align, g: Grid, stride: int = 6, pix_step: int = 8,
               max_range: float = 5.0, stop_short: float = 0.10, n_samples: int = 160):
    """2D free-space counts by carving the camera-to-surface rays.

    A cell crossed by a ray before it hits a surface is empty space. This marks
    the inside of rooms even where the floor itself was never measured (dark
    fabric, glossy tiles, under furniture), which floor and furniture evidence
    alone cannot. Rays stop `stop_short` before the hit so walls stay solid.
    """
    from ..io.capture import backproject

    free = np.zeros(g.shape[0] * g.shape[1], np.int64)
    sub = None
    for f in capture.frames[::stride]:
        d = capture.read_depth(f)
        if sub is None:
            sub = np.zeros(d.shape, bool)
            sub[pix_step // 2::pix_step, pix_step // 2::pix_step] = True
        mask = sub & (d > 0.2) & (d < max_range)
        W, rng, _, _ = backproject(d, f.K, f.T_wc, mask)
        if len(W) == 0:
            continue
        W = W @ R_align.T
        cam = f.T_wc[:3, 3] @ R_align.T
        frac = np.linspace(0.0, 1.0, n_samples)[None, :] * (1 - stop_short / rng)[:, None]
        S = cam[:2] + (W[:, None, :2] - cam[:2]) * frac[:, :, None]
        r = ((S[..., 1] - g.y0) / RES).astype(np.int64).ravel()
        c = ((S[..., 0] - g.x0) / RES).astype(np.int64).ravel()
        ok = (r >= 0) & (r < g.shape[0]) & (c >= 0) & (c < g.shape[1])
        flat = np.unique(r[ok] * g.shape[1] + c[ok])          # count each cell once per frame
        free += np.bincount(flat, minlength=free.size)
    return free.reshape(g.shape)
