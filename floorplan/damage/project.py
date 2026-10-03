"""Turn image detections into measured damage regions on the plan's surfaces (LiDAR tier).
For a detection box in a video frame, the depth pixels inside the box are placed
in 3D with that frame's (drift-corrected) pose, moved into the plan frame, and
assigned to one surface of one room: floor, ceiling, or the nearest wall. The
region's extent is measured in that surface's own 2D frame (walls: u along the
wall from its start, v up from the floor; floor/ceiling: the room frame).
A real stain stays put while the camera moves, so it is detected in several
frames at the same place on the same surface; one-off boxes do not repeat.
Regions are reported only when seen in at least MIN_VIEWS frames.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import cv2
import numpy as np
from matplotlib.path import Path as MPath
from ..io.capture import backproject
MIN_VIEWS = 2
MIN_POINTS = 20
WALL_DIST = 0.25          # m: points this close to a wall line belong to it
FLOOR_BAND = 0.15         # m above the floor counts as floor
CEIL_BAND = 0.20          # m below the ceiling counts as ceiling
MERGE_DIST = 0.30         # m between region centres to treat as the same damage
ROT = {0: None, 90: cv2.ROTATE_90_COUNTERCLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_CLOCKWISE}
def upright_rotation(T_wc, up_world):
    """Counter-clockwise rotation (deg) that turns a frame upright, from the pose's gravity."""
    up = T_wc[:3, :3].T @ up_world
    return int(round(np.degrees(np.arctan2(up[0], -up[1])) / 90.0)) % 4 * 90
def unrotate_point(x, y, k, W, H):
    """Map a pixel in the image rotated CCW by k back to the original W x H image."""
    if k == 0:
        return x, y
    if k == 90:                       # original (x, y) -> rotated (y, W-1-x)
        return W - 1 - y, x
    if k == 180:
        return W - 1 - x, H - 1 - y
    return y, H - 1 - x               # k == 270: original (x, y) -> rotated (H-1-y, x)
@dataclass
class Hit:
    frame: int
    label: str
    score: float
    room: str
    surface: str
    kind: str                 # wall | floor | ceiling
    u: tuple                  # (lo, hi) along the surface's first axis, m
    v: tuple                  # (lo, hi) along its second axis, m
@dataclass
class Region:
    room: str
    surface: str
    kind: str
    label: str
    hits: list = field(default_factory=list)
def surfaces_from_doc(doc):
    """Room polygons and wall segments in the property (plan) frame, from output.json rooms."""
    out = []
    for r in doc["rooms"]:
        o = np.array(r["transform_to_property"]["translation"])
        poly = np.array(r["floor_polygon"]) + o
        walls = [(w["surface_id"], np.array(w["start"]) + o, np.array(w["end"]) + o) for w in r["walls"]]
        out.append({"id": r["id"], "poly": poly, "origin": o, "walls": walls,
                    "ceiling": float(r["ceiling_height"]["value"])})
    return out
def assign(P, rooms, floor_z):
    """Points (plan frame) -> (room, surface, kind, u, v) by majority vote, or None."""
    xy, h = P[:, :2], P[:, 2] - floor_z
    best = None
    for room in rooms:
        path = MPath(room["poly"])
        inside = path.contains_points(xy, radius=0.2) | path.contains_points(xy, radius=-0.2)
        if inside.sum() < MIN_POINTS:
            continue
        q, hq = xy[inside], h[inside]
        votes = {}
        fl = hq < FLOOR_BAND
        ce = hq > room["ceiling"] - CEIL_BAND
        if fl.sum():
            votes[(f"{room['id']}-F", "floor")] = fl
        if ce.sum():
            votes[(f"{room['id']}-C", "ceiling")] = ce
        mid = ~fl & ~ce
        if mid.sum():
            d = []
            for sid, a, b in room["walls"]:
                t = b - a
                L = np.linalg.norm(t)
                t = t / L
                s = np.clip((q[mid] - a) @ t, 0, L)
                d.append(np.linalg.norm(q[mid] - (a + np.outer(s, t)), axis=1))
            d = np.array(d)
            near = d.min(0) < WALL_DIST
            idx = d.argmin(0)
            for k, (sid, a, b) in enumerate(room["walls"]):
                m = np.zeros(len(q), bool)
                m[np.nonzero(mid)[0][near & (idx == k)]] = True
                if m.sum():
                    votes[(sid, "wall")] = m
        if not votes:
            continue
        (sid, kind), m = max(votes.items(), key=lambda kv: kv[1].sum())
        if m.sum() < MIN_POINTS or m.sum() < 0.6 * len(q):
            continue
        if best is None or m.sum() > best[0]:
            if kind == "wall":
                a, b = next((a, b) for s, a, b in room["walls"] if s == sid)
                t = (b - a) / np.linalg.norm(b - a)
                u, v = (q[m] - a) @ t, hq[m]
            else:
                u, v = (q[m] - room["origin"]).T
            best = (int(m.sum()), room["id"], sid, kind, u, v)
    return None if best is None else best[1:]
def hit_from_box(frame_meta, depth, conf, box_rot, k, rgb_size, R_align, rooms, floor_z, label, score, index):
    """One detection box (in the upright image, already scaled to rgb_size) -> Hit, or None."""
    W, H = rgb_size                                    # original RGB size (before rotation)
    x0, y0, x1, y1 = box_rot
    corners = [unrotate_point(x, y, k, W, H) for x, y in ((x0, y0), (x1, y0), (x0, y1), (x1, y1))]
    xs, ys = zip(*corners)
    sx, sy = depth.shape[1] / W, depth.shape[0] / H
    c0, c1 = int(max(0, min(xs) * sx)), int(min(depth.shape[1], max(xs) * sx + 1))
    r0, r1 = int(max(0, min(ys) * sy)), int(min(depth.shape[0], max(ys) * sy + 1))
    mask = np.zeros(depth.shape, bool)
    mask[r0:r1, c0:c1] = True
    mask &= depth > 0.2
    if conf is not None:
        mask &= conf >= 1
    if mask.sum() < MIN_POINTS:
        return None
    Pw, _, _, _ = backproject(depth, frame_meta.K, frame_meta.T_wc, mask)
    P = Pw @ R_align.T
    a = assign(P, rooms, floor_z)
    if a is None:
        return None
    room, sid, kind, u, v = a
    return Hit(index, label, score, room, sid, kind,
               (float(np.percentile(u, 5)), float(np.percentile(u, 95))),
               (float(np.percentile(v, 5)), float(np.percentile(v, 95))))
def merge(hits):
    """Group hits of the same class on the same surface whose centres are within MERGE_DIST;
    keep groups seen in at least MIN_VIEWS distinct frames."""
    regions = []
    for h in sorted(hits, key=lambda h: -h.score):
        c = np.array([np.mean(h.u), np.mean(h.v)])
        for r in regions:
            if r.surface == h.surface and r.label == h.label:
                rc = np.array([np.median([np.mean(x.u) for x in r.hits]), np.median([np.mean(x.v) for x in r.hits])])
                if np.linalg.norm(c - rc) < MERGE_DIST:
                    r.hits.append(h)
                    break
        else:
            regions.append(Region(h.room, h.surface, h.kind, h.label, [h]))
    return [r for r in regions if len({h.frame for h in r.hits}) >= MIN_VIEWS]
