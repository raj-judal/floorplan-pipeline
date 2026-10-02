"""LiDAR tier, rooms stage: capture folder -> room polygons -> plan PNG.

Usage: python tools/plan_lidar.py CAPTURE_DIR OUT_DIR [--stride 5]

Development tool for the rooms stage; the full pipeline command (JSON output,
intervals, openings, drift correction) builds on these functions.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from floorplan.geometry.fusion import fuse
from floorplan.geometry.planes import gravity_align, horizontal_planes, manhattan_yaw, with_normals, yaw_matrix
from floorplan.io import load_capture
from floorplan.plan.render import render_plan
from floorplan.plan.rooms import build_grids, extract_room_polygons, floor_height, free_space, segment_rooms

ap = argparse.ArgumentParser()
ap.add_argument("capture", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("--stride", type=int, default=5)
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)

t0 = time.time()
cap = load_capture(a.capture)
pc = fuse(cap, stride=a.stride)
Rg = gravity_align(cap.up_axis, cap.up_sign)
pc.rotate(Rg, center=(0, 0, 0))
P, N = with_normals(pc)
floor, ceiling = horizontal_planes(P, N)
yaw = manhattan_yaw(N, P, floor)
Ry = yaw_matrix(yaw)
P, N = P @ Ry.T, N @ Ry.T
cams = np.array([f.T_wc[:3, 3] for f in cap.frames]) @ Rg.T @ Ry.T
t_fuse = time.time() - t0

fz = floor_height(P, N)
g, fl, fu, wa = build_grids(P, N, fz)
free = free_space(cap, Ry @ Rg, g)
rooms, wall = segment_rooms(fl, fu, wa, g.to_cell(cams[:, :2]), free=free)
rooms = extract_room_polygons(rooms, g, P, N)
render_plan(rooms, g, wall, a.out / "plan.png")

summary = {
    "capture": cap.name, "frames": len(cap.frames), "stride": a.stride,
    "ceiling_observed": ceiling is not None, "wall_band_m": [round(x, 2) for x in g.wall_band],
    "yaw_deg": round(float(np.degrees(yaw)), 2), "seconds": {"fuse_align": round(t_fuse, 1), "total": round(time.time() - t0, 1)},
    "rooms": [{"id": f"R{r.label}", "area_m2": round(r.area_m2, 2), "entered": r.entered,
               "edges_m": None if r.polygon is None else [round(e.span[1] - e.span[0], 3) for e in r.edges],
               "edges_snapped": None if r.polygon is None else sum(e.snapped for e in r.edges)} for r in rooms],
}
(a.out / "rooms_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
