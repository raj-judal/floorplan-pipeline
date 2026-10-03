"""Thin tiers on rendered protocol views of a real LiDAR scan.
usage: python research/thin_rendered_protocol_test.py LIDAR_RUN_DIR
LIDAR_RUN_DIR is the output of `python -m floorplan run <full scan> --tier lidar`
(uses its output.json rooms and debug/fused_plan_frame.ply). For each room, depth
is rendered from the middle of each wall, 0.3 m in, 1.40 m up, looking across,
for three protocol variants, then measured by the thin-tier code. Compared with
the LiDAR tier's room bounding box (exact only for rectangular rooms).
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import open3d as o3d
from floorplan.thin.rooms_from_images import combine_room, finish_room, measure_image
run = Path(sys.argv[1])
doc = json.loads((run / "output.json").read_text())
fz = json.loads((run / "debug" / "alignment.json").read_text())["floor_z"]
P = np.asarray(o3d.io.read_point_cloud(str(run / "debug" / "fused_plan_frame.ply")).points)
P[:, 2] -= fz
W, H = 320, 240
def render(cam, fwd, hfov, tilt):
    f = W / 2 / np.tan(np.radians(hfov / 2))
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    up = np.array([0, 0, 1.0])
    r = np.cross(fwd, up)
    t = np.radians(tilt)
    f2 = np.cos(t) * fwd - np.sin(t) * up
    Q = (P - cam) @ np.stack([r, np.cross(f2, r), f2]).T
    Q = Q[Q[:, 2] > 0.2]
    u = (Q[:, 0] / Q[:, 2] * K[0, 0] + K[0, 2]).astype(int)
    v = (Q[:, 1] / Q[:, 2] * K[1, 1] + K[1, 2]).astype(int)
    dep = np.full((H, W), np.inf)
    for du in (0, 1, 2):
        for dv in (0, 1, 2):
            uu, vv = u + du, v + dv
            ok = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            np.minimum.at(dep, (vv[ok], uu[ok]), Q[ok, 2])
    dep[~np.isfinite(dep)] = 0
    return dep.astype(np.float32), K
for label, hfov, tilt in [("1x level", 69, 0), ("0.5x level", 108, 0), ("0.5x tilted 10 deg down", 108, 10)]:
    print(label)
    for room in doc["rooms"]:
        V = np.array(room["floor_polygon"]) + np.array(room["transform_to_property"]["translation"])
        lo, hi = V.min(0), V.max(0)
        c = (lo + hi) / 2
        ms = []
        for fwd, cm in [((1, 0, 0), (lo[0] + 0.3, c[1])), ((-1, 0, 0), (hi[0] - 0.3, c[1])),
                        ((0, 1, 0), (c[0], lo[1] + 0.3)), ((0, -1, 0), (c[0], hi[1] - 0.3))]:
            dep, K = render(np.array([cm[0], cm[1], 1.40]), np.array(fwd, float), hfov, tilt)
            ms.append(measure_image(str(fwd), dep, K, np.array([0, -1.0, 0])))
        s = finish_room(ms, None)
        res = combine_room(ms) if s else {"dims": []}
        est = [f"{d['value']:.2f} [{d['value'] - 1.645 * d['sigma']:.2f}-{d['value'] + 1.645 * d['sigma']:.2f}]" for d in res["dims"]]
        print(f"  {room['id']} ({len(room['walls'])} walls): LiDAR box {hi[0] - lo[0]:.2f} x {hi[1] - lo[1]:.2f} m | "
              f"floor seen in {sum(m.scale is not None for m in ms)}/4 views | estimate {est or 'none'}")
