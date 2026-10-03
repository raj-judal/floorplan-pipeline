"""End-to-end LiDAR-tier pipeline: capture folder -> output.json (schema v1.1.0) + plan.png.

Every number carries a 90% interval built from an explicit error budget
(CONFIG below). The systematic terms come from the ARKitScenes laser study in
docs/lab_notebook.md and are deliberately conservative until the benchmark
calibrates them; stages that do not exist yet say so in quality.warnings
instead of emitting empty-but-confident output.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi

from . import __version__
from .geometry.fusion import fuse
from .geometry.planes import (_robust_hfit, gravity_align, horizontal_planes, manhattan_yaw,
                              with_normals, yaw_matrix)
from .io import load_capture
from .plan.render import render_plan
from .plan.rooms import (RES, _vertices, build_grids, extract_room_polygons, floor_height, free_space,
                         segment_rooms)

SCHEMA_VERSION = "1.1.0"
Z90 = 1.645
CONFIG = {
    # Error budget v1, calibrated 2026-10-03 on the LiDAR benchmark
    # (benchmarks/arkitscenes_421383/lidar_benchmark_v0.json: 8 wall lengths and
    # 2 ceilings against Faro laser). See docs/lab_notebook.md, "Calibration".
    "error_budget_version": "v1-calibrated-421383",
    "stride": 5,
    "sys_face_m": 0.035,        # per wall face; 90% of measured wall-length errors are within ~80 mm
                                # (= 1.645 * sqrt(2) * 35 mm). Was 15 mm (v0 guess): only 40% coverage.
    "unsnapped_face_m": 0.06,   # edge placed from the 3 cm grid only: never tighter than a measured face
    "sys_ceiling_m": 0.020,     # measured ceiling errors -9.9 / -27.7 mm, both inside +/-33 mm
    "ceiling_min_coverage": 0.25,
    "ceiling_upper_prior_m": 3.2,
    "opening_width_sigma_m": 0.05,   # no ground truth for openings yet: uncalibrated
    "drift_correction": True,
}


def _meas(value, sigma, unit="m", method=None, n=None):
    v = float(value)
    return {"value": round(v, 4), "unit": unit,
            "ci": {"lower": round(v - Z90 * sigma, 4), "upper": round(v + Z90 * sigma, 4), "level": 0.9},
            "method": method, "n_samples": n}


def _meas_bounds(lower, upper, unit="m", method=None):
    return {"value": round((lower + upper) / 2, 4), "unit": unit,
            "ci": {"lower": round(lower, 4), "upper": round(upper, 4), "level": 0.9},
            "method": method, "n_samples": None}


def _face_sigma(e):
    if not e.snapped:
        return CONFIG["unsnapped_face_m"]
    se = e.face_std / np.sqrt(max(e.n_support, 1))
    return float(np.hypot(se, CONFIG["sys_face_m"]))


def _git_commit(repo: Path):
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def _room_ceiling(room, g, P, N, H):
    """Floor and ceiling planes fitted inside one room; height averaged over the room."""
    r, c = g.to_cell(P[:, :2])
    ok = (r >= 0) & (r < g.shape[0]) & (c >= 0) & (c < g.shape[1])
    inside = np.zeros(len(P), bool)
    inside[ok] = room.mask[r[ok], c[ok]]
    horiz = np.abs(N[:, 2]) > 0.95
    fl = inside & horiz & (np.abs(H) < 0.06)
    ce = inside & horiz & (H > 1.9)
    cells_room = room.mask.sum()
    rc = np.argwhere(room.mask)
    xy = g.to_world(rc.astype(float))
    top_seen = float(np.percentile(H[inside], 99.5)) if inside.sum() > 100 else 0.0
    if ce.sum() > 200 and fl.sum() > 200:
        cr, cc = g.to_cell(P[ce, :2])
        coverage = len(set(zip(cr // 3, cc // 3))) / max(len(set(zip(rc[:, 0] // 3, rc[:, 1] // 3))), 1)
        if coverage >= CONFIG["ceiling_min_coverage"]:
            fcoef, fkeep, _ = _robust_hfit(P[fl])
            ccoef, ckeep, _ = _robust_hfit(P[ce])
            h = (xy @ ccoef[:2] + ccoef[2]) - (xy @ fcoef[:2] + fcoef[2])
            return _meas(np.mean(h), CONFIG["sys_ceiling_m"], method="lidar_plane_fit_footprint_mean",
                         n=int(fkeep.sum() + ckeep.sum())), None
    lower = max(top_seen, 2.0)
    upper = max(CONFIG["ceiling_upper_prior_m"], lower + 0.1)
    return (_meas_bounds(lower, upper, method="bounded_not_observed"),
            f"ceiling not observed: height bounded below by the highest observed surface ({lower:.2f} m) "
            f"and above by a {upper:.1f} m residential prior")


def _openings_between(rooms, g):
    """Passages between rooms: cells where two dilated room masks meet."""
    found = []
    dil = [ndi.binary_dilation(r.mask, iterations=4) for r in rooms]
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            contact = dil[i] & dil[j]
            if contact.sum() < 6:
                continue
            lab, n = ndi.label(contact)
            for k in range(1, n + 1):
                cells = np.argwhere(lab == k)
                if len(cells) < 6:
                    continue
                found.append((i, j, g.to_world(cells.astype(float))))
    return found


def _assign_opening(room, pts):
    """Find the edge of `room` the passage sits on; return (edge index, width, offset, start, end)."""
    c = pts.mean(axis=0)
    best = None
    for i, e in enumerate(room.edges):
        along = 1 - e.axis
        d = abs(c[e.axis] - e.coord)
        inside_span = e.span[0] - 0.2 <= c[along] <= e.span[1] + 0.2
        if inside_span and (best is None or d < best[0]):
            best = (d, i)
    if best is None or best[0] > 0.3:
        return None
    e = room.edges[best[1]]
    along = 1 - e.axis
    lo, hi = pts[:, along].min() - RES / 2, pts[:, along].max() + RES / 2
    lo, hi = max(lo, e.span[0]), min(hi, e.span[1])
    if hi - lo < 0.3:
        return None
    return best[1], hi - lo, lo - e.span[0]


def _overlap_m2(rooms, g):
    if len(rooms) < 2:
        return 0.0
    scale = 1 / 0.01
    masks = []
    for r in rooms:
        m = np.zeros((int(g.shape[0] * RES * scale) + 1, int(g.shape[1] * RES * scale) + 1), np.uint8)
        q = np.c_[(r.polygon[:, 0] - g.x0) * scale, (r.polygon[:, 1] - g.y0) * scale].astype(np.int32)
        cv2.fillPoly(m, [q], 1)
        masks.append(m.astype(bool))
    worst = 0.0
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            worst = max(worst, (masks[i] & masks[j]).sum() * 1e-4)
    return float(worst)


def run(capture_dir, out_dir, stride=None, command="python -m floorplan run", drift=None, drift_result=None, diag=None):
    t0 = time.time()
    capture_dir, out_dir = Path(capture_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stride = stride or CONFIG["stride"]
    warnings = []

    cap = load_capture(capture_dir)
    drift = CONFIG["drift_correction"] if drift is None else drift
    drift_doc = {"method": "none", "enabled": False, "loop_closures": None,
                 "residual_before_m": None, "residual_after_m": None}
    if drift:
        from .geometry.drift import apply_correction, correct_drift
        dr = drift_result or correct_drift(cap)
        cap = apply_correction(cap, dr)
        drift_doc = {"method": "pose_graph_loop_closure", "enabled": True,
                     "loop_closures": dr.stats["loop_closures"],
                     "residual_before_m": dr.stats["residual_before_m"],
                     "residual_after_m": dr.stats["residual_after_m"]}
    pc = fuse(cap, stride=stride)
    Rg = gravity_align(cap.up_axis, cap.up_sign)
    pc.rotate(Rg, center=(0, 0, 0))
    P, N = with_normals(pc)
    floor, _ = horizontal_planes(P, N)
    yaw = manhattan_yaw(N, P, floor)
    Ry = yaw_matrix(yaw)
    P, N = P @ Ry.T, N @ Ry.T
    cams = np.array([f.T_wc[:3, 3] for f in cap.frames]) @ Rg.T @ Ry.T

    fz = floor_height(P, N)
    H = P[:, 2] - fz
    g, fl, fu, wa = build_grids(P, N, fz)
    free = free_space(cap, Ry @ Rg, g)
    rooms, wall = segment_rooms(fl, fu, wa, g.to_cell(cams[:, :2]), free=free)
    rooms = [r for r in extract_room_polygons(rooms, g, P, N) if r.polygon is not None]
    for k, r in enumerate(rooms, start=1):
        r.label = k

    room_docs, adjacency = [], []
    opening_count = {r.label: 0 for r in rooms}
    for r in rooms:
        V = r.polygon
        if 0.5 * np.sum(V[:, 0] * np.roll(V[:, 1], -1) - np.roll(V[:, 0], -1) * V[:, 1]) < 0:
            # keep counter-clockwise order; edge i runs from vertex i-1 to vertex i
            r.edges = r.edges[::-1]
            r.polygon = V = _vertices(r.edges)
        origin = V.min(axis=0)
        ceiling, note = _room_ceiling(r, g, P, N, H)
        rid = f"R{r.label}"
        if note:
            warnings.append({"code": "ceiling_not_observed", "message": note, "affects": [f"{rid}-C"]})
        if not r.entered:
            warnings.append({"code": "room_not_entered", "message": f"{rid} was seen through an opening only",
                             "affects": [rid]})
        sig = [_face_sigma(e) for e in r.edges]
        s_h = (ceiling["ci"]["upper"] - ceiling["value"]) / Z90
        walls, surfaces = [], []
        for i, e in enumerate(r.edges):
            a, b = V[i - 1] - origin, V[i] - origin
            L = float(np.linalg.norm(b - a))
            s_len = float(np.hypot(sig[i - 1], sig[(i + 1) % len(sig)]))
            sid = f"{rid}-W{i + 1}"
            walls.append({"surface_id": sid, "start": [round(float(x), 4) for x in a],
                          "end": [round(float(x), 4) for x in b],
                          "length": _meas(L, s_len, method="lidar_wall_face_snap"),
                          "height": ceiling})
            surfaces.append({"id": sid, "kind": "wall",
                             "area": _meas(L * ceiling["value"], float(np.hypot(L * s_h, ceiling["value"] * s_len)), "m2")})
            if not e.snapped:
                warnings.append({"code": "edge_not_snapped", "affects": [sid],
                                 "message": f"{sid} has no measured wall face; placed from the occupancy grid"})
        lens = np.array([w["length"]["value"] for w in walls])
        area = 0.5 * abs(np.sum(V[:, 0] * np.roll(V[:, 1], -1) - np.roll(V[:, 0], -1) * V[:, 1]))
        s_area = float(np.sqrt(np.sum((lens * np.array(sig)) ** 2)))
        surfaces += [{"id": f"{rid}-F", "kind": "floor", "area": _meas(area, s_area, "m2")},
                     {"id": f"{rid}-C", "kind": "ceiling", "area": _meas(area, s_area, "m2")}]
        w, h = np.ptp(V, axis=0)
        room_docs.append({
            "id": rid, "label": "room", "is_connector": bool(min(w, h) < 1.4 and max(w, h) > 2.5 * min(w, h)),
            "floor_polygon": [[round(float(x), 4) for x in v - origin] for v in V],
            "transform_to_property": {"translation": [round(float(x), 4) for x in origin], "rotation_deg": 0.0},
            "surfaces": surfaces, "walls": walls, "openings": [],
            "ceiling_height": ceiling,
            "floor_area": _meas(area, s_area, "m2", method="polygon_of_snapped_faces"),
            "perimeter": _meas(float(lens.sum()), float(np.sqrt(np.sum((2 * np.array(sig)) ** 2))),
                               method="polygon_of_snapped_faces")})

    by_label = {d["id"]: d for d in room_docs}
    for i, j, pts in _openings_between(rooms, g):
        ids = []
        for room, other in ((rooms[i], rooms[j]), (rooms[j], rooms[i])):
            hit = _assign_opening(room, pts)
            if hit is None:
                ids.append(None)
                continue
            ei, width, offset = hit
            opening_count[room.label] += 1
            oid = f"R{room.label}-O{opening_count[room.label]}"
            by_label[f"R{room.label}"]["openings"].append({
                "id": oid, "type": "open_passage", "wall_id": f"R{room.label}-W{ei + 1}",
                "width": _meas(width, CONFIG["opening_width_sigma_m"], method="room_contact_extent"),
                "height": _meas_bounds(1.9, 2.4, method="not_measured_prior"),
                "sill_height": None,
                "offset_along_wall": _meas(offset, CONFIG["opening_width_sigma_m"], method="room_contact_extent"),
                "connects_to": f"R{other.label}", "detection_confidence": 0.6})
            ids.append(oid)
        if ids[0] or ids[1]:
            a_room, b_room = (rooms[i], rooms[j]) if ids[0] else (rooms[j], rooms[i])
            adjacency.append({"room_a": f"R{a_room.label}", "room_b": f"R{b_room.label}",
                              "opening_a": ids[0] or ids[1], "opening_b": ids[1] if ids[0] else None,
                              "evidence": "shared_tracking", "confidence": 0.6})

    total_area = sum(d["floor_area"]["value"] for d in room_docs)
    s_total = float(np.sqrt(sum(((d["floor_area"]["ci"]["upper"] - d["floor_area"]["value"]) / Z90) ** 2
                                for d in room_docs)))
    allV = np.concatenate([r.polygon for r in rooms]) if rooms else np.zeros((1, 2))
    ext = np.ptp(allV, axis=0)

    if not drift:
        warnings.append({"code": "stage_not_implemented", "affects": ["property_plan"],
                         "message": "drift correction disabled for this run: poses used as-is"})
    warnings += [
        {"code": "stage_not_implemented", "message": "door and window detection on walls not implemented; "
         "only passages between segmented rooms are reported", "affects": ["openings"]},
        {"code": "stage_not_implemented", "message": "damage detection not implemented", "affects": ["damage_regions"]},
    ]

    if diag is not None:
        snapped = [e for r in rooms for e in r.edges if e.snapped]
        diag["face_concentration_median"] = float(np.median([e.concentration for e in snapped])) if snapped else None
        diag["face_std_median_mm"] = float(1000 * np.median([e.face_std for e in snapped])) if snapped else None
        diag["snapped_edges"] = len(snapped)
    render_plan(rooms, g, wall, out_dir / "plan.png")
    # debug artefacts for the benchmark: the rotation from the capture's world
    # frame to the plan frame, and the fused cloud in the plan frame
    dbg = out_dir / "debug"
    dbg.mkdir(exist_ok=True)
    (dbg / "alignment.json").write_text(json.dumps(
        {"R_world_to_plan": (Ry @ Rg).round(12).tolist(), "yaw_deg": float(np.degrees(yaw)),
         "floor_z": fz, "drift_correction": drift}, indent=1))
    import open3d as o3d
    pc_plan = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    o3d.io.write_point_cloud(str(dbg / "fused_plan_frame.ply"), pc_plan)
    cfg_hash = hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()[:16]
    doc = {
        "schema_version": SCHEMA_VERSION,
        "capture": {"capture_id": cap.name, "tier": "lidar",
                    "device": {"model": "iPad Pro (ARKitScenes)" if cap.source_format == "arkitscenes" else "unknown (not in export)",
                               "os_version": None},
                    "capture_app": {"name": "Stray Scanner" if cap.source_format == "stray" else "ARKitScenes recorder",
                                    "version": None},
                    "source_files": [str(capture_dir)], "captured_at": None},
        "pipeline": {"version": __version__, "git_commit": _git_commit(Path(__file__).resolve().parents[1]),
                     "command": command, "config_hash": f"sha256:{cfg_hash}", "seed": 0, "used_cache": False,
                     "models": [], "hardware": None, "runtime_seconds": round(time.time() - t0, 1)},
        "rooms": room_docs,
        "property_plan": {"adjacency": adjacency, "footprint_area": _meas(total_area, s_total, "m2"),
                          "bounding_extent": {"x": _meas(ext[0], CONFIG["sys_face_m"] * 1.414),
                                              "y": _meas(ext[1], CONFIG["sys_face_m"] * 1.414)},
                          "max_room_overlap_m2": round(_overlap_m2(rooms, g), 4), "unplaced_rooms": [],
                          "drift_correction": drift_doc},
        "damage_regions": [], "concealed_damage_flags": [], "scope_line_items": [],
        "quality": {"warnings": warnings},
        "artifacts": {"rendered_plan": str(out_dir / "plan.png"), "per_room_plans": [], "debug_dir": str(dbg)},
    }
    (out_dir / "output.json").write_text(json.dumps(doc, indent=2))
    return doc
