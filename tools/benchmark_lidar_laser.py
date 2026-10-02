"""Score LiDAR-tier pipeline outputs against Faro laser ground truth (ARKitScenes).

Usage:
    python tools/benchmark_lidar_laser.py LASER_CROP.ply OUT_REPORT.json \
        --run 42444966=out/ark_42444966 --run 42444968=out/ark_42444968

Each --run is CAPTURE_ID=PIPELINE_OUTPUT_DIR (from `python -m floorplan run`).
Registration iPad->laser is read from benchmarks/arkitscenes_421383/ and refined
with ICP on the pipeline's own fused cloud (drift correction moves the frame
slightly). Registration is rigid with no scale, so it cannot hide scale errors,
and ground truth is MEASURED from laser points, never copied from the pipeline:
the pipeline's walls only say where to look.

Gates (brief, Part 2): ceiling height <= 15 mm per room; ceiling spread across
repeat captures <= 10 mm; repeat captures agree within max(1 cm, 0.5%) per wall.
Calibration: share of ground-truth values inside the 90% intervals.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import open3d as o3d
from matplotlib.path import Path as MPath

from floorplan.geometry.planes import horizontal_planes, with_normals

BENCH = ROOT / "benchmarks" / "arkitscenes_421383"
GATE_CEIL, GATE_CEIL_SPREAD = 0.015, 0.010


def _h(T, P):
    return P @ T[:3, :3].T + T[:3, 3]


def plan_to_laser(run_dir, capture_id, laser):
    reg = json.loads((BENCH / f"registration_{capture_id}.json").read_text())
    T_wl = np.array(reg["T_ipad_to_laser"])
    al = json.loads((run_dir / "debug" / "alignment.json").read_text())
    R = np.eye(4)
    R[:3, :3] = np.array(al["R_world_to_plan"])
    T0 = T_wl @ np.linalg.inv(R)
    src = o3d.io.read_point_cloud(str(run_dir / "debug" / "fused_plan_frame.ply")).voxel_down_sample(0.02)
    tgt = laser.voxel_down_sample(0.02)
    tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.06, max_nn=30))
    T = T0
    for thr in (0.10, 0.05, 0.025):
        r = o3d.pipelines.registration.registration_icp(
            src, tgt, thr, T, o3d.pipelines.registration.TransformationEstimationPointToPlane())
        T = r.transformation
    return T, float(r.inlier_rmse)


def gt_face(LP, LN, Lfloor, s, d, n, L):
    """Measured laser wall face near the line s + t*d (plan frame); offset along n, or None."""
    rel = LP[:, :2] - s
    off = rel @ n
    t = rel @ d
    H = LP[:, 2] - Lfloor.z_at(LP[:, 0], LP[:, 1])
    sel = (np.abs(off) < 0.12) & (t > 0.1) & (t < L - 0.1) & (H > 0.3) & (H < 2.3) & \
          (np.abs(LN[:, :2] @ n) > 0.95)
    if sel.sum() < 50:
        return None
    hist, edges = np.histogram(off[sel], bins=np.arange(-0.12, 0.125, 0.01))
    cands = [(abs(edges[i] + 0.005), edges[i] + 0.005) for i in range(len(hist)) if hist[i] >= 50]
    if not cands:
        return None
    peak = min(cands)[1]
    near = sel & (np.abs(off - peak) < 0.015)
    return float(np.median(off[near]))


def score_run(capture_id, run_dir, LP_l, LN_l, laser_pc):
    doc = json.loads((run_dir / "output.json").read_text())
    T, rmse = plan_to_laser(run_dir, capture_id, laser_pc)
    Tinv = np.linalg.inv(T)
    LP, LN = _h(Tinv, LP_l), LN_l @ Tinv[:3, :3].T                 # laser in the plan frame
    Lfloor, Lceil = horizontal_planes(LP, LN)
    rooms = []
    for r in doc["rooms"]:
        o = np.array(r["transform_to_property"]["translation"])
        poly = np.array(r["floor_polygon"]) + o
        # ceiling ground truth: laser planes averaged over the pipeline's room footprint
        xs = np.arange(poly[:, 0].min(), poly[:, 0].max(), 0.05)
        ys = np.arange(poly[:, 1].min(), poly[:, 1].max(), 0.05)
        G = np.array(np.meshgrid(xs, ys)).reshape(2, -1).T
        G = G[MPath(poly).contains_points(G)]
        gt_ceil = float(np.mean(Lceil.z_at(G[:, 0], G[:, 1]) - Lfloor.z_at(G[:, 0], G[:, 1])))
        ch = r["ceiling_height"]
        # walls: measured laser face for each pipeline wall, then corners and lengths
        walls = r["walls"]
        lines = []
        for w in walls:
            s, e = np.array(w["start"]) + o, np.array(w["end"]) + o
            L = float(np.linalg.norm(e - s))
            d = (e - s) / L
            n = np.array([-d[1], d[0]])
            f = gt_face(LP, LN, Lfloor, s, d, n, L)
            lines.append(None if f is None else (s + f * n, d))
        wall_rows = []
        for i, w in enumerate(walls):
            a, b, c = lines[i - 1], lines[i], lines[(i + 1) % len(lines)]
            if a is None or b is None or c is None:
                continue

            def cross(l1, l2):
                A = np.c_[l1[1], -l2[1]]
                t = np.linalg.solve(A, l2[0] - l1[0])
                return l1[0] + t[0] * l1[1]
            gt_len = float(np.linalg.norm(cross(b, c) - cross(a, b)))
            m = w["length"]
            mid_laser = _h(T, np.r_[(np.array(w["start"]) + np.array(w["end"])) / 2 + o, 0.0][None])[0]
            dir_laser = T[:3, :3] @ np.r_[np.array(w["end"]) - np.array(w["start"]), 0.0]
            wall_rows.append({"surface_id": w["surface_id"], "pred_m": m["value"], "gt_m": round(gt_len, 4),
                              "err_mm": round(1000 * (m["value"] - gt_len), 1),
                              "in_ci": m["ci"]["lower"] <= gt_len <= m["ci"]["upper"],
                              "mid_laser": mid_laser.round(3).tolist(),
                              "dir_laser": (dir_laser / np.linalg.norm(dir_laser)).round(4).tolist()})
        rooms.append({"room": r["id"], "ceiling_pred_m": ch["value"], "ceiling_gt_m": round(gt_ceil, 4),
                      "ceiling_err_mm": round(1000 * (ch["value"] - gt_ceil), 1),
                      "ceiling_method": ch["method"],
                      "ceiling_in_ci": ch["ci"]["lower"] <= gt_ceil <= ch["ci"]["upper"],
                      "ceiling_pass": abs(ch["value"] - gt_ceil) <= GATE_CEIL,
                      "centroid_laser": _h(T, np.r_[poly.mean(axis=0), 0.0][None])[0].round(3).tolist(),
                      "walls": wall_rows})
    return {"capture": capture_id, "registration_rmse_mm": round(1000 * rmse, 1), "rooms": rooms}


def repeatability(runs):
    """Match rooms and walls across captures in the laser frame."""
    out = {"ceiling": [], "walls": []}
    if len(runs) < 2:
        return out
    a, b = runs[0], runs[1]
    for ra in a["rooms"]:
        rb = min(b["rooms"], key=lambda r: np.linalg.norm(np.subtract(r["centroid_laser"], ra["centroid_laser"])))
        if np.linalg.norm(np.subtract(rb["centroid_laser"], ra["centroid_laser"])) > 1.0:
            continue
        spread = abs(ra["ceiling_pred_m"] - rb["ceiling_pred_m"])
        bias = np.mean([ra["ceiling_err_mm"], rb["ceiling_err_mm"]])
        out["ceiling"].append({"rooms": [ra["room"], rb["room"]], "spread_mm": round(1000 * spread, 1),
                               "mean_err_mm": round(float(bias), 1), "pass": spread <= GATE_CEIL_SPREAD})
        for wa in ra["walls"]:
            best = None
            for wb in rb["walls"]:
                if abs(np.dot(wa["dir_laser"], wb["dir_laser"])) < 0.99:
                    continue
                dist = np.linalg.norm(np.subtract(wa["mid_laser"], wb["mid_laser"]))
                if dist < 0.3 and (best is None or dist < best[0]):
                    best = (dist, wb)
            if best:
                wb = best[1]
                diff = abs(wa["pred_m"] - wb["pred_m"])
                tol = max(0.01, 0.005 * (wa["pred_m"] + wb["pred_m"]) / 2)
                out["walls"].append({"walls": [wa["surface_id"], wb["surface_id"]], "lengths_m": [wa["pred_m"], wb["pred_m"]],
                                     "diff_mm": round(1000 * diff, 1), "tol_mm": round(1000 * tol, 1), "pass": diff <= tol})
    return out


def summarise(runs, rep):
    ceil = [r for run in runs for r in run["rooms"]]
    walls = [w for run in runs for r in run["rooms"] for w in r["walls"]]
    cov = [r["ceiling_in_ci"] for r in ceil] + [w["in_ci"] for w in walls]
    errs = [abs(w["err_mm"]) for w in walls]
    rw = rep["walls"]
    rc = rep["ceiling"]
    return {
        "ceiling_height": {"gate": "<= 15 mm per room", "errors_mm": [r["ceiling_err_mm"] for r in ceil],
                           "pass_rate": float(np.mean([r["ceiling_pass"] for r in ceil])) if ceil else None},
        "ceiling_repeatability": {"gate": "spread <= 10 mm", "spread_mm": [c["spread_mm"] for c in rc],
                                  "diagnosis": None if not rc else (
                                      "repeatable-but-biased" if all(c["pass"] for c in rc) and any(abs(c["mean_err_mm"]) > 15 for c in rc)
                                      else "unrepeatable" if not all(c["pass"] for c in rc) else "pass")},
        "wall_repeatability": {"gate": "max(1 cm, 0.5%) per wall", "pairs": len(rw),
                               "pass_rate": float(np.mean([w["pass"] for w in rw])) if rw else None,
                               "median_diff_mm": float(np.median([w["diff_mm"] for w in rw])) if rw else None},
        "wall_length_accuracy": {"walls_with_ground_truth": len(walls),
                                 "median_abs_err_mm": float(np.median(errs)) if errs else None,
                                 "p90_abs_err_mm": float(np.percentile(errs, 90)) if errs else None},
        "calibration": {"target": 0.90, "coverage": float(np.mean(cov)) if cov else None, "n": len(cov)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("laser", type=Path)
    ap.add_argument("report", type=Path)
    ap.add_argument("--run", action="append", required=True, help="CAPTURE_ID=OUTPUT_DIR")
    a = ap.parse_args()
    laser = o3d.io.read_point_cloud(str(a.laser))
    LP, LN = with_normals(laser, voxel=0.02)
    runs = []
    for spec in a.run:
        cid, d = spec.split("=", 1)
        runs.append(score_run(cid, Path(d), LP, LN, laser))
    rep = repeatability(runs)
    report = {"summary": summarise(runs, rep), "repeatability": rep, "runs": runs}
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
