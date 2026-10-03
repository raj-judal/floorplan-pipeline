"""Thin-tier command path: photo folders or per-room video clips -> output.json + plan.png.
Input layout (capture protocol):
  photo: CAPTURE/<room name>/*.jpg|jpeg|png|heic   one folder per room, 6-8 photos,
         0.5x camera, landscape, back to the middle of each wall, tilted slightly down
  video: CAPTURE/<room name>.mp4|mov                one clip per room, same viewpoints
Depth: Depth Anything V2 Metric Indoor (Small), upright images, then the
camera-height scale correction. Rooms are measured individually and are NOT
stitched into one plan (multi-image assembly failed, lab notebook); the output
says so in quality.warnings and property_plan.unplaced_rooms.
"""
from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps
from .. import __version__
from .rooms_from_images import H_PRIOR, combine_room, finish_room, measure_image
MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
Z90 = 1.645
PHOTO_EXT = {".jpg", ".jpeg", ".png", ".heic"}
VIDEO_EXT = {".mp4", ".mov"}
WORK_WIDTH = 320                    # px; depth is resized to this width for measurement
BACK_GAP = (0.30, 0.10)             # m, camera to the wall behind it (mean, sigma): protocol 'back to the wall'
CEILING_PRIOR = (2.2, 3.2)
def _model_depth_fn():
    import torch
    from transformers import pipeline
    pipe = pipeline("depth-estimation", model=MODEL_ID, device=0 if torch.cuda.is_available() else -1)
    def fn(rgb: np.ndarray):
        return pipe(Image.fromarray(rgb))["predicted_depth"].squeeze().cpu().numpy().astype(np.float32)
    return fn
def _focal35(img: Image.Image):
    try:
        exif = img.getexif()
        sub = exif.get_ifd(0x8769)
        f35 = sub.get(0xA405) or exif.get(0xA405)
        return float(f35) if f35 else None
    except Exception:
        return None
def _intrinsics(w, h, focal35):
    fx = focal35 / 36.0 * max(w, h)          # 35 mm equivalent focal length over a 36 mm frame width
    return np.array([[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1.0]])
def load_photos(folder: Path):
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except ImportError:
        pass
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in PHOTO_EXT:
            img = Image.open(p)
            f35 = _focal35(img)
            img = ImageOps.exif_transpose(img).convert("RGB")     # upright, as the phone displayed it
            yield p.name, np.asarray(img), f35
def load_video(path: Path, fps=2.0):
    cap = cv2.VideoCapture(str(path))                           # OpenCV applies the rotation metadata
    rate = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(rate / fps)))
    i = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if i % step == 0:
            yield f"{path.stem}#{i}", cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), None
        i += 1
def _meas(v, s, unit="m", method=None, n=None):
    return {"value": round(float(v), 4), "unit": unit,
            "ci": {"lower": round(float(v - Z90 * s), 4), "upper": round(float(v + Z90 * s), 4), "level": 0.9},
            "method": method, "n_samples": n}
def _bounds(lo, hi, method):
    return {"value": round((lo + hi) / 2, 4), "unit": "m", "ci": {"lower": lo, "upper": hi, "level": 0.9},
            "method": method, "n_samples": None}
def run_thin(capture: Path, out_dir: Path, tier: str, depth_fn=None, focal35_default=13.0,
             command="python -m floorplan run"):
    t0 = time.time()
    capture, out_dir = Path(capture), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    depth_fn = depth_fn or _model_depth_fn()
    if tier == "photo":
        sources = [(d.name, load_photos(d)) for d in sorted(capture.iterdir()) if d.is_dir()]
    else:
        sources = [(p.stem, load_video(p)) for p in sorted(capture.iterdir()) if p.suffix.lower() in VIDEO_EXT]
    if not sources:
        raise ValueError(f"{capture}: no room {'folders' if tier == 'photo' else 'clips'} found")
    warnings, room_docs, x_cursor = [], [], 0.0
    for k, (room_name, items) in enumerate(sources, start=1):
        rid = f"R{k}"
        measures = []
        for name, rgb, f35 in items:
            depth = depth_fn(rgb)
            h, w = rgb.shape[:2]
            scale_px = WORK_WIDTH / w
            depth = cv2.resize(depth, (WORK_WIDTH, int(round(h * scale_px))), interpolation=cv2.INTER_AREA)
            K = _intrinsics(WORK_WIDTH, depth.shape[0], f35 or focal35_default)
            measures.append(measure_image(name, depth, K, np.array([0, -1.0, 0])))   # upright: up is image-up
        s_room = finish_room(measures, None) if measures else None
        res = combine_room(measures) if s_room else {"dims": [], "ceiling": None, "relative_scale_sigma": None}
        dims = res["dims"]
        rel = res["relative_scale_sigma"] or 0.15
        if s_room is None:
            warnings.append({"code": "scale_ambiguous", "affects": [rid],
                             "message": f"{rid} ({room_name}): no image showed a usable floor, so real-world scale is unknown"})
        # fill missing dimensions from one-sided distances (camera assumed at the opposite wall)
        one = sorted(v for m in measures for v in m.one_sided)
        while len(dims) < 2 and one:
            v = one.pop() + BACK_GAP[0]
            dims.append({"value": v, "sigma": float(np.hypot(rel * v, BACK_GAP[1])) * 1.5, "n": 1, "method": "far_wall_plus_back_gap"})
            warnings.append({"code": "insufficient_overlap", "affects": [rid],
                             "message": f"{rid}: a dimension came from one wall plus an assumed {BACK_GAP[0]} m gap behind the camera; interval widened"})
        if len(dims) < 2:
            warnings.append({"code": "few_images", "affects": [rid],
                             "message": f"{rid} ({room_name}): could not measure both room dimensions; room shown as a 1 m placeholder with no claimed size"})
            dims = (dims + [{"value": 1.0, "sigma": 5.0, "n": 0, "method": "placeholder_not_measured"}] * 2)[:2]
        a, b = sorted(dims, key=lambda d: d["value"])[:2]
        A, B = a["value"], b["value"]
        if res["ceiling"]:
            ceil = _meas(res["ceiling"]["value"], res["ceiling"]["sigma"], method="thin_ceiling_above_camera_plus_prior", n=res["ceiling"]["n"])
        else:
            ceil = _bounds(*CEILING_PRIOR, "bounded_not_observed")
            warnings.append({"code": "ceiling_not_observed", "affects": [f"{rid}-C"],
                             "message": f"{rid}: ceiling not seen; residential prior {CEILING_PRIOR[0]}-{CEILING_PRIOR[1]} m"})
        poly = [[0, 0], [A, 0], [A, B], [0, B]]
        walls, surfaces = [], []
        for i, (s, e, d) in enumerate([((0, 0), (A, 0), a), ((A, 0), (A, B), b), ((A, B), (0, B), a), ((0, B), (0, 0), b)]):
            sid = f"{rid}-W{i + 1}"
            walls.append({"surface_id": sid, "start": list(map(float, s)), "end": list(map(float, e)),
                          "length": _meas(d["value"], d["sigma"], method=d["method"], n=d["n"]), "height": ceil})
            surfaces.append({"id": sid, "kind": "wall", "area": _meas(d["value"] * ceil["value"], d["sigma"] * ceil["value"], "m2")})
        s_area = A * B * np.hypot(a["sigma"] / A, b["sigma"] / B)
        surfaces += [{"id": f"{rid}-F", "kind": "floor", "area": _meas(A * B, s_area, "m2")},
                     {"id": f"{rid}-C", "kind": "ceiling", "area": _meas(A * B, s_area, "m2")}]
        room_docs.append({
            "id": rid, "label": room_name, "is_connector": False, "floor_polygon": poly,
            "transform_to_property": {"translation": [round(x_cursor, 4), 0.0], "rotation_deg": 0.0},
            "surfaces": surfaces, "walls": walls, "openings": [], "ceiling_height": ceil,
            "floor_area": _meas(A * B, s_area, "m2", method="rectangle_of_measured_dimensions"),
            "perimeter": _meas(2 * (A + B), 2 * np.hypot(a["sigma"], b["sigma"]), method="rectangle_of_measured_dimensions")})
        x_cursor += A + 0.5
    warnings += [
        {"code": "room_unstitched", "affects": [d["id"] for d in room_docs],
         "message": "rooms are measured individually and laid out side by side; they are NOT stitched into one plan (thin-tier assembly not delivered)"},
        {"code": "stage_not_implemented", "affects": ["openings"], "message": "openings are not detected in the photo and video tiers"},
        {"code": "insufficient_overlap", "affects": [d["id"] for d in room_docs],
         "message": "photo/video room dimensions are rough: on rendered protocol views their 90% intervals contained 9 of 13 true dimensions; large or L-shaped rooms are underestimated by 30-45%"},
        {"code": "stage_not_implemented", "affects": ["damage_regions"], "message": "damage detection not implemented"},
    ]
    total = sum(d["floor_area"]["value"] for d in room_docs)
    s_total = float(np.sqrt(sum(((d["floor_area"]["ci"]["upper"] - d["floor_area"]["value"]) / Z90) ** 2 for d in room_docs)))
    doc = {
        "schema_version": "1.1.0",
        "capture": {"capture_id": capture.name, "tier": tier, "device": {"model": "unknown (from image metadata if present)", "os_version": None},
                    "capture_app": {"name": "Camera (native)", "version": None}, "source_files": [str(capture)], "captured_at": None},
        "pipeline": {"version": __version__, "git_commit": None, "command": command,
                     "config_hash": "sha256:" + hashlib.sha256(json.dumps({"H_PRIOR": H_PRIOR, "BACK_GAP": BACK_GAP, "WORK_WIDTH": WORK_WIDTH}).encode()).hexdigest()[:16],
                     "seed": 0, "used_cache": False,
                     "models": [{"name": "Depth Anything V2 Metric Indoor Small", "version": MODEL_ID, "used_for": "per-image metric depth (scale corrected by camera height)", "source_url": f"https://huggingface.co/{MODEL_ID}", "license": None, "weights_sha256": None}],
                     "hardware": None, "runtime_seconds": round(time.time() - t0, 1)},
        "rooms": room_docs,
        "property_plan": {"adjacency": [], "footprint_area": _meas(total, s_total, "m2"),
                          "max_room_overlap_m2": 0.0, "unplaced_rooms": [d["id"] for d in room_docs],
                          "drift_correction": {"method": "none", "enabled": False, "loop_closures": None, "residual_before_m": None, "residual_after_m": None}},
        "damage_regions": [], "concealed_damage_flags": [], "scope_line_items": [],
        "quality": {"warnings": warnings},
        "artifacts": {"rendered_plan": str(out_dir / "plan.png"), "per_room_plans": [], "debug_dir": None},
    }
    _render(room_docs, out_dir / "plan.png")
    (out_dir / "output.json").write_text(json.dumps(doc, indent=2))
    return doc
def _render(rooms, path, px_per_m=120, pad=40):
    xs = [r["transform_to_property"]["translation"][0] + p[0] for r in rooms for p in r["floor_polygon"]]
    ys = [p[1] for r in rooms for p in r["floor_polygon"]]
    W, H = int(max(xs) * px_per_m) + 2 * pad, int(max(ys) * px_per_m) + 2 * pad + 30
    img = np.full((H, W, 3), 255, np.uint8)
    for r in rooms:
        ox = r["transform_to_property"]["translation"][0]
        q = np.array([[pad + (ox + x) * px_per_m, H - pad - y * px_per_m] for x, y in r["floor_polygon"]], np.int32)
        cv2.polylines(img, [q], True, (60, 60, 60), 2)
        A, B = r["walls"][0]["length"], r["walls"][1]["length"]
        c = q.mean(0).astype(int)
        cv2.putText(img, f"{r['id']} {r['label']}", (c[0] - 40, c[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        for m, off in ((A, 0), (B, 18)):
            cv2.putText(img, f"{m['value']:.2f} m [{m['ci']['lower']:.2f}-{m['ci']['upper']:.2f}]", (c[0] - 70, c[1] + 12 + off), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 160), 1)
    cv2.putText(img, "Rooms measured individually; not stitched", (pad, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 1)
    cv2.imwrite(str(path), img)
