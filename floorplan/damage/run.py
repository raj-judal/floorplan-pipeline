"""Damage stage of the LiDAR tier: sample video frames, detect, place on surfaces, merge, rules."""
from __future__ import annotations
import cv2
import numpy as np
from .project import ROT, hit_from_box, merge, surfaces_from_doc, upright_rotation
from .rules import to_schema
def detect_damage(cap, R_align, floor_z, room_docs, n_frames=40, threshold=0.3, model="owlv2", detector=None):
    """Returns (damage_regions, concealed_damage_flags, scope_line_items, warnings, stats)."""
    warnings = []
    video = cap.root / "rgb.mp4"
    if cap.source_format != "stray" or not video.exists():
        warnings.append({"code": "stage_not_implemented", "affects": ["damage_regions"],
                         "message": "damage detection needs the capture's RGB video (Stray Scanner); none in this capture"})
        return [], [], [], warnings, {}
    if detector is None:
        try:
            from .detector import Detector
            detector = Detector(model)
        except Exception as e:                                   # no torch/transformers, or no download
            warnings.append({"code": "stage_not_implemented", "affects": ["damage_regions"],
                             "message": f"damage detector unavailable ({type(e).__name__}); damage not assessed"})
            return [], [], [], warnings, {}
    up = np.zeros(3)
    up[cap.up_axis] = cap.up_sign
    rooms = surfaces_from_doc({"rooms": room_docs})
    vc = cv2.VideoCapture(str(video))
    W, H = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = len(cap.frames)
    picks = np.unique(np.linspace(n * 0.03, n * 0.97, n_frames).astype(int))
    hits, raw = [], 0
    for i in picks:
        f = cap.frames[i]
        vc.set(cv2.CAP_PROP_POS_FRAMES, int(f.index))
        ok, bgr = vc.read()
        if not ok:
            continue
        k = upright_rotation(f.T_wc, up)
        img = bgr if ROT[k] is None else cv2.rotate(bgr, ROT[k])
        small = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
        dets = detector(cv2.cvtColor(small, cv2.COLOR_BGR2RGB), threshold=threshold)
        raw += len(dets)
        if not dets:
            continue
        depth, conf = cap.read_depth(f), cap.read_confidence(f)
        for d in dets:
            box = tuple(2 * c for c in d["box"])                 # back to full resolution, still upright
            h = hit_from_box(f, depth, conf, box, k, (W, H), R_align, rooms, floor_z, d["label"], d["score"], int(f.index))
            if h is not None:
                hits.append(h)
    regions = merge(hits)
    damage, flags, scope = to_schema(regions)
    warnings.append({"code": "detector_unvalidated", "affects": ["damage_regions"],
                     "message": (f"damage detection (OWLv2, threshold {threshold}) checked {len(picks)} frames: {raw} detections, "
                                 f"{len(hits)} placed on surfaces, {len(damage)} regions seen in 2+ frames. Never tested on real "
                                 "damage in a captured room. On close-up photos of real damage at this threshold it named the right "
                                 "class for 75% of water stains, 25% of peeling paint, 12% of holes and no cracks; on an undamaged "
                                 "apartment it raised 5 false detections in 60 frames before the two-view rule. Treat regions as "
                                 "leads to inspect, not measurements.")})
    return damage, flags, scope, warnings, {"frames": int(len(picks)), "raw": raw, "placed": len(hits), "regions": len(damage)}
