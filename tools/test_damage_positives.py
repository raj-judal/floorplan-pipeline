"""How often the detector finds REAL damage (photos from Wikimedia Commons).
usage: python tools/test_damage_positives.py data/damage_examples OUT_DIR [--model owlvit|owlv2]
For each class folder: share of images where the detector reports that class
(and any damage class) at each confidence threshold. Pair with
tools/test_damage_detector.py (false alarms on an undamaged capture) to pick
the threshold. Annotated images of the best detection per image go to OUT_DIR.
"""
import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from floorplan.damage.detector import Detector
ap = argparse.ArgumentParser()
ap.add_argument("examples", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("--model", choices=["owlvit", "owlv2"], default="owlvit")
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
det = Detector(a.model)
TH = (0.05, 0.1, 0.2, 0.3)
summary = []
for cls_dir in sorted(d for d in a.examples.iterdir() if d.is_dir()):
    cls = cls_dir.name
    best_cls, best_any = [], []
    for p in sorted(cls_dir.glob("*.jpg")):
        rgb = np.asarray(ImageOps.exif_transpose(Image.open(p)).convert("RGB"))
        dets = det(rgb, threshold=0.01)
        sc = [d["score"] for d in dets if d["label"] == cls]
        best_cls.append(max(sc) if sc else 0.0)
        best_any.append(max([d["score"] for d in dets], default=0.0))
        if dets:
            d = max(dets, key=lambda d: d["score"])
            im = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            x0, y0, x1, y1 = map(int, d["box"])
            cv2.rectangle(im, (x0, y0), (x1, y1), (0, 0, 255), 3)
            cv2.putText(im, f"{d['label']} {d['score']:.2f}", (x0, max(25, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.imwrite(str(a.out / f"{cls}_{p.stem}.jpg"), im)
    bc, ba = np.array(best_cls), np.array(best_any)
    row = " | ".join(f">={t}: right class {np.mean(bc >= t):.0%}, any {np.mean(ba >= t):.0%}" for t in TH)
    print(f"{cls} ({len(bc)} images): {row}")
