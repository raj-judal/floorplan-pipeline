"""Drift ablation: run the same capture with drift correction off and on.

Usage: python tools/drift_ablation.py CAPTURE_DIR OUT_DIR [--stride 5]

Writes OUT_DIR/off and OUT_DIR/on (output.json + plan.png each) and
OUT_DIR/ablation.json comparing footprint, rooms and wall-face sharpness.
Drift shows up as doubled walls, which widens the spread of points on a
snapped wall face; correction should tighten it.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from floorplan.geometry.drift import correct_drift
from floorplan.io import load_capture
from floorplan.pipeline import run

ap = argparse.ArgumentParser()
ap.add_argument("capture", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("--stride", type=int, default=None)
a = ap.parse_args()

dr = correct_drift(load_capture(a.capture))
rows = {}
for name, on in (("off", False), ("on", True)):
    diag = {}
    doc = run(a.capture, a.out / name, stride=a.stride, drift=on, drift_result=dr if on else None,
              command=f"python tools/drift_ablation.py {a.capture} {a.out}", diag=diag)
    rows[name] = {
        "rooms": len(doc["rooms"]),
        "footprint_m2": doc["property_plan"]["footprint_area"]["value"],
        "bounding_extent_m": [doc["property_plan"]["bounding_extent"][k]["value"] for k in ("x", "y")],
        "room_areas_m2": [d["floor_area"]["value"] for d in doc["rooms"]],
        **diag,
    }
rows["drift_stats"] = dr.stats
(a.out / "ablation.json").write_text(json.dumps(rows, indent=2))
print(json.dumps(rows, indent=2))
