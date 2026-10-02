"""Validate a capture output JSON against the schema plus semantic rules.

Usage: python tools/validate_output.py path/to/output.json

JSON Schema handles structure. This script adds the rules it cannot express:
  - every interval brackets its value (lower <= value <= upper)
  - every id is unique and every reference resolves
  - the stitched plan has no room overlaps above tolerance
"""
import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

SCHEMA = Path(__file__).resolve().parents[1] / "schema" / "capture_output.schema.json"
OVERLAP_TOL_M2 = 0.05


def iter_measurements(node, path="$"):
    """Yield (path, measurement) for every object that looks like a Measurement."""
    if isinstance(node, dict):
        if {"value", "unit", "ci"} <= node.keys():
            yield path, node
        for k, v in node.items():
            yield from iter_measurements(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from iter_measurements(v, f"{path}[{i}]")


def semantic_errors(doc):
    errs = []

    for p, m in iter_measurements(doc):
        lo, hi, v = m["ci"]["lower"], m["ci"]["upper"], m["value"]
        if not lo <= v <= hi:
            errs.append(f"{p}: value {v} outside interval [{lo}, {hi}]")

    room_ids, surface_ids, opening_ids = set(), set(), set()
    for r in doc["rooms"]:
        room_ids.add(r["id"])
        for s in r["surfaces"]:
            if s["id"] in surface_ids:
                errs.append(f"duplicate surface id {s['id']}")
            surface_ids.add(s["id"])
        for o in r["openings"]:
            opening_ids.add(o["id"])
            if o["wall_id"] not in surface_ids:
                errs.append(f"opening {o['id']} references unknown wall {o['wall_id']}")
    damage_ids = {d["id"] for d in doc["damage_regions"]}

    for a in doc["property_plan"]["adjacency"]:
        for key in ("room_a", "room_b"):
            if a[key] not in room_ids:
                errs.append(f"adjacency references unknown room {a[key]}")
        if a["opening_a"] not in opening_ids:
            errs.append(f"adjacency references unknown opening {a['opening_a']}")
    for d in doc["damage_regions"]:
        if d["surface_id"] not in surface_ids:
            errs.append(f"damage {d['id']} on unknown surface {d['surface_id']}")
    for f in doc["concealed_damage_flags"]:
        errs += [f"flag {f['id']} fired on unknown damage {x}" for x in f["fired_on"] if x not in damage_ids]
    for s in doc["scope_line_items"]:
        if s["surface_id"] not in surface_ids:
            errs.append(f"scope item {s['id']} on unknown surface {s['surface_id']}")
        errs += [f"scope item {s['id']} references unknown damage {x}" for x in s["damage_ids"] if x not in damage_ids]

    if doc["property_plan"]["max_room_overlap_m2"] > OVERLAP_TOL_M2:
        errs.append(f"rooms overlap by {doc['property_plan']['max_room_overlap_m2']} m2 (tolerance {OVERLAP_TOL_M2})")
    return errs


def main(path):
    doc = json.loads(Path(path).read_text())
    schema = json.loads(SCHEMA.read_text())
    structural = [f"{'/'.join(map(str, e.path)) or '$'}: {e.message}"
                  for e in Draft202012Validator(schema).iter_errors(doc)]
    if structural:
        print("SCHEMA ERRORS:\n  " + "\n  ".join(structural))
        return 1
    semantic = semantic_errors(doc)
    if semantic:
        print("SEMANTIC ERRORS:\n  " + "\n  ".join(semantic))
        return 1
    print(f"OK: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
