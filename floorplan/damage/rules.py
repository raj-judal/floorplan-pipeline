"""Damage regions -> schema damage_regions, concealed-damage flags (with the rule that
fired) and scope line items keyed to surfaces.
Rules are deliberately simple and readable; each flag records its rule_id and text.
"""
from __future__ import annotations
import numpy as np
Z90 = 1.645
EXTENT_REL_SIGMA = 0.35      # box-based extent: detector boxes are loose around irregular stains
MARGIN = 0.15                # m added on each side for patch/replace/paint quantities
RULES = [
    {"id": "CEILING_WATER", "text": "A water stain on a ceiling indicates a leak above; the cavity may hold wet insulation or damaged joists.",
     "when": lambda r: r["class"] == "water_stain" and r["kind"] == "ceiling",
     "severity": lambda r: "high" if r["area"] > 0.25 else "medium"},
    {"id": "LOW_WALL_MOISTURE", "text": "Water staining, mould or salt deposits within 0.3 m of the floor suggest a plumbing leak or rising damp behind the wall base.",
     "when": lambda r: r["class"] in ("water_stain", "mold", "efflorescence") and r["kind"] == "wall" and r["v_lo"] < 0.3,
     "severity": lambda r: "medium"},
    {"id": "MOULD_SOURCE", "text": "Visible mould larger than 0.1 m2 implies a persistent moisture source, which may extend behind the surface.",
     "when": lambda r: r["class"] == "mold" and r["area"] > 0.1,
     "severity": lambda r: "high"},
    {"id": "SCORCH_BEHIND", "text": "A scorch mark on a wall or ceiling can indicate heat damage to wiring or framing behind it.",
     "when": lambda r: r["class"] == "scorch" and r["kind"] in ("wall", "ceiling"),
     "severity": lambda r: "high"},
    {"id": "WARPED_FLOOR", "text": "Warped flooring indicates moisture in the subfloor.",
     "when": lambda r: r["class"] == "warping" and r["kind"] == "floor",
     "severity": lambda r: "medium"},
]
ACTION = {"water_stain": "prime_and_paint", "mold": "treat_mold", "crack": "seal_crack", "hole": "patch_drywall",
          "peeling_paint": "prime_and_paint", "efflorescence": "clean", "scorch": "replace_drywall", "warping": "replace_flooring"}
def _meas(v, s, unit, method):
    return {"value": round(float(v), 4), "unit": unit,
            "ci": {"lower": round(float(max(0.0, v - Z90 * s)), 4), "upper": round(float(v + Z90 * s), 4), "level": 0.9},
            "method": method, "n_samples": None}
def to_schema(regions):
    damage, flags, scope = [], [], []
    for i, r in enumerate(regions, start=1):
        u0, u1 = np.median([h.u[0] for h in r.hits]), np.median([h.u[1] for h in r.hits])
        v0, v1 = np.median([h.v[0] for h in r.hits]), np.median([h.v[1] for h in r.hits])
        w, h = max(u1 - u0, 0.02), max(v1 - v0, 0.02)
        did = f"D{i}"
        score = max(x.score for x in r.hits)
        damage.append({
            "id": did, "surface_id": r.surface, "class": r.label, "class_confidence": round(float(score), 3),
            "polygon_surface": [[round(float(a), 3), round(float(b), 3)] for a, b in ((u0, v0), (u1, v0), (u1, v1), (u0, v1))],
            "area": _meas(w * h, EXTENT_REL_SIGMA * np.sqrt(2) * w * h, "m2", "detector_box_projected_to_surface"),
            "extent": {"width": _meas(w, EXTENT_REL_SIGMA * w, "m", "detector_box_projected_to_surface"),
                       "height": _meas(h, EXTENT_REL_SIGMA * h, "m", "detector_box_projected_to_surface")},
            "evidence_frames": [str(x.frame) for x in sorted(r.hits, key=lambda x: -x.score)[:5]],
            "detector": "OWLv2 open-vocabulary (google/owlv2-base-patch16-ensemble)"})
        info = {"class": r.label, "kind": r.kind, "area": w * h, "v_lo": v0}
        flagged = False
        for rule in RULES:
            if rule["when"](info):
                flags.append({"id": f"CF{len(flags) + 1}", "rule_id": rule["id"], "rule_text": rule["text"],
                              "fired_on": [did], "suspected_surfaces": [r.surface], "severity": rule["severity"](info),
                              "rationale": f"{r.label} on {r.kind} {r.surface}, {w * h:.2f} m2, lowest point {v0:.2f} m"})
                flagged = True
        act = ACTION[r.label]
        if act == "seal_crack":
            L = max(w, h)
            q = _meas(L, EXTENT_REL_SIGMA * L, "m", "crack_length_from_extent")
        else:
            A = (w + 2 * MARGIN) * (h + 2 * MARGIN)
            q = _meas(A, EXTENT_REL_SIGMA * A, "m2", "extent_plus_margin")
        scope.append({"id": f"S{len(scope) + 1}", "surface_id": r.surface, "damage_ids": [did], "action": act,
                      "quantity": q, "description": f"{act.replace('_', ' ')} for {r.label.replace('_', ' ')} {did}"})
        if flagged:
            scope.append({"id": f"S{len(scope) + 1}", "surface_id": r.surface, "damage_ids": [did], "action": "inspect_cavity",
                          "quantity": _meas(1, 0, "each", "rule"), "description": f"inspect behind {r.surface} (concealed-damage flag)"})
    return damage, flags, scope
