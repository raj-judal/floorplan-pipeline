"""Command line: one command per capture.

    python -m floorplan run CAPTURE_DIR --tier lidar --out OUT_DIR
"""
import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m floorplan")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="process one capture")
    r.add_argument("capture", type=Path)
    r.add_argument("--tier", choices=["lidar", "video", "photo"], default="lidar")
    r.add_argument("--out", type=Path, default=None, help="output folder (default: out/<capture name>)")
    r.add_argument("--stride", type=int, default=None, help="use every n-th frame (default from CONFIG)")
    r.add_argument("--no-drift", action="store_true", help="disable drift correction (for the ablation)")
    r.add_argument("--no-damage", action="store_true", help="skip damage detection (LiDAR tier)")
    a = ap.parse_args(argv)

    out = a.out or Path("out") / a.capture.name
    cmd = "python -m floorplan " + " ".join(argv if argv is not None else sys.argv[1:])
    if a.tier == "lidar":
        from .pipeline import run
        doc = run(a.capture, out, stride=a.stride, command=cmd, drift=not a.no_drift, damage=not a.no_damage)
    else:
        from .thin.pipeline_thin import run_thin
        doc = run_thin(a.capture, out, a.tier, command=cmd)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from validate_output import main as validate
    status = validate(str(out / "output.json"))
    rooms = [(d["id"], d["floor_area"]["value"]) for d in doc["rooms"]]
    print(f"rooms: {rooms}")
    print(f"warnings: {len(doc['quality']['warnings'])}, runtime {doc['pipeline']['runtime_seconds']} s")
    print(f"wrote {out / 'output.json'} and {out / 'plan.png'}")
    return status


if __name__ == "__main__":
    sys.exit(main())
