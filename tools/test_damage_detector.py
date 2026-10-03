"""False-alarm test for the damage detector on an UNDAMAGED capture.
usage: python tools/test_damage_detector.py STRAY_CAPTURE OUT_DIR [--frames 60] [--model owlv2|gdino]
Every detection on a capture with no damage is a false alarm. Reports, per
confidence threshold, how many frames raise one, plus timing; saves the most
confident detections as annotated images to see what fooled the detector.
Frames are turned upright using the capture's gravity (as the pipeline does).
"""
import argparse
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.transform import Rotation
MODELS = {"owlv2": "google/owlv2-base-patch16-ensemble", "gdino": "IDEA-Research/grounding-dino-tiny",
          "owlvit": "google/owlvit-base-patch32"}
PROMPTS = {
    "water stain on a wall or ceiling": "water_stain",
    "mould on a wall": "mold",
    "crack in a wall": "crack",
    "hole in a wall": "hole",
    "peeling paint": "peeling_paint",
    "white salt deposits on a wall": "efflorescence",
    "burn or scorch mark": "scorch",
    "warped floorboards": "warping",
}
ROT = {0: None, 90: cv2.ROTATE_90_COUNTERCLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_CLOCKWISE}
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--model", choices=["owlv2", "owlvit"], default="owlvit")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from floorplan.damage.detector import Detector
    t0 = time.time()
    det = Detector(a.model)
    print(f"model {a.model} on {det.device} ({det.dtype}), loaded in {time.time() - t0:.1f} s")
    odo = pd.read_csv(a.capture / "odometry.csv", skipinitialspace=True)
    cap = cv2.VideoCapture(str(a.capture / "rgb.mp4"))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rows, times, images = [], [], {}
    for i in np.linspace(n * 0.03, n * 0.97, a.frames).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, bgr = cap.read()
        if not ok:
            continue
        r = odo.iloc[int(i)]
        up = Rotation.from_quat([r.qx, r.qy, r.qz, r.qw]).as_matrix().T @ np.array([0, 1.0, 0])
        k = int(round(np.degrees(np.arctan2(up[0], -up[1])) / 90.0)) % 4 * 90
        img = bgr if ROT[k] is None else cv2.rotate(bgr, ROT[k])
        img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
        t = time.time()
        res = det(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), threshold=0.05)
        times.append(time.time() - t)
        print(f"  frame {int(i)}: {times[-1]:.1f} s, {len(res)} raw detections", flush=True)
        images[int(i)] = img
        for d in res:
            rows.append({"frame": int(i), "label": d["label"], "score": d["score"], "box": d["box"]})
    df = pd.DataFrame(rows)
    nf = len(times)
    print(f"{nf} frames, {np.median(times):.2f} s per frame")
    for th in (0.1, 0.2, 0.3, 0.4, 0.5):
        s = df[df.score >= th] if len(df) else df
        fr = s.frame.nunique() if len(s) else 0
        print(f"threshold {th:.1f}: {len(s)} detections in {fr} of {nf} frames"
              + (f"; by class {s.label.value_counts().to_dict()}" if len(s) else ""))
    if len(df):
        top = df.sort_values("score", ascending=False).head(12)
        for j, d in enumerate(top.itertuples()):
            im = images[d.frame].copy()
            x0, y0, x1, y1 = map(int, d.box)
            cv2.rectangle(im, (x0, y0), (x1, y1), (0, 0, 255), 3)
            cv2.putText(im, f"{d.label} {d.score:.2f}", (x0, max(20, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imwrite(str(a.out / f"top{j:02d}_frame{d.frame}_{d.label}_{d.score:.2f}.jpg"), im)
        df.to_csv(a.out / "detections.csv", index=False)
        print(f"saved the 12 most confident detections to {a.out}")
if __name__ == "__main__":
    main()
