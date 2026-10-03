"""Real test of camera-height scale correction for the thin tiers.
Usage: python tools/test_depth_model.py PATH/TO/STRAY_CAPTURE [--frames 30] [--height 1.40]
Runs the metric depth model on upright frames, finds the floor in the
model's own depth, and rescales each frame so the camera sits --height m
above the floor. Compares raw and corrected scale against LiDAR.
Gravity comes from the capture's poses here (plain video will need its own).
"""
import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.transform import Rotation
MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
ROT = {0: None, 90: cv2.ROTATE_90_COUNTERCLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_CLOCKWISE}
BACK = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
def ccw_needed(up_img):
    ang = np.degrees(np.arctan2(up_img[0], -up_img[1]))
    return int(round(ang / 90.0)) % 4 * 90
def rot(img, k, table):
    return img if table[k] is None else cv2.rotate(img, table[k])
def camera_height(depth, K, R_wc, up=np.array([0.0, 1.0, 0.0]), min_share=0.03, k=3):
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    z = depth
    P = np.stack([(u + 0.5 - K[0, 2]) / K[0, 0] * z, (v + 0.5 - K[1, 2]) / K[1, 1] * z, z], -1) @ R_wc.T
    dx = P[k:-k, 2 * k:] - P[k:-k, :-2 * k]
    dy = P[2 * k:, k:-k] - P[:-2 * k, k:-k]
    n = np.cross(dx, dy)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    Pc = P[k:-k, k:-k]
    valid = (z[k:-k, k:-k] > 0) & (z[k:-k, 2 * k:] > 0) & (z[k:-k, :-2 * k] > 0) & (z[2 * k:, k:-k] > 0) & (z[:-2 * k, k:-k] > 0)
    if valid.sum() < 500:
        return None
    h = Pc[valid] @ up
    horiz = np.abs(n[valid] @ up) > 0.9
    scale = np.median(z[z > 0])
    width = 0.02 * scale
    hh = h[horiz]
    if len(hh) < 0.03 * len(h):
        return None
    bins = np.arange(hh.min(), hh.max() + width, width)
    c, e = np.histogram(hh, bins=bins)
    share = (c + np.r_[c[1:], 0] + np.r_[0, c[:-1]]) / len(h)
    cand = np.nonzero(share >= min_share)[0]
    if len(cand) == 0:
        return None
    i = cand[0]
    floor = np.median(hh[(hh >= e[max(i - 1, 0)]) & (hh <= e[min(i + 2, len(e) - 1)])])
    if floor > -0.4 * scale:
        return None
    return float(-floor)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", type=Path)
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--height", type=float, default=1.40)
    a = ap.parse_args()
    import torch
    from transformers import pipeline
    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("depth-estimation", model=MODEL, device=device)
    odo = pd.read_csv(a.capture / "odometry.csv", skipinitialspace=True)
    cap = cv2.VideoCapture(str(a.capture / "rgb.mp4"))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        sys.exit("could not read rgb.mp4")
    raw, corr, hms = [], [], []
    for i in np.linspace(n * 0.05, n * 0.95, a.frames).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, bgr = cap.read()
        if not ok:
            continue
        lidar = np.asarray(Image.open(a.capture / "depth" / f"{i:06d}.png"), dtype=np.float32) / 1000
        conf = np.asarray(Image.open(a.capture / "confidence" / f"{i:06d}.png"))
        m = (conf == 2) & (lidar > 0.3) & (lidar < 5)
        if m.sum() < 500:
            continue
        r = odo.iloc[int(i)]
        R_wc = Rotation.from_quat([r.qx, r.qy, r.qz, r.qw]).as_matrix()
        k = ccw_needed((R_wc.T @ np.array([0.0, 1.0, 0.0]))[:2])
        img = rot(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), k, ROT)
        pred = pipe(Image.fromarray(img))["predicted_depth"].squeeze().cpu().numpy().astype(np.float32)
        pred = cv2.resize(rot(pred, k, BACK), (lidar.shape[1], lidar.shape[0]), interpolation=cv2.INTER_AREA)
        s = float(np.median(pred[m] / lidar[m]))
        sx = lidar.shape[1] / 1920.0
        K = np.array([[r.fx * sx, 0, r.cx * sx], [0, r.fy * sx, r.cy * sx], [0, 0, 1.0]])
        hm = camera_height(pred, K, R_wc)
        raw.append(s)
        if hm is None:
            print(f"frame {i:5d}: raw scale {s:.3f}  (no floor visible)")
        else:
            c = s * a.height / hm
            corr.append(c)
            hms.append(hm)
            print(f"frame {i:5d}: raw scale {s:.3f}  camera height in model {hm:.2f} m  corrected scale {c:.3f}")
    raw, corr = np.array(raw), np.array(corr)
    print(f"\nraw      : {len(raw)} frames, median {np.median(raw):.3f}, spread (10-90%) {np.percentile(raw, 10):.3f}-{np.percentile(raw, 90):.3f}")
    if len(corr):
        print(f"corrected: {len(corr)} frames, median {np.median(corr):.3f}, spread (10-90%) {np.percentile(corr, 10):.3f}-{np.percentile(corr, 90):.3f}")
        # the pipeline's plausibility rule (floorplan/thin/rooms_from_images.py PLAUSIBLE) rejects tabletop 'floors'
        hms = np.array(hms)
        keep = (hms / a.height >= 0.9) & (hms / a.height <= 1.8)
        v = corr[keep]
        print(f"plausible: {len(v)} frames, median {np.median(v):.3f}, spread (10-90%) {np.percentile(v, 10):.3f}-{np.percentile(v, 90):.3f}, 90% within {100 * np.percentile(np.abs(v - 1), 90):.0f}%")
        rng = np.random.default_rng(0)
        bs = [np.median(rng.choice(v, len(v))) for _ in range(5000)]
        print(f"whole-capture scale {np.median(v):.3f}, bootstrap 90% interval {np.percentile(bs, 5):.3f}-{np.percentile(bs, 95):.3f}")
        for n in (3, 5, 8):
            sub = [np.median(rng.choice(v, n, replace=False)) for _ in range(5000)]
            print(f"  room from {n} images: 90% within {100 * np.percentile(np.abs(np.array(sub) - 1), 90):.1f}%")
if __name__ == "__main__":
    main()
