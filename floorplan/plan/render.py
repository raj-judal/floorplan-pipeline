"""Render room polygons with edge lengths over the wall-evidence grid (PNG)."""
import cv2
import numpy as np

from .rooms import RES

COLOURS = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48),
           (145, 30, 180), (0, 128, 128), (128, 0, 0), (170, 110, 40)]


def render_plan(rooms, g, wall, path, scale=4, min_label_len=0.6):
    H, W = g.shape
    img = np.full((H * scale, W * scale, 3), 255, np.uint8)
    wm = cv2.resize(wall.astype(np.uint8), (W * scale, H * scale), interpolation=cv2.INTER_NEAREST)
    img[wm[::-1].astype(bool)] = (205, 205, 205)          # grid row 0 is the lowest y: flip for display

    def px(xy):
        return np.c_[(xy[:, 0] - g.x0) / RES * scale, (H - (xy[:, 1] - g.y0) / RES) * scale].astype(np.int32)

    for k, r in enumerate(rooms):
        if r.polygon is None:
            continue
        col = COLOURS[k % len(COLOURS)][::-1]               # OpenCV is BGR
        q = px(r.polygon)
        cv2.polylines(img, [q], True, col, 2)
        c = q.mean(axis=0).astype(int)
        label = f"R{r.label} {r.area_m2:.1f} m2" + ("" if r.entered else " (not entered)")
        cv2.putText(img, label, (c[0] - 45, c[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        for i, e in enumerate(r.edges):
            L = e.span[1] - e.span[0]
            if L >= min_label_len:
                m = ((q[i - 1] + q[i]) / 2).astype(int)
                cv2.putText(img, f"{L:.2f}", (m[0] - 14, m[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)
    cv2.imwrite(str(path), img)
