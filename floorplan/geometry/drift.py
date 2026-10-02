"""Drift correction: 4-DoF fragment pose graph with validated ICP loop closures.

Phone tracking (ARKit VIO) is accurate over seconds but accumulates drift over
minutes, which shows up as doubled walls when a room is revisited. Gravity is
observed directly by the IMU and is not what drifts (measured against laser:
0.15-0.19 deg), so only x, y, z and heading (yaw about up) are corrected.
A first 6-DoF version tilted fragments by up to 4.6 deg and made walls
blurrier; see docs/lab_notebook.md.

  1. Split the capture into fragments of FRAG_SECONDS; fuse each into a small
     cloud in its anchor frame (the middle frame's camera pose).
  2. Odometry edges between consecutive fragments come from the phone (trusted
     short-range, tight weights).
  3. Loop closures: point-to-plane ICP between non-consecutive fragments that
     observed the same space (shared coarse voxels). Accepted only if overlap and residual pass
     and the implied drift is physically plausible: no tilt, bounded shift.
  4. Robust (Huber) least squares over per-fragment corrections (yaw, t),
     fragment 0 fixed. Corrections are interpolated in time per frame, so
     there are no jumps at fragment boundaries.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import open3d as o3d
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from ..io.capture import Capture, backproject

FRAG_SECONDS = 3.0
VOXEL = 0.05
OVERLAP_CELL = 0.20          # m, coarse voxels for deciding whether two fragments saw the same space
MIN_OVERLAP = 0.30           # share of the smaller fragment's voxels also seen by the other
MAX_CANDIDATES = 8           # strongest-overlap partners tested per fragment
MIN_GAP_FRAGMENTS = 4
FITNESS_MIN = 0.35
RMSE_MAX = 0.03
MAX_LOOP_TILT_DEG = 1.0      # implied drift must not tilt gravity
MAX_LOOP_SHIFT_M = 0.6
SIG_ODO_T, SIG_ODO_YAW = 0.01, np.radians(0.2)
SIG_LOOP_T, SIG_LOOP_YAW = 0.02, np.radians(0.5)


@dataclass
class DriftResult:
    frame_corrections: list      # 4x4 world correction per frame
    stats: dict = field(default_factory=dict)


def _fragment_cloud(cap, idx, anchor, stride):
    inv = np.linalg.inv(anchor)
    pts = []
    for i in idx[::stride]:
        f = cap.frames[i]
        d = cap.read_depth(f)
        m = (d > 0.2) & (d < 4.0)
        c = cap.read_confidence(f)
        if c is not None:
            m &= c >= 2
        pts.append(backproject(d, f.K, inv @ f.T_wc, m)[0])
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(pts))).voxel_down_sample(VOXEL)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL * 3, max_nn=30))
    return pc


def _residual(src, tgt, T):
    d = np.asarray(copy.deepcopy(src).transform(T).compute_point_cloud_distance(tgt))
    d = d[d < 0.2]
    return float(np.median(d)) if len(d) else np.nan


def _corr_matrix(yaw, t, up):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(up * yaw).as_matrix()
    T[:3, 3] = t
    return T


def _tilt_deg(R, up):
    return float(np.degrees(np.arccos(np.clip((R @ up) @ up, -1, 1))))


def correct_drift(cap: Capture, stride: int = 6) -> DriftResult:
    up = np.zeros(3)
    up[cap.up_axis] = cap.up_sign
    ts = np.array([f.timestamp for f in cap.frames])
    frag = np.floor((ts - ts[0]) / FRAG_SECONDS).astype(int)
    ids = np.unique(frag)
    groups = [np.nonzero(frag == k)[0] for k in ids]
    anchors = [cap.frames[g[len(g) // 2]].T_wc.copy() for g in groups]
    clouds = [_fragment_cloud(cap, g, a, stride) for g, a in zip(groups, anchors)]
    n = len(groups)
    reg = o3d.pipelines.registration

    edges = [(k, k + 1, np.linalg.inv(anchors[k + 1]) @ anchors[k], SIG_ODO_T, SIG_ODO_YAW)
             for k in range(n - 1)]
    centres = np.array([a[:3, 3] for a in anchors])
    # Loop candidates: fragments that SAW the same space (shared coarse voxels in
    # world coordinates), not fragments where the camera stood close together.
    # Two passes along one wall from different positions are now compared.
    # (fix loop, docs/fix_declaration.md; previously: camera anchors within 2.5 m)
    vox = []
    for c, a in zip(clouds, anchors):
        W = np.asarray(c.points) @ a[:3, :3].T + a[:3, 3]
        vox.append(set(map(tuple, np.floor(W / OVERLAP_CELL).astype(np.int64))))
    pairs = []
    for i in range(n):
        scored = []
        for j in range(i + MIN_GAP_FRAGMENTS, n):
            ov = len(vox[i] & vox[j]) / max(1, min(len(vox[i]), len(vox[j])))
            if ov >= MIN_OVERLAP:
                scored.append((ov, j))
        pairs += [(i, j) for _, j in sorted(scored, reverse=True)[:MAX_CANDIDATES]]
    loops, rejected, before = [], {"fit": 0, "tilt": 0, "shift": 0}, []
    for i, j in pairs:
        T0 = np.linalg.inv(anchors[j]) @ anchors[i]
        icp = reg.registration_icp(clouds[i], clouds[j], VOXEL * 2, T0, reg.TransformationEstimationPointToPlane())
        icp = reg.registration_icp(clouds[i], clouds[j], VOXEL, icp.transformation,
                                   reg.TransformationEstimationPointToPlane())
        if icp.fitness < FITNESS_MIN or icp.inlier_rmse > RMSE_MAX:
            rejected["fit"] += 1
            continue
        M = anchors[j] @ icp.transformation @ np.linalg.inv(anchors[i])   # implied world drift i -> j
        if _tilt_deg(M[:3, :3], up) > MAX_LOOP_TILT_DEG:
            rejected["tilt"] += 1
            continue
        if np.linalg.norm(M[:3, 3] - (np.eye(3) - M[:3, :3]) @ centres[i]) > MAX_LOOP_SHIFT_M:
            rejected["shift"] += 1
            continue
        edges.append((i, j, icp.transformation, SIG_LOOP_T, SIG_LOOP_YAW))
        loops.append((i, j))
        before.append(_residual(clouds[i], clouds[j], T0))

    def corrected(x):
        return [_corr_matrix(x[4 * k], x[4 * k + 1:4 * k + 4], up) @ anchors[k] for k in range(n)]

    def residuals(x):
        A = corrected(x)
        out = [x[:4] * 1e3]                                   # fragment 0 is the reference
        for i, j, Z, st, sy in edges:
            E = np.linalg.inv(Z) @ np.linalg.inv(A[j]) @ A[i]
            up_i = A[i][:3, :3].T @ up
            yaw_err = Rotation.from_matrix(E[:3, :3]).as_rotvec() @ up_i
            out.append(np.r_[E[:3, 3] / st, yaw_err / sy])
        return np.concatenate(out)

    x0 = np.zeros(4 * n)
    sol = least_squares(residuals, x0, loss="huber", f_scale=2.0, max_nfev=200) if loops else None
    x = sol.x if sol is not None else x0
    A = corrected(x)
    after = [_residual(clouds[i], clouds[j], np.linalg.inv(A[j]) @ A[i]) for i, j in loops]

    # per-frame correction: interpolate (yaw, t) between fragment centres in time
    t_centre = np.array([ts[g[len(g) // 2]] for g in groups])
    params = x.reshape(n, 4)
    frame_corr = []
    for t in ts:
        p = np.array([np.interp(t, t_centre, params[:, c]) for c in range(4)])
        frame_corr.append(_corr_matrix(p[0], p[1:], up))
    jumps = [np.linalg.norm(params[k + 1, 1:] - params[k, 1:]) for k in range(n - 1)]
    stats = {"fragments": n, "loop_candidates": len(pairs), "loop_closures": len(loops), "loops_rejected": rejected,
             "residual_before_m": float(np.nanmedian(before)) if before else None,
             "residual_after_m": float(np.nanmedian(after)) if after else None,
             "max_shift_m": float(np.max(np.linalg.norm(params[:, 1:], axis=1))),
             "max_yaw_deg": float(np.degrees(np.max(np.abs(params[:, 0])))),
             "max_step_between_fragments_m": float(max(jumps)) if jumps else 0.0}
    return DriftResult(frame_corr, stats)


def apply_correction(cap: Capture, res: DriftResult) -> Capture:
    out = copy.copy(cap)
    out.frames = []
    for f, C in zip(cap.frames, res.frame_corrections):
        g = copy.copy(f)
        g.T_wc = C @ f.T_wc
        out.frames.append(g)
    return out
