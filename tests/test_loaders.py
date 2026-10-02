"""Unit tests for the geometry conventions the loaders depend on."""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from floorplan.io.arkitscenes import interpolate_poses  # noqa: E402
from floorplan.io.capture import backproject            # noqa: E402


def _traj_row(t, R_wc, C):
    """Build an ARKitScenes .traj row (world-to-camera) from a camera-to-world pose."""
    R_cw = R_wc.T
    return [t, *Rotation.from_matrix(R_cw).as_rotvec(), *(-R_cw @ C)]


def test_traj_is_inverted_to_camera_centre():
    R_wc = Rotation.from_euler("z", 30, degrees=True).as_matrix()
    C = np.array([1.0, 2.0, 1.5])
    traj = np.array([_traj_row(0.0, R_wc, C), _traj_row(0.1, R_wc, C)])
    T, ok = interpolate_poses(traj, np.array([0.0]))
    assert ok[0]
    np.testing.assert_allclose(T[0, :3, 3], C, atol=1e-9)
    np.testing.assert_allclose(T[0, :3, :3], R_wc, atol=1e-9)


def test_interpolation_midpoint_and_gap():
    I = np.eye(3)
    traj = np.array([_traj_row(0.0, I, np.zeros(3)),
                     _traj_row(0.1, I, np.array([1.0, 0, 0])),
                     _traj_row(1.0, I, np.array([2.0, 0, 0]))])   # 0.9 s gap
    T, ok = interpolate_poses(traj, np.array([0.05, 0.5, 2.0]))
    np.testing.assert_allclose(T[0, :3, 3], [0.5, 0, 0], atol=1e-9)
    assert ok[0] and not ok[1] and not ok[2]   # inside gap, and outside span, are rejected


def test_backproject_opencv_convention_and_range_correction():
    K = np.array([[100.0, 0, 50], [0, 100.0, 50], [0, 0, 1]])
    depth = np.full((100, 100), 2.0, dtype=np.float32)
    W, rng, v, u = backproject(depth, K, np.eye(4))
    np.testing.assert_allclose(W[:, 2], 2.0)
    # pixel below the principal point maps to +y (OpenCV: y down)
    assert W[(v == 99) & (u == 50)][0, 1] > 0
    W2, rng2, _, _ = backproject(depth, K, np.eye(4), range_correction=lambda r: np.full_like(r, 0.01))
    np.testing.assert_allclose(rng2 - rng, 0.01, atol=1e-6)


def test_floor_is_lowest_surface_not_largest():
    from floorplan.plan.rooms import floor_height
    rng = np.random.default_rng(0)
    floor = np.c_[rng.uniform(0, 4, (2000, 2)), np.zeros(2000)]
    ceiling = np.c_[rng.uniform(0, 4, (6000, 2)), np.full(6000, 2.4)]   # ceiling has 3x more points
    P = np.vstack([floor, ceiling])
    N = np.tile([0, 0, 1.0], (len(P), 1))
    assert abs(floor_height(P, N)) < 0.01
