# Lab notebook

Working record of findings, kept so every number in the report traces back to a script.

## 2026-10-03: Output contract

- Defined `schema/capture_output.schema.json` v1.0.0 (no published schema was provided, so this is our own; Round 1 gates were not provided and are not referenced).
- Every measured quantity is a Measurement object with a 90% interval, method and sample count.
- All damage, scope items and openings key to surface IDs (`R1-W2`, `R1-F`, `R1-C`).
- `tools/validate_output.py` adds checks JSON Schema cannot express: interval brackets value, references resolve, room overlap within tolerance.

## 2026-10-03: Company sample data

Three Stray Scanner LiDAR captures of one apartment:

| Capture | Frames | Duration | Notes |
|---|---|---|---|
| single_scan_with_ceiling | 9,745 | 215 s | whole apartment, ceiling observed |
| single_scan_floor_only | 5,251 | 115 s | whole apartment, ceiling barely observed (0.8% of points) |
| single_room | 1,715 | 37 s | one bedroom + bathroom, ceiling barely observed (0.3%) |

Format facts the loader must honour:
- Depth 256x192 uint16 millimetres; confidence 0/1/2; RGB 1920x1440 HEVC 60 fps, one frame per depth frame.
- Intrinsics vary per frame (fx 1581 to 1618 px, autofocus). `camera_matrix.csv` holds one frame only; using it for every frame costs up to ~2% scale. Use per-frame values from `odometry.csv`.
- Poses use the OpenCV camera convention (x right, y down, z forward), world +y up. Verified: the ARKit convention smears the floor across metres.
- Video frames are stored rotated 90 degrees (portrait capture); rotate before any image model.
- Doubled walls are visible in the fused whole-apartment scans: accumulated drift, needed for the drift ablation.
- No ground truth was provided (confirmed by the company).

Implication: two of three captures barely see the ceiling, so ceiling height must widen its interval and warn rather than report a confident number.

## 2026-10-03: ARKitScenes as laser ground truth

- Apple ARKitScenes, raw split, visit 421383 (validation). Captures 42444966 and 42444968 of the same room. Faro laser scans provided per visit.
- Laser folder: 4 PLY files, which are 2 distinct scans each duplicated (identical point counts before and after downsampling). Merged and downsampled to 1 cm: 1,046,490 points.
- `.traj` lines are world-to-camera transforms (camera centre = -R^T t), confirmed in ARKitScenes' own loader. Poses are 10 Hz, depth 60 Hz: match by timestamp; about 20% of frames had no pose within 20 ms, so the loader should interpolate.
- ARKitScenes depth also uses the OpenCV convention.
- iPad and laser clouds are in different frames. Rigid registration (no scale, so it cannot absorb scale error): FPFH + RANSAC then point-to-plane ICP. Four random seeds agree; gravity tilt between frames 0.15 to 0.19 degrees.
- Laser ground truth, room 421383: floor-to-ceiling 2329.6 mm (median of ~100k points per surface, spread 6 to 8 mm).
- Disclosure: ARKitScenes was captured on iPad Pro, not iPhone; licence read and use disclosed.

## 2026-10-03: Ceiling height bias (candidate for fix loop)

Raw iPad LiDAR, poses as-is, median of points:

| Capture | Floor | Ceiling | Height error |
|---|---|---|---|
| 42444966 | +8.6 to +11.3 mm | -10 to -11 mm | -20 to -27 mm |
| 42444968 | +17.4 mm | -11.5 mm | -25 to -29 mm |

Ranges reflect sensitivity to measurement choices (downsampling, confidence filter): about +/-5 mm. The final pipeline needs one fixed measurement method.

Classification: repeatable-but-biased (captures agree within ~9 mm, both fail the 15 mm gate).

Hypothesis 1, depth scale error of ~1%: rejected. Error along the viewing ray does not grow with range:

| Range | 0.3-0.75 m | 0.75-1.25 m | 1.25-1.75 m | 1.75-2.5 m | 2.5-3.5 m |
|---|---|---|---|---|---|
| Offset toward camera (conf=2) | +19.8 mm | +10.9 mm | +14.0 mm | +10.9 mm | +5.4 mm |

Hypothesis 2, surfaces sit ~1 to 1.5 cm in front of their true position at typical ranges: consistent with the symmetric pull of floor and ceiling toward the camera. Low-confidence pixels show a larger offset.

Correction test: range-dependent offset calibrated on 42444966 only, pushed back along each ray:

| Capture | Raw | Corrected |
|---|---|---|
| 42444966 (calibration) | -27.0 mm | -10.5 mm |
| 42444968 (held out) | -24.8 mm | -12.2 mm |

Open questions: one room and one device only; iPhone may differ; ~10 mm residual, possibly from grazing views of floor and ceiling.

## 2026-10-03: Shared loaders

- `floorplan/io/` defines one capture model for all formats: OpenCV camera convention, camera-to-world poses, intrinsics at depth resolution, depth in metres, pixels read lazily.
- Stray Scanner loader uses per-frame intrinsics from `odometry.csv`, scaled from 1920x1440 to the depth resolution.
- ARKitScenes loader inverts the world-to-camera `.traj` and interpolates poses to each depth timestamp (slerp + linear), rejecting gaps over 0.15 s. Usable frames rose from about 80% (nearest pose within 20 ms) to 1120 of 1129 and 745 of 754.
- `tools/inspect_capture.py` loads any capture, fuses it and prints a summary; unit tests in `tests/` cover pose inversion, interpolation, gap rejection and back-projection.
