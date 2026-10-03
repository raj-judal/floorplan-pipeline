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

## 2026-10-03: Reproducibility check

- `tools/inspect_capture.py` on the sample single-room capture gave identical results on Windows (laptop) and Linux (sandbox): 1,715 frames, 1,384,766 fused points, floor at -1.487 m.

## 2026-10-03: Plane extraction vs laser (visit 421383)

- `floorplan/geometry/planes.py`: gravity alignment, robust floor/ceiling plane fits (tilt allowed), Manhattan yaw from wall normals, wall offsets from 1D histogram peaks.
- Laser ground truth: floor tilt 0.03 deg, ceiling tilt 0.18 deg, plane residuals 1.7 to 1.8 mm. The ceiling tilt alone changes height by ~13 mm across a 4 m room, so ceiling height must be reported as a footprint average, not at one point.
- Ceiling height from plane fits, evaluated at the room centre (no bias correction): -9.6 mm (42444966), -26.6 mm (42444968). Spread 17 mm: with plane fits the result is UNREPEATABLE, whereas the earlier median method was repeatable-but-biased. Measurement method changes the diagnosis; it must be fixed before the benchmark.
- Walls: histogram peaks include cabinet and furniture faces in this kitchen. Keeping only "structural" surfaces (at least 15% of points more than 1.8 m above the floor) removes most furniture.
- Structural wall offsets vs laser: +11 to +58 mm (42444966), -41 to +60 mm (42444968). Confounded by registration (ICP RMSE ~14 mm) and in-capture drift. Fair comparison is wall-to-wall distance within one capture; registration only supplies correspondence.
- Registration transforms and crop box are stored in `benchmarks/arkitscenes_421383/` so the evaluation regenerates from raw data.

## 2026-10-03: Rooms stage, first version (LiDAR tier)

Method (`floorplan/plan/rooms.py`): 3 cm occupancy grids (floor, furniture, structural-wall evidence), interior = observed space minus walls, room seeds = interior more than 0.42 m from any boundary (door-width passages separate rooms), watershed growth, rectilinear polygons, each edge snapped to the measured wall face (from points, not the grid).

Results on the company sample data (no ground truth; visual check against the point cloud):
- Full scan with ceiling: 5 rooms, polygons follow the walls. Missed: rooms the camera did not enter, and a bathroom whose tiles/glass register as wall evidence.
- Floor-only scan: 4x fewer wall points above 1.5 m (99th percentile point height 2.16 m vs 3.34 m), so rooms merged. Fix: the wall-evidence band adapts to how high the capture reached (here 1.19 to 1.99 m). Corridor and one room still merge.
- Single-room scan: horizontal floor points spread over 4 cm in height (vertical drift, or a sunken bathroom floor), and large interior areas had no measurement at all (dark sofa, low-confidence pixels). Fixes: floor found per 0.3 m tile as the lowest horizontal surface; interior also marked by carving camera-to-surface rays (free space). The bedroom is now detected; its upper part is still cut off, and the bathroom merges with the corridor.

Timing: fuse + align is 17 s for 1,715 frames at stride 3, but 190 s for the 9,745-frame apartment at stride 5. Too slow for a comfortable live walk-in; needs a faster path.

Known failure modes to report: unentered rooms, wide openings merging rooms, glass/tile false walls, vertical drift.

## 2026-10-03: One command per capture (LiDAR tier)

- `python -m floorplan run CAPTURE --tier lidar --out OUT` writes `output.json` (schema 1.1.0, validated on every run) and `plan.png`.
- Schema 1.1.0 adds warning codes `ceiling_not_observed`, `room_not_entered`, `feature_not_implemented`, so missing capabilities are stated in the output rather than silently absent.
- Error budget v0 in `floorplan/config.py` (hashed into every output): 12 mm per snapped wall face, 40 mm per grid-only face, 15 mm ceiling when observed, prior 2.25 to 3.10 m (90%) when the ceiling covers under 30% of the room.
- Ceiling height is the footprint mean of (ceiling plane minus floor plane) per room, which handles tilt.
- Bug fixed: reported edge lengths came from the pre-snap grid outline; they now come from the snapped walls (single-room R1 changed from 1.79 x 2.80 m to 1.91 x 3.00 m).
- Bug fixed: floor height was the most populated horizontal level; on ARKitScenes the ceiling has more points than the floor, so the "floor" was the ceiling. Now the lowest well-supported level.
- Bug fixed: sliver edges (4 mm) left after snapping are merged away, keeping the better-supported wall face.
- First pipeline ceiling numbers vs laser (2329.6 mm): 2314.4 mm (90% interval 2289.8 to 2339.1, contains truth) and 2303.1 mm (2278.5 to 2327.8, misses truth by 1.8 mm). Calibration needs work.

## 2026-10-03: One command per capture (LiDAR tier)

- `python -m floorplan run CAPTURE --tier lidar --out OUT` writes `output.json` and `plan.png`, then validates the JSON. First valid end-to-end output on the sample single-room capture in 18.5 s (stride 3).
- Schema bumped to v1.1.0: new warning codes `ceiling_not_observed`, `room_not_entered`, `edge_not_snapped`, `stage_not_implemented`, so missing stages are declared instead of returning empty-but-confident output.
- Post-snap cleanup removes sliver edges (under 6 cm between near-collinear neighbours); wall spans are recomputed from the snapped neighbours.
- Error budget (`floorplan/pipeline.py` CONFIG, provisional until calibrated on the benchmark): 15 mm per snapped wall face, 50 mm per unsnapped edge, 20 mm on observed ceiling height. Wall length sigma combines its two bounding faces.
- Unobserved ceiling: reported as a bounded interval (highest observed surface to a 3.2 m residential prior) with a warning, never as a confident number.
- Room-to-room passages come from where room masks touch; currently one of them lands on an unsnapped false boundary inside the bedroom (the bedroom split), a known failure.

## 2026-10-03: Drift correction and ablation (full scan with ceiling)

Attempt 1, 6-DoF fragment pose graph (Open3D, ICP loop closures): loop residual 32.6 -> 27.5 mm, but walls got blurrier (share of wall points within 2 cm of the snapped face 0.565 -> 0.512; face spread 8.7 -> 9.9 mm). Diagnosis: the optimiser tilted fragments by a median 1.3 deg (max 4.6 deg) and consecutive fragments jumped by up to 56 cm. Phone gravity is accurate to ~0.15-0.19 deg (laser check), so tilt was error introduced by bad loop closures.

Attempt 2, 4-DoF (x, y, z, yaw; gravity fixed), loop closures validated for overlap, residual, zero tilt (13 bad matches rejected) and plausible shift; Huber least squares; corrections interpolated per frame (largest step between fragments 2.9 cm, largest shift 18.6 cm, largest yaw 0.76 deg).

Ablation, same capture, stride 8 (`tools/drift_ablation.py`):

| | Drift OFF | Drift ON (4-DoF) |
|---|---|---|
| Rooms | 7 | 7 |
| Footprint | 64.5 m2 | 66.5 m2 |
| Bounding extent | 14.57 x 9.70 m | 13.92 x 9.72 m |
| Wall points within 2 cm of face | 0.565 | 0.609 |
| Snapped edges | 47 | 46 |

Walls are sharper with correction and the top-right room becomes a clean rectangle (render: drift_ablation_full_scan.png). The footprint changes by 3% and the x extent by 65 cm, which is large; without ground truth we cannot say which is right, only that the corrected walls are more self-consistent. Loop residual measured as median point distance did not move (30.3 vs 30.6 mm) because it is dominated by 5 cm voxel noise; a better drift metric is needed for the report.

Timing: drift correction 143 s plus pipeline 134 s for the 215 s apartment scan in the sandbox; slower on the laptop. 436 of 472 ICP loop attempts fail the overlap test, so a cheap overlap prefilter would save most of that time.

## 2026-10-03: Reproducibility of drift ablation, a floor bug, and the first LiDAR benchmark

- Drift ablation regenerated on the Windows laptop: identical room count, room areas and footprint to 4 decimals; sharpness metric differs in the 6th decimal.
- Bug: the room stage took the floor as the LARGEST horizontal surface. On ARKitScenes 42444966 the ceiling has more points than the floor, so "floor" was the ceiling, every wall failed to snap and the ceiling read as unobserved. Fixed: lowest well-supported surface (same rule as the plane fitter); regression test added.
- `tools/benchmark_lidar_laser.py` scores pipeline outputs against the laser. Ground truth is measured from laser points (the pipeline's walls only say where to look); registration rigid, refined by ICP on the pipeline's own fused cloud (RMSE 15 mm). Report: `benchmarks/arkitscenes_421383/lidar_benchmark_v0.json`.

| Gate | Result | Pass |
|---|---|---|
| Ceiling height <= 15 mm | -9.9 mm, -27.7 mm | 1 of 2 |
| Ceiling spread across captures <= 10 mm | 17.8 mm (unrepeatable) | no |
| Wall repeatability <= max(1 cm, 0.5%) | 4 pairs, median difference 44 mm | 0 of 4 |
| Wall length vs laser (no LiDAR gate in brief) | median abs error 44.8 mm, 90th pct 76.8 mm | n/a |
| Calibration of 90% intervals | 4 of 10 truths inside | coverage 0.40 |

Observations: all 8 wall lengths and both ceilings are SHORT (errors -5 to -96 mm), consistent with surfaces pulled toward the camera, but often larger than the ~2.5 cm per wall-to-wall distance that the depth offset alone predicts, so a second cause (snapping to cabinet fronts in this kitchen, or drift) is likely. Intervals are far too narrow (0.40 coverage): the provisional error budget is overconfident and must be recalibrated, since confident garbage caps the score. One repeatability pair (254 mm) is a mismatched wall, not a measurement.

## 2026-10-03: Fix loop attempt 1 (failed)

Declared fix (overlap-based loop candidates) shipped; wall repeatability median difference got worse, 44 -> 72 mm, 0 of 4 pairs still failing; ceiling spread 17.8 -> 43.2 mm. The declaration's falsification test fired: walls still doubled. Splitting doubled walls by time shows the second copy comes from a few seconds of mis-tracked frames (capture start on 42444968; a 1 s blip on 42444966), not slow drift between passes. Full write-up: `docs/fix_postmortem_v1.md`.

## 2026-10-03: Fix loop attempt 2, hypothesis tested BEFORE declaring (refuted)

Lesson from attempt 1 applied: test the mechanism before writing a declaration.
Hypothesis 2: the second wall copy comes from short pose glitches (a few seconds of frames placed ~6 cm off), so re-aligning each half-second segment against what other passes saw would merge the copies.
Detection test on 42444968 (no gate measured): if a whole segment were mis-posed by 6 cm, all its surfaces would disagree with other passes. They do not: every segment, including the 2-6 s segments that produced the outer copy of R1-W4, has a median distance of 1.2 to 1.9 cm to other passes, the same as everywhere else. ICP "corrections" of 2 to 22 cm were flagged on almost every segment (72 of 147), which is ICP sliding along planar walls, not glitches. The detector cannot separate glitch segments, and the outer copy is local to the wall rather than a whole-frame pose error.
Result: hypothesis 2 refuted before any declaration or gate measurement; code removed. The cause of the doubled walls is still unknown (candidates: viewpoint-dependent depth error on specific surfaces, or something that moved in the scene).

## 2026-10-03: Calibration of the error budget (v0 -> v1)

Problem: v0 intervals (15 mm per wall face, a guess) contained the laser truth for only 4 of 10 measurements (target 9 of 10). Confident garbage caps the total score, so this outranks any single gate.

Tried first, rejected: quality-aware intervals. If clean walls could be told apart from doubled ones, clean walls could keep tight intervals. Measured: the share of a wall's points within 2 cm of its face does not predict its error (correlation 0.03 over 14 faces; R1-W12 is the most concentrated face, 0.85, and is 57 mm off). No honest per-wall signal, so the widening is uniform.

v1 budget (`floorplan/pipeline.py` CONFIG): 35 mm per wall face (90% of wall-length errors within ~80 mm), 60 mm per unsnapped edge, ceiling unchanged at 20 mm, opening width 50 mm (uncalibrated: no opening ground truth).

| | v0 | v1 |
|---|---|---|
| Truths inside 90% intervals | 4 of 10 (0.40) | 9 of 10 (0.90) |
| Measured values | unchanged | unchanged |

Limits, stated plainly:
- In-sample: v1 is fitted to the same 10 values it is scored on, so 0.90 is close to guaranteed here.
- Leave-one-capture-out shows how fragile it is: calibrating on 42444966 alone gives a 26 mm wall-length sigma, which covers 0 of 4 walls in 42444968; calibrating on 42444968 alone gives 53 mm, which covers 4 of 4 in 42444966. Two captures of one room are not enough to calibrate; more laser rooms would be.
- All 8 wall errors are negative (walls too short, mean -49 mm) and the intervals are centred on the biased values; width, not centring, provides the coverage. The bias is not corrected because its estimate is unstable between captures (-29 vs -70 mm mean).
