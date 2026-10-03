# Benchmark report
Ground truth: Faro laser scans from ARKitScenes visit 421383 (two iPad Pro LiDAR captures of one open-plan kitchen/dining room). Ground truth is measured from laser points; the pipeline's walls only say where to look. Registration iPad-to-laser is rigid with no scale (ICP RMSE ~15 mm). Reports: `benchmarks/arkitscenes_421383/lidar_benchmark_*.json`.
## LiDAR tier gates
| Gate | Result | Pass |
|---|---|---|
| Ceiling height <= 15 mm per room | -9.9 mm (42444966), -27.7 mm (42444968) | 1 of 2 |
| Ceiling spread across captures <= 10 mm | 17.8 mm: unrepeatable | No |
| Repeatability: walls agree within max(1 cm, 0.5%) | 4 pairs, median difference 44 mm | 0 of 4 |
| Wall length vs laser (no LiDAR gate in the brief) | median abs error 44.8 mm, 90th pct 76.8 mm; all 8 walls short | - |
| Interval calibration (target 0.90) | v0 budget 0.40; v1 calibrated budget 0.90 (in-sample) | v1 only |
| Opening widths | not measured: no openings in the ground-truth room are detected | - |
## Repeatability table (same room, same tier)
| Wall pair (42444966 / 42444968) | Lengths (m) | Difference | Tolerance | Pass |
|---|---|---|---|---|
| R1-W6 / R1-W3 | 1.361 / 1.615 | 254 mm (walls mismatched, not a measurement) | 10 mm | No |
| R1-W7 / R1-W4 | 2.432 / 2.405 | 26.5 mm | 12.1 mm | No |
| R1-W8 / R1-W5 | 3.615 / 3.553 | 61.6 mm | 17.9 mm | No |
| R1-W9 / R1-W6 | 2.981 / 2.959 | 21.6 mm | 14.8 mm | No |
Cause (fix loop): two copies of the same wall ~6 cm apart in the iPad data, seen at different times; snapping takes the larger copy. See `docs/fix_postmortem_v1.md`.
## Fix loop numbers
| | Before | Predicted | After fix 1 |
|---|---|---|---|
| Repeatability median difference | 44 mm | <= 15 mm | 72 mm |
| Pairs passing | 0 of 4 | 2 of 4 | 0 of 4 |
| Ceiling spread | 17.8 mm | - | 43.2 mm |
## Drift ablation (company full-apartment scan, no ground truth)
| | Drift off | Drift on (4-DoF) |
|---|---|---|
| Rooms | 7 | 7 |
| Footprint | 64.5 m2 | 66.5 m2 |
| Bounding extent | 14.57 x 9.70 m | 13.92 x 9.72 m |
| Wall points within 2 cm of the snapped face | 0.565 | 0.609 |
23 loop closures accepted; 13 rejected for implying a tilt. Regenerated identically on a second machine.
## Thin tiers (photo, video)
| Test | Result |
|---|---|
| Depth model scale vs LiDAR, real frames, upright (single-room capture) | raw median 1.30 (spread 1.08-1.63) |
| After camera-height correction, same capture | median 0.998; 90% of frames within 14% |
| Same, held-out capture (rule not re-tuned) | median 0.988 (bootstrap 90% interval 0.948-1.007) |
| Gravity from image geometry (LiDAR stand-in depth, 180 frames) | median 0.49-0.74 deg; 2 frames failed |
| Rendered protocol views of the real apartment scan (7 rooms, `research/thin_rendered_protocol_test.py`): floor visible | 1x level: 0-1 of 4 views per room; 0.5x tilted 10 deg down: 1-4 of 4 |
| Same, room dimensions vs LiDAR-tier box, 0.5x tilted | 90% intervals contained 7 of 13 before, 9 of 13 after adding a geometry term; larger rooms underestimated by 30-45% |
| Real protocol photos or video | not measured (no device) |
## Head-to-head vs consumer app
Not performed: it needs the same rooms captured with our pipeline and with a consumer app on a LiDAR iPhone; none was available.
## Timing
| Run | Machine | Time |
|---|---|---|
| LiDAR, single room (37 s capture, stride 3), no drift correction | Windows laptop / Linux sandbox | 31.6 s / 18.5 s |
| LiDAR, same, drift correction on | sandbox | 25.5 s |
| LiDAR, full apartment (215 s capture, stride 8): drift correction + pipeline | sandbox | 143 s + 134 s |
| Depth model per image (photo/video tiers) | laptop, NVIDIA T500 | 0.36 s (first model load 31.6 s) |
