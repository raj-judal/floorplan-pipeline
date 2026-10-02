# Fix declaration

Written 2026-10-03, before any fix code. Baseline: `benchmarks/arkitscenes_421383/lidar_benchmark_v0.json`
(LiDAR tier, ARKitScenes visit 421383, captures 42444966 and 42444968, Faro laser ground truth).

## 1. Worst-performing gate

**Repeatability: two captures of the same room at the same tier agree within max(1 cm, 0.5%) per wall.**
Failing number: **0 of 4 wall pairs pass; median difference 44 mm** (tolerance 10 to 18 mm).

It is the furthest from its threshold of all scored gates (about 3.7x the tolerance, against 1.8x for
ceiling spread at 17.8 mm and 1.8x for the worst ceiling error at -27.7 mm). One of the four pairs
(254 mm) is a wall-matching failure rather than a measurement; without it the median is 26.5 mm, still failing.

## 2. Root-cause hypothesis and evidence

**Hypothesis.** Residual tracking drift between two passes along the same wall leaves two copies of
that wall 5 to 6 cm apart in the fused iPad cloud. Wall snapping takes the copy with more points, and
which copy wins differs between captures, so the same wall gets different lengths. Drift correction
does not remove the doubling because it only tries loop closures between fragments whose camera
positions are within 2.5 m; two passes over the same wall from different positions are never compared.

**Evidence.**
- Point profiles across walls (`docs/figures/doubled_wall_profiles.png`): the iPad shows two peaks
  about 6 cm apart on 42444968 R1-W4 and 42444966 R1-W7; the laser shows a single surface. On W4 the
  laser matches the outer copy, on W7 it is nearer the inner copy, so no fixed rule for choosing a copy
  can be right.
- Face errors split into two groups: walls with a single iPad peak sit 1 to 7 mm from the laser face;
  walls with doubled peaks sit 22 to 66 mm off.

**Alternatives considered.**
- Depth offset toward the camera (~1.2 cm per surface, `docs/lab_notebook.md`): real, and it shortens
  every wall by ~2.5 cm, but it shifts both captures alike, so it cannot explain the repeatability failure.
- Snapping to cabinet fronts instead of walls: ruled out for these walls. The laser has no surface at
  the snapped face position; its nearest layers are 3.5 to 6.6 cm outside, where the second iPad copy is.

## 3. The fix and the predicted number

**Fix.** Choose loop-closure candidates by overlap of the fragment clouds in world space, not by
camera proximity, so every pair of passes that saw the same surfaces is tested. Loop validation
(overlap, residual, zero tilt, plausible shift) is unchanged. Nothing else changes, so the effect is
attributable to this one change.

**Prediction.** Wall repeatability median difference falls from 44 mm to **15 mm or less**, and pairs
passing rise from 0 of 4 to **2 of 4** (the mismatched pair will still fail). Wall length median error
against the laser falls from 45 mm to about **25 mm**, the remainder being the depth offset, which this
fix does not address.

**What would falsify the hypothesis.** If the walls still show two peaks after the fix, the loops were
still not closed. If the peaks merge but repeatability does not improve, the doubling was not the cause.

## Regenerating before and after

    python -m floorplan run DATA/subset_42444966 --tier lidar --out out/ark_42444966 --stride 1
    python -m floorplan run DATA/subset_42444968 --tier lidar --out out/ark_42444968 --stride 1
    python tools/benchmark_lidar_laser.py DATA/laser_room_crop.ply out/benchmark.json \
        --run 42444966=out/ark_42444966 --run 42444968=out/ark_42444968

Before: the commit tagged `fix-loop-before`. After: the commit that follows this declaration.
