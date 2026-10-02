# Exploratory scripts

These scripts were used to understand the input data before building the pipeline.
They are kept for traceability of the findings in `docs/lab_notebook.md`. They are
not part of the pipeline and contain constants specific to the data they were run on.

| Script | Purpose |
|---|---|
| `stray_quick_fuse.py` | Fuse sampled Stray Scanner depth frames into a point cloud (run inside a capture folder) |
| `stray_analyze.py` | Trajectory stats, floor/ceiling peaks, and a top-down render for a Stray Scanner capture |
| `arkitscenes_find_scenes.py` | List ARKitScenes validation rooms with laser ground truth (run inside the ARKitScenes repo) |
| `arkitscenes_downsample_laser.py` | Merge and voxel-downsample the Faro laser scans of one visit |
| `arkitscenes_fuse.py` | Fuse ARKitScenes LiDAR depth using matched poses and per-frame intrinsics |
| `arkitscenes_register_to_laser.py` | Rigid (no-scale) registration of the iPad cloud to the laser cloud: FPFH + RANSAC, then point-to-plane ICP |
| `depth_bias_vs_laser.py` | Signed error of iPad depth points along the viewing ray, binned by range and confidence |
| `ceiling_bias_correction_test.py` | Calibrate a range-dependent depth offset on one capture, test it on a held-out capture |

Hardcoded constants (floor and ceiling heights, room crop box) are for ARKitScenes visit 421383.
