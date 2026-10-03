# Reproducing every reported number
Commands are shown for Windows PowerShell; on macOS/Linux use `/` in paths. Install first (README).
Data goes in `data/` (ignored by git). Expected values are those in `docs/benchmark_report.md`.
## 1. Data
**Company sample data** (three Stray Scanner zips provided with the assessment): extract to
`data/sample/single_room/`, `data/sample/single_scan_with_ceiling/`, `data/sample/single_scan_floor_only/`.
Each contains one recording folder (`c00a170fe1`, `c7d28f72c6`, `1a8384c3f6`).
**ARKitScenes, visit 421383** (Apple; read its LICENSE first):
    git clone https://github.com/apple/ARKitScenes.git data/ARKitScenes
    # Windows only: the script calls `unzip`; use tar instead
    (Get-Content data/ARKitScenes/download_data.py) -replace 'unzip -oq \{filepath\} -d \{dst\}', 'tar -xf {filepath} -C {dst}' | Set-Content data/ARKitScenes/download_data.py
    python data/ARKitScenes/download_data.py raw --split Validation --video_id 42444966 42444968 --download_dir data/arkitscenes --raw_dataset_assets lowres_depth confidence lowres_wide lowres_wide_intrinsics lowres_wide.traj --download_laser_scanner_point_cloud
    python research/arkitscenes_downsample_laser.py data/arkitscenes/laser_scanner_point_clouds/421383 --voxel 0.01
    python research/arkitscenes_crop_laser.py data/arkitscenes/laser_scanner_point_clouds/421383_merged_10mm.ply data/arkitscenes/laser_room_crop.ply
    python tools/make_arkitscenes_subset.py data/arkitscenes/raw/Validation/42444966 data/arkitscenes/subset_42444966
    python tools/make_arkitscenes_subset.py data/arkitscenes/raw/Validation/42444968 data/arkitscenes/subset_42444968
The laser scans are ~7 GB before downsampling. Expected: 1,046,490 merged points; 937,294 after cropping.
## 2. LiDAR benchmark (gates, repeatability, calibration)
    python -m floorplan run data/arkitscenes/subset_42444966 --tier lidar --out out/ark_42444966 --stride 1
    python -m floorplan run data/arkitscenes/subset_42444968 --tier lidar --out out/ark_42444968 --stride 1
    python tools/benchmark_lidar_laser.py data/arkitscenes/laser_room_crop.ply out/benchmark.json --run 42444966=out/ark_42444966 --run 42444968=out/ark_42444968
Expected at HEAD: ceiling -9.9 / -27.7 mm; repeatability 0 of 4, median 44 mm; calibration 0.90.
The v0 calibration (0.40) is regenerated at tag `fix-loop-before`.
## 3. Fix loop before / after
    git checkout fix-loop-before      # then run section 2: repeatability median 44 mm
    git checkout fix-loop-v1-after    # then run section 2: repeatability median 72 mm
    git checkout main
## 4. Drift ablation
    python tools/drift_ablation.py data/sample/single_scan_with_ceiling/c7d28f72c6 out/ablation --stride 8
Expected: 7 rooms both ways; footprint 64.5 -> 66.5 m2; face concentration 0.565 -> 0.609.
## 5. Thin tiers
    python tools/test_depth_model.py data/sample/single_room/c00a170fe1
    python tools/test_depth_model.py data/sample/single_scan_floor_only/1a8384c3f6
    python research/thin_gravity_check.py data/sample/single_room/c00a170fe1 data/sample/single_scan_floor_only/1a8384c3f6 data/sample/single_scan_with_ceiling/c7d28f72c6
    python -m floorplan run data/sample/single_scan_with_ceiling/c7d28f72c6 --tier lidar --out out/full_scan --stride 10
    python research/thin_rendered_protocol_test.py out/full_scan
Expected: plausible-frame scale 0.998 and 0.988; gravity medians 0.66 / 0.74 / 0.49 deg;
rendered views as in the benchmark report. The depth-model test needs the GPU/CPU model download;
its numbers can vary slightly across hardware.
## 6. Plane extraction vs laser
    python research/eval_planes_vs_laser.py data/arkitscenes/laser_room_crop.ply data/arkitscenes/subset_42444966 data/arkitscenes/subset_42444968
