# Compliance matrix
Status: Done / Partial / Not done. Every number cited is in `docs/benchmark_report.md` or `docs/lab_notebook.md`.
| Requirement (brief) | File path | Artifact | Status |
|---|---|---|---|
| Capture route (Route 2: stock app + one-page protocol) | `docs/capture_protocol.md` | Protocol for all three tiers | Done |
| Device matrix | `docs/device_matrix.md` | Tier x hardware x stated accuracy | Done |
| LiDAR tier | `floorplan/io/stray.py`, `floorplan/pipeline.py` | Stray Scanner and ARKitScenes loaders, full pipeline | Done |
| Video tier | `floorplan/thin/` | Per-room measurement from clips | Partial: runs; rooms not stitched; not tested on real protocol video |
| Photo tier (per-room folders, any iPhone 15+) | `floorplan/thin/` | Per-room measurement from photos | Partial: runs; rooms not stitched; not tested on real protocol photos |
| Per-room plan: walls, ceiling height, floor area | `floorplan/pipeline.py`, `floorplan/plan/rooms.py` | `output.json` rooms[] | Done (LiDAR); Partial (thin tiers: rectangles) |
| Openings | `floorplan/pipeline.py` | Passages between segmented rooms | Partial: no door/window detection on walls |
| Stitched multi-room plan with adjacency | `floorplan/plan/rooms.py` | property_plan, adjacency | Done (LiDAR); Not done (photo, video) |
| Damage regions with class and metric extent | - | - | Not done (declared in output warnings) |
| Concealed-damage flags with rule | - | - | Not done |
| Scope line items keyed to surfaces | `schema/capture_output.schema.json` | Schema supports it | Not done (no damage input) |
| Confidence interval on every measurement | `schema/`, `tools/validate_output.py` | Schema requires it; validator checks interval brackets value | Done |
| One command per capture | `floorplan/__main__.py` | `python -m floorplan run CAPTURE --tier lidar|photo|video` | Done |
| JSON to the published schema | `schema/capture_output.schema.json` | Own schema v1.1.0 (none was published to us) | Done |
| Rendered plan | `floorplan/plan/render.py`, `floorplan/thin/pipeline_thin.py` | `plan.png` | Done |
| Benchmark: multi-room capture, 3+ rooms + connector | company sample data | Two full-apartment scans | Partial: no ground truth |
| Benchmark: furnished room with staged damage (2 classes) | - | - | Not done (no device) |
| Benchmark: same rooms at all three tiers | - | - | Not done (no device) |
| Benchmark: one room captured twice, same tier | `benchmarks/arkitscenes_421383/` | ARKitScenes 42444966 + 42444968 | Done |
| Laser or tape ground truth on everything | `benchmarks/arkitscenes_421383/`, `research/arkitscenes_*` | Faro laser, one room | Partial: one room only |
| Gate: opening widths | - | - | Not measured (no opening ground truth) |
| Gate: ceiling height and spread | `docs/benchmark_report.md` | -9.9 / -27.7 mm; spread 17.8 mm | Measured: fails |
| Gate: repeatability | `docs/benchmark_report.md` | 0 of 4 wall pairs; median 44 mm | Measured: fails |
| Gate: drift accountability (ablation on/off) | `floorplan/geometry/drift.py`, `tools/drift_ablation.py` | Footprint and wall sharpness, on vs off | Done |
| Gate: photo-tier whole-property stitch | - | - | Not done (fails) |
| Gates: photo +/-8%, video +/-3% | - | - | Not measured on real captures |
| Calibration scored at every tier | `docs/benchmark_report.md` | LiDAR coverage 0.40 -> 0.90 (in-sample) | Partial: thin-tier intervals from measured scale spread, not checked end-to-end |
| Head-to-head vs consumer app (Part 3) | - | - | Not done (no LiDAR device to capture with the app) |
| Fix loop (Part 4) | `docs/fix_declaration.md`, `docs/fix_postmortem_v1.md` | Declaration before code; shipped fix; worse result; post-mortem; tags `fix-loop-before`, `fix-loop-v1-after` | Done (fix did not move the gate) |
| Process evidence (Part 5) | git history | Commits in working order | Done |
| README to running in under 15 min | `README.md`, `requirements-lock.txt` | Install + one command | Done: fresh clone to validated LiDAR run in 5.6 min (warm pip cache); video tier also ran on the clean install |
| Reproduction bundle | `docs/reproduce.md`, `benchmarks/` | Commands + reference files | Partial: see reproduce.md |
| Benchmark report | `docs/benchmark_report.md` | Gates, repeatability, drift, timing | Done (LiDAR); thin tiers only partly |
| Technical report (max 6 pages) | `docs/technical_report.md` | | In progress |
| Raw benchmark data | ARKitScenes (official script), company sample data | IDs and crop box in `benchmarks/` | Partial: no app exports |
| Pretrained models/datasets disclosed | `output.json` pipeline.models; report | Depth Anything V2 Metric Indoor Small; ARKitScenes | Done |
| Runs without our infrastructure; weights fetched by script | `floorplan/thin/pipeline_thin.py` | Weights downloaded from Hugging Face on first run | Done |
| Mirrors, glass, wet-look surfaces, low light covered | `docs/capture_protocol.md`, report | Protocol rules, confidence filter, observed failures | Partial: no detection; warning codes exist but are not raised |
