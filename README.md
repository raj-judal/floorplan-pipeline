# Phone capture to dimensioned floor plan
One command turns a phone capture into a dimensioned floor plan (`plan.png`) and a
schema-valid JSON (`output.json`) with a 90% confidence interval on every number.
- **LiDAR tier** (Stray Scanner on a Pro iPhone/iPad): rooms, wall lengths, ceiling heights,
  passages between rooms, stitched multi-room plan, drift correction.
- **Photo and video tiers** (any iPhone 15+): each room measured on its own from images with a
  depth model; rough, and rooms are not stitched. Every output says so.
- Damage (LiDAR tier): open-vocabulary detection on video frames, placed on surfaces, with concealed-damage
  rules and repair line items. Never tested on real damage in a captured room; every output says so.
- Not built: door/window detection on walls. See `docs/compliance_matrix.md`.
## Install (about 10 minutes; Python 3.10 or newer)
    git clone https://github.com/raj-judal/floorplan-pipeline.git
    cd floorplan-pipeline
    python -m venv .venv
    .venv\Scripts\Activate.ps1          # Windows PowerShell
    # source .venv/bin/activate         # macOS / Linux
    pip install -r requirements.txt
    python -m pytest -q tests           # expect: 4 passed
Tested on a clean Windows machine: clone to first validated LiDAR run in 5.6 minutes (warm pip cache).
If a newer package release breaks the install, use the exact tested versions:
`pip install -r requirements-lock.txt`.
For a GPU on Windows/Linux (optional, photo and video tiers only):
`pip install torch --index-url https://download.pytorch.org/whl/cu121`.
The depth model (~100 MB) downloads automatically on the first photo or video run.
## Run one capture
    python -m floorplan run PATH/TO/CAPTURE --tier lidar --out out/my_capture
    python -m floorplan run PATH/TO/CAPTURE --tier photo --out out/my_capture
    python -m floorplan run PATH/TO/CAPTURE --tier video --out out/my_capture
What to capture and how to hand it over: `docs/capture_protocol.md`. Inputs:
- lidar: a Stray Scanner recording folder (contains `odometry.csv`, `depth/`, `rgb.mp4`), or an ARKitScenes raw capture
- photo: a folder containing one subfolder of photos per room
- video: a folder containing one clip per room
Each run writes `output.json` (validated against `schema/capture_output.schema.json`
before the command finishes) and `plan.png` to the output folder.
## Documents
| | |
|---|---|
| `docs/technical_report.md` | Architecture, tiers, drift, error budget, calibration, fix loop, failure modes |
| `docs/benchmark_report.md` | Gates, repeatability, drift ablation, thin tiers, timing |
| `docs/compliance_matrix.md` | Every requirement: file, artifact, status |
| `docs/capture_protocol.md` | One-page capture protocol for all tiers |
| `docs/device_matrix.md` | Which tier runs on which device, with stated accuracy |
| `docs/fix_declaration.md`, `docs/fix_postmortem_v1.md` | Fix loop (tags `fix-loop-before`, `fix-loop-v1-after`) |
| `docs/reproduce.md` | Commands that regenerate every reported number |
| `docs/lab_notebook.md` | Dated record of every experiment, including failures |
## Layout
`floorplan/` pipeline package; `schema/` output contract; `tools/` runnable tools;
`research/` scripts behind specific findings; `benchmarks/` small reference files
(registrations, crop box, benchmark reports); `tests/` unit tests.
