# Phone capture to dimensioned floor plan

Pipeline that turns a phone capture (photos, video, or LiDAR) into a dimensioned,
stitched floor plan with damage regions and scope line items, with a 90%
confidence interval on every measurement.

Status: the LiDAR tier runs end to end (rooms, wall lengths, ceiling heights,
passages between rooms, stitched plan, schema-valid JSON). Video and photo tiers,
door/window detection, drift correction and damage detection are not built yet,
and every output says so in `quality.warnings`.

## Run one capture
    pip install -r requirements.txt
    python -m floorplan run PATH/TO/CAPTURE --tier lidar --out out/my_capture

Accepts Stray Scanner exports and ARKitScenes raw captures. Writes `output.json`
(validated against the schema on every run) and `plan.png`.

## Layout
- `floorplan/` the pipeline package (`python -m floorplan`)
- `schema/` output contract (JSON Schema v1.1.0) and an example output
- `tools/` utilities: `validate_output.py`, `inspect_capture.py`, `plan_lidar.py`
- `benchmarks/` small reference files needed to regenerate reported numbers
- `research/` exploratory scripts behind the findings (see `research/README.md`)
- `docs/` working notes, including the lab notebook
- `tests/` unit tests (`python -m pytest -q tests`)
