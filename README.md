# Phone capture to dimensioned floor plan

Pipeline that turns a phone capture (photos, video, or LiDAR) into a dimensioned,
stitched floor plan with damage regions and scope line items, with a confidence
interval on every measurement.

Status: early development. The output contract (`schema/`) and its validator are
defined; the pipeline is being built.

## Layout
- `schema/` output contract (JSON Schema) and an example output
- `tools/` utilities, including `validate_output.py`
- `research/` exploratory scripts used to understand the input data (see `research/README.md`)
- `docs/` working notes, including the lab notebook

## Validate an output
    pip install -r requirements.txt
    python tools/validate_output.py schema/examples/example_photo_tier.json
