"""List ARKitScenes raw captures that have Faro laser-scanner ground truth.

Run from inside the cloned ARKitScenes repo:
    python3 find_arkitscenes_scenes.py

Prints visits (rooms) from the Validation split that have laser scans,
grouped so you can see which rooms were captured more than once. A room
with 2+ captures gives you a repeatability test WITH laser ground truth.
"""
import pandas as pd

URL = "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/metadata.csv"

meta = pd.read_csv(URL)
meta.to_csv("raw_metadata.csv", index=False)  # keep a local copy for reproducibility
print("metadata columns:", list(meta.columns))

has_laser = meta["has_laser_scanner_point_clouds"].astype(str).str.lower().isin(["true", "1", "1.0"])
sel = meta[has_laser & meta["visit_id"].notna()]

if "fold" in sel.columns:
    sel = sel[sel["fold"] == "Validation"]
else:
    splits = pd.read_csv("raw/raw_train_val_splits.csv")
    val_ids = set(splits.loc[splits["fold"] == "Validation", "video_id"].astype(int))
    sel = sel[sel["video_id"].astype(int).isin(val_ids)]

groups = (sel.assign(video_id=sel["video_id"].astype(int), visit_id=sel["visit_id"].astype(int))
             .groupby("visit_id")["video_id"].apply(list)
             .reset_index(name="video_ids"))
groups["n_captures"] = groups["video_ids"].str.len()
groups = groups.sort_values("n_captures", ascending=False)

print(f"\nValidation visits with laser ground truth: {len(groups)}")
print("Top 10 (rooms with the most repeat captures first):\n")
print(groups.head(10).to_string(index=False))
