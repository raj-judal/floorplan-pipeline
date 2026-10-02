"""Capture loaders. load_capture() detects the format from the folder contents."""
from pathlib import Path

from .arkitscenes import is_arkitscenes, load_arkitscenes
from .capture import Capture, FrameMeta, backproject
from .stray import is_stray, load_stray


def load_capture(root) -> Capture:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"{root}: folder does not exist")
    if is_stray(root):
        return load_stray(root)
    if is_arkitscenes(root):
        return load_arkitscenes(root)
    inner = [d for d in root.iterdir() if d.is_dir() and (is_stray(d) or is_arkitscenes(d))]
    if len(inner) == 1:
        raise ValueError(f"{root}: the capture is one level deeper; use {inner[0]}")
    found = sorted(p.name for p in root.iterdir())[:10]
    raise ValueError(f"{root}: not a recognised capture folder. Expected a Stray Scanner export "
                     f"(odometry.csv + depth/) or an ARKitScenes capture (lowres_depth/ + *.traj); "
                     f"found: {found}")


__all__ = ["Capture", "FrameMeta", "backproject", "load_capture"]
