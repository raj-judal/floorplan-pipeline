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
    raise ValueError(f"{root}: not a recognised capture folder (expected Stray Scanner or ARKitScenes layout)")


__all__ = ["Capture", "FrameMeta", "backproject", "load_capture"]
