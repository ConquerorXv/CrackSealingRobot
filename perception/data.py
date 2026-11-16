from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

RGB_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEPTH_EXT = RGB_EXT | {".npy"}


def read_rgb(path: str | Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if arr is None:
        raise OSError(f"Cannot read RGB image: {path}")
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def read_depth(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    else:
        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise OSError(f"Cannot read depth map: {path}")
    if arr.ndim != 2:
        raise ValueError(f"Depth must be single-channel; got {arr.shape}: {path}")
    return arr.astype(np.float32)


def depth_to_float(depth: np.ndarray, scale: float = 1000.0,
                   offset: float = 0.0, min_value: float = 0.0,
                   max_value: float = 5.0, normalization: str = "fixed") -> np.ndarray:
    if normalization == "per_image_max":
        x = np.where(np.isfinite(depth), depth, 0).astype(np.float32)
        x = np.maximum(x, 0)
        maximum = float(x.max())
        return x / maximum if maximum > 0 else x
    if normalization != "fixed":
        raise ValueError(f"Unknown depth normalization: {normalization}")
    if scale <= 0 or max_value <= min_value:
        raise ValueError("Invalid depth normalization settings")
    x = (depth.astype(np.float32) - offset) / scale
    x = np.where(np.isfinite(x), x, min_value)
    return np.clip(x, min_value, max_value).astype(np.float32)
