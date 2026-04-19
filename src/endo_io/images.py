from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def list_image_paths(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Not a directory: {root}")
    paths: list[Path] = []
    for p in sorted(root.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(p)
    return paths


def load_rgb(path: str | Path) -> np.ndarray:
    """Load image as uint8 HxWx3 RGB."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as im:
        rgb = im.convert("RGB")
        return np.asarray(rgb, dtype=np.uint8)


def rgb_to_hsv_u8(rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 -> HSV float in [0,1] for H and S, V in [0,1], shape HxWx3."""
    x = rgb.astype(np.float32) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin
    h = np.zeros_like(cmax)
    mask = delta > 1e-8
    idx = (cmax == r) & mask
    h = np.where(idx, ((g - b) / delta) % 6, h)
    idx = (cmax == g) & mask
    h = np.where(idx, (b - r) / delta + 2, h)
    idx = (cmax == b) & mask
    h = np.where(idx, (r - g) / delta + 4, h)
    h = h / 6.0
    s = np.where(cmax > 1e-8, delta / cmax, 0.0)
    v = cmax
    return np.stack([h, s, v], axis=-1)
