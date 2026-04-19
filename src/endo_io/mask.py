from __future__ import annotations

from pathlib import Path

import numpy as np

from endo_io.images import load_rgb


def border_mask_from_black(rgb: np.ndarray, black_max: int) -> np.ndarray:
    """
    True where pixel is treated as letterbox border (near black in all channels).
    black_max: max channel value (0-255) for a pixel to count as border.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected HxWx3 RGB uint8")
    m = np.max(rgb, axis=-1)
    return m <= int(black_max)


def aggregate_border_mask_from_calibration(
    paths: list[str | Path],
    black_max: int,
    *,
    min_fraction: float = 0.5,
) -> np.ndarray | None:
    """
    Build a conservative border mask: pixel is border if it is black in at least
    `min_fraction` of calibration images (same resolution required).

    Calibration images should have uniform non-black disk and black letterbox.
    Returns None if paths is empty.
    """
    if not paths:
        return None
    first = load_rgb(paths[0])
    h, w = first.shape[:2]
    acc = np.zeros((h, w), dtype=np.float32)
    for p in paths:
        rgb = load_rgb(p)
        if rgb.shape[:2] != (h, w):
            raise ValueError(
                f"Calibration images must share resolution; got {rgb.shape[:2]} vs {(h, w)} for {p}"
            )
        acc += border_mask_from_black(rgb, black_max).astype(np.float32)
    n = float(len(paths))
    return (acc / n) >= float(min_fraction)


def valid_mask_for_histogram(
    rgb: np.ndarray,
    black_max: int,
    calibration_border_mask: np.ndarray | None,
) -> np.ndarray:
    """
    Pixels included in histogram: not border.

    If calibration_border_mask is provided (HxW bool), use it as border;
    else use per-pixel black threshold only.
    """
    if calibration_border_mask is not None:
        if calibration_border_mask.shape != rgb.shape[:2]:
            raise ValueError("Calibration mask shape must match image")
        return ~calibration_border_mask
    return ~border_mask_from_black(rgb, black_max)
