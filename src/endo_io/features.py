from __future__ import annotations

import numpy as np

from endo_io.images import rgb_to_hsv_u8


def histogram_feature(
    rgb: np.ndarray,
    valid_mask: np.ndarray,
    *,
    color_space: str = "rgb",
    bins_per_channel: int = 16,
    normalize: str = "l1",
) -> np.ndarray:
    """
    Concatenated per-channel histograms over valid pixels only.

    color_space: 'rgb' or 'hsv' (HSV uses bins on h in [0,1], s,v in [0,1]).
    normalize: 'l1' (sum to 1), 'l2', or 'none'.
    """
    if rgb.shape[:2] != valid_mask.shape:
        raise ValueError("valid_mask must match image spatial shape")
    flat_ok = valid_mask.reshape(-1)
    if not np.any(flat_ok):
        raise ValueError("No valid pixels for histogram")

    if color_space == "rgb":
        data = rgb.reshape(-1, 3)[flat_ok].astype(np.float32) / 255.0
        edges = [np.linspace(0, 1, bins_per_channel + 1) for _ in range(3)]
    elif color_space == "hsv":
        hsv = rgb_to_hsv_u8(rgb)
        data = hsv.reshape(-1, 3)[flat_ok]
        edges = [
            np.linspace(0, 1, bins_per_channel + 1),
            np.linspace(0, 1, bins_per_channel + 1),
            np.linspace(0, 1, bins_per_channel + 1),
        ]
    else:
        raise ValueError("color_space must be 'rgb' or 'hsv'")

    parts: list[np.ndarray] = []
    for c in range(3):
        hist, _ = np.histogram(data[:, c], bins=edges[c])
        parts.append(hist.astype(np.float64))
    h = np.concatenate(parts, axis=0)

    if normalize == "l1":
        s = float(np.sum(h))
        if s > 0:
            h = h / s
    elif normalize == "l2":
        n = float(np.linalg.norm(h))
        if n > 0:
            h = h / n
    elif normalize == "none":
        pass
    else:
        raise ValueError("normalize must be l1, l2, or none")

    return h.astype(np.float64)
