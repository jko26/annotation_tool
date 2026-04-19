from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from endo_io.features import histogram_feature
from endo_io.images import load_rgb
from endo_io.mask import aggregate_border_mask_from_calibration, valid_mask_for_histogram

DistanceName = Literal["l1", "l2", "chi_square", "hellinger"]


def distance(a: np.ndarray, b: np.ndarray, name: DistanceName, *, eps: float = 1e-10) -> float:
    if name == "l1":
        return float(np.sum(np.abs(a - b)))
    if name == "l2":
        return float(np.linalg.norm(a - b))
    if name == "chi_square":
        return float(np.sum((a - b) ** 2 / (a + b + eps)))
    if name == "hellinger":
        return float(np.sqrt(np.sum((np.sqrt(np.maximum(a, 0)) - np.sqrt(np.maximum(b, 0))) ** 2)))
    raise ValueError(f"Unknown distance: {name}")


@dataclass
class PrototypeModel:
    prototype_inside: np.ndarray
    prototype_outside: np.ndarray
    black_max: int
    color_space: str
    bins_per_channel: int
    normalize: str
    distance: DistanceName
    calibration_paths: list[str] | None
    image_size: tuple[int, int] | None
    border_mask: np.ndarray | None

    def predict_one(self, rgb: np.ndarray) -> tuple[str, float, float]:
        """Returns label, dist_inside, dist_outside."""
        mask = valid_mask_for_histogram(rgb, self.black_max, self.border_mask)
        h = histogram_feature(
            rgb,
            mask,
            color_space=self.color_space,
            bins_per_channel=self.bins_per_channel,
            normalize=self.normalize,
        )
        d_in = distance(h, self.prototype_inside, self.distance)
        d_out = distance(h, self.prototype_outside, self.distance)
        label = "inside" if d_in <= d_out else "outside"
        return label, d_in, d_out


def fit_prototypes(
    inside_paths: list[Path],
    outside_paths: list[Path],
    calibration_paths: list[Path] | None,
    *,
    black_max: int = 18,
    color_space: str = "rgb",
    bins_per_channel: int = 16,
    normalize: str = "l1",
    aggregation: str = "mean",
    min_cal_fraction: float = 0.5,
    distance: DistanceName = "l2",
) -> PrototypeModel:
    if not inside_paths:
        raise ValueError("Need at least one inside training image")
    if not outside_paths:
        raise ValueError("Need at least one outside training image")

    border_mask = None
    if calibration_paths:
        border_mask = aggregate_border_mask_from_calibration(
            calibration_paths, black_max, min_fraction=min_cal_fraction
        )

    def collect(paths: list[Path]) -> list[np.ndarray]:
        vecs: list[np.ndarray] = []
        for p in paths:
            rgb = load_rgb(p)
            mask = valid_mask_for_histogram(rgb, black_max, border_mask)
            vecs.append(
                histogram_feature(
                    rgb,
                    mask,
                    color_space=color_space,
                    bins_per_channel=bins_per_channel,
                    normalize=normalize,
                )
            )
        return vecs

    v_in = collect(inside_paths)
    v_out = collect(outside_paths)
    stack_in = np.stack(v_in, axis=0)
    stack_out = np.stack(v_out, axis=0)
    if aggregation == "mean":
        p_in = np.mean(stack_in, axis=0)
        p_out = np.mean(stack_out, axis=0)
    elif aggregation == "median":
        p_in = np.median(stack_in, axis=0)
        p_out = np.median(stack_out, axis=0)
    else:
        raise ValueError("aggregation must be mean or median")

    ref = load_rgb(inside_paths[0])
    h, w = ref.shape[:2]
    cal_list = [str(p) for p in calibration_paths] if calibration_paths else None

    return PrototypeModel(
        prototype_inside=p_in,
        prototype_outside=p_out,
        black_max=black_max,
        color_space=color_space,
        bins_per_channel=bins_per_channel,
        normalize=normalize,
        distance=distance,
        calibration_paths=cal_list,
        image_size=(h, w),
        border_mask=border_mask,
    )
