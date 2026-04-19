from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from endo_io.prototype import DistanceName, PrototypeModel


def save_model(path: str | Path, model: PrototypeModel) -> None:
    path = Path(path)
    npz_path = path if path.suffix == ".npz" else path.with_suffix(".npz")
    json_path = npz_path.with_suffix(".json")

    bm = model.border_mask
    if bm is None:
        bm_arr = np.zeros((0, 0), dtype=np.bool_)
    else:
        bm_arr = bm.astype(np.bool_)

    np.savez_compressed(
        npz_path,
        prototype_inside=model.prototype_inside,
        prototype_outside=model.prototype_outside,
        border_mask=bm_arr,
    )

    meta: dict[str, Any] = {
        "black_max": model.black_max,
        "color_space": model.color_space,
        "bins_per_channel": model.bins_per_channel,
        "normalize": model.normalize,
        "distance": model.distance,
        "calibration_paths": model.calibration_paths,
        "image_size": list(model.image_size) if model.image_size else None,
        "has_border_mask": model.border_mask is not None,
    }
    json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_model(path: str | Path) -> PrototypeModel:
    path = Path(path)
    npz_path = path if path.suffix == ".npz" else path.with_suffix(".npz")
    json_path = npz_path.with_suffix(".json")
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)
    if not json_path.is_file():
        raise FileNotFoundError(f"Missing metadata JSON: {json_path}")

    meta = json.loads(json_path.read_text(encoding="utf-8"))
    data = np.load(npz_path)
    p_in = np.asarray(data["prototype_inside"], dtype=np.float64)
    p_out = np.asarray(data["prototype_outside"], dtype=np.float64)
    bm = np.asarray(data["border_mask"], dtype=bool)
    border_mask = None
    if meta.get("has_border_mask") and bm.size > 0 and bm.shape[0] > 0:
        border_mask = bm

    d_raw = meta.get("distance", "l2")
    d_name: DistanceName = cast(
        DistanceName, d_raw if d_raw in ("l1", "l2", "chi_square", "hellinger") else "l2"
    )

    return PrototypeModel(
        prototype_inside=p_in,
        prototype_outside=p_out,
        black_max=int(meta["black_max"]),
        color_space=str(meta["color_space"]),
        bins_per_channel=int(meta["bins_per_channel"]),
        normalize=str(meta["normalize"]),
        distance=d_name,
        calibration_paths=meta.get("calibration_paths"),
        image_size=tuple(meta["image_size"]) if meta.get("image_size") else None,
        border_mask=border_mask,
    )
