from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from endo_io.model import load_model, save_model
from endo_io.prototype import fit_prototypes


def _disk_frame(h: int, w: int, disk_rgb: tuple[int, int, int]) -> np.ndarray:
    img = Image.new("RGB", (w, h), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    cx, cy = w // 2, h // 2
    r = min(w, h) // 2 - 4
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=disk_rgb)
    return np.asarray(img, dtype=np.uint8)


def _save_png(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(arr, mode="RGB").save(path)


@pytest.fixture()
def tmp_media(tmp_path: Path) -> dict[str, Path]:
    inside_dir = tmp_path / "inside"
    outside_dir = tmp_path / "outside"
    cal_dir = tmp_path / "cal"
    for d in (inside_dir, outside_dir, cal_dir):
        d.mkdir()

    for i in range(3):
        _save_png(inside_dir / f"in{i}.png", _disk_frame(64, 64, (220, 60, 60)))
        _save_png(outside_dir / f"out{i}.png", _disk_frame(64, 64, (80, 120, 200)))
    _save_png(cal_dir / "cal0.png", _disk_frame(64, 64, (255, 255, 255)))

    return {"inside": inside_dir, "outside": outside_dir, "cal": cal_dir, "tmp": tmp_path}


def test_fit_predict_roundtrip(tmp_media: dict[str, Path]) -> None:
    inside = sorted(tmp_media["inside"].glob("*.png"))
    outside = sorted(tmp_media["outside"].glob("*.png"))
    cal = sorted(tmp_media["cal"].glob("*.png"))

    model = fit_prototypes(inside, outside, cal, black_max=18, distance="l2")
    assert model.border_mask is not None

    label_in, d_in_i, d_out_i = model.predict_one(np.array(Image.open(inside[0])))
    assert label_in == "inside"
    assert d_in_i <= d_out_i

    label_out, d_in_o, d_out_o = model.predict_one(np.array(Image.open(outside[0])))
    assert label_out == "outside"
    assert d_out_o <= d_in_o

    out = tmp_media["tmp"] / "m.npz"
    save_model(out, model)
    loaded = load_model(out)
    l2, _, _ = loaded.predict_one(np.array(Image.open(inside[0])))
    assert l2 == "inside"
