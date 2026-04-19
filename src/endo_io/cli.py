from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import replace
from pathlib import Path

from endo_io.images import list_image_paths, load_rgb
from endo_io.model import load_model, save_model
from endo_io.prototype import fit_prototypes


def _read_train_csv(path: Path) -> tuple[list[Path], list[Path]]:
    inside: list[Path] = []
    outside: list[Path] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "path" not in reader.fieldnames:
            raise ValueError("CSV must have a 'path' column")
        label_key = "label" if "label" in reader.fieldnames else None
        if label_key is None:
            raise ValueError("CSV must have a 'label' column (inside/outside)")
        for row in reader:
            p = Path(row["path"].strip())
            lab = row[label_key].strip().lower()
            if lab in ("inside", "in", "1", "true"):
                inside.append(p)
            elif lab in ("outside", "out", "0", "false"):
                outside.append(p)
            else:
                raise ValueError(f"Unknown label {lab!r} for {p}")
    return inside, outside


def _read_paths_csv(path: Path) -> list[Path]:
    out: list[Path] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "path" not in reader.fieldnames:
            raise ValueError("CSV must have a 'path' column")
        for row in reader:
            out.append(Path(row["path"].strip()))
    return out


def cmd_fit(args: argparse.Namespace) -> int:
    inside: list[Path] = []
    outside: list[Path] = []
    if args.train_csv:
        inside, outside = _read_train_csv(Path(args.train_csv))
    else:
        if not args.inside_dir or not args.outside_dir:
            print("fit: provide --train-csv or both --inside-dir and --outside-dir", file=sys.stderr)
            return 2
        inside = list_image_paths(args.inside_dir, recursive=args.recursive)
        outside = list_image_paths(args.outside_dir, recursive=args.recursive)

    cal_paths: list[Path] | None = None
    if args.calibration_dir:
        cal_paths = list_image_paths(args.calibration_dir, recursive=args.recursive)
        if not cal_paths:
            print("Warning: calibration directory has no images", file=sys.stderr)

    model = fit_prototypes(
        inside,
        outside,
        cal_paths,
        black_max=args.black_max,
        color_space=args.color_space,
        bins_per_channel=args.bins,
        normalize=args.normalize,
        aggregation=args.aggregation,
        min_cal_fraction=args.min_cal_fraction,
        distance=args.distance,
    )
    out = Path(args.out)
    save_model(out, model)
    print(f"Wrote {out.with_suffix('.npz')} and metadata JSON.")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    model = load_model(args.model)
    paths: list[Path] = []
    if args.input_csv:
        paths = _read_paths_csv(Path(args.input_csv))
    elif args.input_dir:
        paths = list_image_paths(args.input_dir, recursive=args.recursive)
    else:
        print("predict: provide --input-dir or --input-csv", file=sys.stderr)
        return 2

    out_path = Path(args.out)
    rows: list[dict[str, str | float]] = []
    for p in paths:
        rgb = load_rgb(p)
        m = model
        if model.border_mask is not None and rgb.shape[:2] != model.border_mask.shape:
            if args.fallback_black_mask:
                m = replace(model, border_mask=None)
            else:
                raise ValueError(
                    f"{p}: size {rgb.shape[:2]} != calibration mask {model.border_mask.shape}; "
                    "use --fallback-black-mask or match resolution to training/calibration."
                )
        label, d_in, d_out = m.predict_one(rgb)
        rows.append(
            {
                "path": str(p),
                "label": label,
                "dist_inside": d_in,
                "dist_outside": d_out,
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "dist_inside", "dist_outside"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="endo-io", description="Histogram prototype inside/outside classifier")
    sub = p.add_subparsers(dest="command", required=True)

    pf = sub.add_parser("fit", help="Build class prototypes from labeled images")
    pf.add_argument("--inside-dir", type=str, default=None, help="Directory of inside images")
    pf.add_argument("--outside-dir", type=str, default=None, help="Directory of outside images")
    pf.add_argument("--train-csv", type=str, default=None, help="CSV with path,label (inside/outside)")
    pf.add_argument("--calibration-dir", type=str, default=None, help="Uniform-disk calibration frames")
    pf.add_argument("--out", "-o", type=str, required=True, help="Output prefix (writes .npz + .json)")
    pf.add_argument("--black-max", type=int, default=18, help="Max channel value for border black (0-255)")
    pf.add_argument("--bins", type=int, default=16, help="Histogram bins per channel")
    pf.add_argument("--color-space", choices=("rgb", "hsv"), default="rgb")
    pf.add_argument("--normalize", choices=("l1", "l2", "none"), default="l1")
    pf.add_argument("--aggregation", choices=("mean", "median"), default="mean")
    pf.add_argument(
        "--min-cal-fraction",
        type=float,
        default=0.5,
        help="Pixel is border if black in at least this fraction of calibration images",
    )
    pf.add_argument(
        "--distance",
        type=str,
        choices=("l1", "l2", "chi_square", "hellinger"),
        default="l2",
    )
    pf.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Include images in subdirectories (e.g. clips/clips_0/, clips_1/...)",
    )
    pf.set_defaults(func=cmd_fit)

    pp = sub.add_parser("predict", help="Classify images using a saved model")
    pp.add_argument("--model", "-m", type=str, required=True, help="Path to saved .npz (same stem as JSON)")
    pp.add_argument("--input-dir", type=str, default=None)
    pp.add_argument("--input-csv", type=str, default=None)
    pp.add_argument("--out", "-o", type=str, required=True, help="Output CSV path")
    pp.add_argument(
        "--fallback-black-mask",
        action="store_true",
        help="If image size differs from calibration mask, use per-pixel black threshold only",
    )
    pp.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Include images in subdirectories under --input-dir",
    )
    pp.set_defaults(func=cmd_predict)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    code = args.func(args)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
