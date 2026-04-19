# endo-io

Histogram prototype classifier for **inside vs outside** endoscope frames (v1: nearest reference to mean class histograms).

## Install

```bash
pip install -e ".[dev]"
```

## Usage

**Fit** prototypes from labeled folders and optional border-calibration images (uniform disk so black pixels are letterbox only):

```bash
endo-io fit \
  --inside-dir /path/to/inside \
  --outside-dir /path/to/outside \
  --calibration-dir /path/to/calibration \
  --out model.npz
```

If inside images live in subfolders (e.g. `clips/clips_0/`, `clips/clips_1/`, …), add **`--recursive`** (or **`-r`**) so all nested images are included:

```bash
endo-io fit -r --inside-dir /path/to/clips --outside-dir /path/to/outside ...
```

Or a CSV with columns `path,label` where `label` is `inside` or `outside`:

```bash
endo-io fit --train-csv train.csv --calibration-dir /path/to/cal --out model.npz
```

**Predict** on a directory of images or a manifest CSV (`path` column):

```bash
endo-io predict --model model.npz --input-dir /path/to/frames --out preds.csv
endo-io predict --model model.npz --input-csv manifest.csv --out preds.csv
```

See `endo-io fit --help` and `endo-io predict --help` for histogram bins, distance metrics, and black-pixel threshold.

## Tests

```bash
pytest
```
