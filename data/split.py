"""
Create a stratified train/val/test split from the NPZ produced by preprocess.py.
Each key in the input NPZ is a class label; each value is an array of samples.
Outputs a new NPZ with `_split` appended containing train_x/train_y, val_x/val_y,
test_x/test_y, and classes.
"""

import argparse
from pathlib import Path

import numpy as np


RATIOS = (0.70, 0.10, 0.20)


def split_class(arr, rng):
    n = len(arr)
    if n == 0:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)
    idx = rng.permutation(n)
    t = int(n * RATIOS[0])
    v = int(n * RATIOS[1])
    return idx[:t], idx[t : t + v], idx[t + v :]


def stitch(xs, ys, sample_shape, sample_dtype, rng):
    if not xs:
        return np.empty((0, *sample_shape), dtype=sample_dtype), np.empty((0,), dtype=np.int64)
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    order = rng.permutation(len(y))
    return x[order], y[order]


def split_dataset(data, rng):
    classes = sorted(data.keys())
    sample = next(iter(data.values()))
    sample_shape = sample.shape[1:]
    sample_dtype = sample.dtype

    buckets = {k: {"x": [], "y": []} for k in ("train", "val", "test")}

    for class_idx, label in enumerate(classes):
        samples = data[label]
        train_idx, val_idx, test_idx = split_class(samples, rng)

        buckets["train"]["x"].append(samples[train_idx])
        buckets["val"]["x"].append(samples[val_idx])
        buckets["test"]["x"].append(samples[test_idx])

        buckets["train"]["y"].append(np.full(len(train_idx), class_idx, dtype=np.int64))
        buckets["val"]["y"].append(np.full(len(val_idx), class_idx, dtype=np.int64))
        buckets["test"]["y"].append(np.full(len(test_idx), class_idx, dtype=np.int64))

    out = {}
    for name in ("train", "val", "test"):
        x, y = stitch(buckets[name]["x"], buckets[name]["y"], sample_shape, sample_dtype, rng)
        out[f"{name}_x"] = x
        out[f"{name}_y"] = y

    out["classes"] = np.array(classes)
    return out


def main():
    parser = argparse.ArgumentParser(description="Stratified split NPZ -> train/val/test")
    parser.add_argument("input", type=Path, help="NPZ file from preprocess.py")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    src = args.input
    if not src.exists():
        raise FileNotFoundError(src)

    data = dict(np.load(src))
    if not data:
        raise ValueError(f"No data found in {src}")

    rng = np.random.default_rng(args.seed)
    split = split_dataset(data, rng)
    dst = src.with_name(f"{src.stem}_split{src.suffix}")
    np.savez(dst, **split)

    tn, vn, tsn = len(split["train_y"]), len(split["val_y"]), len(split["test_y"])
    total = tn + vn + tsn
    print(f"Input:  {src}")
    print(f"Output: {dst}")
    print(
        f"Split sizes -> train: {tn} ({tn/total:.1%}), "
        f"val: {vn} ({vn/total:.1%}), "
        f"test: {tsn} ({tsn/total:.1%})"
    )


if __name__ == "__main__":
    main()
