"""
Convert class-organized CSV samples under data/samples into a class-keyed NPZ.
Each subdirectory of data/samples is treated as a class label and should
contain CSV files (one sample per file). Shorter-than-MIN_LEN samples are
skipped; longer ones are cropped so arrays stack cleanly for preprocess.py.
"""

from pathlib import Path

import numpy as np

SAMPLES_DIR = Path("./samples")
OUT = Path("./samples.npz")
MIN_LEN = 500


def load_class_dir(class_dir: Path):
    samples = []
    first_shape = None
    for csv_path in sorted(class_dir.glob("*.csv")):
        arr = np.loadtxt(csv_path, delimiter=",")
        arr = np.atleast_2d(arr)
        if arr.shape[0] < MIN_LEN:
            continue
        arr = arr[:MIN_LEN]
        if first_shape is None:
            first_shape = arr.shape[1]
        if arr.shape[1] != first_shape:
            raise ValueError(f"Column mismatch in {class_dir.name}: {csv_path}")
        samples.append(arr)
    return samples


def main():
    data = {}
    for class_dir in sorted(SAMPLES_DIR.iterdir()):
        if not class_dir.is_dir():
            continue
        label = class_dir.name
        samples = load_class_dir(class_dir)
        if not samples:
            print(f"Skipping {label}: no valid samples")
            continue
        data[label] = np.stack(samples)
        print(f"{label}: {data[label].shape[0]} samples, shape {data[label].shape[1:]}" )

    if not data:
        raise ValueError("No data collected; check input directory")

    np.savez(OUT, **data)
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
