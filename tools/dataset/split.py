"""Split a dataset into train/val/test (images + labels).

Usage:
  python tools/dataset/split.py --dataset dataset --val 0.1 --test 0.1 --seed 42
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="dataset")
    p.add_argument("--val", type=float, default=0.1)
    p.add_argument("--test", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    d = Path(a.dataset)
    imgs = sorted((d / "images").glob("*.jpg"))
    random.seed(a.seed)
    random.shuffle(imgs)
    n_test = int(len(imgs) * a.test)
    n_val = int(len(imgs) * a.val)
    splits = {
        "test": imgs[:n_test],
        "val": imgs[n_test:n_test + n_val],
        "train": imgs[n_test + n_val:],
    }
    for name, files in splits.items():
        (d / name / "images").mkdir(parents=True, exist_ok=True)
        (d / name / "labels").mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy(f, d / name / "images" / f.name)
            lbl = d / "labels" / (f.stem + ".txt")
            if lbl.exists():
                shutil.copy(lbl, d / name / "labels" / lbl.name)
        print(f"{name}: {len(files)}")


if __name__ == "__main__":
    main()
