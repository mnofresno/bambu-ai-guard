"""Copy camera frames into a YOLO-style dataset/images/ directory.

Usage:
  python tools/dataset/extract_frames.py --src frames --dst dataset --limit 200
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="frames")
    p.add_argument("--dst", default="dataset")
    p.add_argument("--limit", type=int, default=200)
    a = p.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "labels").mkdir(parents=True, exist_ok=True)
    (dst / "events").mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*.jpg")) + sorted(src.glob("*.jpeg"))
    files = files[: a.limit]
    for f in files:
        shutil.copy(f, dst / "images" / f.name)
    print(f"copied {len(files)} frames -> {dst}/images")


if __name__ == "__main__":
    main()
