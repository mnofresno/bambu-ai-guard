"""Generate YOLO label files from a simple annotations file.

Annotations format (JSON list):
[
  {"image": "0001.jpg", "class": "spaghetti",
   "boxes": [[x1,y1,x2,y2], ...]}   # pixel coords
]

Class names must match the ids in --class-map (name -> id).

Usage:
  python tools/dataset/make_yolo.py --annotations ann.json \
      --images dataset/images --out dataset/labels \
      --class-map spaghetti=0 blob=1 adhesion_loss=2 collapse=3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_class_map(s: str) -> dict[str, int]:
    out = {}
    for part in s.split():
        name, cid = part.split("=")
        out[name] = int(cid)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--annotations", required=True)
    p.add_argument("--images", default="dataset/images")
    p.add_argument("--out", default="dataset/labels")
    p.add_argument("--class-map", default="spaghetti=0 blob=1 adhesion_loss=2 collapse=3 air_printing=4")
    a = p.parse_args()

    cmap = parse_class_map(a.class_map)
    images = Path(a.images)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    anns = json.loads(Path(a.annotations).read_text())
    for ann in anns:
        img = images / ann["image"]
        if not img.exists():
            continue
        from PIL import Image
        w, h = Image.open(img).size
        lines = []
        for (x1, y1, x2, y2) in ann.get("boxes", []):
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lines.append(f"{cmap[ann['class']]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (out / (img.stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
    print(f"wrote labels for {len(anns)} images -> {out}")


if __name__ == "__main__":
    main()
