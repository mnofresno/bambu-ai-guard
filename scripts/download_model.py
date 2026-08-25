"""Download the default YOLOv8n ONNX model into models/.

The model weights are not committed (see .gitignore). Run:
  uv run python scripts/download_model.py

Source: a public YOLOv8n detection ONNX (COCO, 80 classes).
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

URL = ("https://huggingface.co/salim4n/yolov8n-detect-onnx/resolve/main/"
       "yolov8n-onnx-web/yolov8n.onnx")
DEST = Path(__file__).resolve().parent.parent / "models" / "yolov8n.onnx"


def main() -> None:
    if DEST.exists() and DEST.stat().st_size > 1_000_000:
        print(f"already present: {DEST}")
        return
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {URL} -> {DEST}")
    urllib.request.urlretrieve(URL, DEST)
    print(f"done ({DEST.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
