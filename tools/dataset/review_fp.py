"""Review recorded failure events to hunt false positives.

Lists every event dir under events/ with its metadata; optionally filters by
min confidence or failure type, and can mark an event as a false positive by
writing a `fp.json` marker (excluded from future fine-tuning).

Usage:
  python tools/dataset/review_fp.py --events events --min-conf 0.8
  python tools/dataset/review_fp.py --events events --mark <event-dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--events", default="events")
    p.add_argument("--min-conf", type=float, default=0.0)
    p.add_argument("--type", default=None)
    p.add_argument("--mark", default=None, help="event dir to mark as false positive")
    a = p.parse_args()
    root = Path(a.events)

    if a.mark:
        d = root / a.mark if not Path(a.mark).is_absolute() else Path(a.mark)
        (d / "fp.json").write_text(json.dumps({"false_positive": True}))
        print(f"marked {d.name} as false positive")
        return

    for d in sorted(root.iterdir()):
        meta = d / "metadata.json"
        if not meta.exists():
            continue
        m = json.loads(meta.read_text())
        if m.get("confidence", 0) < a.min_conf:
            continue
        if a.type and m.get("failure_type") != a.type:
            continue
        fp = " [FP]" if (d / "fp.json").exists() else ""
        print(f"{d.name}  {m.get('failure_type'):<18} conf={m.get('confidence'):<6} "
              f"decision={m.get('decision')}{fp}")


if __name__ == "__main__":
    main()
