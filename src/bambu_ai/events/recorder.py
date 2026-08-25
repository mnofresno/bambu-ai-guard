"""Evidence recorder: saves before/trigger/after frames + metadata per event.

Keeps a small ring buffer of recent frames so an event can include context
from before the trigger frame.
"""
from __future__ import annotations

import json
import re
import time
from collections import deque
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from ..models import Frame


def _slug(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H-%M-%S", time.gmtime(ts))


def _jsonable(v: Any) -> Any:
    if is_dataclass(v):
        return _jsonable(asdict(v))
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


class EventRecorder:
    def __init__(self, root: str | Path, ring_buffer_size: int = 30):
        self.root = Path(root)
        self.ring_buffer_size = ring_buffer_size
        self._ring: deque[Frame] = deque(maxlen=ring_buffer_size)

    def push_frame(self, frame: Frame) -> None:
        self._ring.append(frame)

    def snapshot(self, n: int = 3) -> list[Frame]:
        items = list(self._ring)
        return items[-n:]

    def record(
        self,
        trigger: Frame,
        metadata: dict,
        before: list[Frame] | None = None,
    ) -> Path:
        ts = trigger.timestamp
        safe = re.sub(r"[^0-9A-Za-z-]+", "", _slug(ts))
        event_dir = self.root / safe
        suffix = 0
        while event_dir.exists():
            suffix += 1
            event_dir = self.root / f"{safe}-{suffix:02d}"
        event_dir.mkdir(parents=True, exist_ok=True)

        before = before if before is not None else self.snapshot(3)
        # avoid duplicates: before should be frames prior to the trigger
        prior = [f for f in before if f.timestamp < ts][-2:]
        if prior:
            for i, f in enumerate(prior):
                (event_dir / f"before_{i}.jpg").write_bytes(f.data)
        (event_dir / "trigger.jpg").write_bytes(trigger.data)

        meta = _jsonable(metadata)
        (event_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        return event_dir
