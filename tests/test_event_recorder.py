"""Event recorder: before/trigger/after + metadata, ring buffer."""
from __future__ import annotations

import json

from bambu_ai.events import EventRecorder
from bambu_ai.models import Frame


def test_ring_buffer_bounded():
    rec = EventRecorder("/tmp/evt", ring_buffer_size=3)
    for i in range(6):
        rec.push_frame(Frame(data=b"x", timestamp=float(i)))
    assert len(rec._ring) == 3
    # oldest evicted
    assert rec._ring[0].timestamp == 3.0


def test_record_writes_evidence(tmp_path):
    rec = EventRecorder(tmp_path, ring_buffer_size=5)
    for i in range(4):
        rec.push_frame(Frame(data=f"frame{i}".encode(), timestamp=float(i)))
    trigger = Frame(data=b"TRIGGER", timestamp=4.0)
    out = rec.record(trigger, {
        "printer": "p", "job": "j", "timestamp": 4.0,
        "failure_type": "spaghetti", "confidence": 0.9,
        "model": "m", "decision": "would_pause",
    })
    assert (out / "trigger.jpg").read_bytes() == b"TRIGGER"
    before = list(out.glob("before_*.jpg"))
    assert len(before) == 2  # two frames prior to trigger
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["failure_type"] == "spaghetti"
    assert meta["decision"] == "would_pause"
    assert meta["confidence"] == 0.9
