"""Temporal anomaly detection across frames.

A single frame is never enough: a falling tower does not look like spaghetti
in one snapshot. This module keeps a short history of object positions / sizes
and emits temporal signals that the decision engine combines with per-frame
vision scores.

Signals (all 0..1):
  object_displacement  object center drifted from its stable baseline
  collapse             abrupt bbox size change / loss of expected object
  air_printing         nozzle printing with no geometry beneath (heuristic)
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from ..models import Detection, DetectionResult


@dataclass
class _Track:
    cx: float
    cy: float
    w: float
    h: float
    ts: float

    @property
    def area(self) -> float:
        return self.w * self.h


@dataclass
class TemporalSignals:
    object_displacement: float = 0.0
    collapse: float = 0.0
    air_printing: float = 0.0
    details: dict = field(default_factory=dict)


class TemporalAnalyzer:
    """Stateful across frames; safe to unit-test with synthetic sequences.

    Parameters (normalized, 0..1 image coords unless noted):
      displacement_threshold: center movement (fraction of image) that is anomalous
      area_change_threshold:  fractional bbox area change that is anomalous
      stable_frames:          how many early frames define the baseline
      max_age_seconds:        drop older history (bounds memory)
    """

    def __init__(
        self,
        displacement_threshold: float = 0.10,
        area_change_threshold: float = 0.45,
        stable_frames: int = 5,
        max_age_seconds: float = 30.0,
    ):
        self.displacement_threshold = displacement_threshold
        self.area_change_threshold = area_change_threshold
        self.stable_frames = stable_frames
        self.max_age_seconds = max_age_seconds
        self._history: deque[_Track] = deque()
        self._baseline: _Track | None = None
        self._object_count = 0

    def _main_object(self, result: DetectionResult) -> Detection | None:
        """Pick the dominant object to track.

        Prefer explicit object-like labels (failure weights); otherwise fall
        back to the single largest detection — with a generic COCO model the
        largest thing in the printer's field of view is the print itself.
        """
        boxes = [d for d in result.detections if d.bbox is not None]
        if not boxes:
            return None
        preferred = [d for d in boxes if d.label.lower() in (
            "object", "part", "bottle", "cup", "vase",
        )]
        pool = preferred or boxes
        return max(pool, key=lambda d: (d.bbox[2] * d.bbox[3]))

    def _bbox_center(self, d: Detection) -> tuple[float, float, float, float]:
        x, y, w, h = d.bbox
        return x + w / 2, y + h / 2, w, h

    def update(self, result: DetectionResult, now: float | None = None) -> TemporalSignals:
        now = now if now is not None else time.time()
        self._prune(now)
        obj = self._main_object(result)
        if obj is not None:
            cx, cy, w, h = self._bbox_center(obj)
            self._history.append(_Track(cx, cy, w, h, now))
            self._object_count += 1
        else:
            self._object_count = 0

        signals = TemporalSignals()
        if self._baseline is None and self._object_count >= self.stable_frames:
            recent = list(self._history)[-self.stable_frames:]
            self._baseline = _Track(
                sum(t.cx for t in recent) / len(recent),
                sum(t.cy for t in recent) / len(recent),
                sum(t.w for t in recent) / len(recent),
                sum(t.h for t in recent) / len(recent),
                now,
            )

        if self._baseline is not None and self._history:
            last = self._history[-1]
            dx = abs(last.cx - self._baseline.cx)
            dy = abs(last.cy - self._baseline.cy)
            dist = max(dx, dy)
            signals.object_displacement = min(
                1.0, dist / (2 * self.displacement_threshold)
            ) if dist > self.displacement_threshold else 0.0

            base_area = self._baseline.area
            if base_area > 0:
                area_ratio = last.area / base_area
                deviation = abs(1.0 - area_ratio)
                if deviation > self.area_change_threshold:
                    signals.collapse = min(
                        1.0, deviation / (2 * self.area_change_threshold)
                    )
                # object vanished entirely relative to a previously stable object
                if obj is None and self._baseline is not None:
                    signals.collapse = max(signals.collapse, 0.6)

            # air printing heuristic: object present but small/shrinking with
            # high displacement => material being laid without support.
            if (
                obj is not None
                and signals.object_displacement > 0.5
                and signals.collapse > 0.2
            ):
                signals.air_printing = min(1.0, 0.5 * (signals.object_displacement + signals.collapse))

        signals.details = {
            "n_frames": self._object_count,
            "has_baseline": self._baseline is not None,
        }
        return signals

    def _prune(self, now: float) -> None:
        while self._history and (now - self._history[0].ts) > self.max_age_seconds:
            self._history.popleft()

    def reset(self) -> None:
        self._history.clear()
        self._baseline = None
        self._object_count = 0
