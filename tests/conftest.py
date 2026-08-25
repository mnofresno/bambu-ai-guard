"""Shared fixtures and synthetic frame-sequence builders."""
from __future__ import annotations

import pytest

from bambu_ai.models import (
    Detection,
    DetectionContext,
    DetectionResult,
    FailureType,
    Frame,
    PrinterState,
)
from bambu_ai.temporal import TemporalAnalyzer
from bambu_ai.vision.mock import MockModel


def make_object(cx: float, cy: float, w: float, h: float, conf: float = 0.9):
    return Detection(label="object", confidence=conf, bbox=(cx - w / 2, cy - h / 2, w, h))


def make_result(signals: dict[str, float], detections: list[Detection] | None = None) -> DetectionResult:
    return DetectionResult(detections=detections or [], signal_scores=signals)


def normal_frame(cx=0.5, cy=0.5, w=0.2, h=0.3) -> DetectionResult:
    return make_result(
        {"spaghetti": 0.0, "blob": 0.0, "object": 0.9},
        [make_object(cx, cy, w, h)],
    )


def moved_frame(dx=0.15, dy=0.0, w=0.2, h=0.3) -> DetectionResult:
    return make_result(
        {"spaghetti": 0.0, "blob": 0.0, "object": 0.9},
        [make_object(0.5 + dx, 0.5 + dy, w, h)],
    )


def fallen_frame() -> DetectionResult:
    # object displaced + shrunk (collapsed) with spaghetti signal
    return make_result(
        {"spaghetti": 0.95, "blob": 0.0, "collapse": 0.0, "object": 0.9},
        [make_object(0.62, 0.55, 0.24, 0.12)],
    )


def mock_model_from_sequence(results: list[DetectionResult]) -> MockModel:
    def factory(i: int) -> DetectionResult:
        # repeat the last result once we run out (stays anomalous)
        return results[min(i, len(results) - 1)]
    return MockModel(factory=factory)


@pytest.fixture
def sequence() -> list[DetectionResult]:
    """normal, normal, normal, tower_moves, tower_falls."""
    return [
        normal_frame(),
        normal_frame(),
        normal_frame(),
        moved_frame(),
        fallen_frame(),
    ]


@pytest.fixture
def context() -> DetectionContext:
    return DetectionContext(printer_state=PrinterState.PRINTING, elapsed_seconds=100.0)


@pytest.fixture
def analyzer() -> TemporalAnalyzer:
    return TemporalAnalyzer(
        displacement_threshold=0.10,
        area_change_threshold=0.45,
        stable_frames=3,
    )
