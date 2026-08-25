"""Mock vision model: scripted results for tests and offline dev."""
from __future__ import annotations

from typing import Callable

from ..models import Detection, DetectionContext, DetectionResult, Frame
from .base import VisionModel


class MockModel(VisionModel):
    """Callable drives per-frame results.

    factory(frame_index) -> DetectionResult
    """
    name = "mock"

    def __init__(self, factory: Callable[[int], DetectionResult] | None = None):
        self.factory = factory
        self.index = 0
        self.calls = 0

    async def analyze(self, frame: Frame, context: DetectionContext) -> DetectionResult:
        self.calls += 1
        if self.factory is not None:
            result = self.factory(self.index)
            self.index += 1
            return result
        return DetectionResult(detections=[], signal_scores={
            "spaghetti": 0.0, "blob": 0.0, "adhesion_loss": 0.0,
            "collapse": 0.0, "air_printing": 0.0, "object": 0.5,
        })
