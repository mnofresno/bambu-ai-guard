"""Pluggable vision model interface."""
from __future__ import annotations

import abc

from ..models import DetectionContext, DetectionResult, Frame


class VisionModel(abc.ABC):
    name: str = "abstract"

    @abc.abstractmethod
    async def analyze(self, frame: Frame, context: DetectionContext) -> DetectionResult: ...

    async def close(self) -> None:
        return None
