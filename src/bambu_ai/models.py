"""Shared data types for the bambu-ai-guard pipeline."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class FailureType(str, Enum):
    NONE = "none"
    SPAGHETTI = "spaghetti"
    BLOB = "blob"
    ADHESION_LOSS = "adhesion_loss"
    COLLAPSE = "collapse"
    AIR_PRINTING = "air_printing"
    OBJECT_DISPLACEMENT = "object_displacement"


class PrinterState(str, Enum):
    UNKNOWN = "unknown"
    IDLE = "idle"
    PRINTING = "printing"
    PAUSED = "paused"
    FINISHED = "finished"
    ERROR = "error"


class GuardState(str, Enum):
    NORMAL = "NORMAL"
    SUSPICIOUS = "SUSPICIOUS"
    CONFIRMED_FAILURE = "CONFIRMED_FAILURE"
    PAUSING = "PAUSING"
    PAUSED_BY_AI = "PAUSED_BY_AI"


@dataclass
class Frame:
    """A single camera frame. `data` is JPEG bytes; `bgr` (optional) a numpy array."""
    data: bytes
    timestamp: float = field(default_factory=time.time)
    bgr: "object | None" = None

    @property
    def ts(self) -> float:
        return self.timestamp


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: tuple[float, float, float, float] | None = None  # x, y, w, h (normalized)


@dataclass
class DetectionContext:
    """Everything a vision model may use besides the raw frame."""
    printer_state: PrinterState
    elapsed_seconds: float
    layer_height_mm: float | None = None
    previous_detections: list[Detection] = field(default_factory=list)


@dataclass
class DetectionResult:
    detections: list[Detection]
    signal_scores: dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0

    def score(self, failure: FailureType) -> float:
        return self.signal_scores.get(failure.value, 0.0)


@dataclass
class PrinterStatus:
    state: PrinterState
    job_name: str = ""
    progress_pct: float = 0.0
    elapsed_seconds: float = 0.0
    temperature: float | None = None


@dataclass
class GuardEvent:
    """A confirmed (or would-be) failure event."""
    timestamp: float
    state: GuardState
    failure_type: FailureType
    confidence: float
    decision: str  # "pause", "would_pause", "observed"
    details: dict = field(default_factory=dict)
