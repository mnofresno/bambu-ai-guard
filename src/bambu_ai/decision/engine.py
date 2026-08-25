"""Failure decision engine: a small, testable state machine.

Combines per-frame vision scores with temporal signals into an overall risk,
requires *consecutive* high-risk frames within an observation window before
confirming a failure, and enforces a cooldown between pauses. A single
mediocre frame never triggers a pause.

The engine is pure (no I/O). It emits :class:`Decision` objects; the caller
decides whether to actually pause the printer (auto) or log "would pause"
(shadow mode).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..models import GuardState, PrinterState
from ..temporal.analyzer import TemporalSignals

# severity weight per failure signal type
_SEVERITY = {
    "spaghetti": 1.0,
    "collapse": 1.0,
    "object_displacement": 0.9,
    "adhesion_loss": 0.9,
    "blob": 0.7,
    "air_printing": 0.8,
}


@dataclass
class DecisionConfig:
    pause_threshold: float = 0.90
    consecutive_frames: int = 3
    observation_window_seconds: float = 15.0
    cooldown_seconds: float = 60.0
    suspicious_threshold: float = 0.5


@dataclass
class Decision:
    state: GuardState
    risk: float
    failure_type: str
    action: str  # "none" | "pause" | "would_pause" | "resume"
    reason: str
    ts: float = field(default_factory=time.time)


class DecisionEngine:
    def __init__(self, cfg: DecisionConfig, auto_pause: bool = False):
        self.cfg = cfg
        self.auto_pause = auto_pause
        self.state = GuardState.NORMAL
        self._consecutive = 0
        self._window_start: float | None = None
        self._last_pause_ts: float = 0.0
        self._confirmed_failure = "none"
        self._peak_risk = 0.0

    # -- inputs --------------------------------------------------------------

    def combined_risk(self, vision_signals: dict[str, float], temporal: TemporalSignals) -> tuple[float, str]:
        """Merge frame + temporal signals into (risk 0..1, dominant failure type)."""
        merged: dict[str, float] = dict(vision_signals)
        merged["object_displacement"] = max(
            merged.get("object_displacement", 0.0), temporal.object_displacement
        )
        merged["collapse"] = max(merged.get("collapse", 0.0), temporal.collapse)
        merged["air_printing"] = max(merged.get("air_printing", 0.0), temporal.air_printing)
        # drop the neutral "object" signal from risk
        merged.pop("object", None)
        if not merged:
            return 0.0, "none"
        best_type, best = max(merged.items(), key=lambda kv: kv[1])
        # risk = weighted by severity, capped at 1
        risk = min(1.0, best * _SEVERITY.get(best_type, 0.5))
        return risk, best_type

    # -- core transition ------------------------------------------------------

    def step(
        self,
        risk: float,
        failure_type: str,
        printer_state: PrinterState,
        now: float | None = None,
    ) -> Decision:
        now = now if now is not None else time.time()
        self._peak_risk = max(self._peak_risk, risk)

        # cooldown: ignore new confirmations right after a pause
        if self.state in (GuardState.PAUSED_BY_AI, GuardState.PAUSING) and \
                (now - self._last_pause_ts) < self.cfg.cooldown_seconds:
            return Decision(self.state, risk, failure_type, "none", "cooldown", now)

        # only act on failures while actively printing
        printing = printer_state in (PrinterState.PRINTING, PrinterState.PAUSED)

        if risk >= self.cfg.pause_threshold and failure_type != "none":
            self._count(risk, now)
        elif risk >= self.cfg.suspicious_threshold:
            # suspicious but not confirmatory: keep tracking, don't reset hard
            self.state = self._escalate(self.state, GuardState.SUSPICIOUS)
            return Decision(self.state, risk, failure_type, "none", "suspicious", now)
        else:
            # clean frame: decay
            self._decay(now)
            return Decision(self.state, risk, failure_type, "none", "normal", now)

        if self._consecutive >= self.cfg.consecutive_frames:
            return self._confirm(failure_type, risk, printer_state, now, printing)
        self.state = self._escalate(self.state, GuardState.SUSPICIOUS)
        return Decision(self.state, risk, failure_type, "none", "accumulating", now)

    def _count(self, risk: float, now: float) -> None:
        # require the consecutive frames to fall inside the observation window
        if self._window_start is None or \
                (now - self._window_start) > self.cfg.observation_window_seconds:
            self._window_start = now
            self._consecutive = 0
        self._consecutive += 1

    def _decay(self, now: float) -> None:
        if self.state in (GuardState.SUSPICIOUS,):
            self.state = GuardState.NORMAL
        self._consecutive = 0
        self._window_start = None
        self._peak_risk = 0.0

    def _confirm(
        self,
        failure_type: str,
        risk: float,
        printer_state: PrinterState,
        now: float,
        printing: bool,
    ) -> Decision:
        self.state = GuardState.CONFIRMED_FAILURE
        self._confirmed_failure = failure_type
        if not printing:
            return Decision(
                self.state, risk, failure_type, "none",
                "not_printing", now,
            )
        if self.auto_pause:
            self.state = GuardState.PAUSING
            self._last_pause_ts = now
            self._reset_tracking()
            return Decision(
                self.state, risk, failure_type, "pause",
                f"{failure_type} confirmed over {self.cfg.consecutive_frames} frames",
                now,
            )
        self._last_pause_ts = now
        self._reset_tracking()
        return Decision(
            self.state, risk, failure_type, "would_pause",
            f"shadow mode: {failure_type} confirmed",
            now,
        )

    def _reset_tracking(self) -> None:
        self._consecutive = 0
        self._window_start = None
        self._peak_risk = 0.0

    @staticmethod
    def _escalate(current: GuardState, target: GuardState) -> GuardState:
        order = [
            GuardState.NORMAL, GuardState.SUSPICIOUS, GuardState.CONFIRMED_FAILURE,
            GuardState.PAUSING, GuardState.PAUSED_BY_AI,
        ]
        return target if order.index(target) > order.index(current) else current

    # -- external transitions -------------------------------------------------

    def mark_paused(self, now: float | None = None) -> None:
        self.state = GuardState.PAUSED_BY_AI
        self._last_pause_ts = now if now is not None else time.time()
        self._reset_tracking()

    def mark_resumed(self, now: float | None = None) -> None:
        self.state = GuardState.NORMAL
        self._reset_tracking()
