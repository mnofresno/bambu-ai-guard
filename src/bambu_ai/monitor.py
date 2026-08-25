"""Orchestrates the monitoring pipeline.

frame -> vision -> temporal -> decision -> (pause | would_pause) -> evidence
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from .camera import CameraProvider
from .config import Config
from .decision import DecisionConfig, DecisionEngine
from .events import EventRecorder
from .models import DetectionContext, GuardState, PrinterStatus
from .printer import PrinterController
from .temporal import TemporalAnalyzer
from .vision import VisionModel

log = logging.getLogger("bambu_ai.monitor")

# structured event names (for observability / metrics)
EVT = {
    "printer_connected": "printer.connected",
    "camera_connected": "camera.connected",
    "frame_sampled": "frame.sampled",
    "inference_completed": "inference.completed",
    "anomaly_detected": "anomaly.detected",
    "failure_confirmed": "failure.confirmed",
    "pause_requested": "printer.pause.requested",
    "pause_success": "printer.pause.success",
    "pause_failed": "printer.pause.failed",
}


@dataclass
class MonitorStatus:
    enabled: bool = True
    state: GuardState = GuardState.NORMAL
    printer: PrinterStatus | None = None
    last_signals: dict = field(default_factory=dict)
    last_risk: float = 0.0
    last_failure: str = "none"
    events: list = field(default_factory=list)
    frames_processed: int = 0
    confirmed_failures: int = 0
    pauses: int = 0
    last_inference_ms: float = 0.0
    last_frame_ts: float = 0.0


class Monitor:
    def __init__(
        self,
        cfg: Config,
        camera: CameraProvider,
        printer: PrinterController,
        vision: VisionModel,
        temporal: TemporalAnalyzer | None = None,
        decision: DecisionEngine | None = None,
        recorder: EventRecorder | None = None,
    ):
        self.cfg = cfg
        self.camera = camera
        self.printer = printer
        self.vision = vision
        self.temporal = temporal or TemporalAnalyzer()
        self.decision = decision or DecisionEngine(
            DecisionConfig(
                pause_threshold=cfg.pause_threshold,
                consecutive_frames=cfg.consecutive_frames,
                observation_window_seconds=cfg.observation_window_seconds,
                cooldown_seconds=cfg.cooldown_seconds,
            ),
            auto_pause=cfg.auto_pause,
        )
        self.recorder = recorder or EventRecorder(cfg.events_dir, cfg.ring_buffer_size)
        self.status = MonitorStatus()
        self._stop = asyncio.Event()

    def stop(self) -> None:
        self._stop.set()

    # -- main loop -----------------------------------------------------------

    async def run(self) -> None:
        period = 1.0 / max(0.05, self.cfg.inference_fps)
        log.info("monitor start: fps=%.2f auto_pause=%s", self.cfg.inference_fps, self.cfg.auto_pause)
        await self.camera.connect()
        log.info(EVT["camera_connected"])
        try:
            await self.printer.connect()
            log.info(EVT["printer_connected"])
        except Exception:
            log.exception("printer connect failed; continuing with printer offline")
        try:
            while not self._stop.is_set():
                t0 = time.monotonic()
                try:
                    await self._tick()
                except Exception:
                    log.exception("tick error")
                elapsed = time.monotonic() - t0
                await asyncio.sleep(max(0.05, period - elapsed))
        finally:
            await self.camera.close()
            await self.printer.close()
            await self.vision.close()
            log.info("monitor stopped")

    async def _tick(self) -> None:
        frame = await self.camera.get_frame()
        self.recorder.push_frame(frame)
        self.status.frames_processed += 1
        self.status.last_frame_ts = frame.timestamp
        log.debug(EVT["frame_sampled"])

        try:
            status = await self.printer.get_status()
        except Exception:
            status = PrinterStatus(state=PrinterState.UNKNOWN)
        self.status.printer = status

        ctx = DetectionContext(
            printer_state=status.state,
            elapsed_seconds=status.elapsed_seconds,
        )
        result = await self.vision.analyze(frame, ctx)
        self.status.last_inference_ms = result.latency_ms
        log.debug(EVT["inference_completed"] + f" latency_ms={result.latency_ms:.1f}")

        tsignals = self.temporal.update(result, now=frame.timestamp)
        risk, failure = self.decision.combined_risk(result.signal_scores, tsignals)
        self.status.last_signals = {**result.signal_scores,
                                    "object_displacement": tsignals.object_displacement,
                                    "collapse": tsignals.collapse,
                                    "air_printing": tsignals.air_printing}
        self.status.last_risk = risk
        self.status.last_failure = failure

        decision = self.decision.step(risk, failure, status.state, now=frame.timestamp)
        self.status.state = self.decision.state

        if risk >= self.decision.cfg.suspicious_threshold:
            log.info(EVT["anomaly_detected"] + f" risk={risk:.2f} type={failure}")

        if decision.action == "pause":
            self.status.confirmed_failures += 1
            log.info(EVT["failure_confirmed"] + f" type={failure} risk={risk:.2f}")
            self._record(frame, failure, risk, "pause")
            await self._do_pause(failure, risk, status)
        elif decision.action == "would_pause":
            self.status.confirmed_failures += 1
            log.info(EVT["failure_confirmed"] + f" type={failure} risk={risk:.2f} (shadow)")
            self._record(frame, failure, risk, "would_pause")

    # -- actions -------------------------------------------------------------

    async def _do_pause(self, failure: str, risk: float, status: PrinterStatus) -> None:
        # re-verify before touching the printer
        fresh = await self.printer.get_status()
        if fresh.state not in (status.state,):
            log.info("re-verify: state changed to %s; skip pause", fresh.state)
            return
        reason = f"AI failure detected: {failure} (risk={risk:.2f})"
        log.info(EVT["pause_requested"] + f" reason={reason}")
        try:
            await self.printer.pause(reason)
            self.decision.mark_paused()
            self.status.pauses += 1
            log.info(EVT["pause_success"])
        except Exception:
            log.exception(EVT["pause_failed"])

    def _record(self, frame, failure: str, risk: float, decision: str) -> None:
        self.recorder.record(
            frame,
            {
                "printer": self.cfg.printer_host,
                "job": (self.status.printer.job_name if self.status.printer else ""),
                "timestamp": frame.timestamp,
                "failure_type": failure,
                "confidence": round(risk, 3),
                "model": self.vision.name,
                "decision": decision,
            },
        )
        self.status.events.append({
            "ts": frame.timestamp, "type": failure,
            "risk": round(risk, 3), "decision": decision,
        })
        self.status.events = self.status.events[-20:]
