"""Mock printer for tests and offline development."""
from __future__ import annotations

from ..models import PrinterState, PrinterStatus
from .base import PrinterController


class MockPrinter(PrinterController):
    def __init__(self, state: PrinterState = PrinterState.PRINTING):
        self._state = state
        self._connected = False
        self.paused = False
        self.resume_calls = 0
        self.pause_reasons: list[str] = []

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        self._connected = True

    async def close(self) -> None:
        self._connected = False

    async def get_status(self) -> PrinterStatus:
        effective = PrinterState.PAUSED if self.paused else self._state
        return PrinterStatus(state=effective, job_name="mock-job", progress_pct=42.0, elapsed_seconds=100.0)

    async def pause(self, reason: str) -> None:
        self.paused = True
        self.pause_reasons.append(reason)

    async def resume(self) -> None:
        self.paused = False
        self.resume_calls += 1
