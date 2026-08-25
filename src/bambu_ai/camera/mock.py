"""Mock camera for tests and offline development."""
from __future__ import annotations

import asyncio
import time
from typing import Callable

from ..models import Frame
from .base import CameraProvider


class MockCamera(CameraProvider):
    def __init__(self, frame_factory: Callable[[], bytes] | None = None):
        self.frame_factory = frame_factory or (lambda: b"mock-jpeg")
        self._connected = False
        self._latest: Frame | None = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        self._connected = True

    def push(self, data: bytes) -> None:
        self._latest = Frame(data=data, timestamp=time.time())

    async def get_frame(self, timeout: float = 2.0) -> Frame:
        if not self._connected:
            raise RuntimeError("not connected")
        deadline = time.monotonic() + timeout
        while self._latest is None and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        if self._latest is None:
            self._latest = Frame(data=self.frame_factory(), timestamp=time.time())
        frame, self._latest = self._latest, None
        return frame

    async def close(self) -> None:
        self._connected = False
