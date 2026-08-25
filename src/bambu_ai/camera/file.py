"""File camera: plays back a directory of JPEGs in order. For testing/dataset work."""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

from ..models import Frame
from .base import CameraProvider


class FileCamera(CameraProvider):
    def __init__(self, directory: str | Path, interval: float = 1.0):
        self.directory = Path(directory)
        self.interval = interval
        self._files: list[Path] = []
        self._index = 0
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        self._files = sorted(self.directory.glob("*.jpg")) + sorted(self.directory.glob("*.jpeg"))
        if not self._files:
            raise FileNotFoundError(f"no JPEGs in {self.directory}")
        self._connected = True

    async def get_frame(self, timeout: float = 5.0) -> Frame:
        if not self._connected:
            raise RuntimeError("not connected")
        await asyncio.sleep(self.interval)
        path = self._files[self._index % len(self._files)]
        self._index += 1
        return Frame(data=path.read_bytes(), timestamp=time.time())

    async def close(self) -> None:
        self._connected = False
