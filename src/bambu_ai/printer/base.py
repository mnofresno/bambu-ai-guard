"""Printer control abstraction."""
from __future__ import annotations

import abc

from ..models import PrinterStatus


class PrinterController(abc.ABC):
    @abc.abstractmethod
    async def connect(self) -> None: ...

    @abc.abstractmethod
    async def get_status(self) -> PrinterStatus: ...

    @abc.abstractmethod
    async def pause(self, reason: str) -> None:
        """Pause the print. Implementation must re-verify before acting (see engine)."""

    @abc.abstractmethod
    async def resume(self) -> None: ...

    @abc.abstractmethod
    async def close(self) -> None: ...

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool: ...
