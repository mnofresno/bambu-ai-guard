"""Camera abstraction: the rest of the app never knows where frames come from."""
from __future__ import annotations

import abc

from ..models import Frame


class CameraProvider(abc.ABC):
    @abc.abstractmethod
    async def connect(self) -> None: ...

    @abc.abstractmethod
    async def get_frame(self) -> Frame:
        """Return the latest available frame (blocking up to a short timeout)."""

    @abc.abstractmethod
    async def close(self) -> None: ...

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool: ...
