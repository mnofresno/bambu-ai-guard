"""Bambu Lab A1 camera.

The A1 (unlike X1) has no RTSP/MJPEG endpoint. It serves JPEG frames over a
plain TLS socket on port 6000: a binary handshake (32-char "bblp" username +
32-char access code, little-endian, zero padded) followed by a continuous
stream where each JPEG is preceded by a 16-byte header (LE payload size).

Protocol reference: bambu-connect (mattcar15) and the pybambu library that
ships inside greghesp/ha-bambulab. Frames arrive roughly every 1-2 s.
"""
from __future__ import annotations

import asyncio
import logging
import socket
import ssl
import struct
import time

from ..models import Frame
from .base import CameraProvider

log = logging.getLogger(__name__)

JPEG_SOI = b"\xff\xd8"
JPEG_EOI = b"\xff\xd9"


def build_auth_packet(username: str, access_code: str) -> bytes:
    """Build the 80-byte handshake packet (16 header + 32 user + 32 code)."""
    data = bytearray()
    data += struct.pack("<I", 0x40)
    data += struct.pack("<I", 0x3000)
    data += struct.pack("<I", 0)
    data += struct.pack("<I", 0)
    data += username.encode("ascii").ljust(32, b"\x00")
    data += access_code.encode("ascii").ljust(32, b"\x00")
    assert len(data) == 80, len(data)
    return bytes(data)


def extract_jpeg(buf: bytearray, start: int) -> tuple[bytes | None, int]:
    """Find the first complete JPEG in buf[start:]; returns (jpeg, new_start)."""
    soi = buf.find(JPEG_SOI, start)
    if soi == -1:
        return None, start
    eoi = buf.find(JPEG_EOI, soi + len(JPEG_SOI))
    if eoi == -1:
        # keep the partial frame for the next read
        return None, soi
    return bytes(buf[soi : eoi + len(JPEG_EOI)]), eoi + len(JPEG_EOI)


class BambuCamera(CameraProvider):
    def __init__(self, host: str, access_code: str, port: int = 6000):
        self.host = host
        self.access_code = access_code
        self.port = port
        self._latest: Frame | None = None
        self._connected = False
        self._task: asyncio.Task | None = None
        self._last_frame_at = 0.0

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        self._task = asyncio.get_running_loop().create_task(self._run())

    async def close(self) -> None:
        self._connected = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    async def get_frame(self, timeout: float = 10.0) -> Frame:
        """Wait for the latest frame; timeout if the stream is silent."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._latest and self._last_frame_at >= time.monotonic() - 5.0:
                return self._latest
            await asyncio.sleep(0.1)
        if self._latest:
            return self._latest
        raise TimeoutError(f"no camera frame from {self.host} within {timeout:.0f}s")

    # -- internals ---------------------------------------------------------

    def _socket_read(self, sock: socket.socket, buf: bytearray) -> bytes | None:
        """Blocking read loop over a TLS socket; returns one complete JPEG."""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                return None
            buf += chunk
            img, new_start = extract_jpeg(buf, 0)
            if img is not None:
                del buf[:new_start]
                return img

    async def _run(self) -> None:
        while True:
            try:
                self._stream_once()
            except Exception:
                log.exception("camera stream error; retrying in 5s")
            self._connected = False
            await asyncio.sleep(5)

    def _stream_once(self) -> None:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with socket.create_connection((self.host, self.port), timeout=15) as raw:
            with ctx.wrap_socket(raw, server_hostname=self.host) as ssock:
                ssock.sendall(build_auth_packet("bblp", self.access_code))
                log.info("camera connected: %s:%s", self.host, self.port)
                self._connected = True
                buf = bytearray()
                while self._connected:
                    img = self._socket_read(ssock, buf)
                    if img is None:
                        break
                    self._latest = Frame(data=img, timestamp=time.time())
                    self._last_frame_at = time.monotonic()
