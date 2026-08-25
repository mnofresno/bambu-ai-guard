"""Bambu Lab A1 camera.

The A1 (unlike X1) has no RTSP/MJPEG endpoint. It serves JPEG frames over a
plain TLS socket on port 6000: a binary handshake (32-char "bblp" username +
32-char access code, little-endian, zero padded) followed by a continuous
stream where each JPEG is preceded by framing bytes. Frames arrive roughly
every 1-2 s.

Protocol reference: bambu-connect (mattcar15) and the pybambu library that
ships inside greghesp/ha-bambulab.

The blocking socket I/O runs in a dedicated worker thread so it never blocks
the asyncio event loop; frames are handed out through a thread-safe buffer.
"""
from __future__ import annotations

import asyncio
import logging
import socket
import ssl
import struct
import threading
import time
from collections import deque

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
        return None, soi  # keep partial frame for the next read
    return bytes(buf[soi : eoi + len(JPEG_EOI)]), eoi + len(JPEG_EOI)


class BambuCamera(CameraProvider):
    def __init__(self, host: str, access_code: str, port: int = 6000):
        self.host = host
        self.access_code = access_code
        self.port = port
        self._latest: Frame | None = None
        self._last_frame_at = 0.0
        self._connected = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._frames: deque[Frame] = deque(maxlen=16)
        self._seq = 0
        self._last_returned = 0

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    async def close(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None
        self._connected = False

    async def get_frame(self, timeout: float = 10.0) -> Frame:
        """Return the newest frame not yet returned; wait up to `timeout`."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._seq > self._last_returned and self._frames:
                    frame = self._frames[-1]
                    self._last_returned = self._seq
                    return frame
            await asyncio.sleep(0.05)
        with self._lock:
            if self._frames:
                return self._frames[-1]
        raise TimeoutError(f"no camera frame from {self.host} within {timeout:.0f}s")

    # -- internals (worker thread) -----------------------------------------

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                self._stream_once()
            except Exception:
                log.exception("camera stream error; retrying in 5s")
            self._connected = False
            self._stop.wait(5)

    def _socket_read(self, sock: socket.socket, buf: bytearray) -> bytes | None:
        """Blocking read over a TLS socket; returns one complete JPEG."""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                return None
            buf += chunk
            img, new_start = extract_jpeg(buf, 0)
            if img is not None:
                del buf[:new_start]
                return img

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
                while not self._stop.is_set():
                    img = self._socket_read(ssock, buf)
                    if img is None:
                        break
                    self._on_frame(img)

    def _on_frame(self, img: bytes) -> None:
        frame = Frame(data=img, timestamp=time.time())
        self._last_frame_at = time.monotonic()
        with self._lock:
            self._frames.append(frame)
            self._seq += 1
        self._latest = frame
