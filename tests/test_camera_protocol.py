"""Bambu A1 camera protocol: auth packet + JPEG framing (pure, no network)."""
from __future__ import annotations

import struct

from bambu_ai.camera.bambu import build_auth_packet, extract_jpeg


def test_auth_packet_layout():
    pkt = build_auth_packet("bblp", "1234")
    assert len(pkt) == 80
    # 4-byte header words
    assert struct.unpack("<I", pkt[0:4])[0] == 0x40
    assert struct.unpack("<I", pkt[4:8])[0] == 0x3000
    # username zero padded to 32 (offset 16..48)
    assert pkt[16:20] == b"bblp"
    assert pkt[20:48] == b"\x00" * 28
    # access code zero padded to 32 (offset 48..80)
    assert pkt[48:52] == b"1234"
    assert pkt[52:80] == b"\x00" * 28


def test_extract_jpeg_complete():
    jpeg = b"\xff\xd8\xff\xe0HELLO\xff\xd9"
    buf = bytearray(b"garbage" + jpeg + b"tail\xff\xd8\xff\xe0NEXT\xff\xd9")
    out, start = extract_jpeg(buf, 0)
    assert out == jpeg
    # resume offset is right after the first EOI
    assert buf[start:start + 4] == b"tail"


def test_extract_jpeg_partial_keeps_buffer():
    buf = bytearray(b"\xff\xd8\xff\xe0PARTIAL")  # no EOI yet
    out, start = extract_jpeg(buf, 0)
    assert out is None
    assert start == 0  # keeps from SOI for next read


def test_extract_jpeg_no_soi():
    out, start = extract_jpeg(bytearray(b"no image here"), 0)
    assert out is None
    assert start == 0
