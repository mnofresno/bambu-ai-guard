"""Config loading and ${ENV} expansion."""
from __future__ import annotations

import os

from bambu_ai.config import Config, expand_env


def test_expand_env(monkeypatch):
    monkeypatch.setenv("FOO", "bar")
    assert expand_env("${FOO}") == "bar"
    assert expand_env("${MISSING:-dflt}") == "dflt"
    assert expand_env("${MISSING}") == ""


def test_load_yaml(tmp_path, monkeypatch):
    monkeypatch.setenv("BAMBU_HOST", "10.0.0.5")
    p = tmp_path / "config.yaml"
    p.write_text(
        "printer:\n  host: ${BAMBU_HOST}\n  serial: ABC123\n  access_code: code\n"
        "camera:\n  inference_fps: 2\n"
        "vision:\n  provider: onnx\n  model: models/m.onnx\n"
        "actions:\n  auto_pause: true\n"
    )
    cfg = Config.load(p)
    assert cfg.printer_host == "10.0.0.5"
    assert cfg.printer_serial == "ABC123"
    assert cfg.inference_fps == 2.0
    assert cfg.auto_pause is True
    assert cfg.vision_model == "models/m.onnx"


def test_defaults(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("printer:\n  host: h\n  serial: s\n  access_code: c\n")
    cfg = Config.load(p)
    assert cfg.pause_threshold == 0.90
    assert cfg.consecutive_frames == 3
    assert cfg.auto_pause is False
