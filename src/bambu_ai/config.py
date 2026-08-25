"""Configuration loading with ${ENV} / ${ENV:-default} expansion."""
from __future__ import annotations

import os
import re
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_ENV_RE = re.compile(r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}")


def expand_env(value: str) -> str:
    def repl(m: re.Match[str]) -> str:
        default = m.group("default")
        return os.environ.get(m.group("name"), default if default is not None else "")

    return _ENV_RE.sub(repl, value)


def _expand(obj: Any) -> Any:
    if isinstance(obj, str):
        return expand_env(obj)
    if isinstance(obj, dict):
        return {k: _expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand(v) for v in obj]
    return obj


def _load_dotenv(path: Path) -> None:
    """Load simple KEY=value entries without overriding the process env."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"invalid boolean value: {value!r}")


def _finite(name: str, value: float, minimum: float, maximum: float) -> float:
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be finite and between {minimum} and {maximum}")
    return value


@dataclass
class Config:
    raw: dict = field(default_factory=dict)

    # printer
    printer_host: str = ""
    printer_serial: str = ""
    printer_access_code: str = ""
    printer_mqtt_port: int = 8883
    printer_tls_ca_file: str = ""
    printer_tls_insecure: bool = False

    # camera
    camera_provider: str = "bambu"
    camera_port: int = 6000
    inference_fps: float = 1.0
    camera_file_dir: str = "frames"

    # vision
    vision_provider: str = "onnx"
    vision_model: str = "models/yolov8n.onnx"
    vision_backend: str = "coreml"
    vision_input_size: int = 640
    vision_confidence: float = 0.35
    remote_base_url: str = ""
    remote_api_key: str = ""
    remote_model_name: str = ""

    # decision
    pause_threshold: float = 0.90
    consecutive_frames: int = 3
    observation_window_seconds: float = 15.0
    cooldown_seconds: float = 60.0

    # actions
    auto_pause: bool = False

    # server
    server_host: str = "127.0.0.1"
    server_port: int = 8710
    server_auth_token: str = ""

    # events
    events_dir: str = "events"
    ring_buffer_size: int = 30

    # logging
    log_level: str = "INFO"
    log_file: str = "logs/bambu-ai.log"

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        path = Path(path)
        _load_dotenv(path.parent / ".env")
        data = yaml.safe_load(path.read_text()) or {}
        data = _expand(data)
        p = data.get("printer", {})
        c = data.get("camera", {})
        v = data.get("vision", {})
        d = data.get("decision", {})
        a = data.get("actions", {})
        s = data.get("server", {})
        e = data.get("events", {})
        l = data.get("logging", {})
        cfg = cls(
            raw=data,
            printer_host=str(p.get("host", "")),
            printer_serial=str(p.get("serial", "")),
            printer_access_code=str(p.get("access_code", "")),
            printer_mqtt_port=int(p.get("mqtt_port", 8883)),
            printer_tls_ca_file=str(p.get("tls_ca_file", "")),
            printer_tls_insecure=_bool(p.get("tls_insecure", False)),
            camera_provider=str(c.get("provider", "bambu")),
            camera_port=int(c.get("port", 6000)),
            inference_fps=float(c.get("inference_fps", 1.0)),
            camera_file_dir=str(c.get("file_dir", "frames")),
            vision_provider=str(v.get("provider", "onnx")),
            vision_model=str(v.get("model", "models/yolov8n.onnx")),
            vision_backend=str(v.get("backend", "coreml")),
            vision_input_size=int(v.get("input_size", 640)),
            vision_confidence=float(v.get("confidence_threshold", 0.35)),
            remote_base_url=str(v.get("base_url", "")),
            remote_api_key=str(v.get("api_key", "")),
            remote_model_name=str(v.get("model_name", "")),
            pause_threshold=float(d.get("pause_threshold", 0.90)),
            consecutive_frames=int(d.get("consecutive_frames", 3)),
            observation_window_seconds=float(d.get("observation_window_seconds", 15.0)),
            cooldown_seconds=float(d.get("cooldown_seconds", 60.0)),
            auto_pause=_bool(a.get("auto_pause", False)),
            server_host=str(s.get("host", "127.0.0.1")),
            server_port=int(s.get("port", 8710)),
            server_auth_token=str(s.get("auth_token", os.environ.get("BAMBU_AI_AUTH_TOKEN", ""))),
            events_dir=str(e.get("dir", "events")),
            ring_buffer_size=int(e.get("ring_buffer_size", 30)),
            log_level=str(l.get("level", "INFO")),
            log_file=str(l.get("file", "logs/bambu-ai.log")),
        )
        if not 1 <= cfg.printer_mqtt_port <= 65535 or not 1 <= cfg.camera_port <= 65535:
            raise ValueError("printer and camera ports must be between 1 and 65535")
        if not 1 <= cfg.server_port <= 65535:
            raise ValueError("server.port must be between 1 and 65535")
        _finite("camera.inference_fps", cfg.inference_fps, 0.05, 30.0)
        _finite("vision.confidence_threshold", cfg.vision_confidence, 0.0, 1.0)
        _finite("decision.pause_threshold", cfg.pause_threshold, 0.0, 1.0)
        _finite("decision.observation_window_seconds", cfg.observation_window_seconds, 0.1, 86400.0)
        _finite("decision.cooldown_seconds", cfg.cooldown_seconds, 0.0, 86400.0)
        if cfg.consecutive_frames < 1 or cfg.ring_buffer_size < 1 or cfg.vision_input_size < 32:
            raise ValueError("consecutive_frames, ring_buffer_size and input_size must be positive")
        if cfg.server_host not in {"127.0.0.1", "localhost", "::1"} and not cfg.server_auth_token:
            raise ValueError("server.auth_token is required when server.host is not loopback")
        return cfg
