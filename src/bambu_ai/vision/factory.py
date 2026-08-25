"""Build a VisionModel from config. Provider is swappable without code changes."""
from __future__ import annotations

import json
from pathlib import Path

from ..config import Config
from .base import VisionModel
from .mock import MockModel
from .onnx import OnnxYoloModel
from .remote import RemoteOpenAIModel


def _label_map(path: str | None) -> dict[str, str] | None:
    if path:
        p = Path(path)
        if p.exists():
            return json.loads(p.read_text())
    return None


def build_vision_model(cfg: Config) -> VisionModel:
    provider = cfg.vision_provider.lower()
    if provider == "mock":
        return MockModel()
    if provider == "onnx":
        return OnnxYoloModel(
            model_path=cfg.vision_model,
            backend=cfg.vision_backend,
            input_size=cfg.vision_input_size,
            confidence=cfg.vision_confidence,
            label_map=_label_map(cfg.raw.get("vision", {}).get("label_map")),
        )
    if provider in ("remote_openai_compatible", "remote"):
        return RemoteOpenAIModel(
            base_url=cfg.remote_base_url,
            model_name=cfg.remote_model_name,
            api_key=cfg.remote_api_key,
        )
    raise ValueError(f"unknown vision provider: {provider}")
