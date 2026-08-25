"""Remote OpenAI-compatible vision provider (fallback / heavy-model option).

NOT intended for frame-by-frame use on a budget Mac — keep it for occasional
verification or for servers with GPUs.
"""
from __future__ import annotations

import base64
import json
import time

from ..models import Detection, DetectionContext, DetectionResult, Frame
from .base import VisionModel


class RemoteOpenAIModel(VisionModel):
    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str = "",
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key
        self.timeout = timeout
        self.name = f"remote:{self.model_name}"

    async def analyze(self, frame: Frame, context: DetectionContext) -> DetectionResult:
        import httpx

        b64 = base64.b64encode(frame.data).decode()
        payload = {
            "model": self.model_name,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        "Analyze this 3D printer camera frame for print failures. "
                        "Return ONLY JSON: {\"spaghetti\":0-1,\"blob\":0-1,"
                        "\"adhesion_loss\":0-1,\"collapse\":0-1,"
                        "\"air_printing\":0-1,\"objects\":[{\"bbox\":[x,y,w,h]}]}."
                    )},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }],
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions", json=payload, headers=headers
            )
            resp.raise_for_status()
            data = resp.json()
        text = data["choices"][0]["message"]["content"]
        signals = self._parse(text)
        dets = [
            Detection(label="object", confidence=0.5, bbox=ob.get("bbox"))
            for ob in signals.pop("objects", [])
        ]
        return DetectionResult(
            detections=dets,
            signal_scores=signals,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )

    @staticmethod
    def _parse(text: str) -> dict[str, float]:
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            obj = json.loads(text[start:end])
            return {
                k: float(obj.get(k, 0.0)) for k in
                ("spaghetti", "blob", "adhesion_loss", "collapse", "air_printing")
            } | {"objects": obj.get("objects", [])}
        except (ValueError, json.JSONDecodeError):
            return {
                "spaghetti": 0.0, "blob": 0.0, "adhesion_loss": 0.0,
                "collapse": 0.0, "air_printing": 0.0, "objects": [],
            }
