"""Local FastAPI monitoring UI + control endpoints."""
from __future__ import annotations

import base64
import hmac
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from .monitor import Monitor

STATIC_DIR = Path(__file__).parent / "static"


class SettingsUpdate(BaseModel):
    enabled: bool | None = None
    auto_pause: bool | None = None
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    fps: float | None = Field(default=None, ge=0.05, le=30.0)


def create_app(monitor: Monitor) -> FastAPI:
    app = FastAPI(title="Bambu A1 AI Monitor")

    def authorize(token: str | None) -> None:
        expected = monitor.cfg.server_auth_token
        if expected and not token:
            raise HTTPException(401, "authentication required")
        if expected and not hmac.compare_digest(token or "", expected):
            raise HTTPException(403, "invalid authentication token")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (STATIC_DIR / "index.html").read_text()

    @app.get("/api/status")
    def status(x_api_key: str | None = Header(default=None)) -> JSONResponse:
        authorize(x_api_key)
        s = monitor.status
        return JSONResponse({
            "enabled": s.enabled,
            "state": s.state.value,
            "auto_pause": monitor.cfg.auto_pause,
            "inference_fps": monitor.cfg.inference_fps,
            "pause_threshold": monitor.cfg.pause_threshold,
            "model": monitor.vision.name,
            "printer": asdict(s.printer) if s.printer else None,
            "signals": s.last_signals,
            "risk": s.last_risk,
            "failure": s.last_failure,
            "frames": s.frames_processed,
            "confirmed": s.confirmed_failures,
            "pauses": s.pauses,
            "inference_ms": round(s.last_inference_ms, 1),
            "events": s.events,
        })

    @app.get("/api/frame")
    async def frame(x_api_key: str | None = Header(default=None)) -> JSONResponse:
        authorize(x_api_key)
        try:
            f = await monitor.camera.get_frame(timeout=3)
        except Exception:
            raise HTTPException(503, "camera frame unavailable")
        return JSONResponse({"b64": base64.b64encode(f.data).decode(),
                             "ts": f.timestamp})

    @app.post("/api/settings")
    def settings(u: SettingsUpdate, x_api_key: str | None = Header(default=None)) -> JSONResponse:
        authorize(x_api_key)
        if u.enabled is not None:
            monitor.status.enabled = u.enabled
        if u.auto_pause is not None:
            monitor.cfg.auto_pause = u.auto_pause
            monitor.decision.auto_pause = u.auto_pause
        if u.threshold is not None:
            monitor.cfg.pause_threshold = u.threshold
            monitor.decision.cfg.pause_threshold = u.threshold
        if u.fps is not None:
            monitor.cfg.inference_fps = u.fps
        return JSONResponse({"ok": True})

    @app.post("/api/pause")
    async def pause(x_api_key: str | None = Header(default=None)) -> JSONResponse:
        authorize(x_api_key)
        status = await monitor.printer.get_status()
        if status.state.value != "printing":
            raise HTTPException(409, "printer is not actively printing")
        await monitor.printer.pause("manual")
        return JSONResponse({"ok": True})

    @app.post("/api/resume")
    async def resume(x_api_key: str | None = Header(default=None)) -> JSONResponse:
        authorize(x_api_key)
        await monitor.printer.resume()
        return JSONResponse({"ok": True})

    return app


def serve(monitor: Monitor, host: str, port: int) -> None:
    import uvicorn
    app = create_app(monitor)
    uvicorn.run(app, host=host, port=port, log_level="info")
