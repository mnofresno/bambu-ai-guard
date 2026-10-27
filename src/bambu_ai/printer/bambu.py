"""Bambu Lab A1 printer controller over LAN MQTT.

Protocol (same as pybambu / bambu-connect):
  - TLS MQTT on the printer's own broker, port 8883, self-signed cert.
  - username "bblp", password = printer access code.
  - subscribe:  device/{serial}/report   (printer pushes state)
  - publish:    device/{serial}/request  (commands: pause/resume/...)
"""
from __future__ import annotations

import asyncio
import json
import logging
import ssl
import threading
from typing import Any

import paho.mqtt.client as mqtt

from ..models import PrinterState, PrinterStatus
from .base import PrinterController

log = logging.getLogger(__name__)

_PAUSED_STATES = {"PAUSE", "PAUSED", "FINISH_PAUSE"}
_RUNNING_STATES = {"RUNNING", "PREHEATING", "PRINT_END_PAUSE"}


class BambuPrinter(PrinterController):
    def __init__(self, host: str, serial: str, access_code: str, port: int = 8883,
                 ca_file: str = "", tls_insecure: bool = False):
        self.host = host
        self.serial = serial
        self.access_code = access_code
        self.port = port
        self.ca_file = ca_file
        self.tls_insecure = tls_insecure
        self._connected = False
        self._report: dict[str, Any] = {}
        self._client: mqtt.MQTTClient | None = None
        self._thread: threading.Thread | None = None
        self._pause_reason: str = ""
        self._lock = threading.Lock()

    @property
    def is_connected(self) -> bool:
        return self._connected

    # -- lifecycle ----------------------------------------------------------

    def _on_connect(self, client: mqtt.MQTTClient, _u: Any, _f: Any, rc: int) -> None:
        if rc == 0:
            client.subscribe(f"device/{self.serial}/report")
            client.subscribe(f"device/{self.serial}/stat")
            self._connected = True
            log.info("printer mqtt connected: %s:%s serial=%s", self.host, self.port, self.serial)
        else:
            self._connected = False

    def _on_disconnect(self, _c: mqtt.MQTTClient, _u: Any, _rc: int) -> None:
        self._connected = False

    def _on_message(self, _c: mqtt.MQTTClient, _u: Any, msg: mqtt.MQTTMessage) -> None:
        try:
            data = json.loads(msg.payload.decode())
        except (UnicodeDecodeError, json.JSONDecodeError):
            return
        if msg.topic.endswith("/report"):
            with self._lock:
                self._report.update(data)
        else:
            with self._lock:
                self._report.update(data)

    async def connect(self) -> None:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
        client.username_pw_set("bblp", self.access_code)
        if self.tls_insecure:
            client.tls_set(tls_version=ssl.PROTOCOL_TLS, cert_reqs=ssl.CERT_NONE)
            client.tls_insecure_set(True)
        elif self.ca_file:
            client.tls_set(ca_certs=self.ca_file, tls_version=ssl.PROTOCOL_TLS)
        else:
            raise ssl.SSLError("MQTT TLS requires ca_file or explicit tls_insecure=true")
        client.on_connect = self._on_connect
        client.on_disconnect = self._on_disconnect
        client.on_message = self._on_message
        self._client = client
        await asyncio.to_thread(client.connect, self.host, self.port, 60)
        self._thread = threading.Thread(target=client.loop_forever, daemon=True)
        self._thread.start()
        # wait briefly for the first report
        for _ in range(20):
            if self._report:
                break
            await asyncio.sleep(0.25)
        if not self._connected:
            raise ConnectionError("MQTT connection did not become ready")

    async def close(self) -> None:
        self._connected = False
        if self._client:
            await asyncio.to_thread(self._client.loop_stop)
            await asyncio.to_thread(self._client.disconnect)
            self._client = None
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        self._thread = None

    # -- status --------------------------------------------------------------

    def _status(self) -> PrinterStatus:
        with self._lock:
            r = json.loads(json.dumps(self._report))
        gcode_state = str(r.get("print", {}).get("gcode_state", ""))
        if gcode_state in _RUNNING_STATES:
            state = PrinterState.PRINTING
        elif gcode_state in _PAUSED_STATES:
            state = PrinterState.PAUSED
        elif gcode_state in {"FINISH", "FINISHED"}:
            state = PrinterState.FINISHED
        elif gcode_state == "ERROR":
            state = PrinterState.ERROR
        elif gcode_state in {"", "COMPLETE", "INIT"}:
            state = PrinterState.IDLE
        else:
            state = PrinterState.UNKNOWN
        job = r.get("print", {})
        return PrinterStatus(
            state=state,
            job_name=str(job.get("subtask_name", "")),
            progress_pct=float(job.get("print_percent", 0) or 0),
            elapsed_seconds=float(job.get("remain_time", 0) or 0),
        )

    async def get_status(self) -> PrinterStatus:
        return self._status()

    # -- commands -------------------------------------------------------------

    def _publish(self, payload: dict) -> None:
        if not self._client:
            raise RuntimeError("printer not connected")
        info = self._client.publish(f"device/{self.serial}/request", json.dumps(payload), qos=1)
        if info.rc != mqtt.MQTT_ERR_SUCCESS:
            raise ConnectionError(f"MQTT publish failed: {info.rc}")

    async def pause(self, reason: str) -> None:
        self._pause_reason = reason
        log.info("printer pause requested: %s", reason)
        await asyncio.to_thread(
            self._publish, {"print": {"sequence_id": "0", "command": "pause"}}
        )

    async def resume(self) -> None:
        log.info("printer resume requested")
        await asyncio.to_thread(
            self._publish, {"print": {"sequence_id": "0", "command": "resume"}}
        )
