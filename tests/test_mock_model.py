"""Mock vision model + end-to-end pipeline (mock camera/printer/model)."""
from __future__ import annotations

import pytest

from bambu_ai.models import DetectionContext, PrinterState
from bambu_ai.vision.mock import MockModel
from conftest import make_result


@pytest.mark.asyncio
async def test_mock_model_scripted():
    model = MockModel(factory=lambda i: make_result({"spaghetti": 0.5 if i == 2 else 0.0}))
    ctx = DetectionContext(printer_state=PrinterState.PRINTING, elapsed_seconds=0)
    from bambu_ai.models import Frame
    r0 = await model.analyze(Frame(data=b"x"), ctx)
    r2 = await model.analyze(Frame(data=b"x"), ctx)
    r2 = await model.analyze(Frame(data=b"x"), ctx)
    assert r0.signal_scores["spaghetti"] == 0.0
    assert r2.signal_scores["spaghetti"] == 0.5
    assert model.calls == 3


@pytest.mark.asyncio
async def test_monitor_shadow_end_to_end(sequence, context, tmp_path):
    """Feed a normal->fall sequence through a Monitor in shadow mode:
    it must confirm a failure and record evidence WITHOUT pausing."""
    from bambu_ai.camera.mock import MockCamera
    from bambu_ai.decision import DecisionConfig, DecisionEngine
    from bambu_ai.events import EventRecorder
    from bambu_ai.monitor import Monitor
    from bambu_ai.models import Frame
    from bambu_ai.printer.mock import MockPrinter
    from bambu_ai.temporal import TemporalAnalyzer

    cfg = _cfg(tmp_path)
    cam = MockCamera()
    printer = MockPrinter(state=PrinterState.PRINTING)
    model = MockModel(factory=lambda i: sequence[min(i, len(sequence) - 1)])
    eng = DecisionEngine(
        DecisionConfig(pause_threshold=0.9, consecutive_frames=3,
                       observation_window_seconds=15.0, cooldown_seconds=60.0),
        auto_pause=False,
    )
    rec = EventRecorder(tmp_path / "events", ring_buffer_size=10)
    mon = Monitor(cfg, cam, printer, model, TemporalAnalyzer(), eng, rec)
    await cam.connect()
    await printer.connect()
    t = 0.0
    for i, res in enumerate(sequence * 2):
        frame = Frame(data=b"jpeg", timestamp=t + i)
        await mon._tick()
    assert printer.paused is False  # shadow never pauses
    assert mon.status.confirmed_failures >= 1
    events = list((tmp_path / "events").glob("*/metadata.json"))
    assert len(events) >= 1
    await cam.close()
    await printer.close()


def _cfg(tmp_path):
    from bambu_ai.config import Config
    return Config(
        printer_host="x", printer_serial="x", printer_access_code="x",
        events_dir=str(tmp_path / "events"), auto_pause=False,
    )
