"""Decision engine: threshold, consecutive, cooldown, shadow vs auto."""
from __future__ import annotations

from bambu_ai.decision import DecisionConfig, DecisionEngine
from bambu_ai.models import GuardState, PrinterState
from bambu_ai.temporal import TemporalSignals


def make_engine(auto_pause: bool = False) -> DecisionEngine:
    return DecisionEngine(
        DecisionConfig(
            pause_threshold=0.90,
            consecutive_frames=3,
            observation_window_seconds=15.0,
            cooldown_seconds=60.0,
        ),
        auto_pause=auto_pause,
    )


def test_single_rare_frame_does_not_pause():
    eng = make_engine(auto_pause=True)
    t = 0.0
    # one anomalous frame, then a normal one
    d1 = eng.step(0.95, "spaghetti", PrinterState.PRINTING, now=t)
    assert d1.action != "pause"
    assert eng.state != GuardState.PAUSING
    d2 = eng.step(0.1, "none", PrinterState.PRINTING, now=t + 1)
    assert d2.action == "none"
    assert eng.state == GuardState.NORMAL


def test_consistent_anomaly_confirms_failure_shadow():
    eng = make_engine(auto_pause=False)
    t = 0.0
    actions = []
    for i in range(5):
        d = eng.step(0.96, "spaghetti", PrinterState.PRINTING, now=t + i)
        actions.append(d.action)
    assert "would_pause" in actions
    assert "pause" not in actions  # shadow never actually pauses
    assert eng.state in (GuardState.CONFIRMED_FAILURE, GuardState.PAUSED_BY_AI)


def test_consistent_anomaly_pauses_in_auto():
    eng = make_engine(auto_pause=True)
    t = 0.0
    saw_pause = False
    for i in range(5):
        d = eng.step(0.96, "spaghetti", PrinterState.PRINTING, now=t + i)
        if d.action == "pause":
            saw_pause = True
    assert saw_pause


def test_below_threshold_never_confirms():
    eng = make_engine(auto_pause=True)
    t = 0.0
    for i in range(20):
        d = eng.step(0.5, "blob", PrinterState.PRINTING, now=t + i)
        assert d.action != "pause"


def test_cooldown_prevents_immediate_repause():
    eng = make_engine(auto_pause=True)
    t = 0.0
    for i in range(5):
        eng.step(0.96, "spaghetti", PrinterState.PRINTING, now=t + i)
    assert eng.state == GuardState.PAUSING
    # immediately after, still anomalous but within cooldown -> no new pause
    d = eng.step(0.96, "spaghetti", PrinterState.PRINTING, now=t + 6)
    assert d.action == "none"


def test_not_printing_does_not_pause():
    eng = make_engine(auto_pause=True)
    t = 0.0
    for i in range(5):
        d = eng.step(0.96, "spaghetti", PrinterState.IDLE, now=t + i)
        assert d.action != "pause"


def test_combined_risk_uses_temporal():
    eng = make_engine()
    temporal = TemporalSignals(object_displacement=0.8, collapse=0.0, air_printing=0.0)
    risk, ftype = eng.combined_risk({"object": 0.9, "spaghetti": 0.0}, temporal)
    assert ftype == "object_displacement"
    assert risk >= 0.7
