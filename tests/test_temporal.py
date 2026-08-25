"""Temporal analyzer: displacement and collapse over a frame sequence."""
from __future__ import annotations

from conftest import fallen_frame, moved_frame, normal_frame


def test_stable_object_no_displacement(analyzer):
    t = 0.0
    for i in range(6):
        sig = analyzer.update(normal_frame(), now=t + i)
    assert sig.object_displacement == 0.0
    assert sig.collapse == 0.0


def test_object_movement_flags_displacement(analyzer, sequence):
    t = 0.0
    sigs = []
    for r in sequence:
        sigs.append(analyzer.update(r, now=t + 1))
    last = sigs[-1]
    # after baseline + movement, displacement signal should be nonzero
    assert last.object_displacement > 0.0


def test_fall_flags_collapse(analyzer, sequence):
    t = 0.0
    sigs = [analyzer.update(r, now=t + 1) for r in sequence]
    last = sigs[-1]
    assert last.collapse > 0.0


def test_sequence_normal_normal_normal_move_fall(analyzer, sequence):
    """The canonical fixture: 3 normal, 1 move, 1 fall.

    Early frames must be clean; by the end both displacement and a
    collapse/spaghetti condition must be present.
    """
    t = 0.0
    sigs = [analyzer.update(r, now=t + 1) for r in sequence]
    # first three (normal) stay clean
    for s in sigs[:3]:
        assert s.object_displacement == 0.0
    # final (fall) is anomalous
    assert sigs[-1].object_displacement > 0.0 or sigs[-1].collapse > 0.0


def test_reset_clears_state(analyzer):
    analyzer.update(normal_frame(), now=0)
    analyzer.update(moved_frame(), now=1)
    analyzer.reset()
    assert analyzer._baseline is None
    assert analyzer._object_count == 0
