from copy import deepcopy
import pytest
from last_mile.workspace import Workspace, metrics


@pytest.fixture
def ws():
    w = Workspace(
        [{"ds": "2026-01-01", "y": 42}], ["2026-01-02", "2026-01-03"], [100, 200]
    )
    w.add_evidence("history", "known history", [{"ds": "2026-01-01", "y": 42}])
    return w


def proposal(ws, **kwargs):
    return {
        "event": "test",
        "start_date": "2026-01-02",
        "end_date": "2026-01-03",
        "mode": "multiply",
        "value": 0.5,
        "reason": "historical evidence",
        "confidence": 0.7,
        "evidence_ids": [ws.evidence[0]["id"]],
    } | kwargs


def test_immutable_history_and_baseline(ws):
    before = ws.snapshot()
    ws.apply(proposal(ws))
    assert ws.snapshot()["history"] == before["history"]
    assert [r["y_baseline"] for r in ws.snapshot()["forecast"]] == [100, 200]
    assert [r["y_final"] for r in ws.snapshot()["forecast"]] == [50, 100]
    external = ws.snapshot()
    external["forecast"][0]["y_baseline"] = -1
    assert ws.snapshot()["forecast"][0]["y_baseline"] == 100


@pytest.mark.parametrize(
    "changes",
    [
        {"start_date": "2026-01-01"},
        {"end_date": "2026-01-04"},
        {"evidence_ids": ["invented"]},
        {"confidence": float("nan")},
        {"value": float("inf")},
        {"value": -1},
        {"reason": ""},
        {"mode": "override", "values": {"2026-01-02": 30}},
        {"mode": "clip", "value": [50, 20]},
    ],
)
def test_invalid_revisions_are_atomic(ws, changes):
    before = ws.snapshot()
    with pytest.raises(ValueError):
        ws.apply(proposal(ws, **changes))
    assert ws.snapshot() == before


def test_duplicate_rejected_distinct_evidence_allowed(ws):
    ws.apply(proposal(ws))
    with pytest.raises(ValueError):
        ws.apply(proposal(ws))
    new = ws.add_evidence("user", "new campaign", {})
    ws.apply(proposal(ws, evidence_ids=[new["id"]]))
    assert ws.snapshot()["forecast"][0]["y_final"] == 25


def test_batch_rollback(ws):
    before = ws.snapshot()
    with pytest.raises(ValueError):
        ws.apply_batch([proposal(ws), proposal(ws, end_date="2026-01-04")])
    assert ws.snapshot() == before


def test_scope_and_undo(ws):
    with pytest.raises(ValueError):
        ws.apply_batch([proposal(ws)], scope=("2026-01-02", "2026-01-02"))
    ws.apply(proposal(ws))
    ws.undo()
    assert [r["y_final"] for r in ws.snapshot()["forecast"]] == [100, 200]
    assert ws.actions[0]["undone"]
    assert ws.actions[-1]["mode"] == "undo"
    with pytest.raises(ValueError):
        ws.undo()
    ws.apply(proposal(ws))


def test_mape_zero_explicit():
    m = metrics([0, 100], [20, 80])
    assert (
        m["mae"] == 20
        and m["mape"] == 20
        and m["mape_n"] == 1
        and m["zero_actuals"] == 1
    )
    assert metrics([0], [10])["mape"] is None
