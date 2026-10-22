import json
from copy import deepcopy

import pytest

from last_mile.data import synthetic, default_calendar, analog_mapping, parse_series
from last_mile.experiment import Experiment
from last_mile.memory import ExperienceBank


def test_memory_progressive_disclosure_and_crud():
    bank = ExperienceBank()
    entry = {
        "description": "Holiday context",
        "observation": "Observed a trough",
        "experience": "Check holiday offsets",
    }
    bank.upsert("Spring Festival effect", entry)
    assert bank.directory() == [
        {"title": "Spring Festival effect", "description": "Holiday context"}
    ]
    assert bank.directory("holiday") == bank.directory()
    assert bank.directory("unrelated") == []
    exposed = bank.read("Spring Festival effect")
    exposed["experience"] = "Changed copy"
    assert bank.read("Spring Festival effect") == entry
    bank.upsert(
        "Spring Festival effect", entry | {"experience": "Use conditional offsets"}
    )
    assert bank.changes[-1]["operation"] == "update"
    bank.delete("Spring Festival effect")
    assert bank.snapshot() == {}
    with pytest.raises(ValueError):
        bank.upsert("Invalid", {"description": "Missing fields"})


def test_daily_cumulative_reflection_and_next_forecast_only(tmp_path):
    rows = synthetic()["rows"]
    exp = Experiment(tmp_path, calendar=default_calendar())
    run = exp.predict([r for r in rows if r["ds"] < "2026-02-15"], "2026-02-15", 2)
    frozen = deepcopy(run)
    assert run["memory_snapshot"] == {}
    truth = {r["ds"]: r["y"] for r in rows}
    exp.observe(run["id"], "2026-02-15", truth["2026-02-15"])
    assert (
        "Observed 1 days"
        in exp.state["experiences"]["Recent forecast error"]["observation"]
    )
    with pytest.raises(ValueError):
        exp.observe(run["id"], "2026-02-15", 0)
    exp.observe(run["id"], "2026-02-16", truth["2026-02-16"])
    assert (
        "Observed 2 days"
        in exp.state["experiences"]["Recent forecast error"]["observation"]
    )
    for key in ["workspace", "trajectory", "memory_snapshot"]:
        assert exp.state["runs"][0][key] == frozen[key]
    next_run = exp.predict([r for r in rows if r["ds"] < "2026-02-17"], "2026-02-17", 2)
    assert next_run["memory_cutoff"] == "2026-02-16"
    assert next_run["memory_snapshot"]
    evidence = next_run["workspace"]["evidence"]
    directory = next(e for e in evidence if e["source"] == "memory_directory")
    assert set(directory["payload"][0]) == {"title", "description"}
    assert any(e["source"] == "memory" for e in evidence)
    persisted = json.loads((tmp_path / "experiences.json").read_text())
    assert persisted == exp.state["experiences"]
    assert Experiment(tmp_path).state == exp.state
    assert Experiment(tmp_path / "independent").state["experiences"] == {}


def test_reflection_failure_does_not_commit_actuals_or_memory(tmp_path, monkeypatch):
    rows = synthetic()["rows"]
    exp = Experiment(tmp_path)
    run = exp.predict([r for r in rows if r["ds"] < "2026-02-15"], "2026-02-15", 2)
    before = deepcopy(exp.state)

    def fail(session):
        session.bank.upsert(
            "Staged",
            {
                "description": "Example",
                "observation": "Example",
                "experience": "Example",
            },
        )
        raise ValueError("Simulated model failure")

    monkeypatch.setattr("last_mile.experiment.ReflectionSession.run_session", fail)
    with pytest.raises(ValueError):
        exp.observe(run["id"], "2026-02-15", 100)
    assert exp.state == before
    assert Experiment(tmp_path).state == before
    assert list(tmp_path.glob("failed-reflection-*.json"))


def test_observation_order_and_history_validation(tmp_path):
    rows = synthetic()["rows"]
    exp = Experiment(tmp_path)
    history = [r for r in rows if r["ds"] < "2026-02-15"]
    with pytest.raises(ValueError):
        exp.predict(history[:-1], "2026-02-15", 2)
    with pytest.raises(ValueError):
        exp.predict(rows, "2026-02-15", 2)
    run = exp.predict(history, "2026-02-15", 2)
    for ds, value in [
        ("2026-02-16", 1),
        ("2026-02-15", float("nan")),
        ("2026-02-15", -1),
    ]:
        with pytest.raises(ValueError):
            exp.observe(run["id"], ds, value)
    assert exp.state["runs"][0]["actuals"] == []


def test_synthetic_and_calendar():
    assert synthetic()["rows"] == synthetic()["rows"]
    assert len(synthetic()["rows"]) == 856
    event = next(
        e
        for e in default_calendar()
        if e["event"] == "spring_festival" and e["start_date"].startswith("2026")
    )
    mapping = analog_mapping(event, event["start_date"], event["end_date"])
    assert len(mapping) == 9
    assert mapping[0] == {"ds": "2026-02-15", "reference_ds": "2025-01-27"}


@pytest.mark.parametrize(
    "text",
    ["ds,y\n2026-01-01,nan", "ds,y\n2026-01-01,-1", "ds,y\n2026-01-01,1\n2026-01-03,2"],
)
def test_invalid_series(text):
    with pytest.raises(ValueError):
        parse_series(text, minimum=1)
