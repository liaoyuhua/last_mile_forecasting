import json
from copy import deepcopy

import pytest

pytest.importorskip("smolagents")
from smolagents import ChatMessage, Model
from last_mile.data import synthetic, default_calendar
from last_mile.experiment import Experiment
from last_mile.memory import ExperienceBank
from last_mile.reflection import ReflectionSession


class ScriptedModel(Model):
    def __init__(self, code):
        super().__init__(model_id="test-script")
        self.code = code

    def generate(self, messages, **kwargs):
        return ChatMessage(
            role="assistant", content="<code>\n" + self.code + "\n</code>"
        )


def fixture_run(tmp_path):
    rows = synthetic()["rows"]
    exp = Experiment(tmp_path, calendar=default_calendar())
    return exp.predict([r for r in rows if r["ds"] < "2026-02-15"], "2026-02-15", 7)


def test_independent_live_reflection_crud_and_trajectory(tmp_path):
    run = fixture_run(tmp_path)
    frozen = deepcopy(run)
    bank = ExperienceBank(
        {"Obsolete": {"description": "Old", "observation": "Old", "experience": "Old"}}
    )
    code = """import json
outcomes = json.loads(inspect_outcomes())
assert len(outcomes["rows"]) == 1
assert outcomes["observed_through"] == "2026-02-15"
catalog = json.loads(list_experiences())
assert "observation" not in catalog[0]
read_experience("Obsolete")
list_trajectory()
read_trajectory("revisions", 0)
upsert_experience("Spring Festival effect", json.dumps({"description":"Holiday demand near the new year", "observation":"Observed a trough on 2026-02-15", "experience":"Check date alignment before adopting historical levels"}))
delete_experience("Obsolete")
final_answer("Updated conditional holiday experience")"""
    session = ReflectionSession(
        run,
        [{"ds": "2026-02-15", "y": 300}],
        bank,
        {"mode": "live", "max_steps": 2},
        lambda: ScriptedModel(code),
    )
    assert session.run_session() == "Updated conditional holiday experience"
    assert set(session.bank.entries) == {"Spring Festival effect"}
    assert "Obsolete" in bank.entries  # staging preserves original until caller commits
    assert run == frozen
    assert any(r["tool"] == "read_trajectory" for r in session.tool_trace)
    saved = json.dumps(session.trajectory)
    assert "model_input_messages" in saved
    assert "code_action" in saved
    assert "Updated conditional holiday experience" in saved


def test_reflection_cannot_write_without_reading_evidence(tmp_path):
    session = ReflectionSession(
        fixture_run(tmp_path),
        [{"ds": "2026-02-15", "y": 300}],
        ExperienceBank(),
        {"mode": "live"},
    )
    tools = {t.name: t for t in session.tools()}
    with pytest.raises(ValueError):
        tools["upsert_experience"](
            "Unsupported",
            json.dumps({"description": "x", "observation": "x", "experience": "x"}),
        )
    outcomes = json.loads(tools["inspect_outcomes"]())
    assert all(r["ds"] <= "2026-02-15" for r in outcomes["rows"])
    assert "2026-02-16" not in json.dumps(outcomes)
    assert not any(
        name in tools for name in ["apply_json_policies", "forecast", "revise"]
    )


def test_live_sequence_reads_reflected_experience_on_next_prediction(tmp_path):
    calls = []
    forecast_code = """import json
ws = json.loads(inspect_workspace())
catalog = json.loads(query_memory_bank())["payload"]
for item in catalog:
    read_experience_tool(item["title"])
last = ws["recent_history"][-1]["ds"]
e = json.loads(retrieve_history_tool(last, last))
ds = ws["forecast"][0]["ds"]
p = {"event":"test", "start_date":ds, "end_date":ds, "mode":"multiply", "value":0.9, "reason":"Test historical evidence", "confidence":0.5, "evidence_ids":[e["id"]]}
apply_json_policies(json.dumps([p]))
final_answer("Forecast completed")"""
    reflection_code = """import json
inspect_outcomes()
read_trajectory("revisions", 0)
upsert_experience("Conditional holiday lesson", json.dumps({"description":"Relevant holiday context", "observation":"One realized day", "experience":"Check analogues cautiously"}))
final_answer("Experience updated")"""

    def factory():
        index = len(calls)
        calls.append(index)
        return ScriptedModel(reflection_code if index == 1 else forecast_code)

    settings = {
        "mode": "live",
        "backbone": "seasonal",
        "provider": "openai",
        "max_steps": 2,
    }
    exp = Experiment(tmp_path, settings=settings, model_factory=factory)
    rows = synthetic()["rows"]
    first = exp.predict([r for r in rows if r["ds"] < "2026-02-15"], "2026-02-15", 1)
    exp.observe(
        first["id"], "2026-02-15", next(r["y"] for r in rows if r["ds"] == "2026-02-15")
    )
    second = exp.predict([r for r in rows if r["ds"] < "2026-02-16"], "2026-02-16", 1)
    assert len(calls) == 3
    assert first["memory_snapshot"] == {}
    assert "Conditional holiday lesson" in second["memory_snapshot"]
    assert any(t["tool"] == "read_experience_tool" for t in second["tool_trace"])
    assert exp.state["runs"][0]["workspace"] == first["workspace"]
