import json
import pytest

smolagents = pytest.importorskip("smolagents")
from smolagents import Model, ChatMessage
from last_mile.agents import RevisionSession, make_model
from last_mile.workspace import Workspace


class ScriptedModel(Model):
    """Exercise the actual CodeAgent executor without external API credentials."""

    def __init__(self, code):
        super().__init__(model_id="test-script")
        self.code = code

    def generate(self, messages, **kwargs):
        return ChatMessage(
            role="assistant", content="<code>\n" + self.code + "\n</code>"
        )


def test_codeagent_tools_execute_validated_revision():
    ws = Workspace([{"ds": "2026-01-01", "y": 50}], ["2026-01-02"], [100])
    code = """import json
e = json.loads(retrieve_history_tool("2026-01-01", "2026-01-01"))
p = {"event":"test","start_date":"2026-01-02","end_date":"2026-01-02","mode":"multiply","value":0.5,"reason":"Observed analogue", "confidence":0.7,"evidence_ids":[e["id"]]}
apply_json_policies(json.dumps([p]))
final_answer("Completed")"""
    session = RevisionSession(
        ws,
        [],
        [],
        {"mode": "live", "provider": "openai", "max_steps": 2},
        lambda *x: None,
        lambda: False,
        model_factory=lambda: ScriptedModel(code),
    )
    assert session.run() == "Completed"
    assert ws.snapshot()["forecast"][0]["y_final"] == 50


def test_local_reasoners_are_read_only_and_scoped():
    ws = Workspace(
        [{"ds": "2026-01-01", "y": 50}], ["2026-01-02", "2026-01-03"], [100, 100]
    )
    session = RevisionSession(
        ws,
        [],
        [],
        {"mode": "live", "provider": "openai"},
        lambda *x: None,
        lambda: False,
    )
    evidence = session.history("2026-01-01", "2026-01-01")
    sink = []
    tools = session.tools(("2026-01-02", "2026-01-02"), sink)
    names = {t.name for t in tools}
    assert "apply_json_policies" not in names
    writer = next(t for t in tools if t.name == "write_signal_envelope")
    p = {
        "event": "test",
        "start_date": "2026-01-02",
        "end_date": "2026-01-03",
        "mode": "multiply",
        "value": 0.5,
        "reason": "history",
        "confidence": 0.7,
        "evidence_ids": [evidence["id"]],
    }
    with pytest.raises(ValueError):
        writer(json.dumps([p]))
    p["end_date"] = "2026-01-02"
    writer(json.dumps([p]))
    assert ws.snapshot()["forecast"][0]["y_final"] == 100
    with pytest.raises(ValueError):
        writer("[]")


@pytest.mark.parametrize("provider", ["openai", "anthropic", "deepseek"])
def test_provider_routing(monkeypatch, provider):
    seen = {}
    monkeypatch.setattr(
        smolagents, "LiteLLMModel", lambda **kwargs: seen.update(kwargs) or kwargs
    )
    model = make_model(
        {
            "provider": provider,
            "model": "chosen-model",
            "base_url": "https://gateway.example/v1",
        },
        {provider: "test-only"},
    )
    assert model["model_id"] == f"{provider}/chosen-model"
    assert model["api_base"] == "https://gateway.example/v1"
    assert model["timeout"] == 90


def test_map_reduce_runs_real_codeagents_with_read_only_plans():
    ws = Workspace(
        [{"ds": "2026-01-01", "y": 50}], ["2026-01-02", "2026-01-03"], [100, 100]
    )
    events = [
        {
            "event": f"event{i}",
            "name": f"Event {i}",
            "start_date": f"2026-01-0{i}",
            "end_date": f"2026-01-0{i}",
            "anchor_date": f"2026-01-0{i}",
            "reference_anchor": "2026-01-01",
            "source": "test",
        }
        for i in [2, 3]
    ]
    created = []

    def factory():
        index = len(created)
        created.append(index)
        if index == 0:
            return ScriptedModel(
                'plans = run_map_reduce_planners()\napply_json_policies(plans)\nfinal_answer("merged")'
            )
        ds = f"2026-01-0{index + 1}"
        return ScriptedModel(f'''import json
e=json.loads(retrieve_history_tool("2026-01-01","2026-01-01"))
p={{"event":"local", "start_date":"{ds}", "end_date":"{ds}", "mode":"multiply", "value":0.5, "reason":"Historical evidence", "confidence":0.7,"evidence_ids":[e["id"]]}}
write_signal_envelope(json.dumps([p]))
final_answer("planned")''')

    session = RevisionSession(
        ws,
        events,
        [],
        {"mode": "live", "provider": "openai", "max_steps": 3},
        lambda *x: None,
        lambda: False,
        model_factory=factory,
    )
    assert session.run() == "merged"
    assert len(created) == 3
    assert [r["y_final"] for r in ws.snapshot()["forecast"]] == [50, 50]
