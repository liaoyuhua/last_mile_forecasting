"""Synchronous forecast/observe lifecycle for one isolated experiment sequence."""

from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import json
import os
import tempfile

from .agents import RevisionSession
from .backbones import forecast
from .data import day, dates, parse_series, calendar_events
from .memory import ExperienceBank
from .reflection import ReflectionSession
from .utils import now, uid
from .workspace import Workspace, evaluate


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=".pending-")
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(text + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


class Experiment:
    """Use one directory per independent sequence, and a single writer per directory.

    experiment.json is the atomic authoritative checkpoint; experiences.json is
    the exact title-keyed bank projection, repaired from that checkpoint on open.
    """

    def __init__(self, directory, settings=None, calendar=None, model_factory=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.model_factory = model_factory
        checkpoint = self.directory / "experiment.json"
        if checkpoint.exists():
            self.state = json.loads(checkpoint.read_text())
            if settings is not None and settings != self.state["settings"]:
                raise ValueError(
                    "Existing experiment has different settings; use a new directory"
                )
            if calendar is not None and calendar != self.state["calendar"]:
                raise ValueError(
                    "Existing experiment has a different calendar; use a new directory"
                )
        else:
            self.state = {
                "schema_version": 1,
                "settings": settings
                or {
                    "mode": "demo",
                    "backbone": "seasonal",
                    "provider": "anthropic",
                    "model": "",
                    "context_length": 1024,
                    "max_steps": 12,
                },
                "calendar": calendar or [],
                "experiences": {},
                "observed_through": None,
                "runs": [],
                "reflections": [],
            }
        self._persist()

    def _persist(self):
        write_json(self.directory / "experiment.json", self.state)
        write_json(self.directory / "experiences.json", self.state["experiences"])

    def predict(self, history, start, horizon, context=""):
        day(start)
        if (
            isinstance(horizon, bool)
            or not isinstance(horizon, int)
            or not 1 <= horizon <= 365
        ):
            raise ValueError("Horizon must be an integer from 1 to 365")
        # This interface accepts history only. Callers must keep held-out data separate.
        parse_series("ds,y\n" + "\n".join(f"{r['ds']},{r['y']}" for r in history))
        if history != sorted(history, key=lambda r: r["ds"]):
            raise ValueError("History must be sorted by date")
        if history[-1]["ds"] != (day(start) - timedelta(days=1)).isoformat():
            raise ValueError("History must end exactly one day before forecast start")
        if self.state["observed_through"] and self.state["observed_through"] >= start:
            raise ValueError("Experience cutoff must precede forecast start")
        if self.state["runs"] and start <= self.state["runs"][-1]["end"]:
            raise ValueError(
                "Sequence forecast windows must not overlap; use a separate experiment"
            )
        historical_values = {row["ds"]: row["y"] for row in history}
        for previous in self.state["runs"]:
            for actual in previous["actuals"]:
                if historical_values.get(actual["ds"]) != actual["y"]:
                    raise ValueError(
                        "History must retain previously observed actuals in this sequence"
                    )
        settings = deepcopy(self.state["settings"])
        target = dates(start, (day(start) + timedelta(days=horizon - 1)).isoformat())
        baseline = forecast(
            history, target, settings["backbone"], settings.get("context_length", 1024)
        )
        ws = Workspace(history, target, baseline)
        activity = []
        run = {
            "id": uid(),
            "created_at": now(),
            "start": start,
            "end": target[-1],
            "settings": settings,
            "context": context,
            "memory_snapshot": deepcopy(self.state["experiences"]),
            "memory_cutoff": self.state["observed_through"],
            "actuals": [],
        }
        session = RevisionSession(
            ws,
            self.state["calendar"],
            run["memory_snapshot"],
            settings,
            lambda kind, message, detail=None: activity.append(
                {"kind": kind, "message": message, "detail": detail}
            ),
            lambda: False,
            model_factory=self.model_factory,
        )
        try:
            run["summary"] = session.run(context)
        except Exception:
            write_json(
                self.directory / f"failed-forecast-{run['id']}.json",
                {
                    "workspace": ws.snapshot(),
                    "trajectory": session.trajectory,
                    "tool_trace": session.tool_trace,
                },
            )
            raise
        run.update(
            workspace=ws.snapshot(),
            trajectory=session.trajectory,
            tool_trace=session.tool_trace,
            activity=activity,
        )
        self.state["runs"].append(run)
        self._persist()
        return deepcopy(run)

    def observe(self, run_id, ds, value):
        run = next((r for r in self.state["runs"] if r["id"] == run_id), None)
        if run is None:
            raise KeyError("Unknown forecast run")
        expected = (day(run["start"]) + timedelta(days=len(run["actuals"]))).isoformat()
        if ds != expected or ds > run["end"]:
            raise ValueError("Actuals must arrive once per day in forecast order")
        if self.state["observed_through"] and ds <= self.state["observed_through"]:
            raise ValueError("Observation dates must advance across the sequence")
        if self.state["runs"][-1]["id"] != run_id:
            raise ValueError(
                "Cannot update an earlier run after the next forecast has started"
            )
        actual = parse_series(f"ds,y\n{ds},{value}", minimum=1)[0]
        actuals = run["actuals"] + [actual]
        bank = ExperienceBank(self.state["experiences"])
        session = ReflectionSession(
            run, actuals, bank, self.state["settings"], self.model_factory
        )
        try:
            summary = session.run_session()
        except Exception:
            write_json(
                self.directory / f"failed-reflection-{run_id}-{ds}.json",
                {
                    "trajectory": session.trajectory,
                    "tool_trace": session.tool_trace,
                    "notice": "No actuals or experience edits committed; retry this date.",
                },
            )
            raise
        # The forecast, original trajectory and its experience snapshot are immutable.
        run["actuals"] = actuals
        run["metrics"] = evaluate(
            run["workspace"],
            actuals,
            calendar_events(self.state["calendar"], run["start"], run["end"]),
        )
        self.state["experiences"] = session.bank.snapshot()
        self.state["observed_through"] = ds
        record = {
            "run_id": run_id,
            "observed_through": ds,
            "summary": summary,
            "changes": session.bank.changes,
            "trajectory": session.trajectory,
            "tool_trace": session.tool_trace,
            "created_at": now(),
        }
        self.state["reflections"].append(record)
        self._persist()
        return deepcopy(record)
