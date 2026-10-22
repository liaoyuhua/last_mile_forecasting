"""Immutable baseline/history, transactional revisions, auditable evidence."""

from copy import deepcopy
import math

from .data import dates
from .utils import now, uid


def finite(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError("Revision parameters must be finite numbers")
    return float(value)


class Workspace:
    def __init__(self, history, forecast_dates, baseline):
        if len(baseline) != len(forecast_dates) or not forecast_dates:
            raise ValueError("Baseline length must match the forecast window")
        if dates(forecast_dates[0], forecast_dates[-1]) != forecast_dates:
            raise ValueError("Forecast dates must be continuous and unique")
        if history and max(r["ds"] for r in history) >= forecast_dates[0]:
            raise ValueError("History must precede the forecast window")
        self._history = deepcopy(history)
        self._baseline = {ds: finite(y) for ds, y in zip(forecast_dates, baseline)}
        if any(y < 0 for y in self._baseline.values()):
            raise ValueError("Demand forecasts must be nonnegative")
        self._final = dict(self._baseline)
        self.evidence = []
        self.actions = []

    def snapshot(self):
        return deepcopy(
            {
                "history": self._history,
                "forecast": [
                    {"ds": ds, "y_baseline": y, "y_final": self._final[ds]}
                    for ds, y in self._baseline.items()
                ],
                "evidence": self.evidence,
                "actions": self.actions,
            }
        )

    @classmethod
    def restore(cls, snapshot):
        ws = cls(
            snapshot["history"],
            [r["ds"] for r in snapshot["forecast"]],
            [r["y_baseline"] for r in snapshot["forecast"]],
        )
        ws._final = {r["ds"]: r["y_final"] for r in snapshot["forecast"]}
        ws.evidence = deepcopy(snapshot["evidence"])
        ws.actions = deepcopy(snapshot["actions"])
        return ws

    def add_evidence(self, source, description, payload):
        item = {
            "id": uid(),
            "source": source,
            "description": description,
            "payload": deepcopy(payload),
            "created_at": now(),
        }
        self.evidence.append(item)
        return deepcopy(item)

    def apply(self, proposal, actor="agent"):
        if not isinstance(proposal, dict):
            raise ValueError("A revision must be a JSON object")
        p = deepcopy(proposal)
        mode = p.get("mode")
        if mode not in {"multiply", "add", "clip", "override", "none"}:
            raise ValueError("Unsupported revision operator")
        refs = p.get("evidence_ids", [])
        if (
            not isinstance(refs, list)
            or not refs
            or not all(isinstance(x, str) for x in refs)
            or not set(refs) <= {e["id"] for e in self.evidence}
        ):
            raise ValueError("Every revision must cite retrieved evidence IDs")
        confidence = finite(p.get("confidence"))
        if (
            not 0 <= confidence <= 1
            or not isinstance(p.get("reason"), str)
            or not p["reason"].strip()
        ):
            raise ValueError("A reason and confidence between 0 and 1 are required")
        affected = dates(p.get("start_date"), p.get("end_date"))
        if not set(affected) <= set(self._baseline):
            raise ValueError("Revisions must stay within the forecast window")
        if mode == "override":
            points = p.get("values")
            if not isinstance(points, dict) or set(points) != set(affected):
                raise ValueError(
                    "Overrides must provide a complete date-to-value mapping"
                )
        before = {ds: self._final[ds] for ds in affected}
        after = dict(before)
        for old in self.actions:
            if old.get("undone") or old["mode"] in {"undo", "none"}:
                continue
            overlap = set(old["after"]) & set(affected)
            if overlap and set(old["evidence_ids"]) == set(refs) and mode != "none":
                raise ValueError(
                    "Overlapping revisions cannot reuse the same evidence set; undo before retrying"
                )
        for ds, y in before.items():
            if mode == "multiply":
                after[ds] = y * finite(p.get("value"))
            elif mode == "add":
                after[ds] = y + finite(p.get("value"))
            elif mode == "override":
                after[ds] = finite(p["values"][ds])
            elif mode == "clip":
                bounds = p.get("value")
                if not isinstance(bounds, list) or len(bounds) != 2:
                    raise ValueError("clip requires value=[lower, upper]")
                low, high = map(finite, bounds)
                if low > high:
                    raise ValueError("clip lower bound must not exceed upper bound")
                after[ds] = min(high, max(low, y))
            after[ds] = round(finite(after[ds]), 4)
            if after[ds] < 0:
                raise ValueError("Revisions cannot produce negative demand")
        action = {
            "id": uid(),
            "event": str(p.get("event", "Adjustment")),
            "mode": mode,
            "start_date": affected[0],
            "end_date": affected[-1],
            "reason": p["reason"],
            "confidence": confidence,
            "evidence_ids": refs,
            "actor": actor,
            "created_at": now(),
            "before": before,
            "after": after,
            "proposal": p,
            "undone": False,
        }
        self._final.update(after)
        self.actions.append(action)
        return deepcopy(action)

    def apply_batch(self, proposals, actor="agent", scope=None):
        if not isinstance(proposals, list) or len(proposals) > 100:
            raise ValueError("At most 100 revisions per batch")
        staged = Workspace.restore(self.snapshot())
        accepted = []
        for p in proposals:
            if scope and not (
                scope[0] <= p.get("start_date", "") <= p.get("end_date", "") <= scope[1]
            ):
                raise ValueError("Local reasoner proposal exceeds its assigned window")
            accepted.append(staged.apply(p, actor))
        self._final, self.actions = staged._final, staged.actions
        return accepted

    def undo(self):
        candidates = [
            a for a in self.actions if a["mode"] != "undo" and not a.get("undone")
        ]
        if not candidates:
            raise ValueError("No revision available to undo")
        last = candidates[-1]
        self._final.update(last["before"])
        last["undone"] = True
        self.actions.append(
            {
                "id": uid(),
                "mode": "undo",
                "event": last["event"],
                "target_id": last["id"],
                "reason": "Undo the latest revision",
                "actor": "user",
                "created_at": now(),
            }
        )


def metrics(actual, predicted):
    if len(actual) != len(predicted) or not actual:
        raise ValueError("Evaluation requires nonempty aligned observations")
    errors = [abs(a - p) for a, p in zip(actual, predicted)]
    percentages = [
        abs(a - p) / abs(a) * 100 for a, p in zip(actual, predicted) if a != 0
    ]
    return {
        "mae": sum(errors) / len(errors),
        "mape": sum(percentages) / len(percentages) if percentages else None,
        "n": len(actual),
        "mape_n": len(percentages),
        "zero_actuals": len(actual) - len(percentages),
    }


def evaluate(snapshot, actuals, events):
    actual = {r["ds"]: r["y"] for r in actuals}
    forecast = snapshot["forecast"]
    windows = [("full", "Full forecast window", forecast)] + [
        (
            e["event"] + ":" + e["start_date"],
            e["name"],
            [r for r in forecast if e["start_date"] <= r["ds"] <= e["end_date"]],
        )
        for e in events
    ]
    result = []
    for key, name, rows in windows:
        matched = [r for r in rows if r["ds"] in actual]
        if not matched:
            continue
        truth = [actual[r["ds"]] for r in matched]
        base = metrics(truth, [r["y_baseline"] for r in matched])
        final = metrics(truth, [r["y_final"] for r in matched])
        result.append(
            {
                "key": key,
                "name": name,
                "coverage": len(matched),
                "total": len(rows),
                "baseline": base,
                "final": final,
                "improvement": (base["mae"] - final["mae"]) / base["mae"] * 100
                if base["mae"]
                else None,
            }
        )
    return result
