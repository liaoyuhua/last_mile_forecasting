"""Daily series validation, local calendar and clearly labeled synthetic data."""

import csv
import io
import math
import random
from datetime import date, timedelta
from pathlib import Path

from .utils import now, uid

ROOT = Path(__file__).resolve().parent.parent


def day(value):
    if not isinstance(value, str):
        raise ValueError("Dates must use YYYY-MM-DD")
    result = date.fromisoformat(value)
    if result.isoformat() != value:
        raise ValueError("Dates must use YYYY-MM-DD")
    return result


def dates(start, end):
    a, b = day(start), day(end)
    if b < a or (b - a).days > 50000:
        raise ValueError("Invalid or excessive date range")
    return [(a + timedelta(days=i)).isoformat() for i in range((b - a).days + 1)]


def parse_series(text, minimum=28, contiguous=True):
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
    if not {"ds", "y"} <= set(reader.fieldnames or []):
        raise ValueError("CSV must contain ds,y columns")
    values = {}
    for row in reader:
        ds = row["ds"].strip()
        day(ds)
        if ds in values:
            raise ValueError(f"Duplicate date: {ds}")
        try:
            y = float(row["y"])
        except (ValueError, TypeError):
            raise ValueError(f"{ds} requires a numeric y value") from None
        if not math.isfinite(y) or y < 0:
            raise ValueError(f"{ds} requires a finite nonnegative y value")
        values[ds] = y
        if len(values) > 50000:
            raise ValueError("At most 50,000 rows are supported")
    keys = sorted(values)
    if len(keys) < minimum:
        raise ValueError(f"At least {minimum} daily observations are required")
    if contiguous and dates(keys[0], keys[-1]) != keys:
        raise ValueError(
            "Dates must be continuous; missing observations are not filled automatically"
        )
    return [{"ds": ds, "y": values[ds]} for ds in keys]


def parse_calendar(text):
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
    required = {"event", "name", "start_date", "end_date", "anchor_date"}
    if not required <= set(reader.fieldnames or []):
        raise ValueError(
            "Calendar requires event,name,start_date,end_date,anchor_date columns"
        )
    result, seen = [], set()
    for row in reader:
        row = {
            k: (v or "").strip()
            for k, v in row.items()
            if k in required | {"reference_anchor", "source"}
        }
        if not row["event"] or not row["name"]:
            raise ValueError("Event identifiers and names must not be empty")
        period = dates(row["start_date"], row["end_date"])
        day(row["anchor_date"])
        if row["anchor_date"] not in period:
            raise ValueError("anchor_date must fall within the event window")
        if row.get("reference_anchor"):
            day(row["reference_anchor"])
            if row["reference_anchor"] >= row["start_date"]:
                raise ValueError("The reference anchor must precede the event window")
        key = row["event"], row["start_date"]
        if key in seen:
            raise ValueError("Duplicate calendar event")
        seen.add(key)
        row["source"] = row.get("source") or "Local calendar"
        result.append(row)
        if len(result) > 2000:
            raise ValueError("Calendar supports at most 2,000 events")
    if not result:
        raise ValueError("Calendar is empty")
    return sorted(result, key=lambda r: r["start_date"])


def default_calendar():
    return parse_calendar((ROOT / "data" / "holidays.csv").read_text())


def calendar_events(calendar, start, end):
    return [
        dict(e) for e in calendar if e["start_date"] <= end and e["end_date"] >= start
    ]


def analog_mapping(event, start, end):
    """Relative-anchor mapping, including days outside last year's holiday window.

    This is an explicit reconstruction choice, not a recovered paper algorithm.
    """
    anchor = event.get("reference_anchor")
    if not anchor:
        return []
    target_anchor = day(event["anchor_date"])
    source_anchor = day(anchor)
    return [
        {
            "ds": ds,
            "reference_ds": (source_anchor + (day(ds) - target_anchor)).isoformat(),
        }
        for ds in dates(max(start, event["start_date"]), min(end, event["end_date"]))
    ]


def synthetic(seed=42):
    rng = random.Random(seed)
    calendar = default_calendar()
    rows = []
    for i, ds in enumerate(dates("2024-01-01", "2026-05-05")):
        d = day(ds)
        normal = (
            1300
            + 95 * math.cos(2 * math.pi * (d.timetuple().tm_yday - 20) / 365.25)
            + 0.12 * i
        )
        weekly = [80, 55, 0, -35, -90, -470, -530][d.weekday()]
        y = normal + weekly + rng.gauss(0, 65)
        for event in calendar:
            start, end = day(event["start_date"]), day(event["end_date"])
            if (
                event["event"] == "spring_festival"
                and start - timedelta(days=7) <= d < start
            ):
                progress = (d - (start - timedelta(days=7))).days / 7
                y = normal * (0.8 - 0.42 * progress) + rng.gauss(0, 35)
            if start <= d <= end:
                offset = (d - start).days
                if event["event"] == "spring_festival":
                    y = normal * (0.23 + 0.012 * offset) + rng.gauss(0, 22)
                else:
                    y = (
                        normal
                        * {
                            "new_year": 0.48,
                            "qingming": 0.49,
                            "labor_day": 0.39,
                            "national_day": 0.34,
                        }.get(event["event"], 0.6)
                        + 30 * offset
                        + rng.gauss(0, 28)
                    )
        rows.append({"ds": ds, "y": round(max(50, y), 2)})
    return {
        "id": uid(),
        "name": "Synthetic daily domestic route demand",
        "synthetic": True,
        "seed": seed,
        "description": "Seasonal shape inspired by Figure 6; synthetic, not recovered or digitized observations.",
        "created_at": now(),
        "rows": rows,
    }
