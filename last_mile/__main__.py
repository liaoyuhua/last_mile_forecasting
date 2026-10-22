"""Run a two-window demonstration of forecast, daily reflection and retrieval."""

import argparse
import csv
from datetime import timedelta
from pathlib import Path

from dotenv import load_dotenv

from .data import ROOT, day, default_calendar, parse_calendar, parse_series, synthetic
from .experiment import Experiment, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, help="Daily ds,y CSV; defaults to seeded synthetic data"
    )
    parser.add_argument(
        "--calendar", type=Path, help="Local holiday CSV; defaults to data/holidays.csv"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "runtime/demo")
    parser.add_argument("--start", default="2026-02-15")
    parser.add_argument(
        "--horizon",
        type=int,
        default=7,
        help="Days in each of two consecutive forecasts",
    )
    parser.add_argument(
        "--backbone", choices=["seasonal", "timesfm"], default="seasonal"
    )
    parser.add_argument("--mode", choices=["demo", "live"], default="demo")
    parser.add_argument(
        "--provider", choices=["openai", "anthropic", "deepseek"], default="anthropic"
    )
    parser.add_argument(
        "--model", default="", help="Provider model ID without a provider/ prefix"
    )
    parser.add_argument("--base-url", default="")
    parser.add_argument("--max-steps", type=int, default=12)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    if not 1 <= args.horizon <= 365 or not 1 <= args.max_steps <= 100:
        parser.error("horizon must be 1-365 and max-steps must be 1-100")
    if (args.output / "experiment.json").exists():
        parser.error(
            "Output already contains an experiment; choose a new --output directory"
        )
    if args.mode == "live" and not args.model:
        parser.error(
            "Live mode requires --model and a provider API key in the environment"
        )
    rows = parse_series(args.data.read_text()) if args.data else synthetic()["rows"]
    calendar = (
        parse_calendar(args.calendar.read_text())
        if args.calendar
        else default_calendar()
    )
    starts = [day(args.start) + timedelta(days=i * args.horizon) for i in range(2)]
    by_date = {r["ds"]: r["y"] for r in rows}
    for start in starts:
        history = [r for r in rows if r["ds"] < start.isoformat()]
        if (
            len(history) < 28
            or history[-1]["ds"] != (start - timedelta(days=1)).isoformat()
        ):
            parser.error(
                "Each forecast requires at least 28 continuous historical observations"
            )
        for offset in range(args.horizon):
            if (start + timedelta(days=offset)).isoformat() not in by_date:
                parser.error("This demo requires actuals covering both windows")
    settings = {
        "mode": args.mode,
        "backbone": args.backbone,
        "provider": args.provider,
        "model": args.model,
        "base_url": args.base_url,
        "max_steps": args.max_steps,
        "context_length": 1024,
    }
    experiment = Experiment(args.output, settings, calendar)
    print(
        f"Mode: {args.mode}; backbone: {args.backbone}; synthetic: {not bool(args.data)}"
    )
    if args.mode == "live":
        print(
            f"Live demo: 2 forecasts and {2 * args.horizon} daily reflections; each may make multiple paid API requests."
        )
    records = []
    for index, start in enumerate(starts, 1):
        run = experiment.predict(
            [r for r in rows if r["ds"] < start.isoformat()],
            start.isoformat(),
            args.horizon,
        )
        print(
            f"Window {index}: {run['start']} to {run['end']}; inherited experiences: {len(run['memory_snapshot'])}"
        )
        for offset in range(args.horizon):
            ds = (start + timedelta(days=offset)).isoformat()
            result = experiment.observe(run["id"], ds, by_date[ds])
            print(f"  Observed {ds}; experience edits: {len(result['changes'])}")
        completed = experiment.state["runs"][-1]
        write_json(args.output / f"forecast-{index}.json", completed)
        with (args.output / f"forecast-{index}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=["ds", "y_baseline", "y_final", "y_actual"]
            )
            writer.writeheader()
            writer.writerows(
                r | {"y_actual": by_date[r["ds"]]} for r in run["workspace"]["forecast"]
            )
        records.append(
            {"window": index, "run_id": run["id"], "metrics": completed["metrics"]}
        )
    write_json(
        args.output / "summary.json",
        {
            "synthetic": not bool(args.data),
            "settings": settings,
            "notice": "Functional demonstration, not reproduced paper scores or an ablation study.",
            "windows": records,
        },
    )
    print(f"Artifacts saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
