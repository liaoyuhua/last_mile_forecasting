# Last Mile Forecasting

Research implementation of [Bridging the Last Mile of Time Series Forecasting with LLM Agents](https://arxiv.org/pdf/2606.02497).

Built with TimesFM 2.5 and `smolagents.CodeAgent`. Supports Claude, GPT, and DeepSeek.

## How it works

1. **Forecast:** TimesFM generates a baseline from observed history. The forecast agent queries historical data, a local holiday CSV, and relevant experiences to propose evidence-backed revisions. For multiple events, read-only local agents produce proposals for the main agent to merge.
2. **Freeze:** Preserve the baseline, revised forecast, supporting evidence, and observable agent trajectory. Future actuals are excluded from prediction inputs.
3. **Reflect daily:** As each actual arrives, an independent agent reviews the cumulative observed interval and original trajectory, then creates, updates, or deletes experiences. Issued forecasts stay unchanged; experiences inform **only subsequent forecasts**.

Each experiment starts with an empty JSON bank mapping experience titles to `description`, `observation`, and `experience`. Retrieval progressively exposes titles and descriptions before selected full entries. Only forecasts within the same sequence share experiences.

## Setup

Python 3.11+ required. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,agent,timesfm]'
cp .env.example .env
```

For offline use only, install `pip install -e '.[dev]'` instead. TimesFM downloads its weights on first use.

## Demo

```bash
# Offline: synthetic data, seasonal baseline, deterministic policies, no API calls.
python -m last_mile --output runtime/demo-offline

# Live: set ANTHROPIC_API_KEY in .env first.
python -m last_mile --backbone timesfm --mode live \
  --provider anthropic --model YOUR_SONNET_MODEL_ID \
  --output runtime/demo-live
```

Alternatively, use `--provider openai` with `OPENAI_API_KEY`, or `--provider deepseek` with `DEEPSEEK_API_KEY`, and the corresponding model ID.

The demo runs two consecutive seven-day forecasts with daily reflection: two forecast sessions and fourteen reflection sessions in live mode. Use `--horizon 2` for a shorter run. Choose a fresh output directory for each experiment.

To supply local data, add `--data path/to/daily.csv --calendar data/holidays.csv`. Daily CSV requires continuous `ds,y` rows, at least 28 historical days, and actuals covering both demo windows. Adjust `--start YYYY-MM-DD` as needed.

Outputs include forecast CSVs, metrics, execution trajectories, and `experiences.json`.

## Notes

Original code and data were unavailable; this is a reconstruction. Synthetic demo results do not reproduce the paper's scores. Offline mode verifies the workflow, not LLM reasoning.

- Tests: `python -m pytest -q`
- [MIT License](LICENSE)
