# Premier League Forecast Demo

A Streamlit demonstration of a statistical football forecasting model using the supplied historical EPL CSV files.

## Run locally

```bash
cd football_demo
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The command-line model and walk-forward backtest can also be run with:

```bash
python dc_demo.py
```

## Included data

The ZIP currently includes six seasons: 2015-16 through 2020-21 (2,280 match rows).

## Model

- Poisson goal scoring model with Dixon-Coles low-score correction.
- Exponential time-decay weighting.
- Home/draw/away probabilities and top-five correct-score probabilities.
- Walk-forward backtest: each fixture is predicted from matches before its date.
- Outcome accuracy, log loss, Brier score, top-five score hit rate and exact top-one score rate.
- Baseline comparisons for log loss and Brier score.

## Important scope limitations

This is a working historical-data prototype, not the complete requested production system. The supplied dataset does **not** include the five most recent completed Premier League seasons or the current 2026-27 season. It also does not yet connect live sports APIs, team news, bookmaker odds, n8n workflows, scheduled updates, or Telegram reporting. Those require data-provider credentials, source selection and separate integration work.

Backtest metrics are computed from the included CSV data when the app runs; they are not hardcoded in the dashboard.
