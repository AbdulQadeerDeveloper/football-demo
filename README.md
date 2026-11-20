# Football Intelligence System (Dixon-Coles + n8n)

## Files
- `model.py`       Dixon-Coles model (time-decayed), outcome + correct-score probabilities
- `sources.py`     API clients: football-data.org (fixtures/results), The Odds API (odds), API-Football (injuries), Telegram
- `main.py`        Pipeline: sync | prematch | results | report
- `db.py`          SQLite schema (matches, predictions, odds snapshots, team news)
- `teams.py`       Team-name normalisation across sources
- `n8n/`           Two importable workflows (pre-match every 3h; results + accuracy report daily)
- `app.py`, `dc_demo.py`  Streamlit demo UI and original backtest script

## Quick test (no API keys)
    pip install -r requirements.txt
    python main.py prematch --demo --db demo.db
    python main.py results  --demo --db demo.db
    python main.py report   --demo --db demo.db

## Live setup
1. Copy `.env.example` values into environment variables (API keys, Telegram bot token + chat id).
2. `python main.py sync`          (5 past seasons + current; free football-data.org tier is rate limited)
3. `python main.py prematch --hours 24 --send`
4. In n8n (self-hosted; the Execute Command node is unavailable on n8n Cloud): import both JSON files,
   set the project path in the command, your Telegram credential and chat id, then activate.

## Status / limits (be honest with the client)
- Offline demo mode is tested. API calls in `sources.py` are written from provider docs and NOT yet tested with live keys.
- Provider plans decide what history/odds/injury data is available; verify coverage before promising it.
- Team news = injury/suspension feed only; it is reported, not yet fed into the model.
- Odds are stored as snapshots for movement tracking; they are not used to train the model.
