import os
FOOTBALL_DATA_TOKEN = os.getenv("FOOTBALL_DATA_TOKEN", "")   # football-data.org (fixtures + results)
ODDS_API_KEY        = os.getenv("ODDS_API_KEY", "")          # the-odds-api.com
API_FOOTBALL_KEY    = os.getenv("API_FOOTBALL_KEY", "")      # api-football.com (injuries / team news)
TELEGRAM_TOKEN      = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID    = os.getenv("TELEGRAM_CHAT_ID", "")
DB_PATH             = os.getenv("DB_PATH", "football.db")
CURRENT_SEASON_START = int(os.getenv("CURRENT_SEASON_START", "2026"))   # 2026 = 2026-27
SEASONS_BACK        = 5                                      # five completed seasons + current
MODEL_VERSION       = "dc-v1"
