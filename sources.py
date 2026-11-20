"""API clients. NOT tested against live APIs (no keys in build environment) - verify with your keys."""
import time, datetime as dt, requests
import config as C
from teams import norm, match_team

def _season_label(y): return f"{y}-{str(y+1)[2:]}"

# ---------- football-data.org: fixtures + results ----------
def fetch_matches(seasons):
    """seasons: list of start years. Free tier = 10 requests/min, so we pause between calls."""
    out = []
    for y in seasons:
        r = requests.get("https://api.football-data.org/v4/competitions/PL/matches",
                         params={"season": y}, headers={"X-Auth-Token": C.FOOTBALL_DATA_TOKEN}, timeout=30)
        r.raise_for_status()
        for m in r.json()["matches"]:
            ft = m["score"]["fullTime"]
            out.append(dict(id=str(m["id"]), date=m["utcDate"], season=_season_label(y),
                            home=norm(m["homeTeam"]["name"]), away=norm(m["awayTeam"]["name"]),
                            hg=ft["home"], ag=ft["away"], status=m["status"]))
        time.sleep(7)
    return out

# ---------- The Odds API: bookmaker 1X2 odds ----------
def fetch_odds(known_teams):
    """Returns {(home, away): [(bookmaker, o_home, o_draw, o_away), ...]}"""
    r = requests.get("https://api.the-odds-api.com/v4/sports/soccer_epl/odds",
                     params={"apiKey": C.ODDS_API_KEY, "regions": "uk", "markets": "h2h", "oddsFormat": "decimal"}, timeout=30)
    r.raise_for_status(); res = {}
    for ev in r.json():
        h, a = match_team(ev["home_team"], known_teams), match_team(ev["away_team"], known_teams)
        if not h or not a: continue
        for bk in ev["bookmakers"]:
            mk = next((m for m in bk["markets"] if m["key"] == "h2h"), None)
            if not mk: continue
            px = {o["name"]: o["price"] for o in mk["outcomes"]}
            if ev["home_team"] in px and ev["away_team"] in px and "Draw" in px:
                res.setdefault((h, a), []).append((bk["key"], px[ev["home_team"]], px["Draw"], px[ev["away_team"]]))
    return res

# ---------- API-Football: injuries / suspensions ----------
def fetch_injuries(date, known_teams):
    """Returns [(team, player, kind, reason)] for fixtures on `date` (YYYY-MM-DD)."""
    r = requests.get("https://v3.football.api-sports.io/injuries",
                     params={"league": 39, "season": C.CURRENT_SEASON_START, "date": date},
                     headers={"x-apisports-key": C.API_FOOTBALL_KEY}, timeout=30)
    r.raise_for_status(); out = []
    for x in r.json().get("response", []):
        t = match_team(x["team"]["name"], known_teams)
        if t: out.append((t, x["player"]["name"], x["player"].get("type", ""), x["player"].get("reason", "")))
    return out

def telegram_send(text):
    if not (C.TELEGRAM_TOKEN and C.TELEGRAM_CHAT_ID): return False
    r = requests.post(f"https://api.telegram.org/bot{C.TELEGRAM_TOKEN}/sendMessage",
                      json={"chat_id": C.TELEGRAM_CHAT_ID, "text": text}, timeout=30)
    return r.ok
