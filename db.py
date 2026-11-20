import sqlite3, json
from config import DB_PATH
SCHEMA = """
CREATE TABLE IF NOT EXISTS matches(id TEXT PRIMARY KEY, date TEXT, season TEXT, home TEXT, away TEXT, hg INTEGER, ag INTEGER, status TEXT);
CREATE TABLE IF NOT EXISTS predictions(fixture_key TEXT, created_at TEXT, kickoff TEXT, home TEXT, away TEXT, xg_h REAL, xg_a REAL,
  p_home REAL, p_draw REAL, p_away REAL, top_scores TEXT, model TEXT, actual_hg INTEGER, actual_ag INTEGER, outcome TEXT, PRIMARY KEY(fixture_key, created_at));
CREATE TABLE IF NOT EXISTS odds(fixture_key TEXT, ts TEXT, bookmaker TEXT, o_home REAL, o_draw REAL, o_away REAL);
CREATE TABLE IF NOT EXISTS news(fixture_key TEXT, ts TEXT, team TEXT, player TEXT, kind TEXT, reason TEXT);
"""
def conn(path=None):
    c = sqlite3.connect(path or DB_PATH); c.row_factory = sqlite3.Row; c.executescript(SCHEMA); return c
def fkey(date, home, away): return f"{str(date)[:10]}_{home}_{away}".replace(" ", "")
