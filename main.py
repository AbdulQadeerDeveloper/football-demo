"""Football intelligence pipeline.
  python main.py sync                 # load 5 past seasons + current season into SQLite
  python main.py prematch --hours 24  # predict upcoming fixtures, snapshot odds + news, print/send Telegram reports
  python main.py results              # fetch finished results, score stored predictions
  python main.py report               # accuracy summary of all tracked predictions
Add --demo to run fully offline on the bundled CSV data (no API keys needed).
"""
import argparse, json, datetime as dt, glob
import numpy as np, pandas as pd
import config as C, model, db
from teams import norm

def now(): return dt.datetime.utcnow().replace(microsecond=0).isoformat()

def load_history(c):
    d = pd.read_sql("SELECT * FROM matches WHERE status='FINISHED' AND hg IS NOT NULL", c)
    d["date"] = pd.to_datetime(d["date"]).dt.tz_localize(None)
    return d

def implied(o):  # remove bookmaker margin
    p = 1 / np.array(o); return p / p.sum()

def build_message(row, odds_rows, news):
    top = ", ".join(f"{s} ({p:.1%})" for s, p in json.loads(row["top_scores"]))
    t = [f"⚽ {row['home']} vs {row['away']}", f"🕐 Kick-off {row['kickoff'][:16].replace('T',' ')} UTC",
         f"xG {row['xg_h']:.2f} – {row['xg_a']:.2f}",
         f"1: {row['p_home']:.1%} | X: {row['p_draw']:.1%} | 2: {row['p_away']:.1%}", f"Top scores: {top}"]
    if odds_rows:
        first, last = odds_rows[0], odds_rows[-1]
        mk = implied(last[1:4]); mv = (implied(last[1:4]) - implied(first[1:4])) * 100
        t.append(f"📈 Market (latest of {len(odds_rows)} snapshots): 1 {mk[0]:.1%} X {mk[1]:.1%} 2 {mk[2]:.1%}")
        t.append(f"   Move since first snapshot (pp): 1 {mv[0]:+.1f} X {mv[1]:+.1f} 2 {mv[2]:+.1f}")
        edge = np.array([row['p_home'], row['p_draw'], row['p_away']]) - mk
        t.append(f"   Model vs market (pp): 1 {edge[0]*100:+.1f} X {edge[1]*100:+.1f} 2 {edge[2]*100:+.1f}")
    else: t.append("📈 Odds: not available")
    t.append("📋 Team news: " + ("; ".join(f"{p} ({tm}, {k or reason})" for tm, p, k, reason in news) if news else "none reported"))
    return "\n".join(t)

def cmd_sync(a):
    import sources
    c = db.conn(); yrs = list(range(C.CURRENT_SEASON_START - C.SEASONS_BACK, C.CURRENT_SEASON_START + 1))
    rows = sources.fetch_matches(yrs)
    c.executemany("INSERT OR REPLACE INTO matches VALUES(:id,:date,:season,:home,:away,:hg,:ag,:status)", rows); c.commit()
    print(f"synced {len(rows)} matches for seasons {yrs[0]}-{yrs[-1]}")

def cmd_prematch(a):
    c = db.conn(a.db)
    if a.demo:   # last 3 matches of bundled CSVs act as 'upcoming' fixtures; model never sees them
        d = model.load_csv_dir("data"); up = d.tail(3).copy(); hist = d.iloc[:-3]
        fixtures = [dict(kickoff=r.date.isoformat(), home=r.home, away=r.away) for r in up.itertuples()]
    else:
        import sources
        hist = load_history(c); lim = (dt.datetime.utcnow() + dt.timedelta(hours=a.hours)).isoformat()
        fx = c.execute("SELECT date,home,away FROM matches WHERE status IN('TIMED','SCHEDULED') AND date<=? AND date>=?", (lim, now())).fetchall()
        fixtures = [dict(kickoff=r["date"], home=r["home"], away=r["away"]) for r in fx]
    if not fixtures: print("[]"); return
    m = model.fit(hist, pd.Timestamp(dt.datetime.utcnow()) if not a.demo else hist.date.max() + pd.Timedelta(days=1))
    odds_now = news_all = {}
    if not a.demo:
        known = set(m["ix"])
        try: odds_now = sources.fetch_odds(known)
        except Exception as e: print(f"odds error: {e}", file=__import__('sys').stderr)
    msgs = []
    for f in fixtures:
        if f["home"] not in m["ix"] or f["away"] not in m["ix"]: continue   # newly promoted team without history
        P, lh, la = model.score_matrix(m, f["home"], f["away"]); H, D, A = model.outcome_probs(P)
        key = db.fkey(f["kickoff"], f["home"], f["away"]); ts = now()
        top = [(s, float(p)) for s, p in model.top_scores(P, 5)]
        c.execute("INSERT OR REPLACE INTO predictions(fixture_key,created_at,kickoff,home,away,xg_h,xg_a,p_home,p_draw,p_away,top_scores,model) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                  (key, ts, f["kickoff"], f["home"], f["away"], lh, la, H, D, A, json.dumps(top), C.MODEL_VERSION))
        for bk, oh, od, oa in odds_now.get((f["home"], f["away"]), []):
            c.execute("INSERT INTO odds VALUES(?,?,?,?,?,?)", (key, ts, bk, oh, od, oa))
        news = []
        if not a.demo and C.API_FOOTBALL_KEY:
            try:
                news = [n for n in sources.fetch_injuries(f["kickoff"][:10], set(m["ix"])) if n[0] in (f["home"], f["away"])]
                c.executemany("INSERT INTO news VALUES(?,?,?,?,?,?)", [(key, ts, *n) for n in news])
            except Exception as e: print(f"news error: {e}", file=__import__('sys').stderr)
        # average odds per snapshot time for movement: first snapshot vs latest
        snaps = c.execute("SELECT ts, AVG(o_home) h, AVG(o_draw) d, AVG(o_away) a FROM odds WHERE fixture_key=? GROUP BY ts ORDER BY ts", (key,)).fetchall()
        orows = [(s["ts"], s["h"], s["d"], s["a"]) for s in snaps]
        row = dict(kickoff=f["kickoff"], home=f["home"], away=f["away"], xg_h=lh, xg_a=la, p_home=H, p_draw=D, p_away=A, top_scores=json.dumps(top))
        msgs.append(build_message(row, orows, news))
    c.commit()
    if a.send:
        import sources
        for t in msgs: sources.telegram_send(t)
    print(json.dumps(msgs, ensure_ascii=False))

def cmd_results(a):
    c = db.conn(a.db)
    if a.demo:
        d = model.load_csv_dir("data"); fin = {db.fkey(r.date, r.home, r.away): (r.hg, r.ag) for r in d.itertuples()}
    else:
        import sources
        rows = sources.fetch_matches([C.CURRENT_SEASON_START]); fin = {}
        for r in rows:
            if r["status"] == "FINISHED": fin[db.fkey(r["date"], r["home"], r["away"])] = (r["hg"], r["ag"])
    n = 0
    for p in c.execute("SELECT fixture_key FROM predictions WHERE actual_hg IS NULL").fetchall():
        if p["fixture_key"] in fin:
            hg, ag = fin[p["fixture_key"]]; o = "H" if hg > ag else ("D" if hg == ag else "A")
            c.execute("UPDATE predictions SET actual_hg=?,actual_ag=?,outcome=? WHERE fixture_key=? AND actual_hg IS NULL", (hg, ag, o, p["fixture_key"])); n += 1
    c.commit(); print(f"scored {n} predictions")

def cmd_report(a):
    c = db.conn(a.db)
    r = c.execute("SELECT * FROM predictions WHERE outcome IS NOT NULL").fetchall()
    if not r: print("No scored predictions yet."); return
    hit = ll = br = top5 = 0
    for x in r:
        p = np.array([x["p_home"], x["p_draw"], x["p_away"]]); k = "HDA".index(x["outcome"])
        hit += p.argmax() == k; ll += -np.log(p[k]); br += ((p - np.eye(3)[k]) ** 2).sum()
        top5 += f"{x['actual_hg']}-{x['actual_ag']}" in [s for s, _ in json.loads(x["top_scores"])]
    n = len(r)
    txt = (f"📊 Prediction tracker ({n} matches)\nAccuracy {hit/n:.1%} | Log loss {ll/n:.3f} | Brier {br/n:.3f}\nActual score in top 5: {top5/n:.1%}")
    if a.send:
        import sources; sources.telegram_send(txt)
    print(txt)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["sync", "prematch", "results", "report"])
    ap.add_argument("--hours", type=int, default=24); ap.add_argument("--demo", action="store_true")
    ap.add_argument("--send", action="store_true", help="send to Telegram"); ap.add_argument("--db", default=None)
    a = ap.parse_args(); {"sync": cmd_sync, "prematch": cmd_prematch, "results": cmd_results, "report": cmd_report}[a.cmd](a)
