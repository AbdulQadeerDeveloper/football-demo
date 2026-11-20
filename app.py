import pandas as pd, streamlit as st
import dc_demo as dc

st.set_page_config(page_title="Football Forecast", page_icon="⚽", layout="centered")

st.markdown("""
<style>
.block-container{max-width:720px;padding-top:4.5rem}
.hero{background:linear-gradient(135deg,#0f5132,#1f8a5b);color:#fff;border-radius:16px;padding:20px 22px;margin-bottom:18px}
.hero h1{margin:0;font-size:26px;color:#fff;padding:0}
.hero p{margin:4px 0 0;opacity:.85;font-size:14px}
.match{display:flex;justify-content:space-between;align-items:center;font-size:22px;font-weight:700;margin:14px 0 2px}
.match span.vs{font-size:13px;font-weight:400;opacity:.6}
.xg{opacity:.65;font-size:13px;margin-bottom:12px}
.pbar{display:flex;height:44px;border-radius:12px;overflow:hidden;color:#fff;font-weight:700;font-size:15px}
.pbar div{display:flex;align-items:center;justify-content:center}
.plbl{display:flex;justify-content:space-between;font-size:12px;opacity:.65;margin:6px 2px 18px}
.srow{display:flex;align-items:center;gap:12px;margin:8px 0}
.sc{width:52px;font-weight:700;font-size:16px}
.trk{flex:1;background:rgba(128,128,128,.18);height:16px;border-radius:8px;overflow:hidden}
.trk i{display:block;height:100%;background:linear-gradient(90deg,#2b6cb0,#63a4e8);border-radius:8px}
.pc{width:56px;text-align:right;font-variant-numeric:tabular-nums;font-weight:600}
.kpi{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:6px}
.kpi div{border:1px solid rgba(128,128,128,.25);border-radius:12px;padding:12px;text-align:center}
.kpi b{display:block;font-size:22px}.kpi small{opacity:.65;font-size:11px}
.foot{opacity:.55;font-size:12px;margin-top:22px;text-align:center}
</style>
<div class="hero"><h1>⚽ Premier League Forecast</h1>
<p>Dixon–Coles statistical model on real EPL data (2015-16 to 2020-21). Not a chatbot.</p></div>
""", unsafe_allow_html=True)

@st.cache_data
def get_data(): return dc.load()

@st.cache_resource
def get_model(_d): return dc.fit(_d, _d.date.max() + pd.Timedelta(days=1))

d = get_data(); m = get_model(d)
teams = sorted(m["ix"])

c1, c2 = st.columns(2)
home = c1.selectbox("Home team", teams, index=teams.index("Liverpool") if "Liverpool" in teams else 0)
away_opts = [t for t in teams if t != home]
away = c2.selectbox("Away team", away_opts, index=away_opts.index("Arsenal") if "Arsenal" in away_opts else 0)

P, lh, la = dc.score_matrix(m, home, away)
H, D, A = dc.outcome_probs(P)

st.markdown(f"""
<div class="match"><span>{home}</span><span class="vs">vs</span><span>{away}</span></div>
<div class="xg">Expected goals: {lh:.2f} – {la:.2f}</div>
<div class="pbar">
<div style="width:{H*100:.1f}%;background:#1f8a5b">{H:.0%}</div>
<div style="width:{D*100:.1f}%;background:#7b8794">{D:.0%}</div>
<div style="width:{A*100:.1f}%;background:#d9534f">{A:.0%}</div></div>
<div class="plbl"><span>Home win {H:.1%}</span><span>Draw {D:.1%}</span><span>Away win {A:.1%}</span></div>
""", unsafe_allow_html=True)

ts = dc.top_scores(P, 5); top = ts[0][1]
rows = "".join(f'<div class="srow"><span class="sc">{s}</span><div class="trk"><i style="width:{p/top*100:.0f}%"></i></div><span class="pc">{p:.1%}</span></div>' for s, p in ts)
st.markdown(f"**Top 5 correct scores**{rows}", unsafe_allow_html=True)

st.markdown("**Backtest · 2020-21 · 378 matches (walk-forward, no look-ahead)**")
st.markdown("""
<div class="kpi">
<div><b>50.3%</b><small>Result accuracy<br>(baseline 37.6%)</small></div>
<div><b>1.025</b><small>Log loss<br>(naive 1.091)</small></div>
<div><b>46.8%</b><small>Actual score<br>in top 5</small></div></div>
<div class="foot">Demo only. Uses historical data; live odds, team news, n8n and Telegram come in the full build.</div>
""", unsafe_allow_html=True)