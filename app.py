import pandas as pd
import streamlit as st
import dc_demo as dc

st.set_page_config(page_title="Premier League Forecast", page_icon="⚽", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
header, [data-testid="stHeader"], [data-testid="stAppHeader"], .stAppHeader, [data-testid="stToolbar"], [data-testid="stAppToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"], .stAppDeployButton, #MainMenu, footer {display: none !important; visibility: hidden !important; height: 0 !important;}
.block-container, [data-testid="stMainBlockContainer"] {max-width: 1100px; padding: clamp(1rem, 3vw, 2rem) clamp(1rem, 4vw, 2.5rem) 3rem !important;}
[data-testid="stHorizontalBlock"] {gap: 1rem;}
[data-testid="stMetric"] {min-width: 0; overflow-wrap: anywhere;}
[data-testid="stDataFrame"] {max-width: 100%;}
.stSelectbox, [data-testid="stSelectbox"] {min-width: 0;}
.hero {box-sizing: border-box; width: 100%;}
.hero {background: linear-gradient(110deg,#10583d,#20885f); padding: 22px 26px;
       border-radius: 16px; color: white; margin-bottom: 22px;}
.hero h1 {font-size: 28px; margin: 0 0 4px 0; color: white;}
.hero p {margin: 0; opacity: .9; font-size: 14px;}
.metric-card {background: #f3f5f4; padding: 14px; border-radius: 12px;}
.small-muted {color: #777; font-size: 12px;}
.score-row {display:flex; align-items:center; gap:12px; margin: 9px 0;}
.score-label {width:42px; font-weight:700;}
.score-bar {height:12px; background:#e7e9e8; border-radius:8px; flex:1; overflow:hidden;}
.score-fill {height:100%; background:linear-gradient(90deg,#286eae,#65a6e7); border-radius:8px;}
.score-pct {width:58px; text-align:right; font-weight:600; white-space:nowrap;}
@media (max-width: 640px) {
  .block-container, [data-testid="stMainBlockContainer"] {padding: .8rem .8rem 2rem !important;}
  .hero {padding: 17px 16px; border-radius: 13px; margin-bottom: 16px;}
  .hero h1 {font-size: 22px; line-height: 1.2;}
  .hero p {font-size: 12px; line-height: 1.45; margin-top: 5px;}
  h2 {font-size: 1.25rem !important;}
  h3 {font-size: 1.1rem !important;}
  [data-testid="stHorizontalBlock"] {flex-wrap: wrap; gap: .55rem;}
  [data-testid="stHorizontalBlock"] > [data-testid="column"], [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {flex: 1 1 100% !important; min-width: 0 !important; width: 100% !important;}
  [data-testid="stMetric"] {padding: 10px 12px; background: #f3f5f4; border-radius: 10px;}
  [data-testid="stMetricLabel"] {font-size: 12px;}
  [data-testid="stMetricValue"] {font-size: 1.55rem;}
  .score-row {gap: 8px; margin: 11px 0;}
  .score-label {width: 36px; flex: 0 0 36px; font-size: 13px;}
  .score-bar {min-width: 0; height: 10px;}
  .score-pct {width: 52px; flex: 0 0 52px; font-size: 12px;}
  [data-testid="stDataFrame"] {overflow-x: auto;}
  [data-testid="stDataFrame"] iframe {max-width: 100%;}
}
</style>
<div class="hero">
  <h1>⚽ Premier League Forecast</h1>
  <p>Dixon–Coles statistical model · historical EPL results · walk-forward evaluation</p>
</div>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return dc.load()

@st.cache_data
def run_backtest(_data, season):
    return dc.backtest(_data, season)

@st.cache_resource
def fit_latest(_data):
    return dc.fit(_data, _data.date.max() + pd.Timedelta(days=1))

try:
    data = load_data()
except Exception as exc:
    st.error(f"Could not load the CSV data: {exc}")
    st.stop()

seasons = sorted(data.season.unique())
model = fit_latest(data)
teams = sorted(model["ix"])

st.caption(
    f"Dataset: {len(data):,} completed matches · {len(seasons)} seasons "
    f"({seasons[0]} to {seasons[-1]})."
)

st.subheader("Match prediction")
col1, col2 = st.columns(2)
home = col1.selectbox("Home team", teams, index=teams.index("Aston Villa") if "Aston Villa" in teams else 0)
away_options = [team for team in teams if team != home]
away = col2.selectbox("Away team", away_options, index=away_options.index("Brighton") if "Brighton" in away_options else 0)

P, home_xg, away_xg = dc.score_matrix(model, home, away)
home_p, draw_p, away_p = dc.outcome_probs(P)

st.markdown(f"### {home} <span style='color:#999;font-size:14px'>vs</span> {away}", unsafe_allow_html=True)
st.caption(f"Expected goals: **{home_xg:.2f} – {away_xg:.2f}** · Model fitted using all supplied historical matches.")
a, b, c = st.columns(3)
a.metric("Home win", f"{home_p:.1%}")
b.metric("Draw", f"{draw_p:.1%}")
c.metric("Away win", f"{away_p:.1%}")

st.markdown("#### Outcome probability")
st.markdown(
    f"""<div style="display:flex;height:40px;border-radius:10px;overflow:hidden;color:white;
        font-weight:700;text-align:center;line-height:40px">
        <div style="width:{home_p*100:.4f}%;background:#20885f">{home_p:.0%}</div>
        <div style="width:{draw_p*100:.4f}%;background:#7d8997">{draw_p:.0%}</div>
        <div style="width:{away_p*100:.4f}%;background:#d9534f">{away_p:.0%}</div>
        </div>""",
    unsafe_allow_html=True
)

st.markdown("#### Top 5 correct scores")
top = dc.top_scores(P, 5)
maxp = max(p for _, p in top)
for score, probability in top:
    width = probability / maxp * 100
    st.markdown(
        f"""<div class="score-row">
        <div class="score-label">{score}</div>
        <div class="score-bar"><div class="score-fill" style="width:{width:.2f}%"></div></div>
        <div class="score-pct">{probability:.1%}</div>
        </div>""",
        unsafe_allow_html=True
    )

st.divider()
st.subheader("Historical backtest")
st.caption("Walk-forward testing: each match is predicted using only matches played before its date.")
default_season = "2020-21" if "2020-21" in seasons else seasons[-1]
bt_season = st.selectbox("Test season", seasons, index=seasons.index(default_season))
with st.spinner("Running walk-forward backtest…"):
    try:
        metrics = run_backtest(data, bt_season)
    except Exception as exc:
        st.error(f"Backtest could not be completed: {exc}")
        st.stop()

m1, m2, m3 = st.columns(3)
m1.metric("Matches tested", f"{metrics['matches']:,}")
m2.metric("Outcome accuracy", f"{metrics['outcome_accuracy']:.1%}")
m3.metric("Actual score in top 5", f"{metrics['top5_score_rate']:.1%}")

metrics_table = pd.DataFrame([
    ["Outcome accuracy", f"{metrics['outcome_accuracy']:.1%}", "Higher is better"],
    ["Log loss", f"{metrics['log_loss']:.3f}", f"Baseline: {metrics['baseline_log_loss']:.3f} · lower is better"],
    ["Brier score", f"{metrics['brier_score']:.3f}", f"Baseline: {metrics['baseline_brier_score']:.3f} · lower is better"],
    ["Actual score in top 5", f"{metrics['top5_score_rate']:.1%}", "Share of matches where actual score was among five most likely"],
    ["Exact top-1 score", f"{metrics['top1_score_rate']:.1%}", "Share of matches where most likely score was exact"],
], columns=["Metric", "Model", "Interpretation"])
st.dataframe(metrics_table, hide_index=True, use_container_width=True)

with st.expander("Model methodology and limitations"):
    st.markdown("""
    - Attack and defence strengths are estimated from historical goals.
    - The model uses a Poisson goal model with the Dixon–Coles low-score adjustment (`rho`).
    - Recent matches receive greater weight through exponential time decay.
    - The backtest refits on a rolling historical window and excludes matches on or after each fixture date.
    - Log loss and Brier score are compared with a historical outcome-frequency baseline.

    **Data limitation:** This ZIP contains six completed seasons only, from 2015–16 through 2020–21.
    It does not contain the five most recent completed seasons or the current 2026–27 season.
    The prediction screen is a historical-data demonstration, not a live fixture feed.
    """)

st.caption("Prototype only · No live team-news feed, bookmaker odds, n8n scheduling or Telegram delivery is connected yet.")