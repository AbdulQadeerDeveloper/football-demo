import numpy as np, pandas as pd, streamlit as st
import dc_demo as dc

st.set_page_config(page_title="Football Forecast Demo", page_icon="⚽", layout="centered")
st.title("⚽ Premier League Forecast (Dixon–Coles)")
st.caption("Demo on real EPL data, 2015-16 to 2020-21. Statistical model, not a chatbot.")

@st.cache_data
def get_data(): return dc.load()

@st.cache_resource
def get_model(_d):
    return dc.fit(_d, _d.date.max() + pd.Timedelta(days=1))

d = get_data(); m = get_model(d)
teams = sorted(m["ix"])

c1, c2 = st.columns(2)
home = c1.selectbox("Home team", teams, index=teams.index("Liverpool") if "Liverpool" in teams else 0)
away = c2.selectbox("Away team", [t for t in teams if t != home], index=0)

P, lh, la = dc.score_matrix(m, home, away)
H, D, A = dc.outcome_probs(P)

st.subheader(f"{home} vs {away}")
st.caption(f"Expected goals: {lh:.2f} – {la:.2f}")
a, b, c = st.columns(3)
a.metric("Home win", f"{H:.1%}"); b.metric("Draw", f"{D:.1%}"); c.metric("Away win", f"{A:.1%}")

st.markdown("**Top 5 correct scores**")
ts = dc.top_scores(P, 5)
st.bar_chart(pd.DataFrame({"probability": [p for _, p in ts]}, index=[s for s, _ in ts]))

with st.expander("Backtest (2020-21, 378 matches, walk-forward)"):
    st.table(pd.DataFrame({
        "Metric": ["Result accuracy", "Log loss", "Brier score", "Actual score in top 5"],
        "Model": ["50.3%", "1.025", "0.610", "46.8%"],
        "Naive baseline": ["37.6%", "1.091", "0.664", "-"]}))
st.caption("Demo only: no live odds, team news, n8n or Telegram yet.")
