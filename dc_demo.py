"""Dixon-Coles football forecasting demo (real EPL data 2015-16 .. 2020-21)."""
import glob, numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

NAMES = {"Brighton & Hove Albion":"Brighton","Manchester Utd":"Manchester United","Newcastle Utd":"Newcastle United",
 "Sheffield Utd":"Sheffield United","Tottenham Hotspur":"Tottenham","West Bromwich Albion":"West Brom",
 "West Ham United":"West Ham","Wolverhampton Wanderers":"Wolves","AFC Bournemouth":"Bournemouth"}

def load():
    fr = []
    for f in sorted(glob.glob("data/*.csv")):
        d = pd.read_csv(f); d["season"] = f.split("/")[-1][:-4]; fr.append(d)
    d = pd.concat(fr); d["FT"] = d["FT"].astype(str).str.replace("–","-")
    d = d[d["FT"].str.match(r"^\d+-\d+$", na=False)].copy()
    d[["hg","ag"]] = d["FT"].str.split("-", expand=True).astype(int)
    d["date"] = pd.to_datetime(d["Date"].str.extract(r"(\w{3} \d{1,2} \d{4})")[0], format="%b %d %Y")
    d["home"] = d["Team 1"].str.replace(" FC","").str.replace(" AFC","")
    d["away"] = d["Team 2"].str.replace(" FC","").str.replace(" AFC","")
    d["home"]=d.home.replace(NAMES); d["away"]=d.away.replace(NAMES)
    return d.sort_values("date").reset_index(drop=True)[["date","season","home","away","hg","ag"]]

def tau(x, y, lh, la, rho):
    t = np.ones_like(lh)
    m = (x==0)&(y==0); t[m] = 1 - lh[m]*la[m]*rho
    m = (x==0)&(y==1); t[m] = 1 + lh[m]*rho
    m = (x==1)&(y==0); t[m] = 1 + la[m]*rho
    m = (x==1)&(y==1); t[m] = 1 - rho
    return t

def fit(df, ref_date, xi=0.0019):
    teams = sorted(set(df.home)|set(df.away)); n = len(teams); ix = {t:i for i,t in enumerate(teams)}
    h = df.home.map(ix).values; a = df.away.map(ix).values
    x = df.hg.values; y = df.ag.values
    w = np.exp(-xi*(ref_date-df.date).dt.days.values)       # time decay
    def nll(p):
        att, dfn = p[:n], p[n:2*n]; g, hadv, rho = p[2*n], p[2*n+1], p[2*n+2]
        lh = np.exp(g+hadv+att[h]+dfn[a]); la = np.exp(g+att[a]+dfn[h])
        ll = np.log(np.clip(tau(x,y,lh,la,rho),1e-9,None)) + poisson.logpmf(x,lh) + poisson.logpmf(y,la)
        return -(w*ll).sum() + 50*(att.sum()**2)             # identifiability
    p0 = np.concatenate([np.zeros(2*n), [0.2,0.25,-0.05]])
    r = minimize(nll, p0, method="L-BFGS-B", bounds=[(-2,2)]*(2*n)+[(-1,1),(-1,1),(-0.3,0.3)])
    p = r.x
    return dict(ix=ix, att=p[:n], dfn=p[n:2*n], g=p[2*n], hadv=p[2*n+1], rho=p[2*n+2])

def score_matrix(m, home, away, mx=8):
    i, j = m["ix"][home], m["ix"][away]
    lh = np.exp(m["g"]+m["hadv"]+m["att"][i]+m["dfn"][j]); la = np.exp(m["g"]+m["att"][j]+m["dfn"][i])
    X, Y = np.meshgrid(range(mx+1), range(mx+1), indexing="ij")
    P = poisson.pmf(X,lh)*poisson.pmf(Y,la)*tau(X,Y,np.full(X.shape,lh),np.full(X.shape,la),m["rho"])
    return P/P.sum(), lh, la

def outcome_probs(P):
    return np.tril(P,-1).sum(), np.trace(P), np.triu(P,1).sum()   # H, D, A

def top_scores(P, k=5):
    idx = np.dstack(np.unravel_index(np.argsort(-P.ravel())[:k], P.shape))[0]
    return [(f"{i}-{j}", P[i,j]) for i,j in idx]

if __name__ == "__main__":
    d = load()
    print(f"Matches loaded: {len(d)} | seasons: {d.season.nunique()} ({d.season.min()} to {d.season.max()})")
    train = d[d.season < "2020-21"]; test = d[d.season == "2020-21"].reset_index(drop=True)

    # ---- DEMO MATCH ----
    row = test.iloc[200]
    m = fit(d[d.date < row.date], row.date)
    P, lh, la = score_matrix(m, row.home, row.away)
    H, D, A = outcome_probs(P)
    print(f"\nDEMO: {row.home} vs {row.away} ({row.date.date()})  [model trained on matches before this date]")
    print(f"Expected goals: {lh:.2f} - {la:.2f} | rho={m['rho']:.3f} home_adv={np.exp(m['hadv']):.2f}x")
    print(f"Home {H:.1%} | Draw {D:.1%} | Away {A:.1%}")
    print("Top 5 correct scores:", ", ".join(f"{s} ({p:.1%})" for s,p in top_scores(P)))
    print(f"Actual result: {row.hg}-{row.ag}")

    # ---- BACKTEST: walk-forward, refit every 10 days on past data only ----
    recs = []; m = None; last = None
    for _, r in test.iterrows():
        if m is None or (r.date-last).days >= 10:
            m = fit(d[d.date < r.date], r.date); last = r.date
        if r.home not in m["ix"] or r.away not in m["ix"]: continue   # newly promoted, no history yet
        P, _, _ = score_matrix(m, r.home, r.away)
        p = np.array(outcome_probs(P)); res = 0 if r.hg>r.ag else (1 if r.hg==r.ag else 2)
        ts = [s for s,_ in top_scores(P,5)]
        recs.append((p, res, f"{r.hg}-{r.ag}" in ts, f"{r.hg}-{r.ag}"==ts[0]))
    probs = np.array([r[0] for r in recs]); res = np.array([r[1] for r in recs])
    base = np.array([train.assign(r=np.sign(train.hg-train.ag)).r.map({1:0,0:1,-1:2}).value_counts(normalize=True).sort_index().values]*len(res))
    ll = lambda p: -np.mean(np.log(p[np.arange(len(res)),res]))
    br = lambda p: np.mean(((p-np.eye(3)[res])**2).sum(1))
    print(f"\nBACKTEST: {len(res)} matches of {test.season[0]} (walk-forward, no look-ahead)")
    print(f"Outcome accuracy:     {np.mean(probs.argmax(1)==res):.1%}   (always-home baseline: {np.mean(res==0):.1%})")
    print(f"Log loss:             {ll(probs):.3f}   (naive base-rate: {ll(base):.3f}; lower = better)")
    print(f"Brier score:          {br(probs):.3f}   (naive base-rate: {br(base):.3f}; lower = better)")
    print(f"Actual score in top-5: {np.mean([r[2] for r in recs]):.1%} | exact top-1: {np.mean([r[3] for r in recs]):.1%}")
    # calibration
    allp = probs.ravel(); alle = np.eye(3)[res].ravel()
    bins = [0,.2,.3,.4,.5,.6,1]
    print("Calibration (predicted -> actual):")
    for lo,hi in zip(bins[:-1],bins[1:]):
        k=(allp>=lo)&(allp<hi)
        if k.sum()>10: print(f"  {lo:.0%}-{hi:.0%}: {allp[k].mean():.1%} -> {alle[k].mean():.1%}  (n={k.sum()})")
