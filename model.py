"""Dixon-Coles football forecasting demo (real EPL data 2015-16 .. 2020-21)."""
import glob, numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

NAMES = {"Brighton & Hove Albion":"Brighton","Manchester Utd":"Manchester United","Newcastle Utd":"Newcastle United",
 "Sheffield Utd":"Sheffield United","Tottenham Hotspur":"Tottenham","West Bromwich Albion":"West Brom",
 "West Ham United":"West Ham","Wolverhampton Wanderers":"Wolves","AFC Bournemouth":"Bournemouth"}

def load_csv_dir(path="data"):
    fr = []
    for f in sorted(glob.glob(path+"/*.csv")):
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

