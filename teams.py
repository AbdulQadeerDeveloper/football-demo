"""Team-name normalisation across data sources (football-data.org, The Odds API, API-Football, CSVs)."""
import difflib
ALIASES = {
 "brighton & hove albion":"Brighton","brighton and hove albion":"Brighton","manchester utd":"Manchester United",
 "newcastle utd":"Newcastle United","sheffield utd":"Sheffield United","tottenham hotspur":"Tottenham",
 "west bromwich albion":"West Brom","west ham united":"West Ham","wolverhampton wanderers":"Wolves",
 "afc bournemouth":"Bournemouth","nottm forest":"Nottingham Forest","leeds":"Leeds United","man city":"Manchester City",
 "man united":"Manchester United","spurs":"Tottenham"}
def norm(name: str) -> str:
    n = name.strip()
    for suf in (" FC"," AFC"):
        if n.endswith(suf): n = n[:-len(suf)]
    return ALIASES.get(n.lower(), n)
def match_team(name, known):
    """Map an arbitrary API name to a team known by the model; None if no close match."""
    n = norm(name)
    if n in known: return n
    c = difflib.get_close_matches(n, list(known), n=1, cutoff=0.8)
    return c[0] if c else None
