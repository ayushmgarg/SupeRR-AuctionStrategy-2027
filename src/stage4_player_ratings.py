"""Stage 4: per-player components.

For each player and role (bat / bowl):
  press_v   value per pressure ball, skill units, opponent-adjusted
  press_w   wins added per 100 pressure balls (win units)
  base_v    the same player's value on live, non-pressure balls, re-weighted to the overs in which
            his pressure balls happened (so a death specialist is compared with his own death overs)
  lift      press_v - base_v
  x_*       process rates on all live balls, each measured against the average ball in the same
            situation (innings, phase, pressure or not, pace or spin): scoring, dots, dismissals,
            and for bowlers error balls (wide, no-ball, full toss)
Value components come with a match-clustered sampling variance and an empirical-Bayes shrunk version.

Everything is computed from a per-(player, match, phase, category) table so that the validation
stage can re-run it on any subset or resample of matches in milliseconds.
"""
import numpy as np
import pandas as pd

from common import ATTACK_STROKES, MIN_BALLS, MIN_MATCHES, OUT, eb_shrink, read

ROLE = {
    "bat": dict(key="striker_id", name="striker_name", v="bat_v", w="bat_w", ok="bat_ok", proc=["x_runs", "x_dot", "x_out"]),
    "bowl": dict(key="bowler_id", name="bowler_name", v="bowl_v", w="bowl_w", ok="bowl_ok", proc=["x_tot", "x_dot", "x_wkt", "x_err"]),
}
MIN_BASE_PHASE = 30   # baseline balls needed in a phase before it is used
K_PROC = 400          # process rates are shrunk toward the situation average with the weight of 400 balls
SUMS = ["n", "sv", "sw", "n_resc", "sv_resc", "sv_close"]


def add_process(b: pd.DataFrame) -> pd.DataFrame:
    b = b.copy()
    raw = {"x_runs": b.runs, "x_tot": b.tot, "x_dot": b.tot == 0, "x_out": b.striker_out, "x_wkt": b.is_wicket & ~b.runout,
           "x_err": b.wide | b.noball | b.delivery_style.isin(["Full Toss", "Beamer"])}
    keys = [b.innings, b.phase, b.P, b.pace_or_spin]
    for k, s in raw.items():
        s = s.astype(float)
        b[k] = s - s.groupby(keys).transform("mean")
    return b


def match_agg(b: pd.DataFrame, role: str) -> pd.DataFrame:
    r = ROLE[role]
    # Filter first, so the situation averages come from the same balls the player is credited with.
    d = add_process(b[b[r["ok"]] & (b.P | b.live)])
    d["cat"] = np.where(d.P, "P", "B")
    d["v"], d["w"] = d[r["v"]], d[r["w"]]
    job = d.bat_job if role == "bat" else d.bat_job.map({"rescue": "close", "close": "rescue"})
    d["resc"] = (job == "rescue") & d.P
    d["v_resc"], d["v_close"] = d.v * d.resc, d.v * (d.P & ~d.resc)
    spec = dict(n=("v", "size"), sv=("v", "sum"), sw=("w", "sum"), n_resc=("resc", "sum"), sv_resc=("v_resc", "sum"), sv_close=("v_close", "sum"))
    spec.update({p: (p, "sum") for p in r["proc"]})
    a = d.groupby([r["key"], "match_id", "phase", "cat"]).agg(**spec).reset_index()
    return a.rename(columns={r["key"]: "pid"})


def _ratio(g: pd.DataFrame, num: str, den: str = "n") -> pd.DataFrame:
    """Ratio-of-sums estimate per player with match-clustered variance (linearisation)."""
    t = g.groupby("pid")[[num, den]].transform("sum")
    x = t[num] / t[den]
    resid = (g[num] - x * g[den]) ** 2
    out = pd.DataFrame({"x": x.groupby(g.pid).first(), "N": t[den].groupby(g.pid).first(), "M": g.groupby("pid").size(), "ss": resid.groupby(g.pid).sum()})
    out["var"] = out.M / (out.M - 1).clip(lower=1) * out.ss / out.N ** 2
    return out


def components(a: pd.DataFrame, mult: pd.Series | None = None) -> pd.DataFrame:
    """Player components from a match-aggregate table. `mult` = match multiplicities for a bootstrap."""
    proc = [c for c in a.columns if c.startswith("x_")]
    if mult is not None:
        # A match drawn k times is k separate matches: repeating its rows under new ids keeps the
        # match counts and the match-clustered variances right (scaling the sums by k inflates them).
        a = a.merge(mult.rename("k"), left_on="match_id", right_index=True).reset_index(drop=True)
        a = a.loc[a.index.repeat(a.k)]
        a["match_id"] = a.match_id * 1000 + a.groupby(level=0).cumcount()
    P, B = a[a.cat == "P"], a[a.cat == "B"]

    pm = P.groupby(["pid", "match_id"])[SUMS].sum().reset_index()
    rv, rw = _ratio(pm, "sv"), _ratio(pm, "sw")
    c = pd.DataFrame({"n_p": rv.N, "m_p": rv.M, "press_v": rv.x, "var_press": rv["var"], "press_w": rw.x, "var_w": rw["var"]})
    # "matches helped": among matches with 3+ credited pressure balls, the share where those balls added win value
    q = pm[pm.n >= 3].assign(pos=lambda d: d.sw > 0).groupby("pid").pos
    c["consistency"], c["m_helped"], c["m_counted"] = q.mean().reindex(c.index), q.sum().reindex(c.index), q.size().reindex(c.index)
    tot = pm.groupby("pid")[["n", "n_resc", "sv_resc", "sv_close"]].sum()
    c["n_rescue"], c["n_close"] = tot.n_resc, tot.n - tot.n_resc
    c["rescue_v"], c["close_v"] = tot.sv_resc / tot.n_resc.replace(0, np.nan), tot.sv_close / (tot.n - tot.n_resc).replace(0, np.nan)

    # Baseline standardised to the player's own pressure-phase mix.
    pp = P.groupby(["pid", "phase"]).n.sum().rename("nP")
    bp = B.groupby(["pid", "phase"]).agg(nB=("n", "sum"), svB=("sv", "sum"))
    ph = pd.concat([pp, bp], axis=1).dropna()
    ph = ph[ph.nB >= MIN_BASE_PHASE]
    ph["share"] = ph.nP / ph.groupby("pid").nP.transform("sum")
    ph["mean"] = ph.svB / ph.nB
    c["base_v"] = (ph.share * ph["mean"]).groupby("pid").sum().reindex(c.index)
    c["n_b"] = ph.groupby("pid").nB.sum().reindex(c.index)
    bm = B.merge(ph[["share", "mean", "nB"]].reset_index(), on=["pid", "phase"])
    bm["c"] = bm.share / bm.nB * (bm.sv - bm["mean"] * bm.n)
    cm = bm.groupby(["pid", "match_id"]).c.sum()
    g = cm.groupby(level="pid")
    c["var_base"] = (g.size() / (g.size() - 1).clip(lower=1) * (cm ** 2).groupby(level="pid").sum()).reindex(c.index)
    c["lift"], c["var_lift"] = c.press_v - c.base_v, c.var_press + c.var_base
    # Naive baseline for the audit: all non-pressure balls, ignoring which overs they came from.
    nb = B.groupby("pid")[["sv", "n"]].sum()
    c["base_naive"] = (nb.sv / nb.n).reindex(c.index)

    # Process rates over all live balls, shrunk toward the situation average.
    s = a.groupby("pid")[proc + ["n"]].sum()
    for p in proc:
        c[p] = (s[p] / (s.n + K_PROC)).reindex(c.index)
    c["n_live"] = s.n.reindex(c.index)
    return c


def shrink(c: pd.DataFrame, min_n: int = 40, tau: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Empirical-Bayes shrinkage of press_v, press_w, base_v and lift. Returns the between-player variances."""
    c = c.copy()
    # A variance is usable only if it is really positive: one match (or copies of one match in a
    # resample) leaves pure rounding noise (~1e-33), which would otherwise get a near-infinite weight.
    ok = (c.n_p >= min_n) & c.base_v.notna() & (c.var_press > 1e-12) & (c.var_base > 1e-12)
    taus = {}
    for col, var in (("press_v", "var_press"), ("press_w", "var_w"), ("base_v", "var_base"), ("lift", "var_lift")):
        x, v = c.loc[ok, col].values, c.loc[ok, var].values
        mu = np.average(x, weights=1 / v)
        if tau is None:
            post, sd, t2 = eb_shrink(x, v, mu)
        else:
            t2 = tau[col]
            k = t2 / (t2 + v)
            post, sd = mu + k * (x - mu), np.sqrt(k * v)
        c.loc[ok, col + "_post"], c.loc[ok, col + "_sd"] = post, sd
        taus[col] = t2
    return c, taus


def profile(b: pd.DataFrame, role: str) -> pd.DataFrame:
    """Scorecard-style numbers on pressure balls, for the evidence tables."""
    r = ROLE[role]
    d = b[b.P].copy()
    k = r["key"]
    d["err"] = d.wide | d.noball | d.delivery_style.isin(["Full Toss", "Beamer"])
    if role == "bat":
        d = d[~d.wide]
        g = d.groupby(k)
        p = pd.DataFrame({"runs": g.runs.sum(), "balls": g.size(), "outs": g.striker_out.sum(), "boundary_pct": g.runs.apply(lambda s: (s >= 4).mean() * 100),
                          "dot_pct": g.runs.apply(lambda s: (s == 0).mean() * 100), "attack_pct": g.stroke.apply(lambda s: s.isin(ATTACK_STROKES).mean() * 100)})
        p["strike_rate"], p["avg"] = p.runs / p.balls * 100, p.runs / p.outs.replace(0, np.nan)
    else:
        g = d.groupby(k)
        conceded = d.tot - np.where(d.extras_type.isin(["byes", "legbyes"]), d.extras1_runs, 0)
        p = pd.DataFrame({"balls": g.legal.sum(), "runs": conceded.groupby(d[k]).sum(), "wickets": (d.is_wicket & ~d.runout).groupby(d[k]).sum(),
                          "dot_pct": (d.tot == 0).groupby(d[k]).mean() * 100, "error_pct": g.err.mean() * 100, "boundary_pct": g.runs.apply(lambda s: (s >= 4).mean() * 100)})
        p["economy"] = p.runs / p.balls * 6
    allb = b.groupby(k)
    p["name"] = allb[r["name"]].first()
    p["last_match"] = allb.match_date.max()
    p["ipl_share_p"] = (d.level == "IPL").groupby(d[k]).mean()
    p["death_share_p"] = (d.phase == "death").groupby(d[k]).mean()
    return p


def rate(b: pd.DataFrame, role: str) -> tuple[pd.DataFrame, dict]:
    c, taus = shrink(components(match_agg(b, role)))
    c = c.join(profile(b, role))
    c["eligible"] = (c.n_p >= MIN_BALLS) & (c.m_p >= MIN_MATCHES) & c.press_v_post.notna()
    return c, taus


if __name__ == "__main__":
    b = read("balls_val")
    for role in ("bat", "bowl"):
        c, taus = rate(b, role)
        e = c[c.eligible]
        print(f"\n== {role}: {len(e)} eligible ==")
        print("between-player spread (sd, skill units/ball):", {k: round(float(np.sqrt(v)), 4) for k, v in taus.items()})
        print("typical sampling sd of one player's estimate:", {k: round(float(np.sqrt(e[v].median())), 4) for k, v in (("press_v", "var_press"), ("base_v", "var_base"), ("lift", "var_lift"))})
        c.to_csv(OUT / f"components_{role}.csv")
