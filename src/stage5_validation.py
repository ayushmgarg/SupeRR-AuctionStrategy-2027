"""Stage 5: validation. Nothing here changes a player's numbers; it decides how much each number is trusted.

  A. Clock check      - is play slower where our pressure score is high? (independent of outcomes; an
                        association, also checked within the same over of the same innings)
  B. Clutch audit     - is "rising under pressure" repeatable, naive vs role-controlled?
  C. Rating check     - which components predict a player's pressure performance in matches not used to compute them?
  D. Forecast test    - rate on 2022-24 only (venue level, win model, values and opponent quality all
                        refitted on those years alone), score against 2025-26, compare with scorecard baselines
  E. Sensitivity      - does the top 10 survive other pressure thresholds / IPL-only data?
  F. Stability        - resample matches: how often does each player stay in the top 10?
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import MIN_BALLS, MIN_MATCHES, OUT, SEED, TEST_FROM, read
from stage1_match_states import venue_par
from stage2_win_model import flag
from stage2_win_model import run as win_model
from stage3_ball_value import run as value_model
from stage3_ball_value import value
from stage4_player_ratings import ROLE, components, match_agg, profile, shrink

CORE = ["press_v_post", "base_v_post"]
RIDGE = 1.0  # penalty per 100 training rows on standardised features


def feats(role: str) -> list:
    return CORE + ROLE[role]["proc"]


# ---------------------------------------------------------------- A. clock
def clock(b: pd.DataFrame) -> dict:
    b = b.sort_values(["match_id", "innings", "innings_ball_num"]).copy()
    ts = pd.to_datetime(b.delivery_ist_time, utc=True)
    gap = ts.groupby([b.match_id, b.innings]).diff().dt.total_seconds()
    go = b.groupby(["match_id", "innings", "overid"])
    ok = go.cumcount().gt(0) & gap.between(8, 150) & ~go.is_wicket.shift().fillna(False).astype(bool)
    b["gap"] = np.where(ok, gap, np.nan)
    b["gap_rel"] = b.gap - b.groupby("bowler_id").gap.transform("median")  # seconds slower than this bowler's own normal
    band = pd.cut(b.li.rank(pct=True), [0, .5, .85, .95, .99, .997, 1], labels=["bottom 50%", "50-85%", "85-95%", "95-99%", "99-99.7%", "top 0.3%"])
    # Later overs are slower anyway (field changes, tactics). Second version: also net of the average
    # for that over of that innings, so only balls from the same stage of an innings are compared.
    b["gap_same_over"] = b.gap_rel - b.groupby(["innings", "overid"]).gap_rel.transform("mean")
    by_band = b.groupby(band, observed=True).agg(pressure_score=("li", "mean"), extra_seconds=("gap_rel", "mean"), extra_seconds_same_over=("gap_same_over", "mean"), balls=("gap_rel", "count")).round(2)
    b["q5"] = pd.qcut(b.li, 5, labels=False)
    ko = b.groupby(["q5", "knock_out"]).gap_rel.mean().unstack().round(2)
    d = b[b.phase == "death"]
    d = d.assign(err=d.wide | d.noball | d.delivery_style.isin(["Full Toss", "Beamer"]))
    beh = d.groupby("P").agg(bowler_error_pct=("err", lambda s: s.mean() * 100), wicket_pct=("is_wicket", lambda s: s.mean() * 100),
                             boundary_pct=("runs", lambda s: (s >= 4).mean() * 100), balls=("runs", "size")).round(2)
    print("\n== A. clock check: seconds slower than the bowler's own normal, by pressure score (second column: also within the same over of the same innings) ==")
    print(by_band.to_string())
    print("knockout (1) vs other (0) matches at equal ball pressure (5 bands, low to high):\n", ko.to_string())
    print("death overs, pressure balls vs other balls:\n", beh.to_string())
    return dict(by_band=by_band.reset_index().to_dict("records"), knockout=ko.reset_index().to_dict("records"), death_behaviour=beh.reset_index().to_dict("records"))


# ---------------------------------------------------------------- B + C. repeated split-half experiments
def split_rows(a: pd.DataFrame, matches: np.ndarray, reps: int, min_half: int, seed: int) -> pd.DataFrame:
    rng, rows = np.random.default_rng(seed), []
    for rep in range(reps):
        A = set(rng.permutation(matches)[: len(matches) // 2])
        inA = a.match_id.isin(A)
        cA, _ = shrink(components(a[inA]))
        cB = components(a[~inA])
        for c in (cA, cB):
            c["lift_naive"] = c.press_v - c.base_naive
        j = cA.join(cB[["n_p", "press_v", "press_w", "base_v", "lift", "lift_naive"]], rsuffix="_B", how="inner")
        j = j[(j.n_p >= min_half) & (j.n_p_B >= min_half) & j.press_v_post.notna() & j.base_v_B.notna()]
        rows.append(j.assign(rep=rep))
    return pd.concat(rows)


def per_rep_corr(rows: pd.DataFrame, x: str, y: str) -> tuple[float, float, float]:
    r = rows.groupby("rep").apply(lambda d: d[x].corr(d[y]))
    return float(r.mean()), float(r.quantile(.05)), float(r.quantile(.95))


def fit_model(rows: pd.DataFrame, cols: list) -> dict:
    """Ridge regression of unseen-half pressure value on standardised features (weighted by target sample size)."""
    mu, sd = rows[cols].mean(), rows[cols].std()
    X = np.c_[np.ones(len(rows)), ((rows[cols] - mu) / sd).values]
    w = np.sqrt(rows.n_p_B.values)
    pen = RIDGE * len(rows) / 100 * np.eye(X.shape[1])
    pen[0, 0] = 0
    beta = np.linalg.solve((X * w[:, None]).T @ (X * w[:, None]) + pen, (X * w[:, None]).T @ (rows.press_v_B.values * w))
    return dict(cols=cols, mu=mu.tolist(), sd=sd.tolist(), intercept=float(beta[0]), beta=beta[1:].tolist())


def rating(c: pd.DataFrame, m: dict) -> pd.Series:
    """Weighted sum of standardised components."""
    z = (c[m["cols"]] - m["mu"]) / m["sd"]
    return m["intercept"] + z @ np.array(m["beta"])


def equal_model(c: pd.DataFrame) -> dict:
    """ADOPTED RATING: equal weight on the shrunk pressure record and the shrunk usual level, each
    standardised across the players in `c`. Unit: standard deviations above the average of that pool.

    Why equal weights: both components predict unseen pressure performance about equally, and the
    weights a regression picks swing from run to run (the signal is too weak to estimate them), so
    they are fixed rather than fitted.
    """
    d = c[c.press_v_post.notna()]
    sd = d[CORE].std()
    # A component with no measurable spread between players (shrinkage pulled everyone to the same
    # value) carries no information: it gets weight 0 instead of being divided by ~0.
    live = sd > 1e-6
    return dict(cols=CORE, mu=d[CORE].mean().tolist(), sd=sd.where(live, 1.0).tolist(), intercept=0.0, beta=(0.5 * live).tolist())


def pool(c: pd.DataFrame) -> pd.Series:
    """Eligibility is sample size only: no filter on nationality, league or when the player last appeared."""
    return (c.n_p >= MIN_BALLS) & (c.m_p >= MIN_MATCHES) & c.press_v_post.notna()


def audit(a: pd.DataFrame, role: str, reps: int = 100) -> dict:
    rows = split_rows(a, a.match_id.unique(), reps, 50, SEED)
    n = rows.groupby("rep").size().mean()
    rel = {lab: per_rep_corr(rows, x, x + "_B") for lab, x in (("naive lift (pressure minus all other balls)", "lift_naive"), ("role-controlled lift (same overs)", "lift"),
                                                                ("pressure value", "press_v"), ("usual value, same overs", "base_v"), ("pressure wins added", "press_w"))}
    print(f"\n== B. {role}: repeatability across random halves of matches ({reps} splits, ~{n:.0f} players each) ==")
    for k, (m, lo, hi) in rel.items():
        print(f"  {k:46s} r = {m:+.2f}  [{lo:+.2f}, {hi:+.2f}]")

    odd, even = rows[rows.rep % 2 == 1], rows[rows.rep % 2 == 0].copy()
    cands = {"raw pressure average": ["press_v"], "shrunk pressure record": ["press_v_post"], "usual level only": ["base_v_post"],
             "fitted weights: record + usual": CORE, "process only": ROLE[role]["proc"], "fitted weights: record + usual + process": feats(role)}
    pred = {}
    print(f"== C. {role}: predicting pressure performance in the unseen half ==")
    for lab, cols in cands.items():
        even["p"] = rating(even, fit_model(odd, cols))
        pred[lab] = per_rep_corr(even, "p", "press_v_B")
    even["p"] = pd.concat([rating(d, equal_model(d)) for _, d in even.groupby("rep")])
    pred["EQUAL WEIGHTS: record + usual (adopted)"] = per_rep_corr(even, "p", "press_v_B")
    for lab, (m, lo, hi) in pred.items():
        print(f"  {lab:46s} r = {m:+.2f}  [{lo:+.2f}, {hi:+.2f}]")
    # How unstable fitted weights are: refit on each fifth of the splits.
    shares = []
    for k in range(5):
        beta = np.abs(fit_model(rows[rows.rep % 5 == k], CORE)["beta"])
        shares.append(float(beta[0] / beta.sum() * 100))
    print(f"  weight a regression would put on the pressure record, five refits: {', '.join(f'{x:.0f}%' for x in shares)}")
    return dict(repeatability=rel, prediction=pred, fitted_weight_on_record_pct=shares, players_per_split=float(n))


# ---------------------------------------------------------------- D. forecast test
def scorecard(b: pd.DataFrame, role: str) -> pd.DataFrame:
    k = ROLE[role]["key"]
    if role == "bat":
        d = b[~b.wide]
        f = lambda x: x.groupby(k).runs.sum() / x.groupby(k).size() * 100
        return pd.DataFrame({"death strike rate": f(d[d.phase == "death"]).where(d[d.phase == "death"].groupby(k).size() >= 60), "overall strike rate": f(d),
                             "pressure strike rate": f(d[d.P]), "batting average": d.groupby(k).runs.sum() / d.groupby(k).striker_out.sum().replace(0, np.nan)})
    f = lambda x: -(x.groupby(k).tot.sum() / x.groupby(k).legal.sum() * 6)
    return pd.DataFrame({"death economy": f(b[b.phase == "death"]).where(b[b.phase == "death"].groupby(k).legal.sum() >= 60), "overall economy": f(b), "pressure economy": f(b[b.P]),
                         "bowling strike rate": -(b.groupby(k).legal.sum() / (b.is_wicket & ~b.runout).groupby(b[k]).sum().replace(0, np.nan))})


def period_values(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Everything a rating depends on, rebuilt from the matches of one period alone: venue scoring
    level, win model, pressure score and flag, outcome values and opponent quality. (Team strength is
    a pre-match Elo, which only ever looks backwards.)"""
    bt = read("balls")
    bt = bt[(bt.match_date >= (start or "0")) & (bt.match_date < (end or "9"))].reset_index(drop=True)
    bt["par"] = venue_par(bt)
    w, _ = flag(win_model(bt, repeats=4))
    return value_model(w, verbose=False)[0]


def boot_ci(x, y, reps: int = 1000) -> tuple[float, float]:
    """90% interval for a rank correlation, resampling players."""
    rng, x, y = np.random.default_rng(SEED + 3), np.asarray(x, float), np.asarray(y, float)
    r = [spearmanr(x[i], y[i])[0] for i in (rng.integers(0, len(x), len(x)) for _ in range(reps))]
    return float(np.nanquantile(r, .05)), float(np.nanquantile(r, .95))


def forecast(b: pd.DataFrame, a: pd.DataFrame, role: str, train_val: pd.DataFrame, test_val: pd.DataFrame) -> dict:
    test_m = b[b.match_date >= TEST_FROM].match_id.unique()
    cS = components(a[a.match_id.isin(test_m)])[["n_p", "press_v", "press_w"]]   # what actually happened in 2025-26
    tr = match_agg(train_val, role)
    cT, _ = shrink(components(tr))
    cT = cT[(cT.n_p >= 60) & cT.press_v_post.notna()]
    rows = split_rows(tr, tr.match_id.unique(), 60, 40, SEED + 1)
    j = cT.assign(rating=rating(cT, equal_model(cT)), rating_fit=rating(cT, fit_model(rows, CORE)), rating_proc=rating(cT, fit_model(rows, feats(role))))
    j = j.join(cS, rsuffix="_test", how="inner")
    j = j[j.n_p_test >= 40].join(scorecard(train_val, role))
    preds = ["rating", "rating_fit", "rating_proc", "press_v"] + [c for c in j.columns if "strike rate" in c or "economy" in c or "average" in c]
    labels = {"rating": "PRESSURE-PROOF RATING", "rating_fit": "fitted weights (tested)", "rating_proc": "fitted + process (tested)", "press_v": "raw pressure average"}
    out = {}
    print(f"\n== D. {role}: rated on 2022-24 only (everything refitted on those years), scored on 2025-26 ({len(j)} players) ==")
    for p in preds:
        d = j[j[p].notna()]
        rv, rw = spearmanr(d[p], d.press_v_test)[0], spearmanr(d[p], d.press_w_test)[0]
        (lo, hi), (lo_w, hi_w) = boot_ci(d[p], d.press_v_test), boot_ci(d[p], d.press_w_test)
        out[labels.get(p, p)] = dict(rank_corr_value=float(rv), ci90=[lo, hi], rank_corr_wins=float(rw), ci90_wins=[lo_w, hi_w], n=int(len(d)))
        print(f"  {labels.get(p, p):26s} rank corr with 2025-26 pressure value {rv:+.2f} [{lo:+.2f}, {hi:+.2f}] | with wins added {rw:+.2f} [{lo_w:+.2f}, {hi_w:+.2f}] (n={len(d)})")
    print("  (intervals resample the players in the test; they do not include model or season uncertainty)")
    # Strictest version: the 2025-26 outcome is also measured by models that saw 2025-26 matches only,
    # so the two sides of the comparison share no fitted quantity at all.
    cX = components(match_agg(test_val, role))[["n_p", "press_v"]]
    x = j[["rating"]].join(cX, how="inner")
    x = x[x.n_p >= 40]
    out["rating_vs_outcome_rebuilt_from_test_period_only"] = dict(rank_corr_value=float(spearmanr(x.rating, x.press_v)[0]), ci90=list(boot_ci(x.rating, x.press_v)), n=int(len(x)))
    r = out["rating_vs_outcome_rebuilt_from_test_period_only"]
    print(f"  (rating vs a 2025-26 outcome rebuilt from 2025-26 matches alone: {r['rank_corr_value']:+.2f} [{r['ci90'][0]:+.2f}, {r['ci90'][1]:+.2f}], n={r['n']})")
    # For comparison: the same test with the main pipeline's values, which were fitted on all years.
    cA, _ = shrink(components(a[~a.match_id.isin(test_m)]))
    cA = cA[(cA.n_p >= 60) & cA.press_v_post.notna()]
    k = cA.assign(rating=rating(cA, equal_model(cA))).join(cS, rsuffix="_test", how="inner")
    k = k[k.n_p_test >= 40]
    out["rating_with_values_fitted_on_all_years"] = dict(rank_corr_value=float(spearmanr(k.rating, k.press_v_test)[0]), n=int(len(k)))
    print(f"  (same rating with values fitted on all years: {out['rating_with_values_fitted_on_all_years']['rank_corr_value']:+.2f}, n={len(k)})")
    return out


# ---------------------------------------------------------------- E + F. ranking, stability, sensitivity
def stability(a: pd.DataFrame, c: pd.DataFrame, w: dict, tau: dict, reps: int) -> pd.DataFrame:
    """Resample matches. Gives each player's rating uncertainty and how often he makes the top 10 / 20
    UNDER THE FINAL RULE (conservative rating), not under the raw rating."""
    rng, matches, elig = np.random.default_rng(SEED + 2), a.match_id.unique(), c.index[c.final]
    vals = []
    for _ in range(reps):
        mult = pd.Series(rng.choice(matches, len(matches))).value_counts()
        cb, _ = shrink(components(a, mult), tau=tau)
        vals.append(rating(cb, w).reindex(elig))
    V = pd.concat(vals, axis=1)
    sd = V.std(axis=1)
    R = V.sub(0.674 * sd, axis=0).rank(ascending=False)
    return pd.DataFrame({"rating_sd": sd, "p_top10": (R <= 10).mean(axis=1), "p_top20": (R <= 20).mean(axis=1), "rank_lo": R.quantile(.1, axis=1), "rank_hi": R.quantile(.9, axis=1)})


def ranked(b: pd.DataFrame, role: str, reps: int = 300) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """The whole ranking for one role: components, shrinkage, equal-weight rating standardised on the
    eligible pool, and the conservative rating (the level we are 75% sure the player is above)."""
    a = match_agg(b, role)
    c, taus = shrink(components(a))
    w = equal_model(c[pool(c)])
    c = c.join(profile(b, role)[["name", "last_match"]])
    c["rating"], c["final"] = rating(c, w), pool(c)
    c = c.join(stability(a, c, w, taus, reps))
    c["rating_safe"] = c.rating - 0.674 * c.rating_sd
    return c, a, w, taus


def top10(c: pd.DataFrame) -> list:
    return c[c.final].nlargest(10, "rating_safe").index.tolist()


def sensitivity(b: pd.DataFrame, role: str, full: pd.DataFrame) -> dict:
    """Redo the whole ranking (same rule, conservative rating) under other definitions; count shared top-10 names."""
    base, out = top10(full), {}
    variants = {"winnable 10-90%": dict(lo=.10, hi=.90, q=.85), "winnable 20-80%": dict(lo=.20, hi=.80, q=.85), "tough top 10%": dict(lo=.15, hi=.85, q=.90),
                "tough top 25%": dict(lo=.15, hi=.85, q=.75), "IPL matches only": dict(lo=.15, hi=.85, q=.85, ipl=True)}
    for lab, v in variants.items():
        d = b.assign(P=b.wp.between(v["lo"], v["hi"]) & (b.li >= b.li.quantile(v["q"])))
        # dwp is centred within pressure / other balls, so a new flag needs a new centring and new values
        # (opponent qualities are kept as fitted).
        d = value(d.assign(dwp=d.dwp_raw - d.groupby(["innings", "phase", "P"]).dwp_raw.transform("mean")))
        if v.get("ipl"):
            d = d[d.level == "IPL"]
        c = ranked(d, role, reps=60)[0]
        out[lab] = dict(overlap=len(set(top10(c)) & set(base)), eligible=int(c.final.sum()))
    # How much does the opponent adjustment matter? Redo the ranking with it switched off.
    c = ranked(value(b.assign(opp_bowl_q=0.0, opp_bat_q=0.0)), role, reps=60)[0]
    out["no opponent adjustment"] = dict(overlap=len(set(top10(c)) & set(base)), eligible=int(c.final.sum()))
    # How much of the list is the pressure record alone? Rank on the conservative pressure record only.
    e = full[full.final]
    only = (e.press_v_post - 0.674 * e.press_v_sd).nlargest(10).index
    out["pressure record only"] = dict(overlap=len(set(only) & set(base)), eligible=int(len(e)))
    return out


if __name__ == "__main__":
    b = read("balls_val")
    res = {"clock": clock(b)}
    train_val, test_val = period_values(end=TEST_FROM), period_values(start=TEST_FROM)
    for role in ("bat", "bowl"):
        full, a, w, taus = ranked(b, role)
        aud = audit(a, role)
        sens = sensitivity(b, role, full)
        print(f"== E. {role}: top-10 names shared with the default definition (same ranking rule throughout) ==")
        for k, v in sens.items():
            print(f"  {k:22s} {v['overlap']}/10 same players ({v['eligible']} eligible)")
        e = full[full.final]
        t = e.loc[top10(full)]
        above = dict(pressure_record_above_pool_average=int((t.press_v_post > e.press_v_post.mean()).sum()), usual_level_above_pool_average=int((t.base_v_post > e.base_v_post.mean()).sum()))
        print(f"  of the top 10: {above['pressure_record_above_pool_average']} have a pressure record above the eligible-pool average, {above['usual_level_above_pool_average']} a usual level above it")
        print(f"== F. {role}: top 16 by conservative rating; p_top10 = share of 300 match resamples in which the player makes the top 10 under that same rule ==")
        cols = ["name", "n_p", "m_p", "rating", "rating_safe", "press_v_post", "base_v_post", "lift_post", "press_w", "consistency", "p_top10", "rank_lo", "rank_hi"]
        print(e.sort_values("rating_safe", ascending=False)[cols].head(16).round(3).to_string())
        res[role] = dict(audit=aud, forecast=forecast(b, a, role, train_val, test_val), sensitivity=sens, top10_components=above, taus={k: float(v) for k, v in taus.items()})
        full.to_csv(OUT / f"ratings_{role}.csv")
        json.dump(w, open(OUT / f"model_{role}.json", "w"), indent=1)
    json.dump(res, open(OUT / "validation.json", "w"), indent=1, default=float)
    print("\nsaved outputs/validation.json, ratings_*.csv, model_*.json")
