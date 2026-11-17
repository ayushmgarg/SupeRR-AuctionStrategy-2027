"""Stage 2: win-probability model, pressure score per ball, pressure flag.

Two gradient-boosted classifiers (setting a total, chasing a target), trained with match-grouped
5-fold cross-validation, repeated over several random splits and averaged. Every probability used downstream is out-of-fold: the model that scores a
ball never saw any ball from that match.

Pressure score (leverage) = expected absolute swing in win probability over the possible outcomes
of the ball (including wides, no-balls, byes and leg-byes), divided by the average swing of all balls.
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from common import LIVE_HI, LIVE_LO, OUT, SEED, TOUGH_Q, WIN_HI, WIN_LO, read, write

F1 = ["score", "bl", "wk", "elo_diff", "par"]   # setting
F2 = ["need", "bl", "wk", "elo_diff", "par"]    # chasing
MONO1 = [1, 1, -1, 1, -1]    # more runs / balls in hand / stronger team help; wickets lost and a high-scoring venue hurt
MONO2 = [-1, 1, -1, 1, 1]
PARAMS = dict(objective="binary", n_estimators=400, learning_rate=0.03, num_leaves=12, min_child_samples=300,
              subsample=0.8, subsample_freq=1, random_state=SEED, verbose=-1)
N_REPEATS = 10  # random 5-fold splits averaged for the out-of-fold win probability


def fit(d, feats, mono):
    d = d[d.y != 0.5]  # ties carry no win/loss label
    return lgb.LGBMClassifier(**PARAMS, monotone_constraints=mono).fit(d[feats], d.y.astype(int))


def wp_chase(m2, need, bl, wk, elo, par):
    need, bl, wk = (np.asarray(v, float) for v in (need, bl, wk))
    X = pd.DataFrame({"need": np.maximum(need, 1), "bl": np.maximum(bl, 1), "wk": np.minimum(wk, 9), "elo_diff": elo, "par": par})
    p = m2.predict_proba(X[m2.feature_name_])[:, 1]
    p = np.where(need <= 0, 1.0, p)
    done = (need > 0) & ((bl <= 0) | (wk >= 10))  # chase over short of the target: lost, or tied if scores are level
    return np.where(done, np.where(need == 1, 0.5, 0.0), p)


def wp_set(m1, m2, score, bl, wk, elo, par):
    score, bl, wk, elo = (np.asarray(v, float) for v in (score, bl, wk, elo))
    X = pd.DataFrame({"score": score, "bl": np.maximum(bl, 1), "wk": np.minimum(wk, 9), "elo_diff": elo, "par": par})
    p = m1.predict_proba(X[m1.feature_name_])[:, 1]
    over = (bl <= 0) | (wk >= 10)  # innings finished: hand over to the chase model at ball one
    n = len(score)
    p_end = 1 - wp_chase(m2, score + 1, np.full(n, 120), np.zeros(n), -elo, par)
    return np.where(over, p_end, p)


def outcome_table(b: pd.DataFrame) -> tuple[list, pd.DataFrame]:
    """What can happen on the next ball, taken from the data: (runs added to the total, wicket, ball
    used up), with probabilities by innings and phase. Extras are included - a wide or no-ball adds
    runs without using up a ball. Outcomes rarer than 1 in 1,000 are dropped and the rest renormalised.
    """
    o = pd.DataFrame({"innings": b.innings, "ph5": b.ph5, "r": b.tot.clip(upper=7), "w": b.is_wicket.astype(int), "lg": b.legal.astype(int)})
    overall = o.groupby(["r", "w", "lg"]).size() / len(o)
    classes = [k for k, v in overall.items() if v >= 0.001]
    t = o.groupby(["innings", "ph5", "r", "w", "lg"]).size().unstack(["r", "w", "lg"]).fillna(0)
    t = t.reindex(columns=pd.MultiIndex.from_tuples(classes, names=["r", "w", "lg"]), fill_value=0)
    return classes, t.div(t.sum(axis=1), axis=0)


def flag(b: pd.DataFrame, thr: float | None = None) -> tuple[pd.DataFrame, float]:
    """Pressure flag, live flag, rescue / close tag and centred ball values. Returns the frame and the swing threshold."""
    b = b.copy()
    thr = float(b.li.quantile(TOUGH_Q)) if thr is None else thr
    b["P"] = b.wp.between(WIN_LO, WIN_HI) & (b.li >= thr)
    b["live"] = b.wp.between(LIVE_LO, LIVE_HI)
    b["bat_job"] = np.where(b.wp < 0.5, "rescue", "close")
    # Remove any leftover model drift so the average ball in each context is worth exactly zero.
    # The uncentred change is kept because the centring depends on P (the sensitivity runs redo it).
    b["dwp_raw"] = b.dwp
    b["dwp"] = b.dwp - b.groupby(["innings", "phase", "P"]).dwp.transform("mean")
    return b, thr


def run(b: pd.DataFrame, feats1=F1, feats2=F2, mono1=MONO1, mono2=MONO2, repeats: int = N_REPEATS) -> pd.DataFrame:
    """Out-of-fold win probability, averaged over `repeats` different random 5-fold splits of matches.

    One split leaves fold-to-fold model noise in every ball's value; averaging several splits removes
    most of it, so the downstream ratings do not depend on how the matches happened to be folded.
    """
    b = b.reset_index(drop=True)
    b["ph5"] = pd.cut(b.overid, [0, 6, 11, 15, 18, 20], labels=False)
    outcomes, probs = outcome_table(b)

    matches, rng, n = np.sort(b.match_id.unique()), np.random.default_rng(SEED), len(b)
    wp, wp_sq, swing, wp_end = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    for _ in range(repeats):
        fold = b.match_id.map(pd.Series(rng.permutation(len(matches)) % 5, index=matches)).values
        for f in range(5):
            d = b[fold != f]
            m1, m2 = fit(d[d.innings == 1], feats1, mono1), fit(d[d.innings == 2], feats2, mono2)
            for inn in (1, 2):
                pos = np.flatnonzero((fold == f) & (b.innings.values == inn))
                s = b.iloc[pos]
                if inn == 1:
                    now = wp_set(m1, m2, s.score, s.bl, s.wk, s.elo_diff, s.par)
                    nxt = [wp_set(m1, m2, s.score + r, s.bl - lg, s.wk + w, s.elo_diff, s.par) for r, w, lg in outcomes]
                    wp_end[pos] += 1 - wp_chase(m2, s.score + s.tot + 1, np.full(len(s), 120), np.zeros(len(s)), -s.elo_diff.values, s.par)
                else:
                    now = wp_chase(m2, s.need, s.bl, s.wk, s.elo_diff, s.par)
                    nxt = [wp_chase(m2, s.need - r, s.bl - lg, s.wk + w, s.elo_diff, s.par) for r, w, lg in outcomes]
                pr = probs.loc[inn].reindex(s.ph5.values).values
                wp[pos] += now
                wp_sq[pos] += now ** 2
                swing[pos] += sum(pr[:, k] * np.abs(nxt[k] - now) for k in range(len(outcomes)))

    b["wp"], b["swing"] = wp / repeats, swing / repeats
    b["wp_noise"] = np.sqrt(np.clip(wp_sq / repeats - b.wp ** 2, 0, None))   # split-to-split sd of one model's estimate
    b["li"] = b.swing / b.swing.mean()
    nxt = b.groupby(["match_id", "innings"]).wp.shift(-1)
    terminal = np.where(b.innings == 1, wp_end / repeats, b.y)
    b["dwp"] = np.where(nxt.isna(), terminal, nxt) - b.wp
    return b.drop(columns=["ph5"])


def metrics(b: pd.DataFrame) -> dict:
    out = {}
    for inn, name in ((1, "setting"), (2, "chasing")):
        s = b[(b.innings == inn) & (b.y != 0.5)]
        out[name] = dict(auc=roc_auc_score(s.y, s.wp), brier=brier_score_loss(s.y, s.wp), logloss=log_loss(s.y, s.wp.clip(1e-4, 1 - 1e-4)))
    return out


if __name__ == "__main__":
    b0 = read("balls")
    base = run(b0, ["score", "bl", "wk"], ["need", "bl", "wk"], [1, 1, -1], [-1, 1, -1], repeats=1)
    b = run(b0)
    m_base, m_full = metrics(base), metrics(b)
    print("== out-of-fold accuracy (Brier: 0.25 = coin flip, lower is better) ==")
    for k in m_full:
        print(f"{k:8s} state only: AUC {m_base[k]['auc']:.3f} Brier {m_base[k]['brier']:.4f}  |  + team strength & venue: AUC {m_full[k]['auc']:.3f} Brier {m_full[k]['brier']:.4f}")
    s = b[b.y != 0.5]
    cal = s.groupby(pd.cut(s.wp, np.linspace(0, 1, 11)), observed=True).agg(predicted=("wp", "mean"), actual=("y", "mean"), balls=("y", "size")).round(3)
    print(cal.to_string())
    print(f"mean ball-to-ball change in win probability (should be ~0): {b.dwp.mean():+.5f}")
    print(f"model noise: one 5-fold split moves a win probability by {b.wp_noise.mean() * 100:.2f} points on average; {N_REPEATS} splits are averaged")

    b, thr = flag(b)
    print(f"\npressure threshold: swing >= {thr:.2f}x average | pressure balls {int(b.P.sum()):,} ({b.P.mean() * 100:.1f}%) in {b[b.P].match_id.nunique()} of {b.match_id.nunique()} matches")
    print(f"decided-match balls excluded from baselines (win chance outside 5-95%): {(~b.live).mean() * 100:.1f}%")
    print("pressure balls by innings:", b[b.P].innings.value_counts().to_dict(), "| by phase:", b[b.P].phase.value_counts().to_dict())
    print("pressure score by phase / innings:\n", b.pivot_table(index="innings", columns="phase", values="li", aggfunc="mean")[["pp", "mid", "death"]].round(2).to_string())
    for key, lab in (("striker_id", "batters"), ("bowler_id", "bowlers")):
        n = b[b.P].groupby(key).size()
        print(f"{lab} with >=100 pressure balls: {(n >= 100).sum()} | >=150: {(n >= 150).sum()}")
    write(b, "balls_wp")
    json.dump(dict(model=m_full, state_only=m_base, pressure_threshold=float(thr), pressure_balls=int(b.P.sum()),
                   wp_noise_single_split=float(b.wp_noise.mean()), repeats=N_REPEATS,
                   calibration=cal.reset_index(drop=True).to_dict("records")), open(OUT / "metrics_winprob.json", "w"), indent=1)
    print("saved outputs/balls_wp.parquet, outputs/metrics_winprob.json")
