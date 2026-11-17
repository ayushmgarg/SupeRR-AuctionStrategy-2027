"""Stage 3: value of every ball for the batter and the bowler, adjusted for who they were up against.

Raw value of a ball = change in the batting side's win probability (already measured against the
average outcome for that match state). Two units are kept:
  * win units  (dwp)        - what the ball did to the match; big on high-pressure balls
  * skill units (v)         - the ball's outcome valued at its average worth in that kind of situation
                              (linear weights on dwp/li), comparable across situations and far less noisy

Opponent quality: a ridge regression estimates every batter's and bowler's skill in each phase
jointly (batter effect minus bowler effect), so runs against a weak attack are worth less. The
ridge penalty is the Gaussian prior of a hierarchical model; its strength is chosen by
match-grouped cross-validation.
"""
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from common import SEED, read, write

ALPHAS = [100, 300, 1000, 3000, 10000]


def design(d: pd.DataFrame, bat_ix: dict, bowl_ix: dict):
    n = len(d)
    rb = np.flatnonzero(d.bat_ok.values)
    cb = np.array([bat_ix[k] for k in zip(d.striker_id.values[rb], d.phase.values[rb])])
    rw = np.flatnonzero(d.bowl_ok.values)
    cw = np.array([bowl_ix[k] for k in zip(d.bowler_id.values[rw], d.phase.values[rw])]) + len(bat_ix)
    vals = np.r_[np.ones(len(rb)), -np.ones(len(rw))]
    return sparse.csr_matrix((vals, (np.r_[rb, rw], np.r_[cb, cw])), shape=(n, len(bat_ix) + len(bowl_ix)))


def value(b: pd.DataFrame) -> pd.DataFrame:
    """Value columns for batter and bowler from dwp, li and the opponent qualities on the frame."""
    exact = 100 * b.dwp / b.li.clip(lower=0.25)
    # Linear weights (as in baseball's wOBA): every outcome - dot, single, four, six, wicket, wide... -
    # is valued at its AVERAGE worth in that kind of situation (innings, phase, pressure or not),
    # not at its exact swing in that exact match state. A player controls the outcome, not the
    # state-specific remainder, which is mostly model noise. Tested: this roughly doubles how well
    # a rating predicts the same player's pressure performance in unseen matches.
    ext = b.extras_type.fillna("none")
    worth = exact.groupby([b.innings, b.phase, b.P, b.runs, b.is_wicket, ext]).mean()
    # Byes and leg-byes are neither the batter's runs nor the bowler's: such a ball is credited to
    # both as the same ball without the extras (normally a dot).
    credited = ext.where(~ext.isin(["byes", "legbyes"]), "none")
    own = worth.reindex(pd.MultiIndex.from_arrays([b.innings, b.phase, b.P, b.runs, b.is_wicket, ext])).values
    b["v"] = pd.Series(worth.reindex(pd.MultiIndex.from_arrays([b.innings, b.phase, b.P, b.runs, b.is_wicket, credited])).values, index=b.index).fillna(pd.Series(own, index=b.index))
    b["bat_v"], b["bowl_v"] = b.v + b.opp_bowl_q, -b.v + b.opp_bat_q   # skill units
    b["bat_w"] = 100 * b.dwp + b.opp_bowl_q * b.li                      # win units (x100)
    b["bowl_w"] = -100 * b.dwp + b.opp_bat_q * b.li
    return b


def run(b: pd.DataFrame, alpha: float | None = None, verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    b = value(b.assign(opp_bowl_q=0.0, opp_bat_q=0.0))  # unadjusted v first; the ridge below supplies opponent quality
    # Credit rules: a wide is not the batter's doing; a run-out is not the bowler's; a batter is
    # not charged when his partner is run out; byes and leg-byes count for nobody (see value()).
    b["bat_ok"] = ~b.wide & ~(b.is_wicket & ~b.striker_out)
    b["bowl_ok"] = ~b.runout

    d = b[b.live & (b.bat_ok | b.bowl_ok)]
    bat_ix = {k: i for i, k in enumerate(sorted(set(zip(d.striker_id, d.phase))))}
    bowl_ix = {k: i for i, k in enumerate(sorted(set(zip(d.bowler_id, d.phase))))}
    X, y = design(d, bat_ix, bowl_ix), d.v.values

    if alpha is None:
        cv = {}
        for a in ALPHAS:
            err, err0 = [], []
            for tr, te in GroupKFold(5).split(X, groups=d.match_id):
                m = Ridge(alpha=a, solver="sparse_cg", random_state=SEED).fit(X[tr], y[tr])
                err.append(np.mean((y[te] - m.predict(X[te])) ** 2))
                err0.append(np.mean((y[te] - y[tr].mean()) ** 2))
            cv[a] = 1 - np.mean(err) / np.mean(err0)
        alpha = max(cv, key=cv.get)
        if verbose:
            print("out-of-sample R2 by ridge penalty:", {a: round(r, 5) for a, r in cv.items()}, "-> chosen", alpha)

    m = Ridge(alpha=alpha, solver="sparse_cg", random_state=SEED).fit(X, y)
    q = pd.concat([
        pd.DataFrame([(p, ph, m.coef_[i]) for (p, ph), i in bat_ix.items()], columns=["player_id", "phase", "q"]).assign(role="bat"),
        pd.DataFrame([(p, ph, m.coef_[i + len(bat_ix)]) for (p, ph), i in bowl_ix.items()], columns=["player_id", "phase", "q"]).assign(role="bowl"),
    ])
    qb = q[q.role == "bowl"].set_index(["player_id", "phase"]).q
    qs = q[q.role == "bat"].set_index(["player_id", "phase"]).q
    opp_bowl = pd.Series(list(zip(b.bowler_id, b.phase))).map(qb).fillna(0).values   # quality of the bowler faced
    opp_bat = pd.Series(list(zip(b.striker_id, b.phase))).map(qs).fillna(0).values   # quality of the batter bowled to
    b["opp_bowl_q"], b["opp_bat_q"] = opp_bowl, opp_bat
    return value(b), q


if __name__ == "__main__":
    b, q = run(read("balls_wp"))
    names = pd.concat([b[["striker_id", "striker_name"]].set_axis(["player_id", "name"], axis=1),
                       b[["bowler_id", "bowler_name"]].set_axis(["player_id", "name"], axis=1)]).drop_duplicates("player_id").set_index("player_id").name
    q["name"] = q.player_id.map(names)
    for role, ph in (("bowl", "death"), ("bowl", "pp"), ("bat", "death"), ("bat", "pp")):
        s = q[(q.role == role) & (q.phase == ph)].sort_values("q", ascending=False)
        print(f"strongest {role} {ph}: " + ", ".join(f"{n} ({v:+.2f})" for n, v in zip(s.name.head(6), s.q.head(6))))
    lev = b[b.live].groupby("level").agg(bowl_q_faced=("opp_bowl_q", "mean"), bat_q_faced=("opp_bat_q", "mean")).round(3)
    print("average opponent quality by competition level (skill units per ball):\n", lev.to_string())
    write(b, "balls_val")
    q.to_csv(read.__globals__["OUT"] / "opponent_quality.csv", index=False)
    print("saved outputs/balls_val.parquet, outputs/opponent_quality.csv")
