"""Stage 1: turn the raw ball table into one row per ball with the match state BEFORE that ball.

Adds: score, wickets down, balls left, runs needed, pre-match team strength (Elo), venue scoring
level, and the flags later stages need. Runs data-quality checks and prints them.
"""
import re

import numpy as np
import pandas as pd

from common import load, phase_of, write

TIE = r"super over|tied|one over eliminator"


def match_table() -> pd.DataFrame:
    m = load("matches").merge(load("competitions"), on="comp_id", how="left")
    m["tie"] = m.result.str.contains(TIE, case=False, na=False)
    # The revised-overs fields are not always filled: also trust a D/L or DLS note in the result text.
    m["revised"] = (m[["first_inn_revised_overs", "second_inn_revised_overs", "revised_target"]].notna().any(axis=1)
                    | m.result.str.contains(r"D/L|DLS", case=False, na=False))
    m["no_result"] = m.win_team_id.isna() & ~m.tie
    m["level"] = np.select([m.comp_name.str.startswith("IPL"), m.match_format.eq("T20I")], ["IPL", "T20I"], "Other")
    m = m.sort_values(["match_date", "match_id"]).reset_index(drop=True)

    # Pre-match Elo: team strength known before the first ball, updated after each result.
    elo, home, away = {}, [], []
    for r in m.itertuples():
        h, a = elo.get(r.home_team_id, 1500.0), elo.get(r.away_team_id, 1500.0)
        home.append(h)
        away.append(a)
        if pd.notna(r.win_team_id):
            exp_h = 1 / (1 + 10 ** ((a - h) / 400))
            res_h = 1.0 if r.win_team_id == r.home_team_id else 0.0
            elo[r.home_team_id] = h + 20 * (res_h - exp_h)
            elo[r.away_team_id] = a - 20 * (res_h - exp_h)
    m["elo_home"], m["elo_away"] = home, away
    return m


def venue_par(b: pd.DataFrame) -> pd.Series:
    """Venue scoring level: runs per legal ball at the venue over the matches in `b`, leaving the
    current match out, shrunk toward the overall rate with the weight of ~10 innings."""
    vm = b.groupby(["venue_id", "match_id"]).agg(r=("tot", "sum"), n=("legal", "sum")).reset_index()
    vt = vm.groupby("venue_id")[["r", "n"]].transform("sum")
    glob, k = b.tot.sum() / b.legal.sum(), 1200
    vm["par"] = (vt.r - vm.r + k * glob) / (vt.n - vm.n + k)
    return b.match_id.map(vm.set_index("match_id").par)


def build() -> pd.DataFrame:
    m = match_table()
    keep = m[~m.no_result & ~m.revised]
    b = load("ball_by_ball")
    b = b[(b.innings <= 2) & b.match_id.isin(keep.match_id)].sort_values(["match_id", "innings", "innings_ball_num"]).reset_index(drop=True)

    b["tot"] = b.runs + b.extras1_runs + b.extras_2_runs
    b["wide"] = b.extras_type.eq("wide")
    b["noball"] = b.extras_type.isin(["Overstep", "other"])
    b["legal"] = ~(b.wide | b.noball)
    b["runout"] = b.dismissal_type.eq("RunOut")
    b["striker_out"] = b.is_wicket & (b.dismissed_player_id == b.striker_id)

    g = b.groupby(["match_id", "innings"])
    b["score"] = g.tot.cumsum() - b.tot
    b["wk"] = g.is_wicket.cumsum() - b.is_wicket
    b["bl"] = 120 - (g.legal.cumsum() - b.legal)
    b["target"] = b.match_id.map(b[b.innings == 1].groupby("match_id").tot.sum() + 1)
    b["need"] = np.where(b.innings == 2, b.target - b.score, np.nan)
    b["phase"] = phase_of(b.overid)

    # Rain-shortened matches with no revised-overs fields and no D/L note: an innings stops with an
    # over or more unbowled, wickets in hand (9 down counts as all out) and the target not reached.
    end = b.groupby(["match_id", "innings"]).tail(1)
    cut = end[(end.bl - end.legal >= 6) & (end.wk + end.is_wicket < 9) & ~(end.need - end.tot <= 0)].match_id.unique()
    b = b[~b.match_id.isin(cut)]
    print(f"dropped {len(cut)} more matches cut short by rain with no revised-overs data")

    cols = ["match_id", "match_date", "level", "comp_name", "season", "knock_out", "stage", "venue_id", "tie",
            "win_team_id", "home_team_id", "elo_home", "elo_away"]
    b = b.merge(m[cols], on="match_id")
    bat_home = b.batting_team_id == b.home_team_id
    b["elo_diff"] = np.where(bat_home, b.elo_home - b.elo_away, b.elo_away - b.elo_home)
    b["y"] = np.where(b.tie, 0.5, (b.win_team_id == b.batting_team_id).astype(float))

    b["par"] = venue_par(b)

    qa(b, m)
    b = b[(b.bl > 0) & ((b.innings == 1) | (b.need > 0))].reset_index(drop=True)
    return b.drop(columns=["home_team_id", "elo_home", "elo_away", "win_team_id"])


def qa(b: pd.DataFrame, m: pd.DataFrame) -> None:
    print(f"matches: {len(m)} total | dropped no-result {int(m.no_result.sum())}, rain-revised {int((m.revised & ~m.no_result).sum())} | kept with ball data {b.match_id.nunique()} (ties {b[b.tie].match_id.nunique()})")
    print(f"balls kept: {len(b):,} | by level: {b.groupby('level').size().to_dict()}")
    legal = b.groupby(["match_id", "innings"]).legal.sum()
    print(f"legal balls per innings: max {legal.max()} | innings over 120: {(legal > 120).sum()}")
    # Scorecard check: our innings totals must reproduce the official margin for 'won by N runs' results.
    tot = b.groupby(["match_id", "innings"]).tot.sum().unstack()
    r = m.set_index("match_id").result.reindex(tot.index)
    margin = r.str.extract(r"by\s*(\d+)\s*runs?", flags=re.I)[0].astype(float)
    ok = margin.notna() & tot[2].notna()
    print(f"scorecard check ('won by N runs'): {(tot[1] - tot[2])[ok].eq(margin[ok]).mean() * 100:.1f}% of {int(ok.sum())} matches reproduce the official margin exactly")
    wk = r.str.extract(r"by\s*(\d+)\s*(?:wickets?|wkts?)", flags=re.I)[0].astype(float)
    w2 = b[b.innings == 2].groupby("match_id").is_wicket.sum().reindex(tot.index)
    ok = wk.notna() & w2.notna()
    print(f"scorecard check ('won by N wickets'): {(10 - w2)[ok].eq(wk[ok]).mean() * 100:.1f}% of {int(ok.sum())} matches")
    print(f"elo_diff range {b.elo_diff.min():.0f} to {b.elo_diff.max():.0f} | venue par runs/ball {b.par.min():.2f} to {b.par.max():.2f}")


if __name__ == "__main__":
    write(build(), "balls")
    print("saved outputs/balls.parquet")
