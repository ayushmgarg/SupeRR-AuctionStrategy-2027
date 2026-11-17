"""Stage 6: final rankings, evidence tables, figures and the dashboard file (player-level aggregates only)."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIG, OUT, load, read

PINK, NAVY, GREY = "#E6007E", "#14213D", "#9AA3B2"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": GREY, "axes.labelcolor": NAVY, "xtick.color": NAVY, "ytick.color": NAVY, "figure.dpi": 150})
EVID = {"bat": ["runs", "balls", "strike_rate", "avg", "boundary_pct", "dot_pct"], "bowl": ["balls", "runs", "economy", "wickets", "dot_pct", "error_pct"]}


def table(role: str) -> pd.DataFrame:
    r = pd.read_csv(OUT / f"ratings_{role}.csv", index_col=0)
    c = pd.read_csv(OUT / f"components_{role}.csv", index_col=0)
    t = r.join(c[EVID[role] + ["ipl_share_p", "death_share_p"]])
    t = t[t.final].sort_values("rating_safe", ascending=False)
    t.insert(0, "rank", range(1, len(t) + 1))
    official = load("competition_team_player_mapping").drop_duplicates("player_id").set_index("player_id").player_name
    t["dropdown_name"] = official.reindex(t.index)
    return t


def fig_pressure_map(b):
    d = b[(b.innings == 2) & (b.bl <= 30) & (b.need <= 70)]
    g = d.assign(x=d.bl, y=(d.need // 3) * 3).pivot_table(index="y", columns="x", values="li", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    im = ax.imshow(g.values, origin="lower", aspect="auto", cmap="magma", extent=[g.columns.min() - .5, g.columns.max() + .5, g.index.min(), g.index.max() + 3], vmax=10)
    ax.invert_xaxis()
    ax.set(xlabel="balls left in the chase", ylabel="runs needed", title="Pressure is where the match is still winnable,\nnot simply the last over")
    for bl, need, lab in ((3, 5, "5 off 3"), (4, 15, "15 off 4"), (6, 34, "34 off 6: no pressure")):
        ax.annotate(lab, (bl, need), xytext=(bl + 5, need + 9), color="white", fontsize=9, arrowprops=dict(arrowstyle="-", color="white", lw=.8))
    fig.colorbar(im, label="pressure score (x average ball)")
    fig.tight_layout(); fig.savefig(FIG / "pressure_map.png", bbox_inches="tight"); plt.close(fig)


def fig_clock(v):
    d = pd.DataFrame(v["clock"]["by_band"])
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(d.li.astype(str), d.extra_seconds, color=[GREY] * 2 + [PINK] * 4)
    for i, (s, p) in enumerate(zip(d.extra_seconds, d.pressure_score)):
        ax.text(i, s + .3, f"+{s:.1f}s", ha="center", color=NAVY, fontweight="bold")
        ax.text(i, .3, f"{p:.1f}x", ha="center", color="white", fontsize=8)
    ax.set(xlabel="balls ranked by pressure score", ylabel="seconds slower than\nthe bowler's own normal", title="The clock agrees: play slows where our pressure score is high")
    fig.tight_layout(); fig.savefig(FIG / "clock.png", bbox_inches="tight"); plt.close(fig)


def fig_calibration(m):
    d = pd.DataFrame(m["calibration"])
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    ax.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1)
    ax.plot(d.predicted, d.actual, "o-", color=PINK)
    ax.set(xlabel="win chance the model gave", ylabel="how often that side actually won", title="Win model: predicted vs actual\n(matches never seen in training)")
    fig.tight_layout(); fig.savefig(FIG / "calibration.png", bbox_inches="tight"); plt.close(fig)


def fig_top10(t, role):
    d = t.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.barh(d.name, d.rating, xerr=d.rating_sd, color=PINK, ecolor=NAVY, capsize=2, alpha=.9)
    for y, (r, p, sd) in enumerate(zip(d.rating, d.p_top10, d.rating_sd)):
        ax.text(max(r + sd, 0) + .03, y, f"top-10 in {p * 100:.0f}% of resamples", va="center", fontsize=8, color=NAVY)
    ax.set(xlabel="rating (SDs above the average eligible player); line = uncertainty", title=f"Top 10 {'batters' if role == 'bat' else 'bowlers'} under pressure")
    ax.set_xlim(right=ax.get_xlim()[1] * 1.22)
    fig.tight_layout(); fig.savefig(FIG / f"top10_{role}.png", bbox_inches="tight"); plt.close(fig)


def fig_quadrant(t, role):
    fig, ax = plt.subplots(figsize=(6.4, 5))
    ax.scatter(t.base_v_post, t.press_v_post, s=t.n_p / 6, color=GREY, alpha=.6)
    top = t.head(10)
    ax.scatter(top.base_v_post, top.press_v_post, s=top.n_p / 6, color=PINK)
    for _, r in top.iterrows():
        ax.annotate(r["name"].split()[-1], (r.base_v_post, r.press_v_post), xytext=(4, 3), textcoords="offset points", fontsize=8, color=NAVY)
    lim = [min(t.base_v_post.min(), t.press_v_post.min()) - .02, max(t.base_v_post.max(), t.press_v_post.max()) + .02]
    ax.plot(lim, lim, color=NAVY, ls="--", lw=.8)
    ax.text(lim[1], lim[1], "same under pressure\nas usual", ha="right", va="top", fontsize=7, color=NAVY)
    ax.set(xlabel="usual level, same overs (value per ball)", ylabel="pressure level (value per ball)", title=f"{'Batters' if role == 'bat' else 'Bowlers'}: usual level vs pressure level\n(dot size = pressure balls)")
    fig.tight_layout(); fig.savefig(FIG / f"quadrant_{role}.png", bbox_inches="tight"); plt.close(fig)


def fig_forecast(v):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for ax, role in zip(axes, ("bat", "bowl")):
        f = {k: x for k, x in v[role]["forecast"].items() if isinstance(x, dict) and "ci90_wins" in x and "tested" not in k}
        labs, vals = list(f)[::-1], [f[k]["rank_corr_value"] for k in list(f)[::-1]]
        err = np.array([[f[k]["rank_corr_value"] - f[k]["ci90"][0], f[k]["ci90"][1] - f[k]["rank_corr_value"]] for k in labs]).T
        ax.barh(labs, vals, xerr=err, color=[PINK if "PRESSURE" in k else GREY for k in labs], ecolor=NAVY, capsize=2)
        ax.axvline(0, color=NAVY, lw=.8)
        ax.set(title=f"{'Batters' if role == 'bat' else 'Bowlers'}: rated on 2022-24, scored on 2025-26", xlabel="rank correlation with actual pressure value\n(line = 90% interval)")
    fig.tight_layout(); fig.savefig(FIG / "forecast.png", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    b, v, m = read("balls_val"), json.load(open(OUT / "validation.json")), json.load(open(OUT / "metrics_winprob.json"))
    fig_pressure_map(b); fig_clock(v); fig_calibration(m); fig_forecast(v)
    dash = {"definition": dict(winnable=[0.15, 0.85], tough_threshold=m["pressure_threshold"], pressure_balls=m["pressure_balls"]), "players": {}}
    for role in ("bat", "bowl"):
        t = table(role)
        fig_top10(t, role); fig_quadrant(t, role)
        t.to_csv(OUT / f"final_{role}.csv")
        mism = t.head(15)[t.head(15).name != t.head(15).dropdown_name]
        cols = ["rank", "dropdown_name", "n_p", "m_p", "rating_safe", "press_v_post", "base_v_post", "lift_post", "press_w", "consistency", "p_top10"] + EVID[role][2:]
        print(f"\n== FINAL {role.upper()} : {len(t)} eligible ==")
        print(t[cols].head(12).round(3).to_string(index=False))
        print("name differs from dropdown spelling:", mism[["name", "dropdown_name"]].to_dict("records") if len(mism) else "none in top 15")
        keep = ["rank", "dropdown_name", "n_p", "m_p", "rating", "rating_safe", "rating_sd", "press_v_post", "base_v_post", "lift_post", "press_w", "consistency",
                "n_rescue", "n_close", "rescue_v", "close_v", "p_top10", "rank_lo", "rank_hi", "ipl_share_p", "death_share_p"] + EVID[role]
        dash["players"][role] = json.loads(t[keep].round(4).to_json(orient="records"))
    dash["validation"] = v
    json.dump(dash, open(OUT / "dashboard.json", "w"), indent=1)
    print("\nsaved outputs/final_bat.csv, final_bowl.csv, dashboard.json and outputs/figures/*.png")
