"""Shared paths, constants and loaders. Every threshold the analysis depends on lives here."""
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("PP_DATA", ROOT / "6af458274c1e8_data_pack" / "Data Pack"))  # folder with the data pack CSVs
OUT = ROOT / os.environ.get("PP_OUT", "outputs")   # PP_OUT / PP_SEED exist only for the seed-robustness check
FIG = OUT / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

SEED = int(os.environ.get("PP_SEED", 7))

# --- pressure definition ---
WIN_LO, WIN_HI = 0.15, 0.85   # "winnable": batting side's win chance inside this band
TOUGH_Q = 0.85                # "tough": ball swing in the top 15% of all balls
LIVE_LO, LIVE_HI = 0.05, 0.95  # outside this band the match is decided ("garbage time")

# --- eligibility ---
MIN_BALLS = 100
MIN_MATCHES = 15

# --- forecast test: rate on matches before this date, score on matches from it ---
TEST_FROM = "2025-01-01"

PHASES = {"pp": (1, 6), "mid": (7, 15), "death": (16, 20)}
ATTACK_STROKES = ["Slog", "Slog Sweep", "Pull shot", "Hook Shot", "Paddle Scoop", "Reverse Sweep", "UpperCut",
                  "Switch", "Reverse Paddle Scoop", "PickUp", "Inside out"]


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / f"{name}.csv", low_memory=False)


def phase_of(over: pd.Series) -> pd.Series:
    return pd.cut(over, [0, 6, 15, 20], labels=["pp", "mid", "death"]).astype(str)


def read(name: str) -> pd.DataFrame:
    return pd.read_parquet(OUT / f"{name}.parquet")


def write(df: pd.DataFrame, name: str) -> None:
    df.to_parquet(OUT / f"{name}.parquet", index=False)


def eb_shrink(x: np.ndarray, var: np.ndarray, prior_mean: float = 0.0):
    """Empirical-Bayes shrinkage of noisy estimates x with sampling variances var.

    Between-player variance tau2 is estimated by method of moments. Returns (posterior mean,
    posterior sd, tau2). A player with a small sample is pulled toward prior_mean.
    """
    x, var = np.asarray(x, float), np.asarray(var, float)
    w = 1 / var
    tau2 = max(0.0, np.average((x - prior_mean) ** 2, weights=w) - np.average(var, weights=w))
    k = tau2 / (tau2 + var)
    return prior_mean + k * (x - prior_mean), np.sqrt(k * var), tau2
