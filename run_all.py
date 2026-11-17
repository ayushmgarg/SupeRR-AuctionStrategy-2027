"""Run the whole pipeline in order (about 25 minutes on a laptop).

    python run_all.py            # every stage
    python run_all.py --from 4   # resume from stage 4, reusing earlier outputs
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).parent / "src"
STAGES = [
    "stage1_match_states",    # one row per ball, with the match state before it
    "stage2_win_model",       # win chance before every ball, pressure score, pressure flag
    "stage3_ball_value",      # what each ball was worth to batter and bowler
    "stage4_player_ratings",  # pressure record and usual level per player
    "stage5_validation",      # rating, repeatability, forecast test, sensitivity runs
    "stage6_export",          # final tables, figures, dashboard file
]

ap = argparse.ArgumentParser()
ap.add_argument("--from", dest="start", type=int, default=1, help="first stage to run (1-6)")
args = ap.parse_args()

for i, stage in enumerate(STAGES[args.start - 1:], start=args.start):
    print(f"\n{'=' * 18} {i}/6  {stage} {'=' * 18}", flush=True)
    t = time.time()
    subprocess.run([sys.executable, "-W", "ignore", str(SRC / f"{stage}.py")], check=True)
    print(f"--- done in {time.time() - t:.0f}s", flush=True)
