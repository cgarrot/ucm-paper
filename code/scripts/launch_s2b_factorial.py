#!/usr/bin/env python3
"""S2b factorial launcher (lead 14:15, committed — lesson: never /tmp).

Usage:
  .venv/bin/python scripts/launch_s2b_factorial.py --smoke   # 1 cell, updates=1
  .venv/bin/python scripts/launch_s2b_factorial.py            # full: 5 seeds × 4 cells
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ucm.eval.factorial_2x2 import run_factorial, load_tgk_train_records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--updates", type=int, default=2000)
    args = ap.parse_args()

    records = load_tgk_train_records(n=5492)  # TGK train, ordre fichier déterministe
    n_seeds = 1 if args.smoke else args.seeds
    updates = 1 if args.smoke else args.updates
    n_eval = 8 if args.smoke else 78

    for i in range(n_seeds):
        seed = 100 + i
        ts = time.strftime("%Y%m%dT%H%M%S")
        r = run_factorial(records, None, updates=updates, seed=seed,
                          n_eval=n_eval, f_recovery=0.3, T=32,
                          world="tgk", d_band=(13, 15), ts=ts)
        cells = list(r.get("cells", r.get("artifacts", {})))
        print(f"[s2b] seed {seed} ({'smoke' if args.smoke else 'full'}): "
              f"{len(cells)} cells -> ts={ts}", flush=True)
    print("[s2b] COMPLETE", flush=True)


if __name__ == "__main__":
    main()
