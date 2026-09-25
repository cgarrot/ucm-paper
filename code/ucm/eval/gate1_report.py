"""GATE-1 OFFICIAL runner (sequence step (b); requires tagi-5 green light on
the sealed canon):

    1. extract the first N=100 train episodes from the sealed canon
       (deterministic: file order of episode_ref) → diagnostic JSONL artifact
    2. overfit Model A on it (5k updates cap, §8.2 diagnostic regime,
       order randomization active — T8)
    3. report: optimal-action rate (§9.2 diagnostic), exact parameter count,
       closed-loop success on the same episodes, p50/p95/p99 batch-1 latency
       (mx.eval forced, warm-up 50, ≥1000 measured decisions)

Usage:
    .venv/bin/python -m ucm.eval.gate1_report --episodes 100 --seed 42 [--arch A]
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import mlx.core as mx
import mlx.nn as nn

mx.set_default_device(mx.cpu)

from ucm.eval.gate5_report import (CANON, closed_loop, load_canon_episodes,
                                   load_model, rebuild_layout)
from ucm.eval.timing import bench_and_save
from ucm.model.train import TrainConfig, train as train_model


def extract(episodes, path):
    with open(path, "w") as fh:
        for (_, _, _, recs) in episodes:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
    n_trans = sum(len(r) for (_, _, _, r) in episodes)
    return n_trans


def main(episodes_n: int, seed: int, arch: str):
    os.makedirs("artifacts/gate1-official", exist_ok=True)
    eps = load_canon_episodes(CANON, split="train", limit=episodes_n)
    diag_path = f"artifacts/gate1-official/diagnostic-{episodes_n}ep.jsonl"
    n_trans = extract(eps, diag_path)
    d_hist = {}
    for (_, _, d0, _) in eps:
        d_hist[str(d0)] = d_hist.get(str(d0), 0) + 1
    print(f"diagnostic: {episodes_n} episodes / {n_trans} transitions | d* hist {d_hist}")

    cfg = TrainConfig(name=f"gate1-official-{arch}", data=diag_path, overfit=True,
                      eval_every=500, seed=seed, arch=arch,
                      d_model=(128 if arch == "A" else 192))
    out = train_model(cfg)
    run_dir = sorted(glob.glob("artifacts/*gate1-official-" + arch))[-1]

    m, arch_name = load_model(run_dir)
    res = closed_loop(m, arch_name, eps, seed=seed)
    from ucm.eval import metrics as M
    summary = M.summarize(res)

    # M5 (audit tagi-5): d*=0 episodes are trivial STOP cells — §4.5 forbids
    # letting them inflate non-trivial scores. Report both, gate on non-trivial.
    from ucm.eval.gate5_report import transition_oa_strata
    oa_all, oa_nontrivial = transition_oa_strata(m, eps)

    obs_list = [recs[0]["policy_input"] for (_, _, _, recs) in eps[:8]]
    lat = bench_and_save(obs_list, m, "artifacts/gate1-official", warmup=50,
                         n_measured=1000, seed=seed)

    report = {
        "gate": "GATE-1 OFFICIAL", "arch": arch_name,
        "n_params": out["n_params"],
        "diagnostic": {"episodes": episodes_n, "transitions": n_trans,
                       "d_star_hist": d_hist, "canon": CANON, "extract": diag_path},
        "optimal_action_rate_final": out["final_train"]["optimal_action_rate"],
        "best_optimal_action_rate": out["best"].get("metric"),
        "optimal_action_rate_all_transitions": oa_all,          # per-transition unit (m11)
        "optimal_action_rate_nontrivial_d_gt_0": oa_nontrivial,  # §4.5 gate metric
        "gate_threshold_OA": 0.99,
        "PASS_OA": oa_nontrivial["rate"] >= 0.99,
        "closed_loop_same_episodes": summary,
        "latency": {"model_only": lat["model_only"], "full_decision": lat["full_decision"],
                    "gates": lat["gates"]},
        "run_dir": run_dir,
    }
    with open(f"artifacts/gate1-official/gate1-official-{arch}.json", "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(json.dumps({k: report[k] for k in ("arch", "n_params",
                                             "optimal_action_rate_final", "PASS_OA",
                                             "closed_loop_same_episodes")}, indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arch", default="A", choices=["A", "B"])
    args = ap.parse_args()
    main(args.episodes, args.seed, args.arch)
