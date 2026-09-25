"""GATE-5 criterion 1 on the VAL pool — the 12:38 freeze (M4 disclosure):
evaluation stratum = VAL episodes with d*≥3 (layout-disjoint from train by
construction, WS-B GATE-0), deterministic extraction (all usable val episodes);
model selection = best OA on TRAIN (the gate5v2 checkpoints already follow
this: best-diagnostic-OA checkpointing, M6). Ties are a TRAIN-diagnostic
concept, reported separately, not part of the frozen val stratum.

Cost criterion re-verified per the frozen protocol: BOTH arms mx.compile'd,
5 repetitions each, median of the p95s, ratio ≤ 2.
"""

from __future__ import annotations

import glob
import json
import os

import mlx.core as mx

mx.set_default_device(mx.cpu)

from ucm.eval import metrics as M
from ucm.eval.gate5_report import (CANON, bench_decisions_compiled, closed_loop,
                                   load_canon_episodes, load_model)
from ucm.eval.timing import bench_decisions

OUT = "artifacts/gate5-report"


def main(seeds: list[int], seed: int = 42):
    val_eps = load_canon_episodes(CANON, split="val")
    usable = [e for e in val_eps
              if min(r["provenance"].get("step", 0) for r in e[3]) == 0]
    strat = [e for e in usable if e[2] >= 3]
    safe = [e for e in usable if e[2] <= 2]
    print(f"val: {len(usable)} usable | stratum d*>=3: {len(strat)} | d*<=2: {len(safe)}")

    resA_all, resB_all = [], []
    per_seed = {}
    for s in seeds:
        run_a = sorted(glob.glob(f"artifacts/*gate5v2-A-s{s}"))[-1]
        run_b = sorted(glob.glob(f"artifacts/*gate5v2-B-s{s}"))[-1]
        mA, _ = load_model(run_a)
        mB, _ = load_model(run_b)
        resA = closed_loop(mA, "A", usable, seed=seed)
        resB = closed_loop(mB, "B", usable, seed=seed)
        for r in resA + resB:
            r.seed = s
        resA_all.extend(resA)
        resB_all.extend(resB)
        sA, sB = M.summarize(resA), M.summarize(resB)
        per_seed[s] = {"A": sA["success_rate"], "B": sB["success_rate"],
                       "runs": {"A": run_a, "B": run_b}}
        print(f"seed {s}: val success A {sA['success_rate']:.3f} | B {sB['success_rate']:.3f}", flush=True)

    strat_ids = {e[0] for e in strat}
    safe_ids = {e[0] for e in safe}
    sA_st = [r for r in resA_all if r.episode_id in strat_ids]
    sB_st = [r for r in resB_all if r.episode_id in strat_ids]
    sA_sf = [r for r in resA_all if r.episode_id in safe_ids]
    sB_sf = [r for r in resB_all if r.episode_id in safe_ids]

    pair_strat = M.paired_hierarchical_bootstrap(sB_st, sA_st, n_boot=5000, seed=seed)
    pair_safe = M.paired_hierarchical_bootstrap(sA_sf, sB_sf, n_boot=5000, seed=seed)
    A_st, B_st = M.summarize(sA_st), M.summarize(sB_st)

    # cost: BOTH arms compiled, 5 repetitions, median p95
    mA, _ = load_model(sorted(glob.glob(f"artifacts/*gate5v2-A-s{seeds[-1]}"))[-1])
    mB, _ = load_model(sorted(glob.glob(f"artifacts/*gate5v2-B-s{seeds[-1]}"))[-1])
    obs_list = [rs[0]["policy_input"] for (_, _, _, rs) in usable[:8]]
    p95A, p95B = [], []
    for rep in range(5):
        p95A.append(bench_decisions_compiled(mA, obs_list, warmup=50, n_measured=1000)["p95_ms"])
        p95B.append(bench_decisions_compiled(mB, obs_list, warmup=50, n_measured=1000)["p95_ms"])
    p95A.sort(); p95B.sort()
    medA, medB = p95A[2], p95B[2]
    ratio = medB / max(medA, 1e-9)

    verdict = {
        "gate": "GATE-5 criterion 1 — VAL-stratum (12:38 freeze, promotion proof)",
        "pool": {"split": "val", "usable_episodes": len(usable),
                 "stratum_d_ge_3": len(strat), "safe_d_le_2": len(safe),
                 "layout_disjoint_from_train": True,
                 "model_selection": "best OA on TRAIN diagnostic (gate5v2 checkpoints, M6)"},
        "criterion_1_val_stratum_success": {
            "success_A": A_st["success_rate"], "success_B": B_st["success_rate"],
            "diff_pts": (B_st["success_rate"] - A_st["success_rate"]) * 100,
            "paired_bootstrap_B_minus_A": pair_strat,
            "PASS": (B_st["success_rate"] - A_st["success_rate"]) >= 0.05
                    and pair_strat["ci_low"] > 0},
        "criterion_2_val_safe_d_le_2": {
            "success_A": M.summarize(sA_sf)["success_rate"],
            "success_B": M.summarize(sB_sf)["success_rate"],
            "loss_pts": (M.summarize(sA_sf)["success_rate"]
                         - M.summarize(sB_sf)["success_rate"]) * 100,
            "PASS": (M.summarize(sA_sf)["success_rate"]
                     - M.summarize(sB_sf)["success_rate"]) <= 0.02},
        "criterion_3_cost_frozen_protocol": {
            "protocol": "both arms mx.compile'd, 5 repetitions, median p95",
            "p95_A_ms_reps": p95A, "p95_B_ms_reps": p95B,
            "median_p95_A_ms": medA, "median_p95_B_ms": medB,
            "ratio": ratio, "PASS": ratio <= 2.0},
        "per_seed_val_success": {str(s): {k: v for k, v in per_seed[s].items() if k != "runs"}
                                 for s in seeds},
        "per_seed_runs": {str(s): per_seed[s]["runs"] for s in seeds},
        "note_train_stratum_reference": ("train-diagnostic stratum (+11.8 pts, "
                                         "gate5-official-v2.json) reported separately; "
                                         "ties are a train concept"),
    }
    with open(f"{OUT}/gate5-val-stratum.json", "w") as fh:
        json.dump(verdict, fh, indent=2, default=str)
    print(json.dumps({k: verdict[k] for k in
                      ("criterion_1_val_stratum_success", "criterion_2_val_safe_d_le_2",
                       "criterion_3_cost_frozen_protocol")}, indent=1, default=str))


if __name__ == "__main__":
    main([0, 1, 2, 3, 4])
