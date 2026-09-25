"""GATE-2 CONFIRMATION (frozen protocol; lead GO 15:53):

    - B144 (primary V0), 5 training seeds on the sealed canon TRAIN pool
      (standard regime: ≤10 epochs / 10k updates, canon-val selection,
      best-val checkpointing)
    - arm chosen PER SEED on val ONLY (greedy vs τ grid)
    - ONE read of test_g1 per seed with the chosen arm = OFFICIAL number
    - hierarchical bootstrap seeds→layouts→episodes; thresholds: success
      ≥ 95 %, CI low ≥ 90 %, per goal type ≥ 85 % (spec GATE-2)
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import mlx.core as mx

mx.set_default_device(mx.cpu)

from ucm.eval import metrics as M
from ucm.eval.gate5_report import (CANON, load_canon_episodes, load_model,
                                   rebuild_layout, task_from_record)
from ucm.eval.rollout import ModelPolicy, run_episode
from ucm.env.tinygraph import TinyGraphKey
from ucm.model.train import TrainConfig, train as train_model

OUT = "artifacts/gate2-confirmation"
TAUS = [2.0, 1.0, 0.5]


def train_seed(seed: int) -> str:
    existing = sorted(glob.glob(f"artifacts/*gate2c-B144-s{seed}"))
    if existing:
        cand = existing[-1]
        if os.path.exists(cand + "/checkpoint.npz") and os.path.exists(cand + "/final.json"):
            return cand
    cfg = TrainConfig(name=f"gate2c-B144-s{seed}", data=CANON, arch="B", d_model=144,
                      train_split="train", val_split="val",
                      max_updates=10_000, max_epochs=10, eval_every=100, seed=seed)
    train_model(cfg)
    return sorted(glob.glob(f"artifacts/*gate2c-B144-s{seed}"))[-1]


def rollout(model, episodes, seed, arm, split):
    name = f"B144-{arm}"
    if arm == "greedy":
        pol = ModelPolicy(model, name=name, seed=seed)
    elif arm == "mask":
        pol = ModelPolicy(model, name=name, seed=seed, validity_mask=True)
    else:
        pol = ModelPolicy(model, name=name, seed=seed, sample=True,
                          temperature=float(arm.replace("tau", "")))
    lay_cache = {}
    results = []
    for (ref, lay_h, d0, recs) in episodes:
        if lay_h not in lay_cache:
            lay_cache[lay_h] = rebuild_layout(recs)
        env = TinyGraphKey(lay_cache[lay_h])
        env.reset(task_from_record(recs))
        r = run_episode(env, pol, f"{split}-{ref}", seed, name, d_star=d0)
        r.layout_id = lay_h
        r.seed = seed
        r.goal_type = recs[0]["policy_input"]["goal"]["predicate"]
        results.append(r)
    return results


def main(seeds: list[int]):
    os.makedirs(OUT, exist_ok=True)
    val_eps = [e for e in load_canon_episodes(CANON, split="val")
               if min(r["provenance"].get("step", 0) for r in e[3]) == 0]
    g1_eps = [e for e in load_canon_episodes(CANON, split="test_g1")
              if min(r["provenance"].get("step", 0) for r in e[3]) == 0]
    print(f"val: {len(val_eps)} | test_g1: {len(g1_eps)}")

    chosen, runs = {}, {}
    g1_results = []
    val_selection_cost = {}
    per_seed_g1 = {}
    for s in seeds:
        # deterministic sampling: MLX RNG seeded per (seed, phase) before ANY
        # stochastic evaluation (marking condition, tagi-1 16:01)
        mx.random.seed(10_000 + s)
        run_dir = train_seed(s)
        runs[s] = run_dir
        m, _ = load_model(run_dir)
        # arm selection on VAL ONLY
        tau_scores = {}
        for tau in TAUS:
            res = rollout(m, val_eps, s, f"tau{tau}", "val")
            tau_scores[f"tau{tau}"] = M.summarize(res)["success_rate"]
        res_g = rollout(m, val_eps, s, "greedy", "val")
        g = M.summarize(res_g)["success_rate"]
        best_tau = max(tau_scores, key=tau_scores.get)
        arm = "greedy" if g >= tau_scores[best_tau] else best_tau
        chosen[s] = {"arm": arm, "val_greedy": g, "tau_scores": tau_scores}
        val_selection_cost[s] = (len(TAUS) + 1) * len(val_eps)
        print(f"seed {s}: val greedy {g:.3f} | τ {tau_scores} → {arm}", flush=True)
        # ONE official test_g1 read with the chosen arm (fresh declared RNG state)
        mx.random.seed(20_000 + s)
        res = rollout(m, g1_eps, s, arm, "g1")
        g1_results.extend(res)
        per_seed_g1[s] = {"arm": arm, "summary": M.summarize(res),
                          "mx_random_seed": 20_000 + s}
        print(f"seed {s}: test_g1 (chosen={arm}) success {M.summarize(res)['success_rate']:.4f}", flush=True)

    summary = M.summarize(g1_results)
    ci = M.hierarchical_bootstrap(g1_results, metric=M.success, n_boot=5000, seed=42)
    per_goal = {}
    for gt in ("REACH", "HAVE", "AT"):
        sub = [r for r in g1_results if r.goal_type == gt]
        if sub:
            per_goal[gt] = {"success": M.summarize(sub)["success_rate"], "n": len(sub)}

    verdict = {
        "gate": "GATE-2 CONFIRMATION (B144 primary, frozen protocol)",
        "thresholds": {"success": 0.95, "ci_low": 0.90, "per_goal": 0.85},
        "official_test_g1": {**summary, "hierarchical_bootstrap": ci},
        "PASS_success": summary["success_rate"] >= 0.95,
        "PASS_ci_low": ci["ci_low"] >= 0.90,
        "per_goal_type": per_goal,
        "PASS_per_goal": all(v["success"] >= 0.85 for v in per_goal.values()),
        "chosen_arm_per_seed": {str(s): chosen[s] for s in seeds},
        "per_seed_test_g1": {str(s): per_seed_g1[s] for s in seeds},
        "rng_discipline": {"training": "mx.random.seed(cfg.seed) in train()",
                            "val_selection": "mx.random.seed(10000+seed) per seed",
                            "official_g1_read": "mx.random.seed(20000+seed) per seed"},
        "val_selection_cost_episodes": val_selection_cost,
        "runs": {str(s): runs[s] for s in seeds},
        "protocol_notes": ("arms chosen on val ONLY; test_g1 read ONCE per seed with the "
                           "chosen arm; no selection on test cells at any point"),
    }
    with open(f"{OUT}/gate2-confirmation.json", "w") as fh:
        json.dump(verdict, fh, indent=2, default=str)
    print(json.dumps({k: verdict[k] for k in
                      ("official_test_g1", "PASS_success", "PASS_ci_low",
                       "per_goal_type", "PASS_per_goal")}, indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4")
    args = ap.parse_args()
    main([int(s) for s in args.seeds.split(",")])
