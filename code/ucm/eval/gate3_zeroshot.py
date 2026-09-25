"""GATE-3 / M2b — ZERO-SHOT strict on the reserved composition cell (g2-extended).

The reserved combination AT(key, junction room deg≥3) was EXCLUDED from
train/val by construction (WS-B seal). Question (last V0 question): does B144
generalize to it WITHOUT training? Protocol: the five GATE-2 checkpoints,
UNCHANGED (no fine-tuning), arms frozen as chosen on val at GATE-2, ONE read
per seed of the reserved cell. Threshold (spec GATE-3): success ≥ 80 %.
"""

from __future__ import annotations

import glob
import json
import os

import mlx.core as mx

mx.set_default_device(mx.cpu)

from ucm.eval import metrics as M
from ucm.eval.gate5_report import load_canon_episodes, load_model, rebuild_layout, task_from_record
from ucm.eval.rollout import ModelPolicy, run_episode
from ucm.env.tinygraph import TinyGraphKey

G2EXT = "artifacts/data/m0-transitions-g2ext.jsonl"
OUT = "artifacts/gate3-zeroshot"


def rollout(model, episodes, seed, arm):
    name = f"B144-{arm}"
    kw = {"sample": True, "temperature": float(arm.replace("tau", ""))} if arm.startswith("tau") else {}
    pol = ModelPolicy(model, name=name, seed=seed, **kw)
    lay_cache = {}
    results = []
    for (ref, lay_h, d0, recs) in episodes:
        if lay_h not in lay_cache:
            lay_cache[lay_h] = rebuild_layout(recs)
        env = TinyGraphKey(lay_cache[lay_h])
        env.reset(task_from_record(recs))
        r = run_episode(env, pol, f"g2-{ref}", seed, name, d_star=d0)
        r.layout_id = lay_h
        r.seed = seed
        r.goal_type = recs[0]["policy_input"]["goal"]["predicate"]
        results.append(r)
    return results


def main():
    os.makedirs(OUT, exist_ok=True)
    eps = [e for e in load_canon_episodes(G2EXT, split="test_g2")
           if min(r["provenance"].get("step", 0) for r in e[3]) == 0]
    print(f"g2-extended reserved cell: {len(eps)} episodes")
    chosen = json.load(open("artifacts/gate2-confirmation/gate2-confirmation.json"))["chosen_arm_per_seed"]

    per_seed, all_res = {}, []
    for s in range(5):
        model, _ = load_model(sorted(glob.glob(f"artifacts/*gate2c-B144-s{s}"))[-1])
        arm = chosen[str(s)]["arm"]
        mx.random.seed(30_000 + s)  # declared evaluation RNG (GATE-3 read)
        res = rollout(model, eps, s, arm)
        all_res.extend(res)
        per_seed[s] = {"arm": arm, "success": M.summarize(res)["success_rate"],
                       "mx_random_seed": 30_000 + s}
        print(f"seed {s} ({arm}): {per_seed[s]['success']:.4f}", flush=True)

    summary = M.summarize(all_res)
    ci = M.hierarchical_bootstrap(all_res, metric=M.success, n_boot=5000, seed=42)
    by_band = {}
    for band in sorted({r.d_star for r in all_res}):
        sub = [r for r in all_res if r.d_star == band]
        by_band[str(band)] = {"n": len(sub), "success": M.summarize(sub)["success_rate"]}

    verdict = {
        "gate": "GATE-3 / M2b — ZERO-SHOT composition (AT(key, junction), reserved cell)",
        "threshold": 0.80,
        "protocol": ("5 GATE-2 checkpoints UNCHANGED (no fine-tuning), arms frozen from "
                     "val selection at GATE-2, ONE read per seed, declared RNG seeds"),
        "zero_shot": True, "reserved_cell_disjoint_from_train_val": True,
        "overall": {**summary, "hierarchical_bootstrap": ci},
        "success_by_d_star_band": by_band,
        "per_seed": {str(s): per_seed[s] for s in range(5)},
        "PASS": summary["success_rate"] >= 0.80,
    }
    with open(f"{OUT}/gate3-zeroshot.json", "w") as fh:
        json.dump(verdict, fh, indent=2, default=str)
    print(json.dumps({k: verdict[k] for k in ("overall", "success_by_d_star_band", "PASS")},
                     indent=1, default=str))


if __name__ == "__main__":
    main()
