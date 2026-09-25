"""M2 pilot runner (approved arms; lead 12:30): B primary, 3 exploratory seeds
on the sealed canon train pool; selection on val ONLY (τ chosen on val);
declared privileged secondary arm (validity mask); ONE read of test_g1 at the
end, no selection on it. Budget logged per run (24h envelope).

Arms (spec §9.1): greedy | τ* sampling (τ selected on val) | validity-mask
(DECLARED PRIVILEGED — observable preconditions; never compared to equal arms).
Also reported: local heuristic + random valid on the same val episodes
(non-network baselines, no selection).
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import mlx.core as mx

mx.set_default_device(mx.cpu)

from ucm.eval import baselines as B
from ucm.eval import metrics as M
from ucm.eval.gate5_report import (CANON, closed_loop, load_canon_episodes,
                                   load_model)
from ucm.eval.rollout import ModelPolicy
from ucm.model.train import TrainConfig, train as train_model

OUT = "artifacts/m2-pilot"
TAUS = [2.0, 1.0, 0.5]  # sampling temperatures tried on val (greedy = ∞, always reported)


def train_seed(seed: int) -> str:
    cfg = TrainConfig(name=f"m2pilot-B-s{seed}", data=CANON, arch="B", d_model=192,
                      train_split="train", val_split="val",
                      max_updates=10_000, max_epochs=10, eval_every=100, seed=seed)
    out = train_model(cfg)
    return out, sorted(glob.glob(f"artifacts/*m2pilot-B-s{seed}"))[-1]


def rollout_arm(model, arch, episodes, seed, arm):
    name = f"B-{arm}"
    if arm == "greedy":
        pol = ModelPolicy(model, name=name, seed=seed)
    elif arm == "mask":
        pol = ModelPolicy(model, name=name, seed=seed, validity_mask=True)
    else:
        pol = ModelPolicy(model, name=name, seed=seed, sample=True,
                          temperature=float(arm.replace("tau", "")))
    results = []
    lay_cache = {}
    from ucm.env.tinygraph import TinyGraphKey
    from ucm.eval.gate5_report import rebuild_layout, task_from_record
    from ucm.eval.rollout import run_episode
    for (ref, lay_h, d0, recs) in episodes:
        if lay_h not in lay_cache:
            lay_cache[lay_h] = rebuild_layout(recs)
        env = TinyGraphKey(lay_cache[lay_h])
        env.reset(task_from_record(recs))
        r = run_episode(env, pol, ref, seed, name, d_star=d0)
        r.layout_id = lay_h
        r.seed = seed
        results.append(r)
    return results


def main(seeds: list[int]):
    os.makedirs(OUT, exist_ok=True)
    val_eps = load_canon_episodes(CANON, split="val")
    usable_val = [e for e in val_eps
                  if min(r["provenance"].get("step", 0) for r in e[3]) == 0]
    print(f"val episodes: {len(usable_val)}/{len(val_eps)} usable")

    per_seed, val_arms, best_ckpts = {}, {}, {}
    for s in seeds:
        out, run_dir = train_seed(s)
        best_ckpts[s] = run_dir
        m, _ = load_model(run_dir)
        # τ selection ON VAL ONLY (selection cost = one pass per τ over val)
        tau_scores = {}
        for tau in TAUS:
            res = rollout_arm(m, "B", usable_val, s, f"tau{tau}")
            tau_scores[tau] = M.summarize(res)["success_rate"]
        res_g = rollout_arm(m, "B", usable_val, s, "greedy")
        best_tau_val = max(tau_scores, key=tau_scores.get)
        if tau_scores[best_tau_val] <= M.summarize(res_g)["success_rate"]:
            chosen = "greedy"
        else:
            chosen = f"tau{best_tau_val}"
        per_seed[s] = {"val_greedy": M.summarize(res_g)["success_rate"],
                       "tau_scores_on_val": tau_scores, "chosen_arm": chosen,
                       "best_val_OA_in_training": out["best"]["metric"],
                       "run": run_dir}
        print(f"seed {s}: val greedy {per_seed[s]['val_greedy']:.3f} | τ {tau_scores} | chosen {chosen}", flush=True)

    # final arms on val for each seed (greedy, chosen τ, mask) + baselines once
    report = {"m2_pilot": {"canon": CANON, "seeds": seeds,
                           "val_episodes": len(usable_val),
                           "selection_pool": "val only; test_g1 read once at end"},
              "per_seed": {str(s): {k: v for k, v in per_seed[s].items()} for s in seeds}}
    val_by_arm = {}
    for s in seeds:
        m, _ = load_model(best_ckpts[s])
        for arm in ("greedy", per_seed[s]["chosen_arm"] if per_seed[s]["chosen_arm"] != "greedy" else "greedy", "mask"):
            res = rollout_arm(m, "B", usable_val, s, arm)
            for r in res:
                r.goal_type = ""
            val_by_arm.setdefault(arm, []).extend(res)
    for arm, res in val_by_arm.items():
        report.setdefault("val_arms", {})[arm] = M.summarize(res)
        if arm == "mask":
            report["val_arms"][arm]["DECLARED"] = "privileged (validity mask, observable preconditions)"
    # non-network baselines on val (seed 0, no selection)
    m0, _ = load_model(best_ckpts[seeds[0]])
    base_res = {}
    for name, pol in (("local_heuristic", B.LocalHeuristic(0)),
                      ("random_valid", B.make_random_valid(0))):
        results = []
        lay_cache = {}
        from ucm.env.tinygraph import TinyGraphKey
        from ucm.eval.gate5_report import rebuild_layout, task_from_record
        from ucm.eval.rollout import run_episode
        for (ref, lay_h, d0, recs) in usable_val:
            if lay_h not in lay_cache:
                lay_cache[lay_h] = rebuild_layout(recs)
            env = TinyGraphKey(lay_cache[lay_h])
            env.reset(task_from_record(recs))
            r = run_episode(env, pol, ref, 0, name, d_star=d0)
            r.layout_id = lay_h
            r.seed = 0
            results.append(r)
        base_res[name] = M.summarize(results)
    report["val_baselines"] = base_res

    # ONE read of test_g1: chosen arm per seed, greedy default; reported once
    g1_eps = load_canon_episodes(CANON, split="test_g1")
    usable_g1 = [e for e in g1_eps
                 if min(r["provenance"].get("step", 0) for r in e[3]) == 0]
    g1_by_arm = {}
    for s in seeds:
        m, _ = load_model(best_ckpts[s])
        arm = per_seed[s]["chosen_arm"]
        res = rollout_arm(m, "B", usable_g1, s, arm)
        g1_by_arm.setdefault(arm, []).extend(res)
    report["test_g1_single_read"] = {arm: M.summarize(res) for arm, res in g1_by_arm.items()}
    report["test_g1_single_read"]["note"] = "one read, no selection on test cells"

    with open(f"{OUT}/m2-pilot-report.json", "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(json.dumps({k: report[k] for k in ("per_seed", "val_arms", "test_g1_single_read")},
                     indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()
    main([int(s) for s in args.seeds.split(",")])
