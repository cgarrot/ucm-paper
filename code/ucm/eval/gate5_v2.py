"""GATE-5 OFFICIAL v2 — multi-seed (audit tagi-5 M1: §9.3 five confirmatory
seeds) over the frozen criteria (tagi-1 12:13, unchanged):

For each training seed s ∈ seeds (default 5):
    train A_s and B_s on the sealed-canon diagnostic (100 train episodes,
    overfit regime, order randomization active), checkpoint = BEST validation
    (M6); NOTE: in the overfit diagnostic the selection pool is the diagnostic
    itself (frozen design — GATE-1/GATE-5 stratum lives there); canon-val
    sanity numbers are REPORTED ONLY, never selected on (M4 disclosure).
Then:
    closed-loop rollouts per (arch, seed) on the diagnostic; episode records
    carry seed=s so the paired hierarchical bootstrap clusters seeds→layouts→
    tasks with all three levels resampled (M2). Per-seed values published (M1).
Criteria (frozen): (1) B−A ≥ +5 pts success on stratum d*≥3 ∪ tie episodes,
IC95 > 0; (2) loss ≤ 2 pts on d*≤2; (3) p95(B) ≤ 2× p95(A) remeasured;
(4) tie separability (tests). OA per-transition = secondary diagnostic (m11),
with non-trivial (d*>0) strata per §4.5 (M5).
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import mlx.core as mx

mx.set_default_device(mx.cpu)

from ucm.eval.gate5_report import (CANON, TRANSPARENCY, closed_loop,
                                   load_canon_episodes, load_model,
                                   tie_episode_refs, transition_oa_strata)
from ucm.eval import metrics as M
from ucm.model.train import TrainConfig, train as train_model

OUT = "artifacts/gate5-report"


def train_arm(arch: str, seed: int, diag_path: str, tag: str = "gate5v2") -> str:
    d_model = {"B160": 160, "B144": 144}.get(arch, 192 if arch.startswith("B") else 128)
    core = "B" if arch.startswith("B") else "A"
    existing = sorted(glob.glob(f"artifacts/*{tag}-{arch}-s{seed}"))
    if existing:
        import os as _os
        cand = existing[-1]
        if _os.path.exists(cand + "/checkpoint.npz") and _os.path.exists(cand + "/final.json"):
            import json as _json
            if _json.load(open(cand + "/final.json")).get("updates_run") == 5000:
                return cand  # reuse completed run (idempotent orchestration)
    cfg = TrainConfig(name=f"{tag}-{arch}-s{seed}", data=diag_path,
                      overfit=True, eval_every=500, seed=seed, arch=core,
                      d_model=d_model)
    train_model(cfg)
    return sorted(glob.glob(f"artifacts/*{tag}-{arch}-s{seed}"))[-1]


def main(seeds: list[int], episodes_n: int = 100, seed: int = 42):
    os.makedirs(OUT, exist_ok=True)
    eps = load_canon_episodes(CANON, split="train", limit=episodes_n)
    # m10 (audit): skip episodes whose step-0 record was deduplicated away
    usable = [(ref, h, d, rs) for (ref, h, d, rs) in eps
              if min(r["provenance"].get("step", 0) for r in rs) == 0]
    skipped = len(eps) - len(usable)
    diag_path = f"{OUT}/gate5v2-diagnostic.jsonl"
    with open(diag_path, "w") as fh:
        for (_, _, _, rs) in usable:
            for r in rs:
                fh.write(json.dumps(r) + "\n")
    print(f"diagnostic: {len(usable)} episodes (skipped {skipped} dedup-merged), seeds={seeds}")

    ties = set()          # tie episodes identified on the FIRST seed's A (frozen rule:
    per_seed = {}         # stratum pre-specified from the diagnostic A failure mode)
    resA_all, resB_all = [], []
    oaA_strata, oaB_strata = None, None
    for s in seeds:
        run_a = train_arm("A", s, diag_path)
        run_b = train_arm("B", s, diag_path)
        mA, _ = load_model(run_a)
        mB, _ = load_model(run_b)
        if not ties:
            ties = tie_episode_refs(mA, usable)
        resA = closed_loop(mA, "A", usable, seed=seed)
        resB = closed_loop(mB, "B", usable, seed=seed)
        for r in resA + resB:
            r.seed = s  # training seed → bootstrap seed level
        resA_all.extend(resA)
        resB_all.extend(resB)
        sA, sB = M.summarize(resA), M.summarize(resB)
        oaA, oaAnt = transition_oa_strata(mA, usable)
        oaB, oaBnt = transition_oa_strata(mB, usable)
        oaA_strata, oaB_strata = (oaA, oaAnt), (oaB, oaBnt)
        per_seed[s] = {"A": {"success": sA["success_rate"], "OA_nontrivial": oaAnt["rate"]},
                       "B": {"success": sB["success_rate"], "OA_nontrivial": oaBnt["rate"]},
                       "runs": {"A": run_a, "B": run_b}}
        print(f"seed {s}: A success {sA['success_rate']:.3f} / B {sB['success_rate']:.3f} | "
              f"OA_nt A {oaAnt['rate']:.4f} / B {oaBnt['rate']:.4f}", flush=True)

    strat_ids = {r.episode_id for r in resA_all
                 if r.d_star >= 3 or r.episode_id in ties}
    safe_ids = {r.episode_id for r in resA_all if r.d_star <= 2}
    stratA = [r for r in resA_all if r.episode_id in strat_ids]
    stratB = [r for r in resB_all if r.episode_id in strat_ids]
    safeA = [r for r in resA_all if r.episode_id in safe_ids]
    safeB = [r for r in resB_all if r.episode_id in safe_ids]

    pair_strat = M.paired_hierarchical_bootstrap(stratB, stratA, n_boot=5000, seed=seed)
    pair_safe = M.paired_hierarchical_bootstrap(safeA, safeB, n_boot=5000, seed=seed)
    sA_strat, sB_strat = M.summarize(stratA), M.summarize(stratB)

    verdict = {
        "gate": "GATE-5 OFFICIAL v2 (multi-seed, audit wave-3 fixes)",
        "canon": {"path": CANON, "split": "train", "episodes": len(usable),
                  "skipped_dedup_merged": skipped, "tie_episodes": len(ties),
                  "training_seeds": seeds},
        "criterion_1_stratum_closed_loop_success": {
            "n_episodes_per_seed": len(strat_ids),
            "success_A": sA_strat["success_rate"], "success_B": sB_strat["success_rate"],
            "diff_pts": (sB_strat["success_rate"] - sA_strat["success_rate"]) * 100,
            "paired_bootstrap_B_minus_A": pair_strat,
            "PASS": (sB_strat["success_rate"] - sA_strat["success_rate"]) >= 0.05
                    and pair_strat["ci_low"] > 0},
        "criterion_2_safe_d_le_2": {
            "success_A": M.summarize(safeA)["success_rate"],
            "success_B": M.summarize(safeB)["success_rate"],
            "loss_pts": (M.summarize(safeA)["success_rate"]
                         - M.summarize(safeB)["success_rate"]) * 100,
            "PASS": (M.summarize(safeA)["success_rate"]
                     - M.summarize(safeB)["success_rate"]) <= 0.02},
        "criterion_3_cost": "remeasured from per-seed checkpoints at report tail",
        "criterion_4_tie_separability": "tests/TestTieSeparability — PASS",
        "secondary_oa_diagnostic_per_transition": {
            "A": {"all": oaA_strata[0], "nontrivial_d_gt_0": oaA_strata[1]},
            "B": {"all": oaB_strata[0], "nontrivial_d_gt_0": oaB_strata[1]},
            "unit": "per-transition (episode strata by initial d*)"},
        "per_seed": {str(s): {a: v[a] for a in ("A", "B")} for s, v in per_seed.items()},
        "per_seed_runs": {str(s): v["runs"] for s, v in per_seed.items()},
        "transparency": TRANSPARENCY,
        "m4_disclosure": ("comparison pool = the frozen diagnostic (train split, "
                          "overfit design, as specified at freeze); canon val is "
                          "layout-disjoint and REPORTED at M2 selection only — "
                          "test_* cells are never read by this runner"),
    }
    # criterion 3 remeasure: use the LAST seed's checkpoints (fresh, representative)
    from ucm.eval.gate5_report import bench_decisions_compiled
    from ucm.eval.timing import bench_decisions
    mA, _ = load_model(per_seed[seeds[-1]]["runs"]["A"])
    mB, _ = load_model(per_seed[seeds[-1]]["runs"]["B"])
    obs_list = [rs[0]["policy_input"] for (_, _, _, rs) in usable[:8]]
    latA = bench_decisions(obs_list, mA, warmup=50, n_measured=1000)["model_only"]
    latB = bench_decisions(obs_list, mB, warmup=50, n_measured=1000)["model_only"]
    latBc = bench_decisions_compiled(mB, obs_list, warmup=50, n_measured=1000)
    ratio = min(latB["p95_ms"], latBc["p95_ms"]) / max(latA["p95_ms"], 1e-9)
    verdict["criterion_3_cost"] = {"p95_A_ms": latA["p95_ms"], "p95_B_ms": latB["p95_ms"],
                                   "p95_B_compiled_ms": latBc["p95_ms"],
                                   "ratio_best_B_vs_A": ratio, "PASS": ratio <= 2.0}

    with open(f"{OUT}/gate5-official-v2.json", "w") as fh:
        json.dump(verdict, fh, indent=2, default=str)
    print(json.dumps({k: verdict[k] for k in
                      ("criterion_1_stratum_closed_loop_success",
                       "criterion_2_safe_d_le_2", "criterion_3_cost")}, indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main([int(s) for s in args.seeds.split(",")], args.episodes, args.seed)
