"""GATE-3 OFFICIAL (frozen 16:20:52) — zero-shot STRICT on the COMBINED
reserved cell: 496 episodes (193 canon test_g2 + 303 g2ext), 102 layouts.

    - B144 checkpoints from GATE-2, UNCHANGED (no retraining, no fine-tuning)
    - arms per seed frozen (same val choices as GATE-2)
    - ONE read of the combined cell
    - evaluation RNG: mx.random.seed(20_000 + seed) — same scheme as the
      GATE-2 official g1 reads
    - hierarchical bootstrap, threshold: success ≥ 80 % (spec §11 GATE-3)

Run ONLY after tagi-5's green light on (a) the GATE-2 v2 / M3 audit and
(b) the g2ext data verification. A --dry-run flag performs the wiring check
on 10 episodes WITHOUT publishing.
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

G2EXT = "artifacts/data/m0-transitions-g2ext.jsonl"
OUT = "artifacts/gate3-zeroshot"


def combined_cell():
    canon_all = load_canon_episodes(CANON, split="test_g2")
    ext_all = load_canon_episodes(G2EXT, split="test_g2")
    canon = [e for e in canon_all if min(r["provenance"].get("step", 0) for r in e[3]) == 0]
    ext = [e for e in ext_all if min(r["provenance"].get("step", 0) for r in e[3]) == 0]
    skipped = (len(canon_all) - len(canon)) + (len(ext_all) - len(ext))  # dedup-merged (no step 0)
    return canon, ext, skipped


def rollout(model, episodes, seed, arm, tag):
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
        r = run_episode(env, pol, f"{tag}-{ref}", seed, name, d_star=d0)
        r.layout_id = lay_h
        r.seed = seed
        r.goal_type = recs[0]["policy_input"]["goal"]["predicate"]
        results.append(r)
    return results


def main(dry_run: bool = False, dry_n: int = 10):
    canon, ext, skipped = combined_cell()
    print(f"cellule combinée: {len(canon)} canon + {len(ext)} g2ext = {len(canon) + len(ext)} "
          f"épisodes EFFECTIFS (skipped_dedup_merged={skipped})")
    hashes_c = {e[1] for e in canon}
    hashes_e = {e[1] for e in ext}
    print(f"layouts: canon {len(hashes_c)} + g2ext {len(hashes_e)} = {len(hashes_c | hashes_e)} (attendu 102)")

    chosen = json.load(open("artifacts/gate2-confirmation/gate2-confirmation.json"))["chosen_arm_per_seed"]
    if dry_run:
        ext = ext[:dry_n]
        print(f"DRY-RUN: {len(ext)} épisodes g2ext, wiring check SANS publication")
    cell = canon + ext if not dry_run else ext

    per_seed, all_res = {}, []
    for s in range(5):
        model, _ = load_model(sorted(glob.glob(f"artifacts/*gate2c-B144-s{s}"))[-1])
        arm = chosen[str(s)]["arm"]
        mx.random.seed(20_000 + s)  # frozen scheme (same as GATE-2 official reads)
        res = rollout(model, cell, s, arm, "g3")
        all_res.extend(res)
        per_seed[s] = {"arm": arm, "success": M.summarize(res)["success_rate"],
                       "mx_random_seed": 20_000 + s}
        print(f"seed {s} ({arm}): {per_seed[s]['success']:.4f}", flush=True)

    if dry_run:
        print("DRY-RUN OK — wiring vérifié, rien de publié")
        return

    summary = M.summarize(all_res)
    ci = M.hierarchical_bootstrap(all_res, metric=M.success, n_boot=5000, seed=42)
    by_band = {}
    for band in sorted({r.d_star for r in all_res}):
        sub = [r for r in all_res if r.d_star == band]
        by_band[str(band)] = {"n": len(sub), "success": M.summarize(sub)["success_rate"]}
    verdict = {
        "gate": "GATE-3 OFFICIAL (frozen 16:20:52) — zero-shot STRICT, COMBINED reserved cell",
        "threshold": 0.80,
        "cell": {"canon_test_g2": len(canon), "g2ext": len(ext),
                 "n_effective": len(cell), "skipped_dedup_merged": skipped,
                 "layouts": len(hashes_c | hashes_e)},
        "protocol": ("B144 GATE-2 checkpoints unchanged; arms frozen from val; ONE read; "
                     "mx.random.seed(20000+seed); no selection on test cells at any point"),
        "overall": {**summary, "hierarchical_bootstrap": ci},
        "success_by_d_star_band": by_band,
        "per_seed": {str(s): per_seed[s] for s in range(5)},
        "PASS": summary["success_rate"] >= 0.80,
        "pilot_disclosure_ref": "gate3-PILOT-premature.json (premature g2ext-only read, disclosed 7aa7fda)",
    }
    with open(f"{OUT}/gate3-official.json", "w") as fh:
        json.dump(verdict, fh, indent=2, default=str)
    print(json.dumps({k: verdict[k] for k in ("overall", "PASS")}, indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dry-n", type=int, default=10)
    args = ap.parse_args()
    main(args.dry_run, args.dry_n)
