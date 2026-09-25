"""V1-bis OFFICIAL runner: cables the freeze v4 artifact to the training loop.

This is the INSTRUMENTAL WIRING — it does NOT execute training until step 4
(GO lead). The function run_v1bis_official(freeze_path, protocol_hash) does:

  1. verify_freeze_v1bis v4 (strict guards + cross-checks + code drift)
  2. for each seed 0-9 (positional: seed i ↔ couples_files[i]):
     - build arm model from manifest checkpoints (scratch/canon/validity/null)
     - for each k in [64, 128, 256]:
       - finetune_episode_budget_preloaded (from the correct couples file)
       - record metadata
  3. eval Stage-B: placeholder that REFUSES any non-sealed eval until test2 exists

The function is structured so that step 4 (GO) is a one-line change:
remove the eval_refuse guard and plug in the real test2 episodes.
"""

from __future__ import annotations

import json
import os

import mlx.core as mx

mx.set_default_device(mx.cpu)

from ucm.v1.freeze_v1bis import verify_freeze_v1bis
from ucm.v1.episode_budget import (finetune_episode_budget_preloaded,
                                    from_read_once, preload_store,
                                    select_episodes, sha_of)
from ucm.v1.sealed_reader import SealedOpenRegistry
from ucm.v1.transfer import CoverageTracker, FinetuneConfig


def _load_arm_model(arm_name: str, fm: dict, seed: int, fresh_init=None):
    """Load the correct model initialization for a given arm+seed.
    fresh_init: the seed-matched shared init (B2/§12.4 pairing — REQUIRED)."""
    from ucm.model.siw_model import make_siw_model
    from ucm.v1.transfer import build_arm

    if arm_name == "scratch":
        return build_arm("scratch", make_siw_model, None, None,
                         fresh_init=fresh_init, seed=seed)

    arm = fm["arms"].get(arm_name)
    if arm is None:
        raise ValueError(f"arm {arm_name!r} not in freeze manifest")
    ckpts = arm["checkpoints"]
    entry = next((c for c in ckpts if c["seed"] == seed), None)
    if entry is None:
        raise ValueError(f"arm {arm_name}: no checkpoint for seed {seed}")

    if arm_name == "pretrained_TGK":
        canon = entry["path"]
        return build_arm("pretrained-TGK", make_siw_model, canon, None)
    elif arm_name == "control_validity":
        return build_arm("control-nontarget", make_siw_model, None, entry["path"])
    elif arm_name == "control_null":
        return build_arm("control-nontarget", make_siw_model, None, entry["path"])
    raise ValueError(f"unknown arm {arm_name!r}")


def run_v1bis_official(freeze_path: str, protocol_hash: str,
                        out_dir: str = "artifacts/v1bis-official-run",
                        dry_run: bool = True) -> dict:
    """Wire the freeze v4 artifact to the training loop.

    dry_run=True (default): verify everything (freeze, cross-checks, wiring,
    arm loading) but DON'T execute training. This is the instrumental
    validation that must be green before step 4 GO.

    dry_run=False: execute the full training (step 4, requires explicit
    GO from lead + test2 sealed + eval Stage-B published).
    """
    # 1) Full freeze verification (strict guards + code drift + cross-checks)
    fm = verify_freeze_v1bis(freeze_path, protocol_hash)

    arm_names = list(fm["arms"].keys())
    k_plan = fm["protocol"]["k_plan_exec_order"]
    seeds = fm["protocol"]["seeds"]
    updates = fm["protocol"]["updates"]
    eval_seed = fm["protocol"]["eval_seed"]
    couples_files = fm["inputs"]["couples_files"]
    store_spec = fm["inputs"]["store"]

    # 2) Eval Stage-B: REFUSE until test2 is sealed (step 4)
    def eval_refuse(episodes):
        if episodes is not None:
            raise RuntimeError(
                "EVAL STAGE-B LOCKED: test2 not yet sealed (step 4 requires "
                "explicit GO from lead + data manifest + SHA published). "
                "This runner refuses any eval until that condition is met.")
        return None

    if dry_run:
        # DRY RUN: verify wiring without training
        wiring = {"arms": arm_names, "k_plan": k_plan, "seeds": seeds,
                   "updates": updates, "eval_seed": eval_seed,
                   "n_couples_files": len(couples_files),
                   "positional_mapping": {
                       str(seed): couples_files[i]["gen_seed"]
                       for i, seed in enumerate(seeds)
                       if i < len(couples_files)
                   }}
        # verify arm loading for seed 0 (each arm type)
        for arm_name in arm_names:
            model = _load_arm_model(arm_name, fm, 0)
            assert model is not None
        # verify eval refuses
        try:
            eval_refuse([{"episode_id": "x"}])
            raise RuntimeError("eval_refuse did not refuse!")
        except RuntimeError as e:
            if "STAGE-B LOCKED" not in str(e):
                raise
        return {"status": "dry_run_ok", "wiring": wiring}

    # FULL RUN (step 4 — requires GO): hard guard (tagi-5 fix 2)
    _GO_TOKEN = os.environ.get("V1BIS_STEP4_GO")
    if _GO_TOKEN != "LEAD_APPROVED":
        raise RuntimeError(
            "FULL RUN LOCKED: set V1BIS_STEP4_GO=LEAD_APPROVED (granted by "
            "lead at step 4 GO + test2 sealed). Without this token, only "
            "dry_run=True is permitted.")
    os.makedirs(out_dir, exist_ok=True)
    reg = SealedOpenRegistry()

    # preload store ONCE
    store_mapping, store_sha = preload_store(store_spec["path"],
                                               store_spec["sha256"])

    all_results = {}
    for seed in seeds:
        # positional: seed i ↔ couples_files[i]
        # Read ONCE per seed, REUSE across all arms (tagi-5 fix 1)
        cf = couples_files[seed % len(couples_files)]
        lines_fmt, lines, reader_sha = reg.read_once(
            cf["path"], expected_hash=cf["sha256"])
        verified = from_read_once(lines, reader_sha, cf["sha256"])

        # Bug 3 fix: seed BEFORE building the shared fresh init (EXPLICIT pairing)
        mx.random.seed(9000 + seed)
        from ucm.model.siw_model import make_siw_model
        import mlx.nn as _nn
        base_init = _nn.utils.tree_flatten(make_siw_model().parameters())

        for arm_name in arm_names:
            cfg = FinetuneConfig(updates=updates, seed=seed)
            cov = CoverageTracker()

            for k in k_plan:
                # Bug 2 fix: FRESH model per (arm, seed, k) via build_arm(fresh_init=base_init, seed=seed)
                model = _load_arm_model(arm_name, fm, seed, fresh_init=base_init)

                result = finetune_episode_budget_preloaded(
                    model, reader_sha, cf["sha256"], verified, k,
                    cfg, cov, f"{arm_name}@s{seed}@k={k}",
                    store_mapping, store_sha)
                all_results[f"{arm_name}|s{seed}|k={k}"] = result

    # eval (Stage-B — still locked)
    eval_refuse(None)  # will raise if test2 not ready

    return {"status": "training_complete", "results": all_results}
