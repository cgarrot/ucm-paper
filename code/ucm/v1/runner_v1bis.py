"""V1-bis DEV runner (global opt-in) — lead 13:43 mission.

Consumes the preloaded path end-to-end:
    read_once couples (SHA expected) → from_read_once → preload_store (1×)
    → for each k in plan: finetune_episode_budget_preloaded
    → artifact JSON (tmp+rename DEV)

runner_confirm.main is UNTOUCHED. No sealed/test2. Multi-k without data file
re-open. Wrong SHA refuses before any training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile

import mlx.core as mx

mx.set_default_device(mx.cpu)

from ucm.v1.episode_budget import (finetune_episode_budget_preloaded,
                                    from_read_once, preload_store,
                                    prefix_hashes, sha_of)
from ucm.v1.sealed_reader import SealedOpenRegistry
from ucm.v1.transfer import CoverageTracker, FinetuneConfig


def run_v1bis(couples_path: str, couples_sha: str,
              store_path: str, store_sha: str,
              k_plan: list[int], out_dir: str,
              updates: int = 1, seed: int = 0,
              bundle: bool = False,
              freeze_manifest: str | None = None,
              protocol_hash: str | None = None,
              eval_episodes: list[dict] | None = None,
              eval_seed: int = 50_000) -> dict:
    """Full V1-bis DEV run: one read of couples + one read of store, then
    multi-k training without any file re-open. Returns the artifact dict.
    If freeze_manifest+protocol_hash: verifies code drift BEFORE any read."""
    if freeze_manifest and protocol_hash:
        from ucm.v1.freeze_v1bis import verify_freeze_v1bis
        verify_freeze_v1bis(freeze_manifest, protocol_hash,
                            couples_sha=couples_sha, store_sha=store_sha,
                            k_plan=k_plan, seed=seed, updates=updates)
    elif freeze_manifest or protocol_hash:
        raise SystemExit("freeze-manifest and protocol-hash must be used together")
    os.makedirs(out_dir, exist_ok=True)
    reg = SealedOpenRegistry()

    # 1) Read couples ONCE (SHA validated)
    fmt, lines, reader_sha = reg.read_once(couples_path, expected_hash=couples_sha)
    assert fmt == "couples", f"unexpected format {fmt}"

    # 2) Verify canonical + expected SHA (from_read_once refuses if wrong)
    verified = from_read_once(lines, reader_sha, couples_sha)

    # 3) Pre-load store ONCE
    store_mapping, loaded_store_sha = preload_store(store_path, store_sha)

    # 4) Compute prefix hashes for ALL k in plan (single pass on lines)
    prefix_shas = prefix_hashes(verified, k_plan)
    full_sha = prefix_hashes(verified, [max(k_plan)])[max(k_plan)]

    # 5) Multi-k training: no file re-open
    mx.random.seed(9000 + seed)
    from ucm.model.siw_model import make_siw_model
    model = make_siw_model()
    cfg = FinetuneConfig(updates=updates, seed=seed)
    cov = CoverageTracker()
    cells = {}
    for k in k_plan:
        result = finetune_episode_budget_preloaded(
            model, reader_sha, couples_sha, verified, k,
            cfg, cov, f"v1bis@k={k}", store_mapping, loaded_store_sha)
        meta = result.get("episode_budget_metadata", {})
        cells[f"k={k}"] = {
            "n_episodes": meta.get("n_episodes", k),
            "n_records": meta.get("n_records", 0),
            "episode_ids": meta.get("episode_ids", []),
            "per_episode_record_counts": meta.get("per_episode_record_counts", {}),
            "prefix_sha256": prefix_shas.get(k, ""),
            "finetune": {kk: result.get(kk) for kk in ("updates", "wall_s") if kk in result},
        }

        # EVAL closed-loop (14:27): per-episode raw, STOP on native state
        if eval_episodes:
            from ucm.v1.runner import _build_env
            from ucm.v1.policy_siw import SIWModelPolicy
            from ucm.eval.rollout import run_episode
            mx.random.seed(eval_seed + seed)
            pol = SIWModelPolicy(model, name=f"v1bis-eval@k={k}", seed=eval_seed + seed)
            raw_eps = []
            n_success = 0
            for ep in eval_episodes:
                env = _build_env(ep)
                r = run_episode(env, pol, ep["episode_id"],
                                eval_seed + seed, f"v1bis-eval@k={k}",
                                d_star=ep.get("d_star"))
                raw_eps.append({
                    "arm": f"v1bis@k={k}", "train_seed": seed,
                    "eval_seed": eval_seed + seed, "k": k,
                    "episode_id": r.episode_id, "layout_hash": r.layout_id,
                    "goal_type": r.goal_type, "success": r.success,
                    "outcome": r.outcome, "length": r.length,
                    "n_invalid": r.n_invalid,
                    "goal_reached_without_stop": r.goal_reached_without_stop,
                    "d_star": r.d_star, "L_star": r.L_star,
                })
                n_success += r.success
            cells[f"k={k}"]["eval"] = {
                "n_episodes": len(raw_eps),
                "success_rate": n_success / len(raw_eps) if raw_eps else None,
                "raw": raw_eps,
            }

    # 6) Artifact (tmp + rename)
    artifact = {
        "runner": "v1bis-dev",
        "couples": {"path": couples_path, "sha256": couples_sha,
                     "reader_sha256": reader_sha},
        "store": {"path": store_path, "sha256": store_sha,
                   "loaded_sha256": loaded_store_sha},
        "full_sha256": full_sha,
        "k_plan": k_plan,
        "cells": cells,
        "coverage": cov.coverage(),
        "seed": seed,
        "eval_seed": eval_seed,
        "updates": updates,
    }
    if bundle:
        artifact["bundle_published"] = True  # BEFORE ALL serialization (14:02)
    out_path = os.path.join(out_dir, "v1bis-artifact.json")
    fd, tmp = tempfile.mkstemp(dir=out_dir, suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        json.dump(artifact, fh, indent=2, sort_keys=True, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    os.rename(tmp, out_path)

    # OPT-IN bundle/pointer v9 (lead 13:55): default stays tmp+rename DEV;
    # pass bundle=True to ALSO publish via atomic_publish.BundleWriter
    # (manifest fsynced in-bundle, pointer hardlink no-replace).
    if bundle:
        from ucm.v1.atomic_publish import BundleWriter, capability_probe
        capability_probe(out_dir)  # hardlink+rename capability or raise
        bw = BundleWriter(out_dir, "v1bis")
        bw.add_bytes("v1bis-artifact.json",
                      json.dumps(artifact, indent=2, sort_keys=True,
                                 default=str).encode())
        bw.finalize()
        bw.publish_pointer(os.path.join(out_dir, "v1bis.pointer"))

    return artifact


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--couples-file", required=True)
    ap.add_argument("--couples-sha", required=True)
    ap.add_argument("--store-file", required=True)
    ap.add_argument("--store-sha", required=True)
    ap.add_argument("--k-plan", default="1,2,3")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--updates", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--freeze-manifest", default=None)
    ap.add_argument("--protocol-hash", default=None)
    args = ap.parse_args()
    art = run_v1bis(args.couples_file, args.couples_sha,
                    args.store_file, args.store_sha,
                    [int(k) for k in args.k_plan.split(",")],
                    args.out_dir, args.updates, args.seed,
                    freeze_manifest=args.freeze_manifest,
                    protocol_hash=args.protocol_hash)
    print(json.dumps({"out": os.path.join(args.out_dir, "v1bis-artifact.json"),
                       "cells": list(art["cells"].keys())}))
