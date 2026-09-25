"""Parallel V1-bis runner v2 (Option B, lead 16:06/16:20 corrections).

Delegates ALL merge/sealing to the CANONICAL `ucm.data.interactions_merge`
module (contract-passed, frozen at gel v5). No local merge.

Workers produce entries with REAL sha chains via `entry_seal` (genesis per
worker). The parent calls `merge_worker_logs` from the canonical module.
"""

from __future__ import annotations

import concurrent.futures
import json
import os

import mlx.core as mx

mx.set_default_device(mx.cpu)

from ucm.data.interactions_merge import entry_seal, merge_worker_logs, verify_merge
from ucm.v1.freeze_v1bis import verify_freeze_v1bis
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ucm.v1.episode_budget import (finetune_episode_budget_preloaded,
                                    from_read_once, preload_store)
from ucm.v1.sealed_reader import SealedOpenRegistry
from ucm.v1.transfer import CoverageTracker, FinetuneConfig


def _worker_cell(args):
    """Single (arm, seed) cell in a subprocess. Returns results + worker log
    entries WITH real sha chains (entry_seal, genesis per worker)."""
    arm_name, seed, fm, couples_files_spec, store_spec, k_plan, updates = args
    worker_id = f"{arm_name}-s{seed}"

    # Build log entries with sha chains
    log_entries = []
    prev_seal = "genesis"
    seq = 0

    def log(motive, target="siw-adaptation", k=0):
        nonlocal prev_seal, seq
        entry = {"worker_id": worker_id, "seq": seq, "arm": arm_name,
                 "k": k, "motive": motive, "target": target,
                 "t": _timestamp()}
        seal = entry_seal(entry, prev_seal)
        entry["seal"] = seal
        prev_seal = seal
        seq += 1
        log_entries.append(entry)

    def _timestamp():
        # DETERMINISTIC (tagi-5 B4 fix): derived from worker_id + seq,
        # NOT wall-clock. The field is still metadata, never an order key.
        return f"deterministic:{worker_id}:{seq}"

    log("worker_start")

    from ucm.v1.runner_official import _load_arm_model

    # Read couples (positional: seed i ↔ couples_files[i])
    cf = couples_files_spec[seed % len(couples_files_spec)]
    reg = SealedOpenRegistry()
    fmt, lines, reader_sha = reg.read_once(cf["path"], expected_hash=cf["sha256"])
    verified = from_read_once(lines, reader_sha, cf["sha256"])

    # Preload store (v02 wrapper unwrapped by preload_store — crash GO-run fix)
    store_mapping, store_sha = preload_store(store_spec["path"], store_spec["sha256"])
    log("data_loaded")

    # Fresh init per seed (B2/§12.4 pairing, same discipline as runner_official):
    # seed BEFORE building the shared init, then a FRESH model per (arm, seed, k).
    mx.random.seed(9000 + seed)
    from ucm.model.siw_model import make_siw_model
    import mlx.nn as _nn
    base_init = _nn.utils.tree_flatten(make_siw_model().parameters())

    cells = {}
    for k in k_plan:
        # k-independent cells (Bug 2 fix, parallel path): rebuild per k so
        # k=128 never inherits from k=64 — same protocol as runner_official.
        cfg = FinetuneConfig(updates=updates, seed=seed)
        cov = CoverageTracker()
        model = _load_arm_model(arm_name, fm, seed, fresh_init=base_init)
        result = finetune_episode_budget_preloaded(
            model, reader_sha, cf["sha256"], verified, k,
            cfg, cov, f"{arm_name}@s{seed}@k={k}",
            store_mapping, store_sha)
        meta = result.get("episode_budget_metadata", {})
        cells[f"k={k}"] = {
            "n_episodes": meta.get("n_episodes", k),
            "n_records": meta.get("n_records", 0),
            "prefix_sha256": meta.get("prefix_sha256", {}).get(k, ""),
        }

        # PERSISTENCE (lead 06:53 — run must never depend on caller's print):
        # per-cell model checkpoint, O_EXCL fail-closed (refuses overwrite).
        cells_dir = os.path.join(
            os.environ.get("UCM_RUN_DIR", os.path.join(_REPO, "artifacts/v1bis-run")),
            "cells")
        os.makedirs(cells_dir, exist_ok=True)
        ckpt_path = os.path.join(cells_dir, f"{arm_name}-s{seed}-k{k}.npz")
        if os.path.exists(ckpt_path):
            raise RuntimeError(f"cell checkpoint exists (O_EXCL): {ckpt_path}")
        import mlx.nn as _nnl
        _params = dict(_nnl.utils.tree_flatten(model.parameters()))
        mx.savez(ckpt_path, **_params)

        log(f"cell_complete@k={k}", k=k)

    return {"worker_id": worker_id, "arm": arm_name, "seed": seed,
            "cells": cells, "log_entries": log_entries}


def run_v1bis_parallel(freeze_path: str, protocol_hash: str,
                       n_workers: int = 4,
                       dry_run: bool = True) -> dict:
    """Parallel V1-bis runner using canonical interactions_merge."""
    fm = verify_freeze_v1bis(freeze_path, protocol_hash)

    arm_names = list(fm["arms"].keys())
    k_plan = fm["protocol"]["k_plan_exec_order"]
    seeds = fm["protocol"]["seeds"]
    updates = fm["protocol"]["updates"]
    couples_files = fm["inputs"]["couples_files"]
    store_spec = fm["inputs"]["store"]

    expected_cells = len(arm_names) * len(seeds)

    if dry_run:
        return {"status": "dry_run_ok", "n_workers": min(n_workers, 6),
                "expected_cells": expected_cells,
                "arms": arm_names, "k_plan": k_plan, "seeds": seeds}

    # FULL RUN — GO token required
    token = os.environ.get("V1BIS_STEP4_GO")
    if token != "LEAD_APPROVED":
        raise RuntimeError("FULL RUN LOCKED: V1BIS_STEP4_GO=LEAD_APPROVED required")

    # Build work items
    work_items = []
    for arm_name in arm_names:
        for seed in seeds:
            work_items.append((arm_name, seed, fm, couples_files,
                               store_spec, k_plan, updates))

    # Execute pool
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(n_workers, 6)) as pool:
        results = list(pool.map(_worker_cell, work_items))

    # COMPLETENESS check (lead 16:20 fix 4)
    if len(results) != expected_cells:
        raise RuntimeError(f"INCOMPLETE: {len(results)}/{expected_cells} cells returned")
    for r in results:
        for k in k_plan:
            if f"k={k}" not in r["cells"]:
                raise RuntimeError(f"INCOMPLETE: {r['worker_id']} missing k={k}")

    # Build worker_logs dict for canonical merge
    worker_logs = {}
    for r in results:
        worker_logs[r["worker_id"]] = r["log_entries"]

    # CANONICAL merge (delegates to interactions_merge.py)
    merge = merge_worker_logs(worker_logs)

    # Verify round-trip
    assert verify_merge(merge, worker_logs), "merge verification failed"

    # PERSISTENCE (lead 06:53): full results + merge manifest, O_EXCL. The
    # runner NEVER depends on the caller printing the return value.
    run_dir = os.environ.get("UCM_RUN_DIR", os.path.join(_REPO, "artifacts/v1bis-run"))
    os.makedirs(run_dir, exist_ok=True)
    import datetime as _dt
    ts = os.environ.get("UCM_RUN_TS") or _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    out = {"status": "parallel_complete",
           "n_workers": min(n_workers, 6),
           "n_cells": len(results),
           "merge": merge,
           "cells": {f"{r['arm']}|s{r['seed']}": r["cells"] for r in results}}

    manifest_path = os.path.join(run_dir, f"merge-manifest-{ts}.json")
    _fd = os.open(manifest_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(_fd, "w") as fh:
        json.dump({"merge": merge, "ts": ts}, fh, sort_keys=True, default=str)

    run_path = os.path.join(run_dir, f"v1bis-run-{ts}.json")
    _fd = os.open(run_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(_fd, "w") as fh:
        json.dump(out, fh, sort_keys=True, default=str, indent=2)

    out["persisted_run"] = run_path
    out["persisted_manifest"] = manifest_path
    return out
