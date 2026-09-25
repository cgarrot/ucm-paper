#!/usr/bin/env python3
"""Profil DEV borné de `runner_v1bis` (mission OPS, report-only) — AUCUNE optimisation de code.

Mesures (DEV fixture ~40+ épisodes, plan k=8/16/32, 1-2 seeds, updates réduits) :
- temps wall par phase : read (read_once) / adapter (from_read_once + preload_store) /
  prefix_hashes / finetune (par k + total) ; « eval » = ABSENT dans runner_v1bis (noté) ;
- mémoire (ru_maxrss par run, processus enfant isolé) ;
- ratio adapter vs finetune ;
- coût du **bundle opt-in** vs tmp+rename (delta end-to-end, même fixture/seed).

Méthode : wrappers de chronométrage posés dans le namespace `ucm.v1.runner_v1bis` (aucune
modification du code du runner) ; chaque configuration tourne dans un **processus enfant** (RSS
propre) ; fixture et sorties sous `tmp DEV` uniquement. Aucun scellé/test2. Report-only : les
nombres alimentent le dimensionnement budget/calendrier, aucun seuil figé, aucun code optimisé.

Usage :
  python scripts/profile_runner_v1bis.py --report reports/runner-v1bis-dev-profile.json
  (interne : --child <workdir> <seed> <bundle 0|1> <k-plan> <updates>)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_REPO = Path(__file__).resolve().parents[1]
_LAYOUT_STORE = _REPO / "artifacts" / "inventory-siw-dev-layouts.json"
DEFAULTS = {"episodes": 48, "n_layouts": 8, "band": (2, 4), "k_plan": "8,16,32",
            "seeds": "0,1", "updates": 2}


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _generate_fixture(path: str, seed: int = 42, n: int = 48, band=(2, 4), n_layouts: int = 8):
    """Réplique DEV-only de `tests/test_e2e_v08._generate_v08` (mêmes appels)."""
    from ucm.data.siw_pipeline import generate_adaptation_episodes_rstar
    from ucm.env.siw import SIWLayout
    from ucm.v1.data_adapter import _spec_of
    store = json.load(open(_LAYOUT_STORE))
    layouts = [SIWLayout(_spec_of(e)) for e in list(store.values())[:n_layouts]]
    eps, meta = generate_adaptation_episodes_rstar(layouts, seed=seed, n_episodes=n, d0_band=band)
    with open(path, "w") as fh:
        for e in eps:
            fh.write(json.dumps(e, sort_keys=True, default=str) + "\n")
    return meta


def _child(workdir: str, seed: int, bundle: int, k_plan: list[int], updates: int) -> int:
    """Un run instrumenté, dans SON processus (RSS propre). Écrit workdir/child-<seed>-<b>.json."""
    import ucm.v1.runner_v1bis as rv
    from ucm.v1.runner_v1bis import run_v1bis
    from ucm.v1 import episode_budget as eb
    from ucm.v1.sealed_reader import SealedOpenRegistry

    couples = os.path.join(workdir, "v08.jsonl")
    store = os.path.join(workdir, "store.json")
    couples_sha = _sha256_file(couples)
    store_sha = _sha256_file(store)

    durations: dict[str, list[float]] = {"read": [], "adapter": [], "prefix": [], "finetune": []}
    per_k: dict[str, float] = {}

    def timed(name: str, fn):
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            out = fn(*args, **kwargs)
            durations[name].append(time.perf_counter() - t0)
            return out
        return wrapper

    def timed_finetune(fn):
        def wrapper(model, reader_sha, expected_sha, verified, k, cfg, cov, arm, store_mapping, loaded_sha):
            t0 = time.perf_counter()
            out = fn(model, reader_sha, expected_sha, verified, k, cfg, cov, arm, store_mapping, loaded_sha)
            dt = time.perf_counter() - t0
            durations["finetune"].append(dt)
            per_k[f"k={k}"] = dt
            return out
        return wrapper

    # wrappers posés UNIQUEMENT dans le namespace du runner (aucune modif de code)
    rv.from_read_once = timed("adapter", eb.from_read_once)
    rv.preload_store = timed("adapter", eb.preload_store)
    rv.prefix_hashes = timed("prefix", eb.prefix_hashes)
    rv.finetune_episode_budget_preloaded = timed_finetune(eb.finetune_episode_budget_preloaded)
    orig_read_once = SealedOpenRegistry.read_once
    SealedOpenRegistry.read_once = timed("read", orig_read_once)

    t0 = time.perf_counter()
    art = run_v1bis(couples, couples_sha, store, store_sha, k_plan,
                    os.path.join(workdir, f"out-{seed}-{bundle}"), updates, seed,
                    bundle=bool(bundle))
    total = time.perf_counter() - t0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = int(rss) if sys.platform == "darwin" else int(rss) * 1024

    out = {
        "seed": seed, "bundle": bool(bundle), "k_plan": k_plan, "updates": updates,
        "total_wall_s": round(total, 3),
        "phases_s": {k: round(sum(v), 4) for k, v in durations.items()},
        "finetune_per_k_s": {k: round(v, 4) for k, v in per_k.items()},
        "finetune_reported_wall_s": {f"k={k}": art["cells"][f"k={k}"]["finetune"].get("wall_s")
                                     for k in k_plan},
        "rss_mb": round(rss_bytes / 1e6, 1),
        "cells": {f"k={k}": {"n_episodes": art["cells"][f"k={k}"]["n_episodes"],
                                 "n_records": art["cells"][f"k={k}"]["n_records"]} for k in k_plan},
        "full_sha256": art["full_sha256"],
    }
    Path(workdir, f"child-{seed}-{bundle}.json").write_text(json.dumps(out, indent=2))
    print(f"[child] seed={seed} bundle={bundle} total={out['total_wall_s']}s "
          f"phases={out['phases_s']} rss={out['rss_mb']}MB", flush=True)
    return 0


def main() -> int:
    ap_ = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--child", nargs=5, metavar=("WORKDIR", "SEED", "BUNDLE", "KPLAN", "UPDATES"))
    ap_.add_argument("--episodes", type=int, default=DEFAULTS["episodes"])
    ap_.add_argument("--n-layouts", type=int, default=DEFAULTS["n_layouts"])
    ap_.add_argument("--k-plan", default=DEFAULTS["k_plan"])
    ap_.add_argument("--seeds", default=DEFAULTS["seeds"])
    ap_.add_argument("--updates", type=int, default=DEFAULTS["updates"])
    ap_.add_argument("--report", default=None)
    args = ap_.parse_args()
    if args.child:
        return _child(args.child[0], int(args.child[1]), int(args.child[2]),
                      [int(k) for k in args.child[3].split(",")], int(args.child[4]))

    k_plan = [int(k) for k in args.k_plan.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    workdir = tempfile.mkdtemp(prefix="runner-v1bis-profile-")   # tmp DEV uniquement

    t_fix0 = time.perf_counter()
    meta = _generate_fixture(os.path.join(workdir, "v08.jsonl"),
                             n=args.episodes, n_layouts=args.n_layouts)
    shutil.copy(_LAYOUT_STORE, os.path.join(workdir, "store.json"))
    fix_s = time.perf_counter() - t_fix0

    runs: list[dict] = []
    for seed in seeds:
        modes = (0, 1) if seed == seeds[0] else (0,)   # bundle testé sur le 1er seed uniquement
        for bundle in modes:
            cmd = [sys.executable, str(Path(__file__).resolve()), "--child",
                   workdir, str(seed), str(bundle), args.k_plan, str(args.updates)]
            p = subprocess.run(cmd, capture_output=True, text=True)
            if p.returncode != 0:
                raise SystemExit(f"child échec (seed={seed}, bundle={bundle}):\n{p.stderr[-2000:]}")
            runs.append(json.loads(Path(workdir, f"child-{seed}-{bundle}.json").read_text()))

    base = next(r for r in runs if r["seed"] == seeds[0] and not r["bundle"])
    bundled = next((r for r in runs if r["bundle"]), None)
    report = {
        "kind": "runner-v1bis-dev-profile",
        "runner": "ucm/v1/runner_v1bis.py (post-run-only; aucune optimisation de code)",
        "fixture": {"path": "tmp DEV (mkdtemp)", "episodes_requested": args.episodes,
                    "n_layouts": args.n_layouts, "band": [2, 4],
                    "generator": "generate_adaptation_episodes_rstar (DEV)",
                    "store": "artifacts/inventory-siw-dev-layouts.json (copié tmp)",
                    "couples_sha256": _sha256_file(os.path.join(workdir, "v08.jsonl")),
                    "generation_wall_s": round(fix_s, 3), "meta": meta},
        "plan": {"k_plan": k_plan, "seeds": seeds, "updates": args.updates,
                 "device": "cpu (runner_v1bis fixe mx.set_default_device(mx.cpu))"},
        "runs": runs,
        "ratios": {
            "adapter_vs_finetune": round(base["phases_s"]["adapter"] /
                                         max(base["phases_s"]["finetune"], 1e-9), 4),
            "read_vs_finetune": round(base["phases_s"]["read"] /
                                      max(base["phases_s"]["finetune"], 1e-9), 4),
            "prefix_vs_finetune": round(base["phases_s"]["prefix"] /
                                        max(base["phases_s"]["finetune"], 1e-9), 4),
        },
        "bundle_optin": None if bundled is None else {
            "total_bundle_s": bundled["total_wall_s"],
            "total_tmp_rename_s": base["total_wall_s"],
            "delta_s": round(bundled["total_wall_s"] - base["total_wall_s"], 3),
            "delta_pct_of_total": round(100 * (bundled["total_wall_s"] - base["total_wall_s"]) /
                                        max(base["total_wall_s"], 1e-9), 2),
            "note": "bundle=False = tmp+fsync+rename DEV ; bundle=True = capability_probe + "
                    "BundleWriter (manifest fsync in-bundle) + pointeur hardlink no-replace.",
        },
        "limits": [
            "eval ABSENT de runner_v1bis (aucune phase d'évaluation dans ce runner)",
            "updates réduits (profil de dimensionnement, pas un run de performance officiel)",
            "CPU uniquement (device fixé par le runner) ; machine partagée (charge non contrôlée)",
            "aucun seuil figé ; rapport report-only (post-run-only, aucune optimisation)",
            "aucun scellé/test2 ; fixture+sorties sous tmp DEV uniquement",
        ],
        "workdir": workdir,
    }
    for r in runs:
        print(f"[run ] seed={r['seed']} bundle={int(r['bundle'])} total={r['total_wall_s']}s "
              f"read={r['phases_s']['read']} adapter={r['phases_s']['adapter']} "
              f"prefix={r['phases_s']['prefix']} finetune={r['phases_s']['finetune']} "
              f"rss={r['rss_mb']}MB")
    if report["bundle_optin"]:
        b = report["bundle_optin"]
        print(f"[bundle] tmp+rename={b['total_tmp_rename_s']}s bundle={b['total_bundle_s']}s "
              f"delta={b['delta_s']}s ({b['delta_pct_of_total']}%)")
    print(f"[ratio] adapter/finetune={report['ratios']['adapter_vs_finetune']}")
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"rapport → {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
