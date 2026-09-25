#!/usr/bin/env python3
"""Mesure bornée de l'ADAPTATION aux réglages OFFICIELS (correction GO lead 16:02:54, cond. (d)).

1 cellule RÉELLE, chemin runner officiel : `read_once` → `from_read_once` → `preload_store` →
`finetune_episode_budget_preloaded` pour k=64/128/256, **updates=2000, batch 64, CPU**, sur un
**fichier DEV dédié N=256 (seed DEV neuf)** — PAS un des 10 fichiers officiels. **SANS eval**
(wall-time uniquement, aucun peeking). Extrapolation ×120 cellules sur cette base.

Report-only, aucun seuil figé, aucun scellé/test2. Usage :
  python scripts/profile_adaptation_official.py --report reports/adaptation-official-dev-profile.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_REPO = Path(__file__).resolve().parents[1]
_DEV_STORE = _REPO / "artifacts" / "inventory-siw-dev-layouts.json"


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _generate_couples(path: str, n: int, seed: int, n_layouts: int = 8):
    from ucm.data.siw_pipeline import generate_adaptation_episodes_rstar
    from ucm.env.siw import SIWLayout
    from ucm.v1.data_adapter import _spec_of
    store = json.load(open(_DEV_STORE))
    layouts = [SIWLayout(_spec_of(e)) for e in list(store.values())[:n_layouts]]
    eps, meta = generate_adaptation_episodes_rstar(layouts, seed=seed, n_episodes=n, d0_band=(2, 4))
    with open(path, "w") as fh:
        for e in eps:
            fh.write(json.dumps(e, sort_keys=True, default=str) + "\n")
    return meta


def main() -> int:
    ap_ = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--episodes", type=int, default=256, help="N=256 (fichier DEV dédié)")
    ap_.add_argument("--seed", type=int, default=20260929, help="seed DEV NEUF (≠ seeds officiels)")
    ap_.add_argument("--ks", default="64,128,256")
    ap_.add_argument("--updates", type=int, default=2000)
    ap_.add_argument("--report", default=None)
    args = ap_.parse_args()

    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    from ucm.model.siw_model import make_siw_model
    from ucm.v1.episode_budget import (finetune_episode_budget_preloaded,
                                       from_read_once, preload_store)
    from ucm.v1.sealed_reader import SealedOpenRegistry
    from ucm.v1.transfer import CoverageTracker, FinetuneConfig

    ks = [int(k) for k in args.ks.split(",")]
    workdir = tempfile.mkdtemp(prefix="adaptation-profile-")   # tmp DEV uniquement
    t0 = time.perf_counter()
    meta = _generate_couples(os.path.join(workdir, "couples.jsonl"), n=args.episodes, seed=args.seed)
    shutil.copy(_DEV_STORE, os.path.join(workdir, "store.json"))
    gen_s = time.perf_counter() - t0
    couples = os.path.join(workdir, "couples.jsonl")
    store = os.path.join(workdir, "store.json")
    couples_sha = _sha256_file(couples)
    store_sha = _sha256_file(store)
    print(f"[fixture] {args.episodes} épisodes DEV seed={args.seed}, gen={gen_s:.2f}s, "
          f"sha={couples_sha[:16]}", flush=True)

    reg = SealedOpenRegistry()
    _fmt, lines, reader_sha = reg.read_once(couples, expected_hash=couples_sha)
    verified = from_read_once(lines, reader_sha, couples_sha)
    store_mapping, loaded_store_sha = preload_store(store, store_sha)
    model = make_siw_model()
    cov = CoverageTracker()

    runs = []
    for k in ks:
        cfg = FinetuneConfig(updates=args.updates, seed=0)   # batch 64 = défaut officiel
        mx.random.seed(9000)
        t0 = time.perf_counter()
        print(f"[adapt ] k={k} updates={args.updates} … START", flush=True)
        res = finetune_episode_budget_preloaded(
            model, reader_sha, couples_sha, verified, k, cfg, cov,
            f"adapt-profile@k={k}", store_mapping, loaded_store_sha)
        wall = time.perf_counter() - t0
        meta_k = res.get("episode_budget_metadata", {})
        runs.append({
            "k": k, "updates": args.updates, "batch_size": cfg.batch_size,
            "wall_s": round(wall, 3), "per_update_ms": round(1000 * wall / args.updates, 3),
            "n_episodes": meta_k.get("n_episodes"), "n_records": meta_k.get("n_records"),
            "reported_wall_s": res.get("wall_s"),
        })
        print(f"[adapt ] k={k} DONE wall={wall:.1f}s per-update={1000*wall/args.updates:.1f}ms "
              f"n_eps={meta_k.get('n_episodes')} n_rec={meta_k.get('n_records')}", flush=True)

    total_s = sum(40 * args.updates * (r["per_update_ms"] / 1000) for r in runs)  # 40 cellules/k
    report = {
        "kind": "adaptation-official-dev-profile",
        "path": "read_once → from_read_once → preload_store → finetune_episode_budget_preloaded "
                "(chemin runner_official ; runner_confirm/main intacts)",
        "fixture": {"episodes": args.episodes, "seed": args.seed, "gen_wall_s": round(gen_s, 2),
                    "couples_sha256": couples_sha, "store": "DEV (copié tmp)",
                    "note": "fichier DEV dédié N=256, seed DEV NEUF — PAS un des 10 fichiers officiels ; "
                            "aucun test2/scellé ; SANS eval (wall-time uniquement)"},
        "protocol": {"ks": ks, "updates": args.updates, "batch_size": 64, "device": "cpu"},
        "runs": runs,
        "extrapolation_120_cellules": {
            "hypothese": "40 cellules par k × 2000 updates × per-update mesuré sur cellule RÉELLE",
            "total_heures": round(total_s / 3600, 2),
            "detail_heures_par_k": {f"k={r['k']}": round(40 * args.updates *
                                                          (r["per_update_ms"] / 1000) / 3600, 2)
                                    for r in runs},
        },
        "limits": [
            "mesure sous charge machine partagée (contention non contrôlée) — borne haute probable",
            "1 cellule (3 k) mais un seul seed/arm ; modèle partagé entre les k (accumulation d'état)",
            "batch 64 = défaut officiel ; CPU ; sans eval",
            "aucun scellé/test2 ; aucune optimisation ; report-only ; aucun seuil figé",
        ],
    }
    print(f"[extrap] total 120 cellules ≈ {report['extrapolation_120_cellules']['total_heures']} h "
          f"(detail: {report['extrapolation_120_cellules']['detail_heures_par_k']})", flush=True)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"rapport → {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
