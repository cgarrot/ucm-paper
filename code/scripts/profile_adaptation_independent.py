#!/usr/bin/env python3
"""Re-mesure budget — PROTOCOLE INDÉPENDANT (ordre lead 17:57:52) : **modèle FRAIS par k**.

Différence vs `profile_adaptation_official.py` (ancien protocole cumulatif, f80629e) : ici un
`make_siw_model()` neuf est construit pour CHAQUE k (bornes par-k indépendantes, pas d'état
cumulé). Même fixture DEV dédiée N=256 seed `20260929` (sha `137687fd…`), updates=2000,
batch 64, CPU, **sans eval**. 2-3 réps si possible. **Charge journalisée dans le script**
(`os.getloadavg()` avant/après chaque k) — répond à la demande d'auditabilité tagi-5.

Report-only, aucun scellé/test2, aucun seuil figé. Usage :
  python scripts/profile_adaptation_independent.py --reps 2 --report reports/adaptation-independent-profile.json
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
    ap_.add_argument("--episodes", type=int, default=256)
    ap_.add_argument("--seed", type=int, default=20260929)
    ap_.add_argument("--ks", default="64,128,256")
    ap_.add_argument("--updates", type=int, default=2000)
    ap_.add_argument("--reps", type=int, default=2)
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
    workdir = tempfile.mkdtemp(prefix="adaptation-independent-")   # tmp DEV uniquement
    t0 = time.perf_counter()
    _generate_couples(os.path.join(workdir, "couples.jsonl"), n=args.episodes, seed=args.seed)
    shutil.copy(_DEV_STORE, os.path.join(workdir, "store.json"))
    gen_s = time.perf_counter() - t0
    couples = os.path.join(workdir, "couples.jsonl")
    store = os.path.join(workdir, "store.json")
    couples_sha = _sha256_file(couples)
    store_sha = _sha256_file(store)
    print(f"[fixture] {args.episodes} ép. seed={args.seed} gen={gen_s:.2f}s sha={couples_sha[:16]}",
          flush=True)

    reg = SealedOpenRegistry()
    _fmt, lines, reader_sha = reg.read_once(couples, expected_hash=couples_sha)
    verified = from_read_once(lines, reader_sha, couples_sha)
    store_mapping, loaded_store_sha = preload_store(store, store_sha)

    samples = []
    for rep in range(1, args.reps + 1):
        for k in ks:
            model = make_siw_model()          # ← FRAIS par k (protocole indépendant)
            cfg = FinetuneConfig(updates=args.updates, seed=0)
            cov = CoverageTracker()
            mx.random.seed(9000)
            load_before = [round(x, 2) for x in os.getloadavg()]
            t0 = time.perf_counter()
            print(f"[indep ] rep={rep} k={k} … START (fresh model, load {load_before})", flush=True)
            res = finetune_episode_budget_preloaded(
                model, reader_sha, couples_sha, verified, k, cfg, cov,
                f"adapt-indep@rep{rep}@k={k}", store_mapping, loaded_store_sha)
            wall = time.perf_counter() - t0
            load_after = [round(x, 2) for x in os.getloadavg()]
            meta_k = res.get("episode_budget_metadata", {})
            samples.append({
                "rep": rep, "k": k, "updates": args.updates, "batch_size": cfg.batch_size,
                "wall_s": round(wall, 3), "per_update_ms": round(1000 * wall / args.updates, 3),
                "n_episodes": meta_k.get("n_episodes"), "n_records": meta_k.get("n_records"),
                "load_before": load_before, "load_after": load_after,
                "fresh_model_per_k": True,
            })
            print(f"[indep ] rep={rep} k={k} DONE wall={wall:.1f}s "
                  f"per-update={1000*wall/args.updates:.1f}ms load_after={load_after}", flush=True)

    # agrégats par k (mean/min/max + origine du max) — primaire = max (conservatrice)
    per_k = {}
    for k in ks:
        xs = [s for s in samples if s["k"] == k]
        walls = [s["wall_s"] for s in xs]
        mx_i = walls.index(max(walls))
        per_k[f"k={k}"] = {
            "walls": walls, "mean_s": round(sum(walls) / len(walls), 1),
            "min_s": min(walls), "max_s": max(walls),
            "max_origin": {"rep": xs[mx_i]["rep"], "wall_s": xs[mx_i]["wall_s"]},
            "reps": len(xs),
        }
    # extrapolation ×120 : 40 cellules/k, somme des max par k (primaire conservatrice)
    total_max = sum(40 * (per_k[f"k={k}"]["max_s"]) for k in ks)
    total_mean = sum(40 * (per_k[f"k={k}"]["mean_s"]) for k in ks)
    total_min = sum(40 * (per_k[f"k={k}"]["min_s"]) for k in ks)
    report = {
        "kind": "adaptation-independent-profile",
        "protocol": "INDÉPENDANT — modèle FRAIS par k (≠ f80629e cumulatif) ; ordre lead 17:57:52",
        "path": "read_once → from_read_once → preload_store → finetune_episode_budget_preloaded "
                "(chemin runner_official ; runner_confirm/main intacts)",
        "fixture": {"episodes": args.episodes, "seed": args.seed, "gen_wall_s": round(gen_s, 2),
                    "couples_sha256": couples_sha, "store": "DEV (copié tmp)",
                    "note": "fichier DEV dédié N=256 seed neuf — aucun test2/scellé ; SANS eval"},
        "settings": {"ks": ks, "updates": args.updates, "batch_size": 64, "device": "cpu",
                     "reps": args.reps},
        "samples": samples,
        "per_k": per_k,
        "extrapolation_120_cellules": {
            "primaire_max": {"heures": round(total_max / 3600, 2), "detail_per_k_max_s": {
                f"k={k}": per_k[f"k={k}"]["max_s"] for k in ks}},
            "mean": {"heures": round(total_mean / 3600, 2)},
            "min_plancher": {"heures": round(total_min / 3600, 2)},
            "hypothese": "40 cellules/k × wall mesuré (max=primaire conservatrice, min=plancher)",
        },
        "limits": [
            "charge machine partagée (~8-9 pendant la mesure) — le max est la borne haute prudente",
            "charge journalisée par le script (os.getloadavg avant/après chaque k) — pas d'échantillonnage continu",
            "2 réps par défaut ; modèle frais par k (protocole indépendant) ; un seul seed d'arm",
            "aucun scellé/test2 ; aucune optimisation ; report-only ; aucun seuil figé",
        ],
    }
    for r in per_k:
        print(f"[per_k] {r}: walls={per_k[r]['walls']} mean={per_k[r]['mean_s']} "
              f"min={per_k[r]['min_s']} max={per_k[r]['max_s']} (max origin rep {per_k[r]['max_origin']['rep']})")
    print(f"[extrap] ×120 → max(primaire)={report['extrapolation_120_cellules']['primaire_max']['heures']} h "
          f"| mean={report['extrapolation_120_cellules']['mean']['heures']} h "
          f"| min={report['extrapolation_120_cellules']['min_plancher']['heures']} h", flush=True)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"rapport → {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
