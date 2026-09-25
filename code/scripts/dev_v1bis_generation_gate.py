"""Étape 2 V1-bis: génération + gate coverage par fichier (lead 14:45).

Séquence: déclarer seeds → définir espace restreint → générer 10 fichiers →
gate §1 par fichier → §3-bis remplacement si échec → rapport.

Usage: .venv/bin/python scripts/dev_v1bis_generation_gate.py [--out artifacts/dev-v1bis-gen-gate-v01.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ucm.env.siw import generate_siw_layout, SIWState
from ucm.data.siw_pipeline import generate_adaptation_episodes_rstar
from ucm.data.rstar_order import round_robin_episodes, serialize_rr_records, write_rr_couples_file
from ucm.data.writer import seal_manifest

# === (1) SEEDS DÉCLARÉS — TOUS nouveaux, jamais utilisés ===
PRIMARY_SEEDS = (20261030, 20261031, 20261032, 20261033, 20261034,
                 20261035, 20261036, 20261037, 20261038, 20261039)
REPLACEMENT_SEEDS = (20261040, 20261041, 20261042, 20261043, 20261044,
                     20261045, 20261046, 20261047, 20261048, 20261049)

# Réservoir de layouts (comme dimensioning)
RESERVOIR_SEED = 20261010
N_LAYOUTS = 40
N_EPISODES = 256

# Réfs TRAIN pour calibration de l'espace restreint
REF_TRAIN_SEEDS = (20261013, 20261014, 20261015)
COVERAGE_THRESHOLD = 0.005  # 0.5% (protocole v2)


def label_set(c):
    return tuple(sorted({sem.split(":")[0] for sem in c["optimal_semantic"]}))


def sigma_vec(c):
    sk = c["state_key"]
    return (min(len(sk[1]), 2), min(len(sk[2]), 1), bool(sk[3]), min(len(sk[4]), 1))


def cell_fine(c):
    return (c["goal"]["predicate"],) + sigma_vec(c) + (label_set(c), min(c["d_star"], 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/dev-v1bis-gen-gate-v01.json")
    ap.add_argument("--producer-commit", default=None)
    ap.add_argument("--outdir", default="artifacts/v1bis-gen")
    args = ap.parse_args()

    producer = args.producer_commit or subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()

    # === (2) ESPACE RESTREINT: cellules >=0.5% sur union réfs TRAIN ===
    rng0 = random.Random(RESERVOIR_SEED)
    reservoir = [generate_siw_layout(rng0, rng0.randint(2, 5), with_dialog=rng0.random() < 0.5,
                                     profile="small") for _ in range(N_LAYOUTS)]

    union_cells = Counter()
    union_n = 0
    for rs in REF_TRAIN_SEEDS:
        pool_rng = random.Random(rs + 900)
        pool = [generate_siw_layout(pool_rng, pool_rng.randint(2, 5),
                with_dialog=pool_rng.random() < 0.5, profile="small") for _ in range(N_LAYOUTS)]
        couples, _ = generate_adaptation_episodes_rstar(pool, seed=rs, n_episodes=120)
        union_cells.update(cell_fine(c) for c in couples)
        union_n += len(couples)

    restricted_space = {k: v / union_n for k, v in union_cells.items() if v / union_n >= COVERAGE_THRESHOLD}
    excluded = len(union_cells) - len(restricted_space)
    print(f"espace restreint: {len(restricted_space)} cellules >= {COVERAGE_THRESHOLD*100}% ({excluded} exclues, union n={union_n})")

    # === (3) GÉNÉRER 10 FICHIERS + (4) GATE PAR FICHIER ===
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    files = []
    replacements_used = 0
    for i, seed in enumerate(PRIMARY_SEEDS):
        used_seed = seed
        result = None
        for attempt in range(2):  # primaire puis 1 remplacement max
            couples, stats = generate_adaptation_episodes_rstar(reservoir, seed=used_seed, n_episodes=N_EPISODES)
            by_ep = defaultdict(list)
            for c in couples:
                by_ep[c["episode_id"]].append(c)

            # writer RR
            fname = f"v1bis-gen-{used_seed}.jsonl"
            fpath = outdir / fname
            rep = write_rr_couples_file(couples, by_ep, str(fpath))

            # préfixes k=64/128/256
            ep_order = round_robin_episodes(by_ep, sorted(by_ep))
            prefix_hashes = {}
            for k in (64, 128, 256):
                pf = [c for eid in ep_order[:k] for c in by_ep[eid]]
                prefix_hashes[str(k)] = {
                    "sha256": hashlib.sha256(serialize_rr_records(pf)).hexdigest(),
                    "records": len(pf),
                }

            # gate §1: coverage sur l'espace restreint
            prefix_cells = Counter(cell_fine(c) for c in couples)
            missing = [(k, restricted_space[k]) for k in restricted_space
                       if restricted_space[k] > 0 and prefix_cells.get(k, 0) == 0]
            gate_pass = len(missing) == 0

            result = {
                "file_index": i, "seed": used_seed,
                "is_replacement": attempt > 0,
                "replacement_of": PRIMARY_SEEDS[i] if attempt > 0 else None,
                "path": str(fpath),
                "episodes": N_EPISODES, "records": rep["records"],
                "sha256_full": rep["sha256_write_stream"],
                "prefixes": prefix_hashes,
                "gate": {
                    "pass": gate_pass,
                    "missing_cells": len(missing),
                    "missing_mass_pct": round(100 * sum(m for _, m in missing), 3),
                    "missing_detail": [{"cell": str(k), "ref_mass": round(m * 100, 3)} for k, m in missing[:5]],
                    "journal_before_training": True,
                },
            }
            if gate_pass:
                break
            elif attempt == 0 and replacements_used < len(REPLACEMENT_SEEDS):
                print(f"  seed {used_seed}: GATE FAIL ({len(missing)} manquantes) → remplacement")
                replacements_used += 1
                used_seed = REPLACEMENT_SEEDS[replacements_used - 1]
            else:
                break
        files.append(result)
        status = "PASS" if result["gate"]["pass"] else "FAIL"
        print(f"  [{i+1}/10] seed {result['seed']}: {status} "
              f"({result['gate']['missing_cells']} manquantes, masse {result['gate']['missing_mass_pct']}%)")

    # résumé
    n_pass = sum(1 for f in files if f["gate"]["pass"])
    art = seal_manifest({
        "schema": "ucm-siw-v1bis-gen-gate/0.1",
        "what": "étape 2 V1-bis: génération 10 fichiers + gate coverage (lead 14:45)",
        "producer": {"code_commit": producer, "script": "scripts/dev_v1bis_generation_gate.py"},
        "seeds_declared": {
            "primary": list(PRIMARY_SEEDS),
            "replacements": list(REPLACEMENT_SEEDS),
            "all_new": True,
            "never_used": "aucun de ces seeds n'apparaît dans aucun artefact DEV antérieur",
        },
        "restricted_space": {
            "threshold_pct": COVERAGE_THRESHOLD * 100,
            "calibration": "union des réfs TRAIN " + str(list(REF_TRAIN_SEEDS)),
            "n_cells": len(restricted_space),
            "n_excluded": excluded,
            "union_n": union_n,
            "cells": {str(k): round(v, 5) for k, v in sorted(restricted_space.items())},
        },
        "files": files,
        "summary": {
            "n_files": len(files), "n_pass": n_pass, "n_fail": len(files) - n_pass,
            "replacements_used": replacements_used,
        },
        "no_training": True, "no_test2": True,
        "hold": "génération DEV uniquement, aucun entraînement, aucun scellé test2",
    })
    Path(args.out).write_text(json.dumps(art, sort_keys=True, indent=2))
    print(f"\n→ {args.out} (scellé {art['manifest_sha256'][:12]})")
    print(f"gate: {n_pass}/10 PASS, {replacements_used} remplacements")


if __name__ == "__main__":
    main()
