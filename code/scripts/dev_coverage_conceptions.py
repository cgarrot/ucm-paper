"""Les 3 conceptions coverage (A/B/C) en DEV in-memory (lead 14:37).

A: N accru — courbe couverture/masse manquante par N avec rendement.
B: coarse défendable — bandes structurelles d* déclarées response-to-exploratory.
C: estimand restreint — queues exclues assumées.
Validation croisée: paramètres choisis sur réfs 1-3, validés sur réfs 4-5 jamais vues.

Usage: .venv/bin/python scripts/dev_coverage_conceptions.py [--out artifacts/dev-coverage-conceptions-v01.json]
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ucm.env.siw import generate_siw_layout
from ucm.data.siw_pipeline import generate_adaptation_episodes_rstar
from ucm.data.rstar_order import round_robin_episodes
from ucm.data.writer import seal_manifest

REF_SEEDS_TRAIN = (20261013, 20261014, 20261015)   # choix des paramètres
REF_SEEDS_HOLDOUT = (20261016, 20261017)            # validation jamais vue
PILOT_SEED = 13
N_LAYOUTS = 40
N_PILOT = 512   # maximum pour courbe A
N_REF = 120


def label_set(c):
    return tuple(sorted({sem.split(":")[0] for sem in c["optimal_semantic"]}))


def sigma_vec(c):
    sk = c["state_key"]
    return (min(len(sk[1]), 2), min(len(sk[2]), 1), bool(sk[3]), min(len(sk[4]), 1))


def cell_fine(c):
    return (c["goal"]["predicate"],) + sigma_vec(c) + (label_set(c), min(c["d_star"], 4))


def cell_coarse(c):
    """B: bandes structurelles d* (response-to-exploratory): 0-2 = terminal proche,
    3-4+ = navigation étendue. Déclarée APRÈS avoir vu le coverage fin (pas
    pré-enregistrable rétroactivement — documentée comme telle)."""
    d = c["d_star"]
    band = 0 if d <= 2 else 1
    return (c["goal"]["predicate"],) + sigma_vec(c) + (label_set(c), band)


def cell_restricted(c, excluded_masses):
    """C: estimand restreint — cellules de masse réf < seuil EXCLUES du claim.
    Le seuil est choisi sur réfs TRAIN uniquement, validé sur HOLDOUT."""
    return cell_fine(c)


def coverage(prefix_cells, ref_dist, ref_n):
    """Retourne {covered, missing, missing_mass, missing_cells}."""
    qc = Counter(prefix_cells)
    missing = [(k, ref_dist[k]) for k in ref_dist if ref_dist[k] > 0 and qc.get(k, 0) == 0]
    return {
        "covered": len(ref_dist) - len(missing),
        "total": len(ref_dist),
        "missing": len(missing),
        "missing_mass_pct": round(100 * sum(v for _, v in missing) / ref_n, 3),
        "missing_cells": [str(k) for k, _ in missing[:5]],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/dev-coverage-conceptions-v01.json")
    ap.add_argument("--producer-commit", default=None)
    args = ap.parse_args()

    producer = args.producer_commit or subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()

    rng0 = random.Random(20261010)
    reservoir = [generate_siw_layout(rng0, rng0.randint(2, 5), with_dialog=rng0.random() < 0.5,
                                     profile="small") for _ in range(N_LAYOUTS)]

    # réfs TRAIN (choix) et HOLDOUT (validation)
    refs_train, refs_holdout = {}, {}
    for rs in REF_SEEDS_TRAIN + REF_SEEDS_HOLDOUT:
        pool_rng = random.Random(rs + 900)
        pool = [generate_siw_layout(pool_rng, pool_rng.randint(2, 5),
                with_dialog=pool_rng.random() < 0.5, profile="small") for _ in range(N_LAYOUTS)]
        couples, _ = generate_adaptation_episodes_rstar(pool, seed=rs, n_episodes=N_REF)
        fine = Counter(cell_fine(c) for c in couples)
        coarse = Counter(cell_coarse(c) for c in couples)
        d = {"n": len(couples),
             "fine_dist": {k: v / len(couples) for k, v in fine.items()},
             "coarse_dist": {k: v / len(couples) for k, v in coarse.items()},
             "fine_n": len(fine), "coarse_n": len(coarse)}
        if rs in REF_SEEDS_TRAIN:
            refs_train[rs] = d
        else:
            refs_holdout[rs] = d

    # pilot stream
    couples, _ = generate_adaptation_episodes_rstar(reservoir, seed=PILOT_SEED, n_episodes=N_PILOT)
    by_ep = defaultdict(list)
    for c in couples:
        by_ep[c["episode_id"]].append(c)
    ep_order = round_robin_episodes(by_ep, sorted(by_ep))

    # === A: courbe coverage par N ===
    a_results = []
    prev_cov = 0
    for N in (16, 32, 64, 128, 256, 384, 512):
        prefix = [c for eid in ep_order[:N] for c in by_ep[eid]]
        cells = [cell_fine(c) for c in prefix]
        row = {"N": N, "records": len(cells)}
        for rs, ref in list(refs_train.items())[:1]:  # réf representative pour courbe
            cov = coverage(cells, ref["fine_dist"], ref["n"])
            row.update(cov)
            row["rendement"] = round((cov["covered"] - prev_cov) / max(1, N - (16 if N > 16 else 0)), 4)
        a_results.append(row)
        prev_cov = row.get("covered", 0)

    # === B: coarse vs fine, cross-validated ===
    b_results = []
    for N in (64, 128, 256):
        prefix = [c for eid in ep_order[:N] for c in by_ep[eid]]
        fine_cells = [cell_fine(c) for c in prefix]
        coarse_cells = [cell_coarse(c) for c in prefix]
        for rs, ref in {**refs_train, **refs_holdout}.items():
            split = "train" if rs in refs_train else "holdout"
            b_results.append({
                "N": N, "ref": rs, "split": split,
                "fine": coverage(fine_cells, ref["fine_dist"], ref["n"]),
                "coarse": coverage(coarse_cells, ref["coarse_dist"], ref["n"]),
            })

    # === C: estimand restreint (masse > seuil, seuil choisi sur TRAIN) ===
    # seuil choisi pour que les queues rares (<0.5% sur TRAIN moyen) soient exclues
    C_THRESHOLD = 0.005  # 0.5% — choisi sur TRAIN, response-to-exploratory
    c_results = []
    for N in (64, 128, 256):
        prefix = [c for eid in ep_order[:N] for c in by_ep[eid]]
        cells = [cell_fine(c) for c in prefix]
        for rs, ref in {**refs_train, **refs_holdout}.items():
            split = "train" if rs in refs_train else "holdout"
            # cellules restreintes = celles avec masse réf >= seuil
            restricted_dist = {k: v for k, v in ref["fine_dist"].items() if v >= C_THRESHOLD}
            excluded = len(ref["fine_dist"]) - len(restricted_dist)
            cov = coverage(cells, restricted_dist, ref["n"])
            cov["excluded_cells"] = excluded
            cov["threshold"] = C_THRESHOLD
            c_results.append({"N": N, "ref": rs, "split": split, **cov})

    art = seal_manifest({
        "schema": "ucm-siw-coverage-conceptions/0.1",
        "what": "3 conceptions coverage (A/B/C) DEV in-memory (lead 14:37) — report-only",
        "producer": {"code_commit": producer, "script": "scripts/dev_coverage_conceptions.py"},
        "design": {
            "reservoir": {"seed": 20261010, "layouts": N_LAYOUTS},
            "pilot_seed": PILOT_SEED, "n_pilot_max": N_PILOT,
            "refs_train": list(REF_SEEDS_TRAIN), "refs_holdout": list(REF_SEEDS_HOLDOUT),
            "cross_validation": "paramètres choisis sur TRAIN (1-3), validés sur HOLDOUT (4-5) jamais vues",
            "coarse_declaration": "bandes d* 0-2/3+ = response-to-exploratory (PAS pré-enregistrable rétroactivement — documentée)",
            "c_threshold": "0.5% choisi sur TRAIN — response-to-exploratory",
        },
        "A_n_croissant": a_results,
        "B_coarse_vs_fine": b_results,
        "C_estimand_restreint": c_results,
        "thresholds": "AUCUN — report-only",
        "hold": "DEV in-memory, zéro test2/scellé",
    })
    Path(args.out).write_text(json.dumps(art, sort_keys=True, indent=2))
    print(f"→ {args.out} (scellé {art['manifest_sha256'][:12]})")
    # synthèse
    print("\n=== A: courbe coverage ===")
    for r in a_results:
        print(f"  N={r['N']}: {r['covered']}/{r['total']} mass_missing={r['missing_mass_pct']}%")
    print("\n=== B: coarse vs fine (holdout N=256) ===")
    for r in b_results:
        if r["N"] == 256 and r["split"] == "holdout":
            print(f"  réf {r['ref']}: fine {r['fine']['covered']}/{r['fine']['total']} | coarse {r['coarse']['covered']}/{r['coarse']['total']}")
    print("\n=== C: restreint (holdout N=256) ===")
    for r in c_results:
        if r["N"] == 256 and r["split"] == "holdout":
            print(f"  réf {r['ref']}: {r['covered']}/{r['total']} excl={r['excluded_cells']} mass={r['missing_mass_pct']}%")


if __name__ == "__main__":
    main()
