"""Pilote DEV R* rev0.7 — script REPRODUCTIBLE (lead 11:14: script dans le repo,
pas un artefact seul). Toutes les métriques exploratoires, AUCUN seuil.

Usage: .venv/bin/python scripts/dev_pilot_rstar.py [--out artifacts/dev-rstar-v072-pilot.json]

Définitions EXPLICITES des clés TV (audit 11:14 point 2):
- TV_ABSTRAITE: cellule = (prédicat, σ=features≤2, κ=action-label, d*≤4) —
  COMPARAISON CROSS-LAYOUT PAR ABSTRACTION (déclarée: les pools disjoints ne
  partagent aucun layout; cette TV mesure la similarité de DISTRIBUTION
  agrégée, PAS l'identité des états).
- TV_BRUTE: cellule = (prédicat, layout_hash, state_key) — sur pools
  disjoints cette TV vaut ~1.0 PAR CONSTRUCTION (aucun recouvrement
  possible): publiée comme garde-fou pour montrer que la comparaison
  meaningful est l'abstraite.
- importance-ESS: w_i = p_ref(cell)/q_prefix(cell) sur la cellule ABSTRAITE;
  ESS/records publié explicitement; coverage FAIL avec comptes de cellules.
Preuves par seed (audit point 1): hash FULL du stream, per_pred, sampler seed
réellement passé — exécuté et publié à chaque run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))

from ucm.data.siw_pipeline import generate_adaptation_episodes_rstar
from ucm.data.writer import seal_manifest
from ucm.env.siw import generate_siw_layout


def dev_pool(seed: int, n: int = 10):
    rng = random.Random(seed)
    return [generate_siw_layout(rng, rng.randint(2, 5), with_dialog=rng.random() < 0.5,
                                profile="small") for _ in range(n)]


def label_type(c):
    """κ du gate (lead 11:18): ENSEMBLE TRIÉ des types de TOUTES les actions
    optimales (label oracle set-valued) — JAMAIS le premier optimal isolé,
    et pas étiqueté 'type de l'exécuté' (l'exécuté est un tirage dans ce set).
    Terminal d0: {STOP} exact."""
    types = tuple(sorted({sem.split(":")[0] for sem in c["optimal_semantic"]}))
    return types


def sigma_vector(c):
    """σ = vecteur TYPÉ (filled, chosen, dialog, submitted), chacun borné —
    ne FUSIONNE pas les types (défaut B3 cachable par sommation coarse)."""
    sk = c["state_key"]
    return (min(len(sk[1]), 2), min(len(sk[2]), 1), bool(sk[3]), min(len(sk[4]), 1))


def cell_abstract(c):
    # cellule PROSPECTIVE: prédicat × σ-vecteur typé × κ-ensemble × d*
    return (c["goal"]["predicate"],) + sigma_vector(c) + (label_type(c), min(c["d_star"], 4))


def cell_raw(c):
    return (c["goal"]["predicate"], c["layout_hash"], str(c["state_key"]))


def stream_proof(couples, stats, seed):
    return {"seed": seed,
            "stream_sha256": hashlib.sha256(
                json.dumps(couples, sort_keys=True).encode()).hexdigest(),
            "per_predicate": stats["per_predicate_episodes"],
            "records": len(couples)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/dev-rstar-v072-pilot.json")
    args = ap.parse_args()

    refs = {}
    for rs in (20261013, 20261014, 20261015):
        pool = dev_pool(rs)
        couples, _ = generate_adaptation_episodes_rstar(pool, seed=rs, n_episodes=40)
        refs[rs] = {"pool": pool, "couples": couples,
                    "abstract": Counter(cell_abstract(c) for c in couples),
                    "raw": Counter(cell_raw(c) for c in couples),
                    "n": len(couples)}

    proofs, pairs = [], []
    pool_pilot = dev_pool(20261010)
    seen_stream_hashes = set()
    for ps in (13, 20261020, 20261021):
        couples, stats = generate_adaptation_episodes_rstar(pool_pilot, seed=ps, n_episodes=40)
        proof = stream_proof(couples, stats, ps)
        # lead 11:16: ASSERT seeds distincts ⇒ records DISTINCTS (fixture fixe)
        assert proof["stream_sha256"] not in seen_stream_hashes, \
            f"PSEUDO-RÉPLICATION: seed {ps} produit un stream identique à un seed précédent !"
        seen_stream_hashes.add(proof["stream_sha256"])
        proofs.append(proof)
        by_ep = defaultdict(list)
        for c in couples:
            by_ep[c["episode_id"]].append(c)
        ep_ids = sorted(by_ep)
        for N in (12, 24, 40):
            prefix = [c for eid in ep_ids[:N] for c in by_ep[eid]]
            pa = Counter(cell_abstract(c) for c in prefix)
            pr = Counter(cell_raw(c) for c in prefix)
            n = len(prefix)
            for rs, ref in refs.items():
                keys = set(pa) | set(ref["abstract"])
                tv_abs = 0.5 * sum(abs(pa.get(k, 0) / n - ref["abstract"].get(k, 0) / ref["n"])
                                   for k in keys)
                rk = set(pr) | set(ref["raw"])
                tv_raw = 0.5 * sum(abs(pr.get(k, 0) / n - ref["raw"].get(k, 0) / ref["n"])
                                   for k in rk)
                # importance ESS sur cellule abstraite
                zero_viol = {k: (ref["abstract"].get(k, 0), pa.get(k, 0))
                             for k in ref["abstract"] if ref["abstract"][k] > 0 and pa.get(k, 0) == 0}
                weights = []
                for c in prefix:
                    k = cell_abstract(c)
                    p = ref["abstract"].get(k, 0) / ref["n"]
                    q = pa[k] / n
                    if q > 0:
                        weights.append(p / q)
                ess = (sum(weights) ** 2 / sum(w * w for w in weights)) if weights and any(weights) else None
                pairs.append({"pilot_seed": ps, "episodes": N, "ref_seed": rs,
                              "tv_abstract_declared": round(tv_abs, 4),
                              "tv_raw_guard": round(tv_raw, 4),
                              "ess_importance": round(ess, 1) if ess else None,
                              "ess_per_record": round(ess / n, 4) if ess else None,
                              "coverage_fail": bool(zero_viol),
                              "zero_violation_cells": len(zero_viol),
                              "zero_violation_ref_mass": round(
                                  sum(v[0] for v in zero_viol.values()) / ref["n"], 4)})

    # PAS de résumé poolé n=9 (pseudo-réplication 11:16): strates séparées
    by_ps = defaultdict(list)   # par seed pilote (3 refs = corrélation, min/max seulement)
    by_rs = defaultdict(list)   # par réf (3 seeds pilotes correlés)
    for r in pairs:
        by_ps[(r["pilot_seed"], r["episodes"])].append(r["tv_abstract_declared"])
        by_rs[(r["ref_seed"], r["episodes"])].append(r["tv_abstract_declared"])
    summary = {
        "per_pilot_seed": {f"ps{ps}_n{N}": {"min": round(min(v), 4), "max": round(max(v), 4),
                                            "n_refs": len(v)}
                           for (ps, N), v in sorted(by_ps.items())},
        "per_ref_seed": {f"rs{rs}_n{N}": {"min": round(min(v), 4), "max": round(max(v), 4),
                                          "n_pilots": len(v)}
                         for (rs, N), v in sorted(by_rs.items())},
        "warning": "AUCUN résumé poolé — les paires partagent refs/pools (pseudo-réplication): dispersion inter-réfs et inter-seeds rapportée SÉPARÉMENT, jamais agrégée",
    }
    raw_guards = sorted({r["tv_raw_guard"] for r in pairs})

    art = seal_manifest({
        "schema": "ucm-siw-rstar-pilot/0.7.2",
        "what": "pilote DEV v2 — clés TV EXPLICITES + preuves par seed + ESS/records",
        "seed_proofs": proofs,
        "tv_key_definitions": {
            "tv_abstract_declared": "cellule (prédicat, σ≤2, κ=label, d*≤4) — ABSTRACTION cross-layout DÉCLARÉE",
            "tv_raw_guard": "cellule (prédicat, layout_hash, state_key) — garde-fou: ~1.0 par construction sur pools disjoints (montre que seule l'abstraite est meaningful)",
        },
        "pairs": pairs, "summary_tv_abstract": summary,
        "tv_raw_guard_values": raw_guards,
        "ess_definition": "importance w=p_ref(cell_abstraite)/q_prefix; ESS/records publié; coverage FAIL prioritaire (comptes+masse réf des cellules violées)",
        "thresholds": "AUCUN — exploratoire",
        "limitations": {
            "refs_noise": "réfs = 40 ép./10 layouts chacune (MC bruité) — dispersion NON utilisable comme power sans pools DEV plus grands + réfs indépendantes (lead 11:18)",
            "kappa_definition": "κ = ensemble trié des types de TOUTES les optimales (set-valued) — l'exécuté est un tirage DANS ce set, jamais 'le type de l'exécuté' pris au premier optimal",
            "sigma_definition": "σ = vecteur TYPÉ (filled≤2, chosen≤1, dialog, submitted≤1) — la sommation coarse antérieure pouvait cacher un défaut B3, abandonnée pour le prospectif",
        },
        "hold": "DEV, zéro scellé, zéro seed 03/04",
    })
    Path(args.out).write_text(json.dumps(art, sort_keys=True, indent=2))
    print(f"→ {args.out} (scellé {art['manifest_sha256'][:12]})")
    print("seed proofs (streams TOUS distincts):", [p["stream_sha256"][:8] for p in proofs])
    print("tv_raw_guard (attendu ~1.0):", raw_guards[:3])
    print("summary TV abstraite:", summary)


if __name__ == "__main__":
    main()
