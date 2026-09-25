"""Pilote DEV de DIMENSIONNEMENT R* (lead 11:34) — versionné, producteur publié.

Design:
- budgets en ÉPISODES ENTIERS CROISSANTS (8..256), préfixes cumulatifs du stream;
- réservoir DEV PLUS GRAND (40 layouts, replay-only — AUCUN manifeste scellé);
- 5 seeds pilotes × 5 réfs indépendantes (120 épisodes chacune — moins bruitées);
- publié PAR BUDGET: coverage zéros (comptes + masse réf), TV abstraite fine
  (κ-ensemble × σ-vecteur), ESS_importance + ESS/records, records, équilibrage
  prédicat; PAR STRATE (jamais poolé — leçon pseudo-réplication);
- aucun seuil: dimensionnement pour le calcul de puissance seed-cluster (tagi-5).

Usage: .venv/bin/python scripts/dev_dimensioning_rstar.py [--out artifacts/dev-rstar-v075-dimensioning.json]
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

from ucm.data.siw_pipeline import generate_adaptation_episodes_rstar
from ucm.data.writer import seal_manifest
from ucm.env.siw import generate_siw_layout

PILOT_SEEDS = (13, 20261020, 20261021, 20261022, 20261023)
REF_SEEDS = (20261013, 20261014, 20261015, 20261016, 20261017)
BUDGETS = (8, 16, 32, 64, 128, 256)
RESERVOIR_SEED = 20261010
N_LAYOUTS = 40
REF_EPISODES = 120


def dev_pool(seed, n):
    rng = random.Random(seed)
    return [generate_siw_layout(rng, rng.randint(2, 5), with_dialog=rng.random() < 0.5,
                                profile="small") for _ in range(n)]


def label_set(c):
    return tuple(sorted({sem.split(":")[0] for sem in c["optimal_semantic"]}))


def sigma_vec(c):
    sk = c["state_key"]
    return (min(len(sk[1]), 2), min(len(sk[2]), 1), bool(sk[3]), min(len(sk[4]), 1))


def cell(c):
    return (c["goal"]["predicate"],) + sigma_vec(c) + (label_set(c), min(c["d_star"], 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/dev-rstar-v075-dimensioning.json")
    ap.add_argument("--producer-commit", default=None,
                    help="SHA du commit producteur (requis en archive pinnée sans .git; "
                         "défaut: git rev-parse HEAD du repo courant)")
    args = ap.parse_args()

    if args.producer_commit:
        producer = args.producer_commit
    else:
        try:
            producer = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                      text=True, check=True).stdout.strip()
        except Exception:
            raise SystemExit("producer indéterminable: archive sans .git → passer --producer-commit")

    reservoir = dev_pool(RESERVOIR_SEED, N_LAYOUTS)

    # réfs indépendantes (pool propre par seed de réf — indépendance maximale)
    refs = {}
    for rs in REF_SEEDS:
        pool = dev_pool(rs + 900, N_LAYOUTS)
        couples, _ = generate_adaptation_episodes_rstar(pool, seed=rs, n_episodes=REF_EPISODES)
        cnt = Counter(cell(c) for c in couples)
        refs[rs] = {"n": len(couples), "dist": {k: v / len(couples) for k, v in cnt.items()}}

    # streams pilotes (réservoir partagé, seeds distincts)
    # B1 (lead 12:05, raffiné 12:37): helper PARTAGÉ ucm/data/rstar_order.py —
    # UNE définition de l'ordre pour métriques ET matérialisation train DEV.
    # stream_hashes = digest FULL du flux PERMUTÉ (fingerprint l'ordre effectif).
    from ucm.data.rstar_order import round_robin_episodes, serialize_rr_records
    import hashlib as _hl

    pilot_streams = {}
    for ps in PILOT_SEEDS:
        couples, stats = generate_adaptation_episodes_rstar(reservoir, seed=ps, n_episodes=BUDGETS[-1])
        by_ep = defaultdict(list)
        for c in couples:
            by_ep[c["episode_id"]].append(c)
        ep_order = round_robin_episodes(by_ep, sorted(by_ep))
        permuted = [c for eid in ep_order for c in by_ep[eid]]
        sha_perm = _hl.sha256(serialize_rr_records(permuted)).hexdigest()
        pilot_streams[ps] = {"ep_ids": ep_order, "by_ep": by_ep,
                             "sha_permuted": sha_perm,
                             "prefix_digests": {}}
        # digest FULL de chaque préfixe permuté (octets JSONL exacts)
        for N in BUDGETS:
            pf = [c for eid in ep_order[:N] for c in by_ep[eid]]
            pilot_streams[ps]["prefix_digests"][str(N)] = _hl.sha256(
                serialize_rr_records(pf)).hexdigest()

    rows = []
    for ps in PILOT_SEEDS:
        st = pilot_streams[ps]
        for N in BUDGETS:
            prefix = [c for eid in st["ep_ids"][:N] for c in st["by_ep"][eid]]
            qc = Counter(cell(c) for c in prefix)
            n = len(prefix)
            q = {k: v / n for k, v in qc.items()}
            pred_eps = Counter()
            for eid in st["ep_ids"][:N]:
                pred_eps[st["by_ep"][eid][0]["goal"]["predicate"]] += 1
            # B1 assert: N divisible par 4 ⇒ N/4 épisodes par prédicat
            assert N % 4 == 0 and all(v == N // 4 for v in pred_eps.values()), \
                f"round-robin rompu à N={N}: {dict(pred_eps)}"
            for rs, ref in refs.items():
                keys = set(q) | set(ref["dist"])
                tv = 0.5 * sum(abs(q.get(k, 0) - ref["dist"].get(k, 0)) for k in keys)
                zero_v = [(k, ref["dist"][k]) for k in ref["dist"] if ref["dist"][k] > 0 and qc.get(k, 0) == 0]
                # B2 (lead 12:05:26): UN POIDS PAR RECORD — forme agrégée:
                # Σ_k count_q[k]·w_k, Σ_k count_q[k]·w_k², n=records
                # (w=0 si p_ref=0; indépendant du coverage flag)
                sw = sw2 = 0.0
                for k, cnt in qc.items():
                    p_ref_k = ref["dist"].get(k, 0.0)
                    q_k = cnt / n
                    if q_k > 0:
                        w_k = p_ref_k / q_k
                        sw += cnt * w_k
                        sw2 += cnt * w_k * w_k
                ess = (sw * sw / sw2) if sw2 > 0 else None
                rows.append({
                    "pilot_seed": ps, "budget_episodes": N, "ref_seed": rs,
                    "records": n,
                    "tv_abstract": round(tv, 4),
                    "ess_importance": round(ess, 1) if ess else None,
                    "ess_per_record": round(ess / n, 4) if ess else None,
                    "coverage_zero_cells": len(zero_v),
                    "coverage_zero_ref_mass": round(sum(m for _, m in zero_v), 4),
                    "coverage_fail": bool(zero_v),
                    "pred_episode_counts": dict(pred_eps),
                })

    # strates: par budget (dispersion inter-seeds ET inter-réfs rapportée séparément)
    by_budget = defaultdict(list)
    for r in rows:
        by_budget[r["budget_episodes"]].append(r)
    strata = {}
    for N, rs_ in sorted(by_budget.items()):
        tvs = [r["tv_abstract"] for r in rs_]
        essr = [r["ess_per_record"] for r in rs_ if r["ess_per_record"]]
        strata[str(N)] = {
            "tv_min": round(min(tvs), 4), "tv_max": round(max(tvs), 4),
            "ess_per_record_min": round(min(essr), 4) if essr else None,
            "ess_per_record_max": round(max(essr), 4) if essr else None,
            "coverage_fail_count": sum(1 for r in rs_ if r["coverage_fail"]),
            "pairs": len(rs_),
            "warning": "25 paires par budget = 5 seeds × 5 réfs CORRÉLÉS — strates, jamais poolé",
        }

    art = seal_manifest({
        "schema": "ucm-siw-rstar-dimensioning/0.7",
        "what": "pilote DEV de DIMENSIONNEMENT (lead 11:34) — aucun seuil",
        "producer": {"code_commit": producer, "generator_schema": "ucm-siw-adaptation-couples/0.8"},
        "design": {"reservoir": {"seed": RESERVOIR_SEED, "layouts": N_LAYOUTS, "replay_only": True},
                   "pilot_seeds": list(PILOT_SEEDS), "ref_seeds": list(REF_SEEDS),
                   "ref_episodes": REF_EPISODES, "budgets": list(BUDGETS),
                   "cell": "prédicat × σ-vecteur(filled≤2,chosen≤1,dialog,submitted≤1) × κ-ensemble × d*≤4"},
        "stream_sha256_permuted_full": {str(ps): pilot_streams[ps]["sha_permuted"] for ps in PILOT_SEEDS},
        "prefix_sha256_permuted_full": {str(ps): {N: h for N, h in pilot_streams[ps]["prefix_digests"].items()} for ps in PILOT_SEEDS},
        "hash_format": "SHA-256 complet 64 hex — octets JSONL exactement ceux matérialisés pour train",
        "shared_order_helper": "ucm/data/rstar_order.py (round_robin_episodes + permuted_stream — utilisé par métriques ET matérialisation train)",
        "tv_all_distinct_25_per_budget": "constat report-only: les 25 TV d'un budget sont toutes distinctes (non publié dans rows — vérifiable)",
        "rows": rows, "strata_by_budget": strata,
        "power_note": "dimensionnement pour calcul de puissance seed-cluster (tagi-5) — k* JAMAIS sélectionné via test historique/futur",
        "hold": "DEV, zéro scellé, zéro seed 20261003/04",
    })
    Path(args.out).write_text(json.dumps(art, sort_keys=True, indent=2))
    print(f"→ {args.out} (scellé {art['manifest_sha256'][:12]})")
    for N, s in strata.items():
        print(f"N={N}: TV {s['tv_min']}-{s['tv_max']} | ESS/rec {s['ess_per_record_min']}-{s['ess_per_record_max']} | fail {s['coverage_fail_count']}/{s['pairs']}")


if __name__ == "__main__":
    main()
