"""Entrées puissance report-only (lead 13:43, audit 14:34) — script 5-seeds committé.

Usage: .venv/bin/python scripts/dev_power_inputs.py [--out artifacts/dev-power-inputs-v02.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ucm.data.siw_pipeline import generate_adaptation_episodes_rstar, run_s0_baseline_episode
from ucm.env.siw import SIWState, generate_siw_layout
from ucm.data.writer import seal_manifest

PILOT_SEEDS = (13, 20261020, 20261021, 20261022, 20261023)
N_EPISODES = 60
N_LAYOUTS = 20
SEED_LAYOUTS = 20261010


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/dev-power-inputs-v02.json")
    ap.add_argument("--producer-commit", default=None)
    args = ap.parse_args()

    producer = args.producer_commit or subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()

    rng0 = random.Random(SEED_LAYOUTS)
    layouts = [generate_siw_layout(rng0, rng0.randint(2, 5), with_dialog=rng0.random() < 0.5,
                                   profile="small") for _ in range(N_LAYOUTS)]
    by_hash = {l.layout_hash(): l for l in layouts}

    baseline_by_seed = {}
    for ps in PILOT_SEEDS:
        couples, _ = generate_adaptation_episodes_rstar(layouts, seed=ps, n_episodes=N_EPISODES)
        by_ep = defaultdict(list)
        for c in couples:
            by_ep[c["episode_id"]].append(c)
        rh, rr = [], []
        for eid, recs in sorted(by_ep.items()):
            dep = recs[0]
            lay = by_hash[dep["layout_hash"]]
            goal = dep["goal"]
            st = SIWState(dep["state_key"][0], frozenset(dep["state_key"][1]),
                          dict(dep["state_key"][2]), dep["state_key"][3],
                          frozenset(dep["state_key"][4]))
            seed_ep = int(hashlib.sha256(eid.encode()).hexdigest(), 16) % 10000
            rh.append(run_s0_baseline_episode(lay, st, goal, level="goal_heuristic", seed=seed_ep)["success"])
            rr.append(run_s0_baseline_episode(lay, st, goal, level="random_valid", seed=seed_ep)["success"])
        baseline_by_seed[str(ps)] = {
            "heuristic_success": sum(rh), "random_success": sum(rr),
            "n_episodes": len(rh),
            "delta_pct": round(100 * (sum(rh) - sum(rr)) / max(1, len(rh)), 1),
        }

    deltas = [v["delta_pct"] for v in baseline_by_seed.values()]
    n = len(deltas)
    mean_d = sum(deltas) / n
    var_d = sum((d - mean_d) ** 2 for d in deltas) / (n - 1)
    sd_d = math.sqrt(var_d)
    se_d = sd_d / math.sqrt(n)

    # MDE correctes (audit 14:34 point 4):
    # alpha=0.05 sans puissance: MDE = 1.96 × SE ≈ 5.5pp
    # à ~80% puissance (test bilatéral): MDE ≈ (1.96 + 0.84) × SE = 2.8 × SE ≈ 7.8pp
    mde_alpha_only = 1.96 * se_d
    mde_80pct = 2.8 * se_d

    art = seal_manifest({
        "schema": "ucm-siw-power-inputs/0.2",
        "what": "entrées puissance report-only v02 — script committé 5-seeds, contrast_scope explicite",
        "producer": {"code_commit": producer, "script": "scripts/dev_power_inputs.py",
                     "n_seeds": n, "pilot_seeds": list(PILOT_SEEDS)},
        "contrast_scope": {
            "WARNING": "deltas = heuristic-vs-random S0 PROXY — PAS le contraste pretrained-vs-scratch du protocole V1-bis",
            "M_V1b_sd_reference": "sd inter-seeds M-V1b ≈ 10.9pp (source: lead 14:34, rapport exploratoire)",
            "transposition": "INTERDITE — le sd du proxy (6.3pp) ≠ sd du contraste réel; le MDE calculé ici ne s'applique PAS au protocole",
        },
        "s0_by_seed": baseline_by_seed,
        "variability": {
            "delta_mean_pp": round(mean_d, 1), "delta_sd_pp": round(sd_d, 1),
            "se_delta_pp": round(se_d, 1), "effect_over_se": round(mean_d / se_d, 2),
            "n_seeds": n,
        },
        "mde": {
            "alpha_005_no_power": {"formula": "1.96 × SE", "value_pp": round(mde_alpha_only, 1)},
            "power_80pct": {"formula": "2.8 × SE (=1.96+0.84)", "value_pp": round(mde_80pct, 1)},
            "illustrative": "v01 citait 5.6pp sans préciser sans-puissance — corrigé: 5.5pp (α seul) vs 7.8pp (80% puissance)",
        },
        "n_seeds_guidance": {
            "current": n,
            "target": "≥10 seeds (à confirmer par calcul formel — le protocole décidera)",
            "to_halve_mde": "≈ 4× n_seeds (MDE ∝ 1/√n)",
        },
        "budgets_episodes_note": "voir artifacts/dev-rstar-v0710-dimensioning.json pour budgets épisodes/records",
        "thresholds": "AUCUN",
        "hold": "report-only, DEV, zéro test2",
    })
    Path(args.out).write_text(json.dumps(art, sort_keys=True, indent=2))
    print(f"→ {args.out} (scellé {art['manifest_sha256'][:12]})")
    print(f"delta mean={mean_d:.1f} sd={sd_d:.1f} SE={se_d:.1f} | MDE α-only={mde_alpha_only:.1f}pp | MDE 80%={mde_80pct:.1f}pp")


if __name__ == "__main__":
    main()
