"""S0 baselines batch-runner — script committé reproductible (tagi-5 13:47).

Usage: .venv/bin/python scripts/dev_s0_baselines.py [--out artifacts/dev-s0-baselines-v02.json]
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ucm.data.siw_pipeline import generate_adaptation_episodes_rstar, run_s0_baseline_episode
from ucm.env.siw import SIWState
from ucm.data.writer import seal_manifest

LEVELS = ("oracle", "goal_heuristic", "random_valid")
N_EPISODES = 60
N_LAYOUTS = 20
SEED_LAYOUTS = 20261010
SEED_EPISODES = 13


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/dev-s0-baselines-v03.json")
    ap.add_argument("--producer-commit", default=None,
                    help="SHA du commit producteur (requis en archive sans .git)")
    args = ap.parse_args()

    if args.producer_commit:
        producer = args.producer_commit
    else:
        try:
            producer = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                      text=True, check=True).stdout.strip()
        except Exception:
            raise SystemExit("--producer-commit requis en archive sans .git")

    from ucm.env.siw import generate_siw_layout
    rng = random.Random(SEED_LAYOUTS)
    layouts = [generate_siw_layout(rng, rng.randint(2, 5), with_dialog=rng.random() < 0.5,
                                   profile="small") for _ in range(N_LAYOUTS)]
    by_hash = {l.layout_hash(): l for l in layouts}

    couples, _ = generate_adaptation_episodes_rstar(layouts, seed=SEED_EPISODES, n_episodes=N_EPISODES)
    by_ep = defaultdict(list)
    for c in couples:
        by_ep[c["episode_id"]].append(c)

    results = []
    for eid, recs in sorted(by_ep.items()):
        dep = recs[0]
        lay = by_hash[dep["layout_hash"]]
        goal = dep["goal"]
        st = SIWState(dep["state_key"][0], frozenset(dep["state_key"][1]),
                      dict(dep["state_key"][2]), dep["state_key"][3],
                      frozenset(dep["state_key"][4]))
        for level in LEVELS:
            import hashlib as _hl
            seed_ep = int(_hl.sha256(eid.encode()).hexdigest(), 16) % 10000
            r = run_s0_baseline_episode(lay, st, goal, level=level,
                                       seed=seed_ep)
            r["episode_id"] = eid
            r["predicate"] = goal["predicate"]
            r["d_star_depart"] = dep["d_star"]
            results.append(r)

    summary = {}
    for level in LEVELS:
        rs = [r for r in results if r["level"] == level]
        n = len(rs)
        summary[level] = {
            "n_episodes": n,
            "success": sum(r["success"] for r in rs),
            "goal_ever": sum(r["goal_ever"] for r in rs),
            "stopped": sum(r["stopped"] for r in rs),
            "timeout": sum(r["timeout"] for r in rs),
            "avg_steps": round(sum(r["steps"] for r in rs) / n, 1),
            "avg_latency_ms": round(sum(r["latency_ms"] for r in rs) / n, 2),
            "label_information": {
                "oracle": "BORNE SUP PRIVILÉGIÉE — vraies transitions + d* exact",
                "goal_heuristic": "INFO RÉELLEMENT DISPONIBLE — observation seule, heuristique par prédicat",
                "random_valid": "INFO RÉELLEMENT DISPONIBLE — transitions connues, tirage uniforme des valides",
            }[level],
        }

    art = seal_manifest({
        "schema": "ucm-siw-s0-baselines/0.2",
        "what": "S0 baselines CPU DEV v02 — script committé, raw COMPLET (tagi-5 13:47)",
        "producer": {"code_commit": producer, "script": "scripts/dev_s0_baselines.py"},
        "fixtures": {"generator": "R* rev0.8", "seed_episodes": SEED_EPISODES,
                     "seed_layouts": SEED_LAYOUTS, "n_episodes": N_EPISODES,
                     "n_layouts": N_LAYOUTS, "profile": "small", "departures_d*": "[2,8]"},
        "levels": summary,
        "budget": {"calls_per_episode": "≤64 (horizon)", "interactions": "1 env par épisode",
                   "cpu_only": True},
        "raw_sample_rule": "AUCUN échantillonnage — raw COMPLET 60×3=180 records publiés intégralité",
        "raw": results,
        "thresholds": "AUCUN — exploratoire",
        "hold": "DEV-only, zéro test2/scellé",
    })
    Path(args.out).write_text(json.dumps(art, sort_keys=True, indent=2))
    print(f"→ {args.out} (scellé {art['manifest_sha256'][:12]})")
    for lv, s in summary.items():
        print(f"  {lv}: {s['success']}/{s['n_episodes']} steps {s['avg_steps']} lat {s['avg_latency_ms']}ms")


if __name__ == "__main__":
    main()
