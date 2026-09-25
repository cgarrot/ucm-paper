#!/usr/bin/env python3
"""Mesure DEV bornée du chemin EVAL officiel (mission OPS condition (b)) — report-only.

Chemin mesuré : `_build_env` + `SIWModelPolicy` + `run_episode` (§9.4), tel qu'utilisé par
l'évaluation de l'étape 4 (600 épisodes × 120 cellules = 72 000 rollouts). Fixture : épisodes
DEV **générés à la volée** (layouts du store DEV) — **jamais test2, aucun scellé**. Modèle :
fraîche init `make_siw_model()` (le temps d'eval est indépendant de l'état d'entraînement —
noté). Estimation : s/épisode (moyenne/médiane/p95) → extrapolation 72k.

AUCUNE optimisation de code ; aucun seuil figé. Usage :
  python scripts/profile_eval_official.py --report reports/eval-official-dev-profile.json
"""
from __future__ import annotations

import argparse
import json
import resource
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_REPO = Path(__file__).resolve().parents[1]
_DEV_STORE = _REPO / "artifacts" / "inventory-siw-dev-layouts.json"


def _build_env(episode: dict):
    """Réplique exacte de ucm/v1/runner.py::_build_env (même chemin que l'eval officiel)."""
    from ucm.env.siw import SIW, SIWLayout
    spec = dict(episode["layout_spec"])
    if isinstance(spec.get("widgets"), dict):
        spec["widgets"] = list(spec["widgets"].values())
    env = SIW(SIWLayout(spec))
    env.reset(episode["task"])
    return env


def _generate_dev_episodes(n_per_predicate: int, seed: int, band=(2, 8)):
    """Épisodes DEV à la volée (build_siw_test_episodes sur layouts DEV) → format runner."""
    from ucm.data.siw_pipeline import build_siw_test_episodes
    from ucm.env.siw import SIWLayout
    from ucm.v1.data_adapter import _spec_of, episodes_lines_to_runner
    store = json.load(open(_DEV_STORE))
    layouts = [SIWLayout(_spec_of(e)) for e in store.values()]
    _manifest, episodes = build_siw_test_episodes(
        layouts, seed=seed, n_per_predicate=n_per_predicate, band=band)
    lines = [json.dumps(e, sort_keys=True, default=str) for e in episodes]
    return episodes_lines_to_runner(lines, str(_DEV_STORE))


def main() -> int:
    ap_ = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--n-per-predicate", type=int, default=15,
                     help="épisodes par prédicat (15 → 60 épisodes DEV bornés)")
    ap_.add_argument("--seed", type=int, default=20260927)
    ap_.add_argument("--horizon", type=int, default=64)
    ap_.add_argument("--report", default=None)
    args = ap_.parse_args()

    from ucm.model.siw_model import make_siw_model
    from ucm.v1.policy_siw import SIWModelPolicy
    from ucm.eval.rollout import run_episode
    import mlx.core as mx
    mx.set_default_device(mx.cpu)

    t_fix0 = time.perf_counter()
    eps = _generate_dev_episodes(args.n_per_predicate, args.seed)
    fix_s = time.perf_counter() - t_fix0

    model = make_siw_model()
    mx.eval(model.parameters())  # modèle prêt (fraîche init ; le timing eval n'en dépend pas)

    per_ep: list[dict] = []
    for i, ep in enumerate(eps):
        t_env0 = time.perf_counter()
        env = _build_env(ep)
        env_s = time.perf_counter() - t_env0

        pol = SIWModelPolicy(model, name="eval-profile", seed=0)
        pol_times: list[float] = []

        def timed_policy(obs, _pol=pol, _acc=pol_times):
            t0 = time.perf_counter()
            act = _pol(obs)
            _acc.append(time.perf_counter() - t0)
            return act

        mx.eval()  # drain (aucun graphe MLX en attente avant le chrono)
        t0 = time.perf_counter()
        r = run_episode(env, timed_policy, ep["episode_id"], seed=0, policy_name="eval-profile",
                        d_star=ep.get("d_star"), horizon=args.horizon)
        total_s = time.perf_counter() - t0
        per_ep.append({
            "episode_id": ep["episode_id"], "layout_hash": ep.get("layout_hash", ""),
            "d_star": ep.get("d_star"), "length": r.length, "outcome": r.outcome,
            "success": r.success, "n_invalid": r.n_invalid,
            "env_build_s": round(env_s, 6),
            "policy_total_s": round(sum(pol_times), 6),
            "policy_calls": len(pol_times),
            "episode_total_s": round(total_s, 6),
            "per_step_ms": round(1000 * total_s / max(r.length, 1), 3),
        })
        if i == 0:
            mx.eval()  # warm-up implicite sur le 1er épisode (noté dans le rapport)

    totals = [e["episode_total_s"] for e in per_ep]
    policies = [e["policy_total_s"] for e in per_ep]
    envs = [e["env_build_s"] for e in per_ep]
    lengths = [e["length"] for e in per_ep]

    def _stats(xs):
        s = sorted(xs)
        n = len(s)
        return {"mean": round(statistics.fmean(s), 6), "median": round(s[n // 2], 6),
                "p95": round(s[min(n - 1, int(0.95 * n))], 6), "min": round(s[0], 6),
                "max": round(s[-1], 6)}

    mean_s = statistics.fmean(totals)
    p95_s = sorted(totals)[min(len(totals) - 1, int(0.95 * len(totals)))]
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = int(rss) if sys.platform == "darwin" else int(rss) * 1024

    report = {
        "kind": "eval-official-dev-profile",
        "path_mesure": "_build_env + SIWModelPolicy + run_episode (§9.4) — chemin de l'eval étape 4 "
                       "(runner_official Stage-B, actuellement LOCKED ; aucune donnée test2 utilisée)",
        "fixture": {"source": "build_siw_test_episodes sur layouts DEV (store 30 layouts)",
                    "seed": args.seed, "n_per_predicate": args.n_per_predicate,
                    "n_episodes": len(eps), "band": [2, 8], "generation_wall_s": round(fix_s, 3),
                    "note": "épisodes DEV générés à la volée — AUCUN test2/scellé ouvert"},
        "protocol": {"horizon": args.horizon, "model": "make_siw_model() fraîche init (timing "
                     "indépendant de l'état d'entraînement)", "device": "cpu"},
        "per_episode": per_ep,
        "stats": {"episode_total_s": _stats(totals), "policy_total_s": _stats(policies),
                  "env_build_s": _stats(envs), "length": _stats(lengths)},
        "extrapolation": {
            "n_episodes_etape4": 72_000,
            "heures_moyenne": round(mean_s * 72_000 / 3600, 2),
            "heures_p95": round(p95_s * 72_000 / 3600, 2),
            "note": "600 épisodes × 120 cellules ; extrapolation linéaire depuis la mesure DEV "
                    "(report-only, aucun seuil figé)",
        },
        "machine": {"rss_mb": round(rss_bytes / 1e6, 1), "note": "machine partagée (charge non contrôlée)"},
        "limits": [
            "modèle fraîche init (pas de modèle fine-tuné) — le timing eval n'en dépend pas notablement",
            "1 seed d'éval, 60 épisodes DEV (borné) ; machine partagée",
            "premier épisode = warm-up implicite mesuré (pas exclu) ; médiane+p95 fournies",
            "aucun scellé/test2 ; aucun seuil figé ; aucune optimisation de code",
        ],
    }
    for e in per_ep[:3]:
        print(f"[ep ] {e['episode_id']} len={e['length']} total={e['episode_total_s']}s "
              f"policy={e['policy_total_s']}s per-step={e['per_step_ms']}ms out={e['outcome']}")
    print(f"[stats] total mean={report['stats']['episode_total_s']['mean']}s "
          f"median={report['stats']['episode_total_s']['median']}s p95={p95_s}s | "
          f"policy mean={report['stats']['policy_total_s']['mean']}s | "
          f"len mean={report['stats']['length']['mean']}")
    print(f"[extrap] 72k épisodes → {report['extrapolation']['heures_moyenne']} h (moyenne) / "
          f"{report['extrapolation']['heures_p95']} h (p95)")
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"rapport → {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
