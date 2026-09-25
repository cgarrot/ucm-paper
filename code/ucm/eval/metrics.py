"""Evaluation metrics (spec §9.2) and hierarchical paired bootstrap (spec §9.3).

Primary: full-episode success with STOP, verified independently (rollout.py).
Secondary: physically-reached goal, premature STOP, timeouts, invalid rate,
length, truncated cost score (L on success, H+1 on failure — explicit formula,
never presented as exact regret), regret L-L* on successes WITH the associated
failure rate, optimal-action rate stratified by |A*|.

Bootstrap: paired, hierarchical (seeds → layouts → tasks), 95 % — the 500
episodes of a cell are NOT 500 independent layouts; per-seed and effective
cluster counts are published alongside (spec §9.3).
"""

from __future__ import annotations

import random
from collections import defaultdict

from ucm.eval.rollout import EpisodeResult, HORIZON


def summarize(episodes: list[EpisodeResult]) -> dict:
    n = len(episodes)
    if n == 0:
        return {"n": 0}
    succ = [e for e in episodes if e.success]
    regrets = [(e.length - e.L_star) for e in succ if e.L_star is not None]
    total_decisions = sum(e.length for e in episodes)
    return {
        "n": n,
        "success_rate": len(succ) / n,
        "premature_stop_rate": sum(e.outcome == "premature_stop" for e in episodes) / n,
        "timeout_rate": sum(e.outcome == "timeout" for e in episodes) / n,
        "goal_reached_without_stop_rate": sum(e.goal_reached_without_stop for e in episodes) / n,
        "invalid_rate": sum(e.n_invalid for e in episodes) / max(total_decisions, 1),
        "mean_length_success": (sum(e.length for e in succ) / len(succ)) if succ else None,
        "mean_regret_success": (sum(regrets) / len(regrets)) if regrets else None,
        "mean_truncated_cost": sum(e.length if e.success else HORIZON + 1 for e in episodes) / n,
        "mean_L_star": (sum(e.L_star for e in episodes if e.L_star is not None)
                        / max(1, sum(1 for e in episodes if e.L_star is not None))),
    }


def success(e: EpisodeResult) -> float:
    return 1.0 if e.success else 0.0


def truncated_cost(e: EpisodeResult) -> float:
    return float(e.length if e.success else HORIZON + 1)


# ---------------------------------------------------------------------------
# Hierarchical paired bootstrap (spec §9.3)
# ---------------------------------------------------------------------------

def _clusters(episodes: list[EpisodeResult]) -> dict:
    """seed → layout → [episodes] nested clustering. EpisodeResult must carry
    seed and layout_id (rollout fills them; tests may inject)."""
    tree: dict = defaultdict(lambda: defaultdict(list))
    for e in episodes:
        tree[e.seed][e.layout_id].append(e)
    return tree


def _resample(tree: dict, rng: random.Random) -> list[EpisodeResult]:
    """Resample seeds → layouts → tasks, each WITH replacement at every level
    (spec §9.3: graines, layouts puis tâches — the task level is resampled too;
    audit tagi-5 M2). Sample size = #episodes, not #layouts."""
    seeds = list(tree.keys())
    out = []
    for _ in range(len(seeds)):
        s = rng.choice(seeds)
        layouts = list(tree[s].keys())
        for _ in range(len(layouts)):
            lay = rng.choice(layouts)
            eps = tree[s][lay]
            for _ in range(len(eps)):          # tasks resampled with replacement
                out.append(rng.choice(eps))
    return out


def hierarchical_bootstrap(episodes: list[EpisodeResult], metric=success,
                           n_boot: int = 10_000, ci: float = 0.95,
                           seed: int = 0) -> dict:
    """CI for a single arm's metric with seeds→layouts→tasks clustering."""
    rng = random.Random(seed)
    tree = _clusters(episodes)
    stats = []
    for _ in range(n_boot):
        sample = _resample(tree, rng)
        stats.append(sum(metric(e) for e in sample) / len(sample))
    stats.sort()
    lo = stats[int((1 - ci) / 2 * n_boot)]
    hi = stats[min(n_boot - 1, int((1 + ci) / 2 * n_boot))]
    point = sum(metric(e) for e in episodes) / len(episodes)
    n_clusters = sum(len(lays) for lays in tree.values())
    return {"point": point, "ci_low": lo, "ci_high": hi, "level": ci,
            "n_seeds": len(tree), "n_seed_layout_clusters": n_clusters,
            "n_episodes": len(episodes)}


def paired_hierarchical_bootstrap(episodes_a: list[EpisodeResult],
                                  episodes_b: list[EpisodeResult],
                                  metric=success, n_boot: int = 10_000,
                                  ci: float = 0.95, seed: int = 0) -> dict:
    """PAIRED bootstrap for arm A − arm B over the same episode set: matches
    episodes by (seed, layout_id, episode_id); resamples clusters once and
    applies the same resample indices to both arms (spec §9.3 appariement)."""
    key = lambda e: (e.seed, e.layout_id, e.episode_id)
    b_by = {key(e): e for e in episodes_b}
    pairs = [(e, b_by[key(e)]) for e in episodes_a if key(e) in b_by]
    if len(pairs) != len(episodes_a) or len(pairs) != len(episodes_b):
        raise ValueError("paired bootstrap requires identical episode sets")
    rng = random.Random(seed)
    # cluster over paired episodes using arm A structure (identical by pairing)
    tree = _clusters(episodes_a)
    stats = []
    for _ in range(n_boot):
        sample_a = _resample(tree, rng)
        da = sum(metric(a) for a in sample_a) / len(sample_a)
        # same resampled keys applied to arm B
        db_keys = [(e.seed, e.layout_id, e.episode_id) for e in sample_a]
        db = sum(metric(b_by[k]) for k in db_keys) / len(db_keys)
        stats.append(da - db)
    stats.sort()
    lo = stats[int((1 - ci) / 2 * n_boot)]
    hi = stats[min(n_boot - 1, int((1 + ci) / 2 * n_boot))]
    point = (sum(metric(a) for a, _ in pairs) - sum(metric(b) for _, b in pairs)) / len(pairs)
    return {"point_diff": point, "ci_low": lo, "ci_high": hi, "level": ci,
            "ci_excludes_zero": (lo > 0 or hi < 0), "n_pairs": len(pairs),
            "n_seeds": len(tree),
            "n_seed_layout_clusters": sum(len(l) for l in tree.values())}


def optimal_action_rate_by_astar(logits_labels: list[tuple]) -> dict:
    """Stratification of the diagnostic by |A*| (spec §9.2).
    logits_labels: list of (logits: list[float], labels: list[int]) pairs."""
    buckets: dict[int, list[bool]] = defaultdict(list)
    for logits, labels in logits_labels:
        k = int(sum(labels))
        hit = max(range(len(logits)), key=lambda i: logits[i])  # argmax over scores
        buckets[k].append(labels[hit] == 1)
    return {f"|A*|={k}": {"rate": sum(v) / len(v), "n": len(v)} for k, v in sorted(buckets.items())}
