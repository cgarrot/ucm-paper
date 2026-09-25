"""Génération d'épisodes SIW (WS-A V1) — toutes les leçons V0 intégrées:

- quota d*=0 au NIVEAU POOL (leçon M2: jamais par-layout), train seulement
- cycle de prédicats équilibré (leçon M3: le filtrage de bande déséquilibre)
- rejets comptés par motif, jamais silencieux (spec §4.5)
- episode_ref/step en provenance, ordre canonique au stockage
- déterminisme par seed, octets stables

Usage:
    python -m ucm.data.generate_siw --layouts 300 --episodes 1000 \
        --seed 42 --out artifacts/data-siw/pilot.jsonl --pool train
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass, field

from ucm.data.schema import make_state_goal_hash
from ucm.env.siw import (SIW, SIWLayout, SIWState, generate_siw_layout,
                         sample_task)
from ucm.env.siw_oracle import SIWOracle

GENERATOR_VERSION = "siw-0.1.0"
ORACLE_VERSION = "siw-0.1.0"
D_STAR_ZERO_FRACTION = 0.10


@dataclass
class SIWStats:
    layouts: int = 0
    rejected_unreachable: int = 0
    rejected_band: int = 0
    rejected_zero_cap: int = 0
    rejected_k_bounds: int = 0
    episodes: int = 0
    transitions: int = 0
    by_goal: dict = field(default_factory=dict)
    by_d_star: dict = field(default_factory=dict)
    predicate_fallbacks: int = 0

    def summary(self) -> dict:
        return {
            "layouts": self.layouts, "episodes": self.episodes,
            "transitions": self.transitions,
            "rejections": {"unreachable": self.rejected_unreachable,
                           "band": self.rejected_band,
                           "zero_cap": self.rejected_zero_cap,
                           "k_bounds": self.rejected_k_bounds},
            "by_goal": dict(self.by_goal),
            "by_d_star_hist": dict(self.by_d_star),
            "predicate_fallbacks": self.predicate_fallbacks,
        }


def _oracle_for(layout: SIWLayout, goal: dict, cache: dict) -> SIWOracle:
    k = (layout.layout_hash(), goal["predicate"],
         tuple(sorted(goal["args"].items())))
    o = cache.get(k)
    if o is None:
        o = SIWOracle(layout, goal)
        cache[k] = o
    return o


def generate_siw_episode(rng: random.Random, layout: SIWLayout,
                          d_star_band: tuple[int, int], stats: SIWStats,
                          allow_zero_now: bool,
                          oracle_cache: dict,
                          tasks_out: list | None = None,
                          forced_pred: str | None = None,
                          episode_ref: str = "") -> list[dict] | None:
    from ucm.env.siw import GOAL_PREDICATES
    attempts = 200
    for i_attempt in range(attempts):
        state, goal = sample_task(rng, layout)
        # relax après 1/3 des tentatives (leçon V0 M3: le filtrage de bande
        # déséquilibre VIEW, qui a plus de d* petits)
        relax = forced_pred is not None and i_attempt >= attempts // 3
        if relax and goal["predicate"] != forced_pred:
            stats.predicate_fallbacks += 1
            forced_eff = None
        else:
            forced_eff = forced_pred
        if forced_eff and goal["predicate"] != forced_eff:
            # re-tirer le but avec le prédicat forcé (cycle §M3)
            from ucm.env.siw import task_goal
            w = None
            if forced_eff == "VIEW":
                goal = task_goal("VIEW", view=rng.choice(layout.views))
            elif forced_eff == "SET":
                flds = [x for x in layout.widgets.values()
                        if x["type"] == "field"]
                w = rng.choice(flds)
                goal = task_goal("SET", field=w["id"])
            elif forced_eff == "CHOOSE":
                sels = [x for x in layout.widgets.values()
                        if x["type"] == "select"]
                w = rng.choice(sels)
                opts = [o for o in layout.widgets.values()
                        if o.get("select") == w["id"]]
                goal = task_goal("CHOOSE", select=w["id"],
                                 option=rng.choice(opts)["id"])
            else:
                forms = [x for x in layout.widgets.values()
                         if x["type"] == "form"]
                w = rng.choice(forms)
                goal = task_goal("SUBMITTED", form=w["id"])
        oracle = _oracle_for(layout, goal, oracle_cache)
        sol = oracle.solve(state)
        if not sol["reachable"]:
            stats.rejected_unreachable += 1
            continue
        d0 = sol["d_star"]
        if d0 == 0:
            if not allow_zero_now:
                stats.rejected_zero_cap += 1
                continue
        elif not (d_star_band[0] <= d0 <= d_star_band[1]):
            stats.rejected_band += 1
            continue

        env = SIW(layout)
        env.goal = goal
        env.state = state.copy()
        env.step_count = 0
        env.terminal = False
        records: list[dict] = []
        layout_hash = layout.layout_hash()
        t = 0
        while not env.terminal:
            obs = env.observe()
            cands = obs["candidates"]
            state_hash = env.state_hash()
            sol_now = oracle.solve(env.state)
            chosen = rng.choice(sol_now["optimal_actions"])
            records.append({
                "schema_version": "0.2",
                "policy_input": {
                    "goal": obs["goal"],
                    "entities": obs["entities"],
                    "relations": obs["relations"],
                    "candidates": cands,
                },
                "execution": {
                    "action_ref": chosen,
                    "observable_result": None,
                    "next_state_hash": None,
                },
                "supervision": {
                    "optimal_actions": sol_now["optimal_actions"],
                    "d_star": sol_now["d_star"],
                    "reachable": True,
                },
                "provenance": {
                    "layout_hash": layout_hash,
                    "state_goal_hash": make_state_goal_hash(
                        layout_hash, state_hash, goal),
                    "split": "unassigned",
                    "source": "oracle",
                    "generator_version": GENERATOR_VERSION,
                    "oracle_version": ORACLE_VERSION,
                    "episode_ref": episode_ref,
                    "step": t,
                },
            })
            res = env.execute(cands[chosen])
            records[-1]["execution"]["observable_result"] = \
                "valid" if res["valid"] else "invalid"
            records[-1]["execution"]["next_state_hash"] = env.state_hash()
            t += 1

        stats.episodes += 1
        stats.transitions += len(records)
        stats.by_goal[goal["predicate"]] = \
            stats.by_goal.get(goal["predicate"], 0) + 1
        stats.by_d_star[str(d0)] = stats.by_d_star.get(str(d0), 0) + 1
        if tasks_out is not None:
            tasks_out.append({"layout_hash": layout_hash,
                              "goal": goal, "d_star": d0})
        return records
    return None


def generate_siw(episodes: int, n_layouts: int, seed: int,
                 views_range: tuple[int, int] = (2, 5),
                 d_star_band: tuple[int, int] = (2, 24),
                 pool: str = "train", *,
                 collect_details: bool = False,
                 episode_prefix: str = ""):
    """Retourne (lignes JSONL, stats[, détails]). Quota zéro = NIVEAU POOL,
    train uniquement (leçons V0 M2: pool-level, jamais par-layout)."""
    rng = random.Random(seed)
    stats = SIWStats()
    layouts = []
    tries = 0
    while len(layouts) < n_layouts and tries < 10 * n_layouts:
        tries += 1
        try:
            layouts.append(generate_siw_layout(
                rng, rng.randint(*views_range)))
        except Exception as e:
            if "hors bornes" in str(e):
                stats.rejected_k_bounds += 1
            else:
                raise
    stats.layouts = len(layouts)
    oracle_cache: dict = {}
    lines: list[str] = []
    tasks: list[dict] = []
    zero_quota = (max(1, round(episodes * D_STAR_ZERO_FRACTION))
                  if pool == "train" and episodes >= 1 else 0)
    zero_done = 0
    pred_cycle = ["VIEW", "SET", "CHOOSE", "SUBMITTED"]
    for i in range(episodes):
        layout = layouts[rng.randrange(len(layouts))]
        allow_zero = zero_done < zero_quota
        forced = pred_cycle[i % 4]
        ep_ref = f"{episode_prefix}{seed}-{i:05d}" if episode_prefix else ""
        recs = generate_siw_episode(
            rng, layout, d_star_band, stats, allow_zero,
            oracle_cache, tasks_out=tasks, forced_pred=forced,
            episode_ref=ep_ref)
        if recs is None:
            continue
        if len(recs) == 1:
            zero_done += 1
        lines.extend(json.dumps(r, sort_keys=True) for r in recs)
    if collect_details:
        details = {
            "layouts": [{"layout_hash": l.layout_hash(),
                         "views": l.views,
                         "nav_edges": [list(e) for e in l.nav_edges],
                         "widgets": list(l.widgets.values()),
                         "labels": l.labels} for l in layouts],
            "tasks": tasks,
        }
        return lines, stats.summary(), details
    return lines, stats.summary()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layouts", type=int, default=300)
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--views-min", type=int, default=2)
    ap.add_argument("--views-max", type=int, default=5)
    ap.add_argument("--d-star-min", type=int, default=2)
    ap.add_argument("--d-star-max", type=int, default=24)
    ap.add_argument("--pool", choices=["train", "val", "test"], default="train")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--layouts-out", type=str, default=None)
    ap.add_argument("--tasks-out", type=str, default=None)
    ap.add_argument("--episode-prefix", type=str, default="")
    args = ap.parse_args()

    t0 = time.time()
    want = bool(args.layouts_out or args.tasks_out)
    res = generate_siw(
        episodes=args.episodes, n_layouts=args.layouts, seed=args.seed,
        views_range=(args.views_min, args.views_max),
        d_star_band=(args.d_star_min, args.d_star_max),
        pool=args.pool, collect_details=want,
        episode_prefix=args.episode_prefix)
    lines, stats = res[0], res[1]
    stats["generation_seconds"] = round(time.time() - t0, 2)
    stats["lines"] = len(lines)
    stats["seed"] = args.seed
    stats["pool"] = args.pool
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    with open(args.out.rsplit(".", 1)[0] + ".stats.json", "w") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
    if want:
        details = res[2]
        if args.layouts_out:
            with open(args.layouts_out, "w") as f:
                json.dump({l["layout_hash"]: l for l in details["layouts"]},
                          f, indent=1, sort_keys=True)
        if args.tasks_out:
            with open(args.tasks_out, "w") as f:
                json.dump(details["tasks"], f, indent=1)
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
