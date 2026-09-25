"""Génération de layouts et trajectoires oracle TinyGraphKey (WS-A).

Déterministe par seed. Compte les rejets et leurs motifs (spec §4.5) : aucun
rejet silencieux. Records conformes au schéma WS-B (ucm.data.schema) —
validation stricte appliquée dans tests/test_generate.py.

Usage :
    python -m ucm.data.generate --layouts 350 --episodes 5000 --seed 42 \
        --out artifacts/data/pilot.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass, field

from ucm.data.schema import make_state_goal_hash
from ucm.env.oracle import LayoutOracle
from ucm.env.tinygraph import Layout, PhysicalState, TinyGraphKey

GENERATOR_VERSION = "0.2.1"
ORACLE_VERSION = "0.1.0"
D_STAR_ZERO_FRACTION = 0.10  # spec §4.5 : ~10 % d'épisodes déjà satisfaits


# --------------------------------------------------------------------------- #
# Layouts
# --------------------------------------------------------------------------- #

@dataclass
class GenerationStats:
    layouts: int = 0
    rejected_unreachable: int = 0
    rejected_band: int = 0
    rejected_zero_fraction: int = 0
    rejected_g2_reservation: int = 0
    rejected_non_g2: int = 0
    predicate_fallbacks: int = 0
    episodes: int = 0
    transitions: int = 0
    bridge_doors: int = 0
    door_usage: int = 0  # épisodes dont un plan optimal utilise la porte
    by_goal: dict = field(default_factory=dict)
    by_d_star: dict = field(default_factory=dict)  # d* INITIAL par épisode

    def summary(self) -> dict:
        return {
            "layouts": self.layouts, "episodes": self.episodes,
            "transitions": self.transitions,
            "bridge_doors": self.bridge_doors,
            "door_usage_episodes": self.door_usage,
            "rejections": {"unreachable": self.rejected_unreachable,
                           "band": self.rejected_band,
                           "zero_fraction_cap": self.rejected_zero_fraction,
                           "g2_reservation": self.rejected_g2_reservation,
                           "non_g2": self.rejected_non_g2},
            "predicate_fallbacks": self.predicate_fallbacks,
            "by_goal": dict(self.by_goal), "by_d_star_hist": dict(self.by_d_star),
        }


def generate_layout(rng: random.Random, n_rooms: int,
                    bridge_fraction: float) -> Layout:
    """Arbre couvrant aléatoire + arêtes extra ; porte sur un pont avec
    probabilité bridge_fraction (documentée dans layout.meta)."""
    rooms = [f"v{i}" for i in range(n_rooms)]
    rng.shuffle(rooms)
    edges: list[tuple[str, str]] = []
    for i in range(1, n_rooms):
        j = rng.randrange(i)
        edges.append((rooms[i], rooms[j]))
    seen = {frozenset(e) for e in edges}
    n_extra = rng.randint(0, max(0, n_rooms // 2))
    tries = added = 0
    while added < n_extra and tries < 10 * n_rooms:
        tries += 1
        a, b = rng.sample(rooms, 2)
        if frozenset((a, b)) not in seen:
            edges.append((a, b))
            seen.add(frozenset((a, b)))
            added += 1

    tmp = Layout(rooms, edges, 0)
    bridges = [i for i in range(len(edges)) if tmp.is_bridge(i)]
    cycles = [i for i in range(len(edges)) if i not in bridges]
    if (rng.random() < bridge_fraction and bridges):
        door = rng.choice(bridges)
    elif cycles:
        door = rng.choice(cycles)
    else:
        door = rng.choice(bridges) if bridges else 0
    lay = Layout(rooms, edges, door)
    lay.meta = {"door_on_bridge": lay.is_bridge(door), "n_extra_edges": added}
    return lay


# --------------------------------------------------------------------------- #
# Tâches + trajectoires
# --------------------------------------------------------------------------- #

def _sample_init(rng: random.Random, layout: Layout) -> PhysicalState:
    return PhysicalState(agent=rng.choice(layout.rooms),
                         carried=None,
                         key_room=rng.choice(layout.rooms),
                         parcel_room=rng.choice(layout.rooms),
                         door_locked=True)


def _sample_goal(rng: random.Random, layout: Layout,
                  forced_pred: str | None = None) -> dict:
    """Échantillonne un but ; prédicat forcé optionnel (équilibre §4.2 —
    audit tagi-5 M3 : l'échantillonnage uniforme + filtrage de bande
    sous-représente REACH, qui a plus de d*\u2208{0,1})."""
    pred = forced_pred or rng.choice(["REACH", "HAVE", "AT"])
    if pred == "REACH":
        args = {"room": rng.choice(layout.rooms)}
    elif pred == "HAVE":
        args = {"object": rng.choice(["key", "parcel"])}
    else:
        args = {"object": rng.choice(["key", "parcel"]),
                "room": rng.choice(layout.rooms)}
    return {"predicate": pred, "args": args}


def _is_g2_reserved(layout: Layout, goal: dict) -> bool:
    """Combinaison réservée G2 (spec §7.4) : AT(clé, pièce jonction degré>=3).
    Exclue du train/val ; le pool test G2 la requiert sur layouts disjoints."""
    if goal["predicate"] != "AT" or goal["args"].get("object") != "key":
        return False
    room = goal["args"].get("room")
    return room in layout._adj and layout.degree(room) >= 3


def _goal_key(goal: dict) -> tuple:
    return (goal["predicate"], tuple(sorted(goal["args"].items())))


def _oracle_for(layout: Layout, goal: dict, cache: dict) -> LayoutOracle:
    k = (layout.layout_hash(), _goal_key(goal))
    o = cache.get(k)
    if o is None:
        o = LayoutOracle(layout, goal)
        cache[k] = o
    return o


def _plan_uses_door(oracle: LayoutOracle, layout: Layout,
                    state: PhysicalState, rng: random.Random) -> bool:
    """Un plan optimal (choix seedé parmi A*) utilise-t-il la porte au départ
    verrouillée ? Diagnostic de la valeur de la clé (spec §4.5)."""
    env = TinyGraphKey(layout)
    env.goal = oracle.goal
    env.state = state.copy()
    env.step_count = 0
    locked_at_start = state.door_locked
    used = False
    while not env.terminal and env.step_count < 4 * 64:
        if not oracle.reachable(env.state):
            break
        chosen = oracle.candidates[rng.choice(
            oracle.optimal_actions(env.state))]
        if chosen["action"] == "STOP":
            break
        if chosen["action"] == "UNLOCK":
            used = True
        if (chosen["action"] == "MOVE" and locked_at_start
                and frozenset((env.state.agent, chosen["arg"]))
                == frozenset(layout.door_rooms)):
            used = True
        env.execute(chosen)
    return used


def generate_episode(rng: random.Random, layout: Layout,
                     d_star_band: tuple[int, int],
                     stats: GenerationStats,
                     allow_zero: bool,
                     oracle_cache: dict,
                     tasks_out: list | None = None,
                     g2_mode: str = "off",
                     episode_ref: str = "",
                     forced_pred: str | None = None,
                     allow_zero_now: bool = False) -> list[dict] | None:
    """Échantillonne (init, but) jusqu'à d* dans la bande ; trajectoire A*.

    g2_mode: "off" (aucun filtrage), "exclude" (réserve G2 : rejette la
    combinaison AT(clé, jonction) — usage train/val), "require" (ne garde
    QUE cette combinaison — usage pool test G2 sur layouts disjoints).
    forced_pred: équilibre des prédicats (§4.2). allow_zero_now: quota
    d*=0 accordé par l'appelant (§4.5) — sinon les tirages d*=0 sont rejetés
    (comptés). Après 2/3 des tentatives, tout prédicat est accepté (fallback
    compté) pour ne jamais bloquer le pipeline."""
    attempts = 300
    # Re-vér tagi-5 §2: en require (cellule test §7.4), ni zéros (bande
    # stricte) ni relax de prédicat (tirages morts comptés fallback).
    allow_zero_now = allow_zero_now and g2_mode != "require"
    for i_attempt in range(attempts):
        state = _sample_init(rng, layout)
        relax = (forced_pred is not None
                 and i_attempt >= attempts // 3
                 and g2_mode != "require")
        goal = _sample_goal(rng, layout,
                            None if relax else forced_pred)
        if relax and goal["predicate"] != forced_pred:
            stats.predicate_fallbacks += 1
        reserved = _is_g2_reserved(layout, goal)
        if g2_mode == "exclude" and reserved:
            stats.rejected_g2_reservation += 1
            continue
        if g2_mode == "require" and not reserved:
            stats.rejected_non_g2 += 1
            continue
        oracle = _oracle_for(layout, goal, oracle_cache)
        sol = oracle.solve(state)
        if not sol["reachable"]:
            stats.rejected_unreachable += 1
            continue
        d0 = sol["d_star"]
        if d0 == 0:
            if not (allow_zero or allow_zero_now):
                stats.rejected_zero_fraction += 1
                continue
        elif not (d_star_band[0] <= d0 <= d_star_band[1]):
            stats.rejected_band += 1
            continue

        env = TinyGraphKey(layout)
        env.goal = goal
        env.state = state.copy()
        env.step_count = 0
        env.terminal = False
        records: list[dict] = []
        crossing = _plan_uses_door(oracle, layout, state, rng)
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
                    "observable_result": None,   # rempli après execute
                    "next_state_hash": None,     # rempli après execute
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
                    "split": "unassigned",  # assigné par le manifest WS-B
                    "source": "oracle",
                    "generator_version": GENERATOR_VERSION,
                    "oracle_version": ORACLE_VERSION,
                    # champs extra (canal provenance, jamais policy_input)
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
        if crossing:
            stats.door_usage += 1
        if tasks_out is not None:
            tasks_out.append({"layout_hash": layout.layout_hash(),
                              "goal": goal, "d_star": d0})
        return records
    return None


def generate(episodes: int, n_layouts: int, seed: int,
             rooms_range: tuple[int, int] = (4, 8),
             d_star_band: tuple[int, int] = (2, 12),
             bridge_fraction: float = 0.5, *,
             collect_details: bool = False,
             g2_mode: str = "off",
             episode_prefix: str = ""):
    """Retourne (lignes JSONL, stats[, détails]) — (layouts, tâches) si
    collect_details. Le flux RNG est identique avec ou sans collect_details :
    les octets JSONL ne changent jamais."""
    rng = random.Random(seed)
    stats = GenerationStats()
    layouts = [generate_layout(rng, rng.randint(*rooms_range), bridge_fraction)
               for _ in range(n_layouts)]
    stats.layouts = len(layouts)
    stats.bridge_doors = sum(1 for l in layouts if l.meta.get("door_on_bridge"))
    oracle_cache: dict = {}
    lines: list[str] = []
    tasks: list[dict] = []
    # M2 (audit tagi-5): quota d*=0 = premier arrivé jusqu'au quota (plus de
    # gate i%10 qui n'atteignait que ~1.4%% au lieu de ~10%%), min 1 si
    # episodes>=1. M3: cycle REACH/HAVE/AT pour équilibrer les prédicats.
    zero_quota = (max(1, round(episodes * D_STAR_ZERO_FRACTION))
                  if episodes >= 1 else 0)
    zero_done = 0
    pred_cycle = ["REACH", "HAVE", "AT"]
    for i in range(episodes):
        layout = layouts[rng.randrange(len(layouts))]
        allow_zero = zero_done < zero_quota
        forced = pred_cycle[i % 3] if g2_mode != "require" else "AT"
        ep_ref = f"{episode_prefix}{seed}-{i:05d}" if episode_prefix else ""
        recs = generate_episode(rng, layout, d_star_band, stats,
                                False, oracle_cache, tasks_out=tasks,
                                g2_mode=g2_mode, episode_ref=ep_ref,
                                forced_pred=forced,
                                allow_zero_now=allow_zero)
        if recs is not None and len(recs) == 1:
            # épisode INITIALEMENT satisfait (d0=0 ⇒ STOP unique) ; le
            # dernier pas d'un épisode de bande a aussi d*=0 mais ne compte
            # pas (bug détecté par vérification empirique post-M2)
            zero_done += 1
        if recs is None:
            continue
        lines.extend(json.dumps(r, sort_keys=True) for r in recs)
    if collect_details:
        details = {
            "layouts": [{"layout_hash": l.layout_hash(), "rooms": l.rooms,
                         "edges": [list(e) for e in l.edges],
                         "door_edge": l.door_edge, "meta": l.meta}
                        for l in layouts],
            "tasks": tasks,
        }
        return lines, stats.summary(), details
    return lines, stats.summary()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layouts", type=int, default=350)
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bridge-fraction", type=float, default=0.5)
    ap.add_argument("--d-star-min", type=int, default=2)
    ap.add_argument("--d-star-max", type=int, default=12)
    ap.add_argument("--rooms-min", type=int, default=4)
    ap.add_argument("--rooms-max", type=int, default=8)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--layouts-out", type=str, default=None,
                    help="JSON {layout_hash: layout} pour l'inventaire WS-B")
    ap.add_argument("--tasks-out", type=str, default=None,
                    help="JSON [{layout_hash, goal, d_star}] pour l'inventaire")
    ap.add_argument("--g2-mode", choices=["off", "exclude", "require"],
                    default="off",
                    help="exclude: réserve G2 (train/val) ; require: pool test G2")
    ap.add_argument("--episode-prefix", type=str, default="",
                    help="préfixe des episode_ref en provenance")
    args = ap.parse_args()

    t0 = time.time()
    want_details = bool(args.layouts_out or args.tasks_out)
    result = generate(
        episodes=args.episodes, n_layouts=args.layouts, seed=args.seed,
        rooms_range=(args.rooms_min, args.rooms_max),
        d_star_band=(args.d_star_min, args.d_star_max),
        bridge_fraction=args.bridge_fraction,
        collect_details=want_details,
        g2_mode=args.g2_mode,
        episode_prefix=args.episode_prefix)
    lines, stats = result[0], result[1]
    if want_details:
        details = result[2]
        if args.layouts_out:
            with open(args.layouts_out, "w") as f:
                json.dump({l["layout_hash"]: l for l in details["layouts"]},
                          f, indent=1, sort_keys=True)
        if args.tasks_out:
            with open(args.tasks_out, "w") as f:
                json.dump(details["tasks"], f, indent=1)
    stats["generation_seconds"] = round(time.time() - t0, 2)
    stats["lines"] = len(lines)
    stats["seed"] = args.seed
    stats["params"] = {"layouts": args.layouts, "episodes": args.episodes,
                       "bridge_fraction": args.bridge_fraction,
                       "d_star_band": [args.d_star_min, args.d_star_max]}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    stats_out = (args.out.rsplit(".", 1)[0] + ".stats.json")
    with open(stats_out, "w") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
