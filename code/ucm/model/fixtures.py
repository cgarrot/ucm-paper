"""WS-C fixtures/adapters on the CANONICAL WS-A environment (post-M0 landing).

Superseded the early dev-only synthetic world: everything here now wraps the
real ``ucm.env.tinygraph`` / ``ucm.env.oracle`` / ``ucm.data.generate`` code so
that WS-C consumes exactly the M0-canonical policy_input format:

    goal:       {"predicate": "REACH"|"HAVE"|"AT", "args": {...}}   (§4.2)
    entities:   [{"id", "type": room|agent|key|parcel|door, "attrs": {...}}]
                door attrs: {"locked": bool}
    relations:  [{"subj", "pred": adjacent|connects|unlocks|at|held, "obj"}]
    candidates: [{"action": MOVE|PICK|DROP|UNLOCK|STOP, "arg": id|None}]  K=R+6
    supervision.optimal_actions: list[int]  (indices into candidates, canonical order)

permute_obs implements the coherent permutation transform (G5) on this format.
"""

from __future__ import annotations

import random

from ucm.data import generate as gen
from ucm.env.oracle import LayoutOracle, candidate_actions
from ucm.env.tinygraph import Layout, TinyGraphKey, goal_satisfied

# canonical vocabularies (mirror of ucm.env, single source of truth = WS-A)
ENTITY_TYPES = ["room", "agent", "key", "parcel", "door"]
RELATION_PREDS = ["adjacent", "connects", "unlocks", "at", "held"]
GOAL_PREDICATES = ["REACH", "HAVE", "AT"]
ACTION_TYPES = ["MOVE", "PICK", "DROP", "UNLOCK", "STOP"]

MAX_ROOMS = 12


# ---------------------------------------------------------------------------
# canonical record generation (real env + real oracle)
# ---------------------------------------------------------------------------

def make_layout(rng: random.Random, n_rooms: int | None = None,
                bridge_fraction: float = 0.4) -> Layout:
    n = n_rooms if n_rooms is not None else rng.randint(4, 8)
    return gen.generate_layout(rng, n, bridge_fraction)


def make_task(rng: random.Random, layout: Layout,
              d_star_band: tuple[int, int] = (0, 12)) -> tuple[dict, dict] | None:
    """Sample (task, initial_solve) with d* in band on the REAL dynamics.
    Returns None after the sampling budget is exhausted (caller retries)."""
    for _ in range(300):
        state = gen._sample_init(rng, layout)
        goal = gen._sample_goal(rng, layout)
        oracle = LayoutOracle(layout, goal)  # small layouts: rebuild is cheap
        sol = oracle.solve(state)
        if not sol["reachable"]:
            continue
        if d_star_band[0] <= sol["d_star"] <= d_star_band[1]:
            return ({"init": {"agent": state.agent, "carried": state.carried,
                             "key": state.key_room, "parcel": state.parcel_room,
                             "door_locked": state.door_locked},
                    "goal": goal}, sol)
    return None


def episode_records(rng: random.Random, layout: Layout, task: dict) -> list[dict]:
    """Full A*-trajectory records for one episode on the REAL env (canonical)."""
    env = TinyGraphKey(layout)
    env.reset(task)
    goal = task["goal"]
    oracle = LayoutOracle(layout, goal)
    records = []
    while not env.terminal:
        obs = env.observe()
        sol = oracle.solve(env.state)
        records.append({
            "schema_version": "0.2",
            "policy_input": {"goal": obs["goal"], "entities": obs["entities"],
                             "relations": obs["relations"], "candidates": obs["candidates"]},
            "supervision": {"optimal_actions": sol["optimal_actions"],
                            "d_star": sol["d_star"], "reachable": sol["reachable"]},
            "provenance": {"layout_hash": layout.layout_hash(), "split": "ws-c-dev",
                           "source": "oracle"},
        })
        act = obs["candidates"][rng.choice(sol["optimal_actions"])]
        env.execute(act)
    return records


def make_records(rng: random.Random, n_records: int, d_star_band=(0, 12),
                 rooms_range=(4, 8)) -> list[dict]:
    """Collect n_records labelled transitions (initial states of fresh episodes)
    over fresh layouts — canonical format, ready for tensorize_batch."""
    out: list[dict] = []
    while len(out) < n_records:
        lay = gen.generate_layout(rng, rng.randint(*rooms_range), 0.4)
        t = make_task(rng, lay, d_star_band)
        if t is None:
            continue
        task, sol = t
        env = TinyGraphKey(lay)
        obs = env.reset(task)
        out.append({
            "schema_version": "0.2",
            "policy_input": {"goal": obs["goal"], "entities": obs["entities"],
                             "relations": obs["relations"], "candidates": obs["candidates"]},
            "supervision": {"optimal_actions": sol["optimal_actions"],
                            "d_star": sol["d_star"], "reachable": True},
            "provenance": {"layout_hash": lay.layout_hash(), "split": "ws-c-dev",
                           "source": "oracle"},
        })
    return out


# ---------------------------------------------------------------------------
# coherent permutation (G5) on the canonical format
# ---------------------------------------------------------------------------

def permute_obs(obs: dict, labels: list[int] | None,
                rng: random.Random) -> tuple[dict, list[int] | None, dict]:
    """Rename ids, shuffle entity/relation/candidate orders (labels permuted
    together). Same problem, permuted presentation — tensorization + model
    must be invariant (G5, tol 1e-5 FP32). Returns (obs', labels', old→new map)."""
    perm = list(range(len(obs["entities"])))
    rng.shuffle(perm)
    new_ids = {e["id"]: f"e{perm[i]}" for i, e in enumerate(obs["entities"])}
    tr = lambda x: new_ids[x]

    entities = [dict(e, id=tr(e["id"])) for e in obs["entities"]]
    rng.shuffle(entities)
    relations = [dict(r, subj=tr(r["subj"]), obj=tr(r["obj"])) for r in obs["relations"]]
    rng.shuffle(relations)

    order = list(range(len(obs["candidates"])))
    rng.shuffle(order)
    candidates = [dict(obs["candidates"][i],
                       arg=(tr(obs["candidates"][i]["arg"])
                            if obs["candidates"][i]["arg"] is not None else None))
                  for i in order]
    args = obs["goal"]["args"]
    goal = {"predicate": obs["goal"]["predicate"],
            "args": {k: tr(v) if isinstance(v, str) else v for k, v in args.items()}}

    new_labels = None
    if labels is not None:
        new_labels = [0] * len(labels)
        for new_i, old_i in enumerate(order):
            new_labels[new_i] = labels[old_i]
    return ({"goal": goal, "entities": entities, "relations": relations,
             "candidates": candidates}, new_labels, new_ids)


def labels_of(record: dict) -> list[int]:
    """0/1 label mask from supervision indices (binding to candidate order)."""
    K = len(record["policy_input"]["candidates"])
    lab = [0] * K
    for i in record["supervision"]["optimal_actions"]:
        if not (0 <= i < K):
            raise ValueError(f"optimal action index {i} out of range [0,{K})")
        lab[i] = 1
    if sum(lab) == 0:
        raise ValueError("empty optimal set")
    return lab
