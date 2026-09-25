"""Baselines V0 (spec §9.1) on the canonical WS-A observation format.

Each arm declares its information budget:

1. random_syntactic     — uniform over ALL candidates (no precondition knowledge)
2. random_valid         — uniform over physically valid actions; precondition-
                          informed arm, clearly labelled (§9.1)
3. LocalHeuristic       — explicit local rules (documented below): no distances,
                          no oracle plan, no multi-step lookahead
4. bfs_oracle_policy    — exact search via the oracle: SUPERVISION BOUND with
                          privileged transition knowledge, declared as such
5. KNNRetrievalPolicy   — structural-signature retrieval over train records
                          (no test neighbours); fallback random syntactic

LocalHeuristic rules (computed from policy_input only, 1-hop):
    R1  goal satisfied (per observation)               → STOP
    R2  HAVE(o): o in agent's room & hand empty        → PICK(o)
    R3  AT(o,r): holding o & agent in r                → DROP(o)
    R4  holding an object other than the needed one    → DROP(held)
    R5  holding key & locked door at an adjacent edge  → UNLOCK(door)
    R6  otherwise                                      → random VALID move
Navigation is impossible without distances: R6's random walk makes this a weak
local arm — documented, which is what §9.1 requires.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Callable

# ---------------------------------------------------------------------------
# helpers reading ONLY the canonical policy_input observation
# ---------------------------------------------------------------------------

def obs_graph(obs: dict):
    """Parse entities/relations → (entities_by_id, adjacency, located, carried,
    door_locks). Relations use subj/pred/obj (adjacent|connects|unlocks|at|held)."""
    entities = {e["id"]: e for e in obs["entities"]}
    adj: dict[str, set] = {e["id"]: set() for e in obs["entities"]}
    located: dict[str, str] = {}   # entity → room ("at")
    carried = None                 # object id ("held" → agent)
    door_rooms: dict[str, set] = {}
    key_of_door: dict[str, str] = {}
    for r in obs["relations"]:
        s, p, o = r["subj"], r["pred"], r["obj"]
        if p == "adjacent":
            adj[s].add(o)
            adj[o].add(s)
        elif p == "at":
            located[s] = o
        elif p == "held":
            carried = s
        elif p == "connects":
            door_rooms.setdefault(s, set()).add(o)
        elif p == "unlocks":
            key_of_door[o] = s
    door_locked = {d["id"]: bool(d.get("attrs", {}).get("locked", False))
                   for d in obs["entities"] if d["type"] == "door"}
    return entities, adj, located, carried, door_rooms, door_locked, key_of_door


def goal_satisfied_obs(obs: dict) -> bool:
    entities, adj, located, carried, *_ = obs_graph(obs)
    pred, args = obs["goal"]["predicate"], obs["goal"]["args"]
    if pred == "REACH":
        return located.get("agent") == args["room"]
    if pred == "HAVE":
        return carried == args["object"]
    if pred == "AT":
        return located.get(args["object"]) == args["room"]  # held ⇒ None ≠ room
    raise ValueError(pred)


def valid_actions(obs: dict) -> list[dict]:
    """Physically valid actions derived from observable preconditions (§4.3).
    MOVE excludes the locked door edge; PICK/DROP/UNLOCK/STOP per preconditions."""
    (entities, adj, located, carried, door_rooms, door_locked,
     key_of_door) = obs_graph(obs)
    agent_room = located.get("agent")
    acts: list[dict] = []
    for r in sorted(adj.get(agent_room, ())):
        blocked = any(locked and rooms == {agent_room, r}
                      for d, rooms in door_rooms.items()
                      for locked in [door_locked.get(d, False)])
        if not blocked:
            acts.append({"action": "MOVE", "arg": r})
    for e in obs["entities"]:
        if e["type"] in ("key", "parcel"):
            oid = e["id"]
            if located.get(oid) == agent_room and carried is None:
                acts.append({"action": "PICK", "arg": oid})
            if carried == oid:
                acts.append({"action": "DROP", "arg": oid})
    for d, rooms in door_rooms.items():
        if door_locked.get(d) and agent_room in rooms:
            key = key_of_door.get(d)
            if key is not None and carried == key:
                acts.append({"action": "UNLOCK", "arg": d})
    if goal_satisfied_obs(obs):
        acts.append({"action": "STOP", "arg": None})
    return acts


# ---------------------------------------------------------------------------
# 1) random syntactic
# ---------------------------------------------------------------------------

def make_random_syntactic(seed: int = 0) -> Callable:
    rng = random.Random(seed)

    def policy(obs: dict) -> dict:
        return rng.choice(obs["candidates"])

    policy.name = "random_syntactic"
    return policy


# ---------------------------------------------------------------------------
# 2) random valid (precondition-informed, labelled)
# ---------------------------------------------------------------------------

def make_random_valid(seed: int = 0) -> Callable:
    rng = random.Random(seed)

    def policy(obs: dict) -> dict:
        acts = valid_actions(obs)
        if not acts:  # goal unsatisfied & no valid move: invalid STOP is visible
            return {"action": "STOP", "arg": None}
        return rng.choice(acts)

    policy.name = "random_valid"
    return policy


# ---------------------------------------------------------------------------
# 3) local heuristic (documented rules; no distances/oracle)
# ---------------------------------------------------------------------------

class LocalHeuristic:
    name = "local_heuristic"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def __call__(self, obs: dict) -> dict:
        (entities, adj, located, carried, door_rooms, door_locked,
         key_of_door) = obs_graph(obs)
        agent_room = located.get("agent")
        pred, args = obs["goal"]["predicate"], obs["goal"]["args"]
        goal_obj = args.get("object")
        goal_room = args.get("room")

        if goal_satisfied_obs(obs):                                   # R1
            return {"action": "STOP", "arg": None}
        if pred == "HAVE" and located.get(goal_obj) == agent_room and carried is None:
            return {"action": "PICK", "arg": goal_obj}                # R2
        if pred == "AT":
            if carried == goal_obj and agent_room == goal_room:
                return {"action": "DROP", "arg": goal_obj}            # R3
            if carried is not None and carried != goal_obj:
                return {"action": "DROP", "arg": carried}             # R4
            if located.get(goal_obj) == agent_room and carried is None:
                return {"action": "PICK", "arg": goal_obj}
        if pred == "HAVE" and carried is not None and carried != goal_obj:
            return {"action": "DROP", "arg": carried}                 # R4
        va = valid_actions(obs)
        unlocks = [a for a in va if a["action"] == "UNLOCK"]
        if unlocks:                                                   # R5
            return unlocks[0]
        moves = [a for a in va if a["action"] == "MOVE"]
        if moves:                                                     # R6
            return self.rng.choice(moves)
        return {"action": "STOP", "arg": None}


# ---------------------------------------------------------------------------
# 4) exact search via oracle — SUPERVISION BOUND (privileged, declared)
# ---------------------------------------------------------------------------

def make_bfs_oracle_policy(solve_fn) -> Callable:
    """solve_fn(obs) → {"optimal_actions": [int]} binding into the CURRENT
    candidates list. Oracle access at every step = privileged arm (§9.1/§4.6)."""

    def policy(obs: dict) -> dict:
        sol = solve_fn(obs)
        for i in sol.get("optimal_actions", []):
            return obs["candidates"][i]
        raise RuntimeError("oracle returned no optimal action")

    policy.name = "bfs_oracle_bound"
    return policy


# ---------------------------------------------------------------------------
# 5) kNN structural retrieval over train records (no test neighbours)
# ---------------------------------------------------------------------------

def structural_signature(obs: dict) -> tuple:
    """Typed structural signature invariant to ids and orders: sorted entity
    descriptors (type, degree, locked), goal predicate + ref roles, relation
    multiset, carried flag."""
    entities, adj, located, carried, *_ = obs_graph(obs)
    ent_sig = sorted(
        (e["type"], len(adj.get(e["id"], ())),
         bool(e.get("attrs", {}).get("locked", False)))
        for e in obs["entities"])
    args = obs["goal"]["args"]
    ref_roles = tuple(sorted(
        (k, entities[v]["type"]) for k, v in args.items()))
    rel_sig = tuple(sorted(
        (r["pred"], entities[r["subj"]]["type"], entities[r["obj"]]["type"])
        for r in obs["relations"]))
    return (tuple(ent_sig), obs["goal"]["predicate"], ref_roles, rel_sig,
            carried is not None)


class KNNRetrievalPolicy:
    name = "knn_retrieval"

    def __init__(self, train_records: list[dict], seed: int = 0, k: int = 5):
        self.rng = random.Random(seed)
        self.k = k
        self.index: dict[tuple, list[tuple]] = {}
        for rec in train_records:
            sig = structural_signature(rec["policy_input"])
            cands = rec["policy_input"]["candidates"]
            for i in rec["supervision"]["optimal_actions"]:
                c = cands[i]
                self.index.setdefault(sig, []).append((c["action"], c["arg"]))

    def __call__(self, obs: dict) -> dict:
        sig = structural_signature(obs)
        if sig in self.index and self.index[sig]:
            top = Counter(self.index[sig]).most_common()
            best = top[0][1]
            tied = [a for a, c in top if c == best]
            act = self.rng.choice(tied)
            for c in obs["candidates"]:
                if c["action"] == act[0] and c["arg"] == act[1]:
                    return c
            # action not present among this obs's candidates → structural neighbour
            # disagrees with binding; fall through to declared fallback
        return self.rng.choice(obs["candidates"])  # declared fallback (syntactic random)
