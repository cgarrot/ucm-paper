"""Fixtures WS-B sur le VRAI env/oracle WS-A (conformance permanente).

Chaque fixture produit des records exactement conformes au contrat PLAN §3.3,
générés par ``TinyGraphKey`` + ``LayoutOracle`` : si WS-A change l'observation,
ces fixtures (et les tests) cassent immédiatement — plus de dérive possible.

Layout de référence : 4 pièces, porte sur un pont (b–c), jonction room_b (deg 3)
pour les tests G2.
"""

from __future__ import annotations

from typing import Any, Optional

from ..env.oracle import solve
from ..env.tinygraph import Layout, TinyGraphKey, task_goal
from .schema import make_state_goal_hash, policy_input_from_obs

# Layout : a–b, b–c (porte, pont), b–d. room_b = jonction deg 3.
ROOMS = ["room_a", "room_b", "room_c", "room_d"]
EDGES = [("room_a", "room_b"), ("room_b", "room_c"), ("room_b", "room_d")]
DOOR_EDGE = 1  # room_b–room_c

LAYOUT = Layout(ROOMS, EDGES, DOOR_EDGE)

DEFAULT_INIT = {
    "agent": "room_a",
    "carried": None,
    "key": "room_a",
    "parcel": "room_d",
    "door_locked": True,
}

DEFAULT_GOAL = task_goal("AT", object="parcel", room="room_c")


def make_task(goal: Optional[dict] = None, init: Optional[dict] = None, layout: Optional[Layout] = None) -> dict:
    return {"init": dict(init or DEFAULT_INIT), "goal": dict(goal or DEFAULT_GOAL)}


def make_record(
    *,
    action_ref: int = 0,
    goal: Optional[dict] = None,
    init: Optional[dict] = None,
    layout: Optional[Layout] = None,
    split: str = "train",
    source: str = "oracle",
    provenance_extra: Optional[dict] = None,
    generator_version: str = "fixtures-0.1",
    oracle_version: str = "oracle-0.1",
    optimal_actions: Optional[list[int]] = None,
    d_star: Optional[int] = None,
    reachable: Optional[bool] = None,
) -> dict:
    """Une transition réelle : reset → obs → solve → execute → record §3.3."""
    layout = layout or LAYOUT
    task = make_task(goal, init)
    env = TinyGraphKey(layout)
    obs = env.reset(task)
    policy_input = policy_input_from_obs(obs)  # validation d'abord

    state_before = env.state.copy()
    s_hash = env.state_hash()
    l_hash = layout.layout_hash()
    sg_hash = make_state_goal_hash(l_hash, s_hash, env.goal)
    supervision = solve(layout, state_before, env.goal)

    result = env.execute(env.candidates()[action_ref])
    reachable_final = supervision["reachable"] if reachable is None else reachable
    optimal_final = list(optimal_actions) if optimal_actions is not None else list(supervision["optimal_actions"])
    if reachable_final is False and optimal_actions is None:
        optimal_final = []
    if reachable_final is False and d_star is None:
        d_star_final = None
    else:
        d_star_final = (supervision["d_star"] if supervision["reachable"] else None) if d_star is None else d_star
    record = {
        "schema_version": "0.2",
        "policy_input": policy_input,
        "execution": {
            "action_ref": action_ref,
            "observable_result": "valid" if result["valid"] else "invalid",
            "next_state_hash": env.state_hash(),
        },
        "supervision": {
            "optimal_actions": optimal_final,
            "d_star": d_star_final,
            "reachable": reachable_final,
        },
        "provenance": {
            "layout_hash": l_hash,
            "state_goal_hash": sg_hash,
            "split": split,
            "source": source,
            "generator_version": generator_version,
            "oracle_version": oracle_version,
        },
    }
    if provenance_extra:
        record["provenance"].update(provenance_extra)
    return record
