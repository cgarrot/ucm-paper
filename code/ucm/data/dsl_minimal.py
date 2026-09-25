"""DSL minimal P2 (lead 13:28, mission 3) — tests différentiels vs oracles existants.

Le DSL définit des mondes symboliques en YAML/JSON declaratif. Les tests
génèrent le MÊME monde en DSL et en code natif, exécutent les DEUX oracles,
et comparent les sorties épisode par épisode. Toute divergence = bug DSL.

Usage: from ucm.data.dsl_minimal import world_from_dsl, differential_test_tgk
"""
from __future__ import annotations

import json
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Grammar: un monde = {type, spec}
# TGK: {type: "tgk", rooms, edges, door_edge, init, goal}
# SIW: {type: "siw", views, nav_edges, widgets, init, goal}
# ---------------------------------------------------------------------------


_KNOWN_TYPES = frozenset({"tgk", "siw"})
_TGK_REQUIRED = frozenset({"type", "rooms", "edges", "door_edge", "init", "goal"})
_SIW_REQUIRED = frozenset({"type", "views", "nav_edges", "init", "goal"})
_COMPARE_FIELDS = ("d_star", "L_star", "reachable", "optimal_count")  # (b) constante figée


def world_from_dsl(dsl: dict) -> Any:
    """Construit un monde env depuis un dict DSL. Type dispatch.
    Fail-closed: type inconnu ou champ requis manquant → erreur explicite,
    JAMAIS de fallback silencieux (contrat pour l'extension aux nouveaux mondes)."""
    if not isinstance(dsl, dict):
        raise ValueError(f"DSL doit être un dict, reçu {type(dsl).__name__}")
    world_type = dsl.get("type")
    if world_type not in _KNOWN_TYPES:
        raise ValueError(f"type de monde inconnu: {world_type!r} — types connus: {sorted(_KNOWN_TYPES)}")
    required = _TGK_REQUIRED if world_type == "tgk" else _SIW_REQUIRED
    missing = required - set(dsl)
    if missing:
        raise ValueError(f"DSL {world_type}: champs requis manquants {sorted(missing)} — fail-closed")
    if world_type == "tgk":
        return _tgk_from_dsl(dsl)
    return _siw_from_dsl(dsl)


def _tgk_from_dsl(dsl: dict):
    from ..env.tinygraph import Layout, PhysicalState, TinyGraphKey, task_goal
    layout = Layout(dsl["rooms"], [tuple(e) for e in dsl["edges"]], dsl["door_edge"])
    init = dsl["init"]
    state = PhysicalState(
        agent=init["agent"], carried=init.get("carried"),
        key_room=init["key"] if not init.get("carried") == "key" else None,
        parcel_room=init["parcel"] if not init.get("carried") == "parcel" else None,
        door_locked=init.get("door_locked", True),
    )
    goal = task_goal(dsl["goal"]["predicate"], **dsl["goal"]["args"])
    env = TinyGraphKey(layout)
    env.goal = goal
    env.state = state
    env.step_count = 0
    env.terminal = False
    return env


def _siw_from_dsl(dsl: dict):
    from ..env.siw import SIWLayout, SIW, SIWState
    layout = SIWLayout({
        "views": dsl["views"],
        "widgets": dsl.get("widgets", []),
        "nav_edges": [tuple(e) for e in dsl["nav_edges"]],
    })
    init = dsl["init"]
    state = SIWState(
        view=init["view"], filled=frozenset(init.get("filled", [])),
        chosen=dict(init.get("chosen", {})), dialog_open=init.get("dialog_open", False),
        submitted=frozenset(init.get("submitted", [])),
    )
    goal = {"predicate": dsl["goal"]["predicate"], "args": dsl["goal"]["args"]}
    env = SIW(layout)
    env.goal = goal
    env.state = state
    env.step_count = 0
    env.terminal = False
    return env


# ---------------------------------------------------------------------------
# Tests différentiels: même monde en DSL et natif → oracles → comparaison
# ---------------------------------------------------------------------------


def differential_test_tgk(dsl: dict, *, horizon: int = 64) -> dict:
    """Génère le monde TGK depuis le DSL, exécute l'oracle, retourne la trace.
    L'appelant construit le même monde en natif et compare."""
    from ..env.oracle import solve as tgk_solve
    env = world_from_dsl(dsl)
    result = tgk_solve(env.layout, env.state, env.goal)
    return {
        "d_star": result["d_star"], "L_star": result["L_star"],
        "reachable": result["reachable"],
        "optimal_count": len(result["optimal_actions"]),
        "env_type": "tgk",
    }


def differential_test_siw(dsl: dict, *, horizon: int = 64) -> dict:
    """Génère le monde SIW depuis le DSL, exécute l'oracle, retourne la trace."""
    from ..env.siw_oracle import solve as siw_solve
    env = world_from_dsl(dsl)
    result = siw_solve(env.layout, env.state, env.goal)
    return {
        "d_star": result["d_star"], "L_star": result.get("L_star"),
        "reachable": result["reachable"],
        "optimal_count": len(result["optimal_actions"]),
        "env_type": "siw",
    }


def compare_traces(trace_a: dict, trace_b: dict) -> dict:
    """Compare deux traces oracle sur _COMPARE_FIELDS (constante figée)."""
    divergences = [f for f in _COMPARE_FIELDS if trace_a.get(f) != trace_b.get(f)]
    return {"match": not divergences, "divergences": divergences,
            "trace_a": {f: trace_a.get(f) for f in _COMPARE_FIELDS},
            "trace_b": {f: trace_b.get(f) for f in _COMPARE_FIELDS}}


def run_differential_suite() -> list[dict]:
    """Suite complète: mondes DSL vs natifs → oracles → comparaison.
    Toute divergence = bug DSL (le monde natif est la référence)."""
    results = []

    # === TGK: monde simple ligne a-b-c, porte b-c, but AT colis room_c ===
    tgk_dsl = {
        "type": "tgk",
        "rooms": ["room_a", "room_b", "room_c"],
        "edges": [["room_a", "room_b"], ["room_b", "room_c"]],
        "door_edge": 1,
        "init": {"agent": "room_a", "carried": None, "key": "room_a", "parcel": "room_d" if "room_d" in ["room_a","room_b","room_c"] else "room_b", "door_locked": True},
        "goal": {"predicate": "AT", "args": {"object": "parcel", "room": "room_c"}},
    }
    # natif: même monde en code direct
    from ..env.tinygraph import Layout, PhysicalState, TinyGraphKey, task_goal
    from ..env.oracle import solve as tgk_solve
    lay = Layout(["room_a", "room_b", "room_c"],
                 [("room_a", "room_b"), ("room_b", "room_c")], 1)
    st = PhysicalState("room_a", None, "room_a", "room_b", True)
    goal = task_goal("AT", object="parcel", room="room_c")
    native = tgk_solve(lay, st, goal)
    native_trace = {"d_star": native["d_star"], "L_star": native["L_star"],
                    "reachable": native["reachable"],
                    "optimal_count": len(native["optimal_actions"]), "env_type": "tgk"}
    dsl_trace = differential_test_tgk(tgk_dsl)
    results.append({"test": "tgk_line_3rooms", **compare_traces(dsl_trace, native_trace)})

    # === TGK: monde avec jonction deg3, but REACH ===
    tgk_dsl2 = {
        "type": "tgk",
        "rooms": ["a", "b", "c", "d"],
        "edges": [["a", "b"], ["b", "c"], ["b", "d"]],
        "door_edge": 0,
        "init": {"agent": "a", "carried": None, "key": "c", "parcel": "d", "door_locked": False},
        "goal": {"predicate": "REACH", "args": {"room": "d"}},
    }
    lay2 = Layout(["a", "b", "c", "d"], [("a", "b"), ("b", "c"), ("b", "d")], 0)
    st2 = PhysicalState("a", None, "c", "d", False)
    goal2 = task_goal("REACH", room="d")
    native2 = tgk_solve(lay2, st2, goal2)
    native_trace2 = {"d_star": native2["d_star"], "L_star": native2["L_star"],
                     "reachable": native2["reachable"],
                     "optimal_count": len(native2["optimal_actions"]), "env_type": "tgk"}
    dsl_trace2 = differential_test_tgk(tgk_dsl2)
    results.append({"test": "tgk_junction_reach", **compare_traces(dsl_trace2, native_trace2)})

    # === SIW: monde 3 vues, 1 form 1 champ, but SET ===
    siw_dsl = {
        "type": "siw",
        "views": ["vw0", "vw1", "vw2"],
        "nav_edges": [["vw0", "vw1"], ["vw1", "vw2"]],
        "widgets": [
            {"id": "form0", "type": "form", "view": "vw0"},
            {"id": "fd0_0", "type": "field", "view": "vw0", "form": "form0"},
            {"id": "fd0_1", "type": "field", "view": "vw0", "form": "form0"},
            {"id": "bt_s0", "type": "button", "view": "vw0", "kind": "submit", "submit_for": "form0"},
        ],
        "init": {"view": "vw2", "filled": [], "chosen": {}, "dialog_open": False, "submitted": []},
        "goal": {"predicate": "SET", "args": {"field": "fd0_0"}},
    }
    from ..env.siw import SIWLayout, SIW, SIWState
    from ..env.siw_oracle import solve as siw_solve
    lay_s = SIWLayout({"views": ["vw0", "vw1", "vw2"],
                       "widgets": [{"id": "form0", "type": "form", "view": "vw0"},
                                   {"id": "fd0_0", "type": "field", "view": "vw0", "form": "form0"},
                                   {"id": "fd0_1", "type": "field", "view": "vw0", "form": "form0"},
                                   {"id": "bt_s0", "type": "button", "view": "vw0", "kind": "submit", "submit_for": "form0"}],
                       "nav_edges": [("vw0", "vw1"), ("vw1", "vw2")]})
    st_s = SIWState("vw2", frozenset(), {}, False, frozenset())
    goal_s = {"predicate": "SET", "args": {"field": "fd0_0"}}
    native_s = siw_solve(lay_s, st_s, goal_s)
    native_trace_s = {"d_star": native_s["d_star"], "L_star": native_s.get("L_star"),
                      "reachable": native_s["reachable"],
                      "optimal_count": len(native_s["optimal_actions"]), "env_type": "siw"}
    dsl_trace_s = differential_test_siw(siw_dsl)
    results.append({"test": "siw_3views_set", **compare_traces(dsl_trace_s, native_trace_s)})

    return results
