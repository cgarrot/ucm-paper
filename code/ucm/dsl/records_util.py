"""INVARIANT PARTAGÉ (lead 12:15-2c, 6e revue): toute trajectoire générée
par le DSL DOIT se terminer par le record terminal {STOP} (d*=0, but satisfait).
Le bug a existé dans p2/records (fix local 20:23) puis EST REVENU ailleurs —
la centralisation ici le rend impossible à oublier: les générateurs appellent
walk_optimal_plan() et ne construisent JAMAIS leurs records terminaux eux-mêmes.
"""
from __future__ import annotations

from ucm.dsl.core import DSLInterpreter
from ucm.dsl.bfs import DSLBFS


def walk_optimal_plan(interp: DSLInterpreter, task: dict, max_steps: int = 16,
                       emit_record=None):
    """Marche le plan optimal DSL depuis reset(task). Pour CHAQUE état visité
    (y compris l'état terminal d*=0), appelle emit_record(state, obs_builder,
    optimal_indices, d_star). L'état terminal est TOUJOURS émis avec
    optimal={STOP} — l'invariant vit ICI, pas chez l'appelant."""
    interp.reset(task)
    st = interp.state
    records_meta = []
    for _ in range(max_steps):
        interp.state = st
        if interp.goal_satisfied():
            cands = interp.candidates()
            stop_idx = [i for i, c in enumerate(cands) if c["action"] == "STOP"][0]
            meta = {"optimal_actions": [stop_idx], "d_star": 0,
                    "terminal": True}
            if emit_record:
                emit_record(st, meta)
            records_meta.append(meta)
            return records_meta
        sol = DSLBFS(interp).solve(st, max_states=40_000)
        if not sol["reachable"] or not sol["optimal_actions"]:
            return records_meta
        meta = {"optimal_actions": list(sol["optimal_actions"]),
                "d_star": sol["d_star"], "terminal": False}
        if emit_record:
            emit_record(st, meta)
        records_meta.append(meta)
        cand = interp.candidates()[sorted(sol["optimal_actions"])[0]]
        if cand["action"] == "STOP":
            return records_meta
        _, st = interp.successor(cand)
    return records_meta


def assert_terminal_stop_invariant(all_meta: list[dict]):
    """Garde partagée FAIL-CLOSED (lead 12:46): la dernière entrée doit être
    terminal d*=0 {STOP}. RuntimeError (pas assert: strippable sous -O —
    un invariant critique lève toujours)."""
    if not all_meta:
        return
    last = all_meta[-1]
    if not (last.get("terminal") and last["d_star"] == 0
            and len(last["optimal_actions"]) == 1):
        raise RuntimeError(f"INVARIANT STOP terminal violé: {last}")
