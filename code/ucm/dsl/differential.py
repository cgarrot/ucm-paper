"""Différentiel pas-à-pas DSL vs moteur NATIF TGK — jalon 1.

Pour chaque état d'une fermeture avant (construits par RESET uniquement):
  - candidats identiques (liste exacte)
  - validité identique pour CHAQUE candidat
  - successeur identique (forme canonique) pour chaque candidat valide
  - vérité du but identique
  - d* et actions optimales identiques (BFS DSL vs LayoutOracle natif)
Le différentiel est la PREUVE que le natif est une instance fidèle du DSL.
"""
from __future__ import annotations

import json
import random

from ucm.dsl.core import DSLInterpreter
from ucm.dsl.tgk_program import TGK_DSL_PROGRAM, layout_from_native


def _native_state_key(st) -> tuple:
    return (st.agent, st.carried, st.key_room, st.parcel_room,
            bool(st.door_locked))


def _dsl_state_key(st) -> tuple:
    f = st.fields
    return (f["agent"], f["carried"], f["key_room"], f["parcel_room"],
            bool(f["door_locked"]))


def differential_tgk(layout, tasks: list[dict], seed: int = 0,
                     max_states: int = 2000) -> dict:
    """tasks = [{init, goal}] natifs. Retourne le rapport de différentiel."""
    from ucm.env.tinygraph import TinyGraphKey
    from ucm.env.oracle import LayoutOracle

    tables = layout_from_native(layout)
    interp = DSLInterpreter(TGK_DSL_PROGRAM, tables)

    n_states = n_cands = n_valid_match = n_succ_match = 0
    n_goal_match = n_dstar_match = n_opt_match = 0
    states_checked = set()
    mismatches = []

    def check_state(dsl_state, native_state, goal):
        nonlocal n_states, n_cands, n_valid_match, n_succ_match, n_goal_match
        k = (_dsl_state_key(dsl_state), goal["predicate"],
             json.dumps(goal["args"], sort_keys=True))
        if k in states_checked:
            return
        states_checked.add(k)
        n_states += 1
        # env natif replacé à cet état PAR RESET (B1: jamais d'injection)
        env = TinyGraphKey(layout)
        task = {"init": {"agent": native_state.agent,
                         "carried": native_state.carried,
                         "key": native_state.key_room,
                         "parcel": native_state.parcel_room,
                         "door_locked": bool(native_state.door_locked)},
                "goal": goal}
        env.reset(task)
        interp.task = {"init": task["init"], "goal": goal}
        interp.state = dsl_state
        interp.terminal = False

        # 1. candidats identiques
        if interp.candidates() != env.candidates():
            mismatches.append(("candidates", k))
        # 2-3. validité + successeur pas-à-pas
        for cand in env.candidates():
            n_cands += 1
            valid_n = env._is_valid(cand)
            spec = (interp._action_spec(cand["action"])
                    if cand["action"] != "STOP" else None)
            valid_d = interp._guard_ok(spec, cand) if spec else True
            if valid_d == valid_n:
                n_valid_match += 1
            else:
                mismatches.append(("valid", k, cand))
            if cand["action"] != "STOP" and valid_n:
                _, nxt_d = interp.successor(cand)
                # exécuter le natif sur une copie (reset de nouveau)
                env2 = TinyGraphKey(layout)
                env2.reset(task)
                env2.execute(cand)
                if _dsl_state_key(nxt_d) == _native_state_key(env2.state):
                    n_succ_match += 1
                else:
                    mismatches.append(("succ", k, cand))
        # 4. vérité du but
        if interp.goal_satisfied() == env.goal_satisfied():
            n_goal_match += 1
        else:
            mismatches.append(("goal", k))

    for task in tasks:
        goal = task["goal"]
        env = TinyGraphKey(layout)
        try:
            env.reset(task)
        except Exception:
            continue  # tâche rejetée par le NATIF aussi — hors du comparables
        o = LayoutOracle(layout, goal)
        # fermeture avant native depuis l'état initial
        seen = {}
        start_native = env.state
        from collections import deque
        q = deque([(start_native.key(), start_native)])
        seen[start_native.key()] = start_native
        while q and len(seen) < max_states:
            _, st = q.popleft()
            # état DSL correspondant — PAR RESET avec l'état natif re-sérialisé
            init = {"agent": st.agent, "carried": st.carried,
                    "key": st.key_room, "parcel": st.parcel_room,
                    "door_locked": bool(st.door_locked)}
            interp.reset({"init": init, "goal": goal})
            dsl_st = interp.state
            check_state(dsl_st, st, goal)
            # successeurs natifs pour élargir la fermeture
            envx = TinyGraphKey(layout)
            envx.reset({"init": init, "goal": goal})
            for cand in envx.candidates():
                if cand["action"] == "STOP" or not envx._is_valid(cand):
                    continue
                env2 = TinyGraphKey(layout)
                env2.reset({"init": init, "goal": goal})
                env2.execute(cand)
                kk = env2.state.key()
                if kk not in seen:
                    seen[kk] = env2.state
                    q.append((kk, env2.state))
        # 5. d* + optimaux: comparaison par état initial de tâche
        interp.reset(task)
        from ucm.dsl.bfs import DSLBFS
        sol = DSLBFS(interp).solve(interp.state)
        o = LayoutOracle(layout, goal)
        st0 = TinyGraphKey(layout)
        st0.reset(task)
        d_n = o.d_star(st0.state)
        if sol["d_star"] == d_n:
            n_dstar_match += 1
        else:
            mismatches.append(("d_star", sol["d_star"], d_n))
        # optimaux: sémantique d'indices ↔ natif (compare les ACTIONS)
        opt_d = {json.dumps(interp.candidates()[i], sort_keys=True)
                 for i in sol["optimal_actions"]}
        opt_n = {json.dumps(st0.candidates()[i], sort_keys=True)
                 for i in o.optimal_actions(st0.state)}
        if opt_d == opt_n:
            n_opt_match += 1
        else:
            mismatches.append(("optimal", sorted(map(json.dumps, opt_d)),
                               sorted(map(json.dumps, opt_n))))

    return {"n_states": n_states, "n_candidates_checked": n_cands,
            "valid_match": n_valid_match, "succ_match": n_succ_match,
            "goal_match": n_goal_match, "d_star_match": n_dstar_match,
            "optimal_match": n_opt_match,
            "n_mismatches": len(mismatches),
            "mismatches_sample": mismatches[:10],
            "equal": not mismatches}
