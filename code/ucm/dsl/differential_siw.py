"""Différentiel pas-à-pas DSL vs moteurs NATIFS SIW — jalon 2.

Même discipline que le jalon 1 TGK: pour chaque état d'une fermeture avant
(tous construits par RESET uniquement), comparer candidats/validité/
successeur/but; par tâche: d* + ensembles optimaux vs SIWOracle natif.
"""
from __future__ import annotations

import json
from collections import deque

from ucm.dsl.core import DSLInterpreter
from ucm.dsl.siw_program import SIW_DSL_PROGRAM, layout_tables_from_siw


def _native_key(st) -> tuple:
    return (st.view, sorted(st.filled), sorted(st.chosen.items()),
            bool(st.dialog_open), sorted(st.submitted))


def _dsl_key(st) -> tuple:
    f = st.fields
    chosen = sorted((k, v) for k, v in f["chosen"].items()) \
        if isinstance(f["chosen"], dict) else []
    return (f["view"], sorted(f["filled"]), chosen,
            bool(f["dialog_open"]), sorted(f["submitted"]))


def _task_of(native_state, goal) -> dict:
    return {"init": {"view": native_state.view,
                     "filled": sorted(native_state.filled),
                     "chosen": dict(native_state.chosen),
                     "dialog_open": bool(native_state.dialog_open),
                     "submitted": sorted(native_state.submitted)},
            "goal": goal}


def differential_siw(layout, tasks: list[dict], max_states: int = 2000) -> dict:
    from ucm.env.siw import SIW
    from ucm.env.siw_oracle import SIWOracle

    tables = layout_tables_from_siw(layout)
    interp = DSLInterpreter(SIW_DSL_PROGRAM, tables)
    n_states = n_cands = n_valid = n_succ = n_goal = 0
    n_dst = n_opt = 0
    mismatches = []

    for task in tasks:
        goal = task["goal"]
        env = SIW(layout)
        env.reset(task)
        seen = {env.state.key(): env.state}
        q = deque([(env.state.key(), env.state)])
        checked = set()
        while q and len(seen) <= max_states:
            _, st = q.popleft()
            k = json.dumps(_native_key(st), sort_keys=True, default=str)
            if k in checked:
                continue
            checked.add(k)
            n_states += 1
            t = _task_of(st, goal)
            envx = SIW(layout)
            envx.reset(t)
            interp.reset(t)
            if interp.candidates() != envx.candidates():
                mismatches.append(("candidates", _native_key(st)))
            for cand in envx.candidates():
                n_cands += 1
                vn, _eff = envx._is_valid(cand)
                spec = (interp._action_spec(cand["action"])
                        if cand["action"] != "STOP" else None)
                vd = interp._guard_ok(spec, cand) if spec else True
                if vn == vd:
                    n_valid += 1
                else:
                    mismatches.append(("valid", _native_key(st), cand))
                if cand["action"] != "STOP" and vn and vd:
                    _, nxt_d = interp.successor(cand)
                    env2 = SIW(layout)
                    env2.reset(t)
                    env2.execute(cand)
                    if _dsl_key(nxt_d) == _native_key(env2.state):
                        n_succ += 1
                    else:
                        mismatches.append(("succ", _native_key(st), cand,
                                           _dsl_key(nxt_d), _native_key(env2.state)))
            if interp.goal_satisfied() == envx.goal_satisfied():
                n_goal += 1
            else:
                mismatches.append(("goal", _native_key(st)))
            # étendre la fermeture (natif)
            for cand in envx.candidates():
                if cand["action"] == "STOP" or not envx._is_valid(cand)[0]:
                    continue
                env2 = SIW(layout)
                env2.reset(t)
                r = env2.execute(cand)
                if r.get("effect") in ("none", "stop"):
                    continue  # no-op: pas de nouvel état
                kk = env2.state.key()
                if kk not in seen:
                    seen[kk] = env2.state
                    q.append((kk, env2.state))
        # d* + optimaux (état initial de la tâche)
        interp.reset(task)
        from ucm.dsl.bfs import DSLBFS
        sol = DSLBFS(interp).solve(interp.state, max_states=max_states)
        env0 = SIW(layout)
        env0.reset(task)
        o = SIWOracle(layout, goal)
        d_n = o.d_star(env0.state) if o.reachable(env0.state) else None
        if sol["d_star"] == d_n:
            n_dst += 1
        else:
            mismatches.append(("d_star", sol["d_star"], d_n))
        opt_d = {json.dumps(interp.candidates()[i], sort_keys=True)
                 for i in sol["optimal_actions"]}
        if o.reachable(env0.state):
            opt_n = {json.dumps(env0.candidates()[i], sort_keys=True)
                     for i in o.optimal_actions(env0.state)}
        else:
            opt_n = set()
        if opt_d == opt_n:
            n_opt += 1
        else:
            mismatches.append(("optimal", sorted(opt_d), sorted(opt_n)))

    return {"n_states": n_states, "n_candidates_checked": n_cands,
            "valid_match": n_valid, "succ_match": n_succ,
            "goal_match": n_goal, "d_star_match": n_dst,
            "optimal_match": n_opt, "n_mismatches": len(mismatches),
            "mismatches_sample": mismatches[:10],
            "equal": not mismatches}
