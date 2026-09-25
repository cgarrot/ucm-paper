"""P2-1 (D) — Gate oracle-privilégié AVANT entraînement (gel 16275ad §3).

Oracle à information égale: BFS DSL restreint aux GENRES OBSERVÉS dans le
contexte D1 (kinds sans observation = inconnus, non planifiables) — le
budget est kind-default (plan optimal sur genres connus, horizon 2×d*).
Fraction de référence ≥ 30 % exigée; écart oracle-exact vs oracle-égal
publié. Échec ⇒ RuntimeError AVANT tout entraînement (jamais de run sous
un gate échoué).
"""
from __future__ import annotations

from ucm.dsl.core import DSLInterpreter
from ucm.dsl.tgk_program import TGK_DSL_PROGRAM
from ucm.dsl.bfs import DSLBFS

GATE_MIN_FRACTION = 0.30


def _solve_restricted(interp: DSLInterpreter, task, allowed_kinds, max_states=20_000):
    """BFS sur l'état de croyance: seuls les genres observés sont jouables.
    Retourne (solvable, plan_length) — kind-default = plan optimal restreint."""
    interp.reset(task)
    st = interp.state
    goal = task["goal"]
    steps = 0
    for _ in range(48):
        interp.state = st
        if interp.goal_satisfied():
            return True, steps
        # un pas: parmi les candidats des genres autorisés + STOP, choisir
        # par BFS restreint (greedy optimal sur le sous-espace)
        best = None
        # construire dist restreinte: fermeture sur genres autorisés
        dist = _restricted_dist(interp, st, allowed_kinds, max_states)
        if dist is None:
            return False, steps
        d0 = dist.get(st.key())
        if d0 is None:
            return False, steps
        if d0 == 0:
            return True, steps
        for cand in interp.candidates():
            if cand["action"] == "STOP":
                continue
            if cand["action"] not in allowed_kinds:
                continue
            valid, nxt = interp.successor(cand)
            if valid and nxt is not None and dist.get(nxt.key()) == d0 - 1:
                best = cand
                break
        if best is None:
            return False, steps
        _, st = interp.successor(best)
        steps += 1
    return False, steps


def _restricted_dist(interp, start, allowed_kinds, max_states):
    """BFS inverse restreint aux genres autorisés (distance-au-but)."""
    from collections import deque
    interp.state = start
    states = {start.key(): start}
    edges = {}
    q = deque([start.key()])
    while q:
        k = q.popleft()
        interp.state = states[k]
        edges[k] = []
        for cand in interp.candidates():
            if cand["action"] == "STOP" or cand["action"] not in allowed_kinds:
                continue
            valid, nxt = interp.successor(cand)
            if valid and nxt is not None:
                nk = nxt.key()
                edges[k].append(nk)
                if nk not in states:
                    states[nk] = nxt
                    q.append(nk)
                    if len(states) > max_states:
                        return None
    rev = {k: [] for k in states}
    for k, outs in edges.items():
        for nk in outs:
            rev[nk].append(k)
    goals = []
    for k, s in states.items():
        interp.state = s
        if interp.goal_satisfied():
            goals.append(k)
    gdist = {g: 0 for g in goals}
    q = deque(goals)
    while q:
        k = q.popleft()
        for p in rev[k]:
            if p not in gdist:
                gdist[p] = gdist[k] + 1
                q.append(p)
    interp.state = start   # PAS de mutation visible (discipline)
    return gdist


def gate_equal_info(dataset_episodes: dict, ctx_kinds=("MOVE", "PICK", "DROP")):
    """Gate sur les familles primaires. Retourne le rapport — raise si < 30%."""
    import json
    report = {"ctx_kinds": list(ctx_kinds), "min_fraction": GATE_MIN_FRACTION,
              "per_family": {}, "fraction_solved": None}
    total_n = total_ok = 0
    for fam, eps in dataset_episodes.items():
        interp_cache = {}
        ok = 0
        for ep in eps:
            interp = DSLInterpreter(TGK_DSL_PROGRAM, ep["layout_tables"])
            solved, plen = _solve_restricted(interp, ep["task"], set(ctx_kinds))
            ok += 1 if solved else 0
        frac = ok / len(eps) if eps else 0.0
        report["per_family"][fam] = {"n": len(eps), "solved": ok,
                                     "fraction": round(frac, 4)}
        total_n += len(eps)
        total_ok += ok
    report["fraction_solved"] = round(total_ok / max(total_n, 1), 4)
    report["pass"] = report["fraction_solved"] >= GATE_MIN_FRACTION
    # écart oracle-exact (BFS complet) publié
    exact = 0
    for fam, eps in dataset_episodes.items():
        for ep in eps:
            interp = DSLInterpreter(TGK_DSL_PROGRAM, ep["layout_tables"])
            interp.reset(ep["task"])
            sol = DSLBFS(interp).solve(interp.state, max_states=20_000)
            exact += 1 if sol["reachable"] else 0
    report["oracle_exact_fraction"] = round(exact / max(total_n, 1), 4)
    report["gap_equal_vs_exact"] = round(
        report["oracle_exact_fraction"] - report["fraction_solved"], 4)
    if not report["pass"]:
        raise RuntimeError(
            f"GATE oracle-à-information-égale ÉCHOUÉ: {report['fraction_solved']} "
            f"< {GATE_MIN_FRACTION} — arrêt AVANT entraînement "
            f"(redesign des familles requis, protocole §3)")
    return report
