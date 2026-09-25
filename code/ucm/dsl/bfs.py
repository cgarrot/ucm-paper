"""Oracle BFS GÉNÉRIQUE sur le DSL — jalon 1 (v2: distances-au-buts).

Fermeture avant depuis un état (construit par reset) en ENREGISTRANT le
graphe de transitions, puis BFS INVERSE depuis les états-buts:
goal_dist[s] = longueur du plus court chemin s → but. d*(start) =
goal_dist[start]. Actions optimales en s: celles dont le successeur est à
goal_dist[s]-1 (la profondeur AVANT n'est PAS la distance-au-but — bug v1
corrigé: un successeur proche du départ peut être loin du but).
Le BFS ne connaît que l'interpréteur: aucun concept de monde. d*=0 avec but
vrai ⇒ optimal = {STOP} (sémantique §9.4).
"""
from __future__ import annotations

from collections import deque

from ucm.dsl.core import DSLInterpreter, DSLState


class DSLBFS:
    def __init__(self, interp: DSLInterpreter):
        self.interp = interp

    def explore(self, start: DSLState, max_states: int = 200_000):
        """Fermeture avant + graphe de transitions. Retourne
        (states, edges) avec edges[k] = [(succ_key, cand), …]."""
        states = {start.key(): start}
        edges: dict = {}
        q = deque([start.key()])
        while q:
            k = q.popleft()
            self.interp.state = states[k]
            edges[k] = []
            for cand in self.interp.candidates():
                if cand["action"] == "STOP":
                    continue
                valid, nxt = self.interp.successor(cand)
                if not valid or nxt is None:
                    continue
                nk = nxt.key()
                edges[k].append((nk, cand))
                if nk not in states:
                    states[nk] = nxt
                    q.append(nk)
                    if len(states) > max_states:
                        raise RuntimeError("fermeture trop grande (max_states)")
        return states, edges

    def _goal_dist(self, states, edges):
        """BFS inverse depuis TOUS les états-buts."""
        rev: dict = {k: [] for k in states}
        for k, outs in edges.items():
            for nk, _ in outs:
                rev[nk].append(k)
        goals = [k for k, s in states.items() if self._sat(s)]
        gdist = {g: 0 for g in goals}
        q = deque(goals)
        while q:
            k = q.popleft()
            for pred in rev[k]:
                if pred not in gdist:
                    gdist[pred] = gdist[k] + 1
                    q.append(pred)
        return gdist

    def _sat(self, state: DSLState) -> bool:
        self.interp.state = state
        return self.interp.goal_satisfied()

    def solve(self, start: DSLState, max_states: int = 200_000):
        """{reachable, d_star, optimal_actions, goal_dist, n_states}."""
        states, edges = self.explore(start, max_states)
        gdist = self._goal_dist(states, edges)
        if start.key() not in gdist:
            return {"reachable": False, "d_star": None,
                    "optimal_actions": [], "n_states": len(states)}
        d = gdist[start.key()]
        self.interp.state = start
        if d == 0:
            # d*=0: optimal = {STOP} — unique action du terminal (§9.4)
            stop_idx = [i for i, c in enumerate(self.interp.candidates())
                        if c["action"] == "STOP"]
            return {"reachable": True, "d_star": 0,
                    "optimal_actions": stop_idx,
                    "goal_dist": gdist, "n_states": len(states)}
        cands = self.interp.candidates()
        optimal = []
        for i, cand in enumerate(cands):
            if cand["action"] == "STOP":
                continue
            valid, nxt = self.interp.successor(cand)
            if valid and nxt is not None and gdist.get(nxt.key()) == d - 1:
                optimal.append(i)
        return {"reachable": True, "d_star": d,
                "optimal_actions": optimal,
                "goal_dist": gdist, "n_states": len(states)}
