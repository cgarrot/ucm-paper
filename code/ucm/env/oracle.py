"""Oracle exact TinyGraphKey (spec §4.6).

BFS inverse multi-source depuis tous les états satisfaisant le but, sur le graphe
explicite des états physiques d'un layout. Fournit :

    d*(s,g)   distance physique restante (STOP exclu)
    L*(s,g)   longueur optimale d'épisode = d* + 1 (STOP inclus)
    A*(s,g)   actions optimales : {STOP} si d*=0, sinon actions réduisant d* de 1
    reachable indicate d'accessibilité du but depuis s

Côut de chaque décision = 1 (invalides incluses : self-loops, jamais optimales).
Dans ce MDP déterministe, toute action de A* fait strictement décroître d* ;
plusieurs optimums ne permettent pas de boucle optimale (spec §4.6).

Usage supervision/évaluation UNIQUEMENT — jamais une entrée de politique.
"""

from __future__ import annotations

from .tinygraph import Layout, PhysicalState, OBJ_ROOM, goal_satisfied

CARRY_NONE, CARRY_KEY, CARRY_PARCEL = None, "key", "parcel"


def enumerate_states(layout: Layout) -> list[PhysicalState]:
    """Tous les états physiques : 2·(R² + 2R) états."""
    states = []
    for locked in (True, False):
        for agent in layout.rooms:
            # rien porté : deux objets posés
            for kr in layout.rooms:
                for pr in layout.rooms:
                    states.append(PhysicalState(agent, CARRY_NONE, kr, pr, locked))
            # clé portée
            for pr in layout.rooms:
                states.append(PhysicalState(agent, CARRY_KEY, None, pr, locked))
            # colis porté
            for kr in layout.rooms:
                states.append(PhysicalState(agent, CARRY_PARCEL, kr, None, locked))
    return states


def _successor(layout: Layout, s: PhysicalState, action: dict):
    """Transition physique d'une action candidate. Retourne l'état suivant si
    l'action est valide, sinon None (self-loop d'action invalide)."""
    kind, arg = action["action"], action["arg"]
    if kind == "MOVE":
        if arg == s.agent:
            return None
        if not layout.adjacent(s.agent, arg):
            return None
        if (frozenset((s.agent, arg)) == frozenset(layout.door_rooms)
                and s.door_locked):
            return None
        return PhysicalState(arg, s.carried, s.key_room, s.parcel_room, s.door_locked)
    if kind == "PICK":
        if arg not in ("key", "parcel") or s.carried is not None:
            return None
        if getattr(s, OBJ_ROOM[arg]) != s.agent:
            return None
        st = s.copy()
        st.carried = arg
        setattr(st, OBJ_ROOM[arg], None)
        return st
    if kind == "DROP":
        if arg not in ("key", "parcel") or s.carried != arg:
            return None
        st = s.copy()
        setattr(st, OBJ_ROOM[arg], st.agent)
        st.carried = CARRY_NONE
        return st
    if kind == "UNLOCK":
        if arg != "door0" or not s.door_locked or s.agent not in layout.door_rooms:
            return None
        if s.carried != "key":
            return None
        st = s.copy()
        st.door_locked = False
        return st
    if kind == "STOP":
        return None  # terminal, hors graphe physique
    return None


def candidate_actions(layout: Layout) -> list[dict]:
    out = [{"action": "MOVE", "arg": r} for r in layout.rooms]
    out += [{"action": "PICK", "arg": "key"}, {"action": "PICK", "arg": "parcel"},
            {"action": "DROP", "arg": "key"}, {"action": "DROP", "arg": "parcel"},
            {"action": "UNLOCK", "arg": "door0"},
            {"action": "STOP", "arg": None}]
    return out


class LayoutOracle:
    """Oracle par (layout, but). BFS inverse avec cache des distances."""

    def __init__(self, layout: Layout, goal: dict):
        self.layout = layout
        self.goal = goal
        self.candidates = candidate_actions(layout)
        self._phys = [a for a in self.candidates if a["action"] != "STOP"]
        self._dist: dict[tuple, int] = {}
        self._backward_bfs()

    # -- BFS inverse multi-source -------------------------------------------- #

    def _backward_bfs(self):
        """Prédecesseurs : pour chaque état t et action physique valide a menant
        s → t, s est prédécesseur. Multi-source depuis tous les états-buts."""
        # index des états
        all_states = enumerate_states(self.layout)
        index = {st.key(): i for i, st in enumerate(all_states)}
        succs: list[list[tuple[int, int]]] = [[] for _ in all_states]
        for i, s in enumerate(all_states):
            for ai, act in enumerate(self._phys):
                t = _successor(self.layout, s, act)
                if t is not None:
                    succs[index[t.key()]].append((i, ai))  # prédécesseur i, action ai
        from collections import deque
        dq = deque()
        for i, s in enumerate(all_states):
            if goal_satisfied(s, self.goal):
                self._dist[s.key()] = 0
                dq.append(i)
        while dq:
            ti = dq.popleft()
            t = all_states[ti]
            dt = self._dist[t.key()]
            for si, ai in succs[ti]:
                s = all_states[si]
                if s.key() not in self._dist:
                    self._dist[s.key()] = dt + 1
                    dq.append(si)

    # -- API publique --------------------------------------------------------- #

    def d_star(self, state: PhysicalState) -> int:
        d = self._dist.get(state.key())
        if d is None:
            raise ValueError("unreachable goal state (call reachable first)")
        return d

    def reachable(self, state: PhysicalState) -> bool:
        return state.key() in self._dist

    def optimal_actions(self, state: PhysicalState) -> list[int]:
        """Indices dans candidate_actions. {STOP} si d*=0 ; sinon les actions
        physiques dont le successeur est à d*-1.

        Lève ValueError sur un état unreachable : préférer solve(), qui
        protège cet appel (API publique)."""
        d = self.d_star(state)
        if d == 0:
            stop_idx = len(self.candidates) - 1
            assert self.candidates[stop_idx]["action"] == "STOP"
            return [stop_idx]
        out = []
        for ai, act in enumerate(self._phys):
            t = _successor(self.layout, state, act)
            if t is not None and self._dist.get(t.key()) == d - 1:
                out.append(ai)
        return out

    def solve(self, state: PhysicalState) -> dict:
        if not self.reachable(state):
            return {"d_star": -1, "L_star": -1, "optimal_actions": [],
                    "reachable": False}
        d = self.d_star(state)
        return {"d_star": d, "L_star": d + 1,
                "optimal_actions": self.optimal_actions(state),
                "reachable": True}


def solve(layout: Layout, state: PhysicalState, goal: dict) -> dict:
    """API du contrat §3.2 (sans cache inter-appels)."""
    return LayoutOracle(layout, goal).solve(state)
