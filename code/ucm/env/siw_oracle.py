"""Oracle exact SIW — backward BFS multi-source sur l'espace d'états
énumérable (spec SIW §7, structure identique à l'oracle V0).

Espace: views × (filled ⊆ fields) × (chosen: select→option) × dialog ×
(submitted ⊆ forms). Invalides = self-loops coût 1 exclus du graphe utile;
no-ops VALIDES (distracteurs, submit prématuré, confirm sans dialog) =
self-loops à coût 1, jamais optimaux. STOP terminal hors graphe.

Usage supervision/évaluation UNIQUEMENT — jamais une entrée de politique.
"""

from __future__ import annotations

from collections import deque

from .siw import SIWLayout, SIWState, goal_satisfied


def enumerate_states(layout: SIWLayout) -> list[SIWState]:
    views = layout.views
    fields = sorted(w["id"] for w in layout.widgets.values()
                    if w["type"] == "field")
    selects = sorted(w["id"] for w in layout.widgets.values()
                     if w["type"] == "select")
    options_of = {s: sorted(o["id"] for o in layout.widgets.values()
                            if o.get("select") == s) for s in selects}
    forms = sorted(w["id"] for w in layout.widgets.values()
                   if w["type"] == "form")

    from itertools import combinations, product
    states = []
    filled_possibilities = [frozenset(c)
                            for k in range(len(fields) + 1)
                            for c in combinations(fields, k)]
    chosen_possibilities = [dict(zip(selects, combo))
                            for combo in product(*[options_of[s] + [None]
                                                   for s in selects])]
    for view in views:
        for filled in filled_possibilities:
            for chosen in chosen_possibilities:
                chosen_real = {k: v for k, v in chosen.items() if v is not None}
                for dialog in (False, True):
                    for k in range(len(forms) + 1):
                        for sub in combinations(forms, k):
                            states.append(SIWState(
                                view, filled, dict(chosen_real),
                                dialog, frozenset(sub)))
    return states


def candidate_actions(layout: SIWLayout) -> list[dict]:
    out = [{"action": "NAVIGATE", "arg": v} for v in layout.views]
    out += [{"action": "CLICK", "arg": b["id"]} for b in
            layout.widgets.values() if b["type"] == "button"]
    out += [{"action": "TYPE", "arg": f["id"]} for f in
            layout.widgets.values() if f["type"] == "field"]
    out += [{"action": "SELECT", "arg": o["id"]} for o in
            layout.widgets.values() if o["type"] == "option"]
    out += [{"action": "STOP", "arg": None}]
    return out


def _visible(layout: SIWLayout, state: SIWState, widget_id: str) -> bool:
    w = layout.widgets[widget_id]
    if w["type"] == "option":
        return _visible(layout, state, w["select"])
    if w.get("in_dialog"):
        return state.dialog_open
    return w.get("view") == state.view


def _form_complete(layout: SIWLayout, state: SIWState, form_id: str) -> bool:
    for m in layout.widgets.values():
        if m.get("form") == form_id:
            if m["type"] == "field" and m["id"] not in state.filled:
                return False
            if m["type"] == "select" and m["id"] not in state.chosen:
                return False
    return True


def _successor(layout: SIWLayout, s: SIWState, action: dict):
    """Transition physique. Retourne l'état suivant si l'action est valide,
    sinon None (self-loop invalide). No-op VALIDE → retourne s (self-loop,
    jamais optimal). Implémentation indépendante de SIW._is_valid (le
    différentiel env↔oracle est un test requis, leçon T2 V0)."""
    kind, arg = action["action"], action["arg"]
    if kind == "NAVIGATE":
        if arg == s.view or not layout.adjacent(s.view, arg):
            return None
        t = s.copy(); t.view = arg
        return t
    if kind == "CLICK":
        w = layout.widgets.get(arg)
        if w is None or w["type"] != "button" or not _visible(layout, s, arg):
            return None
        k = w.get("kind", "none")
        if k == "submit":
            fid = w.get("submit_for")
            if _form_complete(layout, s, fid) and fid not in s.submitted:
                t = s.copy()
                t.submitted = t.submitted | {fid}
                t.dialog_open = False
                return t
            return s  # no-op valide (submit prématuré)
        if k in ("confirm", "dismiss"):
            if s.dialog_open:
                t = s.copy(); t.dialog_open = False
                return t
            return s  # no-op valide
        return s  # distracteur: no-op valide
    if kind == "TYPE":
        w = layout.widgets.get(arg)
        if (w is None or w["type"] != "field" or arg in s.filled
                or not _visible(layout, s, arg)):
            return None
        t = s.copy(); t.filled = t.filled | {arg}
        return t
    if kind == "SELECT":
        w = layout.widgets.get(arg)
        if (w is None or w["type"] != "option"
                or not _visible(layout, s, arg)
                or w.get("select") in s.chosen):
            return None
        t = s.copy(); t.chosen = dict(t.chosen)
        t.chosen[w["select"]] = arg
        return t
    return None  # STOP: terminal, hors graphe


# --------------------------------------------------------------------------- #
# Graphe d'adjacence par layout, calculé UNE fois (cache) — les successeurs
# ne sont plus recalculés par but (correctif performance: 256k états × ~50
# actions × N buts = explosion).
# --------------------------------------------------------------------------- #

from functools import lru_cache


@lru_cache(maxsize=64)
def _layout_graph(layout_hash: str, views: tuple, widgets_frozen: tuple):
    """Construit (états, index_clé→i, successeurs[i] = [(action_idx, j)],
    prédécesseurs[i] = [j]). widgets_frozen = triplat figé des widgets."""
    layout = SIWLayout({"views": list(views),
                         "nav_edges": _nav_from_widgets(widgets_frozen),
                         "widgets": [dict(w) for w in widgets_frozen]})
    all_states = enumerate_states(layout)
    index = {st.key(): i for i, st in enumerate(all_states)}
    phys = [a for a in candidate_actions(layout) if a["action"] != "STOP"]
    succ: list[list[tuple[int, int]]] = [[] for _ in all_states]
    preds: list[list[int]] = [[] for _ in all_states]
    for i, s in enumerate(all_states):
        for ai, a in enumerate(phys):
            t = _successor(layout, s, a)
            if t is not None and t.key() != s.key():
                j = index[t.key()]
                succ[i].append((ai, j))
                preds[j].append(i)
    return layout, all_states, index, phys, succ, preds


@lru_cache(maxsize=64)
def _layout_graph_cached_impl(layout_hash: str, views: tuple,
                              nav_frozen: tuple, widgets_frozen: tuple):
    layout = SIWLayout({"views": list(views),
                         "nav_edges": [list(e) for e in nav_frozen],
                         "widgets": [dict(w) for w in widgets_frozen]})
    all_states = enumerate_states(layout)
    index = {st.key(): i for i, st in enumerate(all_states)}
    phys = [a for a in candidate_actions(layout) if a["action"] != "STOP"]
    succ: list[list[tuple[int, int]]] = [[] for _ in all_states]
    preds: list[list[int]] = [[] for _ in all_states]
    for i, s in enumerate(all_states):
        for ai, a in enumerate(phys):
            t = _successor(layout, s, a)
            if t is not None and t.key() != s.key():
                j = index[t.key()]
                succ[i].append((ai, j))
                preds[j].append(i)
    return layout, all_states, index, phys, succ, preds


def _layout_graph_cached(layout: SIWLayout):
    # ORDRE D'INSERTION préservé: les indices de candidats du graphe caché
    # doivent coïncider avec candidate_actions(layout) de l'APPELANT.
    widgets_frozen = tuple(
        tuple(sorted((k, v) for k, v in w.items()
                     if k not in ("options",)))
        for w in layout.widgets.values())
    return _layout_graph_cached_impl(
        layout.layout_hash(), tuple(layout.views),
        tuple(tuple(e) for e in layout.nav_edges), widgets_frozen)


class SIWOracle:
    """Oracle par (layout, but): BFS inverse multi-source sur le graphe de
    layout (construit une fois, mis en cache par layout_hash)."""

    def __init__(self, layout: SIWLayout, goal: dict):
        self.layout = layout
        self.goal = goal
        self.candidates = candidate_actions(layout)
        self._layout2, self._states, self._index, self._phys, self._succ, \
            self._preds = _layout_graph_cached(layout)
        # COHÉRENCE: les indices d'actions viennent du layout reconstruit
        # (ordre stable trié), pas de l'original — sinon désalignement.
        self.candidates = candidate_actions(self._layout2)
        self._dist: dict = {}
        self._backward_bfs()

    def _backward_bfs(self):
        dq = deque()
        for i, s in enumerate(self._states):
            if goal_satisfied(s, self._layout2, self.goal):
                self._dist[s.key()] = 0
                dq.append(i)
        while dq:
            ti = dq.popleft()
            dt = self._dist[self._states[ti].key()]
            for si in self._preds[ti]:
                k = self._states[si].key()
                if k not in self._dist:
                    self._dist[k] = dt + 1
                    dq.append(si)

    def d_star(self, state: SIWState) -> int:
        d = self._dist.get(state.key())
        if d is None:
            raise ValueError("unreachable (call reachable first)")
        return d

    def reachable(self, state: SIWState) -> bool:
        return state.key() in self._dist

    def optimal_actions(self, state: SIWState) -> list[int]:
        d = self.d_star(state)
        if d == 0:
            return [len(self.candidates) - 1]  # STOP
        i = self._index[state.key()]
        out = []
        for ai, j in self._succ[i]:
            if self._dist.get(self._states[j].key()) == d - 1:
                out.append(ai)
        return out

    def solve(self, state: SIWState) -> dict:
        if not self.reachable(state):
            return {"d_star": -1, "L_star": -1, "optimal_actions": [],
                    "reachable": False}
        d = self.d_star(state)
        return {"d_star": d, "L_star": d + 1,
                "optimal_actions": self.optimal_actions(state),
                "reachable": True}


def solve(layout: SIWLayout, state: SIWState, goal: dict) -> dict:
    return SIWOracle(layout, goal).solve(state)
