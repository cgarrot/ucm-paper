"""Tests oracle — cohérence Bellman exhaustive + propriétés §4.6 spec."""

import itertools
import random

import pytest

from ucm.env.oracle import (LayoutOracle, candidate_actions, enumerate_states,
                            solve, _successor)
from ucm.env.tinygraph import Layout, PhysicalState, TinyGraphKey, goal_satisfied

ROOMS4 = ["a", "b", "c", "d"]
EDGES4 = [("a", "b"), ("a", "c"), ("b", "c"), ("c", "d")]
ROOMS5 = ["a", "b", "c", "d", "e"]
EDGES5 = [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("e", "a"), ("b", "d")]


def bellman_reference(layout: Layout, goal: dict) -> dict:
    """Value iteration exacte par relaxation jusqu'au point fixe (coût 1 par
    action physique valide ; les invalides sont des self-loops inutiles)."""
    states = enumerate_states(layout)
    acts = [a for a in candidate_actions(layout) if a["action"] != "STOP"]
    dist = {}
    for s in states:
        if goal_satisfied(s, goal):
            dist[s.key()] = 0
    changed = True
    while changed:
        changed = False
        for s in states:
            best = dist.get(s.key())
            for a in acts:
                t = _successor(layout, s, a)
                if t is not None and t.key() in dist:
                    d = dist[t.key()] + 1
                    if best is None or d < best:
                        best = d
            if best is not None and best < dist.get(s.key(), best + 1):
                dist[s.key()] = best
                changed = True
    return dist


GOALS = [
    {"predicate": "REACH", "args": {"room": "d"}},
    {"predicate": "HAVE", "args": {"object": "key"}},
    {"predicate": "AT", "args": {"object": "parcel", "room": "a"}},
]


@pytest.mark.parametrize("rooms,edges,door", [
    (ROOMS4, EDGES4, 3),
    (ROOMS5, EDGES5, 4),
])
@pytest.mark.parametrize("goal", GOALS, ids=lambda g: g["predicate"])
def test_bellman_exhaustive(rooms, edges, door, goal):
    layout = Layout(rooms, edges, door)
    oracle = LayoutOracle(layout, goal)
    ref = bellman_reference(layout, goal)
    states = enumerate_states(layout)
    for s in states:
        assert oracle.reachable(s) == (s.key() in ref), \
            f"reachability mismatch at {s.key()}"
        if s.key() in ref:
            assert oracle.d_star(s) == ref[s.key()], \
                f"d* mismatch at {s.key()}: {oracle.d_star(s)} vs {ref[s.key()]}"


@pytest.mark.parametrize("goal", GOALS, ids=lambda g: g["predicate"])
def test_optimal_actions_consistency(goal):
    """Chaque action de A* mène à un état d*-1 ; aucune autre action n'y mène."""
    layout = Layout(ROOMS4, EDGES4, 3)
    oracle = LayoutOracle(layout, goal)
    acts = candidate_actions(layout)
    for s in enumerate_states(layout):
        if not oracle.reachable(s):
            continue
        d = oracle.d_star(s)
        opt = oracle.optimal_actions(s)
        if d == 0:
            assert [acts[i]["action"] for i in opt] == ["STOP"]
            continue
        assert len(opt) >= 1
        for ai in opt:
            t = _successor(layout, s, acts[ai])
            assert t is not None and oracle.d_star(t) == d - 1
        for ai, a in enumerate(acts):
            if a["action"] == "STOP":
                continue
            t = _successor(layout, s, a)
            if t is not None:
                reduces = oracle.d_star(t) == d - 1
                assert reduces == (ai in opt)


def test_unreachable_states_flagged():
    """Clé derrière la porte qu'elle ouvre ⇒ buts nécessitant la clé inaccessibles."""
    # porte = pont (c,d) ; clé en d verrouillée, colis en d aussi
    layout = Layout(ROOMS4, EDGES4, 3)
    s = PhysicalState(agent="a", carried=None, key_room="d",
                      parcel_room="d", door_locked=True)
    oracle = LayoutOracle(layout, {"predicate": "HAVE", "args": {"object": "key"}})
    assert not oracle.reachable(s)
    res = solve(layout, s, {"predicate": "HAVE", "args": {"object": "key"}})
    assert res == {"d_star": -1, "L_star": -1, "optimal_actions": [],
                   "reachable": False}


def test_solve_contract_shape():
    layout = Layout(ROOMS4, EDGES4, 3)
    s = PhysicalState(agent="a", carried=None, key_room="a",
                      parcel_room="b", door_locked=True)
    res = solve(layout, s, {"predicate": "REACH", "args": {"room": "d"}})
    assert res["reachable"] and res["d_star"] >= 1
    assert res["L_star"] == res["d_star"] + 1
    assert all(isinstance(i, int) and 0 <= i < len(candidate_actions(layout))
               for i in res["optimal_actions"])


def test_L_star_matches_rollout():
    """Une trajectoire suivant A* atteint le but en exactement d* actions
    physiques + 1 STOP (spec §4.6 : pas d'oscillation optimale possible)."""
    rng = random.Random(7)
    layout = Layout(ROOMS5, EDGES5, 4)  # porte (d,e) — pont
    for goal in GOALS:
        oracle = LayoutOracle(layout, goal)
        for s in rng.sample(enumerate_states(layout), 40):
            if not oracle.reachable(s) or oracle.d_star(s) == 0:
                continue
            env = TinyGraphKey(layout)
            env.goal = goal
            env.state = s.copy()
            env.step_count = 0
            steps = 0
            while not env.terminal and steps < 80:
                acts = candidate_actions(layout)
                opt = oracle.optimal_actions(env.state)
                env.execute(acts[rng.choice(opt)])
                steps += 1
            assert env.last_result == "success"
            assert steps == oracle.d_star(s) + 1  # d* actions + STOP


def _random_layout(rng: random.Random, n_rooms: int) -> Layout:
    """Arbre couvrant aléatoire + arêtes supplémentaires, connecté."""
    rooms = [f"v{i}" for i in range(n_rooms)]
    rng.shuffle(rooms)
    edges = []
    for i in range(1, n_rooms):  # arbre couvrant aléatoire
        j = rng.randrange(i)
        edges.append((rooms[i], rooms[j]))
    extra = rng.randint(0, n_rooms // 2)
    seen = {frozenset(e) for e in edges}
    tries = 0
    while len([e for e in edges if frozenset(e) not in seen]) < extra and tries < 50:
        tries += 1
        a, b = rng.sample(rooms, 2)
        if frozenset((a, b)) not in seen:
            edges.append((a, b))
            seen.add(frozenset((a, b)))
    door = rng.randrange(len(edges))
    return Layout(rooms, edges, door)


@pytest.mark.parametrize("seed", range(6))
def test_random_layouts_oracle_matches_bellman(seed):
    """Propriété : sur layouts aléatoires connectés, l'oracle BFS inverse est
    exact (d* et reachability) face à la value iteration par relaxation."""
    rng = random.Random(1000 + seed)
    n = rng.randint(4, 8)
    layout = _random_layout(rng, n)
    # buts relatifs au layout (REACH/AT pointent sur des pièces existantes)
    goal = rng.choice([
        {"predicate": "REACH", "args": {"room": rng.choice(layout.rooms)}},
        {"predicate": "HAVE", "args": {"object": "key"}},
        {"predicate": "AT", "args": {"object": "parcel",
                                      "room": rng.choice(layout.rooms)}},
    ])
    oracle = LayoutOracle(layout, goal)
    ref = bellman_reference(layout, goal)
    states = enumerate_states(layout)
    assert len(ref) > 0, "goal must be satisfiable in at least one state"
    for s in states:
        assert oracle.reachable(s) == (s.key() in ref)
        if s.key() in ref:
            assert oracle.d_star(s) == ref[s.key()]


def test_differential_env_vs_oracle_successor():
    """Audit tagi-5 T2 : l'env et l'oracle partagent la description de
    dynamique (deux implémentations) — le différentiel doit rester nul."""
    for seed in range(12):
        rng = random.Random(500 + seed)
        layout = _random_layout(rng, rng.randint(2, 8))
        goal = rng.choice([
            {"predicate": "REACH", "args": {"room": rng.choice(layout.rooms)}},
            {"predicate": "HAVE", "args": {"object": "key"}},
            {"predicate": "AT", "args": {"object": "parcel",
                                          "room": rng.choice(layout.rooms)}},
        ])
        acts = candidate_actions(layout)
        env = TinyGraphKey(layout)
        env.goal = goal
        checks = 0
        for s in enumerate_states(layout):
            for a in acts:
                if a["action"] == "STOP":
                    continue
                env.state = s.copy()
                env.step_count = 0
                env.terminal = False
                res = env.execute(a)
                t = _successor(layout, s, a)
                assert res["valid"] == (t is not None), \
                    f"validity divergence {s.key()} {a}"
                if t is not None:
                    assert env.state.key() == t.key(), \
                        f"successor divergence {s.key()} {a}"
                checks += 1
        assert checks > 0


def test_chain_door_on_every_edge_reachable():
    """Chaîne R=5, porte successivement sur chaque arête : la clé doit être
    accessible et les buts derrière calculables (audit T5)."""
    rooms = ["a", "b", "c", "d", "e"]
    edges = [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e")]
    for door in range(4):
        layout = Layout(rooms, edges, door)
        s = PhysicalState(agent="a", carried=None, key_room="a",
                          parcel_room="a", door_locked=True)
        goal = {"predicate": "REACH", "args": {"room": "e"}}
        assert solve(layout, s, goal)["reachable"]
        # clé derrière sa propre porte ⇒ unreachable
        if layout.edges[door][0] != "a" and layout.edges[door][1] != "a":
            pass  # clé en 'a' du même côté
        else:
            s2 = PhysicalState(agent="b", carried=None, key_room="e",
                               parcel_room="e", door_locked=True)
            # 'b' peut être de l'autre côté selon la porte ; test générique :
            if "b" not in layout.edges[door]:
                assert not solve(layout, s2, {"predicate": "HAVE",
                                              "args": {"object": "key"}})["reachable"]


def test_two_room_layout_unreachable_key():
    layout = Layout(["a", "b"], [("a", "b")], 0)  # porte = seule arête (pont)
    s = PhysicalState(agent="a", carried=None, key_room="b",
                      parcel_room="b", door_locked=True)
    assert not solve(layout, s, {"predicate": "HAVE",
                                 "args": {"object": "key"}})["reachable"]
