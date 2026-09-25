"""Tests SIW — contrat §12.6 + leçons V0 (M1 STOP, T2 différentiel, T6 types)."""

import random

import pytest

from ucm.env.siw import (GOAL_PREDICATES, SIW, SIWLayout, SIWLayoutError,
                         SIWState, generate_siw_layout, goal_satisfied,
                         sample_task, task_goal)
from ucm.env.siw_oracle import (SIWOracle, _successor, candidate_actions,
                                enumerate_states, solve)


def _mk(rng_seed=1, views=3):
    rng = random.Random(rng_seed)
    lay = generate_siw_layout(rng, views)
    return lay


def _reset_random_goal(env_seed=5, views=3):
    rng = random.Random(env_seed)
    lay = generate_siw_layout(rng, views)
    env = SIW(lay)
    init, goal = sample_task(rng, lay)
    env.reset({"init": {"view": init.view, "dialog_open": init.dialog_open},
               "goal": goal})
    return lay, env, goal


class TestLayout:
    def test_k_bounds_respected(self):
        for seed in range(20):
            rng = random.Random(seed)
            lay = generate_siw_layout(rng, rng.randint(2, 5))
            K = (len(lay.views)
                 + sum(1 for w in lay.widgets.values() if w["type"] == "button")
                 + sum(1 for w in lay.widgets.values() if w["type"] == "field")
                 + sum(1 for w in lay.widgets.values() if w["type"] == "option")
                 + 1)
            assert 30 <= K <= 62

    def test_labels_arbitrary_no_role_link(self):
        """§2.3: les labels ne portent pas l'information de rôle — un submit
        peut porter n'importe quel mot du vocabulaire."""
        rng = random.Random(3)
        found = set()
        for _ in range(15):
            lay = generate_siw_layout(rng, rng.randint(2, 4))
            for w in lay.widgets.values():
                if w["type"] == "button" and w.get("kind") == "submit":
                    found.add(lay.labels.get(w["id"]))
        # aucune corrélation testable structurellement, mais au moins variété
        assert len(found) >= 5

    def test_layout_hash_ignores_labels(self):
        """Isomorphisme = widgets+FSM SANS labels (§2.3, clé pour WS-B)."""
        rng = random.Random(11)
        lay = generate_siw_layout(rng, 3)
        h1 = lay.layout_hash()
        # permuter tous les labels
        ids = list(lay.labels.keys())
        vals = [lay.labels[i] for i in ids]
        rng.shuffle(vals)
        lay.labels = dict(zip(ids, vals))
        assert lay.layout_hash() == h1

    def test_connected_required(self):
        with pytest.raises(SIWLayoutError):
            SIWLayout({"views": ["a", "b", "c"],
                       "nav_edges": [["a", "b"]],
                       "widgets": []})


class TestInformationContract:
    FORBIDDEN = {"reward", "progress", "d_star", "L_star", "success",
                 "terminal", "optimal", "plan", "next_", "timestep", "step",
                 "episode", "split", "source", "oracle"}

    def test_policy_input_no_forbidden_no_numerics(self):
        lay, env, goal = _reset_random_goal()
        obs = env.observe()

        def walk(n, path="policy_input"):
            if isinstance(n, dict):
                for k, v in n.items():
                    assert k not in self.FORBIDDEN, f"{k}"
                    walk(v, path)
            elif isinstance(n, list):
                for v in n:
                    walk(v, path)

        walk(obs)

        def walk_num(n):
            if isinstance(n, dict):
                for v in n.values():
                    walk_num(v)
            elif isinstance(n, list):
                for v in n:
                    walk_num(v)
            else:
                assert not isinstance(n, (int, float)) or isinstance(n, bool), \
                    f"numeric value {n!r} dans policy_input"

        walk_num(obs)

    def test_candidates_goal_and_state_blind(self):
        lay, env, goal = _reset_random_goal()
        c1 = env.candidates()
        env.execute({"action": "NAVIGATE", "arg": env.candidates()[0]["arg"]})
        assert env.candidates() == c1  # ne dépend que du layout


class TestDynamics:
    def test_invalid_vs_valid_noop_distinction(self):
        lay, env, goal = _reset_random_goal(9)
        # trouver un distracteur visible
        cur = env.state.view
        dist = next(w for w in lay.widgets.values()
                    if w["type"] == "button" and w.get("kind") == "none"
                    and w.get("view") == cur)
        r = env.execute({"action": "CLICK", "arg": dist["id"]})
        assert r["valid"] and r["result"] == "ok"  # no-op VALIDE
        # navigate non adjacente = invalide
        non_adj = next(v for v in lay.views if not lay.adjacent(cur, v))
        r2 = env.execute({"action": "NAVIGATE", "arg": non_adj})
        assert not r2["valid"] and r2["result"] == "invalid"

    def test_submit_requires_complete_form(self):
        rng = random.Random(21)
        lay = generate_siw_layout(rng, 2)
        env = SIW(lay)
        form = next(w for w in lay.widgets.values() if w["type"] == "form")
        goal = task_goal("SUBMITTED", form=form["id"])
        env.reset({"init": {"view": form["view"]}, "goal": goal})
        sub = next(w for w in lay.widgets.values()
                   if w.get("submit_for") == form["id"])
        r = env.execute({"action": "CLICK", "arg": sub["id"]})
        assert r["valid"] and r["result"] == "ok"  # no-op valide: incomplet
        assert not env.goal_satisfied()

    def test_stop_costs_one_step(self):
        lay, env, goal = _reset_random_goal(13)
        r = env.execute({"action": "STOP", "arg": None})
        assert env.step_count == 1 and r["step"] == 1

    def test_type_goal_blind_after_revision(self):
        """§5 révisé: TYPE valide sur tout field visible+vide (la contrainte
        'field du but' rendait SUBMITTED insolvable)."""
        rng = random.Random(33)
        lay = generate_siw_layout(rng, 2)
        env = SIW(lay)
        form = next(w for w in lay.widgets.values() if w["type"] == "form")
        env.reset({"init": {"view": form["view"]},
                   "goal": task_goal("SUBMITTED", form=form["id"])})
        field = next(w for w in lay.widgets.values()
                     if w["type"] == "field" and w.get("form") == form["id"])
        r = env.execute({"action": "TYPE", "arg": field["id"]})
        assert r["valid"], "TYPE doit être valide pour SUBMITTED"

    def test_label_permutation_same_physics(self):
        """Permuter les labels ne change ni candidats ni dynamique (§8b)."""
        rng = random.Random(77)
        lay = generate_siw_layout(rng, 2)
        env1 = SIW(lay)
        form = next(w for w in lay.widgets.values() if w["type"] == "form")
        goal = task_goal("SUBMITTED", form=form["id"])
        env1.reset({"init": {"view": lay.views[0]}, "goal": goal})
        ids = list(lay.labels.keys())
        vals = [lay.labels[i] for i in ids]
        rng.shuffle(vals)
        lay.labels = dict(zip(ids, vals))
        env2 = SIW(lay)
        env2.reset({"init": {"view": lay.views[0]}, "goal": goal})
        for c in env1.candidates():
            r1 = env1.execute(c)
            r2 = env2.execute(c)
            assert r1["valid"] == r2["valid"] and r1["result"] == r2["result"]


class TestOracle:
    def test_bellman_exhaustive_small(self):
        rng = random.Random(2)
        lay = generate_siw_layout(rng, 2, k_bounds=(30, 34))
        states = enumerate_states(lay)
        acts = [a for a in candidate_actions(lay) if a["action"] != "STOP"]
        goals = [
            {"predicate": "VIEW", "args": {"view": lay.views[0]}},
            {"predicate": "SET", "args": {
                "field": next(w["id"] for w in lay.widgets.values()
                              if w["type"] == "field")}},
            {"predicate": "SUBMITTED", "args": {
                "form": next(w["id"] for w in lay.widgets.values()
                             if w["type"] == "form")}},
        ]
        sel = next(w for w in lay.widgets.values() if w["type"] == "select")
        opt = next(o["id"] for o in lay.widgets.values()
                   if o.get("select") == sel["id"])
        goals.append({"predicate": "CHOOSE",
                      "args": {"select": sel["id"], "option": opt}})
        for g in goals:
            orc = SIWOracle(lay, g)
            dist = {st.key(): 0 for st in states
                    if goal_satisfied(st, lay, g)}
            changed = True
            while changed:
                changed = False
                for s in states:
                    best = dist.get(s.key())
                    for a in acts:
                        t = _successor(lay, s, a)
                        if t is not None and t.key() != s.key() \
                                and t.key() in dist:
                            d = dist[t.key()] + 1
                            if best is None or d < best:
                                best = d
                    if best is not None and best < dist.get(s.key(), best + 1):
                        dist[s.key()] = best
                        changed = True
            for s in states:
                assert orc.reachable(s) == (s.key() in dist)
                if s.key() in dist:
                    assert orc.d_star(s) == dist[s.key()]

    def test_differential_env_vs_oracle(self):
        """Leçon T2: deux implémentations, zéro divergence tolérée."""
        for seed in range(6):
            rng = random.Random(500 + seed)
            lay = generate_siw_layout(rng, rng.randint(2, 4))
            rng_g = random.Random(900 + seed)
            goal = None
            form = next(w for w in lay.widgets.values() if w["type"] == "form")
            goal = task_goal("SUBMITTED", form=form["id"])
            env = SIW(lay)
            env.goal = goal
            states = enumerate_states(lay)
            acts = candidate_actions(lay)
            checked = 0
            for s in states:
                for a in acts:
                    if a["action"] == "STOP":
                        continue
                    env.state = s.copy()
                    env.step_count = 0
                    env.terminal = False
                    res = env.execute(a)
                    t = _successor(lay, s, a)
                    assert res["valid"] == (t is not None), \
                        f"validité divergente {s.key()} {a}"
                    if t is not None:
                        assert env.state.key() == t.key(), \
                            f"successeur divergent {s.key()} {a}"
                    checked += 1
            assert checked > 1000

    def test_all_predicates_solvable(self):
        """Leçon §5-révisé: chaque prédicat doit avoir des états atteignables."""
        rng = random.Random(4242)
        for _ in range(5):
            lay = generate_siw_layout(rng, rng.randint(2, 4))
            states = enumerate_states(lay)
            empty = SIWState(lay.views[0], frozenset(), {}, False, frozenset())
            goals = []
            goals.append(task_goal("VIEW", view=rng.choice(lay.views)))
            goals.append(task_goal("SET", field=next(
                w["id"] for w in lay.widgets.values() if w["type"] == "field")))
            sel = next(w for w in lay.widgets.values() if w["type"] == "select")
            goals.append(task_goal("CHOOSE", select=sel["id"], option=next(
                o["id"] for o in lay.widgets.values()
                if o.get("select") == sel["id"])))
            goals.append(task_goal("SUBMITTED", form=next(
                w["id"] for w in lay.widgets.values() if w["type"] == "form")))
            for g in goals:
                sol = solve(lay, empty, g)
                assert sol["reachable"], f"{g['predicate']} insolvable!"
                assert sol["d_star"] >= 1

    def test_optimal_rollout_reaches_goal(self):
        rng = random.Random(31415)
        lay = generate_siw_layout(rng, 3)
        form = next(w for w in lay.widgets.values() if w["type"] == "form")
        goal = task_goal("SUBMITTED", form=form["id"])
        oracle = SIWOracle(lay, goal)
        env = SIW(lay)
        env.goal = goal
        env.state = SIWState(lay.views[0], frozenset(), {}, False, frozenset())
        env.step_count = 0
        env.terminal = False
        L = oracle.d_star(env.state) + 1
        while not env.terminal:
            cands = candidate_actions(lay)
            env.execute(cands[oracle.optimal_actions(env.state)[0]])
        assert env.last_result == "success"
        assert env.step_count == L


class TestAllPredicatesReset:
    """Régression tagi-5 (f1aae9f m2): le check de type cassait VIEW (KeyError)
    — les tests aléatoires passaient par chance de seed. Désormais: reset
    EXPLICITE avec chacun des 4 prédicats (leçon: toute contrainte
    goal-conditionnée vérifiée sur TOUS les prédicats)."""

    def _all_predicates(self, lay):
        sel = next(w for w in lay.widgets.values() if w["type"] == "select")
        opt = next(o["id"] for o in lay.widgets.values()
                   if o.get("select") == sel["id"])
        return [
            task_goal("VIEW", view=lay.views[0]),
            task_goal("SET", field=next(w["id"] for w in lay.widgets.values()
                                        if w["type"] == "field")),
            task_goal("CHOOSE", select=sel["id"], option=opt),
            task_goal("SUBMITTED", form=next(w["id"] for w in lay.widgets.values()
                                             if w["type"] == "form")),
        ]

    def test_reset_each_predicate_no_error(self):
        for seed in (1, 2, 3):
            lay = generate_siw_layout(random.Random(seed), 3)
            env = SIW(lay)
            for goal in self._all_predicates(lay):
                env.reset({"init": {"view": lay.views[0]}, "goal": goal})
                obs = env.observe()
                assert obs["goal"]["predicate"] == goal["predicate"]

    def test_type_check_rejects_wrong_refs(self):
        lay = generate_siw_layout(random.Random(1), 3)
        env = SIW(lay)
        btn = next(w["id"] for w in lay.widgets.values()
                   if w["type"] == "button")
        fld = next(w["id"] for w in lay.widgets.values()
                   if w["type"] == "field")
        with pytest.raises(SIWLayoutError):
            env.reset({"init": {"view": lay.views[0]},
                       "goal": {"predicate": "SET", "args": {"field": btn}}})
        with pytest.raises(SIWLayoutError):
            env.reset({"init": {"view": lay.views[0]},
                       "goal": {"predicate": "SUBMITTED",
                                "args": {"form": fld}}})
        with pytest.raises(SIWLayoutError):
            env.reset({"init": {"view": lay.views[0]},
                       "goal": {"predicate": "VIEW",
                                "args": {"view": "vue_fantome"}}})
