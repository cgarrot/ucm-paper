"""Tests environnement TinyGraphKey — contrat §4 spec."""

import pytest

from ucm.env.tinygraph import (GOAL_PREDICATES, Layout, LayoutError,
                               PhysicalState, TinyGraphKey, goal_satisfied,
                               task_goal)

# Layout de référence : 2 triangles partageant une arête-porte
#   a - b      c - d        porte = (b, c)
#   | /        | /
#   e          f  → non, garder simple : voir ci-dessous.

ROOMS = ["a", "b", "c", "d"]
EDGES = [("a", "b"), ("a", "c"), ("b", "c"), ("c", "d")]
# a,b,c triangle ; d attaché à c. Porte sur (c,d) → pont vers d.


def make_layout(door_edge: int = 3) -> Layout:
    return Layout(ROOMS, EDGES, door_edge)


def make_task(pred="REACH", **args):
    if pred == "REACH":
        args.setdefault("room", "a")  # validation stricte : args exacts requis
    return {"init": {"agent": "a", "key": "a", "parcel": "b",
                     "door_locked": True},
            "goal": task_goal(pred, **args)}


class TestLayout:
    def test_rejects_disconnected(self):
        with pytest.raises(LayoutError):
            Layout(["a", "b", "c"], [("a", "b")], 0)

    def test_rejects_duplicate_and_selfloop(self):
        with pytest.raises(LayoutError):
            Layout(["a", "b"], [("a", "b"), ("b", "a")], 0)
        with pytest.raises(LayoutError):
            Layout(["a", "b"], [("a", "a"), ("a", "b")], 1)

    def test_bridge_detection(self):
        lay = make_layout()
        assert lay.is_bridge(3)          # (c,d) pont
        assert not lay.is_bridge(0)      # (a,b) dans le triangle

    def test_layout_hash_stable_and_identifies_labeled_instance(self):
        h1 = make_layout().layout_hash()
        h2 = make_layout().layout_hash()
        assert h1 == h2
        # layout_hash identifie l'instance ÉTIQUETÉE (identité de provenance).
        # La détection d'isomorphisme relève de WS-B (préfiltre + check exact).
        lay2 = Layout(["x", "y", "z", "w"],
                      [("x", "y"), ("x", "z"), ("y", "z"), ("z", "w")], 3)
        assert lay2.layout_hash() != h1  # isomorphe mais étiqueté différemment


class TestCandidates:
    def test_k_equals_r_plus_6(self):
        env = TinyGraphKey(make_layout())
        env.reset(make_task("REACH", room="a"))
        cands = env.candidates()
        assert len(cands) == len(ROOMS) + 6

    def test_goal_blind(self):
        """Candidats identiques quel que soit le but (spec §5.3)."""
        env = TinyGraphKey(make_layout())
        env.reset(make_task("REACH", room="a"))
        c1 = env.candidates()
        env.reset(make_task("AT", object="parcel", room="d"))
        assert env.candidates() == c1

    def test_stop_present_with_null_arg(self):
        env = TinyGraphKey(make_layout())
        env.reset(make_task())
        assert env.candidates()[-1] == {"action": "STOP", "arg": None}


class TestDynamics:
    def test_move_through_locked_door_invalid_noop_costs_step(self):
        env = TinyGraphKey(make_layout())
        env.reset(make_task("REACH", room="d"))
        r = env.execute({"action": "MOVE", "arg": "c"})   # a→c, arête libre
        assert r["valid"]
        r = env.execute({"action": "MOVE", "arg": "d"})   # porte verrouillée
        assert r["result"] == "invalid" and not r["terminal"]
        assert env.state.agent == "c"
        assert env.step_count == 2  # l'invalide coûte un pas

    def test_move_to_own_room_invalid(self):
        env = TinyGraphKey(make_layout())
        env.reset(make_task("REACH", room="a"))
        r = env.execute({"action": "MOVE", "arg": "a"})
        assert not r["valid"]

    def test_pick_drop_roundtrip(self):
        env = TinyGraphKey(make_layout())
        env.reset(make_task("HAVE", object="key"))
        env.execute({"action": "PICK", "arg": "key"})  # clé en 'a' avec agent
        assert env.state.carried == "key" and env.state.key_room is None
        r = env.execute({"action": "PICK", "arg": "parcel"})  # main occupée
        assert not r["valid"]
        env.execute({"action": "MOVE", "arg": "b"})
        env.execute({"action": "DROP", "arg": "key"})
        assert env.state.key_room == "b" and env.state.carried is None

    def test_unlock_requires_key_at_endpoint_and_permanent(self):
        env = TinyGraphKey(make_layout())
        env.reset(make_task("REACH", room="d"))
        env.execute({"action": "MOVE", "arg": "b"})
        env.execute({"action": "MOVE", "arg": "c"})
        r = env.execute({"action": "UNLOCK", "arg": "door0"})  # pas la clé
        assert not r["valid"]
        env.execute({"action": "MOVE", "arg": "a"})
        env.execute({"action": "PICK", "arg": "key"})
        env.execute({"action": "MOVE", "arg": "b"})
        env.execute({"action": "MOVE", "arg": "c"})
        r = env.execute({"action": "UNLOCK", "arg": "door0"})
        assert r["valid"] and not env.state.door_locked
        r2 = env.execute({"action": "UNLOCK", "arg": "door0"})  # déjà ouvert
        assert not r2["valid"]
        r3 = env.execute({"action": "MOVE", "arg": "d"})
        assert r3["valid"]

    def test_carrying_object_moves_with_agent(self):
        env = TinyGraphKey(make_layout())
        env.reset(make_task("AT", object="key", room="b"))
        env.execute({"action": "PICK", "arg": "key"})
        env.execute({"action": "MOVE", "arg": "b"})
        env.execute({"action": "DROP", "arg": "key"})
        assert env.state.key_room == "b"
        assert env.goal_satisfied()

    def test_at_requires_placed_not_carried(self):
        env = TinyGraphKey(make_layout())
        env.reset({"init": {"agent": "b", "key": "a", "parcel": "a",
                            "door_locked": True, "carried": "key"},
                   "goal": task_goal("AT", object="key", room="b")})
        assert not env.goal_satisfied()  # porté en b ≠ posé en b

    def test_execute_rejects_action_outside_candidates(self):
        env = TinyGraphKey(make_layout())
        env.reset(make_task())
        with pytest.raises(ValueError):
            env.execute({"action": "MOVE", "arg": "zzz"})


class TestTermination:
    def test_stop_success_when_satisfied(self):
        env = TinyGraphKey(make_layout())
        env.reset(make_task("REACH", room="a"))  # déjà satisfait
        r = env.execute({"action": "STOP", "arg": None})
        assert r["terminal"] and r["result"] == "success"

    def test_stop_premature_fails(self):
        env = TinyGraphKey(make_layout())
        env.reset(make_task("REACH", room="d"))
        r = env.execute({"action": "STOP", "arg": None})
        assert r["terminal"] and r["result"] == "premature_stop"
        with pytest.raises(RuntimeError):
            env.execute({"action": "STOP", "arg": None})

    def test_timeout_at_horizon_without_stop(self):
        env = TinyGraphKey(make_layout(), horizon=4)
        env.reset(make_task("REACH", room="d"))
        for _ in range(4):
            r = env.execute({"action": "MOVE", "arg": "a"})  # invalide, no-op
        assert r["terminal"] and r["result"] == "timeout"


class TestInformationContract:
    """policy_input ne doit contenir AUCUN champ interdit (spec §5.1)."""

    FORBIDDEN_KEYS = {"reward", "progress", "distance", "d_star", "L_star",
                      "success", "terminal", "optimal_actions", "plan",
                      "next_state", "next_observation", "expert", "timestep",
                      "step", "step_count", "episode", "split", "source",
                      "generator", "oracle"}

    def test_observe_has_no_forbidden_field(self):
        env = TinyGraphKey(make_layout())
        env.reset(make_task("AT", object="parcel", room="d"))
        obs = env.observe()

        def walk(node):
            if isinstance(node, dict):
                for k, v in node.items():
                    assert k not in self.FORBIDDEN_KEYS, f"forbidden key {k}"
                    walk(v)
            elif isinstance(node, list):
                for v in node:
                    walk(v)

        walk(obs)
        assert obs["schema_version"] == "0.2"

    def test_observation_contains_required_information(self):
        env = TinyGraphKey(make_layout())
        env.reset(make_task("AT", object="parcel", room="d"))
        obs = env.observe()
        rels = {(r["subj"], r["pred"], r["obj"]) for r in obs["relations"]}
        # topologie complète (2 orientations par arête)
        for a, b in EDGES:
            assert (a, "adjacent", b) in rels and (b, "adjacent", a) in rels
        assert ("key", "unlocks", "door0") in rels
        assert ("agent", "at", "a") in rels
        assert ("parcel", "at", "b") in rels
        door = [e for e in obs["entities"] if e["type"] == "door"][0]
        assert door["attrs"]["locked"] is True

    def test_goal_change_does_not_alter_state_part(self):
        """Changer le but à état identique ne change pas candidats ni
        normalisation de l'état (spec §5.3)."""
        env = TinyGraphKey(make_layout())
        env.reset(make_task("REACH", room="a"))
        o1 = env.observe()
        env.goal = task_goal("HAVE", object="parcel")
        o2 = env.observe()
        assert o1["entities"] == o2["entities"]
        assert o1["relations"] == o2["relations"]
        assert o1["candidates"] == o2["candidates"]
        assert o1["goal"] != o2["goal"]


class TestPermutation:
    def test_permuted_ids_same_physics(self):
        """Renommage cohérent des IDs ⇒ mêmes prédictions physiques attendues."""
        perm = {"a": "z3", "b": "z1", "c": "z0", "d": "z2"}
        lay2 = Layout([perm[r] for r in ROOMS],
                      [(perm[a], perm[b]) for a, b in EDGES], 3)
        e1, e2 = TinyGraphKey(make_layout()), TinyGraphKey(lay2)
        t1 = make_task("REACH", room="d")
        t2 = {"init": {"agent": perm["a"], "key": perm["a"],
                       "parcel": perm["b"], "door_locked": True},
              "goal": task_goal("REACH", room=perm["d"])}
        e1.reset(t1), e2.reset(t2)
        for act in ({"action": "MOVE", "arg": "b"}, {"action": "MOVE", "arg": "c"}):
            r1 = e1.execute(act)
            r2 = e2.execute({"action": act["action"], "arg": perm[act["arg"]]})
            assert r1["valid"] == r2["valid"]
        assert e1.state.agent == "c"
        assert e2.state.agent == perm["c"]

    def test_relabelled_layout_same_physics_different_identity_hash(self):
        perm = {"a": "z3", "b": "z1", "c": "z0", "d": "z2"}
        lay2 = Layout([perm[r] for r in ROOMS],
                      [(perm[a], perm[b]) for a, b in EDGES], 3)
        # identité étiquetée différente, physique identique (couvert ci-dessus)
        assert lay2.layout_hash() != make_layout().layout_hash()
        assert lay2.degree(perm["a"]) == make_layout().degree("a")


class TestStopCostsOneStep:
    """Audit tagi-5 M1 : chaque décision, STOP inclus, coûte un pas (§4.4)."""

    def test_stop_increments_step_count(self):
        env = TinyGraphKey(make_layout())
        env.reset(make_task("REACH", room="a"))  # déjà satisfait
        r = env.execute({"action": "STOP", "arg": None})
        assert env.step_count == 1 and r["step"] == 1
        assert r["result"] == "success"

    def test_stop_at_horizon_boundary_is_not_timeout(self):
        env = TinyGraphKey(make_layout(), horizon=2)
        env.reset(make_task("REACH", room="c"))
        r1 = env.execute({"action": "MOVE", "arg": "c"})
        assert r1["valid"] and not r1["terminal"]
        r2 = env.execute({"action": "STOP", "arg": None})  # décision H=2
        assert r2["result"] == "success" and env.step_count == 2

    def test_optimal_rollout_length_equals_L_star(self):
        from ucm.env.oracle import LayoutOracle, candidate_actions
        layout = make_layout()
        goal = task_goal("AT", object="parcel", room="d")
        oracle = LayoutOracle(layout, goal)
        env = TinyGraphKey(layout)
        env.goal = goal
        env.state = PhysicalState(agent="a", carried=None, key_room="b",
                                  parcel_room="b", door_locked=True)
        env.step_count = 0
        env.terminal = False
        L = oracle.d_star(env.state) + 1
        while not env.terminal:
            cands = candidate_actions(layout)
            env.execute(cands[oracle.optimal_actions(env.state)[0]])
        assert env.last_result == "success"
        assert env.step_count == L  # d* actions physiques + 1 STOP payant


class TestGoalValidation:
    """Audit tagi-5 m2 : pas d'alias d'objet silencieux, refs validées."""

    def test_unknown_object_raises_at_evaluation(self):
        with pytest.raises(ValueError):
            goal_satisfied(PhysicalState("a", None, "a", "b", True),
                           {"predicate": "AT", "args": {"object": "banana",
                                                        "room": "b"}})

    def test_reset_rejects_bad_goal(self):
        env = TinyGraphKey(make_layout())
        with pytest.raises(Exception):
            env.reset({"init": {"agent": "a", "key": "a", "parcel": "b",
                                "door_locked": True},
                       "goal": {"predicate": "REACH", "args": {}}})
        with pytest.raises(Exception):
            env.reset({"init": {"agent": "a", "key": "a", "parcel": "b",
                                "door_locked": True},
                       "goal": {"predicate": "AT",
                                "args": {"object": "banana", "room": "a"}}})
        with pytest.raises(Exception):
            env.reset({"init": {"agent": "a", "key": "a", "parcel": "b",
                                "door_locked": True},
                       "goal": {"predicate": "REACH", "args": {"room": "zz"}}})


class TestBridgeBounds:
    def test_is_bridge_out_of_range_raises(self):
        with pytest.raises(LayoutError):
            make_layout().is_bridge(999)


class TestInputOrderInvariance:
    """Audit tagi-5 T3 : l'ordre de déclaration (rooms/edges/orientation
    porte) ne doit rien changer à la physique ni au hash canonique."""

    def test_shuffled_declaration_same_hash_and_physics(self):
        lay1 = make_layout()
        lay2 = Layout(["d", "a", "c", "b"],
                      [("c", "a"), ("d", "c"), ("b", "c"), ("b", "a")], 1)
        assert lay2.layout_hash() == lay1.layout_hash()
        for r in lay1.rooms:
            assert lay1.degree(r) == lay2.degree(r)
        assert lay1.is_bridge(3) == lay2.is_bridge(1)


class TestObservationPurity:
    """Audit tagi-5 T6/T7 : scan de TYPES (aucun int/float dans
    policy_input, bools autorisés), candidates state-blind, observation
    invariante après action invalide."""

    def test_policy_input_has_no_numeric_values(self):
        env = TinyGraphKey(make_layout())
        env.reset(make_task("AT", object="parcel", room="d"))

        def walk(n):
            if isinstance(n, dict):
                for k, v in n.items():
                    assert not isinstance(v, (int, float)) or isinstance(v, bool), \
                        f"numeric value in policy_input: {k}={v!r}"
                    walk(v)
            elif isinstance(n, list):
                for v in n:
                    walk(v)
        walk(env.observe())

    def test_candidates_state_blind(self):
        env = TinyGraphKey(make_layout())
        env.reset(make_task("REACH", room="a"))
        c0 = env.candidates()
        env.execute({"action": "MOVE", "arg": "b"})
        assert env.candidates() == c0  # ne dépend pas de l'état, seulement du layout

    def test_observation_identical_after_invalid_action(self):
        env = TinyGraphKey(make_layout())
        env.reset(make_task("REACH", room="d"))
        o1 = env.observe()
        env.execute({"action": "MOVE", "arg": "d"})  # invalide, no-op
        assert env.observe() == o1
