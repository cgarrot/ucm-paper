"""DSL jalon 1 (lead 18:29): interpréteur générique + BFS + différentiel TGK."""
import json
import os
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _eps():
    from ucm.eval.gate5_report import load_canon_episodes
    return load_canon_episodes(
        os.path.join(_REPO, "artifacts/data/m0-transitions.jsonl"),
        split="test_g1", limit=200)


class TestInterpreterGeneric:
    def test_reset_only_construction(self):
        """B1: l'état n'est constructible que par reset — toute la surface
        publique part de reset(task); les successeurs dérivent de ces états."""
        from ucm.dsl.core import DSLInterpreter, DSLError
        from ucm.dsl.tgk_program import TGK_DSL_PROGRAM, layout_from_native
        from ucm.eval.gate5_report import rebuild_layout, task_from_record
        eps = _eps()
        lay = rebuild_layout(eps[0][3])
        interp = DSLInterpreter(TGK_DSL_PROGRAM, layout_from_native(lay))
        with pytest.raises((AttributeError, TypeError)):
            interp.goal_satisfied()   # pas d'état avant reset — échec franc
        interp.reset(task_from_record(eps[0][3]))
        assert interp.state is not None

    def test_schema_guard(self):
        from ucm.dsl.core import DSLInterpreter, DSLError
        with pytest.raises(DSLError):
            DSLInterpreter({"schema": "autre"}, {})

    def test_stop_semantics_9_4(self):
        """STOP: valide toujours, terminal, succès = but."""
        from ucm.dsl.core import DSLInterpreter
        from ucm.dsl.tgk_program import TGK_DSL_PROGRAM, layout_from_native
        from ucm.eval.gate5_report import rebuild_layout, task_from_record
        eps = _eps()
        lay = rebuild_layout(eps[0][3])
        t = task_from_record(eps[0][3])
        interp = DSLInterpreter(TGK_DSL_PROGRAM, layout_from_native(lay))
        interp.reset(t)
        r = interp.execute({"action": "STOP", "arg": None})
        assert r["terminal"] and r["valid"]
        assert r["result"] in ("success", "premature_stop")


class TestBFS:
    def test_d_star_and_optimal_vs_native(self):
        from ucm.dsl.core import DSLInterpreter
        from ucm.dsl.tgk_program import TGK_DSL_PROGRAM, layout_from_native
        from ucm.dsl.bfs import DSLBFS
        from ucm.env.tinygraph import TinyGraphKey
        from ucm.env.oracle import LayoutOracle
        from ucm.eval.gate5_report import rebuild_layout, task_from_record
        eps = _eps()
        for e in eps[:5]:
            lay = rebuild_layout(e[3])
            t = task_from_record(e[3])
            interp = DSLInterpreter(TGK_DSL_PROGRAM, layout_from_native(lay))
            interp.reset(t)
            sol = DSLBFS(interp).solve(interp.state)
            env = TinyGraphKey(lay)
            env.reset(t)
            o = LayoutOracle(lay, t["goal"])
            assert sol["d_star"] == o.d_star(env.state), f"d* mismatch {t}"
            opt_d = {json.dumps(interp.candidates()[i], sort_keys=True)
                     for i in sol["optimal_actions"]}
            opt_n = {json.dumps(env.candidates()[i], sort_keys=True)
                     for i in o.optimal_actions(env.state)}
            assert opt_d == opt_n, f"optimal mismatch {t}: {opt_d} vs {opt_n}"

    def test_forward_depth_is_not_goal_distance_regression(self):
        """Régression bug v1: un successeur proche du DÉPART peut être loin du
        BUT — la profondeur avant ne doit JAMAIS servir d'optimalité."""
        from ucm.dsl.core import DSLInterpreter
        from ucm.dsl.tgk_program import TGK_DSL_PROGRAM, layout_from_native
        from ucm.dsl.bfs import DSLBFS
        from ucm.eval.gate5_report import rebuild_layout, task_from_record
        eps = _eps()
        lay = rebuild_layout(eps[0][3])
        t = task_from_record(eps[0][3])
        interp = DSLInterpreter(TGK_DSL_PROGRAM, layout_from_native(lay))
        interp.reset(t)
        sol = DSLBFS(interp).solve(interp.state)
        # chaque action optimale mène à goal_dist == d*-1 (et pas avant-profondeur)
        for i in sol["optimal_actions"]:
            cand = interp.candidates()[i]
            if cand["action"] == "STOP":
                continue
            valid, nxt = interp.successor(cand)
            assert sol["goal_dist"][nxt.key()] == sol["d_star"] - 1


class TestDifferentialTGK:
    def test_differential_equal_two_layouts(self):
        from ucm.dsl.differential import differential_tgk
        from ucm.eval.gate5_report import rebuild_layout, task_from_record
        from collections import defaultdict
        eps = _eps()
        by = defaultdict(list)
        for e in eps:
            by[e[1]].append(e)
        for lh, group in list(by.items())[:2]:
            lay = rebuild_layout(group[0][3])
            tasks = [task_from_record(e[3]) for e in group[:2]]
            r = differential_tgk(lay, tasks, max_states=250)
            assert r["equal"], r["mismatches_sample"]
            assert r["n_states"] > 50
            assert r["valid_match"] == r["n_candidates_checked"]

    def test_mutation_injected_bug_detected(self, monkeypatch):
        """Test de mutation (lead 17:54-2): bug injecté dans l'interpréteur
        ⇒ le différentiel DÉTECTE la divergence."""
        from ucm.dsl import core as dsl_core
        from ucm.dsl.differential import differential_tgk
        from ucm.eval.gate5_report import rebuild_layout, task_from_record
        eps = _eps()
        lay = rebuild_layout(eps[0][3])
        t = task_from_record(eps[0][3])
        real_eval = dsl_core.DSLInterpreter._eval

        def mutated_eval(self, e):
            # MUTATION: toute négation ignorée (agent≠arg cassé → MOVE
            # vers sa propre pièce devient valide, etc.)
            if e[0] == "not":
                return True
            return real_eval(self, e)

        monkeypatch.setattr(dsl_core.DSLInterpreter, "_eval", mutated_eval)
        r = differential_tgk(lay, [t], max_states=150)
        assert not r["equal"], "mutation NON détectée — différentiel aveugle"
        assert r["n_mismatches"] > 0
