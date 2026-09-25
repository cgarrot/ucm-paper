"""DSL jalons 2+3 (lead 18:43): SIW 2e instance + fermeture ≥10⁴ + mutation système."""
import json
import os
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _siw(n=2, seed=42):
    import random
    from ucm.env.siw import SIWLayout, sample_task
    from ucm.v1.data_adapter import _spec_of
    inv = json.load(open(os.path.join(_REPO, "artifacts/inventory-siw-dev-layouts.json")))
    rng = random.Random(seed)
    out = []
    for h, e in list(inv.items())[:n]:
        lay = SIWLayout(_spec_of(e))
        tasks, tries = [], 0
        while len(tasks) < 2 and tries < 200:
            tries += 1
            st, goal = sample_task(rng, lay)
            tasks.append({"init": {"view": st.view, "filled": sorted(st.filled),
                                   "chosen": dict(st.chosen),
                                   "dialog_open": st.dialog_open,
                                   "submitted": sorted(st.submitted)},
                          "goal": goal})
        out.append((h, lay, tasks))
    return out


class TestSIWSecondInstance:
    def test_differential_siw_equal(self):
        from ucm.dsl.differential_siw import differential_siw
        for h, lay, tasks in _siw(2):
            r = differential_siw(lay, tasks, max_states=200)
            assert r["equal"], r["mismatches_sample"]
            assert r["n_states"] > 50

    def test_siw_reset_only(self):
        from ucm.dsl.core import DSLInterpreter
        from ucm.dsl.siw_program import SIW_DSL_PROGRAM, layout_tables_from_siw
        _, lay, tasks = _siw(1)[0]
        interp = DSLInterpreter(SIW_DSL_PROGRAM, layout_tables_from_siw(lay))
        with pytest.raises((AttributeError, TypeError)):
            interp.goal_satisfied()
        interp.reset(tasks[0])
        assert interp.state is not None

    def test_siw_click_effect_groups_by_kind(self):
        """CLICK: submit → submitted+dialog ; confirm/dismiss → dialog seul.
        Vérifié via le différentiel (successeurs) — ce test cible le mécanisme:
        un layout AVEC dialog, tâche où submit est possible."""
        from ucm.dsl.differential_siw import differential_siw
        # les layouts DEV incluent des dialogs; le différentiel complet le
        # couvre — ici on vérifie juste qu'un layout avec dialog passe
        import random
        from ucm.env.siw import SIWLayout
        from ucm.v1.data_adapter import _spec_of
        inv = json.load(open(os.path.join(_REPO, "artifacts/inventory-siw-dev-layouts.json")))
        for h, e in inv.items():
            lay = SIWLayout(_spec_of(e))
            if any(w["type"] == "dialog" for w in lay.widgets.values()):
                r = differential_siw(lay, _siw(1)[0][2][:1], max_states=150)
                assert r["equal"], "layout avec dialog diverge"
                return
        pytest.skip("aucun layout avec dialog dans les 30 DEV")


class TestClosure10k:
    def test_evidence_artifact_reaches_10k_all_equal_3_mutations(self):
        p = os.path.join(_REPO, "artifacts/dsl-evidence",
                         "dsl-closure-evidence-20260924T185613.json")
        art = json.load(open(p))
        assert art["reached_10k"], f"seulement {art['total_states_differential']} états"
        assert art["all_equal"]
        assert art["aggregate"]["tgk"]["states"] + art["aggregate"]["siw"]["states"] \
            == art["total_states_differential"]
        # sélection documentée (rejouabilité)
        assert len(art["layout_selection"]["tgk"]) >= 10
        assert len(art["layout_selection"]["siw"]) >= 12
        assert all("layout_hash" in s for s in art["layout_selection"]["tgk"])
        # 3 mutations DISTINCTES toutes détectées
        muts = {m["mutation"]: m["detected"] for m in art["mutations"]}
        assert muts == {"M1_negations_ignored": True,
                        "M2_s_add_noop": True,
                        "M3_eq_inverted": True}

    def test_mutation_m2_needs_siw_world(self):
        """Le faux négatif M2-sur-TGK est DOCUMENTÉ: s_add n'existe que dans
        le programme SIW — muter contre le bon monde n'est pas optionnel."""
        from ucm.dsl.closure_evidence import run_mutation_suite
        muts = run_mutation_suite()
        d = {m["mutation"]: m for m in muts}
        assert d["M2_s_add_noop"]["world"] == "siw"
        assert d["M2_s_add_noop"]["detected"]


class TestP2Pilot:  # lead 18:43-4 p_ref et SD pour delta_min
    def test_pilot_artifact_complete(self):
        p = os.path.join(_REPO, "artifacts/p2-pilot", "p2-pilot-20260924T191604.json")
        art = json.load(open(p))
        # 4 familles tenues à l'écart (bucket d2-12), 3 seeds
        assert len(art["families_selection"]) == 4
        assert all(f["signature"][2] == "d2-12" for f in art["families_selection"])
        assert len(art["seeds"]) == 3
        # p_ref + SD par famille, per_seed complet
        for k, v in art["per_family"].items():
            assert 0.0 <= v["p_ref"] <= 1.0
            assert 0.0 <= v["sd_inter_seed"] <= 0.5
            assert len(v["per_seed"]) == 3
        # la signature documente la BORNE de variance pour Δ_min
        sds = [v["sd_inter_seed"] for v in art["per_family"].values()]
        assert max(sds) > 0.0  # familles avec variance réelle présentes

    def test_held_families_selection_skips_empty(self):
        """Le cartésien contient des familles VIDES (DROP+REACH: poser un
        objet ne mène jamais à une pièce) — la sélection les saute."""
        from ucm.eval.p2_pilot import held_families_d2_12
        sel = held_families_d2_12(4, min_eps=13)
        assert len(sel) == 4
        assert all(s[2] == "d2-12" for s in sel)
        assert ["DROP", "REACH", "d2-12"] not in sel  # vide, jamais sélectionnée
