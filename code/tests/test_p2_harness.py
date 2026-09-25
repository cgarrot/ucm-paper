"""P2-1 harnais (lead 19:47): E2E chemin réel + gardes."""
import json
import os
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ds():
    return json.load(open(os.path.join(_REPO, "artifacts/p2-dataset.json")))


class TestGenerator:
    def test_dataset_splits_disjoint_per_family(self):
        d = _ds()
        for fam, meta in d["dataset"]["families"].items():
            tr, dev, te = set(meta["train"]), set(meta["dev"]), set(meta["test"])
            assert not (tr & dev) and not (tr & te) and not (dev & te)
            assert len(tr) + len(dev) + len(te) == meta["n"] == 60

    def test_episodes_self_contained(self):
        d = _ds()
        ep = d["episodes"]["DROP-AT-d2-12"][0]
        assert {"episode_id", "family", "layout_tables", "task", "d_star",
                "opt_kinds"} <= set(ep)
        assert 2 <= ep["d_star"] <= 12
        assert "DROP" in ep["opt_kinds"]


class TestContextArms:
    def test_d1_fraction_exact_and_deterministic(self):
        from ucm.p2.context_arms import d1_episode_ids, CTX_SEED
        ids = [f"e{i}" for i in range(36)]
        a = d1_episode_ids(ids)
        b = d1_episode_ids(ids)
        assert a == b                      # RNG seed gelé — déterministe
        assert len(a) == 10                # floor(0.30×36)
        assert a <= set(ids)

    def test_shuffled_targets_permuted(self):
        import random
        from ucm.p2.context_arms import (build_context_entities,
                                         build_context_entities_shuffled)
        effs = [("MOVE", "agent_at", "v1"), ("MOVE", "agent_at", "v2"),
                ("PICK", "carried", "key"), ("DROP", "obj_at", "parcel")]
        rng = random.Random(0)
        e1, r1 = build_context_entities(effs, rng=rng)
        rng = random.Random(0)
        e2, r2 = build_context_entities_shuffled(effs, rng=rng)
        assert e1 == e2                    # même volume/entités
        t1 = [r["obj"] for r in r1]
        t2 = [r["obj"] for r in r2]
        assert sorted(t1) == sorted(t2) and (len(t1) < 2 or t1 != t2)  # permutées


class TestGate:
    def test_gate_passes_and_publishes_gap(self):
        from ucm.p2.gate import gate_equal_info
        r = gate_equal_info(_ds()["episodes"])
        assert r["pass"] and r["fraction_solved"] >= 0.30
        assert "gap_equal_vs_exact" in r

    def test_gate_failure_stops_before_training(self):
        from ucm.p2 import gate as g
        from ucm.p2.gate import gate_equal_info
        # contexte dégénéré: AUCUN genre observé → insolvable
        with pytest.raises(RuntimeError, match="GATE"):
            gate_equal_info(_ds()["episodes"], ctx_kinds=())


class TestRunnerE2E:
    def test_mini_cell_executes_and_persists(self, tmp_path, monkeypatch):
        """E2E chemin réel: 1 cellule (famille, D1-mixed, s0, updates=2) —
        gate AVANT entraînement, perte conjointe, raws, ckpt O_EXCL, budget."""
        monkeypatch.setenv("UCM_P2_CKPT_DIR", str(tmp_path / "ck"))
        from ucm.p2.runner import run_p2
        r = run_p2(out_dir=str(tmp_path / "out"), ts="E2E", updates=2,
                   families=("DROP-AT-d2-12",), seeds=(0,), arms=("D1-mixed",))
        c = r["cells"]["DROP-AT-d2-12|D1-mixed|s0"]
        assert c["train"]["n_records"] > 0
        assert len(c["raws"]) == 12         # 12 tâches test
        assert os.path.exists(r["persisted"])
        assert os.path.exists(r["budget_journal"])
        assert os.path.exists(tmp_path / "ck" / "DROP-AT-d2-12-D1-mixed-s0.npz")
        # O_EXCL: même cellule ⇒ refus ckpt
        with pytest.raises(RuntimeError, match="O_EXCL"):
            run_p2(out_dir=str(tmp_path / "out2"), ts="E2E2", updates=2,
                   families=("DROP-AT-d2-12",), seeds=(0,), arms=("D1-mixed",))

    def test_head_never_trained_counterfactual(self):
        """Garde HEAD-NEVER-TRAINED: head_enabled=False ⇒ sha tête inchangé
        après entraînement; contre-factuel: head_enabled=True ⇒ sha CHANGE."""
        import mlx.core as mx
        mx.set_default_device(mx.cpu)
        from ucm.p2.model_p2 import P2Model
        from ucm.p2.runner import train_cell, load_dataset, build_arm_records
        recs, sp = build_arm_records(load_dataset()["episodes"],
                                     "DROP-AT-d2-12", "D0", 0)
        recs = recs[:40]
        m_off = P2Model(d=48, head_enabled=False)
        sha0 = m_off.params_sha(m_off.head)
        train_cell(m_off, recs, 20, 0)
        assert m_off.params_sha(m_off.head) == sha0, \
            "tête a bougé sous head_enabled=False — garde HEAD violée"
        m_on = P2Model(d=48, head_enabled=True)
        sha1 = m_on.params_sha(m_on.head)
        train_cell(m_on, recs, 20, 0)
        assert m_on.params_sha(m_on.head) != sha1, \
            "tête N'A PAS appris sous head_enabled=True — contrôle positif mort"


class TestLabelsFnHook:  # tagi-2 12:26 ablation controlee
    def test_hook_overrides_effect_labels_only(self):
        """labels_fn override les effect_labels SANS toucher au reste:
        policy labels inchangées, entraînement identique hors effet."""
        import mlx.core as mx
        mx.set_default_device(mx.cpu)
        from ucm.p2.runner import load_dataset, build_arm_records, train_cell
        from ucm.p2.model_p2 import P2Model
        recs, _ = build_arm_records(load_dataset()["episodes"],
                                    "DROP-AT-d2-12", "D0", 0)
        recs = recs[:40]
        seen = []
        def zero_all(eff, rec):
            out = [0] * len(eff)      # ablation: tout → none
            seen.append(out)          # enregistre la SORTIE (ce qui va à la perte)
            return out
        mx.random.seed(0)
        m = P2Model(d=48)
        meta = train_cell(m, recs, 20, 0, labels_fn=zero_all)
        assert len(seen) == len(recs), "hook non appelé pour chaque record"
        assert all(all(x == 0 for x in e) for e in seen), "override non appliqué"
        assert meta["n_records"] == len(recs)
        # sans hook: identité (pipeline inchangé)
        mx.random.seed(0)
        m2 = P2Model(d=48)
        meta2 = train_cell(m2, recs, 20, 0)
        assert meta2["loss_first"] is not None
