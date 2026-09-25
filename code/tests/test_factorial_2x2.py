"""Factorielle 2×2 S2b — E2E sur le chemin qui exécute (données DEV réelles,
updates=1). Aucune fixture parallèle: le test exécute run_factorial avec le
VRAI inventaire DEV + les VRAIS couples v08 (déterminisme), budget miniature."""
import json
import os
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _dev_inputs(n_lines=12):
    lines = [l for l in open(os.path.join(
        _REPO, "artifacts/test-determinism/v08.jsonl")) if l.strip()][:n_lines]
    store = json.load(open(os.path.join(
        _REPO, "artifacts/inventory-siw-dev-layouts.json")))
    return lines, store


class TestRecurrentBlock:
    def test_param_count_constant_in_T(self):
        import mlx.core as mx
        mx.set_default_device(mx.cpu)
        import mlx.nn as nn
        from ucm.model.recurrent_block import RecurrentRefine
        r4 = RecurrentRefine(64, 7, T=4)
        r32 = RecurrentRefine(64, 7, T=32)
        p = lambda m: sum(int(x.size) for _, x in nn.utils.tree_flatten(m.parameters()))
        assert p(r4) == p(r32)

    def test_T_zero_refused(self):
        from ucm.model.recurrent_block import make_siw_rec_model
        with pytest.raises(ValueError):
            make_siw_rec_model(T=0)

    def test_forward_shape_matches_base(self):
        import mlx.core as mx
        mx.set_default_device(mx.cpu)
        from ucm.model.recurrent_block import make_siw_rec_model
        from ucm.v1.tensorize_siw import tensorize_siw_obs, collate_siw
        m = make_siw_rec_model(d=48, T=2)
        lines, store = _dev_inputs(2)
        from ucm.v1.data_adapter import _couples_core
        recs = _couples_core(store, lines)
        from ucm.v1.tensorize_siw import tensorize_siw_obs
        ex = tensorize_siw_obs(recs[0]["policy_input"])
        ex["labels"] = None
        b = collate_siw([ex])
        out = m(b)
        assert out.shape[0] == 1 and out.shape[1] == len(recs[0]["policy_input"]["candidates"])


class TestRecoveryMix:
    def test_volume_controlled_total_identical(self):
        lines, store = _dev_inputs(12)
        from ucm.data.recovery_mix import build_recovery_mix
        mixed, stats = build_recovery_mix(lines, store, f_recovery=0.5, seed=3)
        assert len(mixed) == len(lines)
        assert stats["n_total"] == len(lines)
        assert stats["n_recovery"] + stats["n_oracle"] == len(lines)

    def test_fraction_zero_is_oracle_pure(self):
        lines, store = _dev_inputs(8)
        from ucm.data.recovery_mix import build_recovery_mix
        mixed, stats = build_recovery_mix(lines, store, f_recovery=0.0, seed=3)
        assert stats["n_recovery"] == 0
        assert mixed == [json.dumps(json.loads(l), sort_keys=True) for l in lines]


class TestFactorialE2E:
    def test_full_factorial_miniature_executes_and_persists(self, tmp_path):
        from ucm.eval.factorial_2x2 import run_factorial
        lines, store = _dev_inputs(8)
        out = str(tmp_path / "out")
        r = run_factorial(lines, store, updates=1, seed=5, n_eval=4,
                          T=2, out_dir=out, ts="E2E")
        assert set(r["cells"]) == {"b144-oracle", "b144-recovery",
                                   "rec-oracle", "rec-recovery"}
        # budgets ÉGAUX entre cellules (la récupération REMPLACE)
        ns = {c: cell["n_couples"] for c, cell in r["cells"].items()}
        assert len(set(ns.values())) == 1, f"budgets inégaux: {ns}"
        # persistance par cellule + summary, O_EXCL
        for c in ("b144", "rec"):
            for d in ("oracle", "recovery"):
                assert os.path.exists(os.path.join(out, f"cell-{c}-{d}-E2E.json"))
        assert os.path.exists(os.path.join(out, "factorial-summary-E2E.json"))
        with pytest.raises(OSError):
            os.open(os.path.join(out, "factorial-summary-E2E.json"),
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        # sondes présentes
        assert "receptive_field" in r["summary"]["probes"]
        assert "log_ratio_vs_b144_oracle" in r["summary"]["probes"]


class TestProbes:
    def test_spearman_known_values(self):
        from ucm.eval.probes_s2b import spearman
        assert spearman([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0
        assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0

    def test_receptive_field_probe_buckets(self):
        from ucm.eval.probes_s2b import receptive_field_probe
        raw = [{"success": True, "d_star": 1}, {"success": False, "d_star": 4},
               {"success": False, "d_star": 5}, {"success": True, "d_star": 2},
               {"success": True, "d_star": 2}, {"success": True, "d_star": 1}]
        p = receptive_field_probe(raw)
        assert p["fail_rate_by_d_star"]["1"] == 0.0
        assert p["fail_rate_by_d_star"]["2"] == 0.0
        assert p["fail_rate_by_d_star"]["4+"] == 1.0
        assert p["spearman_fail_vs_dstar"] > 0.5  # échecs concentrent loin

    def test_log_ratio_robust_at_ceiling(self):
        from ucm.eval.probes_s2b import log_ratio_failures
        near_ceiling_a = [{"success": True}] * 99 + [{"success": False}]  # 1% fail
        perfect_b = [{"success": True}] * 100                              # 0% fail
        lr = log_ratio_failures(near_ceiling_a, perfect_b)
        assert lr["log_ratio"] > 0 and lr["reading"] == "A échoue davantage"
        # symétrie
        lr2 = log_ratio_failures(perfect_b, near_ceiling_a)
        assert abs(lr2["log_ratio"] + lr["log_ratio"]) < 1e-9


class TestCIVNormative:  # tagi-5 12:36 normatif
    def _mini_records(self, n=6):
        lines, store = _dev_inputs(n)
        from ucm.v1.data_adapter import _couples_core
        return _couples_core(store, lines)

    def test_base_frozen_sha_unchanged_after_training(self):
        import mlx.core as mx
        mx.set_default_device(mx.cpu)
        from ucm.model.recurrent_block import make_siw_rec_model
        from ucm.v1.runner import finetune
        from ucm.v1.transfer import CoverageTracker, FinetuneConfig
        m = make_siw_rec_model(d=48, T=2)
        sha0 = m.base_params_sha256()
        finetune(m, self._mini_records(), FinetuneConfig(updates=2, seed=1),
                 CoverageTracker(), "rec-test", 6, trainable="refine")
        assert m.base_params_sha256() == sha0, "base drifted under freeze"

    def test_counterfactual_unfreeze_DETECTED_by_compare(self):
        """Contre-factuel: SANS le gel (trainable=all), le sha base CHANGE —
        la vérification par sha DÉTECTE un unfreeze (le test prouve la
        détection, pas seulement le bon comportement)."""
        import mlx.core as mx
        mx.set_default_device(mx.cpu)
        from ucm.model.recurrent_block import make_siw_rec_model
        from ucm.v1.runner import finetune
        from ucm.v1.transfer import CoverageTracker, FinetuneConfig
        m = make_siw_rec_model(d=48, T=2)
        sha0 = m.base_params_sha256()
        finetune(m, self._mini_records(), FinetuneConfig(updates=5, seed=1),
                 CoverageTracker(), "unfrozen", 6, trainable="all")
        assert m.base_params_sha256() != sha0, \
            "unfreeze NOT detected — sha compare is vacuous"

    def test_T_effective_counter_mechanical(self):
        import mlx.core as mx
        mx.set_default_device(mx.cpu)
        from ucm.model.recurrent_block import make_siw_rec_model
        from ucm.v1.tensorize_siw import tensorize_siw_obs, collate_siw
        m = make_siw_rec_model(d=48, T=7)
        recs = self._mini_records(2)
        ex = tensorize_siw_obs(recs[0]["policy_input"]); ex["labels"] = None
        m(collate_siw([ex]))
        assert m.refine.n_iters == 7, "counter != T"

    def test_T_effective_counterfactual_shortcircuit_FAILS(self):
        """Contre-factuel: boucle court-circuitée (1 itération réelle, T=7
        déclaré) ⇒ la vérification du harnais (T_effective != T) ÉCHOUE —
        preuve mécanique que le compteur détecte un court-circuit."""
        import mlx.core as mx
        mx.set_default_device(mx.cpu)
        from ucm.model.recurrent_block import make_siw_rec_model
        from ucm.v1.tensorize_siw import tensorize_siw_obs, collate_siw
        m = make_siw_rec_model(d=48, T=7)
        recs = self._mini_records(2)
        ex = tensorize_siw_obs(recs[0]["policy_input"]); ex["labels"] = None
        b = collate_siw([ex])

        # court-circuit MONKEYPATCHÉ: une seule itération quoi qu'il arrive
        real_call = type(m.refine).__call__
        def shortcircuited(self, h, batch):
            self.n_iters = 0
            hh = self.ln(self.gru(h, self.mp(h, batch)))
            self.n_iters = 1
            return hh
        type(m.refine).__call__ = shortcircuited
        try:
            m(b)
            T_declared = 7
            # la vérification du harnais: T_effective != T ⇒ RuntimeError
            detected = (m.refine.n_iters != T_declared)
            assert detected, "shortcircuit undetected — counter vacuous"
            # et l'inégalité lève bien (forme exacte du harnais)
            import pytest as _pt
            with _pt.raises(RuntimeError, match="T effectif"):
                if m.refine.n_iters != T_declared:
                    raise RuntimeError(f"T effectif {m.refine.n_iters} != T déclaré {T_declared}")
        finally:
            type(m.refine).__call__ = real_call

    def test_loop_sequential_wall_grows_with_T(self):
        """Boucle RÉELLEMENT séquentielle: le coût par forward croît avec T
        (vectorisation T aurait un coût quasi-constant)."""
        import time
        import mlx.core as mx
        mx.set_default_device(mx.cpu)
        from ucm.model.recurrent_block import make_siw_rec_model
        from ucm.v1.tensorize_siw import tensorize_siw_obs, collate_siw
        recs = self._mini_records(2)
        ex = tensorize_siw_obs(recs[0]["policy_input"]); ex["labels"] = None
        b = collate_siw([ex])
        m2 = make_siw_rec_model(d=48, T=2)
        m16 = make_siw_rec_model(d=48, T=16)
        m2(b); m16(b)  # warm-up (compile/alloc)
        t0 = time.perf_counter(); [m2(b) for _ in range(3)]; mx.eval(m2.parameters())
        t2 = (time.perf_counter() - t0) / 3
        t0 = time.perf_counter(); [m16(b) for _ in range(3)]; mx.eval(m16.parameters())
        t16 = (time.perf_counter() - t0) / 3
        assert t16 > t2, f"wall not growing with T ({t16:.4f}s vs {t2:.4f}s) — vectorized?"

    def test_budgets_paired_civ_vs_b144(self):
        """Budgets APPARIÉS: mêmes records, mêmes updates, éval identique —
        l'artefact prouve l'appariement par champ (n_couples, updates) +
        gel vérifié + T prouvé sur le bras rec."""
        from ucm.eval.factorial_2x2 import run_cell, build_eval_episodes
        lines, store = _dev_inputs(6)
        eval_eps = build_eval_episodes(store, 2, seed=3)
        rb = run_cell("b144", "oracle", lines, store, 1, 3, eval_eps)
        rc = run_cell("rec", "oracle", lines, store, 1, 3, eval_eps, T=2)
        assert rb["n_couples"] == rc["n_couples"]
        assert rb["train"]["updates"] == rc["train"]["updates"]
        assert rb["n_eval"] == rc["n_eval"]
        assert rc["T_effective_proven"] == 2
        assert rc["base_frozen_verified"] is True


class TestHarnessGuards:  # lead 12:46 contre-factuels committés
    """Les deux gardes du harnais, testés PAR pytest.raises sur le
    discriminant EXACT (discipline ci_high), contre-factuels monkeypatchés
    au niveau du HARNAIS (run_cell), pas seulement en unitaire."""

    def _eps(self, store):
        from ucm.eval.factorial_2x2 import build_eval_episodes
        return build_eval_episodes(store, 2, seed=3)

    def test_guard_base_drift_counterfactual(self, monkeypatch):
        """Monkeypatch trainable='all' (gel cassé) ⇒ run_cell lève
        RuntimeError 'CIV base DRIFTED'."""
        import ucm.v1.runner as _runner
        real = _runner.finetune
        def unfrozen(*a, **kw):
            kw["trainable"] = "all"       # GEL CASSÉ (contre-factuel)
            return real(*a, **kw)
        monkeypatch.setattr(_runner, "finetune", unfrozen)
        from ucm.eval.factorial_2x2 import run_cell
        lines, store = _dev_inputs(6)
        with pytest.raises(RuntimeError, match="CIV base DRIFTED"):
            run_cell("rec", "oracle", lines, store, 3, 3, self._eps(store), T=2)

    def test_guard_T_effective_counterfactual(self, monkeypatch):
        """Monkeypatch court-circuit __call__ du refine (1 itération réelle,
        T=4 déclaré) ⇒ run_cell lève RuntimeError 'T_effectif'."""
        from ucm.model.recurrent_block import RecurrentRefine
        real_call = RecurrentRefine.__call__
        def shortcircuited(self, h, batch):
            self.n_iters = 0
            hh = self.ln(self.gru(h, self.mp(h, batch)))
            self.n_iters = 1            # COURT-CIRCUIT (contre-factuel)
            return hh
        monkeypatch.setattr(RecurrentRefine, "__call__", shortcircuited)
        from ucm.eval.factorial_2x2 import run_cell
        lines, store = _dev_inputs(6)
        with pytest.raises(RuntimeError, match="T_effectif"):
            run_cell("rec", "oracle", lines, store, 1, 3, self._eps(store), T=4)


class TestFactorialTGK:  # lead 13:13 le verdict S2b porte sur TGK G4
    """E2E TGK sur le chemin qui exécute: VRAIS records m0-transitions train,
    VRAIE strate test_g4 d*=13-24, 4 cellules, budgets appariés, gardes actifs."""

    def _tgk_records(self, n=8):
        from ucm.eval.factorial_2x2 import load_tgk_train_records
        return load_tgk_train_records(n)

    def test_tgk_strate_g4_exists_13_24(self):
        from ucm.eval.factorial_2x2 import load_tgk_eval_episodes
        eps = load_tgk_eval_episodes(d_band=(13, 24))
        assert len(eps) >= 50, f"strate G4 trop petite: {len(eps)}"
        assert all(13 <= e[2] <= 24 for e in eps)

    def test_tgk_recovery_volume_controlled(self):
        from ucm.data.recovery_mix import build_recovery_mix_tgk
        recs = self._tgk_records(10)
        mixed, stats = build_recovery_mix_tgk(recs, f_recovery=0.5, seed=3)
        assert len(mixed) == len(recs), "volume total doit rester identique"
        assert stats["n_total"] == len(recs)

    def test_tgk_full_factorial_miniature(self, tmp_path, monkeypatch):
        from ucm.eval.factorial_2x2 import run_factorial
        recs = self._tgk_records(8)
        out = str(tmp_path / "out")
        monkeypatch.setenv("UCM_S2B_CKPT_DIR", str(tmp_path / "ck"))
        r = run_factorial(recs, None, updates=1, seed=5, n_eval=3,
                          T=2, out_dir=out, ts="TGKE2E", world="tgk")
        assert set(r["cells"]) == {"b144-oracle", "b144-recovery",
                                   "rec-oracle", "rec-recovery"}
        # budgets appariés (même monde)
        ns = {c: cell["n_couples"] for c, cell in r["cells"].items()}
        assert len(set(ns.values())) == 1, f"budgets inégaux: {ns}"
        # gardes CIV actifs sur TGK: gel + T prouvé dans l'artefact
        assert r["cells"]["rec-oracle"]["base_frozen_verified"] is True
        assert r["cells"]["rec-oracle"]["T_effective_proven"] == 2
        assert r["cells"]["rec-oracle"]["base_frozen_sha256"] is not None
        # CANON (lead 14:05): les DEUX bras partent du canon, sha publié
        for c in r["cells"].values():
            assert c["canon_sha256"] is not None
            assert c["base_is_canon_verified"] is True
        canon_sha = r["cells"]["b144-oracle"]["canon_sha256"]
        assert r["cells"]["rec-oracle"]["canon_sha256"] == canon_sha
        # la base rec est restée == canon (gel + canon)
        assert r["cells"]["rec-oracle"]["base_frozen_sha256"] == canon_sha
        assert r["summary"]["canon_sha256"] == canon_sha
        # sondes présentes sur le monde TGK
        assert "receptive_field" in r["summary"]["probes"]
        assert r["summary"]["world"] == "tgk"
        assert r["summary"]["d_band"] == (13, 24)
        # persistance
        assert os.path.exists(os.path.join(out, "factorial-summary-TGKE2E.json"))

    def test_canon_not_loaded_counterfactual(self, tmp_path):
        """Contre-factuel (lead 14:05-2): canon NON chargé (modèle frais
        substitué) ⇒ la garde CANON NOT LOADED ÉCHOUE le run."""
        import ucm.eval.factorial_2x2 as f2
        real = f2.load_canon_gnnb
        def fake_loader(path, sha):
            from ucm.model.gnn_b import GNNB
            from ucm.model.recurrent_block import SIWRecModel
            import mlx.core as _mx
            _mx.eval(GNNB(d=144).parameters())
            fresh = GNNB(d=144)      # PAS le canon — contre-factuel
            return fresh, "0" * 64   # sha incohérent avec la base fraîche
        f2.load_canon_gnnb = fake_loader
        try:
            from ucm.eval.factorial_2x2 import load_tgk_eval_episodes
            recs = self._tgk_records(4)
            eps = load_tgk_eval_episodes(n=2)
            with pytest.raises(RuntimeError, match="CANON NOT LOADED"):
                f2.run_cell("b144", "oracle", recs, None, 1, 3, eps,
                            T=2, world="tgk",
                            canon_entry={"path": "x", "sha256": "0" * 64})
        finally:
            f2.load_canon_gnnb = real


class TestResumeCheckpointsParallel:  # lead 13:31 3 bloquants avant T0
    def test_resume_interrupted_run_completes_missing_only(self, tmp_path, monkeypatch):
        """Run interrompu simulé: 2/4 cellules artefacts existantes → relance
        complète les 2 manquantes SANS retoucher les existantes (contenu
        bit-identique), budgets appariés préservés."""
        from ucm.eval.factorial_2x2 import run_factorial
        lines, store = _dev_inputs(6)
        out = str(tmp_path / "out")
        # run complet de référence
        r1 = run_factorial(lines, store, updates=1, seed=5, n_eval=2,
                           T=2, out_dir=out, ts="R1")
        # simule l'interruption: garde b144-oracle + rec-oracle, supprime les 2 autres
        for f in ("cell-b144-recovery-R1.json", "cell-rec-recovery-R1.json"):
            os.remove(os.path.join(out, f))
        # fige le contenu des survivantes
        survivors = {f: open(os.path.join(out, f), "rb").read()
                     for f in ("cell-b144-oracle-R1.json", "cell-rec-oracle-R1.json")}
        # ckpt O_EXCL: les ckpts existent déjà → run_cell refuserait —
        # nouveau dir de ckpts pour la reprise (chemin neuf par reprise)
        monkeypatch.setenv("UCM_S2B_CKPT_DIR", str(tmp_path / "ck2"))
        # summary O_EXCL aussi: ts neuf pour la relance? NON — même ts, le
        # summary doit détecter... le summary est O_EXCL: on le retire (il
        # sera réécrit au ts de relance)
        os.remove(os.path.join(out, "factorial-summary-R1.json"))
        r2 = run_factorial(lines, store, updates=1, seed=5, n_eval=2,
                           T=2, out_dir=out, ts="R1")
        assert set(r2["cells"]) == {"b144-oracle", "b144-recovery",
                                    "rec-oracle", "rec-recovery"}
        # les survivantes sont BIT-IDENTIQUES (jamais retouchées)
        for f, content in survivors.items():
            assert open(os.path.join(out, f), "rb").read() == content, f"{f} réécrit!"
        # les manquantes sont complétées avec le MÊME contenu déterministe
        assert json.load(open(os.path.join(out, "cell-b144-recovery-R1.json")))["raw"] == \
            r1["cells"]["b144-recovery"]["raw"]

    def test_resume_corrupt_artifact_explicit_error(self, tmp_path):
        from ucm.eval.factorial_2x2 import run_factorial
        lines, store = _dev_inputs(6)
        out = str(tmp_path / "out")
        run_factorial(lines, store, updates=1, seed=5, n_eval=2,
                      T=2, out_dir=out, ts="C1")
        # corrompt un artefact
        p = os.path.join(out, "cell-b144-oracle-C1.json")
        open(p, "w").write("{CORRUPT")
        os.remove(os.path.join(out, "factorial-summary-C1.json"))
        import shutil
        shutil.rmtree(str(tmp_path / "ck1"), ignore_errors=True)
        with pytest.raises(RuntimeError, match="CORROMPU"):
            run_factorial(lines, store, updates=1, seed=5, n_eval=2,
                          T=2, out_dir=out, ts="C1")

    def test_cell_checkpoints_oexl_exist(self, tmp_path, monkeypatch):
        from ucm.eval.factorial_2x2 import run_factorial
        lines, store = _dev_inputs(6)
        out = str(tmp_path / "out")
        ck = str(tmp_path / "ck1")
        monkeypatch.setenv("UCM_S2B_CKPT_DIR", ck)
        run_factorial(lines, store, updates=1, seed=5, n_eval=2,
                      T=2, out_dir=out, ts="CK1")
        found = os.listdir(ck) if os.path.exists(ck) else []
        assert set(found) == {"b144-oracle-s5.npz", "b144-recovery-s5.npz",
                              "rec-oracle-s5.npz", "rec-recovery-s5.npz"}
        # O_EXCL: relance MÊME cellule (nouveau ts) REFUSE le ckpt existant
        with pytest.raises(RuntimeError, match="O_EXCL"):
            run_factorial(lines, store, updates=1, seed=5, n_eval=2,
                          T=2, out_dir=out, ts="CK2")

    def test_parallel_equals_serial_deterministic(self, tmp_path, monkeypatch):
        """workers>0: même résultats que série (seeds contrôlent tout — v10)."""
        from ucm.eval.factorial_2x2 import run_factorial
        lines, store = _dev_inputs(6)
        monkeypatch.setenv("UCM_S2B_CKPT_DIR", str(tmp_path / "cks"))
        rs = run_factorial(lines, store, updates=1, seed=5, n_eval=2, T=2,
                           out_dir=str(tmp_path / "s"), ts="S")
        monkeypatch.setenv("UCM_S2B_CKPT_DIR", str(tmp_path / "ckp"))
        rp = run_factorial(lines, store, updates=1, seed=5, n_eval=2, T=2,
                           out_dir=str(tmp_path / "p"), ts="P", workers=4)
        for c in ("b144-oracle", "b144-recovery", "rec-oracle", "rec-recovery"):
            assert rs["cells"][c]["raw"] == rp["cells"][c]["raw"], \
                f"{c}: parallèle != série"


class TestSubtreeOptimizerFix:  # lead 14:55 bug racine du 0 pct
    def test_refine_loss_descends_and_params_change(self):
        """Le subtree optimizer ENTRAÎNE réellement le refine: la loss
        descend matériellement ET les params du refine changent (base == canon
        conservée). C'était le bug: intersection vide ⇒ no-op ⇒ 0%."""
        import mlx.core as mx
        mx.set_default_device(mx.cpu)
        from ucm.model.recurrent_block import make_siw_rec_model
        from ucm.v1.runner import finetune
        from ucm.v1.transfer import CoverageTracker, FinetuneConfig
        import hashlib, numpy as np
        import mlx.nn as nn
        m = make_siw_rec_model(d=48, T=2)
        sha = lambda mod: hashlib.sha256("\n".join(
            k + "|" + hashlib.sha256(np.asarray(p.tolist(), dtype=np.float32).tobytes()).hexdigest()
            for k, p in nn.utils.tree_flatten(mod.parameters())).encode()).hexdigest()
        r0 = sha(m.refine)
        b0 = m.base_params_sha256()
        lines, store = _dev_inputs(8)
        from ucm.v1.data_adapter import _couples_core
        recs = _couples_core(store, lines)
        meta = finetune(m, recs, FinetuneConfig(updates=250, seed=1),
                        CoverageTracker(), "fix", len(recs),
                        trainable="refine")
        assert sha(m.refine) != r0, "refine n'a PAS changé — no-op toujours là"
        assert m.base_params_sha256() == b0, "base a dérivé"
        assert meta["loss_last"] < meta["loss_first"], \
            f"loss ne descend pas: {meta['loss_first']} -> {meta['loss_last']}"
        assert meta["n_loss_samples"] >= 1

    def test_ckpt_filename_includes_seed_no_collision(self, tmp_path, monkeypatch):
        """(b) collision ckpt: le nom inclut le seed — deux seeds de la MÊME
        cellule ne crashent plus en O_EXCL."""
        from ucm.eval.factorial_2x2 import run_cell, build_eval_episodes
        lines, store = _dev_inputs(6)
        eps = build_eval_episodes(store, 2, seed=3)
        ck = str(tmp_path / "ck")
        monkeypatch.setenv("UCM_S2B_CKPT_DIR", ck)
        run_cell("b144", "oracle", lines, store, 1, 0, eps)
        run_cell("b144", "oracle", lines, store, 1, 1, eps)  # seed 1: PAS de crash
        import os
        assert sorted(os.listdir(ck)) == ["b144-oracle-s0.npz", "b144-oracle-s1.npz"]

    def test_noop_optimizer_counterfactual_detected(self, tmp_path, monkeypatch):
        """Contre-factuel garde 14:55: finetune no-op (ne touche rien) ⇒
        run_cell lève REFINE NEVER TRAINED."""
        import ucm.eval.factorial_2x2 as f2
        import ucm.v1.runner as _runner
        def noop_finetune(model, records, cfg, cov, arm, k, **kw):
            return {"updates": cfg.updates, "mean_loss_tail": 0.0,
                    "loss_first": None, "loss_last": None, "loss_curve": [],
                    "n_loss_samples": 0, "wall_s": 0.0}
        monkeypatch.setattr(_runner, "finetune", noop_finetune)  # late import in run_cell
        from ucm.eval.factorial_2x2 import build_eval_episodes
        lines, store = _dev_inputs(6)
        eps = build_eval_episodes(store, 2, seed=3)
        monkeypatch.setenv("UCM_S2B_CKPT_DIR", str(tmp_path / "ck"))
        with pytest.raises(RuntimeError, match="REFINE NEVER TRAINED"):
            f2.run_cell("rec", "oracle", lines, store, 5, 3, eps, T=2)
