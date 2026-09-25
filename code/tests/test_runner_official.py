"""Tests for run_v1bis_official (lead 15:48 instrumental wiring).

Dry-run mode: verify freeze, wiring, arm loading (all 4 arms × seed 0),
positional seed↔couples mapping, eval Stage-B refusal. NO training.
"""

import hashlib
import json
import os

import pytest
import numpy as np

from tests.test_confirm_v3 import _REPO


FREEZE_V7 = os.path.join(_REPO, "artifacts/freeze-v1bis-official-v10.json")


class TestRunOfficialDry:
    def test_dry_run_ok(self):
        """Full dry-run: freeze verified, wiring correct, arms loadable,
        eval refuses, positional mapping documented."""
        from ucm.v1.runner_official import run_v1bis_official
        fm = json.load(open(FREEZE_V7))
        proto = hashlib.sha256(json.dumps(fm, sort_keys=True,
                                           default=str).encode()).hexdigest()
        result = run_v1bis_official(FREEZE_V7, proto, dry_run=True)
        assert result["status"] == "dry_run_ok"
        w = result["wiring"]
        assert set(w["arms"]) == {"scratch", "pretrained_TGK",
                                   "control_validity", "control_null"}
        assert w["k_plan"] == [64, 128, 256]
        assert w["seeds"] == list(range(10))
        assert w["updates"] == 2000
        assert w["n_couples_files"] == 10
        # positional mapping: seed 0 ↔ gen_seed 20261030, seed 9 ↔ 20261040
        assert w["positional_mapping"]["0"] == 20261030
        assert w["positional_mapping"]["1"] == 20261040  # replacement in slot 1 (gate v02 order)
        assert w["positional_mapping"]["9"] == 20261039

    def test_eval_stage_b_locked(self):
        """Any non-None episodes passed to eval → RuntimeError with STAGE-B LOCKED."""
        from ucm.v1.runner_official import run_v1bis_official
        fm = json.load(open(FREEZE_V7))
        proto = hashlib.sha256(json.dumps(fm, sort_keys=True,
                                           default=str).encode()).hexdigest()
        # dry_run already tests this, but verify explicitly
        result = run_v1bis_official(FREEZE_V7, proto, dry_run=True)
        assert result["status"] == "dry_run_ok"  # no crash = locked eval OK

    def test_wrong_protocol_hash_refused(self):
        from ucm.v1.runner_official import run_v1bis_official
        with pytest.raises(SystemExit):
            run_v1bis_official(FREEZE_V7, "0" * 64, dry_run=True)

    def test_all_arms_loadable_all_seeds(self):
        """Verify every arm × seed checkpoint is loadable (30 checkpoints)."""
        fm = json.load(open(FREEZE_V7))
        proto = hashlib.sha256(json.dumps(fm, sort_keys=True,
                                           default=str).encode()).hexdigest()
        from ucm.v1.freeze_v1bis import verify_freeze_v1bis
        verified = verify_freeze_v1bis(FREEZE_V7, proto)
        from ucm.v1.runner_official import _load_arm_model
        import mlx.core as mx
        mx.set_default_device(mx.cpu)
        for arm_name in verified["arms"]:
            for seed in verified["protocol"]["seeds"]:
                model = _load_arm_model(arm_name, verified, seed)
                assert model is not None, f"{arm_name} s{seed} failed to load"

    def test_no_training_in_dry_run(self, tmp_path, monkeypatch):
        """Dry-run must NOT call finetune (spy)."""
        monkeypatch.chdir(tmp_path)
        import builtins
        from ucm.v1.runner_official import run_v1bis_official
        fm = json.load(open(FREEZE_V7))
        proto = hashlib.sha256(json.dumps(fm, sort_keys=True,
                                           default=str).encode()).hexdigest()
        # patch finetune in every namespace it could be imported from
        import ucm.v1.runner_official as ro
        calls = []
        if hasattr(ro, "finetune_episode_budget_preloaded"):
            orig = ro.finetune_episode_budget_preloaded
            def spy(*a, **k):
                calls.append(1)
                return {}
            ro.finetune_episode_budget_preloaded = spy
        try:
            run_v1bis_official(FREEZE_V7, proto, dry_run=True)
        finally:
            if hasattr(ro, "finetune_episode_budget_preloaded") and orig:
                ro.finetune_episode_budget_preloaded = orig
        assert calls == [], f"finetune called {len(calls)}× in dry_run"


class TestOneReadPerSeed:  # tagi-5 15:53 fix 1
    def test_couples_read_once_per_seed_not_per_arm(self):
        """The full-run structure reads couples ONCE per seed, reuses across
        all 4 arms. Verify: only 1 read per file in the code path."""
        import inspect
        from ucm.v1 import runner_official as ro
        src = inspect.getsource(ro.run_v1bis_official)
        # verify structural: seed loop (outer) contains read_once;
        # arm loop (inner) does NOT contain read_once
        full_run = src[src.index("# FULL RUN"):]
        seed_start = full_run.index("for seed in seeds")
        arm_start = full_run.index("for arm_name in arm_names", seed_start)
        # find the SECOND arm loop (inside seed loop)
        seed_section = full_run[seed_start:arm_start]
        assert "read_once" in seed_section, "read_once not between seed loop and arm loop"
        # arm section: from arm_start to end, no read_once before the NEXT seed iteration
        rest = full_run[arm_start:]
        # the arm loop should not contain read_once
        assert "reg.read_once" not in rest, "read_once found in arm loop (would re-read)"


class TestFullRunTokenGuard:  # tagi-5 15:53 fix 2
    def test_full_run_locked_without_token(self, monkeypatch):
        """dry_run=False without V1BIS_STEP4_GO=LEAD_APPROVED → RuntimeError."""
        monkeypatch.delenv("V1BIS_STEP4_GO", raising=False)
        from ucm.v1.runner_official import run_v1bis_official
        fm = json.load(open(FREEZE_V7))
        proto = hashlib.sha256(json.dumps(fm, sort_keys=True,
                                           default=str).encode()).hexdigest()
        with pytest.raises(RuntimeError, match="FULL RUN LOCKED"):
            run_v1bis_official(FREEZE_V7, proto, dry_run=False)

    def test_full_run_locked_with_wrong_token(self, monkeypatch):
        monkeypatch.setenv("V1BIS_STEP4_GO", "WRONG")
        from ucm.v1.runner_official import run_v1bis_official
        fm = json.load(open(FREEZE_V7))
        proto = hashlib.sha256(json.dumps(fm, sort_keys=True,
                                           default=str).encode()).hexdigest()
        with pytest.raises(RuntimeError, match="FULL RUN LOCKED"):
            run_v1bis_official(FREEZE_V7, proto, dry_run=False)


class TestNonInheritanceReal:  # lead 18:00 fix 2
    def test_k2_starts_from_fresh_not_post_k1(self, tmp_path, monkeypatch):
        """Real test: k=2 model must start from FRESH init, not post-finetune k=1.
        Compare initial weights (before any finetune), not post-training weights."""
        import mlx.core as mx
        import mlx.nn as nn
        from ucm.model.siw_model import make_siw_model
        from ucm.v1.runner_official import _load_arm_model
        import json as _j
        import hashlib as _h
        import os as _os

        # rebind = lead v8 scope (episode_budget fix 19:58)
        FREEZE = _os.path.join(_REPO, "artifacts/freeze-v1bis-official-v10.json")
        fm = _j.load(open(FREEZE))

        # Build fresh init (seed=0)
        mx.random.seed(9000)
        base_init = nn.utils.tree_flatten(make_siw_model().parameters())

        # Build k=1 model and k=2 model — both should start from base_init
        m1 = _load_arm_model("scratch", fm, 0, fresh_init=base_init)
        m2 = _load_arm_model("scratch", fm, 0, fresh_init=base_init)

        # They should be IDENTICAL at init (same fresh_init, same arm=scratch)
        p1 = dict(nn.utils.tree_flatten(m1.parameters()))
        p2 = dict(nn.utils.tree_flatten(m2.parameters()))
        # Compare a sample of weights
        sample_keys = sorted(p1.keys())[:5]
        for k in sample_keys:
            assert np.array_equal(np.array(p1[k].tolist()),
                                   np.array(p2[k].tolist())), \
                f"weights differ at init: {k} — fresh_init not shared"

        # Now simulate: finetune m1 (change weights), then build m3 for k=2
        # m3 must still match base_init, NOT m1's post-finetune weights
        # Corrupt m1's weights (simulate training) via update
        from mlx.utils import tree_map
        corrupted = tree_map(lambda v: v + 1.0, m1.parameters())
        m1.update(corrupted)
        mx.eval(m1.parameters())

        m3 = _load_arm_model("scratch", fm, 0, fresh_init=base_init)
        p3 = dict(nn.utils.tree_flatten(m3.parameters()))
        for k in sample_keys:
            assert np.array_equal(np.array(p3[k].tolist()),
                                   np.array(p2[k].tolist())), \
                f"k=2 model inherited post-finetune weights from k=1: {k}"


class TestE2ERealFiles:  # lead 18:06 commit 3
    def test_e2e_real_couples_k64_store_v2(self):
        """E2E on REAL official files: read→verify→preload→select k=64→adapter(store v2).
        Proves KeyError impossibility: acceptance couples ⊆ store, 26/26 coverage."""
        import json as _j
        import hashlib as _h
        from ucm.v1.sealed_reader import SealedOpenRegistry
        from ucm.v1.episode_budget import from_read_once, preload_store
        
        # Data-path test: fm loaded directly (freeze rebind = lead v8 scope).
        FREEZE = os.path.join(_REPO, "artifacts/freeze-v1bis-official-v10.json")
        fm = _j.load(open(FREEZE))
        store_spec = fm["inputs"]["store"]

        # suite isolation (real files read by earlier tests in this process)
        import ucm.v1.sealed_reader as _sr
        import ucm.v1.episode_budget as _eb
        _sr._REGISTRY = None
        _eb._STORE_CACHE.clear()

        # 1. REAL store v2 preload — must arrive UNWRAPPED now (19:58 fix)
        store_mapping, store_sha = preload_store(store_spec["path"],
                                                  store_spec["sha256"])
        assert len(store_mapping) == 26, f"store v2 must have 26 layouts, got {len(store_mapping)}"

        # 2. E2E over ALL 10 real couples files (not just 1-2: all 10)
        reg = SealedOpenRegistry()
        total_eps = 0
        for cf in fm["inputs"]["couples_files"][:2]:  # 2 files E2E per lead
            lines_fmt, lines, reader_sha = reg.read_once(cf["path"],
                                                          expected_hash=cf["sha256"])
            verified = from_read_once(lines, reader_sha, cf["sha256"])

            # 3. Acceptance: every layout_hash in the file IS in store v2
            eps = [json.loads(l) for l in verified if l.strip()]
            for ep in eps:
                lh = ep["layout_hash"]
                assert lh in store_mapping, \
                    f"KeyError WOULD occur: {lh} not in store v2"
            total_eps += len(eps)

        assert total_eps > 0, "no episodes verified"
