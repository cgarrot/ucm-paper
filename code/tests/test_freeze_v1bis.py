"""Tests for freeze-manifest DEV (runner_v1bis, lead 14:05).

Covers:
  - build_freeze_v1bis: correct SHAs, O_EXCL, canonical SHA
  - verify_freeze_v1bis: code drift ⇒ abort, hash mismatch ⇒ abort,
    valid manifest passes
  - runner_v1bis with freeze: full run OK, code drift aborts BEFORE any
    data read (spy: 0 opens of couples/store)
  - SHA input divergent ⇒ abort at the sealed reader (before training)
"""

import hashlib
import json
import os
import shutil

import pytest

from tests.test_confirm_v3 import _REPO, _LAYOUT_STORE
from tests.test_e2e_v08 import _generate_v08



def _setup_v1bis(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _generate_v08("v08.jsonl", seed=42, n=4, band=(2, 4))
    shutil.copy(_LAYOUT_STORE, "store.json")
    cs = hashlib.sha256(open("v08.jsonl", "rb").read()).hexdigest()
    ss = hashlib.sha256(open("store.json", "rb").read()).hexdigest()
    return cs, ss


def _build_v1bis(tmp_path, cs, ss, k=(1, 2)):
    from ucm.v1.freeze_v1bis import build_freeze_v1bis
    return build_freeze_v1bis(
        os.path.join(str(tmp_path), "freeze-v1bis.json"),
        couples_path="v08.jsonl", couples_sha=cs,
        store_path="store.json", store_sha=ss,
        k_plan=list(k), seed=0, updates=1)


class TestFreezeV1bis:
    def test_build_and_verify_ok(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cs, ss = _setup_v1bis(tmp_path, monkeypatch)
        proto = _build_v1bis(tmp_path, cs, ss)
        assert len(proto) == 64
        assert os.path.exists("freeze-v1bis.json")
        fm = json.load(open("freeze-v1bis.json"))
        assert set(fm["code"].keys()) >= {"ucm/v1/runner_v1bis.py",
                                           "ucm/v1/episode_budget.py"}
        from ucm.v1.freeze_v1bis import verify_freeze_v1bis
        result = verify_freeze_v1bis("freeze-v1bis.json", proto)
        assert result["protocol"]["k_plan"] == [1, 2]

    def test_oexcl_no_overwrite(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cs, ss = _setup_v1bis(tmp_path, monkeypatch)
        _build_v1bis(tmp_path, cs, ss)
        with pytest.raises((SystemExit, FileExistsError)):
            _build_v1bis(tmp_path, cs, ss)  # second build → refuse

    def test_code_drift_aborts_before_data_read(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cs, ss = _setup_v1bis(tmp_path, monkeypatch)
        _build_v1bis(tmp_path, cs, ss)
        fm = json.load(open("freeze-v1bis.json"))
        fm["code"]["ucm/v1/episode_budget.py"] = "0" * 64
        json.dump(fm, open("freeze-v1bis.json", "w"), indent=2, sort_keys=True)
        from ucm.v1.freeze_v1bis import verify_freeze_v1bis
        tampered = hashlib.sha256(json.dumps(fm, sort_keys=True, default=str).encode()).hexdigest()
        with pytest.raises(SystemExit, match="code drift"):
            verify_freeze_v1bis("freeze-v1bis.json", tampered)

    def test_runner_with_freeze_drift_aborts_no_data_read(self, tmp_path, monkeypatch):
        import builtins
        monkeypatch.chdir(tmp_path)
        cs, ss = _setup_v1bis(tmp_path, monkeypatch)
        _build_v1bis(tmp_path, cs, ss)
        fm = json.load(open("freeze-v1bis.json"))
        fm["code"]["ucm/v1/runner_v1bis.py"] = "0" * 64
        json.dump(fm, open("freeze-v1bis.json", "w"), indent=2, sort_keys=True)
        tampered = hashlib.sha256(json.dumps(fm, sort_keys=True, default=str).encode()).hexdigest()
        from ucm.v1.runner_v1bis import run_v1bis
        opens = []
        real_open = builtins.open
        def spy(f, *a, **k):
            opens.append(os.path.realpath(str(f)))
            return real_open(f, *a, **k)
        monkeypatch.setattr(builtins, "open", spy)
        try:
            with pytest.raises(SystemExit, match="code drift|hash mismatch"):
                run_v1bis("v08.jsonl", cs, "store.json", ss,
                          [1], "out-f", updates=0,
                          freeze_manifest="freeze-v1bis.json",
                          protocol_hash=tampered)
        finally:
            monkeypatch.setattr(builtins, "open", real_open)
        data_opens = [o for o in opens if o.endswith(("v08.jsonl", "store.json"))]
        assert len(data_opens) == 0, f"data opened despite abort: {data_opens}"

    def test_runner_full_run_with_freeze(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cs, ss = _setup_v1bis(tmp_path, monkeypatch)
        from ucm.v1.freeze_v1bis import build_freeze_v1bis
        proto = build_freeze_v1bis("freeze-v1bis.json",
                                    couples_path="v08.jsonl", couples_sha=cs,
                                    store_path="store.json", store_sha=ss,
                                    k_plan=[1, 2], seed=0, updates=0)
        from ucm.v1.runner_v1bis import run_v1bis
        art = run_v1bis("v08.jsonl", cs, "store.json", ss,
                        [1, 2], "out-ok", updates=0,
                        freeze_manifest="freeze-v1bis.json",
                        protocol_hash=proto)
        assert os.path.exists("out-ok/v1bis-artifact.json")
        loaded = json.load(open("out-ok/v1bis-artifact.json"))
        assert loaded["cells"]["k=2"]["n_records"] > 0

class TestFreezeCrossCheck:  # tagi-5 finding 1, 14:10
    def test_couples_sha_divergent_aborts_no_opens(self, tmp_path, monkeypatch):
        """Freeze with couples_sha=X, run with couples_sha=Y → abort BEFORE
        any data open (spy: 0 opens)."""
        import builtins
        monkeypatch.chdir(tmp_path)
        cs, ss = _setup_v1bis(tmp_path, monkeypatch)
        _build_v1bis(tmp_path, cs, ss)
        fm = json.load(open("freeze-v1bis.json"))
        proto = hashlib.sha256(json.dumps(fm, sort_keys=True, default=str).encode()).hexdigest()
        from ucm.v1.runner_v1bis import run_v1bis
        opens = []
        real_open = builtins.open
        def spy(f, *a, **k):
            opens.append(str(f)); return real_open(f, *a, **k)
        monkeypatch.setattr(builtins, "open", spy)
        try:
            with pytest.raises(SystemExit, match="couples SHA disagrees"):
                run_v1bis("v08.jsonl", "f" * 64, "store.json", ss,  # WRONG sha
                          [1], "out-x", updates=0,
                          freeze_manifest="freeze-v1bis.json", protocol_hash=proto)
        finally:
            monkeypatch.setattr(builtins, "open", real_open)
        assert not any("v08.jsonl" in o or "store.json" in o for o in opens)

    def test_k_plan_divergent_aborts(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cs, ss = _setup_v1bis(tmp_path, monkeypatch)
        _build_v1bis(tmp_path, cs, ss, k=(1, 2))
        fm = json.load(open("freeze-v1bis.json"))
        proto = hashlib.sha256(json.dumps(fm, sort_keys=True, default=str).encode()).hexdigest()
        from ucm.v1.runner_v1bis import run_v1bis
        with pytest.raises(SystemExit, match="k_plan.*disagrees"):
            run_v1bis("v08.jsonl", cs, "store.json", ss,
                      [1, 2, 3],  # DIFFERENT k_plan
                      "out-y", updates=0,
                      freeze_manifest="freeze-v1bis.json", protocol_hash=proto)

    def test_seed_divergent_aborts(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cs, ss = _setup_v1bis(tmp_path, monkeypatch)
        _build_v1bis(tmp_path, cs, ss, k=(1,))
        fm = json.load(open("freeze-v1bis.json"))
        proto = hashlib.sha256(json.dumps(fm, sort_keys=True, default=str).encode()).hexdigest()
        from ucm.v1.runner_v1bis import run_v1bis
        with pytest.raises(SystemExit, match="seed disagrees"):
            run_v1bis("v08.jsonl", cs, "store.json", ss,
                      [1], "out-z", updates=0, seed=99,  # DIFFERENT seed
                      freeze_manifest="freeze-v1bis.json", protocol_hash=proto)

    def test_store_sha_divergent_aborts(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cs, ss = _setup_v1bis(tmp_path, monkeypatch)
        _build_v1bis(tmp_path, cs, ss)
        fm = json.load(open("freeze-v1bis.json"))
        proto = hashlib.sha256(json.dumps(fm, sort_keys=True, default=str).encode()).hexdigest()
        from ucm.v1.runner_v1bis import run_v1bis
        with pytest.raises(SystemExit, match="store SHA disagrees"):
            run_v1bis("v08.jsonl", cs, "store.json", "e" * 64,  # WRONG store sha
                      [1], "out-w", updates=0,
                      freeze_manifest="freeze-v1bis.json", protocol_hash=proto)
# module-level helpers already defined above


class TestKPlanOrderBinding:  # 14:18 fix 2
    def test_permutation_aborts(self, tmp_path, monkeypatch):
        """k_plan [1,2,3] frozen, run with [3,2,1] → abort (order matters)."""
        monkeypatch.chdir(tmp_path)
        cs, ss = _setup_v1bis(tmp_path, monkeypatch)
        _build_v1bis(tmp_path, cs, ss, k=(1, 2, 3))
        fm = json.load(open("freeze-v1bis.json"))
        proto = hashlib.sha256(json.dumps(fm, sort_keys=True, default=str).encode()).hexdigest()
        from ucm.v1.runner_v1bis import run_v1bis
        with pytest.raises(SystemExit, match="ORDER"):
            run_v1bis("v08.jsonl", cs, "store.json", ss,
                      [3, 2, 1],  # PERMUTED order
                      "out-perm", updates=0,
                      freeze_manifest="freeze-v1bis.json", protocol_hash=proto)


class TestOfficialVerifyCrossCheck:  # 15:34
    def _mk_official(self, tmp_path):
        import glob as _g
        from ucm.v1.freeze_v1bis import build_freeze_v1bis_official
        def sha(f):
            import hashlib
            h = hashlib.sha256()
            with open(f, "rb") as fh:
                for c in iter(lambda: fh.read(1 << 20), b""): h.update(c)
            return h.hexdigest()
        def ckpt(pattern, role):
            entries = []
            for s in range(10):
                d = sorted(_g.glob(pattern.format(seed=s)))[-1]
                ck = os.path.abspath(os.path.join(d, "checkpoint.npz"))
                fin = json.load(open(os.path.join(d, "final.json")))
                entries.append({"path": ck, "sha256": sha(ck),
                                "checkpoint_policy": "best_validation",
                                "source_run_updates_run": fin["updates_run"],
                                "loaded_checkpoint_update": fin.get("best", {}).get("update", fin["updates_run"]),
                                "role": role, "seed": s})
            return entries
        couples = [{"path": os.path.abspath(f"artifacts/v1bis-gen/v1bis-gen-{s2}.jsonl"),
                    "sha256": sha(f"artifacts/v1bis-gen/v1bis-gen-{s2}.jsonl"),
                    "gen_seed": s2,
                    "status": "replacement_of_31" if s2 == 20261040 else "original"}
                   for s2 in (20261030, 20261032, 20261033, 20261034, 20261035,
                              20261036, 20261037, 20261038, 20261039, 20261040)]
        proto = build_freeze_v1bis_official(
            str(tmp_path / "fm-off.json"),
            couples_files=couples,
            store_path=os.path.abspath("artifacts/inventory-siw-dev-layouts.json"),
            store_sha=sha("artifacts/inventory-siw-dev-layouts.json"),
            k_plan=[64, 128, 256], seeds=list(range(10)),
            updates=2000, eval_seed=50000,
            canon_ckpts=ckpt("artifacts/*gate2c-B144-s{seed}", "canon"),
            control_validity_ckpts=ckpt("artifacts/*v1-control-s{seed}", "control_validity"),
            control_null_ckpts=ckpt("artifacts/*v1-null-s{seed}", "control_null"))
        return couples, ckpt, proto

    def test_couples_files_sha_divergent_aborts(self, tmp_path):
        from ucm.v1.freeze_v1bis import verify_freeze_v1bis
        couples, _, proto = self._mk_official(tmp_path)
        bad_couples = [dict(c) for c in couples]
        bad_couples[0]["sha256"] = "f" * 64  # tamper one sha
        with pytest.raises(SystemExit, match="couples_files.*sha256"):
            verify_freeze_v1bis(str(tmp_path / "fm-off.json"), proto,
                                couples_files=bad_couples)

    def test_arm_checkpoint_sha_divergent_aborts(self, tmp_path):
        from ucm.v1.freeze_v1bis import verify_freeze_v1bis
        couples, ckpt, proto = self._mk_official(tmp_path)
        bad_arms = {"pretrained_TGK": {"checkpoints": ckpt("artifacts/*gate2c-B144-s{seed}", "canon")}}
        # tamper one sha
        bad_arms["pretrained_TGK"]["checkpoints"][3]["sha256"] = "0" * 64
        with pytest.raises(SystemExit, match="arm.*sha256"):
            verify_freeze_v1bis(str(tmp_path / "fm-off.json"), proto,
                                couples_files=couples, arms=bad_arms)

    def test_official_verify_round_trip(self, tmp_path):
        from ucm.v1.freeze_v1bis import verify_freeze_v1bis
        couples, ckpt, proto = self._mk_official(tmp_path)
        fm = verify_freeze_v1bis(str(tmp_path / "fm-off.json"), proto,
                                  couples_files=couples,
                                  arms={"pretrained_TGK": {"checkpoints": ckpt("artifacts/*gate2c-B144-s{seed}", "canon")}})
        assert fm["version"] == 2


class TestStrictOfficialGuards:  # 15:43
    """Strict guards: test fixtures can NEVER feed the official path."""
    def _args(self, **over):
        """Minimal valid args dict with overrides."""
        base = dict(
            couples_files=[{"path": f"/abs/{i}", "sha256": "a" * 64,
                            "gen_seed": 20261030 + i, "status": "original"}
                           for i in range(10)],
            store_path="/abs/store", store_sha="b" * 64,
            k_plan=[64, 128, 256], seeds=list(range(10)),
            updates=2000, eval_seed=50000,
            canon_ckpts=[], control_validity_ckpts=[], control_null_ckpts=[])
        base.update(over)
        return base

    def test_3_files_refused(self):
        from ucm.v1.freeze_v1bis import build_freeze_v1bis_official
        with pytest.raises(SystemExit, match="exactly 10"):
            build_freeze_v1bis_official("/tmp/x.json", **self._args(
                couples_files=self._args()["couples_files"][:3]))

    def test_wrong_k_plan_refused(self):
        from ucm.v1.freeze_v1bis import build_freeze_v1bis_official
        with pytest.raises(SystemExit, match="k_plan"):
            build_freeze_v1bis_official("/tmp/x.json", **self._args(
                k_plan=[100, 500, 2000]))

    def test_wrong_updates_refused(self):
        from ucm.v1.freeze_v1bis import build_freeze_v1bis_official
        with pytest.raises(SystemExit, match="updates"):
            build_freeze_v1bis_official("/tmp/x.json", **self._args(
                updates=100))

    def test_wrong_seeds_refused(self):
        from ucm.v1.freeze_v1bis import build_freeze_v1bis_official
        with pytest.raises(SystemExit, match="seeds"):
            build_freeze_v1bis_official("/tmp/x.json", **self._args(
                seeds=[0, 1, 2]))

    def test_missing_gen_seed_refused(self):
        from ucm.v1.freeze_v1bis import build_freeze_v1bis_official
        bad = self._args()
        del bad["couples_files"][0]["gen_seed"]
        with pytest.raises(SystemExit, match="gen_seed|missing"):
            build_freeze_v1bis_official("/tmp/x.json", **bad)

    def test_v4_artifact_has_batch_64(self):
        """The committed v4 artifact must contain protocol.batch=64."""
        fm = json.load(open(os.path.join(_REPO, "artifacts/freeze-v1bis-official-v10.json")))
        assert fm["protocol"]["batch"] == 64
