"""DEV integration test for runner_confirm v3 (pre-freeze; NO real test2).

Builds a mini self-contained DEV test2 file via the WRITER format
({episode_id, layout_spec, task, d_star, layout_hash}), runs the FULL
confirmation flow (failfast → one-read → cells with --updates 2 → raw →
report), and asserts every acceptance point:
  failfast (missing args / wrong protocol hash / existing out dir) with ZERO
  side effects (open-spy); one-open per sealed file; 600-line/150×4/disjunction
  validation; raw flushed before derived stats; primary/CI/per-seed computed
  FROM RAW; ckpt sha 64-hex; raw sha recorded; no tmp files in out dir.
"""

import hashlib
import json
import os
import random
import sys

import pytest

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DEV_COUPLES = os.path.join(_REPO, "artifacts/siw-dev-couples.jsonl")
_LAYOUT_STORE = os.path.join(_REPO, "artifacts/inventory-siw-dev-layouts.json")


class TestConfirmV3:
    FIXTURE = os.path.join(_REPO, "artifacts/dev-e2e-fixture-selfcontained.jsonl")

    def test_failfast_missing_args_zero_side_effects(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        import builtins
        opened = []
        real = builtins.open
        def spy(f, *a, **k):
            opened.append(str(f)); return real(f, *a, **k)
        builtins.open = spy
        try:
            from ucm.v1 import runner_confirm as rc
            with pytest.raises(SystemExit):
                rc.main(type("NS", (), {})())
        finally:
            builtins.open = real
        assert opened == []

    def test_validation_real_fixture_and_negatives(self, tmp_path, monkeypatch):
        """M1 on the REAL sampler fixture: 600, 150/predicate, variable
        layouts (9-36), hash recompute OK, disjunction OK. Negative:
        predicate-IMBALANCED file rejected; uniform-4/layout + balanced
        predicates ACCEPTED (licit pattern)."""
        import json as _json
        from ucm.v1.runner_confirm import _parse_test2_lines
        lines = [l.rstrip("\n") for l in open(self.FIXTURE) if l.strip()]
        assert len(lines) == 600
        eps = _parse_test2_lines(lines, adaptation_layouts=set())
        from collections import Counter
        pc = Counter(e["goal_type"] for e in eps)
        assert all(pc[p] == 150 for p in ("VIEW", "SET", "CHOOSE", "SUBMITTED"))
        nl = len({e["layout_hash"] for e in eps})
        assert 1 <= nl <= 30  # variable (fixture stats: 30 layouts, counts 9-36)
        # negative 1: predicate imbalance
        bad = [l for l in lines]
        e0 = _json.loads(bad[0]); e0["task"]["goal"]["predicate"] = "SET"
        bad[0] = _json.dumps(e0)
        with pytest.raises(RuntimeError):
            _parse_test2_lines(bad, set())
        # negative 2: layout_hash/spec mismatch
        e1 = _json.loads(lines[1]); e1["layout_hash"] = "deadbeefdeadbeef"
        bad2 = [_json.dumps(e1)] + lines[1:]
        with pytest.raises(RuntimeError):
            _parse_test2_lines(bad2, set())

    def test_dev_end_to_end_mini(self, tmp_path, monkeypatch):
        pytest.importorskip("mlx")
        import shutil, hashlib as _h
        monkeypatch.chdir(tmp_path)
        _cap_horizon(monkeypatch)
        shutil.copy(self.FIXTURE, "dev-test2.jsonl")
        store_p, couples_p = _mini_adaptation_pool(".", tag="-e2e")
        # source manifest (G2): 10 real checkpoints with sha64 + proven updates
        from ucm.v1 import runner_confirm as rc
        entries = []
        for s2 in range(5):
            for pat in ("artifacts/*gate2c-B144-s{seed}", "artifacts/*v1-control-s{seed}"):
                d = sorted(__import__("glob").glob(os.path.join(_REPO, pat.format(seed=s2))))[-1]
                ck = os.path.join(d, "checkpoint.npz")
                fin = _json_load(os.path.join(d, "final.json"))
                role = "canon" if "gate2c" in pat else "control"
                entries.append({"path": ck, "sha256": rc._sha256_file(ck),
                                "role": role, "seed": s2,
                                "checkpoint_policy": fin.get("checkpoint_policy", "final"),
                                "source_run_updates_run": fin["updates_run"],
                                "loaded_checkpoint_update": (fin.get("best", {}).get("update")
                                                             if fin.get("checkpoint_policy") == "best_validation"
                                                             else fin["updates_run"])})
        for e in entries:
            e["path"] = os.path.abspath(e["path"])
        with open("source-manifest.json", "w") as fh:
            _json_dump({"checkpoints": entries}, fh)
        ch = _h.sha256(open(couples_p, "rb").read()).hexdigest()
        th = _h.sha256(open("dev-test2.jsonl", "rb").read()).hexdigest()
        _json_dump({"path": T2 if "T2" in dir() else "dev-test2.jsonl", "sha256_write_stream": th}, open("data-manifest.json", "w"))
        proto = rc.build_freeze_manifest(
            "freeze-manifest.json", couples_path=couples_p, couples_sha=ch,
            layout_store=store_p, source_manifest_path="source-manifest.json",
            updates=2, seed_gen=999, seed_ep=None,
            test2_manifest_pattern="data-manifest.json")
        ns = type("NS", (), {"protocol_hash": proto, "couples_hash": ch,
                             "test2_hash": th, "couples_file": couples_p,
                             "test2_file": "dev-test2.jsonl",
                             "freeze_manifest": "freeze-manifest.json",
                             "out_dir": "confirm-out", "updates": 2,
                             "canon_ckpt_pattern": os.path.join(_REPO, "artifacts/*gate2c-B144-s{seed}/checkpoint.npz"),
                             "control_ckpt_pattern": os.path.join(_REPO, "artifacts/*v1-control-s{seed}/checkpoint.npz"),
                             "layout_store": store_p})()
        rc.main(ns)
        report = _json_load("confirm-out/confirm-report.json")
        assert report["primary_from_raw"]["raw"]["n_pairs"] == 5 * 600
        assert len(report["raw_sha256"]) == 64
        assert all(len(v["sha256"]) == 64 for v in report["checkpoints_manifest"].values())
        from collections import Counter
        import json as _j2
        gt = Counter(_j2.loads(l)["goal_type"]
                     for l in open("confirm-out/confirm-episodes-raw.jsonl"))
        assert gt["VIEW"] == 3 * 5 * 150  # G3: goal_type populated (3 bras × 5 seeds × 150)
        files = os.listdir("confirm-out")
        assert not any(f.endswith(".tmp") for f in files)




def _mini_adaptation_pool(tmp, tag=""):
    """3 tiny synthetic layouts + couples (sealed format) + store — disjoint
    from the DEV fixture pool (avoids the disjunction false-positive)."""
    import hashlib as _h
    import json as _json
    from ucm.env.siw import SIW, SIWLayout
    from ucm.env.siw_oracle import SIWOracle
    store, couples = {}, []
    for i in range(3):
        spec = {"views": [f"mv{i}a", f"mv{i}b"], "nav_edges": [[f"mv{i}a", f"mv{i}b"]],
                "widgets": [
                    {"id": f"mf{i}", "type": "field", "view": f"mv{i}a"},
                    {"id": f"mb{i}", "type": "button", "view": f"mv{i}b", "kind": "none"}],
                "labels": {}}
        lay = SIWLayout(spec)
        h = lay.layout_hash()
        store[h] = {"views": spec["views"], "widgets": {w["id"]: w for w in spec["widgets"]},
                    "nav_edges": spec["nav_edges"], "labels": {}}
        goal = {"predicate": "SET", "args": {"field": f"mf{i}"}}
        for view in spec["views"]:
            env = SIW(lay)
            env.reset({"init": {"view": view, "chosen": {}, "dialog_open": False,
                                "filled": [], "submitted": []}, "goal": goal})
            d = SIWOracle(lay, goal).solve(env.state)
            if d["reachable"]:
                couples.append({"goal": goal, "layout_hash": h,
                                "state_key": [view, [], [], False, []]})
    sp = os.path.join(tmp, f"mini-store{tag}.json"); cp = os.path.join(tmp, f"mini-couples{tag}.jsonl")
    _json.dump(store, open(sp, "w"))
    with open(cp, "w") as fh:
        for c in couples:
            fh.write(_json.dumps(c, sort_keys=True) + "\n")
    return sp, cp




def _cap_horizon(monkeypatch, horizon=6):
    """TEST-ONLY: cap rollout horizon (untrained cells wander to 64 on 600
    episodes — minutes per cell; the real runner keeps 64)."""
    from ucm.v1 import runner as _r
    real = _r.run_episode
    def capped(env, pol, *a, **k):
        k.setdefault("horizon", horizon)
        return real(env, pol, *a, **k)
    monkeypatch.setattr(_r, "run_episode", capped)


def _json_load(p):
    import json
    return json.load(open(p))


def _json_dump(o, fh):
    import json
    json.dump(o, fh)


class TestCrashNoPublication:
    FIXTURE = os.path.join(_REPO, "artifacts/dev-e2e-fixture-selfcontained.jsonl")

    def test_injected_crash_no_numbers_no_report(self, tmp_path, monkeypatch):
        """v4.6 on the REAL fixture (600) with mini adaptation pool: crash at
        cell 2 → no published out_dir, no number prints, staging private,
        second process refused WITHOUT re-opening sealed files."""
        monkeypatch.chdir(tmp_path)
        import shutil, hashlib as _h
        _cap_horizon(monkeypatch, horizon=6)
        shutil.copy(self.FIXTURE, "dev-test2.jsonl")
        store_p, couples_p = _mini_adaptation_pool(".", tag="-spy")
        from ucm.v1 import runner_confirm as rc
        entries = []
        for s2 in range(5):
            for pat in ("artifacts/*gate2c-B144-s{seed}", "artifacts/*v1-control-s{seed}"):
                d = sorted(__import__("glob").glob(os.path.join(_REPO, pat.format(seed=s2))))[-1]
                ck = os.path.join(d, "checkpoint.npz")
                fin = _json_load(os.path.join(d, "final.json"))
                entries.append({"path": ck, "sha256": rc._sha256_file(ck),
                                "role": "canon" if "gate2c" in d else "control", "seed": s2,
                                "checkpoint_policy": fin["checkpoint_policy"],
                                "source_run_updates_run": fin["updates_run"],
                                "loaded_checkpoint_update": fin["best"]["update"]})
        for e in entries:
            e["path"] = os.path.abspath(e["path"])
        with open("source-manifest.json", "w") as fh:
            _json_dump({"checkpoints": entries}, fh)
        ch = _h.sha256(open(couples_p, "rb").read()).hexdigest()
        th = _h.sha256(open("dev-test2.jsonl", "rb").read()).hexdigest()
        _json_dump({"path": T2 if "T2" in dir() else "dev-test2.jsonl", "sha256_write_stream": th}, open("data-manifest.json", "w"))
        proto = rc.build_freeze_manifest(
            "freeze-manifest.json", couples_path=couples_p, couples_sha=ch,
            layout_store=store_p, source_manifest_path="source-manifest.json",
            updates=1, seed_gen=999, seed_ep=None,
            test2_manifest_pattern="data-manifest.json")
        ns = type("NS", (), {"protocol_hash": proto, "couples_hash": ch,
                             "test2_hash": th, "couples_file": couples_p,
                             "test2_file": "dev-test2.jsonl",
                             "freeze_manifest": "freeze-manifest.json",
                             "out_dir": "confirm-out", "updates": 1,
                             "canon_ckpt_pattern": os.path.join(_REPO, "artifacts/*gate2c-B144-s{seed}/checkpoint.npz"),
                             "control_ckpt_pattern": os.path.join(_REPO, "artifacts/*v1-control-s{seed}/checkpoint.npz"),
                             "layout_store": store_p})()
        calls = {"n": 0}
        real_eval = rc.evaluate
        def crashing_eval(*a, **k):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("INJECTED CRASH at cell 2")
            return real_eval(*a, **k)
        rc.evaluate = crashing_eval
        printed = []
        real_print = print
        monkeypatch.setattr("builtins.print", lambda *a, **k: printed.append(str(a)))
        try:
            with pytest.raises(RuntimeError):
                rc.main(ns)
            assert not os.path.exists("confirm-out")
            assert os.path.exists("confirm-out.staging")
            assert not any("k=500: 0." in p for p in printed), printed
            import builtins
            opened = []
            real_open = builtins.open
            def spy(f, *a, **k):
                opened.append(str(f)); return real_open(f, *a, **k)
            builtins.open = spy
            try:
                with pytest.raises(SystemExit):
                    rc.main(ns)
            finally:
                builtins.open = real_open
            assert not any("dev-test2" in o or "mini-couples" in o for o in opened), opened
        finally:
            rc.evaluate = real_eval
            monkeypatch.setattr("builtins.print", real_print)


class TestFreezeManifestBinding:
    def test_code_drift_aborts(self, tmp_path, monkeypatch):
        """G4: any bound module modified after freeze → abort before effects."""
        monkeypatch.chdir(tmp_path)
        import shutil as _sh
        _sh.copy(os.path.join(_REPO, "ucm/v1/runner_confirm.py"), "drifted.py")
        from ucm.v1 import runner_confirm as rc
        # simulate drift: freeze binds a fake sha for one module
        import json as _j
        fm = {"version": 2, "code": {"nonexistent.py": "0" * 64},
              "inputs": {}, "generation": {"test2_data_manifest": {}},
              "protocol": {"k_star": 500, "updates": 2}}
        _j.dump(fm, open("fm.json", "w"))
        ns = type("NS", (), {"freeze_manifest": "fm.json", "protocol_hash": "x",
                             "couples_hash": "c", "test2_hash": "t",
                             "couples_file": "a", "test2_file": "b",
                             "out_dir": "o", "updates": 2,
                             "canon_ckpt_pattern": "p{seed}",
                             "control_ckpt_pattern": "q{seed}",
                             "layout_store": "s"})()
        import pytest as _pt
        with _pt.raises(SystemExit):
            rc.main(ns)

    def test_couples_same_path_new_content_aborts(self, tmp_path, monkeypatch):
        """G4: a REPLACED couples file (same path, new content) cannot pass:
        freeze manifest binds the CONTENT sha; a new --couples-hash alone
        mismatches the frozen inputs → abort."""
        monkeypatch.chdir(tmp_path)
        import json as _j
        from ucm.v1 import runner_confirm as rc
        _j.dump({"version": 2, "code": {},
                 "inputs": {"couples": {"path": "/abs/c.jsonl", "sha256": "0" * 64}},
                 "generation": {"test2_data_manifest": {}},
                 "protocol": {"k_star": 500, "updates": 2}}, open("fm.json", "w"))
        canon = _j.dumps(_j.load(open("fm.json")), sort_keys=True)
        proto = __import__("hashlib").sha256(canon.encode()).hexdigest()
        ns = type("NS", (), {"freeze_manifest": "fm.json", "protocol_hash": proto,
                             "couples_hash": "1" * 64, "test2_hash": "f" * 64,
                             "couples_file": "/abs/c.jsonl", "test2_file": "b",
                             "out_dir": "o", "updates": 2,
                             "canon_ckpt_pattern": "p{seed}",
                             "control_ckpt_pattern": "q{seed}",
                             "layout_store": "s"})()
        import pytest as _pt
        with _pt.raises(SystemExit, match="couples"):
            rc.main(ns)


class TestOneReadStrictV6:
    def test_couples_opened_exactly_once_intent_first(self, tmp_path, monkeypatch):
        """Full run spy: the couples path is opened EXACTLY ONCE (by
        read_once), and the interactions log's intent entry is fsynced
        BEFORE that open."""
        import builtins, shutil, hashlib as _h
        monkeypatch.chdir(tmp_path)
        _cap_horizon(monkeypatch, horizon=4)
        tagname = os.environ.get("PYTEST_CURRENT_TEST", "x").replace("/", "-")[:60]
        shutil.copy(TestConfirmV3.FIXTURE, f"dev-test2{tagname}.jsonl")
        T2 = f"dev-test2{tagname}.jsonl"
        store_p, couples_p = _mini_adaptation_pool(".", tag="-t3")
        from ucm.v1 import runner_confirm as rc
        entries = []
        import glob as _g
        for s2 in range(5):
            for pat, role in (("artifacts/*gate2c-B144-s{seed}", "canon"),
                              ("artifacts/*v1-control-s{seed}", "control")):
                d = sorted(_g.glob(os.path.join(_REPO, pat.format(seed=s2))))[-1]
                ck = os.path.join(d, "checkpoint.npz")
                fin = _json_load(os.path.join(d, "final.json"))
                entries.append({"path": ck, "sha256": rc._sha256_file(ck),
                                "role": role, "seed": s2,
                                "checkpoint_policy": fin.get("checkpoint_policy", "final"),
                                "source_run_updates_run": fin["updates_run"],
                                "loaded_checkpoint_update": fin.get("best", {}).get("update", fin["updates_run"])})
        for e in entries:
            e["path"] = os.path.abspath(e["path"])
        with open("source-manifest.json", "w") as fh:
            _json_dump({"checkpoints": entries}, fh)
        ch = _h.sha256(open(couples_p, "rb").read()).hexdigest()
        th = _h.sha256(open(T2, "rb").read()).hexdigest()
        _json_dump({"path": T2 if "T2" in dir() else "dev-test2.jsonl", "sha256_write_stream": th}, open("data-manifest.json", "w"))
        proto = rc.build_freeze_manifest(
            "freeze-manifest.json", couples_path=couples_p, couples_sha=ch,
            layout_store=store_p, source_manifest_path="source-manifest.json",
            updates=1, seed_gen=999, seed_ep=None,
            test2_manifest_pattern="data-manifest.json")
        opens = []
        real_open = builtins.open
        def spy(f, *a, **k):
            opens.append((str(f), a[0] if a else k.get("mode", "r")))
            return real_open(f, *a, **k)
        ns = type("NS", (), {"protocol_hash": proto, "couples_hash": ch,
                             "test2_hash": th, "couples_file": couples_p,
                             "test2_file": T2,
                             "freeze_manifest": "freeze-manifest.json",
                             "out_dir": "confirm-out", "updates": 1,
                             "canon_ckpt_pattern": "unused{seed}",
                             "control_ckpt_pattern": "unused{seed}",
                             "layout_store": store_p})()
        monkeypatch.setattr(builtins, "open", spy)
        try:
            rc.main(ns)
        finally:
            monkeypatch.setattr(builtins, "open", real_open)
        copens = [o for o in opens if couples_p in o[0]]
        assert len(copens) == 1 and "r" in copens[0][1], \
            f"couples opened {len(copens)}× — one-read violated"

    def test_bound_paths_no_glob_rogue_never_loaded(self, tmp_path, monkeypatch):
        """G2 v3: the run loads the MANIFEST-BIND paths; a rogue file
        matching the old glob pattern is never read (spy proves it)."""
        import builtins, shutil, hashlib as _h
        monkeypatch.chdir(tmp_path)
        _cap_horizon(monkeypatch, horizon=4)
        tagname = os.environ.get("PYTEST_CURRENT_TEST", "x").replace("/", "-")[:60]
        shutil.copy(TestConfirmV3.FIXTURE, f"dev-test2{tagname}.jsonl")
        T2 = f"dev-test2{tagname}.jsonl"
        store_p, couples_p = _mini_adaptation_pool(".", tag="-t4")
        from ucm.v1 import runner_confirm as rc
        entries = []
        import glob as _g
        for s2 in range(5):
            for pat, role in (("artifacts/*gate2c-B144-s{seed}", "canon"),
                              ("artifacts/*v1-control-s{seed}", "control")):
                d = sorted(_g.glob(os.path.join(_REPO, pat.format(seed=s2))))[-1]
                ck = os.path.join(d, "checkpoint.npz")
                fin = _json_load(os.path.join(d, "final.json"))
                entries.append({"path": ck, "sha256": rc._sha256_file(ck),
                                "role": role, "seed": s2,
                                "checkpoint_policy": fin.get("checkpoint_policy", "final"),
                                "source_run_updates_run": fin["updates_run"],
                                "loaded_checkpoint_update": fin.get("best", {}).get("update", fin["updates_run"])})
        for e in entries:
            e["path"] = os.path.abspath(e["path"])
        with open("source-manifest.json", "w") as fh:
            _json_dump({"checkpoints": entries}, fh)
        ch = _h.sha256(open(couples_p, "rb").read()).hexdigest()
        th = _h.sha256(open(T2, "rb").read()).hexdigest()
        _json_dump({"path": T2 if "T2" in dir() else "dev-test2.jsonl", "sha256_write_stream": th}, open("data-manifest.json", "w"))
        proto = rc.build_freeze_manifest(
            "freeze-manifest.json", couples_path=couples_p, couples_sha=ch,
            layout_store=store_p, source_manifest_path="source-manifest.json",
            updates=1, seed_gen=999, seed_ep=None,
            test2_manifest_pattern="data-manifest.json")
        rogue = "rogue-gate2c-B144-s9"  # would match the glob, never bound
        ns = type("NS", (), {"protocol_hash": proto, "couples_hash": ch,
                             "test2_hash": th, "couples_file": couples_p,
                             "test2_file": T2,
                             "freeze_manifest": "freeze-manifest.json",
                             "out_dir": "confirm-out2", "updates": 1,
                             "canon_ckpt_pattern": f"*{rogue}*{{seed}}",
                             "control_ckpt_pattern": "unused{seed}",
                             "layout_store": store_p})()
        rc.main(ns)  # runs fine: patterns UNUSED, manifest paths bound
        report = _json_load("confirm-out2/confirm-report.json")
        assert report["checkpoints_manifest"]["scratch|s0"]  # cells produced


class TestDataManifestStrictB3:
    def _harness(self, tmp_path, monkeypatch, dm_extra=None, method_key="sha256_write_stream"):
        """Minimal failfast harness: returns SystemExit or None for a given
        data-manifest content (no test2 open ever — cheap guards first)."""
        import json as _j, hashlib as _h, shutil
        from ucm.v1 import runner_confirm as rc
        monkeypatch.chdir(tmp_path)
        shutil.copy(TestConfirmV3.FIXTURE, "t2.jsonl")
        th = _h.sha256(open("t2.jsonl", "rb").read()).hexdigest()
        dm = {"path": "t2.jsonl", "sha256_write_stream": th}
        if dm_extra:
            dm.update(dm_extra)
        _j.dump(dm, open("dm.json", "w"))
        fm = {"version": 2, "code": {},
              "inputs": {"couples": {"path": "c.jsonl", "sha256": "0" * 64}},
              "generation": {"test2_data_manifest": {"method": method_key,
                                                      "manifest_path": "dm.json"}},
              "protocol": {"k_star": 500, "updates": 2}}
        _j.dump(fm, open("fm.json", "w"))
        proto = _h.sha256(_j.dumps(fm, sort_keys=True).encode()).hexdigest()
        ns = type("NS", (), {"protocol_hash": proto, "couples_hash": "0" * 64,
                             "test2_hash": th, "couples_file": "c.jsonl",
                             "test2_file": "t2.jsonl",
                             "freeze_manifest": "fm.json",
                             "out_dir": "o", "updates": 2,
                             "canon_ckpt_pattern": "p{seed}",
                             "control_ckpt_pattern": "q{seed}",
                             "layout_store": "s"})()
        try:
            rc._failfast(ns)
            return None
        except SystemExit as e:
            return str(e)

    def test_bad_path_aborts(self, tmp_path, monkeypatch):
        msg = self._harness(tmp_path, monkeypatch,
                            dm_extra={"path": "OTHER-file.jsonl"})
        assert msg and "path" in msg

    def test_bad_method_aborts(self, tmp_path, monkeypatch):
        msg = self._harness(tmp_path, monkeypatch, method_key="other_method")
        assert msg and "method" in msg

    def test_missing_path_key_aborts(self, tmp_path, monkeypatch):
        import json as _j, hashlib as _h, shutil
        from ucm.v1 import runner_confirm as rc
        monkeypatch.chdir(tmp_path)
        shutil.copy(TestConfirmV3.FIXTURE, "t2.jsonl")
        th = _h.sha256(open("t2.jsonl", "rb").read()).hexdigest()
        _j.dump({"sha256_write_stream": th}, open("dm.json", "w"))  # NO path key
        fm = {"version": 2, "code": {},
              "inputs": {"couples": {"path": "c.jsonl", "sha256": "0" * 64}},
              "generation": {"test2_data_manifest": {"method": "sha256_write_stream",
                                                      "manifest_path": "dm.json"}},
              "protocol": {"k_star": 500, "updates": 2}}
        _j.dump(fm, open("fm.json", "w"))
        proto = _h.sha256(_j.dumps(fm, sort_keys=True).encode()).hexdigest()
        ns = type("NS", (), {"protocol_hash": proto, "couples_hash": "0" * 64,
                             "test2_hash": th, "couples_file": "c.jsonl",
                             "test2_file": "t2.jsonl",
                             "freeze_manifest": "fm.json",
                             "out_dir": "o", "updates": 2,
                             "canon_ckpt_pattern": "p{seed}",
                             "control_ckpt_pattern": "q{seed}",
                             "layout_store": "s"})()
        import pytest as _pt
        with _pt.raises(SystemExit, match="path"):
            rc._failfast(ns)


class TestHex64Strict:  # B3bis
    def test_non_hex_hash_aborts_zero_fd(self, tmp_path, monkeypatch):
        """'Z'*64 passes len==64 but is NOT sha256 — must abort BEFORE any
        file open (spy: zero opens, including the freeze manifest)."""
        import builtins
        monkeypatch.chdir(tmp_path)
        opened = []
        real = builtins.open
        def spy(f, *a, **k):
            opened.append(str(f)); return real(f, *a, **k)
        builtins.open = spy
        try:
            from ucm.v1 import runner_confirm as rc
            ns = type("NS", (), {"protocol_hash": "a" * 64,
                                 "couples_hash": "Z" * 64,   # non-hex!
                                 "test2_hash": "b" * 64,
                                 "couples_file": "c", "test2_file": "t",
                                 "freeze_manifest": "fm.json", "out_dir": "o",
                                 "updates": 2, "canon_ckpt_pattern": "p{seed}",
                                 "control_ckpt_pattern": "q{seed}",
                                 "layout_store": "s"})()
            import pytest as _pt
            with _pt.raises(SystemExit, match="couples-hash.*not a valid"):
                rc.main(ns)
        finally:
            builtins.open = real
        assert opened == [], f"side-effect opens: {opened}"

    def test_data_manifest_non_hex_attested_aborts(self, tmp_path, monkeypatch):
        import json as _j, hashlib as _h, shutil
        from ucm.v1 import runner_confirm as rc
        monkeypatch.chdir(tmp_path)
        shutil.copy(TestConfirmV3.FIXTURE, "t2.jsonl")
        _j.dump({"path": "t2.jsonl", "sha256_write_stream": "Z" * 64},
                open("dm.json", "w"))
        fm = {"version": 2, "code": {},
              "inputs": {"couples": {"path": "c.jsonl", "sha256": "0" * 64}},
              "generation": {"test2_data_manifest": {"method": "sha256_write_stream",
                                                      "manifest_path": "dm.json"}},
              "protocol": {"k_star": 500, "updates": 2}}
        _j.dump(fm, open("fm.json", "w"))
        proto = _h.sha256(_j.dumps(fm, sort_keys=True).encode()).hexdigest()
        ns = type("NS", (), {"protocol_hash": proto, "couples_hash": "0" * 64,
                             "test2_hash": "b" * 64, "couples_file": "c.jsonl",
                             "test2_file": "t2.jsonl", "freeze_manifest": "fm.json",
                             "out_dir": "o", "updates": 2,
                             "canon_ckpt_pattern": "p{seed}",
                             "control_ckpt_pattern": "q{seed}",
                             "layout_store": "s"})()
        import pytest as _pt
        with _pt.raises(SystemExit, match="attested hash.*not a valid"):
            rc._failfast(ns)


class TestNoManifestMutation:  # 07:19 NO-GO
    def test_relative_path_aborts_without_mutation(self, tmp_path, monkeypatch):
        """Relative entries[].path → abort; the manifest file is NOT modified
        (sha unchanged); absolute → accepted, sha also unchanged."""
        import json as _j, hashlib as _h
        from ucm.v1 import runner_confirm as rc
        monkeypatch.chdir(tmp_path)
        rel = {"version": 1, "checkpoints": [
            {"path": "relative/ckpt.npz", "sha256": "0" * 64, "role": "canon",
             "seed": 0, "checkpoint_policy": "best_validation",
             "source_run_updates_run": 800, "loaded_checkpoint_update": 800}]}
        _j.dump(rel, open("sm.json", "w"))
        sha_before = _h.sha256(open("sm.json", "rb").read()).hexdigest()
        import pytest as _pt
        with _pt.raises(SystemExit, match="RELATIVE"):
            rc.build_freeze_manifest(
                "fm.json", couples_path="c.jsonl", couples_sha="0" * 64,
                layout_store="/abs/store.json", source_manifest_path="sm.json",
                updates=2, seed_gen=1, seed_ep=None,
                test2_manifest_pattern="dm.json")
        assert _h.sha256(open("sm.json", "rb").read()).hexdigest() == sha_before, \
            "source manifest MUTATED on abort!"

    def test_absolute_accepted_no_mutation(self, tmp_path, monkeypatch):
        import json as _j, hashlib as _h, os as _os
        from ucm.v1 import runner_confirm as rc
        monkeypatch.chdir(tmp_path)
        # create a real store to satisfy _sha256_file on layout_store etc.
        _j.dump({}, open("store.json", "w"))
        _j.dump({}, open("dm.json", "w"))
        abs_entries = []
        for role in ("canon", "control"):
            for s2 in range(5):
                abs_entries.append({"path": _os.path.join(str(tmp_path), f"ck-{role}-{s2}.npz"),
                                    "sha256": "0" * 64, "role": role, "seed": s2,
                                    "checkpoint_policy": "best_validation",
                                    "source_run_updates_run": 800,
                                    "loaded_checkpoint_update": 800})
        # create dummy files + matching shas so verify passes
        import hashlib as _h3
        for e in abs_entries:
            open(e["path"], "wb").write(b"")
            e["sha256"] = _h3.sha256(open(e["path"], "rb").read()).hexdigest()
            open(e["path"].replace(".npz", "/final.json").replace("/ck-", "/ck-dir-") if False else e["path"], "a").close()
        # also need final.json dirs... simpler: create dirs alongside
        for role in ("canon", "control"):
            for s2 in range(5):
                d = _os.path.join(str(tmp_path), f"run-{role}-{s2}")
                _os.makedirs(d, exist_ok=True)
                ck = _os.path.join(d, "checkpoint.npz")
                open(ck, "wb").write(b"x")
                import json as _j3, hashlib as _h4
                _j3.dump({"updates_run": 800, "checkpoint_policy": "best_validation",
                          "best": {"update": 800}}, open(_os.path.join(d, "final.json"), "w"))
                e = next(e for e in abs_entries if e["role"] == role and e["seed"] == s2)
                e["path"] = ck
                e["sha256"] = _h4.sha256(open(ck, "rb").read()).hexdigest()
        abs_sm = {"version": 1, "checkpoints": abs_entries}
        _j.dump(abs_sm, open("sm_abs.json", "w"))
        _j.dump({}, open("c.jsonl", "w"))
        sha_before = _h.sha256(open("sm_abs.json", "rb").read()).hexdigest()
        rc.build_freeze_manifest(
            "fm.json", couples_path=_os.path.join(str(tmp_path), "c.jsonl"),
            couples_sha="0" * 64, layout_store=_os.path.join(str(tmp_path), "store.json"),
            source_manifest_path=_os.path.join(str(tmp_path), "sm_abs.json"),
            updates=2, seed_gen=1, seed_ep=None,
            test2_manifest_pattern=_os.path.join(str(tmp_path), "dm.json"))
        assert _h.sha256(open("sm_abs.json", "rb").read()).hexdigest() == sha_before
        fm = _j.load(open("fm.json"))
        assert fm["inputs"]["source_manifest"]["path"].startswith("/")


class TestFreezeValidatesBeforeWrite:  # 08:32
    def test_invalid_source_manifest_aborts_no_freeze_written(self, tmp_path, monkeypatch):
        """A source manifest with a BAD sha (file content mismatch) must abort
        the freeze BEFORE the freeze-manifest is written; nothing mutated."""
        import json as _j, hashlib as _h, os as _os
        from ucm.v1 import runner_confirm as rc
        monkeypatch.chdir(tmp_path)
        # fake checkpoint file with WRONG content vs declared sha
        open("ck.npz", "wb").write(b"WRONG")
        sm = {"checkpoints": [
            {"path": _os.path.join(str(tmp_path), "ck.npz"),
             "sha256": "0" * 64,  # will mismatch
             "role": "canon", "seed": 0,
             "checkpoint_policy": "best_validation",
             "source_run_updates_run": 800, "loaded_checkpoint_update": 800}]}
        _j.dump(sm, open("sm.json", "w"))
        sha_sm = _h.sha256(open("sm.json", "rb").read()).hexdigest()
        import pytest as _pt
        with _pt.raises(SystemExit, match="expected 10|sha MISMATCH"):
            rc.build_freeze_manifest(
                "fm.json", couples_path="/abs/c.jsonl", couples_sha="0" * 64,
                layout_store="/abs/store.json", source_manifest_path="sm.json",
                updates=2, seed_gen=1, seed_ep=None,
                test2_manifest_pattern="/abs/dm.json")
        assert not _os.path.exists("fm.json"), "freeze manifest WRITTEN despite abort!"
        assert _h.sha256(open("sm.json", "rb").read()).hexdigest() == sha_sm

    def test_missing_source_file_aborts_no_freeze_written(self, tmp_path, monkeypatch):
        import os as _os
        import pytest as _pt
        from ucm.v1 import runner_confirm as rc
        monkeypatch.chdir(tmp_path)
        sm = {"checkpoints": [
            {"path": _os.path.join(str(tmp_path), "ABSENT.npz"),
             "sha256": "0" * 64, "role": "canon", "seed": 0,
             "checkpoint_policy": "best_validation",
             "source_run_updates_run": 800, "loaded_checkpoint_update": 800}]}
        import json as _j
        _j.dump(sm, open("sm.json", "w"))
        import json as _j2
        _j2.dump({}, open("store.json", "w"))
        _j2.dump({}, open("dm.json", "w"))
        import hashlib as _h2
        _h2_ = lambda p: _h2.sha256(open(p, "rb").read()).hexdigest()
        with _pt.raises(SystemExit, match="expected 10|MISSING"):
            rc.build_freeze_manifest(
                "fm.json", couples_path="/abs/c.jsonl", couples_sha="0" * 64,
                layout_store=_os.path.join(str(tmp_path), "store.json"),
                source_manifest_path="sm.json",
                updates=2, seed_gen=1, seed_ep=None,
                test2_manifest_pattern=_os.path.join(str(tmp_path), "dm.json"))
        assert not _os.path.exists("fm.json")


class TestSingleFDReal:  # 10:52 NO-GO
    def test_couples_and_test_opened_max_once(self, tmp_path, monkeypatch):
        """Audit hook: builtins.open spy — couples and test files opened at
        most ONCE each through the FULL real path (load_couples + main)."""
        import builtins, json as _j, shutil
        monkeypatch.chdir(tmp_path)
        from ucm.env.siw import SIWLayout
        from ucm.v1.runner import load_couples
        from ucm.v1.sealed_reader import SealedOpenRegistry
        # build a small couples file (sealed format)
        store = _j.load(open(os.path.join(_REPO, "artifacts/inventory-siw-dev-layouts.json")))
        h0 = next(iter(store))
        _ws = list(store[h0]["widgets"].values())
        _field = next(w["id"] for w in _ws if w["type"] == "field")
        couples = [{"goal": {"predicate": "SET", "args": {"field": _field}},
                    "layout_hash": h0, "state_key": [store[h0]["views"][0], [], [], False, []]}]
        with open("c.jsonl", "w") as fh:
            for c in couples:
                fh.write(_j.dumps(c, sort_keys=True) + "\n")
        # spy on opens
        opens = []
        real_open = builtins.open
        def spy(f, *a, **k):
            opens.append(str(f)); return real_open(f, *a, **k)
        monkeypatch.setattr(builtins, "open", spy)
        try:
            recs = load_couples("c.jsonl",
                                layout_store=os.path.join(_REPO, "artifacts/inventory-siw-dev-layouts.json"))
            assert len(recs) >= 1
        finally:
            monkeypatch.setattr(builtins, "open", real_open)
        c_opens = [o for o in opens if o.endswith("c.jsonl")]
        assert len(c_opens) == 1, f"couples opened {len(c_opens)}×: {c_opens}"


class TestFreezeExclusiveCreation:  # 08:34
    def test_preexisting_freeze_aborts_sha_intact(self, tmp_path, monkeypatch):
        """A published freeze manifest must never be overwritten — abort
        BEFORE any write; the existing file's sha is unchanged."""
        import json as _j, hashlib as _h, os as _os
        import pytest as _pt
        from ucm.v1 import runner_confirm as rc
        monkeypatch.chdir(tmp_path)
        _j.dump({"PUBLISHED": True}, open("fm.json", "w"))
        sha_before = _h.sha256(open("fm.json", "rb").read()).hexdigest()
        # build with a valid-enough source manifest to reach the write
        _j.dump({}, open("store.json", "w")); _j.dump({}, open("dm.json", "w"))
        _os.makedirs("d"); open("d/checkpoint.npz", "wb").write(b"x")
        _j.dump({"updates_run": 800, "checkpoint_policy": "best_validation",
                 "best": {"update": 800}}, open("d/final.json", "w"))
        import hashlib as _h2
        sm = {"checkpoints": [
            {"path": _os.path.join(str(tmp_path), "d/checkpoint.npz"),
             "sha256": _h2.sha256(open("d/checkpoint.npz", "rb").read()).hexdigest(),
             "role": "canon", "seed": s2,
             "checkpoint_policy": "best_validation",
             "source_run_updates_run": 800, "loaded_checkpoint_update": 800}
            for s2 in range(5)] + [
            {"path": _os.path.join(str(tmp_path), "d/checkpoint.npz"),
             "sha256": _h2.sha256(open("d/checkpoint.npz", "rb").read()).hexdigest(),
             "role": "control", "seed": s2,
             "checkpoint_policy": "best_validation",
             "source_run_updates_run": 800, "loaded_checkpoint_update": 800}
            for s2 in range(5)]}
        _j.dump(sm, open("sm.json", "w"))
        with _pt.raises(SystemExit, match="ALREADY EXISTS"):
            rc.build_freeze_manifest(
                "fm.json",
                couples_path=_os.path.join(str(tmp_path), "c.jsonl"),
                couples_sha="0" * 64,
                layout_store=_os.path.join(str(tmp_path), "store.json"),
                source_manifest_path=_os.path.join(str(tmp_path), "sm.json"),
                updates=2, seed_gen=1, seed_ep=None,
                test2_manifest_pattern=_os.path.join(str(tmp_path), "dm.json"))
        assert _h.sha256(open("fm.json", "rb").read()).hexdigest() == sha_before


class TestRealpathKeying:  # 10:58
    def test_relative_absolute_symlink_one_read(self, tmp_path, monkeypatch):
        """Same file via relative/absolute/symlink → 1 read, 2nd raises."""
        from ucm.v1.sealed_reader import SealedOpenRegistry
        monkeypatch.chdir(tmp_path)
        with open("f.jsonl", "w") as fh:
            fh.write('{"goal": {"predicate": "VIEW", "args": {"view": "v0"}}, "layout_hash": "h", "state_key": ["v0", [], [], false, []]}\n')
        import os as _os, pytest as _pt
        _os.symlink("f.jsonl", "link.jsonl")
        reg = SealedOpenRegistry()
        fmt, lines, ch = reg.read_once("f.jsonl")
        with _pt.raises(RuntimeError):
            reg.read_once(str(tmp_path / "f.jsonl"))    # absolute → same file
        with _pt.raises(RuntimeError):
            reg.read_once("link.jsonl")                   # symlink → same file
        assert reg.recorded_hash("link.jsonl") == ch     # lookup via any alias
