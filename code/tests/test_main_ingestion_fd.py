"""Audit-hook test: FULL runner_confirm.main ingestion single-FD proof (11:03/11:15).

NOTE: this tests rc.main (the CONFIRMATION runner path). The legacy
runner.py main is tested separately if/when the V1-bis path reuses it.

Spies builtins.open through the COMPLETE main ingestion path — sealed
couples (pattern per budget), sealed test file — and asserts:
    - exactly ≤1 open (in read mode) per sealed file BY REALPATH
    - for EACH sealed file: a fsync_intent_done event PRECEDES its
      sealed_open in an ordered event stream (not post-hoc journal reads)
    - NOT covered here (separate E2E): labels semantic==oracle cross-check,
      legacy runner.py main
Dev-only synthetic fixtures, no sealed artifacts, no test2.
"""

import builtins
import hashlib
import json
import os
import random
import shutil

import pytest

from tests.test_confirm_v3 import _REPO, _LAYOUT_STORE, _json_dump, _json_load, _cap_horizon


def _mk_sealed_couples(path, store, n=8):
    from ucm.env.siw import SIW, SIWLayout
    from ucm.env.siw_oracle import SIWOracle
    from ucm.v1.data_adapter import _spec_of
    lines = []
    h0 = next(iter(store))  # couples: FIRST layout only
    lay = SIWLayout(_spec_of(store[h0]))
    fields = [w["id"] for w in lay.widgets.values() if w["type"] == "field"]
    rng = random.Random(9)
    for i in range(n):
        goal = {"predicate": "SET", "args": {"field": rng.choice(fields)}}
        view = rng.choice(lay.views)
        st = [view, [], [], False, []]
        env = SIW(lay)
        try:
            env.reset({"init": {"view": view, "chosen": {}, "dialog_open": False,
                                "filled": [], "submitted": []}, "goal": goal})
        except Exception:
            continue
        d = SIWOracle(lay, goal).solve(env.state)
        if d["reachable"]:
            lines.append(json.dumps({"goal": goal, "layout_hash": h0,
                                     "state_key": st}, sort_keys=True))
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return len(lines)


def _mk_sealed_test(path, per_pred=2):
    """Balanced mini test file: per_pred episodes for EACH predicate, built
    from DEV layouts (SET/VIEW real, CHOOSE/SUBMITTED crafted from state)."""
    from ucm.env.siw import SIW, SIWLayout
    from ucm.env.siw_oracle import SIWOracle
    from ucm.v1.data_adapter import _spec_of
    store = _json_load(_LAYOUT_STORE)
    out = []
    _store_items = list(store.items())[1:]  # test: skip the couples layout
    if len(_store_items) < 2:
        _store_items = list(store.items())
    for h0, entry in _store_items[:4]:
        lay = SIWLayout(_spec_of(entry))
        ws = list(lay.widgets.values())
        fields = [w["id"] for w in ws if w["type"] == "field"]
        for f in fields[:per_pred]:  # capped to match crafted
            goal = {"predicate": "SET", "args": {"field": f}}
            task = {"init": {"view": lay.views[0], "chosen": {}, "dialog_open": False,
                             "filled": [], "submitted": []}, "goal": goal}
            env = SIW(lay)
            try: env.reset(task)
            except Exception: continue
            d = SIWOracle(lay, goal).solve(env.state)
            if d["reachable"]:
                out.append({"episode_id": f"t-{h0[:6]}-SET-{len(out)}",
                            "layout_spec": entry, "task": task,
                            "d_star": d["d_star"], "layout_hash": h0})
        for v in lay.views[:per_pred]:
            goal = {"predicate": "VIEW", "args": {"view": v}}
            task = {"init": {"view": lay.views[-1], "chosen": {}, "dialog_open": False,
                             "filled": [], "submitted": []}, "goal": goal}
            env = SIW(lay)
            try: env.reset(task)
            except Exception: continue
            d = SIWOracle(lay, goal).solve(env.state)
            if d["reachable"]:
                out.append({"episode_id": f"t-{h0[:6]}-VIEW-{len(out)}",
                            "layout_spec": entry, "task": task,
                            "d_star": d["d_star"], "layout_hash": h0})
    # CHOOSE/SUBMITTED: crafted on any layout (self-sufficient, balanced)
    _h_craft = _store_items[0][0]  # NOT the couples layout
    _spec_c = _store_items[0][1]
    _ws_c = list(_spec_c["widgets"].values())
    _sel = next((w for w in _ws_c if w["type"] == "select"), None)
    _form = next((w for w in _ws_c if w["type"] == "form"), None)
    _opts = [w["id"] for w in _ws_c if w.get("select") == _sel["id"]] if _sel else []
    # FAIL (never silent fallback): these predicates must be constructible
    assert _sel is not None and _opts, "no select+option in craft layout — fixture broken"
    assert _form is not None, "no form in craft layout — fixture broken"
    _lay_c = SIWLayout(_spec_of(_spec_c))
    for i in range(per_pred * 4):
        p = "CHOOSE" if i % 2 == 0 else "SUBMITTED"
        goal = ({"predicate": "CHOOSE", "args": {"select": _sel["id"], "option": _opts[0]}}
                if p == "CHOOSE" else
                {"predicate": "SUBMITTED", "args": {"form": _form["id"]}})
        task = {"init": {"view": _spec_c["views"][0], "chosen": {}, "dialog_open": False,
                         "filled": [], "submitted": []}, "goal": goal}
        env = SIW(_lay_c)
        try:
            env.reset(task)
        except Exception:
            pytest.fail(f"craft {p} task failed to reset — fixture broken")
        d = SIWOracle(_lay_c, goal).solve(env.state)
        if not d["reachable"]:
            pytest.fail(f"craft {p} unreachable — fixture broken")
        out.append({"episode_id": f"t-craft-{i}", "layout_spec": _spec_c,
                    "task": task, "d_star": d["d_star"], "layout_hash": _h_craft})
    import collections as _cc
    cnt = _cc.Counter(e["task"]["goal"]["predicate"] for e in out)
    mn = min(cnt.values())
    seen = _cc.Counter()
    bal = []
    for e in out:
        p = e["task"]["goal"]["predicate"]
        if seen[p] < mn:
            bal.append(e)
            seen[p] += 1
    with open(path, "w") as fh:
        for e in bal:
            fh.write(json.dumps(e, sort_keys=True, default=str) + "\n")
    return len(bal)


class TestMainIngestionSingleFD:
    def test_full_ingestion_one_open_per_sealed_file(self, tmp_path, monkeypatch):
        """Audit-hook: runner.main ingestion — ≤1 read-FD per sealed file
        (couples pattern + test), intent fsynced before first open."""
        pytest.importorskip("mlx")
        monkeypatch.chdir(tmp_path)
        _cap_horizon(monkeypatch, horizon=4)
        store = _json_load(_LAYOUT_STORE)
        # sealed couples per budget
        n = _mk_sealed_couples("sc-k500.jsonl", store, n=12)
        assert n >= 3
        _n_test = _mk_sealed_test("st.jsonl", per_pred=2)
        assert _n_test >= 8
        # source manifest (10 canon/control)
        import glob as _g
        from ucm.v1 import runner_confirm as rc
        entries = []
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
        with open("sm.json", "w") as fh:
            _json_dump({"checkpoints": entries}, fh)
        _json_dump({"path": os.path.abspath("st.jsonl"), "sha256_write_stream": hashlib.sha256(open("st.jsonl", "rb").read()).hexdigest()},
                   open("dm.json", "w"))
        import hashlib as _h
        ch500 = _h.sha256(open("sc-k500.jsonl", "rb").read()).hexdigest()
        th = _h.sha256(open("st.jsonl", "rb").read()).hexdigest()
        proto = rc.build_freeze_manifest(
            "fm.json", couples_path=os.path.abspath("sc-k500.jsonl"), couples_sha=ch500,
            layout_store=_LAYOUT_STORE, source_manifest_path=os.path.abspath("sm.json"),
            updates=1, seed_gen=1, seed_ep=None,
            test2_manifest_pattern=os.path.abspath("dm.json"))
        ns = type("NS", (), {"protocol_hash": proto, "couples_hash": ch500,
                             "test2_hash": th, "couples_file": "sc-k500.jsonl",
                             "test2_file": "st.jsonl", "freeze_manifest": "fm.json",
                             "out_dir": "confirm-out", "updates": 1,
                             "canon_ckpt_pattern": "unused{seed}",
                             "control_ckpt_pattern": "unused{seed}",
                             "layout_store": _LAYOUT_STORE})()
        # AUDIT HOOK: spy all opens, track by realpath
        opens = []
        real_open = builtins.open
        def spy(f, *a, **k):
            mode = a[0] if a else k.get("mode", "r")
            opens.append((os.path.realpath(str(f)), mode))
            return real_open(f, *a, **k)
        # mini fixture: relax the production constants
        import collections as _cc
        monkeypatch.setattr(rc, "EXPECTED_TOTAL", _n_test, raising=True)
        monkeypatch.setattr(rc, "EXPECTED_PER_PREDICATE", _n_test // 4, raising=True)
        
        # ORDERED event stream: intent-log + fsync + sealed opens interleaved.
        # Each fsync is ATTRIBUTED to the LAST intent logged (spy on
        # InteractionsLog.log), so F1↔couples and F2↔test2 are distinguishable.
        events = []
        _last_intent = {"motive": ""}
        _real_log = type(ilog_probe := None) if False else None  # placeholder
        import ucm.v1.transfer as _transfer_mod
        _realILogLog = None
        class _ILogSpy(_transfer_mod.InteractionsLog):
            def log(self, motive, arm="-", target="SIW-dev"):
                events.append(("intent_log", motive, target))
                return super().log(motive, arm, target)
        _real_fsync = rc._fsync
        def fsync_spy(ilog):
            _real_fsync(ilog)
            # attribute this fsync to the most recent intent
            recent = [e for e in events if e[0] == "intent_log"]
            tag = recent[-1][1] if recent else "?"
            events.append(("fsync_intent_done", tag))
        monkeypatch.setattr(rc, "_fsync", fsync_spy)
        _real_open_spy = spy
        _rp_c = os.path.realpath("sc-k500.jsonl")
        _rp_t = os.path.realpath("st.jsonl")
        def ordered_spy(f, *a, **k):
            mode = a[0] if a else k.get("mode", "r")
            rp = os.path.realpath(str(f))
            if "r" in mode and "w" not in mode and rp in (_rp_c, _rp_t):
                which = "couples" if rp == _rp_c else "test2"
                events.append(("sealed_open", which))
            return _real_open_spy(f, *a, **k)
        monkeypatch.setattr(builtins, "open", ordered_spy)
        # patch InteractionsLog inside runner_confirm's namespace
        import ucm.v1.runner_confirm as _rc_mod
        _origILog = _rc_mod.InteractionsLog
        _rc_mod.InteractionsLog = _ILogSpy
        try:
            rc.main(ns)
        finally:
            _rc_mod.InteractionsLog = _origILog
            monkeypatch.setattr(builtins, "open", real_open)
        # assertions: ≤1 read-mode open per sealed file actually consumed
        for f in ("sc-k500.jsonl", "st.jsonl"):
            rp = os.path.realpath(f)
            reads = [o for o in opens if o[0] == rp and "r" in o[1] and "w" not in o[1]]
            assert len(reads) == 1, f"{f}: {len(reads)} read-opens (expect 1): {reads}"
        assert not os.path.exists("sc-k500.jsonl.tmp")
        # STRICT interleaved order: intent_c → F1(couples) → open_c →
        # intent_t → F2(test2) → open_t. Two DISTINCT attributed fsyncs.
        i_ic = [i for i, e in enumerate(events) if e[0] == "intent_log"
                and "couples" in e[1]]
        # fsync events: ("fsync_intent_done", attributed_intent_text)
        f_all = [i for i, e in enumerate(events) if e[0] == "fsync_intent_done"]
        o_c = [i for i, e in enumerate(events) if e[0] == "sealed_open" and e[1] == "couples"]
        i_it = [i for i, e in enumerate(events) if e[0] == "intent_log"
                and "TEST2" in e[1].upper()]
        o_t = [i for i, e in enumerate(events) if e[0] == "sealed_open" and e[1] == "test2"]
        assert len(o_c) == 1 and len(o_t) == 1, f"opens: c={len(o_c)} t={len(o_t)}"
        assert len(f_all) >= 2, f"need ≥2 distinct fsyncs, got {len(f_all)}"
        # strict chain
        assert i_ic and i_ic[0] < f_all[0], "intent_c must precede F1"
        assert f_all[0] < o_c[0], "F1 must precede open_c"
        assert i_it and i_it[0] > o_c[0], "intent_t after open_c (sequential flow)"
        assert i_it[0] < f_all[1], "intent_t must precede F2"
        assert f_all[1] < o_t[0], "F2 must precede open_t"


class TestAdversarialFailClosed:  # tagi-5 11:35 corrected (was vacuous)
    def test_non_hex_format_gated_before_open(self, tmp_path, monkeypatch):
        """(a) FORMAT gate: truly non-hex char ('z') → SystemExit at
        _require_hex64, BEFORE any file open (spy: 0 opens of c.jsonl)."""
        import builtins, os as _os
        import pytest as _pt
        monkeypatch.chdir(tmp_path)
        from ucm.v1 import runner_confirm as rc
        opens = []
        real_open = builtins.open
        def spy(f, *a, **k):
            opens.append(str(f)); return real_open(f, *a, **k)
        monkeypatch.setattr(builtins, "open", spy)
        ns = type("NS", (), {"protocol_hash": "a" * 64,
                             "couples_hash": "b" * 63 + "z",  # z = NOT hex
                             "test2_hash": "d" * 64,
                             "couples_file": "c.jsonl", "test2_file": "t.jsonl",
                             "freeze_manifest": "fm.json", "out_dir": "o",
                             "updates": 2, "canon_ckpt_pattern": "p{seed}",
                             "control_ckpt_pattern": "q{seed}",
                             "layout_store": "s"})()
        try:
            with _pt.raises(SystemExit, match="not a valid sha256"):
                rc.main(ns)
        finally:
            monkeypatch.setattr(builtins, "open", real_open)
        assert not any("c.jsonl" in o for o in opens), opens

    def test_valid_hex_wrong_content_gated_before_open(self, tmp_path, monkeypatch):
        """(b) CONTENT fail-closed: build a REAL freeze manifest, then pass a
        VALID hex64 that DIFFERS from the recorded sha (1 digit flip) →
        SystemExit 'couples CONTENT sha disagrees', c.jsonl NEVER opened."""
        import builtins, json as _j, hashlib as _h, os as _os, glob as _g
        import pytest as _pt
        monkeypatch.chdir(tmp_path)
        from ucm.v1 import runner_confirm as rc
        store = _j.load(open(os.path.join(_REPO, "artifacts/inventory-siw-dev-layouts.json")))
        from tests.test_main_ingestion_fd import _mk_sealed_couples
        n = _mk_sealed_couples("c.jsonl", store, n=4)
        assert n >= 2
        # source manifest (10 ckpts) — same as main test
        entries = []
        for s2 in range(5):
            for pat, role in (("artifacts/*gate2c-B144-s{seed}", "canon"),
                              ("artifacts/*v1-control-s{seed}", "control")):
                d = sorted(_g.glob(os.path.join(_REPO, pat.format(seed=s2))))[-1]
                ck = os.path.join(d, "checkpoint.npz")
                fin = _j.load(open(os.path.join(d, "final.json")))
                entries.append({"path": ck, "sha256": rc._sha256_file(ck),
                                "role": role, "seed": s2,
                                "checkpoint_policy": fin.get("checkpoint_policy", "final"),
                                "source_run_updates_run": fin["updates_run"],
                                "loaded_checkpoint_update": fin.get("best", {}).get("update", fin["updates_run"])})
        with open("sm.json", "w") as fh:
            _j.dump({"checkpoints": entries}, fh)
        _j.dump({}, open("store.json", "w")); _j.dump({}, open("dm.json", "w"))
        true_sha = _h.sha256(open("c.jsonl", "rb").read()).hexdigest()
        proto = rc.build_freeze_manifest(
            "fm.json", couples_path=os.path.abspath("c.jsonl"), couples_sha=true_sha,
            layout_store=os.path.abspath("store.json"),
            source_manifest_path=os.path.abspath("sm.json"),
            updates=1, seed_gen=1, seed_ep=None,
            test2_manifest_pattern=os.path.abspath("dm.json"))
        # FLIP one hex digit: still valid hex64, WRONG content
        wrong_sha = ("e" if true_sha[0] != "e" else "f") + true_sha[1:]
        opens = []
        real_open = builtins.open
        def spy(f, *a, **k):
            opens.append(str(f)); return real_open(f, *a, **k)
        monkeypatch.setattr(builtins, "open", spy)
        ns = type("NS", (), {"protocol_hash": proto,
                             "couples_hash": wrong_sha,
                             "test2_hash": "d" * 64,
                             "couples_file": "c.jsonl", "test2_file": "t.jsonl",
                             "freeze_manifest": "fm.json", "out_dir": "o2",
                             "updates": 1,
                             "canon_ckpt_pattern": "unused{seed}",
                             "control_ckpt_pattern": "unused{seed}",
                             "layout_store": os.path.abspath("store.json")})()
        try:
            with _pt.raises(SystemExit, match="couples CONTENT sha disagrees"):
                rc.main(ns)
        finally:
            monkeypatch.setattr(builtins, "open", real_open)
        assert not any("c.jsonl" in o for o in opens), \
            f"c.jsonl opened despite WRONG content hash: {[o for o in opens if 'c.jsonl' in o]}"
