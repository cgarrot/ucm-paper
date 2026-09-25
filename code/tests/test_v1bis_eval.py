"""Tests for eval closed-loop in runner_v1bis (lead 14:27).

Covers: eval per k with raw per-episode data, drift module eval ⇒ abort,
eval seeds separate from train seeds (declared in freeze).
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


def _mk_eval_eps(n=4):
    """Small self-contained eval episodes from DEV layouts."""
    store = json.load(open(_LAYOUT_STORE))
    from ucm.env.siw import SIW, SIWLayout
    from ucm.env.siw_oracle import SIWOracle
    from ucm.v1.data_adapter import _spec_of
    eps = []
    for h0, entry in list(store.items())[:3]:
        lay = SIWLayout(_spec_of(entry))
        ws = list(entry["widgets"].values())
        fields = [w["id"] for w in ws if w["type"] == "field"]
        for f in fields[:2]:
            goal = {"predicate": "SET", "args": {"field": f}}
            task = {"init": {"view": entry["views"][0], "chosen": {},
                             "dialog_open": False, "filled": [],
                             "submitted": []},
                    "goal": goal}
            env = SIW(lay)
            try: env.reset(task)
            except Exception: continue
            d = SIWOracle(lay, goal).solve(env.state)
            if d["reachable"]:
                eps.append({"episode_id": f"ev-{h0[:6]}-{len(eps)}",
                            "layout_spec": entry, "task": task,
                            "d_star": d["d_star"], "layout_hash": h0})
        if len(eps) >= n:
            break
    return eps[:n]


class TestEvalClosedLoop:
    def test_eval_raw_per_episode(self, tmp_path, monkeypatch):
        """Run with eval_episodes: per-episode raw data in artifact, success
        rate computed, all required fields present."""
        monkeypatch.chdir(tmp_path)
        cs, ss = _setup_v1bis(tmp_path, monkeypatch)
        eval_eps = _mk_eval_eps(4)
        assert len(eval_eps) >= 2

        from ucm.v1.runner_v1bis import run_v1bis
        art = run_v1bis("v08.jsonl", cs, "store.json", ss,
                        [1], "out-ev", updates=0, seed=0,
                        eval_episodes=eval_eps, eval_seed=50_000)
        cell = art["cells"]["k=1"]
        assert "eval" in cell
        ev = cell["eval"]
        assert ev["n_episodes"] == len(eval_eps)
        assert ev["success_rate"] is not None
        assert len(ev["raw"]) == len(eval_eps)
        for r in ev["raw"]:
            for field in ("arm", "train_seed", "eval_seed", "k", "episode_id",
                          "layout_hash", "goal_type", "success", "outcome",
                          "length", "n_invalid", "goal_reached_without_stop",
                          "d_star"):
                assert field in r, f"missing {field} in eval raw"

    def test_eval_seed_separate_from_train(self, tmp_path, monkeypatch):
        """eval_seed in artifact differs from train seed; raw records carry it."""
        monkeypatch.chdir(tmp_path)
        cs, ss = _setup_v1bis(tmp_path, monkeypatch)
        eval_eps = _mk_eval_eps(2)
        from ucm.v1.runner_v1bis import run_v1bis
        art = run_v1bis("v08.jsonl", cs, "store.json", ss,
                        [1], "out-es", updates=0, seed=7,
                        eval_episodes=eval_eps, eval_seed=42_000)
        assert art["eval_seed"] == 42_000
        assert art["seed"] == 7
        raw = art["cells"]["k=1"]["eval"]["raw"]
        for r in raw:
            assert r["eval_seed"] == 42_007  # 42000 + seed 7
            assert r["train_seed"] == 7

    def test_drift_eval_module_aborts(self, tmp_path, monkeypatch):
        """Code drift in ucm/eval/rollout.py → abort (it's in BOUND_CODE)."""
        monkeypatch.chdir(tmp_path)
        cs, ss = _setup_v1bis(tmp_path, monkeypatch)
        from ucm.v1.freeze_v1bis import build_freeze_v1bis, BOUND_CODE
        assert "ucm/eval/rollout.py" in BOUND_CODE  # eval modules bound
        assert "ucm/v1/policy_siw.py" in BOUND_CODE
        proto = build_freeze_v1bis("fm.json", couples_path="v08.jsonl",
                                    couples_sha=cs, store_path="store.json",
                                    store_sha=ss, k_plan=[1], seed=0, updates=0)
        # tamper
        fm = json.load(open("fm.json"))
        fm["code"]["ucm/eval/rollout.py"] = "0" * 64
        json.dump(fm, open("fm.json", "w"), indent=2, sort_keys=True)
        tampered = hashlib.sha256(json.dumps(fm, sort_keys=True,
                                              default=str).encode()).hexdigest()
        from ucm.v1.runner_v1bis import run_v1bis
        with pytest.raises(SystemExit, match="code drift"):
            run_v1bis("v08.jsonl", cs, "store.json", ss,
                      [1], "out-drift", updates=0,
                      freeze_manifest="fm.json", protocol_hash=tampered)

    def test_no_eval_when_no_episodes(self, tmp_path, monkeypatch):
        """eval_episodes=None → no eval key in cells (backward compat)."""
        monkeypatch.chdir(tmp_path)
        cs, ss = _setup_v1bis(tmp_path, monkeypatch)
        from ucm.v1.runner_v1bis import run_v1bis
        art = run_v1bis("v08.jsonl", cs, "store.json", ss,
                        [1], "out-noev", updates=0)
        assert "eval" not in art["cells"]["k=1"]
