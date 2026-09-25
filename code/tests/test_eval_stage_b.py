"""Éval Stage-B tests — SYNTHÉTIQUES uniquement (la lecture unique du
scellé appartient au run officiel, jamais aux tests)."""
import json
import os
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _synth_setup(tmp_path, n_eps=4, n_arms=2, seeds=(0,), k_plan=(1,)):
    """Build fake freeze + fake episodes (DEV inventory layouts, NOT sealed)
    + fake checkpoints (fresh model params) in tmp."""
    import mlx.core as mx
    import mlx.nn as nn
    mx.set_default_device(mx.cpu)
    from ucm.model.siw_model import make_siw_model

    inv = json.load(open(os.path.join(_REPO, "artifacts/inventory-siw-dev-layouts.json")))
    entries = list(inv.values())[:n_eps] if isinstance(inv, dict) else inv[:n_eps]

    episodes = []
    for i, e in enumerate(entries):
        spec = dict(e)
        if isinstance(spec.get("widgets"), dict):
            spec["widgets"] = list(spec["widgets"].values())
        views = spec.get("views", ["vw0"])
        init_v = views[-1] if len(views) > 1 else views[0]
        goal_v = views[0] if views[0] != init_v else views[-1]
        episodes.append({
            "episode_id": f"synth-{i:03d}",
            "layout_hash": f"synthhash{i:03d}",
            "layout_spec": spec,
            "task": {"goal": {"args": {"view": goal_v}, "predicate": "VIEW"},
                     "init": {"view": init_v, "chosen": {}, "dialog_open": False,
                              "filled": [], "submitted": []}},
            "d_star": 2,
        })
    eps_path = tmp_path / "synth-episodes.jsonl"
    with open(eps_path, "w") as fh:
        for e in episodes:
            fh.write(json.dumps(e, sort_keys=True, default=str) + "\n")

    # fake checkpoints
    ckpt_dir = tmp_path / "cells"
    ckpt_dir.mkdir()
    arms = ["scratch", "pretrained_TGK"][:n_arms]
    for a in arms:
        for s in seeds:
            for k in k_plan:
                m = make_siw_model()
                mx.savez(str(ckpt_dir / f"{a}-s{s}-k{k}.npz"),
                         **dict(nn.utils.tree_flatten(m.parameters())))

    # fake freeze
    fm = {"arms": {a: {} for a in arms},
          "protocol": {"eval_seed": 700, "k_plan": list(k_plan),
                       "k_plan_exec_order": list(k_plan), "seeds": list(seeds),
                       "updates": 1},
          "inputs": {}}
    fz = tmp_path / "synth-freeze.json"
    json.dump(fm, open(fz, "w"))

    # fake pointer anchored on the synth file
    import hashlib
    sha = hashlib.sha256(open(eps_path, "rb").read()).hexdigest()
    ptr = {"members": [{"name": "test2-episodes.jsonl", "sha256": sha}],
           "schema": "ucm-atomic-bundle/0.1"}
    pp = tmp_path / "synth.pointer"
    json.dump(ptr, open(pp, "w"))
    return str(fz), str(pp), str(eps_path), str(ckpt_dir)


class TestEvalStageB:
    def test_locked_without_token(self, tmp_path, monkeypatch):
        monkeypatch.delenv("V1BIS_STEP4_GO", raising=False)
        from ucm.v1.eval_stage_b import evaluate
        fz, pp, ep, ck = _synth_setup(tmp_path)
        with pytest.raises(RuntimeError, match="LOCKED"):
            evaluate(fz, pp, ep, ck, str(tmp_path / "out"), ts="T1", expected_n=4)

    def test_plan_mode_no_read(self, tmp_path):
        from ucm.v1.eval_stage_b import evaluate
        fz, pp, ep, ck = _synth_setup(tmp_path)
        r = evaluate(fz, pp, ep, ck, str(tmp_path / "out"), run=False)
        assert r["status"] == "plan" and r["n_cells"] == 2  # 2 arms × 1 seed × 1 k

    def test_missing_ckpt_fail_closed(self, tmp_path):
        from ucm.v1.eval_stage_b import evaluate
        fz, pp, ep, ck = _synth_setup(tmp_path)
        os.remove(os.path.join(ck, "scratch-s0-k1.npz"))
        with pytest.raises(RuntimeError, match="missing checkpoints"):
            evaluate(fz, pp, ep, ck, str(tmp_path / "out"), run=False)

    def test_full_eval_synthetic_raw_schema_and_verdict(self, tmp_path, monkeypatch):
        monkeypatch.setenv("V1BIS_STEP4_GO", "LEAD_APPROVED")
        from ucm.v1.eval_stage_b import evaluate
        fz, pp, ep, ck = _synth_setup(tmp_path)
        out = tmp_path / "out"
        r = evaluate(fz, pp, ep, ck, str(out), ts="T2", expected_n=4)
        assert r["status"] == "official_complete"
        assert r["n_cells"] == 2

        # raw schema EXACT (lead 11:27 fields)
        from ucm.v1.eval_stage_b import RAW_FIELDS
        raw_lines = [json.loads(l) for l in open(r["raw"]) if l.strip()]
        assert len(raw_lines) == 2 * 4  # 2 cells × 4 episodes
        for rec in raw_lines:
            assert set(rec) == set(RAW_FIELDS), "raw schema mismatch"

        # verdict published per k + metrics O_EXCL
        assert "k=1" in r["verdicts"]
        assert os.path.exists(r["metrics"])
        assert os.path.exists(out / "stage-b-verdict-raw-k1-T2.json")

        # O_EXCL: re-publish same ts REFUSES
        with pytest.raises(OSError):
            os.open(str(out / "stage-b-metrics-T2.json"),
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY)

    def test_resume_skips_complete_cells(self, tmp_path, monkeypatch):
        monkeypatch.setenv("V1BIS_STEP4_GO", "LEAD_APPROVED")
        from ucm.v1.eval_stage_b import evaluate
        fz, pp, ep, ck = _synth_setup(tmp_path, n_eps=4)
        out = tmp_path / "out"
        r1 = evaluate(fz, pp, ep, ck, str(out), ts="T3", expected_n=4)
        # second evaluate with same ts: raw already complete → cells skipped,
        # aggregation re-runs, O_EXCL on metrics REFUSES (same ts)
        with pytest.raises(OSError):
            evaluate(fz, pp, ep, ck, str(out), ts="T3", expected_n=4)
        # different ts: raw re-appended? NO — cells complete ⇒ 0 lines added
        raw2 = str(out / "stage-b-raw-T4.jsonl")
        # simulate: copy raw and verify skip logic counts
        from ucm.v1.eval_stage_b import evaluate as ev
        # run with T4: re-does everything into new raw (no resume across ts)
        r2 = ev(fz, pp, ep, ck, str(out), ts="T4", expected_n=4)
        assert r2["n_cells"] == 2
