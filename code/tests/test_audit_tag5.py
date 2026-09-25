"""Non-régression tagi-5 — harnais d'audit intégrés à la suite (proposition).

Origine : scripts d'audit tagi-5 (repass_integrated.py, gate5v2_repass.py,
verify_postM6.py, verify_final_pipe.py, cost_controlled.py).
Deux familles :
  - tests WS-B rapides (schéma durci, dédup, blobs, reader, splits) — sans artefacts ;
  - tests d'artefacts (canon scellé, strate VAL, critère 4) — skip si absents ;
  - mesure coût intercalée : SKIP par défaut (machine-dépendante), activable
    par AUDIT_TAG5_COST=1 (protocole figé tagi-1 13:31).
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

from ucm.data import fixtures as fx
from ucm.data.reader import iter_records, load_blob, read_all, verify_jsonl
from ucm.data.schema import (SchemaError, canonical_json, policy_input_from_obs,
                             validate_record)
from ucm.data.splits import (canonical_certificate, is_g2_reserved,
                             layout_from_dict, verify_split_manifest)
from ucm.data.writer import (InconsistentDuplicateError, JsonlRecordWriter,
                             seal_manifest, write_blob)
from ucm.env.tinygraph import Layout, TinyGraphKey

REPO = Path(os.environ.get("UCM_REPO", Path(__file__).resolve().parents[1]))
CANON = REPO / "artifacts/data/m0-transitions.jsonl"
MANIFEST = REPO / "artifacts/split-manifest-M0.json"
LAYOUTS = REPO / "artifacts/layouts-M0.json"


def _base():
    return json.loads(canonical_json(fx.make_record()))


def _expect_schema_error(obj, mutate=None):
    r = _base() if obj is None else json.loads(canonical_json(obj))
    if mutate:
        mutate(r)
    with pytest.raises(SchemaError):
        validate_record(r)


class TestSchemaHardening:
    def test_p1a_no_edges_rejected(self):
        _expect_schema_error(None, lambda r: r["policy_input"].__setitem__(
            "relations", [x for x in r["policy_input"]["relations"] if x["pred"] != "adjacent"]))

    def test_p1b_disconnected_rejected(self):
        def drop(r):
            r["policy_input"]["relations"] = [
                x for x in r["policy_input"]["relations"]
                if not (x["pred"] == "adjacent" and {x["subj"], x["obj"]} == {"room_a", "room_b"})]
        _expect_schema_error(None, drop)

    def test_p1c_door_off_edge_rejected(self):
        def off(r):
            for x in r["policy_input"]["relations"]:
                if x["pred"] == "connects" and x["obj"] == "room_b":
                    x["obj"] = "room_a"  # (room_a, room_c) non adjacentes
        _expect_schema_error(None, off)

    def test_p2_d0_requires_exact_stop(self):
        stop = len(fx.LAYOUT.rooms) + 5
        _expect_schema_error(fx.make_record(d_star=0, optimal_actions=[stop, 0]))
        validate_record(fx.make_record(d_star=0, optimal_actions=[stop]))  # exact OK

    def test_p3_unknown_obs_keys_rejected(self):
        env = TinyGraphKey(fx.LAYOUT)
        obs = env.reset({"init": dict(fx.DEFAULT_INIT), "goal": dict(fx.DEFAULT_GOAL)})
        bad = dict(obs)
        bad["d_star"] = 7
        with pytest.raises(SchemaError):
            policy_input_from_obs(bad)

    def test_p14_split_enum(self):
        _expect_schema_error(fx.make_record(split="trian"))

    def test_p11_deep_types_are_schema_errors(self):
        _expect_schema_error(None, lambda r: r["policy_input"].__setitem__("entities", 5))
        _expect_schema_error(None, lambda r: r["policy_input"].__setitem__("relations", 5))

    def test_p11c_reader_annotates_line(self, tmp_path):
        r = _base()
        r["policy_input"]["entities"] = 5
        p = tmp_path / "bad.jsonl"
        p.write_text(canonical_json(r) + "\n", encoding="utf-8")
        with pytest.raises(SchemaError) as exc:
            read_all(p)
        assert ":1:" in str(exc.value)


class TestWriterHardening:
    def test_p4_cross_split_visits_keep_split(self, tmp_path):
        p = tmp_path / "p4.jsonl"
        with JsonlRecordWriter(p, append=False) as w:
            w.write(fx.make_record(split="train"))
            w.write(fx.make_record(split="test_g1"))
        rec = read_all(p)[0]
        visits = rec.provenance.extra.get("visits", [])
        assert rec.provenance.split == "train"
        assert {v.get("split") for v in visits} == {"train", "test_g1"}

    def test_p4b_execution_divergence_rejected(self, tmp_path):
        p = tmp_path / "p4b.jsonl"
        ra = fx.make_record(action_ref=0)
        rb = json.loads(canonical_json(ra))
        rb["execution"]["next_state_hash"] = "0" * 16
        with pytest.raises(InconsistentDuplicateError):
            with JsonlRecordWriter(p, append=False) as w:
                w.write(ra)
                w.write(rb)

    def test_p6_preexisting_duplicates_merged(self, tmp_path):
        p = tmp_path / "p6.jsonl"
        line = canonical_json(fx.make_record(provenance_extra={"episode": 1}))
        p.write_text(line + "\n" + line + "\n", encoding="utf-8")
        with JsonlRecordWriter(p, append=True):
            pass
        assert sum(1 for x in p.read_text().splitlines() if x.strip()) == 1
        assert len(read_all(p)[0].provenance.extra.get("visits", [])) == 2

    def test_p7_blob_tamper_detected(self, tmp_path):
        h = write_blob(tmp_path, {"obs": [1, 2, 3]})
        (tmp_path / f"{h}.json").write_text(canonical_json({"obs": [9]}), encoding="utf-8")
        with pytest.raises(SchemaError):
            load_blob(tmp_path, h)

    def test_p8_verify_jsonl_and_source_filter(self, tmp_path):
        p = tmp_path / "p8.jsonl"
        with JsonlRecordWriter(p, append=False) as w:
            w.write(fx.make_record())
        assert len(list(iter_records(p, source="oracle"))) == 1
        with pytest.raises(ValueError):
            list(iter_records(p, source="ghost"))
        obj = json.loads(p.read_text().splitlines()[0])
        obj["execution"]["observable_result"] = "maybe"
        p.write_text(canonical_json(obj) + "\n", encoding="utf-8")
        assert verify_jsonl(p)["valid"] is False


class TestCertificateAndSplits:
    def test_p9_p10_door_semantics(self):
        rooms = ["v0", "v1", "v2", "v3"]
        end = Layout(rooms, [(rooms[i], rooms[i + 1]) for i in range(3)], 0)
        mid = Layout(rooms, [(rooms[i], rooms[i + 1]) for i in range(3)], 1)
        assert canonical_certificate(end) != canonical_certificate(mid)
        p3a = Layout(["a", "b", "c"], [("a", "b"), ("b", "c")], 0)
        p3b = Layout(["a", "b", "c"], [("a", "b"), ("b", "c")], 1)
        assert canonical_certificate(p3a) == canonical_certificate(p3b)

    def test_p12_wl_hard_pair(self):
        k33 = Layout([f"k{i}" for i in range(6)],
                     [(f"k{i}", f"k{j}") for i in range(3) for j in range(3, 6)], 0)
        prism = Layout(["p0", "p1", "p2", "q0", "q1", "q2"],
                       [("p0", "p1"), ("p1", "p2"), ("p2", "p0"), ("q0", "q1"),
                        ("q1", "q2"), ("q2", "q0"), ("p0", "q0"), ("p1", "q1"),
                        ("p2", "q2")], 0)
        assert canonical_certificate(k33) != canonical_certificate(prism)

    @pytest.mark.skipif(not (MANIFEST.exists() and LAYOUTS.exists()),
                        reason="artefacts canon absents")
    def test_m8_manifest_recheck_and_contamination(self):
        man = json.loads(MANIFEST.read_text())
        layouts = {h: layout_from_dict(d) for h, d in json.loads(LAYOUTS.read_text()).items()}
        assert verify_split_manifest(man, layouts.values())["valid"]
        bad = copy.deepcopy(man)
        a = bad["pools"]["train"].pop(0)
        b = bad["pools"]["val"].pop(0)
        bad["pools"]["train"].append(b)
        bad["pools"]["val"].append(a)
        assert verify_split_manifest(seal_manifest(bad), layouts.values())["valid"] is False

    def test_p13_g2_robust(self):
        assert is_g2_reserved(fx.LAYOUT, {"predicate": "AT", "args": {"object": "key"}}) is False
        assert is_g2_reserved(fx.LAYOUT, {"predicate": "AT", "args": {"object": "key", "room": "zzz"}}) is False


@pytest.mark.skipif(not CANON.exists(), reason="canon scellé absent")
class TestVerdictArtifacts:
    """Reproductions des critères officiels (déterministes ; skip si artefacts absents)."""

    def test_canon_sha_and_double_metric(self):
        import hashlib
        from ucm.data.inventory import build_inventory
        assert hashlib.sha256(CANON.read_bytes()).hexdigest().startswith("9a19d8f4")
        inv = build_inventory(read_all(CANON))
        assert inv["episodes_distinct"] == 2614
        assert inv["episodes_distinct_lines"] == 2599

    def test_canon_immutability(self):
        """Le scellé GATE-0 reste bit-intact après tout run de test."""
        import hashlib
        assert hashlib.sha256(CANON.read_bytes()).hexdigest().startswith("9a19d8f4")
        man = REPO / "artifacts/split-manifest-M0.json"
        if man.exists():
            assert hashlib.sha256(man.read_bytes()).hexdigest().startswith("4d82ecaa")

    def test_val_stratum_b144_reproduction(self):
        """Critères 1/2 GATE-5 B144 reproduits depuis les checkpoints officiels."""
        art = REPO / "artifacts/gate5-report/gate5-v0c-bs144.json"
        if not art.exists():
            pytest.skip("artefact B144 absent")
        from ucm.eval import gate5_report as G5
        from ucm.eval import metrics as M
        r = json.loads(art.read_text())
        runs = r["per_seed_runs"]
        seeds = sorted(int(s) for s in runs)
        val = G5.load_canon_episodes(str(CANON), split="val")
        strat = [e for e in val if e[2] is not None and e[2] >= 3]
        A, B = [], []
        for s in seeds:
            mA, _ = G5.load_model(str(REPO / runs[str(s)]["A"]))
            mB, ab = G5.load_model(str(REPO / runs[str(s)]["B144"]))
            rA = G5.closed_loop(mA, "A", strat, seed=42)
            rB = G5.closed_loop(mB, ab, strat, seed=42)
            for x in rA + rB:
                x.seed = s
            A += rA
            B += rB
        sA, sB = M.summarize(A), M.summarize(B)
        pair = M.paired_hierarchical_bootstrap(B, A, n_boot=2000, seed=42)
        ref = r["criterion_1_val_stratum"]
        assert abs(sA["success_rate"] - ref["success_A"]) < 1e-9
        assert abs(sB["success_rate"] - ref["success_B144"]) < 1e-9
        assert pair["point_diff"] > 0 and pair["ci_low"] > 0

    def test_criterion4_fixture_on_official_checkpoints(self):
        """A bit-exact tie ; B144 sépare (preuve bit-level du critère 4)."""
        import numpy as np
        art = REPO / "artifacts/gate5-report/gate5-v0c-bs144.json"
        if not art.exists():
            pytest.skip("artefact B144 absent")
        import sys
        sys.path.insert(0, str(REPO / "tests"))
        import test_model as T
        from ucm.eval.gate5_report import load_model
        from ucm.model.tensorize import collate, tensorize_obs
        r = json.loads(art.read_text())
        runs = r["per_seed_runs"]["4"]
        obs, r2, r4, _ = T.tie_pair_fixture()
        ex = tensorize_obs(obs)
        ex["labels"] = None
        ids = [e["id"] for e in obs["entities"]]
        i2 = obs["candidates"].index({"action": "MOVE", "arg": r2})
        i4 = obs["candidates"].index({"action": "MOVE", "arg": r4})
        mA, _ = load_model(str(REPO / runs["A"]))
        mB, _ = load_model(str(REPO / runs["B144"]))
        uA = np.array(mA.encode(collate([ex])).tolist())[0]
        lgA = np.array(mA(collate([ex])).tolist())[0]
        uB = np.array(mB.encode(collate([ex])).tolist())[0]
        lgB = np.array(mB(collate([ex])).tolist())[0]
        assert np.array_equal(uA[ids.index(r2)], uA[ids.index(r4)]) and lgA[i2] == lgA[i4]
        assert not np.array_equal(uB[ids.index(r2)], uB[ids.index(r4)]) and lgB[i2] != lgB[i4]


@pytest.mark.skipif(os.environ.get("AUDIT_TAG5_COST") != "1",
                    reason="mesure coût machine-dépendante (AUDIT_TAG5_COST=1 pour activer)")
class TestCostInterleaved:
    def test_b144_official_reproduction(self):
        import statistics
        from ucm.eval import gate5_report as G5
        runs = {"A": REPO / "artifacts/20260922-135941-gate5v2-A-s4",
                "B": REPO / "artifacts/20260922-152651-v0c-bs144-B144-s4"}
        if not runs["B"].exists():
            pytest.skip("checkpoints B144 absents")
        import mlx.core as mx
        from ucm.model.tensorize import collate as _collate, tensorize_obs as _tensorize
        mx.set_default_device(mx.cpu)
        eps = G5.load_canon_episodes(str(CANON), split="train", limit=100)
        obs = [rec["policy_input"] for (_, _, _, rs) in eps for rec in rs]
        import numpy as np
        k = int(np.median([len(o["candidates"]) for o in obs]))
        reps = [o for o in obs if len(o["candidates"]) == k][:8]
        batches = []
        for o in reps:
            ex = _tensorize(o)
            ex["labels"] = None
            batches.append(_collate([ex]))
        mA, _ = G5.load_model(str(runs["A"]))
        mB, _ = G5.load_model(str(runs["B"]))
        keys = [x for x in batches[0] if x != "labels"]
        cfA = mx.compile(lambda *a: mA(dict(zip(keys, a)))[0])
        cfB = mx.compile(lambda *a: mB(dict(zip(keys, a)))[0])

        def p95(cf):
            import time
            for i in range(50):
                mx.eval(cf(*[batches[i % len(batches)][x] for x in keys]))
            t = []
            for i in range(1000):
                args = [batches[i % len(batches)][x] for x in keys]
                mx.eval()
                t0 = time.perf_counter()
                mx.eval(cf(*args))
                t.append(time.perf_counter() - t0)
            t.sort()
            return t[int(0.95 * 1000)] * 1e3

        ratios = []
        for _ in range(6):
            pA = statistics.median([p95(cfA) for _ in range(3)])
            pB = statistics.median([p95(cfB) for _ in range(3)])
            ratios.append(pB / pA)
        assert statistics.median(ratios[1:]) <= 2.0  # médiane blocs 2-6 (protocole figé)
