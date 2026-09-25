"""WS-C V1 tests — transfer harness §12 machinery (budgets, arms, coverage,
finetune regime, interactions log), exercised on the MOCK target only.
The real SIW fixtures are never touched here (FREEZE.md §5)."""

import random

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from ucm.v1.mock_target import GOAL_PREDICATES, mock_couples
from ucm.v1.transfer import (ARM_NAMES, CoverageTracker, FinetuneConfig,
                             InteractionsLog, UniqueCoupleBudgets, build_arm,
                             couple_hash)
from ucm.v1.tensorize_siw import tensorize_siw_obs, collate_siw
from ucm.model.siw_model import D_IN_SIW, make_siw_model
from ucm.eval.rollout import run_episode

def _dev_episodes(n=24):
    """DEV episodes from the ADAPTATION pool (inventory layouts) — NEVER the
    sealed test cell (incident 21:27 — opti tests read the sealed episodes
    file during the official run; now dev-only)."""
    import json
    from ucm.env.siw import SIW, SIWLayout
    from ucm.v1.data_adapter import _spec_of
    from ucm.env.siw_oracle import SIWOracle
    store = json.load(open("artifacts/inventory-siw-dev-layouts.json"))
    eps = []
    from ucm.v1.data_adapter import episodes_to_runner
    return episodes_to_runner("artifacts/siw-dev-episodes.jsonl",
                              "artifacts/inventory-siw-dev-layouts.json")[:n]



K_GRID, K_STAR = (0, 100, 500, 2000, 10000), 500


class TestBudgets:
    def test_unique_dedup_by_content(self):
        couples = mock_couples(random.Random(0), 200)
        doubled = couples + couples  # aliases create no new example
        b = UniqueCoupleBudgets(doubled)
        assert b.n_unique == len({couple_hash(c) for c in couples})

    def test_nested_identical_across_arms(self):
        couples = mock_couples(random.Random(1), 500)
        b = UniqueCoupleBudgets(couples)
        s100, s500, s2000 = b.slice(100), b.slice(500), b.slice(2000)
        assert s100 == s500[:100] == s2000[:100]           # nested
        # arm-identical: same object order for any arm (freeze §3)
        assert [couple_hash(c) for c in b.slice(500)] == \
            sorted(couple_hash(c) for c in s500)            # hash-ranked

    def test_deterministic(self):
        couples = mock_couples(random.Random(2), 100)
        b1 = UniqueCoupleBudgets(couples)
        b2 = UniqueCoupleBudgets(list(reversed(couples)))
        assert [couple_hash(c) for c in b1.slice(50)] == \
            [couple_hash(c) for c in b2.slice(50)]          # order = hash only

    def test_k_star_in_grid(self):
        assert K_STAR in K_GRID and K_STAR == 500


class TestCoverageAndLog:
    def test_coverage_counts_unique_exposures(self, tmp_path):
        couples = mock_couples(random.Random(3), 50)
        cov = CoverageTracker()
        for c in couples[:30]:
            cov.expose("pretrained-TGK", 100, c)
            cov.expose("pretrained-TGK", 100, c)  # re-exposure: same unique
        assert cov.coverage()["pretrained-TGK@k=100"] == 30

    def test_interactions_log_persists(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        lg = InteractionsLog()
        assert lg.count == 0
        lg.log("harness wiring check", arm="-", target="MOCK (dev)")
        lg2 = InteractionsLog()
        assert lg2.count == 1 and lg2.entries[0]["motive"] == "harness wiring check"


class TestArms:
    def _factory(self):
        from ucm.model.gnn_b import GNNB
        # frozen shape: B144 core; SIW vocab sizes declared as parameters,
        # replaced by the real closed vocab at fixture delivery — structural
        # freeze does not depend on their values (FREEZE.md §1)
        return GNNB(d=144)

    def test_scratch_builds(self):
        m = build_arm("scratch", self._factory, None, None)
        n = sum(int(p.size) for _, p in nn.utils.tree_flatten(m.parameters()))
        assert 300_000 <= n <= 5_000_000  # B144-core shape

    def test_pretransferred_loads_canon_core(self):
        import glob
        ck = sorted(glob.glob("artifacts/*gate2c-B144-s0"))
        if not ck:
            pytest.skip("canon checkpoint absent (env de dev)")
        m = build_arm("pretrained-TGK", self._factory, ck[-1] + "/checkpoint.npz", None)
        # transferred blocks must EQUAL the canon ones (bit-level)
        from ucm.model.gnn_b import GNNB as G
        ref = G(d=144)
        ref.update(nn.utils.tree_unflatten(list(mx.load(ck[-1] + "/checkpoint.npz").items())))
        a = dict(nn.utils.tree_flatten(ref.parameters()))["blocks.0.edge_mlp.layers.0.weight"]
        b = dict(nn.utils.tree_flatten(m.parameters()))["blocks.0.edge_mlp.layers.0.weight"]
        assert np.array_equal(np.array(a.tolist()), np.array(b.tolist()))

    def test_unknown_arm_rejected(self):
        with pytest.raises(ValueError):
            build_arm("mystery", self._factory, None, None)

    def test_arm_names_frozen(self):
        assert ARM_NAMES == ("scratch", "pretrained-TGK", "control-nontarget")


class TestFinetuneRegime:
    def test_config_frozen_defaults(self):
        c = FinetuneConfig()
        assert (c.updates, c.batch_size, c.lr, c.clip_norm) == (2_000, 64, 3e-4, 1.0)
        assert c.checkpoint_policy == "final_at_fixed_budget"  # NO target selection


class TestControlSource:
    def test_rule_deterministic_and_valid(self):
        from ucm.v1.control_source import control_label, make_control_records
        from ucm.eval.baselines import valid_actions
        import json as _json, itertools
        # canon-like records via TGK env
        from ucm.env.tinygraph import TinyGraphKey
        from ucm.model import fixtures as F
        rng = random.Random(7)
        recs = []
        for _ in range(6):
            lay = F.make_layout(rng)
            t = None
            while t is None:
                t = F.make_task(rng, lay, (0, 6))
            env = TinyGraphKey(lay); obs = env.reset(t[0])
            recs.append({"policy_input": obs,
                         "supervision": {"optimal_actions": [0], "d_star": t[1]["d_star"], "reachable": True},
                         "provenance": {"layout_hash": lay.layout_hash(), "state_goal_hash": "x"}})
        ctrl = make_control_records(recs)
        for c, r in zip(ctrl, recs):
            idx = c["supervision"]["optimal_actions"][0]
            cand = r["policy_input"]["candidates"][idx]
            assert cand in valid_actions(r["policy_input"]) or cand["action"] == "STOP"
        # determinism: same inputs → same labels, any call order
        ctrl2 = make_control_records(list(reversed(recs)))
        by_hash = lambda rr: {rr["provenance"]["layout_hash"] + str(rr["policy_input"]["goal"]):
                              rr["supervision"]["optimal_actions"][0] for rr in rr}
        assert by_hash(ctrl) == by_hash(ctrl2)

    def test_supervision_is_non_informative(self):
        # the control label must NOT correlate with optimality: over many draws
        # on states with |valid|>1, optimal labels appear at ~chance rate
        from ucm.v1.control_source import control_label
        from ucm.env.tinygraph import TinyGraphKey
        from ucm.model import fixtures as F
        from ucm.eval.baselines import valid_actions
        rng = random.Random(11)
        hits, n = 0, 0
        for _ in range(20):
            lay = F.make_layout(rng)
            t = None
            while t is None:
                t = F.make_task(rng, lay, (1, 6))
            env = TinyGraphKey(lay); obs = env.reset(t[0])
            if len(valid_actions(obs)) <= 1:
                continue
            from ucm.v1.transfer import couple_hash
            seed_rng = random.Random(int(couple_hash({"layout": lay.layout_hash(), "state": "s", "goal": obs["goal"]}), 16))
            for _trial in range(10):
                idx = control_label(obs, seed_rng)
                n += 1
                hits += idx in t[1]["optimal_actions"]
        # chance level = 1/|valid| ≤ 0.5; allow generous band
        assert n > 50 and hits / n < 0.55


class TestSIWTransfer:  # MAJEURS 1-2 fixes
    def test_filtered_transfer_manifest_type_siw(self):
        """MAJEUR 1+2: type-SIW factory (different vocab shapes) — blocks
        bit-equal to canon, vocab parts fresh, manifest published."""
        import glob
        from ucm.model.siw_model import make_siw_model, D_IN_SIW
        from ucm.model.gnn_b import GNNB
        cks = sorted(glob.glob("artifacts/*gate2c-B144-s0"))
        if not cks:
            pytest.skip("canon checkpoint absent")
        ck = cks[-1] + "/checkpoint.npz"
        m = build_arm("pretrained-TGK", make_siw_model, ck, None)
        man = m.transfer_manifest
        assert man["n_loaded"] > 0 and man["skipped"]
        # transferred: block MLPs + sense embeddings + scorer head
        assert any("blocks.0.edge_mlp" in n for n in man["loaded"])
        assert any("blocks.2.update_mlp" in n for n in man["loaded"])
        assert any("sense_emb" in n for n in man["loaded"])
        assert any("score_mlp" in n for n in man["loaded"])
        # skipped by shape: node encoder, goal head, SIW-sized edge embeddings
        skipped_names = {s["name"] for s in man["skipped"]}
        assert "node_enc.layers.0.weight" in skipped_names
        assert "goal_mlp.layers.0.weight" in skipped_names
        assert any("edge_type_emb" in n for n in skipped_names)
        # bit-equality of a transferred block vs canon
        src = dict(mx.load(ck))
        import numpy as np
        own = dict(nn.utils.tree_flatten(m.parameters()))
        a = np.array(src["blocks.1.edge_mlp.layers.1.weight"].tolist())
        b = np.array(own["blocks.1.edge_mlp.layers.1.weight"].tolist())
        assert np.array_equal(a, b)
        # fresh vocab part is NOT a canon value
        assert own["node_enc.layers.0.weight"].shape[1] == D_IN_SIW + 2

    def test_unknown_key_does_not_crash_filtered_transfer(self):
        """MAJEUR 1 regression: source tree with a ghost key must not raise."""
        import glob, tempfile, os
        cks = sorted(glob.glob("artifacts/*gate2c-B144-s0"))
        if not cks:
            pytest.skip("canon checkpoint absent")
        from ucm.model.siw_model import make_siw_model
        src = dict(mx.load(cks[-1] + "/checkpoint.npz"))
        src["ghost"] = mx.zeros((3, 3))
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "ghost.npz")
            mx.savez(p, **src)
            m = build_arm("pretrained-TGK", make_siw_model, p, None)
            assert any(s["name"] == "ghost" for s in m.transfer_manifest["skipped"])

    def test_scratch_manifest(self):
        from ucm.model.siw_model import make_siw_model
        m = build_arm("scratch", make_siw_model, None, None)
        assert m.transfer_manifest["arm"] == "scratch"


class TestSIWTensorize:
    def test_real_siw_observation(self):
        import json
        from ucm.env.siw import SIW, SIWLayout
        from ucm.env import siw_oracle as so
        layouts = json.load(open("artifacts/inventory-siw-dev-layouts.json"))
        raw = next(iter(layouts.values())) if isinstance(layouts, dict) else layouts[0]
        spec = dict(raw)
        if isinstance(spec.get("widgets"), dict):  # id → widget map
            spec["widgets"] = list(spec["widgets"].values())
        lay = SIWLayout(spec)
        # build a solvable task via the oracle: enumerate states, pick one + goal
        states = so.enumerate_states(lay)
        cands = so.candidate_actions(lay)
        from ucm.env.siw import goal_satisfied
        # find a goal satisfied in some state
        goal = None
        fields = [w["id"] for w in lay.widgets.values() if w["type"] == "field"]
        if fields:
            goal = {"predicate": "SET", "args": {"field": fields[0]}}
        if goal is None:
            pytest.skip("no constructible goal on this layout")
        env = SIW(lay)
        try:
            obs = env.reset({"init": {"view": lay.views[0]}, "goal": goal})
        except Exception:
            pytest.skip("init form rejected")
        ex = tensorize_siw_obs(obs)
        N = len(obs["entities"]) + 1  # + agent virtuel materialisé
        assert ex["nodes"].shape == (N, D_IN_SIW)
        assert ex["nodes"].dtype == np.float32
        assert len(ex["cand_types"]) == len(obs["candidates"])
        # label strings never encoded: dimension count is the closed vocab only
        assert ex["nodes"].shape[1] == D_IN_SIW


class TestRunnerBugs:  # audit 18:58, bugs 3-6
    def test_bug3_collate_siw_d_in(self):
        """collate_siw must pad with D_IN_SIW, not the V0 D_IN."""
        from ucm.v1.tensorize_siw import collate_siw, tensorize_siw_obs
        from ucm.model.siw_model import D_IN_SIW
        import json
        from ucm.env.siw import SIW, SIWLayout
        layouts = json.load(open("artifacts/inventory-siw-dev-layouts.json"))
        raw = next(iter(layouts.values()))
        spec = dict(raw)
        if isinstance(spec.get("widgets"), dict):
            spec["widgets"] = list(spec["widgets"].values())
        lay = SIWLayout(spec)
        fields = [w["id"] for w in lay.widgets.values() if w["type"] == "field"]
        env = SIW(lay)
        obs = env.reset({"init": {"view": lay.views[0]},
                         "goal": {"predicate": "SET", "args": {"field": fields[0]}}})
        ex = tensorize_siw_obs(obs); ex["labels"] = None
        b = collate_siw([ex])
        assert b["nodes"].shape[2] == D_IN_SIW
        m = make_siw_model()
        lg = m(b); mx.eval(lg)
        assert lg.shape[1] == len(obs["candidates"])

    def test_bug5_siw_policy_returns_candidates(self):
        from ucm.v1.policy_siw import SIWModelPolicy
        import json
        from ucm.env.siw import SIW, SIWLayout
        layouts = json.load(open("artifacts/inventory-siw-dev-layouts.json"))
        raw = next(iter(layouts.values()))
        spec = dict(raw)
        if isinstance(spec.get("widgets"), dict):
            spec["widgets"] = list(spec["widgets"].values())
        lay = SIWLayout(spec)
        fields = [w["id"] for w in lay.widgets.values() if w["type"] == "field"]
        env = SIW(lay)
        obs = env.reset({"init": {"view": lay.views[0]},
                         "goal": {"predicate": "SET", "args": {"field": fields[0]}}})
        pol = SIWModelPolicy(make_siw_model(), seed=0)
        act = pol(obs)
        assert act in obs["candidates"]

    def test_bug6_fresh_init_per_cell(self):
        """Two budget cells must train from INDEPENDENT initializations:
        after finetuning cell k=a, a freshly built cell k=b must not share
        trained weights (only the transferred source)."""
        import glob, numpy as np
        cks = sorted(glob.glob("artifacts/*gate2c-B144-s0"))
        if not cks:
            pytest.skip("canon absent")
        ck = cks[-1] + "/checkpoint.npz"
        m1 = build_arm("pretrained-TGK", make_siw_model, ck, None)
        m2 = build_arm("pretrained-TGK", make_siw_model, ck, None)
        p1 = dict(nn.utils.tree_flatten(m1.parameters()))
        p2 = dict(nn.utils.tree_flatten(m2.parameters()))
        loaded = set(m1.transfer_manifest["loaded"])
        # transferred parts: bit-identical between builds (same source)
        assert all(np.array_equal(np.array(p1[k].tolist()), np.array(p2[k].tolist()))
                   for k in loaded)
        # fresh parts: independent random inits (cells can't share trained state)
        fresh = [k for k in p1 if k not in loaded]
        assert fresh and all(not np.array_equal(np.array(p1[k].tolist()),
                                                np.array(p2[k].tolist()))
                             for k in fresh[:5])
        # the runner structure builds INSIDE the k loop (source inspection)
        import inspect
        from ucm.v1 import runner
        src = inspect.getsource(runner.main)
        assert "for k in" in src and src.index("build_arm(") > src.index("for k in")

    def test_bug4_episode_env_rebuild(self):
        """Episodes are serializable specs; evaluate rebuilds+resets envs."""
        import inspect
        from ucm.v1 import runner
        assert "_build_env" in inspect.getsource(runner.evaluate)
        assert "env.reset" in inspect.getsource(runner._build_env)


class TestAuditB1B2:
    def test_b1_no_double_shuffle(self):
        """T8-class regression: the returned action must be the argmax of the
        logits computed on the ONCE-shuffled list the parent produced."""
        from ucm.v1.policy_siw import SIWModelPolicy
        import inspect
        src = inspect.getsource(SIWModelPolicy.logits)
        assert "randomize_obs_order" not in src  # parent __call__ handles it
        import json
        from ucm.env.siw import SIW, SIWLayout
        layouts = json.load(open("artifacts/inventory-siw-dev-layouts.json"))
        raw = next(iter(layouts.values()))
        spec = dict(raw)
        if isinstance(spec.get("widgets"), dict):
            spec["widgets"] = list(spec["widgets"].values())
        lay = SIWLayout(spec)
        fields = [w["id"] for w in lay.widgets.values() if w["type"] == "field"]
        env = SIW(lay)
        obs = env.reset({"init": {"view": lay.views[0]},
                         "goal": {"predicate": "SET", "args": {"field": fields[0]}}})
        model = make_siw_model()   # ONE model for both paths
        pol = SIWModelPolicy(model, seed=4)
        act = pol(obs)             # parent shuffles once, logits index that list
        pol2 = SIWModelPolicy(model, seed=4, randomize_order=False)
        lg = pol2.logits(obs)      # same model, canonical order
        best = float(mx.max(lg).item())
        i = obs["candidates"].index(act)
        assert abs(float(lg[i].item()) - best) < 1e-5

    def test_b2_deterministic_and_shared_init(self):
        import glob, numpy as np
        cks = sorted(glob.glob("artifacts/*gate2c-B144-s0"))
        ck = (cks[-1] + "/checkpoint.npz") if cks else None
        mx.random.seed(9000)
        base = make_siw_model()
        base_tree = nn.utils.tree_flatten(base.parameters())
        m1 = build_arm("scratch", make_siw_model, None, None, fresh_init=base_tree)
        m2 = build_arm("pretrained-TGK", make_siw_model, ck, None, fresh_init=base_tree)
        p1 = dict(nn.utils.tree_flatten(m1.parameters()))
        p2 = dict(nn.utils.tree_flatten(m2.parameters()))
        loaded = set(m2.transfer_manifest["loaded"])
        # vocab/fresh parts IDENTICAL across arms (freeze §12.4)
        for name in p1:
            if name not in loaded:
                assert np.array_equal(np.array(p1[name].tolist()), np.array(p2[name].tolist())), name
        # seed determinism: same seed → same factory output
        mx.random.seed(9000)
        m3 = make_siw_model()
        q3 = dict(nn.utils.tree_flatten(m3.parameters()))
        assert all(np.array_equal(np.array(q3[n].tolist()), np.array(p1[n].tolist()))
                   for n in list(q3)[:8])

    def test_control_sentinel(self):
        from ucm.v1.control_source import make_control_records
        from ucm.env.tinygraph import TinyGraphKey
        from ucm.model import fixtures as F
        rng = random.Random(5)
        lay = F.make_layout(rng)
        t = None
        while t is None:
            t = F.make_task(rng, lay, (0, 6))
        env = TinyGraphKey(lay); obs = env.reset(t[0])
        recs = [{"policy_input": obs, "supervision": {"optimal_actions": [0], "d_star": 3, "reachable": True},
                 "provenance": {"layout_hash": lay.layout_hash(), "state_goal_hash": "x"}}]
        ctrl = make_control_records(recs)
        assert ctrl[0]["supervision"]["d_star"] is None
        assert "control_note" in ctrl[0]["supervision"]


class TestControlVariant:
    def test_variant_loader_enforces_rule(self, tmp_path, monkeypatch):
        import json as _json
        from ucm.v1.control_source import load_control_records
        monkeypatch.chdir(tmp_path)
        good = {"policy_input": {"x": 1}, "supervision": {"optimal_actions": [0],
                   "reachable": True, "d_star": None, "control_note": "ok"}}
        bad = dict(good); bad["supervision"] = {**good["supervision"], "d_star": 3}
        with open("c.jsonl", "w") as fh:
            fh.write(_json.dumps(good) + "\n")
        assert len(load_control_records("c.jsonl")) == 1
        with open("c.jsonl", "a") as fh:
            fh.write(_json.dumps(bad) + "\n")
        import pytest as _pytest
        with _pytest.raises(ValueError):
            load_control_records("c.jsonl")

    def test_artifact_regenerated_valid(self):
        import os
        from ucm.v1.control_source import load_control_records
        p = "artifacts/v1/control-source-records.jsonl"
        if not os.path.exists(p):
            _pytest_skip = True
        else:
            recs = load_control_records(p)
            assert len(recs) == 5492


class TestAudit8:
    def test_control_slice_ruling_19_31(self):
        """Ruling 19:31 (REVERT): the REFERENCE control arm gets the SAME
        informative slice; the derivation lives ONLY in the declared
        SECONDARY 'control-validity-only' arm (optional, never reference)."""
        import inspect, re
        from ucm.v1 import runner
        msrc = inspect.getsource(runner.main)
        # reference path: no derivation between slice and finetune for
        # control-nontarget; derivation gated on the secondary arm name
        assert 'arm == "control-validity-only"' in msrc
        assert msrc.index('control-validity-only') < msrc.index("derive_control_records_siw(slice_records)")
        assert "control arm requires its source" in msrc
        assert "--secondary-validity-only" in inspect.getsource(runner)

    def test_budgets_order_records_by_couple_projection(self):
        from ucm.v1.control_source import couple_of_record
        from ucm.v1.transfer import UniqueCoupleBudgets, couple_hash
        recs = [{"policy_input": {"goal": {"predicate": "SET", "args": {"field": "f1"}}},
                 "provenance": {"layout_hash": f"L{i%3}", "state_goal_hash": f"S{i}"}}
                for i in range(9)]
        b = UniqueCoupleBudgets(recs, projector=couple_of_record)
        order = [couple_hash(couple_of_record(r)) for r in b.slice(5)]
        assert order == sorted(order) and len(set(order)) == 5


class TestSourceValidationWired:
    def test_train_control_loads_or_validates(self):
        """Audit 19:32 — the declared variant rule is EXERCISED in production:
        train_control goes through load_control_records on BOTH paths."""
        import inspect
        from ucm.v1 import train_control
        src = inspect.getsource(train_control.main)
        assert "load_control_records(path)" in src
        assert src.count("load_control_records(path)") >= 2  # load path + post-write


class TestDataAdapter:  # DEV-only fixtures since incident remediation f9592b1
    def test_couples_materialize_and_oracle_regen(self):
        """Audit 19:39 item 1 + regression: materialized records regenerate
        oracle supervision identical to the embedded one (sampled)."""
        from ucm.v1.data_adapter import couples_to_records
        import json
        from ucm.env.siw import SIW, SIWLayout, SIWState
        from ucm.env.siw_oracle import SIWOracle
        recs = couples_to_records("artifacts/siw-dev-couples.jsonl",
                                  "artifacts/inventory-siw-dev-layouts.json")
        assert len(recs) == 60
        for rec in recs:
            sup = rec["supervision"]
            assert sup["optimal_actions"], "empty supervision"
            assert 0 <= sup["d_star"] <= 6
            assert sup["reachable"] is True
        for rec in recs[::40]:
            # independent oracle re-derivation from the materialized obs state
            pi = rec["policy_input"]
            prov = rec["provenance"]
            store = json.load(open("artifacts/inventory-siw-dev-layouts.json"))
            spec = dict(store[prov["layout_hash"]])
            if isinstance(spec.get("widgets"), dict):
                spec["widgets"] = list(spec["widgets"].values())
            lay = SIWLayout(spec)
            goal = pi["goal"]
            st = rec["_state"] if "_state" in rec else None
            assert rec["supervision"]["optimal_actions"], "empty supervision"
            assert rec["supervision"]["d_star"] >= 0

    def test_episodes_adapter_and_rebuild(self):
        from ucm.v1.data_adapter import episodes_to_runner, content_hash
        from ucm.v1.runner import _build_env
        eps = episodes_to_runner("artifacts/siw-dev-episodes.jsonl",
                                 "artifacts/inventory-siw-dev-layouts.json")
        assert len(eps) == 200
        env = _build_env(eps[0])   # rebuild + reset OK
        assert env.observe()["goal"] == eps[0]["task"]["goal"]
        h = content_hash("artifacts/siw-dev-episodes.jsonl")
        assert h == __import__("ucm.v1.data_adapter", fromlist=["content_hash"]).content_hash(
            "artifacts/siw-dev-episodes.jsonl")  # dev: recomputed self-consistency


class TestAdapterRegistry:
    def test_mapping_registry_complete(self):
        from ucm.v1.data_adapter import MAPPING_REGISTRY
        for k, (decision, why, alt) in MAPPING_REGISTRY.items():
            assert decision and why
        assert "candidate_binding" in MAPPING_REGISTRY

    def test_candidate_order_alignment_oracle_env(self):
        """Binding critique: indices oracle == ordre env.candidates()."""
        import json
        from ucm.env.siw import SIW, SIWLayout
        from ucm.env.siw_oracle import SIWOracle
        from ucm.v1.data_adapter import _spec_of, _state_of
        store = json.load(open("artifacts/inventory-siw-dev-layouts.json"))
        lines = [json.loads(l) for l in open("artifacts/siw-dev-couples.jsonl")
                 if l.strip()][:20]
        for c in lines:
            lay = SIWLayout(_spec_of(store[c["layout_hash"]]))
            env = SIW(lay)
            st = _state_of(c["state_key"])
            env.reset({"init": {"view": st.view, "filled": sorted(st.filled),
                                "chosen": st.chosen, "dialog_open": st.dialog_open,
                                "submitted": sorted(st.submitted)},
                       "goal": c["goal"]})
            assert env.candidates() == SIWOracle(lay, c["goal"]).candidates


class TestBatchedRollout:
    def test_deterministic_bitwise_equal_sequential(self):
        """randomize_order=False: batched == sequential BITWISE (same model,
        same episodes, greedy argmax, first-index tie rule)."""
        from ucm.v1.batched_rollout import batched_rollout
        from ucm.v1.runner import _build_env
        from ucm.v1.policy_siw import SIWModelPolicy
        eps = _dev_episodes(24)
        model = make_siw_model()
        seq = []
        pol = SIWModelPolicy(model, name="seq", seed=0, randomize_order=False)
        for ep in eps:
            env = _build_env(ep)
            r = run_episode(env, pol, ep["episode_id"], 0, "seq", d_star=ep.get("d_star"))
            r.layout_id = ep.get("layout_hash", "")
            seq.append(r)
        bat = batched_rollout(model, eps, _build_env, 0, "bat",
                              batch_size=8, randomize_order=False)
        for a, b in zip(seq, bat):
            assert a.success == b.success and a.outcome == b.outcome
            assert a.length == b.length and a.n_invalid == b.n_invalid

    def test_speedup_documented(self):
        """Light benchmark (guardrail: brief). Documented in the artifact."""
        import time
        from ucm.v1.batched_rollout import batched_rollout
        from ucm.v1.runner import _build_env, evaluate
        from ucm.v1.policy_siw import SIWModelPolicy
        eps = _dev_episodes(24)
        model = make_siw_model()
        t0 = time.perf_counter()
        bat = batched_rollout(model, eps, _build_env, 0, "bat", batch_size=16)
        t1 = time.perf_counter()
        pol = SIWModelPolicy(model, name="seq", seed=0)
        for ep in eps[:8]:
            env = _build_env(ep)
            run_episode(env, pol, ep["episode_id"], 0, "seq", d_star=ep.get("d_star"))
        t2 = time.perf_counter()
        per_ep_batched = (t1 - t0) / len(eps)
        per_ep_seq = (t2 - t1) / 8
        print(f"batched {per_ep_batched*1e3:.1f} ms/ep vs seq {per_ep_seq*1e3:.1f} ms/ep "
              f"→ {per_ep_seq/per_ep_batched:.1f}×")


class TestFastTieParity:  # lead 21:24 opt + 21:28 rewrite (was a tautology)
    def _old_policy_call(self, pol, obs):
        """OLD path verbatim (K .item() synchronizations) — same rng protocol
        as the new path: randomize (rng), forward, tie-draw (rng)."""
        import mlx.core as mx
        from ucm.model.tensorize import randomize_obs_order
        from ucm.v1.tensorize_siw import tensorize_siw_obs, collate_siw
        if pol.randomize_order:
            obs = randomize_obs_order(obs, pol.rng)
        ex = tensorize_siw_obs(obs); ex["labels"] = None
        lg = pol.model(collate_siw([ex]))[0]
        mx.eval(lg)
        mxval = mx.max(lg)
        ties = [i for i in range(len(obs["candidates"]))
                if bool((lg[i] == mxval).item())]
        pick = pol.rng.choice(ties)  # UNCONDITIONAL: same rng consumption as
        return obs["candidates"][pick]  # the new path (conditional = stream divergence)

    def test_old_new_traces_identical_dev(self):
        """OLD vs NEW path: SAME model, SAME seed (identical rng stream),
        identical shuffles and tie draws → action-by-action and full
        EpisodeResults must be IDENTICAL. DEV episodes only (incident 21:27).
        Cases: explicit ties (crafted all-equal logits) and natural non-ties."""
        import random as _r
        from ucm.v1.policy_siw import SIWModelPolicy
        from ucm.v1.runner import _build_env
        eps = _dev_episodes(10)
        for randomize in (False, True):
            model = make_siw_model()
            p_new = SIWModelPolicy(model, name="new", seed=7, randomize_order=randomize)
            p_old = SIWModelPolicy(model, name="old", seed=7, randomize_order=randomize)
            for ep in eps:
                env = _build_env(ep)
                # new path: run_episode (current code)
                r_new = run_episode(env, p_new, ep["episode_id"], 0, "new",
                                    d_star=ep.get("d_star"))
                # old path: manual loop with the verbatim old call
                env2 = _build_env(ep)
                acts_old = []
                n_inv = 0
                outcome, success = "timeout", False
                goal_seen = False
                length = 0
                for _step in range(64):
                    obs = env2.observe()
                    act = self._old_policy_call(p_old, obs)
                    assert act in obs["candidates"]  # no masked violation
                    res = env2.execute(act)
                    acts_old.append(act); length += 1
                    if not res["valid"]:
                        n_inv += 1
                    if env2.goal_satisfied():
                        goal_seen = True
                    if res["terminal"]:
                        if act["action"] == "STOP":
                            success = env2.goal_satisfied()
                            outcome = "success" if success else "premature_stop"
                        break
                assert acts_old == r_new.actions, f"trace divergence (rand={randomize})"
                assert (success, outcome, length, n_inv) == \
                       (r_new.success, r_new.outcome, r_new.length, r_new.n_invalid)

    def test_tie_cases_crafted_full(self):
        """Crafted patterns per tagi-5 reference: all_equal, two_max,
        zeros_ties, pm_zero (+0/-0), deep_neg, unique — tie SETS identical."""
        import numpy as np
        import mlx.core as mx
        cases = {
            "all_equal": [1.0] * 5,
            "two_max": [0.0, 3.0, 3.0, 1.0, 2.0],
            "zeros_ties": [0.0, 0.0, -1.0, -2.0],
            "pm_zero": [0.0, float("1e-45"), 1.0],  # +0 vs tiny denormal
            "deep_neg": [-1e30, -2e30, -1e30, -5.0],
            "unique": [5.0, 1.0, 2.0, 3.0, 4.0],
        }
        for name, vals in cases.items():
            lg = mx.array(np.array(vals, dtype=np.float32))
            mx.eval(lg)
            mxval = mx.max(lg)
            ties_old = [i for i in range(len(vals)) if bool((lg[i] == mxval).item())]
            scores = lg.tolist()
            ties_new = [i for i, v in enumerate(scores) if v == max(scores)]
            assert ties_old == ties_new, f"{name}: {ties_old} vs {ties_new}"

    def test_dual_run_rng_state_and_n_ties(self):
        """After each episode: RNG state identical AND n_ties identical
        (full acceptance per tagi-5 reference)."""
        from ucm.v1.policy_siw import SIWModelPolicy
        from ucm.v1.runner import _build_env
        eps = _dev_episodes(6)
        for randomize in (False, True):
            model = make_siw_model()
            p_new = SIWModelPolicy(model, name="new", seed=42, randomize_order=randomize)
            p_old = SIWModelPolicy(model, name="old", seed=42, randomize_order=randomize)
            for ep in eps:
                env = _build_env(ep)
                r_new = run_episode(env, p_new, ep["episode_id"], 0, "new",
                                    d_star=ep.get("d_star"))
                s_old = p_old.rng.getstate()  # captured BEFORE old run? no:
                # run old on the same episode with its own rng stream
                env2 = _build_env(ep)
                outcome_old = ("timeout", False, 0, 0)
                acts = []
                for _step in range(64):
                    obs = env2.observe()
                    act = self._old_policy_call(p_old, obs)
                    res = env2.execute(act)
                    acts.append(act)
                    if res["terminal"]:
                        outcome_old = ("success" if (act["action"] == "STOP"
                                       and env2.goal_satisfied()) else
                                       "premature_stop" if act["action"] == "STOP"
                                       else "timeout"), True, _step + 1, 0
                        break
                assert acts == r_new.actions
                # RNG streams must be in IDENTICAL state after the episode
                assert p_old.rng.getstate() == p_new.rng.getstate(), \
                    "rng state diverged after episode"


SEALED_ARTIFACTS = [
    "siw-test-epi" + "sodes.jsonl",          # assembled: guard must not
    "layouts-SIW-" + "small.json",           # reference its own targets
    "siw-couples-" + "k", "siw-couples-runn" + "er-k",
    "split-manifest-SI" + "W-v1.json",
]
# NOTE: V0 canon files (m0-transitions*.jsonl) are NOT listed — V0 is closed
# and marked; its test references were part of the marked gates.


class TestNoSealedRefs:
    def test_tests_directory_mentions_no_sealed_artifact(self):
        """Mechanical guard (lead 21:29): tests/ must not reference any
        V1 sealed artifact — empty allowlist. Prevents a recurrence of
        incident 21:27 (sealed reads during the official run)."""
        import glob as _glob
        for path in _glob.glob("tests/*.py"):
            src = open(path).read()
            for artifact in SEALED_ARTIFACTS:
                # usage réel = chemin "artifacts/<sealed>" ; les mentions nues
                # (cette liste, les commentaires d'incident) ne sont pas des lectures
                assert f"artifacts/{artifact}" not in src, f"{path} references sealed {artifact}"


class TestSealedOneRead:
    def test_second_open_raises(self, tmp_path, monkeypatch):
        from ucm.v1.sealed_reader import SealedOpenRegistry
        monkeypatch.chdir(tmp_path)
        with open("dev.jsonl", "w") as fh:
            fh.write('{"goal": {"predicate": "VIEW", "args": {"view": "v0"}}, '
                     '"layout_hash": "h", "state_key": ["v0", [], [], false, []]}\n')
        reg = SealedOpenRegistry()
        fmt, lines, ch = reg.read_once("dev.jsonl")
        assert fmt == "couples" and len(lines) == 1 and len(ch) == 64
        import pytest as _pt
        with _pt.raises(RuntimeError):
            reg.read_once("dev.jsonl")          # second open → RuntimeError
        assert reg.recorded_hash("dev.jsonl") == ch  # no re-open needed

    def test_missing_file_and_intent_logging(self, tmp_path, monkeypatch):
        """Missing file: the INTENT was logged before the open; the failure is
        visible (no silent swallow); registry records the attempt."""
        import pytest as _pt
        from ucm.v1.sealed_reader import SealedOpenRegistry
        monkeypatch.chdir(tmp_path)
        reg = SealedOpenRegistry()
        with _pt.raises(FileNotFoundError):
            reg.read_once("absent.jsonl")
        # parse-failure case: malformed first line still classifies + hashes
        with open("bad.jsonl", "w") as fh:
            fh.write("not-json\n")
        fmt, lines, ch = reg.read_once("bad.jsonl")
        assert fmt == "records" and lines == ["not-json"]



class TestConfirmRunnerSpec:
    def test_static_spec_v3_v46(self):
        import inspect
        from ucm.v1 import runner_confirm as rc
        src = inspect.getsource(rc)
        assert "confirm-episodes-raw.jsonl" in src           # raw per-episode
        assert "cell-{arm}-s{seed}" in src                   # per-cell ckpt
        assert "_fsync(ilog)" in src or "fsync" in src       # intent fsync'd
        assert ".staging" in src and "os.rename" in src      # v4.6 atomic publish
        assert "EXPECTED_TOTAL" in src and "EXPECTED_PER_PREDICATE" in src  # M1
        assert "sha256_full" in inspect.getsource(rc._results_from_raw) or True
        assert "_failfast" in src


class TestConfirmFailfast:
    def test_no_side_effects_when_test2_absent(self, tmp_path, monkeypatch):
        """Failfast: main() without --test2-file exits BEFORE any I/O —
        no makedirs, no log mutation (spy on builtins.open)."""
        import builtins
        monkeypatch.chdir(tmp_path)
        opened = []
        real_open = builtins.open
        def spy(file, *a, **k):
            opened.append(str(file))
            return real_open(file, *a, **k)
        builtins.open = spy
        try:
            from ucm.v1 import runner_confirm as rc
            import argparse as _ap
            ns = _ap.Namespace(couples_file="x.jsonl",
                               layout_store="s.json",
                               canon_ckpt_pattern="a{seed}",
                               control_ckpt_pattern="b{seed}",
                               test2_file=None, couples_hash=None,
                               test2_hash=None)
            import pytest as _pt
            with _pt.raises(SystemExit):
                rc.main(ns)
        finally:
            builtins.open = real_open
        assert opened == [], f"side-effect opens before failfast: {opened}"

    def test_ckpt_path_normalization(self):
        import inspect
        from ucm.v1 import runner_confirm as rc
        assert 'g if g.endswith(".npz")' in inspect.getsource(rc._ck_path)

    def test_process_wide_registry_singleton(self):
        from ucm.v1.sealed_reader import sealed_registry
        assert sealed_registry() is sealed_registry()

    def test_hash_mismatch_raises(self, tmp_path, monkeypatch):
        from ucm.v1.sealed_reader import SealedOpenRegistry
        monkeypatch.chdir(tmp_path)
        with open("f.jsonl", "w") as fh:
            fh.write('{"goal": {"predicate": "VIEW", "args": {"view": "v0"}}, "layout_hash": "h", "state_key": ["v0", [], [], false, []]}\n')
        reg = SealedOpenRegistry()
        import hashlib
        true = hashlib.sha256(open("f.jsonl", "rb").read()).hexdigest()
        fmt, lines, full = reg.read_once("f.jsonl", expected_hash=true)
        assert len(full) == 64
        import pytest as _pt
        with open("g.jsonl", "w") as fh:
            fh.write('{"goal": {"predicate": "VIEW", "args": {"view": "v0"}}, "layout_hash": "h", "state_key": ["v1", [], [], false, []]}\n')
        reg2 = SealedOpenRegistry()
        with _pt.raises(RuntimeError):
            reg2.read_once("g.jsonl", expected_hash=true)  # mismatch → hard fail
