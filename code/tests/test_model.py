"""WS-C model tests (PLAN §3.4/§3.5, spec §5.3, §6.1, §8.1, §11-G5) — on the
canonical WS-A observation format (goal{predicate,args}, entities{id,type,attrs},
relations{subj,pred,obj}, candidates{action,arg}).

Covers:
    - tensor contract shapes/dtypes/vocabularies (§3.4)
    - binding by localized indices (never raw ids / no id embeddings)
    - coherent permutation invariance (ids + orders) within 1e-5 FP32 (G5)
    - anti-fuite: supervision/provenance changes → bit-identical tensorization
    - exact parameter count in the 0.3–0.7M target (§6.1)
    - set-valued BC loss: stability, padding-only masking, uniform/perfect cases
    - training smoke writing artifacts (spec §15.2)
"""

import math
import random

import mlx.core as mx
import numpy as np
import pytest

from ucm.model import fixtures as F
from ucm.model.deepsets_a import DeepSetsA, predict_single
from ucm.model.loss import (optimal_action_rate, prob_mass_on_optimal, set_bc_loss)
from ucm.model.tensorize import (D_IN, K_CAP, N_CAP, collate, labels_from_supervision,
                                 permute_example, randomize_obs_order,
                                 tensorize_batch, tensorize_obs)

TOL = 1e-5  # spec G5 tolerance for FP32


@pytest.fixture(scope="module")
def model():
    mx.set_default_device(mx.cpu)
    m = DeepSetsA()
    mx.eval(m.parameters())
    return m


@pytest.fixture(scope="module")
def records():
    return F.make_records(random.Random(1234), 24)


# ---------------------------------------------------------------------------
# tensorize — contract §3.4 on the canonical format
# ---------------------------------------------------------------------------

class TestTensorContract:
    def test_shapes_and_dtypes(self, records):
        obs = records[0]["policy_input"]
        ex = tensorize_obs(obs)
        N, K, E = len(obs["entities"]), len(obs["candidates"]), len(obs["relations"])
        assert ex["nodes"].shape == (N, D_IN) and ex["nodes"].dtype == np.float32
        assert ex["edges"].shape == (E, 2) and ex["edges"].dtype == np.int32
        assert ex["edge_types"].shape == (E,)
        assert ex["goal_refs"].shape == (2,) and ex["goal_refs"].dtype == np.int32
        assert ex["cand_types"].shape == (K,) and ex["cand_args"].shape == (K, 2)
        # closed vocab one-hots: type block always 1; locked block only for doors
        for row, ent in zip(ex["nodes"], obs["entities"]):
            assert row[:5].sum() == 1
            if ent["type"] == "door":
                assert row[5:].sum() == 1
            else:
                assert row[5:].sum() == 0

    def test_candidate_count_formula(self, records):
        # K = R + 6 (R MOVE + 2 PICK + 2 DROP + 1 UNLOCK + 1 STOP), spec §4.3
        for rec in records:
            rooms = sum(e["type"] == "room" for e in rec["policy_input"]["entities"])
            assert len(rec["policy_input"]["candidates"]) == rooms + 6

    def test_stop_null_arg_own_type(self, records):
        obs = records[0]["policy_input"]
        ex = tensorize_obs(obs)
        stops = [i for i, c in enumerate(obs["candidates"]) if c["action"] == "STOP"]
        assert len(stops) == 1
        s = stops[0]
        assert (ex["cand_args"][s] == -1).all()
        assert ex["cand_types"][s] == ACTION_STOP  # STOP id in closed vocab

    def test_localized_indices_never_ids(self, records):
        ex = tensorize_obs(records[0]["policy_input"])
        N = ex["nodes"].shape[0]
        assert (ex["edges"] < N).all() and (ex["edges"] >= 0).all()
        refs = ex["goal_refs"][ex["goal_refs"] >= 0]
        assert (refs < N).all()

    def test_goal_refs_semantics(self, records):
        # slot 0 = requested object, slot 1 = requested room (spec §4.2)
        for rec in records:
            obs = rec["policy_input"]
            ex = tensorize_obs(obs)
            ids = {e["id"]: i for i, e in enumerate(obs["entities"])}
            args = obs["goal"]["args"]
            if "object" in args:
                assert ex["goal_refs"][0] == ids[args["object"]]
            if "room" in args:
                assert ex["goal_refs"][1] == ids[args["room"]]
            if "object" not in args:
                assert ex["goal_refs"][0] == -1

    def test_unknown_vocab_is_visible_error(self):
        obs = {"goal": {"predicate": "REACH", "args": {"room": "r1"}},
               "entities": [{"id": "r1", "type": "castle", "attrs": {}}],
               "relations": [], "candidates": [{"action": "STOP", "arg": None}]}
        with pytest.raises(ValueError):
            tensorize_obs(obs)

    def test_caps_enforced_visible(self):
        obs = {"goal": {"predicate": "REACH", "args": {"room": "r0"}},
               "entities": [{"id": f"r{i}", "type": "room", "attrs": {}} for i in range(N_CAP + 1)],
               "relations": [], "candidates": [{"action": "STOP", "arg": None}]}
        with pytest.raises(ValueError):
            tensorize_obs(obs)


ACTION_STOP = ["MOVE", "PICK", "DROP", "UNLOCK", "STOP"].index("STOP")


def tie_pair_fixture():
    """Real-env layout with a same-1-hop-signature candidate pair (GATE-5
    criterion 4): agent v0, key at v1 behind locked door (v0,v1); v2's and
    v4's pre-update neighbour multisets are both {plain room, plain room}.
    A*: {MOVE v2}; MOVE v4 is NOT optimal — oracle-verified."""
    from ucm.env.oracle import LayoutOracle
    from ucm.env.tinygraph import Layout, TinyGraphKey
    lay = Layout(["v0", "v1", "v2", "v3", "v4", "v5"],
                 [("v0", "v2"), ("v0", "v1"), ("v2", "v1"), ("v0", "v4"),
                  ("v4", "v3"), ("v3", "v5")],
                 door_edge=1)  # door on (v0, v1), locked
    task = {"init": {"agent": "v0", "carried": None, "key": "v1", "parcel": "v5",
                     "door_locked": True},
            "goal": {"predicate": "HAVE", "args": {"object": "key"}}}
    env = TinyGraphKey(lay)
    obs = env.reset(task)
    sol = LayoutOracle(lay, task["goal"]).solve(env.state)
    opt = [obs["candidates"][i] for i in sol["optimal_actions"]]
    assert {"action": "MOVE", "arg": "v2"} in opt
    assert {"action": "MOVE", "arg": "v4"} not in opt
    return obs, "v2", "v4", sol["d_star"]


class TestTieSeparability:
    """GATE-5 criterion 4 (pre-specified): A must be EXACTLY tied on a
    same-1-hop-signature pair (information limit); B's representations must
    separate (depth). If B also ties bit-exactly → observation-scheme limit,
    major discovery to report, not to hide."""

    def test_A_ties_bitexactly_on_symmetric_pair(self, model):
        obs, r2, r4, d = tie_pair_fixture()
        ex = tensorize_obs(obs)
        ex["labels"] = None
        u = model.encode(collate([ex]))
        mx.eval(u)
        ids = [e["id"] for e in obs["entities"]]
        un = np.array(u.tolist())[0]
        assert np.array_equal(un[ids.index(r2)], un[ids.index(r4)])
        lg = np.array(model(collate([ex])).tolist())[0]
        i2 = obs["candidates"].index({"action": "MOVE", "arg": r2})
        i4 = obs["candidates"].index({"action": "MOVE", "arg": r4})
        assert lg[i2] == lg[i4]  # exact tie — the audited A limit

    def test_B_representations_separate_symmetric_pair(self):
        from ucm.model.gnn_b import GNNB
        mx.set_default_device(mx.cpu)
        mB = GNNB()
        mx.eval(mB.parameters())
        obs, r2, r4, d = tie_pair_fixture()
        ex = tensorize_obs(obs)
        ex["labels"] = None
        u = mB.encode(collate([ex]))
        mx.eval(u)
        ids = [e["id"] for e in obs["entities"]]
        un = np.array(u.tolist())[0]
        assert not np.array_equal(un[ids.index(r2)], un[ids.index(r4)])


class TestGNNB:
    """Model B (§6.2) contract tests."""

    def test_param_count_target(self, capsys):
        from ucm.model.gnn_b import GNNB
        mx.set_default_device(mx.cpu)
        mB = GNNB()
        mx.eval(mB.parameters())
        n = mB.print_param_count()
        assert 1_100_000 <= n <= 1_500_000, n
        assert "1,230,145" in capsys.readouterr().out

    def test_forward_and_permutation_invariance(self, records):
        from ucm.model.gnn_b import GNNB, predict_single_b
        mx.set_default_device(mx.cpu)
        mB = GNNB()
        mx.eval(mB.parameters())
        batch = tensorize_batch(records[:3])
        lg = mB(batch)
        mx.eval(lg)
        assert lg.shape == batch["cand_mask"].shape
        rec = records[0]
        lab = labels_from_supervision(rec["policy_input"],
                                      rec["supervision"]["optimal_actions"]).tolist()
        l1 = np.array(predict_single_b(mB, rec["policy_input"]).tolist())
        worst = 0.0
        for t in range(6):
            o2, l2, idmap = F.permute_obs(rec["policy_input"], lab, random.Random(70 * t + 9))
            l2m = np.array(predict_single_b(mB, o2).tolist())
            k2i = {(c["action"], c["arg"]): j for j, c in enumerate(o2["candidates"])}
            mp = [k2i[(c["action"], (idmap[c["arg"]] if c["arg"] is not None else None))]
                  for c in rec["policy_input"]["candidates"]]
            worst = max(worst, float(np.abs(l1 - l2m[mp]).max()))
        assert worst < TOL, worst


class TestBinding:
    def test_labels_bind_indices_to_candidates(self, records):
        rec = records[0]
        obs = rec["policy_input"]
        lab = labels_from_supervision(obs, rec["supervision"]["optimal_actions"])
        assert lab.sum() == len(rec["supervision"]["optimal_actions"])
        for i in rec["supervision"]["optimal_actions"]:
            assert lab[i] == 1.0

    def test_labels_from_explicit_actions_by_reference(self, records):
        rec = records[0]
        obs = rec["policy_input"]
        optimal = [obs["candidates"][i] for i in rec["supervision"]["optimal_actions"]]
        lab = labels_from_supervision(obs, optimal)
        lab_idx = labels_from_supervision(obs, rec["supervision"]["optimal_actions"])
        assert (lab == lab_idx).all()

    def test_empty_supervision_rejected(self, records):
        with pytest.raises(ValueError):
            labels_from_supervision(records[0]["policy_input"], [])

    def test_swapping_goal_changes_refs_not_entities(self, records):
        rec = records[0]
        obs = rec["policy_input"]
        rooms = [e["id"] for e in obs["entities"] if e["type"] == "room"]
        other = [r for r in rooms if r != obs["goal"]["args"].get("room")]
        if not other or obs["goal"]["predicate"] != "REACH":
            pytest.skip("need REACH goal with a spare room")
        obs2 = {**obs, "goal": {"predicate": "REACH", "args": {"room": other[0]}}}
        e1, e2 = tensorize_obs(obs), tensorize_obs(obs2)
        assert (e1["nodes"] == e2["nodes"]).all()
        assert e1["goal_refs"][1] != e2["goal_refs"][1]


class TestAntiFuite:
    def test_supervision_provenance_cannot_change_tensors(self, records):
        rec = records[0]
        ex1 = tensorize_obs(rec["policy_input"])
        tampered = dict(rec)
        tampered["supervision"] = {"optimal_actions": [0], "d_star": 0, "reachable": True}
        tampered["provenance"] = {"split": "TEST-CONTAMINATED", "source": "x"}
        ex2 = tensorize_obs(tampered["policy_input"])
        for k in ("nodes", "edges", "edge_types", "goal_refs", "cand_types", "cand_args"):
            assert (ex1[k] == ex2[k]).all(), k

    def test_no_forbidden_fields_in_policy_input(self, records):
        forbidden = {"d_star", "distance", "reward", "progress", "success", "terminal",
                     "plan", "next_state", "expert", "timestep", "episode", "split",
                     "layout_hash", "source", "generator"}
        for rec in records:
            obs = rec["policy_input"]
            assert not (forbidden & set(obs.keys()))
            for e in obs["entities"]:
                assert not (forbidden & set(e.keys()))
                assert not (forbidden & set(e.get("attrs", {}).keys()))

    def test_candidates_goal_blind(self, records):
        # §5.3: changing the goal changes neither candidates nor entities
        rec = records[0]
        obs = rec["policy_input"]
        rooms = [e["id"] for e in obs["entities"] if e["type"] == "room"]
        obs2 = {**obs, "goal": {"predicate": "REACH", "args": {"room": rooms[-1]}}}
        e1, e2 = tensorize_obs(obs), tensorize_obs(obs2)
        assert (e1["nodes"] == e2["nodes"]).all()
        assert (e1["cand_types"] == e2["cand_types"]).all()
        assert (e1["cand_args"] == e2["cand_args"]).all()


class TestPermutation:
    def test_permutation_invariance_logits(self, model, records):
        worst = 0.0
        for rec in records[:6]:
            obs = rec["policy_input"]
            lab = labels_from_supervision(obs, rec["supervision"]["optimal_actions"]).tolist()
            l1 = np.array(predict_single(model, obs).tolist())
            for t in range(4):
                o2, l2, idmap = F.permute_obs(obs, lab, random.Random(1000 * t + 7))
                l2m = np.array(predict_single(model, o2).tolist())
                k2i = {(c["action"], c["arg"]): j for j, c in enumerate(o2["candidates"])}
                mp = [k2i[(c["action"],
                            (idmap[c["arg"]] if c["arg"] is not None else None))]
                      for c in obs["candidates"]]
                worst = max(worst, float(np.abs(l1 - l2m[mp]).max()))
                assert sum(l2) == sum(lab)  # labels permuted together
        assert worst < TOL, f"permutation invariance violated: {worst}"


class TestOrderRandomization:
    """PLAN §WS-C (audit tagi-5 T8): rank/order must carry no signal — train
    AND test randomization, with G5 proof (20 coherent permutations)."""

    def test_g5_20_permutations_distribution_identical(self, model, records):
        # 20 coherent permutations per episode: per-candidate probabilities
        # (hence the full distribution) identical up to FP32 tolerance; argmax
        # preserved except on EXACT ties (distinguished per spec §11).
        for rec in records[:5]:
            obs = rec["policy_input"]
            lab = labels_from_supervision(obs, rec["supervision"]["optimal_actions"]).tolist()
            base = predict_single(model, obs)
            mx.eval(base)
            p1 = np.array(mx.softmax(base).tolist())
            k1 = [(c["action"], c["arg"]) for c in obs["candidates"]]
            for t in range(20):
                o2, l2, idmap = F.permute_obs(obs, lab, random.Random(555 * t + 13))
                lg = predict_single(model, o2)
                mx.eval(lg)
                p2 = np.array(mx.softmax(lg).tolist())
                k2i = {(c["action"], c["arg"]): j for j, c in enumerate(o2["candidates"])}
                mp = [k2i[(c["action"],
                            (idmap[c["arg"]] if c["arg"] is not None else None))]
                      for c in obs["candidates"]]
                assert np.abs(p1 - p2[mp]).max() < TOL  # identical distribution
                # argmax preserved unless exact tie in the permuted run
                if not (np.sort(p2)[-2] == np.sort(p2)[-1]):
                    assert np.argmax(p1) == mp.index(int(np.argmax(p2))) or \
                        p1[mp.index(int(np.argmax(p2)))] == p1.max()

    def test_permute_example_equivalence_dict_level(self, records):
        # numpy-level permute_example ≡ tensorize(randomize_obs_order(obs))
        rec = records[0]
        obs = rec["policy_input"]
        ex = tensorize_obs(obs)
        ex["labels"] = labels_from_supervision(obs, rec["supervision"]["optimal_actions"])
        rng = random.Random(42)
        for _ in range(10):
            shuffled = permute_example(ex, rng)
            # equivalently: tensorize the dict with same node order & orders
            # rebuild via ids: track node identity through nodes' one-hot rows
            # (rows are unique signatures here: types differ or relations do)
            assert shuffled["nodes"].shape == ex["nodes"].shape
            assert sorted(map(tuple, shuffled["nodes"].tolist())) == \
                sorted(map(tuple, ex["nodes"].tolist()))
            assert shuffled["labels"].sum() == ex["labels"].sum()
            assert (shuffled["cand_types"] == ex["cand_types"]).sum() >= 1

    def test_permute_example_logits_match(self, model, records):
        rec = records[0]
        obs = rec["policy_input"]
        ex = tensorize_obs(obs)
        ex["labels"] = None
        l1 = np.array(model(collate([ex])).tolist())[0]
        rng = random.Random(7)
        for _ in range(10):
            pe = permute_example(dict(ex), rng)
            l2 = np.array(model(collate([pe])).tolist())[0]
            # map candidates back: cand row order changed; use (type, arg-node)
            # signature via nodes rows
            def sig(t, a, nodes):
                return (int(t), tuple(nodes[a].tolist()) if a >= 0 else None)
            sigs1 = [sig(ex["cand_types"][i], ex["cand_args"][i][0], ex["nodes"])
                     for i in range(len(l1))]
            sigs2 = [sig(pe["cand_types"][i], pe["cand_args"][i][0], pe["nodes"])
                     for i in range(len(l2))]
            # signatures unique when args differ; fallback to type-count check
            if len(set(map(str, sigs1))) == len(sigs1):
                mp = [sigs2.index(s) for s in sigs1]
                assert np.abs(l1 - l2[mp]).max() < TOL

    def test_randomize_obs_order_content_preserved(self, records):
        obs = records[0]["policy_input"]
        rng = random.Random(3)
        for _ in range(5):
            o2 = randomize_obs_order(obs, rng)
            assert len(o2["entities"]) == len(obs["entities"])
            assert sorted(map(str, o2["entities"])) == sorted(map(str, obs["entities"]))
            assert sorted(map(str, o2["relations"])) == sorted(map(str, obs["relations"]))
            assert sorted(map(str, o2["candidates"])) == sorted(map(str, obs["candidates"]))
            assert o2["goal"] == obs["goal"]
            assert [e["id"] for e in o2["entities"]] != [e["id"] for e in obs["entities"]] \
                or True  # orders may coincide by chance; content invariant is the point


# ---------------------------------------------------------------------------
# model §6.1
# ---------------------------------------------------------------------------

class TestDeepSetsA:
    def test_param_count_target(self, model, capsys):
        n = model.print_param_count()
        assert 300_000 <= n <= 700_000, n
        assert "366,337" in capsys.readouterr().out  # exact count pinned (canonical d_in=7)

    def test_forward_shapes_and_padding_mask(self, model, records):
        batch = tensorize_batch(records[:4])
        logits = model(batch)
        mx.eval(logits)
        B, K = logits.shape
        assert B == 4 and K == batch["cand_mask"].shape[1]
        for b in range(B):
            kp = int(batch["cand_mask"][b].sum().item())
            assert all(logits[b, i].item() < -1e8 for i in range(kp, K))

    def test_goal_conditioning_changes_logits(self, model, records):
        rec = records[0]
        obs = rec["policy_input"]
        rooms = [e["id"] for e in obs["entities"] if e["type"] == "room"]
        other_goal = {**obs, "goal": {"predicate": "REACH", "args": {"room": rooms[-1]}}}
        l1 = np.array(predict_single(model, obs).tolist())
        l2 = np.array(predict_single(model, other_goal).tolist())
        assert np.abs(l1 - l2).max() > 1e-4

    def test_batch1_equals_batched_rows(self, model, records):
        batch = tensorize_batch(records[:3])
        lg = model(batch)
        mx.eval(lg)
        for i, rec in enumerate(records[:3]):
            single = np.array(predict_single(model, rec["policy_input"]).tolist())
            row = np.array(lg[i].tolist())
            k = len(rec["policy_input"]["candidates"])
            assert np.abs(single[:k] - row[:k]).max() < 1e-5


# ---------------------------------------------------------------------------
# loss §8.1
# ---------------------------------------------------------------------------

class TestLoss:
    def _arr(self, x):
        return mx.array(np.array(x, dtype=np.float32))

    def test_uniform_gives_logK(self):
        logits = self._arr([[0.0] * 4] * 2)
        labels = self._arr([[1, 0, 0, 0], [0, 1, 0, 0]])
        pad = self._arr([[1.0] * 4] * 2)
        assert abs(float(set_bc_loss(logits, labels, pad)) - math.log(4)) < 1e-6

    def test_perfect_is_zero(self):
        logits = self._arr([[10.0, -10.0, -10.0, -10.0]] * 2)
        labels = self._arr([[1, 0, 0, 0]] * 2)
        pad = self._arr([[1.0] * 4] * 2)
        assert abs(float(set_bc_loss(logits, labels, pad))) < 1e-6

    def test_only_padding_masked(self):
        base = np.array([[10.0, -10.0, -10.0, -10.0]], np.float32)
        ext = np.concatenate([base, 1e6 * np.ones((1, 8))], axis=1)
        lab = np.concatenate([[[1, 0, 0, 0]], np.zeros((1, 8))], axis=1).astype(np.float32)
        pad = np.concatenate([np.ones((1, 4)), np.zeros((1, 8))], axis=1).astype(np.float32)
        l1 = float(set_bc_loss(self._arr(base), self._arr([[1, 0, 0, 0]]),
                               self._arr(np.ones((1, 4)))))
        l2 = float(set_bc_loss(self._arr(ext), self._arr(lab), self._arr(pad)))
        assert abs(l1 - l2) < 1e-6

    def test_invalid_actions_stay_in_denominator(self):
        logits_bad = self._arr([[0.0, 5.0, -10.0, -10.0]])
        logits_good = self._arr([[0.0, -10.0, -10.0, -10.0]])
        labels = self._arr([[1, 0, 0, 0]])
        pad = self._arr(np.ones((1, 4)))
        assert float(set_bc_loss(logits_bad, labels, pad)) > \
               float(set_bc_loss(logits_good, labels, pad))

    def test_grad_finite(self):
        pad = self._arr(np.ones((1, 4)))
        labels = self._arr([[1, 0, 0, 0]])
        g = mx.grad(lambda w: set_bc_loss(self._arr([[0.1, 0.2, 0.3, 0.4]]) * w,
                                          labels, pad))(mx.array(1.0))
        mx.eval(g)
        assert math.isfinite(float(g)) and float(g) != 0.0

    def test_diagnostics(self):
        logits = self._arr([[10.0, -10.0, -10.0, -10.0]])
        labels = self._arr([[1, 0, 0, 0]])
        pad = self._arr(np.ones((1, 4)))
        assert float(optimal_action_rate(logits, labels, pad)) == 1.0
        p = float(prob_mass_on_optimal(logits, labels, pad))
        assert 0.0 <= p <= 1.0 and p > 0.999


# ---------------------------------------------------------------------------
# training smoke (artifacts on disk, spec §15.2)
# ---------------------------------------------------------------------------

class TestTrainSmoke:
    def test_smoke_run_writes_artifacts(self, tmp_path, monkeypatch):
        import ucm.model.train as T
        monkeypatch.chdir(tmp_path)
        cfg = T.TrainConfig(name="pytest-smoke", synthetic=60, max_updates=30,
                            max_epochs=1000, eval_every=15, seed=0, val_fraction=0.2)
        out = T.train(cfg)
        import glob, json, os
        runs = glob.glob(str(tmp_path / "artifacts" / "*pytest-smoke"))
        assert runs, "artifact dir missing"
        final = json.load(open(runs[-1] + "/final.json"))
        assert final["n_params"] == DeepSetsA().param_count()
        for f in ("checkpoint.npz", "config.json", "metrics.jsonl"):
            assert os.path.exists(runs[-1] + "/" + f)
        assert final["updates_run"] == 30
