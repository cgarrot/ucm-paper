"""WS-C evaluation tests (PLAN WS-C, spec §9) on the REAL WS-A environment.

Uses ucm.env.tinygraph.TinyGraphKey + ucm.env.oracle.LayoutOracle (M0 landed):
closed-loop rollouts, all five baselines, metrics and the paired hierarchical
bootstrap, latency bench smoke.
"""

import random

import mlx.core as mx
import numpy as np
import pytest

from ucm.env.oracle import LayoutOracle
from ucm.env.tinygraph import TinyGraphKey
from ucm.eval import baselines as B
from ucm.eval import metrics as M
from ucm.eval.rollout import HORIZON, EpisodeResult, ModelPolicy, run_episode
from ucm.model import fixtures as F
from ucm.model.deepsets_a import DeepSetsA


def make_task(rng: random.Random, d_min: int = 0, d_max: int = 12,
              rooms: tuple[int, int] = (4, 8)):
    """Solvable task on the REAL env with d* in band. Returns
    (layout, task, solve) where solve = oracle dict on the initial state."""
    for _ in range(500):
        lay = F.make_layout(rng, rng.randint(*rooms))
        t = F.make_task(rng, lay, d_star_band=(d_min, d_max))
        if t is not None:
            task, solve = t
            return lay, task, solve
    raise RuntimeError("no task found")


def make_env(lay, task) -> TinyGraphKey:
    env = TinyGraphKey(lay)
    env.reset(task)
    return env


def replanning_oracle(lay):
    """solve_fn(obs) rebinding the true oracle onto the CURRENT env state
    (privileged bound arm; the env/true state are passed by closure)."""
    env_holder = {}

    def solve(obs):
        oracle = LayoutOracle(lay, lay_goal_holder["goal"])
        return oracle.solve(env_holder["env"].state)

    lay_goal_holder = {}
    return solve, env_holder, lay_goal_holder


# ---------------------------------------------------------------------------
# rollouts
# ---------------------------------------------------------------------------

class TestRollout:
    def test_oracle_policy_succeeds_optimally(self):
        rng = random.Random(5)
        for k in range(10):
            lay, task, solve = make_task(rng)
            env = make_env(lay, task)
            oracle = LayoutOracle(lay, task["goal"])

            def policy(obs, _o=oracle):
                return obs["candidates"][_o.solve(env.state)["optimal_actions"][0]]

            res = run_episode(env, policy, f"e{k}", seed=0, policy_name="oracle",
                              d_star=solve["d_star"])
            assert res.success and res.outcome == "success"
            assert res.length == solve["d_star"] + 1  # L = L* = d*+1, STOP included
            assert res.n_invalid == 0

    def test_random_syntactic_respects_horizon(self):
        rng = random.Random(11)
        lay, task, solve = make_task(rng)
        env = make_env(lay, task)
        res = run_episode(env, B.make_random_syntactic(seed=1), "e", seed=0,
                          policy_name="rand_syn", d_star=solve["d_star"])
        assert res.length <= HORIZON
        if res.outcome == "timeout":
            assert not res.success and res.length == HORIZON

    def test_premature_stop_is_failure(self):
        rng = random.Random(13)
        lay, task, solve = make_task(rng, d_min=2)  # goal NOT satisfied initially
        env = make_env(lay, task)
        res = run_episode(env, lambda obs: {"action": "STOP", "arg": None}, "e", seed=0,
                          policy_name="stopper", d_star=solve["d_star"])
        assert res.outcome == "premature_stop" and not res.success

    def test_satisfied_task_stop_success(self):
        rng = random.Random(17)
        lay, task, solve = make_task(rng, d_min=0, d_max=0)  # A* = {STOP}
        assert solve["d_star"] == 0
        env = make_env(lay, task)
        res = run_episode(env, lambda obs: {"action": "STOP", "arg": None}, "e", seed=0,
                          policy_name="stopper", d_star=0)
        assert res.success and res.length == 1 and res.L_star == 1

    def test_goal_reached_without_stop_flagged(self):
        # walk the optimal physical plan (never STOP), then move until timeout
        rng = random.Random(19)
        lay, task, solve = make_task(rng, d_min=1, d_max=4)
        env = make_env(lay, task)
        oracle = LayoutOracle(lay, task["goal"])
        plan = []
        probe = TinyGraphKey(lay)
        probe.reset(task)
        while True:
            sol = oracle.solve(probe.state)
            if sol["d_star"] == 0:
                break
            act = probe.candidates()[sol["optimal_actions"][0]]
            plan.append(act)
            probe.execute(act)
        assert len(plan) >= 1
        steps = {"i": 0}

        def walker(obs):
            if steps["i"] < len(plan):
                a = plan[steps["i"]]
                steps["i"] += 1
                return a
            va = B.valid_actions(obs)
            mv = [a for a in va if a["action"] == "MOVE"]
            return mv[0] if mv else {"action": "STOP", "arg": None}

        res = run_episode(env, walker, "e", seed=0, policy_name="walker",
                          d_star=solve["d_star"])
        assert res.outcome == "timeout" and not res.success
        assert res.goal_reached_without_stop  # physically reached, never STOPed

    def test_interface_violation_visible(self):
        rng = random.Random(23)
        lay, task, _ = make_task(rng)
        env = make_env(lay, task)
        with pytest.raises(ValueError):
            run_episode(env, lambda obs: {"action": "TELEPORT", "arg": None}, "e",
                        seed=0, policy_name="bad")


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------

class TestBaselines:
    def test_valid_actions_never_invalid(self):
        rng = random.Random(31)
        for k in range(12):
            lay, task, solve = make_task(rng)
            env = make_env(lay, task)
            res = run_episode(env, B.make_random_valid(seed=3), f"b{k}", seed=0,
                              policy_name="random_valid", d_star=solve["d_star"])
            assert res.n_invalid == 0  # precondition-informed arm

    def test_local_heuristic_solves_satisfied_tasks(self):
        rng = random.Random(37)
        lay, task, solve = make_task(rng, d_min=0, d_max=0)
        env = make_env(lay, task)
        res = run_episode(env, B.LocalHeuristic(seed=0), "b", seed=0,
                          policy_name="local_heuristic", d_star=0)
        assert res.success and res.length == 1

    def test_bfs_oracle_bound_always_optimal(self):
        rng = random.Random(41)
        for k in range(6):
            lay, task, solve = make_task(rng)
            env = make_env(lay, task)
            oracle = LayoutOracle(lay, task["goal"])

            def solve_fn(obs):
                return oracle.solve(env.state)

            res = run_episode(env, B.make_bfs_oracle_policy(solve_fn), f"b{k}", seed=0,
                              policy_name="bfs_oracle_bound", d_star=solve["d_star"])
            assert res.success and res.length == solve["d_star"] + 1
            assert res.n_invalid == 0

    def test_knn_memorizes_seen_signatures(self):
        rng = random.Random(43)
        train = F.make_records(rng, 10)
        rec = train[0]
        knn = B.KNNRetrievalPolicy(train, seed=0)
        obs = rec["policy_input"]
        act = knn(obs)
        assert act in obs["candidates"]
        opt_idx = rec["supervision"]["optimal_actions"]
        assert obs["candidates"].index(act) in opt_idx  # memorized arm hits A*
        # unknown signature → declared fallback (still a valid candidate)
        lay, task, _ = make_task(random.Random(99), d_min=1)
        obs2 = TinyGraphKey(lay).reset(task)
        act2 = knn(obs2)
        assert act2 in obs2["candidates"]

    def test_policies_always_return_candidates(self):
        rng = random.Random(47)
        lay, task, _ = make_task(rng)
        obs = TinyGraphKey(lay).reset(task)
        for pol in (B.make_random_syntactic(0), B.make_random_valid(0),
                    B.LocalHeuristic(0)):
            assert pol(obs) in obs["candidates"]

    def test_random_valid_never_unlocks_without_key(self):
        # spot-check precondition derivation on a locked-door layout
        rng = random.Random(53)
        for _ in range(20):
            lay, task, _ = make_task(rng)
            obs = TinyGraphKey(lay).reset(task)
            va = B.valid_actions(obs)
            carried = next((r["subj"] for r in obs["relations"] if r["pred"] == "held"), None)
            for a in va:
                if a["action"] == "UNLOCK":
                    unlock_ok = next(
                        (r["subj"] for r in obs["relations"] if r["pred"] == "unlocks"
                         and r["obj"] == a["arg"]), None)
                    assert carried == unlock_ok  # key actually held


# ---------------------------------------------------------------------------
# metrics §9.2/§9.3
# ---------------------------------------------------------------------------

def ep(success, length, L_star, seed, layout, eid, invalid=0, outcome=None, goal_ns=False):
    return EpisodeResult(episode_id=eid, seed=seed, policy_name="x",
                         success=success, outcome=outcome or ("success" if success else "timeout"),
                         length=length, n_invalid=invalid, goal_reached_without_stop=goal_ns,
                         d_star=(L_star - 1 if L_star else None), L_star=L_star,
                         layout_id=layout)


class TestMetrics:
    def test_summarize_math(self):
        eps = [
            ep(True, 5, 5, 0, "l1", "t1"),
            ep(True, 9, 5, 0, "l1", "t2", invalid=4),
            ep(False, 64, 5, 0, "l2", "t3", outcome="timeout"),
            ep(False, 3, 5, 0, "l2", "t4", outcome="premature_stop"),
        ]
        s = M.summarize(eps)
        assert s["n"] == 4
        assert s["success_rate"] == 0.5
        assert s["timeout_rate"] == 0.25 and s["premature_stop_rate"] == 0.25
        assert s["mean_regret_success"] == 2.0
        # truncated cost: L on success, H+1 on EVERY failure (explicit formula §9.2)
        assert s["mean_truncated_cost"] == (5 + 9 + (HORIZON + 1) + (HORIZON + 1)) / 4
        assert s["invalid_rate"] == 4 / (5 + 9 + 64 + 3)
        assert s["mean_length_success"] == 7.0

    def test_single_arm_bootstrap_contains_point(self):
        eps = ([ep(i % 2 == 0, 4, 4, seed=0, layout=f"l{i}", eid=f"t{i}") for i in range(10)]
               + [ep(i % 3 == 0, 4, 4, seed=1, layout=f"l{i}", eid=f"t{i}") for i in range(10)])
        ci = M.hierarchical_bootstrap(eps, n_boot=2000, seed=0)
        assert ci["ci_low"] <= ci["point"] <= ci["ci_high"]
        assert ci["n_seeds"] == 2 and ci["n_episodes"] == 20

    def test_paired_bootstrap_identical_arms_zero_diff(self):
        eps = [ep(i % 2 == 0, 4, 4, seed=0, layout=f"l{i}", eid=f"t{i}") for i in range(12)]
        out = M.paired_hierarchical_bootstrap(eps, list(eps), n_boot=1000, seed=1)
        assert abs(out["point_diff"]) < 1e-12
        assert out["ci_low"] <= 0.0 <= out["ci_high"]
        assert not out["ci_excludes_zero"]

    def test_paired_bootstrap_convention_first_minus_second(self):
        """Convention lock (audit tagi-5 B1): point_diff = mean(FIRST arm) −
        mean(SECOND arm). GATE-5 criterion 1 calls paired(B, A) so that
        positive = B better."""
        a = [ep(True, 4, 4, seed=0, layout=f"l{i}", eid=f"t{i}") for i in range(10)]
        b = [ep(False, 64, 4, seed=0, layout=f"l{i}", eid=f"t{i}", outcome="timeout")
             for i in range(10)]
        ab = M.paired_hierarchical_bootstrap(a, b, n_boot=500, seed=0)   # A−B
        ba = M.paired_hierarchical_bootstrap(b, a, n_boot=500, seed=0)   # B−A
        assert ab["point_diff"] == 1.0 and ab["ci_low"] > 0
        assert ba["point_diff"] == -1.0 and ba["ci_high"] < 0

    def test_task_level_resampled_bootstrap_width(self):
        """Audit tagi-5 M2: tasks within layouts are resampled — sample size is
        #episodes (bootstrap CIs reflect task-level variance too)."""
        eps = []
        for lay in ("l1", "l2"):
            for t in range(6):
                eps.append(ep(t % 2 == 0, 4, 4, seed=0, layout=lay, eid=f"{lay}-t{t}"))
        ci_full = M.hierarchical_bootstrap(eps, n_boot=3000, seed=1)
        assert ci_full["n_episodes"] == 12
        rng = random.Random(0)
        tree = M._clusters(eps)
        sample = M._resample(tree, rng)
        assert len(sample) == 12  # episode count, not layout count

    def test_paired_bootstrap_detects_difference(self):
        a = [ep(True, 4, 4, seed=s, layout=f"l{i}", eid=f"t{i}")
             for s in range(3) for i in range(10)]
        b = [ep(False, 64, 4, seed=s, layout=f"l{i}", eid=f"t{i}", outcome="timeout")
             for s in range(3) for i in range(10)]
        out = M.paired_hierarchical_bootstrap(a, b, n_boot=1000, seed=2)
        assert out["point_diff"] == 1.0
        assert out["ci_low"] > 0.0 and out["ci_excludes_zero"]

    def test_paired_bootstrap_requires_same_episodes(self):
        a = [ep(True, 4, 4, seed=0, layout="l1", eid="t1")]
        b = [ep(True, 4, 4, seed=0, layout="l1", eid="tX")]
        with pytest.raises(ValueError):
            M.paired_hierarchical_bootstrap(a, b)

    def test_astar_stratification(self):
        out = M.optimal_action_rate_by_astar([
            ([3.0, 1.0, 0.0], [1, 0, 0]),   # |A*|=1 hit
            ([1.0, 3.0, 0.0], [1, 0, 0]),   # |A*|=1 miss
            ([0.0, 3.0, 3.0], [0, 1, 1]),   # |A*|=2 hit
        ])
        assert out["|A*|=1"]["rate"] == 0.5 and out["|A*|=1"]["n"] == 2
        assert out["|A*|=2"]["rate"] == 1.0 and out["|A*|=2"]["n"] == 1


# ---------------------------------------------------------------------------
# latency bench §10.2 (small-n smoke; the full DoD bench runs at M1)
# ---------------------------------------------------------------------------

class TestTiming:
    def test_bench_structure_and_eval_discipline(self):
        from ucm.eval.timing import bench_decisions
        mx.set_default_device(mx.cpu)
        model = DeepSetsA()
        mx.eval(model.parameters())
        rng = random.Random(0)
        obs_list = [TinyGraphKey(l).reset(t)
                    for (l, t, _) in (make_task(rng, rooms=(6, 9)) for _ in range(4))]
        rep = bench_decisions(obs_list, model, warmup=5, n_measured=50)
        for stage in ("model_only", "tensorize", "selection", "full_decision"):
            assert rep[stage]["n"] == 50 and rep[stage]["p50_ms"] >= 0.0
        assert rep["runtime"]["mlx"].startswith("0.")
        assert isinstance(rep["gates"]["p95_model_le_20ms"], bool)

    def test_model_policy_returns_candidates(self):
        mx.set_default_device(mx.cpu)
        model = DeepSetsA()
        mx.eval(model.parameters())
        rng = random.Random(9)
        lay, task, _ = make_task(rng)
        obs = TinyGraphKey(lay).reset(task)
        pol = ModelPolicy(model, seed=0)
        assert pol(obs) in obs["candidates"]

    def test_model_policy_randomized_order_maps_actions_correctly(self):
        """Regression (closed-loop bug, WS-C): under randomize_order=True the
        returned action must be the argmax of the SHUFFLED list — i.e. exactly
        what an unrandomized run would return, up to exact ties."""
        mx.set_default_device(mx.cpu)
        from ucm.model.tensorize import tensorize_obs, collate
        model = DeepSetsA()
        mx.eval(model.parameters())
        rng = random.Random(21)
        for _ in range(5):
            lay, task, solve = make_task(rng)
            obs = TinyGraphKey(lay).reset(task)
            pol_on = ModelPolicy(model, seed=3, randomize_order=True)
            act_on = pol_on(obs)
            assert act_on in obs["candidates"]
            # unrandomized argmax for comparison
            ex = tensorize_obs(obs)
            ex["labels"] = None
            lg = model(collate([ex]))[0]
            mx.eval(lg)
            best = float(mx.max(lg).item())
            # action returned must score max (exact-tie tolerance)
            idx = obs["candidates"].index(act_on)
            assert abs(float(lg[idx].item()) - best) < 1e-6
