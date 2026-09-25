"""M1 diagnostic report runner (WS-C): closed-loop rollouts of the overfit
checkpoint + baselines + oracle bound + latency, on the exact 100-episode
diagnostic set (seed 1007). Writes artifacts/m1-report/."""

import glob
import json
import random
import sys

import mlx.core as mx
import mlx.nn as nn

mx.set_default_device(mx.cpu)

from ucm.data import generate as gen
from ucm.env.oracle import LayoutOracle
from ucm.env.tinygraph import TinyGraphKey
from ucm.eval import baselines as B
from ucm.eval import metrics as M
from ucm.eval.rollout import ModelPolicy, run_episode
from ucm.eval.timing import bench_and_save
from ucm.model.deepsets_a import DeepSetsA

SEED = 1007
RUN_EVAL_SEED = 0


def build_diagnostic(n_episodes=100):
    """Regenerate the diagnostic episodes + task descriptors (deterministic)."""
    rng = random.Random(SEED)
    stats = gen.GenerationStats()
    tasks = []
    hist = {}
    while stats.episodes < n_episodes:
        lay = gen.generate_layout(rng, rng.randint(4, 8), 0.4)
        t_ep = []
        recs = gen.generate_episode(rng, lay, (2, 12), stats, True, {}, tasks_out=t_ep)
        if not recs:
            continue
        d0, goal = t_ep[0]["d_star"], t_ep[0]["goal"]
        obs0 = recs[0]["policy_input"]
        loc = {r["subj"]: r["obj"] for r in obs0["relations"] if r["pred"] == "at"}
        carried = next((r["subj"] for r in obs0["relations"] if r["pred"] == "held"), None)
        locked = next((e["attrs"]["locked"] for e in obs0["entities"] if e["type"] == "door"), None)
        init = {"agent": loc["agent"], "carried": carried,
                "key": None if carried == "key" else loc.get("key"),
                "parcel": None if carried == "parcel" else loc.get("parcel"),
                "door_locked": locked}
        tasks.append((f"ep{len(tasks)}", {"init": init, "goal": goal}, lay, d0))
        hist[str(d0)] = hist.get(str(d0), 0) + 1
    return tasks, hist


def main():
    run = sorted(glob.glob("artifacts/*m1-overfit-100ep"))[-1]  # lr 3e-4 run
    params = mx.load(run + "/checkpoint.npz")
    model = DeepSetsA()
    model.update(nn.utils.tree_unflatten(list(params.items())))
    mx.eval(model.parameters())
    tasks, hist = build_diagnostic()

    arms = {
        "modelA": ModelPolicy(model, name="modelA", seed=RUN_EVAL_SEED),
        "random_syntactic": B.make_random_syntactic(RUN_EVAL_SEED),
        "random_valid": B.make_random_valid(RUN_EVAL_SEED),
        "local_heuristic": B.LocalHeuristic(RUN_EVAL_SEED),
    }
    all_results = {}
    for name, pol in arms.items():
        results = []
        for (tid, task, lay, d0) in tasks:
            env = TinyGraphKey(lay)
            env.reset(task)
            r = run_episode(env, pol, tid, RUN_EVAL_SEED, name, d_star=d0)
            r.layout_id = lay.layout_hash()
            r.goal_type = task["goal"]["predicate"]
            results.append(r)
        all_results[name] = results
        s = M.summarize(results)
        print(f"{name:16s} success {s['success_rate']:.3f} regret {s['mean_regret_success']} "
              f"invalid {s['invalid_rate']:.3f} timeout {s['timeout_rate']:.3f} "
              f"goal_no_stop {s['goal_reached_without_stop_rate']:.3f}", flush=True)

    oracle_res = []
    for (tid, task, lay, d0) in tasks:
        env = TinyGraphKey(lay)
        env.reset(task)
        orc = LayoutOracle(lay, task["goal"])
        pol = B.make_bfs_oracle_policy(lambda obs: orc.solve(env.state))
        r = run_episode(env, pol, tid, RUN_EVAL_SEED, "bfs_oracle_bound", d_star=d0)
        r.layout_id = lay.layout_hash()
        oracle_res.append(r)

    paired = M.paired_hierarchical_bootstrap(all_results["modelA"], all_results["random_valid"],
                                             n_boot=2000, seed=0)

    obs_list = []
    for (tid, task, lay, d0) in tasks[:8]:
        env = TinyGraphKey(lay)
        obs_list.append(env.reset(task))
    trep = bench_and_save(obs_list, model, "artifacts/m1-report", warmup=50,
                          n_measured=1000, seed=0)

    report = {
        "diagnostic": {"episodes": len(tasks), "generator_seed": SEED,
                       "d_star_hist": hist, "checkpoint": run,
                       "note": "overfit diagnostic (train episodes themselves); GATE-1 target >=99% OA"},
        "arms": {n: M.summarize(r) for n, r in all_results.items()},
        "oracle_bound": M.summarize(oracle_res),
        "paired_model_vs_random_valid": paired,
        "latency": trep,
    }
    json.dump(report, open("artifacts/m1-report/m1-report.json", "w"), indent=2, default=str)
    print("\noracle bound success:", report["oracle_bound"]["success_rate"],
          "mean regret:", report["oracle_bound"]["mean_regret_success"])
    print("paired modelA − random_valid:", json.dumps(paired, default=str))
    print("p95 model ms:", trep["model_only"]["p95_ms"],
          "| p95 full ms:", trep["full_decision"]["p95_ms"], "| gates:", trep["gates"])


if __name__ == "__main__":
    main()
