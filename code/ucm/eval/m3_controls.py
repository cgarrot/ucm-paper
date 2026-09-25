"""M3 interface controls (spec §9.1) on the sealed canon, B144 primary.

Controls are BATCH-LEVEL ablations applied after collate (deterministic,
documented; the checkpoints are untouched):

    no_goal          — goal conditioning FULLY removed (lead directive: role
                       tags AND all goal binding): goal_refs → absent, goal
                       predicate one-hot → zeros. goal_vec becomes constant.
    counterfactual   — task goal swapped to another valid entity-referenced
                       goal; rollout judged against the COUNTERFACTUAL goal
                       (oracle d* recomputed). A goal-following policy must
                       transfer.
    candidates_only  — state stripped: node features zeroed AND relations
                       removed (msg_mask=0); candidate structure remains.
    no_relations     — msg_mask=0 only (nodes + goal kept).

G5 permutation control lives in the metamorphic check (orders/ids permuted at
inference; episode-level success must be identical).
"""

from __future__ import annotations

import random

import mlx.core as mx
import numpy as np

from ucm.eval import metrics as M
from ucm.eval.gate5_report import (CANON, load_canon_episodes, load_model,
                                   rebuild_layout, task_from_record)
from ucm.eval.rollout import EpisodeResult, ModelPolicy, run_episode
from ucm.env.oracle import LayoutOracle
from ucm.env.tinygraph import TinyGraphKey
from ucm.model.tensorize import collate, tensorize_obs


def ablate_batch(batch: dict, control: str) -> dict:
    b = dict(batch)
    if control == "no_goal":
        z2 = mx.zeros_like(b["goal_refs"])
        b["goal_refs"] = z2 - 1                       # refs absent → tags 0, uref 0
        b["goal_ref_present"] = mx.zeros_like(b["goal_ref_present"])
        b["goal_pred"] = mx.full_like(b["goal_pred"], 3)   # out-of-vocab → one-hot all zeros
    elif control == "candidates_only":
        b["nodes"] = mx.zeros_like(b["nodes"])
        b["msg_mask"] = mx.zeros_like(b["msg_mask"])
    elif control == "no_relations":
        b["msg_mask"] = mx.zeros_like(b["msg_mask"])
    else:
        raise ValueError(control)
    return b


class ControlPolicy(ModelPolicy):
    """ModelPolicy with a batch-level ablation (deterministic seed discipline)."""

    def __init__(self, model, control: str, **kw):
        super().__init__(model, **kw)
        self.control = control

    def logits(self, obs: dict) -> mx.array:
        ex = tensorize_obs(obs)
        ex["labels"] = None
        if self.control == "counterfactual":
            # task-level control: env reset with the swapped goal; NO batch ablation
            batch = collate([ex])
        else:
            batch = ablate_batch(collate([ex]), self.control)
        out = self.model(batch)[0]
        mx.eval(out)
        return out


def counterfactual_task(recs, rng: random.Random) -> dict:
    """Swap the episode goal to another valid entity-referenced goal (same
    layout, valid per env validation)."""
    obs0 = recs[0]["policy_input"]
    pred, args = obs0["goal"]["predicate"], dict(obs0["goal"]["args"])
    rooms = [e["id"] for e in obs0["entities"] if e["type"] == "room"]
    if pred == "REACH":
        other = [r for r in rooms if r != args["room"]]
        args = {"room": rng.choice(other) if other else args["room"]}
    elif pred == "HAVE":
        args = {"object": "parcel" if args["object"] == "key" else "key"}
    else:  # AT
        if rng.random() < 0.5:
            args = {"object": "parcel" if args["object"] == "key" else "key",
                    "room": args["room"]}
        else:
            other = [r for r in rooms if r != args["room"]]
            args = {"object": args["object"],
                    "room": rng.choice(other) if other else args["room"]}
    task = task_from_record(recs)
    return {"init": task["init"], "goal": {"predicate": pred, "args": args}}


def run_control(model, episodes, control: str, seed: int = 42,
                arm: str = "greedy", limit: int | None = None):
    eps = episodes if limit is None else episodes[:limit]
    lay_cache = {}
    results = []
    rng = random.Random(seed)
    for (ref, lay_h, d0, recs) in eps:
        if control == "counterfactual":
            lay = lay_cache.get(lay_h)
            if lay is None:
                lay = rebuild_layout(recs)
                lay_cache[lay_h] = lay
            task = counterfactual_task(recs, rng)
            env = TinyGraphKey(lay)
            env.reset(task)
            orc = LayoutOracle(lay, task["goal"])
            d0 = orc.solve(env.state)["d_star"]
        else:
            lay = lay_cache.get(lay_h)
            if lay is None:
                lay = lay_cache[lay_h] = rebuild_layout(recs)
            env = TinyGraphKey(lay)
            env.reset(task_from_record(recs))
        kw = {"sample": True, "temperature": float(arm.replace("tau", ""))} if arm.startswith("tau") else {}
        pol = ControlPolicy(model, control, name=f"B144-{control}", seed=seed, **kw)
        r = run_episode(env, pol, f"c-{ref}", seed, f"B144-{control}", d_star=d0)
        r.layout_id = lay_h
        r.seed = seed
        r.goal_type = recs[0]["policy_input"]["goal"]["predicate"]
        results.append(r)
    return results


def main(run_dir: str | None = None, limit: int | None = None):
    import glob as _glob
    run_dir = run_dir or sorted(_glob.glob("artifacts/*gate2c-B144-s2"))[-1]
    model, _ = load_model(run_dir)
    g1_eps = [e for e in load_canon_episodes(CANON, split="test_g1")
              if min(r["provenance"].get("step", 0) for r in e[3]) == 0]
    report = {"checkpoint": run_dir, "cell": "test_g1", "n": limit or len(g1_eps)}
    for control in ("no_goal", "counterfactual", "candidates_only", "no_relations"):
        res = run_control(model, g1_eps, control, limit=limit)
        report[control] = M.summarize(res)
        print(control, "→", report[control]["success_rate"], flush=True)

    # G5 permutation metamorphic control: 20 coherent permutations per episode
    rng = random.Random(0)
    from ucm.model.tensorize import randomize_obs_order
    n_same, n_tot = 0, 0
    for (ref, lay_h, d0, recs) in g1_eps[:10]:
        lay = rebuild_layout(recs)
        task = task_from_record(recs)
        base_pol = ModelPolicy(model, name="B144", seed=42)
        env = TinyGraphKey(lay)
        env.reset(task)
        r_base = run_episode(env, base_pol, f"p-{ref}", 42, "B144", d_star=d0)
        for t in range(20):
            pol = ModelPolicy(model, name="B144", seed=42 + t)
            pol._perm_rng = random.Random(100 + t)  # order permutation source
            _orig = pol.randomize_order
            pol.randomize_order = False
            env2 = TinyGraphKey(lay)
            obs = env2.reset(task)
            # manual randomized-order rollout
            from ucm.eval.rollout import _assert_candidate
            success = None
            perm_rng = random.Random(100 + t)
            outcome = None
            length = n_invalid = 0
            actions = []
            for _step in range(64):
                o2 = randomize_obs_order(obs, perm_rng)
                act = pol(o2)   # action dict from the SHUFFLED candidate list
                act_canon = next(c for c in obs["candidates"]
                                 if c["action"] == act["action"] and c["arg"] == act["arg"])
                res = env2.execute(act_canon)
                actions.append(act_canon)
                length += 1
                if not res["valid"]:
                    n_invalid += 1
                if env2.goal_satisfied():
                    goal_seen = True
                if res["terminal"]:
                    outcome = "success" if env2.goal_satisfied() else "premature_stop"
                    break
                obs = env2.observe()
            same = (outcome == r_base.outcome)
            n_tot += 1
            n_same += same
    report["g5_permutation_control"] = {"episodes": 10, "perms_per_episode": 20,
                                        "outcome_identical_rate": n_same / n_tot,
                                        "n": n_tot}
    print("g5 permutation →", report["g5_permutation_control"], flush=True)

    import json, os
    os.makedirs("artifacts/m3", exist_ok=True)
    with open("artifacts/m3/controls.json", "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    return report


if __name__ == "__main__":
    main()
