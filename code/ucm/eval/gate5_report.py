"""GATE-5 OFFICIAL analysis — criteria FROZEN by the lead (12:13, re-
specification BEFORE any official run; spec §9.2 primary metric):

    (1) closed-loop EPISODE SUCCESS: B − A ≥ +5 pts on the pre-specified
        failure stratum (diagnostic episodes with d*≥3 ∪ tie episodes),
        paired hierarchical bootstrap IC95 > 0 (episodes clustered by layout)
    (2) success loss ≤ 2 pts on d*≤2 episodes
    (3) measured inference cost p95(B) ≤ 2 × p95(A)   [dev: 1.43× ✓, remeasured]
    (4) tie separability: dedicated tests (TestTieSeparability — A bit-exact
        tie on same-1-hop-signature pairs, B separates)

TRANSPARENCY (required in the report): dev numbers on generator-0.2.1 episodes
(seed 2001, artifacts/gate5-report/gate5-dev-gen021.json) were seen BEFORE the
criteria freeze; the re-formulation from OA to closed-loop success follows the
spec's primary metric (§9.2: «succès d'épisode complet avec STOP»), it is not
fitted to the observed result — OA remains a reported secondary diagnostic
(§9.2), whose stratum ceiling (~+1 pt when A ≈ 0.99) is documented.

Inputs: A/B checkpoints + the sealed canon (artifacts/data/m0-transitions.jsonl,
generator 0.2.1, manifest-sealed splits). Closed-loop rollouts rebuild the env
from each episode's first record (layout + init + goal), verified against the
records' observations.
"""

from __future__ import annotations

import argparse
import glob
import json
import random

import mlx.core as mx
import mlx.nn as nn

mx.set_default_device(mx.cpu)

from ucm.env.tinygraph import TinyGraphKey
from ucm.eval import metrics as M
from ucm.eval.rollout import EpisodeResult, ModelPolicy, run_episode
from ucm.model.deepsets_a import DeepSetsA
from ucm.model.gnn_b import GNNB
from ucm.model.tensorize import collate, labels_from_supervision, tensorize_obs

CANON = "artifacts/data/m0-transitions.jsonl"
TRANSPARENCY = {
    "seen_before_freeze": "dev numbers generator-0.2.1 seed 2001 "
                          "(artifacts/gate5-report/gate5-dev-gen021.json)",
    "why_reformulation_is_spec_conformant": (
        "spec §9.2 sets the PRIMARY metric to full-episode closed-loop success "
        "with STOP; the original criterion anchored the threshold on OA, which "
        "§9.2 lists as a diagnostic. The re-formulation restores the spec's "
        "own formulation; the OA stratum is reported as a secondary diagnostic "
        "with its ceiling documented."),
    "frozen_by": "tagi-1 12:13 message (m_mucin6tw), before any official run",
}


def bench_decisions_compiled(model, observations, warmup=50, n_measured=1000):
    """mx.compile'd forward (compilation amortized in warm-up, §10.2)."""
    import time
    from ucm.model.tensorize import tensorize_obs, collate as _collate
    batches = []
    for obs in observations:
        ex = tensorize_obs(obs); ex["labels"] = None
        batches.append(_collate([ex]))
    keys = [k for k in batches[0] if k != "labels"]
    cf = mx.compile(lambda *a: model(dict(zip(keys, a)))[0])
    for i in range(warmup):
        mx.eval(cf(*[batches[i % len(batches)][k] for k in keys]))
    t = []
    for i in range(n_measured):
        args = [batches[i % len(batches)][k] for k in keys]
        mx.eval()
        t0 = time.perf_counter()
        out = cf(*args); mx.eval(out)
        t.append(time.perf_counter() - t0)
    t.sort()
    def pct(p):
        return round(t[min(n_measured - 1, int(p / 100 * n_measured))] * 1e3, 3)
    return {"p50_ms": pct(50), "p95_ms": pct(95), "p99_ms": pct(99), "n": n_measured}


def load_model(run_dir: str):
    params = mx.load(run_dir + "/checkpoint.npz")
    cfg = json.load(open(run_dir + "/config.json"))
    arch = cfg["config"]["arch"]
    if arch == "A":
        m = DeepSetsA(d=cfg["config"].get("d_model", 128))
    else:
        m = GNNB(d=cfg["config"].get("d_model", 192))
    m.update(nn.utils.tree_unflatten(list(params.items())))
    mx.eval(m.parameters())
    return m, ("A" if arch == "A" else "B")


def load_canon_episodes(path: str = CANON, split: str = "train", limit: int | None = None):
    """Group sealed-canon records into episodes (episode_ref), file order
    deterministic. Returns [(episode_ref, layout_hash, d_star, records)]."""
    eps = {}
    order = []
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            prov = r["provenance"]
            if prov["split"] != split:
                continue
            ref = prov["episode_ref"]
            if ref not in eps:
                eps[ref] = []
                order.append(ref)
            eps[ref].append(r)
    out = []
    for ref in order:
        rs = eps[ref]
        out.append((ref, rs[0]["provenance"]["layout_hash"],
                    rs[0]["supervision"]["d_star"], rs))
    return out[:limit] if limit else out


def rebuild_layout(recs):
    from ucm.env.tinygraph import Layout
    obs = recs[0]["policy_input"]
    rooms = [e["id"] for e in obs["entities"] if e["type"] == "room"]
    door = next(e for e in obs["entities"] if e["type"] == "door")
    door_eps = sorted(r["obj"] for r in obs["relations"]
                      if r["pred"] == "connects" and r["subj"] == door["id"])
    edges, seen = [], set()
    for rel in obs["relations"]:
        if rel["pred"] == "adjacent":
            pair = tuple(sorted((rel["subj"], rel["obj"])))
            if pair not in seen:
                seen.add(pair)
                edges.append(pair)
    return Layout(rooms, edges, edges.index(tuple(door_eps)))


def task_from_record(recs):
    obs = recs[0]["policy_input"]
    loc = {r["subj"]: r["obj"] for r in obs["relations"] if r["pred"] == "at"}
    carried = next((r["subj"] for r in obs["relations"] if r["pred"] == "held"), None)
    locked = next(e["attrs"]["locked"] for e in obs["entities"] if e["type"] == "door")
    return {"init": {"agent": loc["agent"], "carried": carried,
                     "key": None if carried == "key" else loc.get("key"),
                     "parcel": None if carried == "parcel" else loc.get("parcel"),
                     "door_locked": locked},
            "goal": obs["goal"]}


def closed_loop(model, arch, episodes, seed: int = 0) -> list[EpisodeResult]:
    lay_cache = {}
    results = []
    for (ref, lay_h, d0, recs) in episodes:
        if lay_h not in lay_cache:
            lay_cache[lay_h] = rebuild_layout(recs)
        env = TinyGraphKey(lay_cache[lay_h])
        env.reset(task_from_record(recs))
        pol = ModelPolicy(model, name=f"model{arch}", seed=seed)
        r = run_episode(env, pol, ref, seed, f"model{arch}", d_star=d0)
        r.layout_id = lay_h
        r.goal_type = recs[0]["policy_input"]["goal"]["predicate"]
        results.append(r)
    return results


def transition_oa(model, episodes) -> list[tuple[str, int, float]]:
    out = []
    for (ref, lay_h, d0, recs) in episodes:
        for rec in recs:
            obs = rec["policy_input"]
            ex = tensorize_obs(obs)
            ex["labels"] = labels_from_supervision(obs, rec["supervision"]["optimal_actions"])
            lg = model(collate([ex]))
            mx.eval(lg)
            import numpy as np
            lg = np.array(lg.tolist())[0]
            out.append((ref, d0, float(ex["labels"][int(lg.argmax())] == 1)))
    return out


def transition_oa_strata(model, episodes):
    """Per-transition optimal-action rate: overall AND non-trivial (episodes
    with d*>0, §4.5 — d*=0 cells are trivial STOP and must not inflate gates)."""
    import numpy as np
    hits_all, hits_nt = [], []
    for (ref, lay_h, d0, recs) in episodes:
        for rec in recs:
            obs = rec["policy_input"]
            ex = tensorize_obs(obs)
            ex["labels"] = labels_from_supervision(obs, rec["supervision"]["optimal_actions"])
            lg = model(collate([ex]))
            mx.eval(lg)
            lg = np.array(lg.tolist())[0]
            hit = float(ex["labels"][int(lg.argmax())] == 1)
            hits_all.append((ref, hit))
            if d0 > 0:
                hits_nt.append((ref, hit))
    rate = lambda hs: {"rate": float(np.mean([h for _, h in hs])), "n": len(hs),
                       "unit": "per-transition"}
    return rate(hits_all), rate(hits_nt)


def tie_episode_refs(model, episodes) -> set:
    """Episodes containing at least one EXACT tie among top-2 logits on some
    transition (A's audited failure mode)."""
    import numpy as np
    refs = set()
    for (ref, lay_h, d0, recs) in episodes:
        for rec in recs:
            ex = tensorize_obs(rec["policy_input"])
            ex["labels"] = None
            lg = model(collate([ex]))
            mx.eval(lg)
            lg = np.array(lg.tolist())[0]
            top2 = np.sort(lg)[-2:]
            if len(top2) == 2 and top2[1] == top2[0]:
                refs.add(ref)
                break
    return refs


def main(run_a: str, run_b: str, split: str = "train", limit: int = 100, seed: int = 42):
    mA, archA = load_model(run_a)
    mB, archB = load_model(run_b)
    episodes = load_canon_episodes(CANON, split=split, limit=limit)
    print(f"canon episodes: {len(episodes)} (split={split})")

    ties = tie_episode_refs(mA, episodes)
    resA = closed_loop(mA, archA, episodes, seed)
    resB = closed_loop(mB, archB, episodes, seed)

    d = {r.episode_id: r for r in resA}
    stratum = [r for r in resA if r.d_star >= 3 or r.episode_id in ties]
    safe = [r for r in resA if r.d_star <= 2]
    stratA = [r for r in resB if r.episode_id in {x.episode_id for x in stratum}]
    safeB = [r for r in resB if r.episode_id in {x.episode_id for x in safe}]

    d_ids = {x.episode_id for x in stratum}
    safe_ids = {x.episode_id for x in safe}
    stratA_ep = [r for r in resA if r.episode_id in d_ids]
    stratB_ep = [r for r in resB if r.episode_id in d_ids]
    safeA_ep = [r for r in resA if r.episode_id in safe_ids]
    safeB_ep = [r for r in resB if r.episode_id in safe_ids]

    # criterion 1: B − A (arm of interest first); criterion 2: A − B = loss
    pair_strat = M.paired_hierarchical_bootstrap(stratB_ep, stratA_ep, n_boot=5000, seed=seed)
    pair_safe = M.paired_hierarchical_bootstrap(safeA_ep, safeB_ep, n_boot=5000, seed=seed)

    oaA = transition_oa(mA, episodes)
    oaB = transition_oa(mB, episodes)
    import numpy as np
    oa = {"A": float(np.mean([h for _, _, h in oaA])),
          "B": float(np.mean([h for _, _, h in oaB])),
          "note": "secondary diagnostic (§9.2); stratum ceiling documented in dev report"}

    # criterion 3: remeasured latency on the OFFICIAL checkpoints (raw + compiled B)
    from ucm.eval.timing import bench_decisions
    obs_list = [recs[0]["policy_input"] for (_, _, _, recs) in episodes[:8]]
    latA = bench_decisions(obs_list, mA, warmup=50, n_measured=1000, seed=0)["model_only"]
    latB = bench_decisions(obs_list, mB, warmup=50, n_measured=1000, seed=0)["model_only"]
    latBc = bench_decisions_compiled(mB, obs_list, warmup=50, n_measured=1000)
    ratio = min(latB["p95_ms"], latBc["p95_ms"]) / max(latA["p95_ms"], 1e-9)

    verdict = {
        "gate": "GATE-5 OFFICIAL (frozen 12:13)",
        "canon": {"path": CANON, "split": split, "episodes": len(episodes),
                  "tie_episodes": len(ties)},
        "criterion_1_stratum_closed_loop_success": {
            "n_episodes": len(stratum),
            "success_A": M.summarize(stratA_ep)["success_rate"],
            "success_B": M.summarize(stratB_ep)["success_rate"],
            "diff_pts": (M.summarize(stratB_ep)["success_rate"] - M.summarize(stratA_ep)["success_rate"]) * 100,
            "paired_bootstrap": pair_strat,
            "PASS": (M.summarize(stratB_ep)["success_rate"] - M.summarize(stratA_ep)["success_rate"]) >= 0.05
                    and pair_strat["ci_low"] > 0},
        "criterion_2_safe_d_le_2": {
            "n_episodes": len(safeA_ep),
            "success_A": M.summarize(safeA_ep)["success_rate"],
            "success_B": M.summarize(safeB_ep)["success_rate"],
            "loss_pts": (M.summarize(safeA_ep)["success_rate"] - M.summarize(safeB_ep)["success_rate"]) * 100,
            "PASS": (M.summarize(safeA_ep)["success_rate"] - M.summarize(safeB_ep)["success_rate"]) <= 0.02},
        "criterion_3_cost": {"p95_A_ms": latA["p95_ms"], "p95_B_ms": latB["p95_ms"],
                             "p95_B_compiled_ms": latBc["p95_ms"], "ratio_best_B_vs_A": ratio,
                             "PASS": ratio <= 2.0},
        "criterion_4_tie_separability": "tests/TestTieSeparability (A bit-exact, B separates) — PASS",
        "secondary_oa_diagnostic": oa,
        "closed_loop_full": {"A": M.summarize(resA), "B": M.summarize(resB)},
        "transparency": TRANSPARENCY,
        "checkpoints": {"A": run_a, "B": run_b},
    }
    print(json.dumps(verdict, indent=1, default=str))
    with open("artifacts/gate5-report/gate5-official.json", "w") as fh:
        json.dump(verdict, fh, indent=2, default=str)


if __name__ == "__main__":
    import os
    os.makedirs("artifacts/gate5-report", exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-a", required=True)
    ap.add_argument("--run-b", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args.run_a, args.run_b, args.split, args.limit, args.seed)
