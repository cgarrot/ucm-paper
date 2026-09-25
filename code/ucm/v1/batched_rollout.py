"""Batched closed-loop rollouts (lead 21:17, task (a)) — evaluation-only
optimization (~5-10× expected), for FUTURE evals; the in-flight M-V1b run is
untouched (sequential reference path).

Semantics contract:
    - randomize_order=False (deterministic policy): batched results are
      BITWISE equal to the sequential path (tested).
    - randomize_order=True: per-episode RNG streams (seed = episode seed +
      step), NOT a shared stream — distribution-equivalent, not bitwise
      (documented; the official/reference numbers come from the sequential
      path unless a re-freeze says otherwise).
    - Tie-breaking: per-episode seeded rng, same rule as ModelPolicy.
"""

from __future__ import annotations

import random

import mlx.core as mx

from ucm.eval.rollout import HORIZON, EpisodeResult
from ucm.model.tensorize import randomize_obs_order
from ucm.v1.tensorize_siw import collate_siw, tensorize_siw_obs


def batched_rollout(model, episodes: list[dict], build_env, seed: int,
                    policy_name: str, batch_size: int = 32,
                    horizon: int = HORIZON, randomize_order: bool = True,
                    sample_temperature: float | None = None) -> list[EpisodeResult]:
    """Run `episodes` (specs with episode_id/task/d_star/layout_hash) in
    lockstep micro-batches of `batch_size` active episodes.

    build_env(ep) → fresh env reset on its task (same as runner._build_env).
    """
    envs = {i: build_env(ep) for i, ep in enumerate(episodes)}
    active = list(range(len(episodes)))
    ep_rng = {i: random.Random(1000 + seed) for i in active}  # tie-break streams
    results = [None] * len(episodes)
    n_invalid = [0] * len(episodes)
    goal_seen = [False] * len(episodes)
    actions_log = [[] for _ in episodes]

    for _step in range(horizon):
        if not active:
            break
        # 1) collect observations (with per-episode order randomization)
        obs_map = {}
        for i in active:
            obs = envs[i].observe()
            if randomize_order:
                # per-episode, per-step stream (distribution-equivalent, doc'd)
                rng = random.Random(hash((i, seed, _step)) & 0xFFFFFFFF)
                obs = randomize_obs_order(obs, rng)
            obs_map[i] = obs
        # 2) batched forward (micro-batches to bound memory)
        picks = {}
        for s in range(0, len(active), batch_size):
            chunk_idx = active[s:s + batch_size]
            exs = []
            for i in chunk_idx:
                ex = tensorize_siw_obs(obs_map[i])
                ex["labels"] = None
                exs.append(ex)
            batch = collate_siw(exs)
            logits = model(batch)
            mx.eval(logits)
            # ONE host transfer per micro-batch (lead 21:26): the per-row
            # mx.max + K .item() loop forced K syncs × episodes — the very
            # anti-pattern the batching was meant to remove
            import numpy as np
            scores_mat = np.asarray(logits)
            for row, i in enumerate(chunk_idx):
                lg = scores_mat[row]
                cands = obs_map[i]["candidates"]
                if sample_temperature is not None:
                    # sampling path stays on device semantics (MLX categorical)
                    lgx = logits[row]
                    picks[i] = int(mx.random.categorical(lgx / sample_temperature).item())
                else:
                    mxval = lg.max()
                    ties = np.flatnonzero(lg == mxval).tolist()
                    picks[i] = ties[0] if len(ties) == 1 else ep_rng[i].choice(ties)
        # 3) execute + bookkeeping
        still = []
        for i in active:
            act = obs_map[i]["candidates"][picks[i]]
            res = envs[i].execute(act)
            actions_log[i].append(act)
            if not res["valid"]:
                n_invalid[i] += 1
            if envs[i].goal_satisfied():
                goal_seen[i] = True
            if res["terminal"]:
                ep = episodes[i]
                success = act["action"] == "STOP" and envs[i].goal_satisfied()
                outcome = ("success" if success else
                           "premature_stop" if act["action"] == "STOP" else "timeout")
                results[i] = EpisodeResult(
                    episode_id=ep["episode_id"], seed=seed, policy_name=policy_name,
                    success=success, outcome=outcome, length=_step + 1,
                    n_invalid=n_invalid[i],
                    goal_reached_without_stop=(goal_seen[i] and not success),
                    d_star=ep.get("d_star"),
                    L_star=(ep["d_star"] + 1) if ep.get("d_star") is not None else None,
                    layout_id=ep.get("layout_hash", ""),
                    goal_type=ep.get("goal_type", ""), actions=actions_log[i])
            else:
                still.append(i)
        active = still
    # horizon exhausted without STOP → timeout
    for i in active:
        ep = episodes[i]
        results[i] = EpisodeResult(
            episode_id=ep["episode_id"], seed=seed, policy_name=policy_name,
            success=False, outcome="timeout", length=horizon, n_invalid=n_invalid[i],
            goal_reached_without_stop=goal_seen[i], d_star=ep.get("d_star"),
            L_star=(ep["d_star"] + 1) if ep.get("d_star") is not None else None,
            layout_id=ep.get("layout_hash", ""), goal_type=ep.get("goal_type", ""),
            actions=actions_log[i])
    return results
