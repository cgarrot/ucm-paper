"""Closed-loop evaluation harness (spec §9, PLAN WS-C) — canonical WS-A env.

The environment contract (§3.1 / ucm.env.tinygraph):
    reset(task) → Obs ; observe() → policy_input Obs
    candidates() → list[{"action","arg"}]        (canonical order, K = R+6)
    execute(action: dict) → {"valid": bool, "terminal": bool, "result": str, "step": int}
    goal_satisfied() / state_hash() — evaluator-side ONLY

Independent evaluator (§9.4): STOP is verified on the env NATIVE state via
goal_satisfied(); a policy/network claim is never an attestation. Invalid
actions and timeouts are counted, never silently corrected (§4.4):

    - STOP with goal true  → terminal success
    - STOP with goal false → terminal failure `premature_stop`
    - H=64 decisions without STOP → terminal failure `timeout`
    - invalid actions consume a step, count in length and invalid rate

A policy is any callable ``policy(obs) -> {"action", "arg"} `` whose returned
action must be one of ``obs["candidates"]`` (binding by reference).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

import mlx.core as mx

from ucm.model.tensorize import collate, randomize_obs_order, tensorize_obs

HORIZON = 64  # spec §4.4

Action = dict  # canonical {"action": str, "arg": str | None}


@dataclass
class EpisodeResult:
    episode_id: str
    seed: int
    policy_name: str
    success: bool
    outcome: str            # "success" | "premature_stop" | "timeout"
    length: int             # decisions incl. STOP (L)
    n_invalid: int
    goal_reached_without_stop: bool  # physical goal true at some point without correct STOP
    d_star: int | None = None
    L_star: int | None = None        # d* + 1 (if oracle available for the initial state)
    layout_id: str = ""
    goal_type: str = ""
    actions: list = field(default_factory=list)


def run_episode(env, policy: Callable[[dict], Action], episode_id: str,
                seed: int, policy_name: str, d_star: int | None = None,
                horizon: int = HORIZON) -> EpisodeResult:
    """One closed-loop episode after the caller has reset the env on its task."""
    obs = env.observe()
    n_invalid = 0
    goal_seen = False
    length = 0
    outcome = "timeout"
    success = False
    actions: list = []
    for _step in range(horizon):
        action = policy(obs)
        _assert_candidate(obs, action, episode_id)
        result = env.execute(action)
        actions.append(action)
        length += 1
        if not result["valid"]:
            n_invalid += 1
        if env.goal_satisfied():  # evaluator-side postcondition (native state)
            goal_seen = True
        if result["terminal"]:
            if action["action"] == "STOP":
                success = env.goal_satisfied()  # re-verified on native state (§9.4)
                outcome = "success" if success else "premature_stop"
            break
        obs = env.observe()
    return EpisodeResult(
        episode_id=episode_id, seed=seed, policy_name=policy_name,
        success=success, outcome=outcome, length=length, n_invalid=n_invalid,
        goal_reached_without_stop=(goal_seen and not success),
        d_star=d_star, L_star=(d_star + 1) if d_star is not None else None,
        actions=actions,
    )


def _assert_candidate(obs: dict, action: Action, episode_id: str) -> None:
    if action not in obs["candidates"]:
        raise ValueError(f"[{episode_id}] policy produced action {action!r} "
                         f"not present among candidates — interface violation")


# ---------------------------------------------------------------------------
# Policy adapter over Model A
# ---------------------------------------------------------------------------

class ModelPolicy:
    """Greedy argmax over model logits with seeded random tie-breaking among
    exactly-tied maxima (ties counted; spec §11/G5 distinguishes ties from
    equivariance violations). Sampling mode optional (temperature).

    randomize_order=True (default, PLAN §WS-C audit T8): entity/relation/
    candidate order of the incoming observation is randomized (seeded, ids
    kept) before tensorization — order/rank must not matter at inference."""

    def __init__(self, model, name: str = "modelA", seed: int = 0,
                 sample: bool = False, temperature: float = 1.0,
                 randomize_order: bool = True, validity_mask: bool = False):
        self.model = model
        self.name = name
        self.rng = random.Random(seed)
        self.sample = sample
        self.temperature = temperature
        self.randomize_order = randomize_order
        # validity_mask: DECLARED PRIVILEGED secondary arm (spec §9.1) — masks
        # logits to physically valid actions derived from OBSERVABLE preconditions.
        # Never compared to equal-information arms.
        self.validity_mask = validity_mask
        self.n_ties = 0

    def logits(self, obs: dict) -> mx.array:
        ex = tensorize_obs(obs)
        ex["labels"] = None
        batch = collate([ex])
        out = self.model(batch)[0]
        if self.validity_mask:
            from ucm.eval.baselines import valid_actions
            valid = {i for i, c in enumerate(obs["candidates"])
                     if c in valid_actions(obs)}
            mask = mx.array([[0.0 if i in valid else 1.0
                              for i in range(len(obs["candidates"]))]], mx.float32)
            out = out + mask[0] * -1e9
        mx.eval(out)
        return out

    def __call__(self, obs: dict) -> Action:
        if self.randomize_order:
            obs = randomize_obs_order(obs, self.rng)   # ids kept, orders shuffled
        lg = self.logits(obs)
        cands = obs["candidates"]  # SAME shuffled list the logits index into
        if self.sample:
            scaled = lg / self.temperature
            pick = int(mx.random.categorical(scaled).item())
        else:
            # single device→host transfer (lead 21:24: K .item() calls forced K
            # MLX synchronizations per decision); float equality is EXACT —
            # same values, same comparisons, no ordering change
            scores = lg.tolist()
            mxval = max(scores)
            ties = [i for i, v in enumerate(scores) if v == mxval]
            if len(ties) > 1:
                self.n_ties += 1
            pick = self.rng.choice(ties)
        return cands[pick]


def evaluate_policy(env_factory, tasks: list, policy: Callable, policy_name: str,
                    seed: int = 0, horizon: int = HORIZON) -> list[EpisodeResult]:
    """env_factory(task, seed) → fresh env already reset on the task.
    tasks: list of (task_id, task, layout_id, d_star) — d_star may be None."""
    results = []
    for (tid, task, layout_id, d_star) in tasks:
        env = env_factory(task, seed)
        results.append(run_episode(env, policy, tid, seed, policy_name,
                                   d_star=d_star, horizon=horizon))
    return results
