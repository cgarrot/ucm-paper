"""NULL control source generator (lead 14:53 arbitrage, re-vérifié 15:16).

Labels are uniformly random among ALL candidates (valid AND invalid) —
NO validity filter, NO information about which actions are physically
possible. This is the SEMANTIC point of the null arm: it removes both
the decisional AND the validity-structure information from the source
pretraining, while keeping the input geometry and action space interface.

Contrast with control_source.py (validity-informed): uniform among VALID
actions only — retains validity structure, removes decisional information.
"""

from __future__ import annotations

import json
import random

from ucm.v1.transfer import couple_hash


def make_null_records(canon_records: list[dict]) -> list[dict]:
    """For each canon record: label = one candidate drawn uniformly at random
    from the COMPLETE candidate list (no validity filter). rng seeded by
    couple hash — deterministic, order-independent."""
    out = []
    for rec in canon_records:
        pi = rec["policy_input"]
        rng = random.Random(int(couple_hash(
            {"layout": rec["provenance"].get("layout_hash", ""),
             "state": rec["provenance"].get("state_goal_hash", ""),
             "goal": pi["goal"]}), 16))
        # UNIFORM AMONG ALL — no valid_actions() filter (semantic point)
        idx = rng.randrange(len(pi["candidates"]))
        new = json.loads(json.dumps(rec))
        new["supervision"] = {"optimal_actions": [idx],
                              "d_star": None, "reachable": True,
                              "control_note": "null: uniform among ALL candidates "
                                               "(no validity filter)"}
        out.append(new)
    return out


def null_label_distribution(records: list[dict]) -> dict:
    """Proof: distribution of valid/invalid labels (for the artifact)."""
    from ucm.eval.baselines import valid_actions
    valid = 0
    invalid = 0
    for rec in records:
        pi = rec["policy_input"]
        idx = rec["supervision"]["optimal_actions"][0]
        if pi["candidates"][idx] in valid_actions(pi):
            valid += 1
        else:
            invalid += 1
    return {"total": valid + invalid, "valid_labels": valid,
            "invalid_labels": invalid,
            "invalid_fraction": invalid / (valid + invalid) if (valid + invalid) else None}
