"""Mock SIW target (DEV SCAFFOLDING — the real SIW env/fixtures come from the
lead; per FREEZE.md the architecture is frozen independently of target
content). Exposes the SAME coupling interface the harness expects:

    couple = {"layout": <canonical layout id/desc>, "state": <physical state
              descriptor>, "goal": {"predicate", "args"}}
    episode rollouts via a tiny deterministic state machine (typed widgets).

Purpose: end-to-end tests of budgets / arms / coverage / finetune / one-read
WITHOUT touching any real target data (interactions log stays empty for the
real target). No scientific claim is ever made on the mock.
"""

from __future__ import annotations

import random

WIDGET_TYPES = ["button", "field", "dropdown", "menu", "form", "dialog",
                "inactive", "distractor"]
GOAL_PREDICATES = ["FILL", "SELECT", "NAVIGATE", "SUBMIT", "REORDER"]
ACTION_TYPES = ["CLICK", "SELECT", "TYPE", "NAVIGATE", "SUBMIT"]


def mock_layout(rng: random.Random) -> dict:
    n = rng.randint(4, 10)
    widgets = [{"id": f"w{i}", "type": rng.choice(WIDGET_TYPES),
                "attrs": {"enabled": rng.random() > 0.2}} for i in range(n)]
    edges = [(f"w{i}", f"w{j}") for i in range(n - 1) for j in (i + 1,)
             if rng.random() < 0.5]
    return {"widgets": widgets, "edges": edges}


def mock_couples(rng: random.Random, n: int) -> list[dict]:
    """Dev couples for machinery tests (NOT target data)."""
    out = []
    for _ in range(n):
        lay = mock_layout(rng)
        state = {"focus": rng.choice(lay["widgets"])["id"],
                 "filled": [w["id"] for w in lay["widgets"] if rng.random() < 0.3]}
        goal = {"predicate": rng.choice(GOAL_PREDICATES),
                "args": {"target": rng.choice(lay["widgets"])["id"]}}
        out.append({"layout": lay, "state": state, "goal": goal})
    return out
