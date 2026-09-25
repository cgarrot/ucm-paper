"""Control-non-targeted source (FREEZE.md §2, validated by lead 17:48).

RULE (exact, seed-controlled): for each canon train record, the supervision
label is replaced by ONE candidate drawn UNIFORMLY AT RANDOM among the
PHYSICALLY VALID actions of that state (observable preconditions only —
ucm.eval.baselines.valid_actions, no oracle information). Deterministic per
record: rng seeded by couple hash (seed 0), so the control dataset is bit-
reproducible. Input distribution is the EXACT canon TGK distribution
(same records, same geometry, same source budget); ONLY the supervision is
made non-informative for the decision.

Interpretation contract (published with results, §12.5):
    control ≈ pretrained-TGK → the transfer gain is EXPOSURE/OPTIMIZATION
                                 (accelération d'optimisation)
    control ≈ scratch        → the gain is DECISIONAL (informative labels)
The control shares the TGK input geometry BY DESIGN (spec §12.2 allows it
without guaranteeing absence of shared structure); interpretation must
account for it. A structurally-different source remains a possible frozen
follow-up if the control proves too strong/weak.
"""

from __future__ import annotations

import json
import random

from ucm.eval.baselines import valid_actions
from ucm.v1.transfer import couple_hash


def control_label(policy_input: dict, rng: random.Random) -> int:
    """Index (into policy_input['candidates']) of one uniformly random VALID
    action; STOP is allowed when goal-satisfied (valid_actions includes it)."""
    obs = policy_input
    valid = [c for c in obs["candidates"] if c in valid_actions(obs)]
    if not valid:  # no valid action: STOP (invalid by construction, visible)
        return len(obs["candidates"]) - 1
    chosen = valid[rng.randrange(len(valid))]
    return obs["candidates"].index(chosen)


def make_control_records(canon_records: list[dict]) -> list[dict]:
    """Canon records with NON-INFORMATIVE supervision (rule above). rng seeded
    per record by couple hash — deterministic, no dependence on call order."""
    out = []
    for rec in canon_records:
        pi = rec["policy_input"]
        rng = random.Random(int(couple_hash(
            {"layout": rec["provenance"].get("layout_hash", ""),
             "state": rec["provenance"].get("state_goal_hash", ""),
             "goal": pi["goal"]}), 16))
        idx = control_label(pi, rng)
        new = json.loads(json.dumps(rec))  # deep copy
        new["supervision"] = {"optimal_actions": [idx],
                              "d_star": None,  # sentinel: never actionable (audit mineur)
                              "reachable": True,
                              "control_note": "non-informative label: uniform among "
                                              "valid candidates, seed=hash(couple)"}
        out.append(new)
    return out


def load_control_records(path: str) -> list[dict]:
    """Load the control-source JSONL under its DECLARED VARIANT rule (lead
    19:15): source=control ⇒ supervision = {optimal_actions, reachable=True,
    d_star=None, control_note} — the exact symmetric inverse of the V0 rule
    (which requires int d_star); never mixed. V0 global validation is NOT
    relaxed; this loader enforces the variant locally."""
    out = []
    for ln, line in enumerate(open(path), 1):
        if not line.strip():
            continue
        rec = json.loads(line)
        sup = rec["supervision"]
        if sup.get("d_star") is not None or not sup.get("control_note") \
                or sup.get("reachable") is not True:
            raise ValueError(f"{path}:{ln}: control variant violated "
                             f"(d_star={sup.get('d_star')!r}, note={bool(sup.get('control_note'))})")
        out.append(rec)
    return out


def couple_of_record(rec: dict) -> dict:
    """Couple projection of a sealed record (ordering identity §12.3)."""
    prov = rec.get("provenance", {})
    return {"layout": prov.get("layout_hash", ""),
            "state": prov.get("state_goal_hash", ""),
            "goal": rec["policy_input"]["goal"]}


def derive_control_records_siw(records: list[dict]) -> list[dict]:
    """Ruling lead 19:21 (option 1): control-arm TARGET slice = the SAME
    couples, labels re-derived NON-INFORMATIVE — one candidate drawn uniformly
    among the SIW-PHYSICALLY-VALID actions of that state (env-reconstructed,
    adaptation-side). rng seeded by couple hash (bit-reproducible). The
    derivation hash is published next to coverage for audit re-derivation."""
    import copy
    from ucm.env.siw import SIW, SIWLayout
    out = []
    for rec in records:
        pi = rec["policy_input"]
        spec = dict(rec.get("layout_spec") or rec.get("provenance", {}).get("layout_spec"))
        if isinstance(spec.get("widgets"), dict):
            spec["widgets"] = list(spec["widgets"].values())
        rng = random.Random(int(couple_hash(couple_of_record(rec)), 16))
        valid_idx = []
        for i, cand in enumerate(pi["candidates"]):
            env = SIW(SIWLayout(spec))  # fresh env per candidate probe
            env.reset(rec["task"])
            try:
                res = env.execute(cand)
                if res.get("valid"):
                    valid_idx.append(i)
            except Exception:
                pass  # non-executable candidate: not valid
        if not valid_idx:
            valid_idx = [len(pi["candidates"]) - 1]  # STOP fallback (visible)
        new = copy.deepcopy(rec)
        new["supervision"] = {"optimal_actions": [rng.choice(valid_idx)],
                              "d_star": None, "reachable": True,
                              "control_note": "non-informative target label: uniform among "
                                              "SIW-valid candidates, seed=hash(couple)"}
        out.append(new)
    return out
