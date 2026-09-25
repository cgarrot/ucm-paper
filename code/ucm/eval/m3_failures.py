"""M3: failure categorization (PRE-SPECIFIED categories, lead 16:01 — the
taxonomy is fixed BEFORE inspection), G3/G4 descriptive cells with d*
histograms and success per band, and the GATE-4 baseline signal on test_g1.

Pre-specified categories (decision tree, first match wins — CODE order,
audit tagi-5 m1: the docstring previously announced 1..7 linearly while the
effective order is 1,2,3,6,5,4,7; this docstring now matches the code):
    1. action_invalide_repetee — repetition rule. STRICT (m2, future runs):
                                ≥3 consecutive identical actions flagged
                                INVALID in the per-action validity trace.
                                NOTE: the published m3-report.json (67b8928)
                                used the LOOSE variant (≥3 identical actions
                                AND n_invalid>0) because the per-action
                                validity was not recorded; tagi-5 verified the
                                inspected cases satisfy the strict rule too.
    2. boucle                  — a physical state cycle (state_hash repeats)
    3. oubli_stop              — timeout with goal physically satisfied at end
    6. binding_representation  — action referencing a wrong-typed entity
                                w.r.t. the goal (checked BEFORE 5/4)
    5. detour_inutile          — timeout, valid-only, ≥4 distinct rooms
    4. mauvaise_cible          — valid-only timeout otherwise
    7. autre                   — anything else
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict

import mlx.core as mx

mx.set_default_device(mx.cpu)

from ucm.eval import baselines as B
from ucm.eval import metrics as M
from ucm.eval.gate5_report import (CANON, load_canon_episodes, load_model,
                                   rebuild_layout, task_from_record)
from ucm.eval.rollout import ModelPolicy, run_episode
from ucm.env.oracle import LayoutOracle
from ucm.env.tinygraph import TinyGraphKey
from ucm.model.tensorize import tensorize_obs, collate

CATEGORIES = ["action_invalide_repetee", "boucle", "oubli_stop", "mauvaise_cible",
              "detour_inutile", "binding_representation", "autre"]


def categorize_failure(env_trace: list, result, task, valid_flags=None) -> str:
    """env_trace: state hashes AFTER each action; valid_flags (optional,
    future runs): per-action validity booleans for the STRICT repetition rule."""
    actions = result.actions
    # 1. repeated identical invalid actions (STRICT when valid_flags present)
    run = 1
    for i in range(1, len(actions)):
        run = run + 1 if actions[i] == actions[i - 1] else 1
        if run >= 3:
            if valid_flags is not None:
                if not all(valid_flags[i - run + 1:i + 1]):
                    return "action_invalide_repetee"
            elif result.n_invalid > 0:  # loose fallback (published variant)
                return "action_invalide_repetee"
    # 2. state loop (hash seen before)
    if len(set(env_trace)) < len(env_trace):
        return "boucle"
    # 3. goal satisfied at end without STOP success
    if result.goal_reached_without_stop:
        return "oubli_stop"
    goal = task["goal"]
    # 4/6 need the final configuration: use the last recorded state
    final = env_trace[-1] if env_trace else None
    # 6. binding: an action referencing the wrong-typed entity w.r.t. goal
    if goal["predicate"] == "AT":
        wrong = [a for a in actions
                 if a["action"] == "DROP" and a["arg"] != goal["args"]["object"]]
        if wrong:
            return "binding_representation"
    if goal["predicate"] == "HAVE":
        wrong = [a for a in actions
                 if a["action"] in ("PICK", "DROP") and a["arg"] != goal["args"]["object"]]
        if wrong:
            return "binding_representation"
    # 5. wandering detour: many distinct rooms, valid-only, timeout
    n_rooms = len({h.split("|")[0] for h in env_trace if "|" in h})
    if result.n_invalid == 0 and result.outcome == "timeout" and n_rooms >= 4:
        return "detour_inutile"
    # 4. wrong target: valid-only, timeout, ends far from goal referent
    if result.n_invalid == 0 and result.outcome == "timeout":
        return "mauvaise_cible"
    return "autre"


def trace_episode(env, pol, horizon=64):
    """Rollout recording state hashes (agent|key|parcel|locked) after each action."""
    obs = env.observe()
    trace, actions, valid_flags, n_invalid, outcome = [], [], [], 0, None
    goal_seen = False
    for _ in range(horizon):
        act = pol(obs)
        res = env.execute(act)
        actions.append(act)
        valid_flags.append(res["valid"])
        if not res["valid"]:
            n_invalid += 1
        trace.append(env.state_hash())
        if env.goal_satisfied():
            goal_seen = True
        if res["terminal"]:
            outcome = "success" if env.goal_satisfied() else "premature_stop"
            break
        obs = env.observe()
    if outcome is None:
        outcome = "timeout"
    return trace, actions, n_invalid, outcome, goal_seen, valid_flags


def main():
    os.makedirs("artifacts/m3", exist_ok=True)
    import glob
    report = {}
    # ---- failures on the GATE-2 official runs (all seeds, chosen arms) ----
    g1_eps = [e for e in load_canon_episodes(CANON, split="test_g1")
              if min(r["provenance"].get("step", 0) for r in e[3]) == 0]
    failures = []
    cats = Counter()
    per_seed = defaultdict(list)
    from ucm.eval.gate2_confirmation import TAUS
    chosen = json.load(open("artifacts/gate2-confirmation/gate2-confirmation.json"))["chosen_arm_per_seed"]
    lay_cache = {}
    for s in range(5):
        model, _ = load_model(sorted(glob.glob(f"artifacts/*gate2c-B144-s{s}"))[-1])
        arm = chosen[str(s)]["arm"]
        mx.random.seed(40_000 + s)  # declared evaluation RNG (audit tagi-5 m3)
        kw = {"sample": True, "temperature": float(arm.replace("tau", ""))} if arm.startswith("tau") else {}
        pol = ModelPolicy(model, name="B144", seed=s, **kw)
        for (ref, lay_h, d0, recs) in g1_eps:
            if lay_h not in lay_cache:
                lay_cache[lay_h] = rebuild_layout(recs)
            env = TinyGraphKey(lay_cache[lay_h])
            task = task_from_record(recs)
            env.reset(task)
            trace, actions, n_invalid, outcome, goal_seen, valid_flags = trace_episode(env, pol)
            if outcome != "success":
                cat = categorize_failure(trace, type("R", (), {
                    "actions": actions, "n_invalid": n_invalid, "outcome": outcome,
                    "goal_reached_without_stop": goal_seen})(), task, valid_flags)
                cats[cat] += 1
                per_seed[s].append((ref, cat))
                if len(failures) < 400:
                    failures.append({"seed": s, "episode": ref, "category": cat,
                                     "outcome": outcome, "n_invalid": n_invalid,
                                     "d_star": d0, "goal": task["goal"],
                                     "actions": actions[:70]})
        print(f"seed {s}: failures {len(per_seed[s])}", flush=True)
    report["failure_categorization"] = {
        "categories_pre_specified": CATEGORIES,
        "counts": dict(cats), "total_failures": sum(cats.values()),
        "total_episodes": 5 * len(g1_eps),
        "note": "taxonomy fixed before inspection (lead 16:01); decision tree order applied"}
    report["failures_20_inspected"] = failures[:20]

    # ---- G3 / G4 descriptive cells ----
    for cell in ("test_g3", "test_g4"):
        eps = [e for e in load_canon_episodes(CANON, split=cell)
               if min(r["provenance"].get("step", 0) for r in e[3]) == 0]
        hist = Counter(e[2] for e in eps)
        res_all = []
        model, _ = load_model(sorted(glob.glob("artifacts/*gate2c-B144-s2"))[-1])
        pol = ModelPolicy(model, name="B144", seed=2)
        lc = {}
        for (ref, lay_h, d0, recs) in eps:
            if lay_h not in lc:
                lc[lay_h] = rebuild_layout(recs)
            env = TinyGraphKey(lc[lay_h])
            env.reset(task_from_record(recs))
            r = run_episode(env, pol, f"{cell}-{ref}", 2, "B144", d_star=d0)
            r.layout_id = lay_h
            r.goal_type = recs[0]["policy_input"]["goal"]["predicate"]
            res_all.append(r)
        by_band = {}
        for band in sorted(hist):
            sub = [r for r in res_all if r.d_star == band]
            if sub:
                by_band[str(band)] = {"n": len(sub), "success": M.summarize(sub)["success_rate"]}
        report[cell] = {"n_episodes": len(eps), "d_star_hist": {str(k): v for k, v in sorted(hist.items())},
                         "success_overall": M.summarize(res_all)["success_rate"],
                         "success_by_d_star_band": by_band}
        print(cell, "→", report[cell]["success_overall"], "hist:", dict(hist), flush=True)

    # ---- GATE-4 signal: baselines on test_g1 (heuristic R1-R6 documented) ----
    base = {}
    for name, pol in (("local_heuristic_R1_R6", B.LocalHeuristic(0)),
                      ("random_valid", B.make_random_valid(0)),
                      ("random_syntactic", B.make_random_syntactic(0))):
        res = []
        lc2 = {}
        for (ref, lay_h, d0, recs) in g1_eps:
            if lay_h not in lc2:
                lc2[lay_h] = rebuild_layout(recs)
            env = TinyGraphKey(lc2[lay_h])
            env.reset(task_from_record(recs))
            r = run_episode(env, pol, f"b-{ref}", 0, name, d_star=d0)
            r.layout_id = lay_h
            res.append(r)
        base[name] = M.summarize(res)
        print("baseline", name, "→", base[name]["success_rate"], flush=True)
    report["gate4_baselines_test_g1"] = {
        "model_B144_official": 0.9750,
        "baselines": base,
        "heuristic_protocol": "R1 STOP if satisfied; R2 PICK goal obj (HAVE); R3 DROP at goal room (AT); "
                              "R4 DROP non-goal carried; R5 UNLOCK if key+locked+adjacent; R6 random valid MOVE. "
                              "No distances, no oracle, 1-hop only."}

    with open("artifacts/m3/m3-report.json", "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(json.dumps({k: report[k] for k in ("failure_categorization",)}, indent=1, default=str))


if __name__ == "__main__":
    main()
