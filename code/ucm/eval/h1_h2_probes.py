"""H1: profondeur vs signature — H2: raccourci but→rôle (lead 11:31-2).
ÉVAL SEULE depuis ckpts existants, zéro entraînement.

H1 GO PRÉENREGISTRÉ: sur le canon B144 0-shot TGK —
  strate A: buts AT d*20-24 SANS traversée de porte longue (UNLOCK ∉ plan
            optimal, l'arête porte ∉ chemin optimal)
  strate B: buts AT d*≤12 AVEC porte longue obligatoire (UNLOCK ∈ plan)
  lecture départage: si A ≫ B ⇒ la traversée domine; si A ≈ B ≈ bas ⇒ la
  profondeur domine; chiffres publiés, pas de seuil binaire (sonde).
H2 GO PRÉENREGISTRÉ: canon 0-shot sur DROP-HAVE fraîches —
  type-match initial: fraction des tâches où le PREMIER argmax du canon
  (0-shot) a le MÊME TYPE d'action que l'optimal (DROP). Seuil: ≥ 40%
  (= 2× chance 20% sur 5 types) ⇒ le raccourci but→type existe;
  succès closed-loop rapporté en complément.
"""
from __future__ import annotations

import json
import os
import random
import time

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_OUT = os.path.join(_REPO, "artifacts/h1h2")


def _canon_model():
    import mlx.core as mx
    import mlx.nn as nn
    mx.set_default_device(mx.cpu)
    from ucm.eval.factorial_2x2 import load_canon_gnnb, canon_checkpoints_from_freeze
    cks = canon_checkpoints_from_freeze()
    m, sha = load_canon_gnnb(cks[0]["path"], cks[0]["sha256"])
    return m


def _gen_tgk_tasks(n, lo, hi, require_unlock, seed):
    """Tâches AT via DSL: d*0∈[lo,hi], UNLOCK ∈/∉ plan optimal."""
    from ucm.dsl.core import DSLInterpreter
    from ucm.dsl.tgk_program import TGK_DSL_PROGRAM, layout_from_native
    from ucm.dsl.bfs import DSLBFS
    from ucm.eval.gate5_report import load_canon_episodes, rebuild_layout
    eps = load_canon_episodes(os.path.join(
        _REPO, "artifacts/data/m0-transitions.jsonl"), split="train", limit=800)
    lays = {}
    for e in eps:
        lays.setdefault(e[1], e[3])
    rng = random.Random(seed)
    items = list(lays.items())
    out = []
    tries = 0
    while len(out) < n and tries < 40_000:
        tries += 1
        lh, recs = items[rng.randrange(len(items))]
        tables = layout_from_native(rebuild_layout(recs))
        rooms = tables["rooms"]
        obj = rng.choice(["key", "parcel"])
        goal = {"predicate": "AT", "args": {"object": obj,
                                            "room": rng.choice(rooms)}}
        agent = rng.choice(rooms)
        carried = rng.choice([None, "key", "parcel"])
        init = {"agent": agent, "carried": carried,
                "key": None if carried == "key" else rng.choice(rooms),
                "parcel": None if carried == "parcel" else rng.choice(rooms),
                "door_locked": rng.choice([True, False])}
        interp = DSLInterpreter(TGK_DSL_PROGRAM, tables)
        task = {"init": init, "goal": goal}
        try:
            interp.reset(task)
        except Exception:
            continue
        sol = DSLBFS(interp).solve(interp.state, max_states=40_000)
        if not sol["reachable"] or not (lo <= sol["d_star"] <= hi):
            continue
        # UNLOCK dans le plan optimal?
        kinds = set()
        st = interp.state
        for _ in range(30):
            interp.state = st
            if interp.goal_satisfied():
                break
            s2 = DSLBFS(interp).solve(st, max_states=40_000)
            cand = interp.candidates()[sorted(s2["optimal_actions"])[0]]
            if cand["action"] == "STOP":
                break
            kinds.add(cand["action"])
            _, st = interp.successor(cand)
        has_unlock = "UNLOCK" in kinds
        if has_unlock != require_unlock:
            continue
        out.append({"layout_tables": tables, "task": task,
                    "d_star": sol["d_star"], "opt_kinds": sorted(kinds),
                    "layout_hash": lh})
    return out


def run_h1h2(out_dir=_OUT):
    import mlx.core as mx
    import numpy as np
    mx.set_default_device(mx.cpu)
    from ucm.env.tinygraph import TinyGraphKey
    from ucm.eval.rollout import run_episode, ModelPolicy
    from ucm.p2.records import _native_layout
    from ucm.model.tensorize import tensorize_obs, collate
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    model = _canon_model()

    # H1: deux strates AT, canon 0-shot closed-loop
    strata = {"A_at_20_24_no_door": _gen_tgk_tasks(25, 20, 24, False, 4242),
              "B_at_le12_long_door": _gen_tgk_tasks(25, 2, 12, True, 1717)}
    h1 = {}
    for name, tasks in strata.items():
        succ = 0
        for t in tasks:
            lay = _native_layout(t["layout_tables"])
            env = TinyGraphKey(lay)
            env.reset(t["task"])
            pol = ModelPolicy(model, name="h1", seed=900)
            r = run_episode(env, pol, "h1", 900, "h1", d_star=t["d_star"])
            succ += r.success
        h1[name] = {"n": len(tasks), "success": round(succ / max(len(tasks), 1), 4)}

    # H2: canon 0-shot sur DROP-HAVE fraîches (corpus P2: test split)
    from ucm.p2.runner import load_dataset
    from ucm.p2.generator import split_family
    sp = split_family(load_dataset()["episodes"]["DROP-HAVE-d2-12"])
    type_match = 0
    succ = 0
    for ep in sp["test"]:
        lay = _native_layout(ep["layout_tables"])
        env = TinyGraphKey(lay)
        obs = env.reset(ep["task"])
        ex = tensorize_obs(obs); ex["labels"] = None
        out = np.asarray(model(collate([ex]))[0].tolist())
        act = obs["candidates"][int(np.argmax(out))]
        # type optimal au départ: l'épisode est DROP-HAVE ⇒ premier optimal
        # contient PICK/MOVE/DROP…; le raccourci testé: argmax TYPE == DROP?
        if act["action"] == "DROP":
            type_match += 1
        pol = ModelPolicy(model, name="h2", seed=900)
        env2 = TinyGraphKey(lay)
        env2.reset(ep["task"])
        r = run_episode(env2, pol, "h2", 900, "h2", d_star=ep["d_star"])
        succ += r.success
    n_h2 = len(sp["test"])
    tm = type_match / max(n_h2, 1)

    art = {"ts": ts,
           "h1_pre_registered": "A: AT d*20-24 sans porte (∉UNLOCK) | B: AT "
                                "d*≤12 avec porte obligatoire (∈UNLOCK); sonde "
                                "départage profondeur/traversée, pas de seuil",
           "h1": h1,
           "h2_pre_registered": "type-match initial argmax==DROP ≥ 40% (2× "
                                "chance) ⇒ raccourci but→type existe",
           "h2": {"n": n_h2, "drop_type_match_initial": round(tm, 4),
                  "shortcut_ge_40pct": tm >= 0.40,
                  "closed_loop_success_0shot": round(succ / max(n_h2, 1), 4)}}
    apath = os.path.join(out_dir, f"h1h2-{ts}.json")
    _fd = os.open(apath, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(_fd, "w") as fh:
        json.dump(art, fh, indent=1, sort_keys=True)
    art["persisted"] = apath
    return art


if __name__ == "__main__":
    print(json.dumps(run_h1h2(), indent=1, default=str))
