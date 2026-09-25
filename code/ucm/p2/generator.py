"""P2-1 (A) — Générateur de tâches des familles primaires via le DSL VÉRIFIÉ.

Le générateur EST l'interpréteur audité (fa02d96): chaque tâche est
construite par RESET UNIQUEMENT, classée par signature (kinds du plan
optimal BFS-DSL, prédicat du but, d*0 ∈ [2,12]), épisodes AUTO-SUFFISANTS
(layout tables embarquées — rejouables sans dépendance externe).
Splits train/dev/test DISJOINTS par famille (ordre déterministe par
episode_id, proportions 60/20/20).
Familles primaires gelées (16275ad): DROP-AT, DROP-HAVE.
"""
from __future__ import annotations

import json
import os
import random

from ucm.dsl.core import DSLInterpreter
from ucm.dsl.tgk_program import TGK_DSL_PROGRAM, layout_from_native
from ucm.dsl.bfs import DSLBFS

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PRIMARY_FAMILIES = (("DROP", "AT", "d2-12"), ("DROP", "HAVE", "d2-12"))
D_BAND = (2, 12)
GEN_SEED = 20260924  # gelé (RNG du générateur)


def _layouts_from_m0(limit=40):
    """Layouts natifs TGK (train) → tables DSL. Les familles sont tenues à
    l'écart par SIGNATURE, pas par layout — tout layout sert."""
    from collections import defaultdict
    from ucm.eval.gate5_report import load_canon_episodes, rebuild_layout
    eps = load_canon_episodes(os.path.join(_REPO, "artifacts/data/m0-transitions.jsonl"),
                              split="train", limit=800)
    by = {}
    for e in eps:
        by.setdefault(e[1], e[3])
        if len(by) >= limit:
            break
    out = []
    for lh, recs in by.items():
        lay = rebuild_layout(recs)
        out.append((lh, layout_from_native(lay)))
    return out


def _optimal_kinds_dsl(interp: DSLInterpreter, start, max_steps=14):
    """Kinds d'un plan optimal DSL (greedy par actions optimales)."""
    sol = DSLBFS(interp).solve(start, max_states=20_000)
    kinds = []
    st = start
    for _ in range(max_steps):
        interp.state = st
        if interp.goal_satisfied():
            kinds.append("STOP")
            return kinds, sol["d_star"], sol
        cands = interp.candidates()
        opts = [i for i in sol["optimal_actions"]
                ] if st is start else None
        # re-solve depuis l'état courant (petits espaces: ok)
        s2 = DSLBFS(interp).solve(st, max_states=20_000)
        if not s2["reachable"]:
            return kinds, None, sol
        opts = s2["optimal_actions"]
        i = sorted(opts)[0]
        cand = cands[i]
        if cand["action"] == "STOP":
            kinds.append("STOP")
            return kinds, s2["d_star"], sol
        kinds.append(cand["action"])
        _, st = interp.successor(cand)
    return kinds, None, sol


def generate_family_tasks(n_per_family=60, seed=GEN_SEED, layouts_limit=40):
    """Génère les tâches des familles primaires via le DSL. Retourne
    {fam_key: [épisode…]} — épisode auto-suffisant:
    {episode_id, family, layout_tables, task, d_star, opt_kinds}."""
    rng = random.Random(seed)
    layouts = _layouts_from_m0(layouts_limit)
    fams = {f: [] for f in PRIMARY_FAMILIES}
    attempts = 0
    # objets/pièces: échantillonnage par reset d'états initiaux aléatoires
    while any(len(v) < n_per_family for v in fams.values()) and attempts < 6000:
        attempts += 1
        lh, tables = layouts[rng.randrange(len(layouts))]
        rooms = tables["rooms"]
        goal_pred = rng.choice(["AT", "HAVE"])
        obj = rng.choice(["key", "parcel"])
        if goal_pred == "AT":
            goal = {"predicate": "AT", "args": {"object": obj,
                                                "room": rng.choice(rooms)}}
        else:
            goal = {"predicate": "HAVE", "args": {"object": obj}}
        agent = rng.choice(rooms)
        other = "parcel" if obj == "key" else "key"
        carried = rng.choice([None, "key", "parcel"])
        init = {"agent": agent,
                "carried": carried,
                "key": None if carried == "key" else rng.choice(rooms),
                "parcel": None if carried == "parcel" else rng.choice(rooms),
                "door_locked": rng.choice([True, False])}
        interp = DSLInterpreter(TGK_DSL_PROGRAM, tables)
        task = {"init": init, "goal": goal}
        try:
            interp.reset(task)
        except Exception:
            continue
        sol = DSLBFS(interp).solve(interp.state, max_states=20_000)
        if not sol["reachable"]:
            continue
        d0 = sol["d_star"]
        if not (D_BAND[0] <= d0 <= D_BAND[1]):
            continue
        # signature: kinds du plan optimal (première action(s) suffisante:
        # le plan contient-il DROP?)
        kinds, _, _ = _optimal_kinds_dsl(interp, interp.state)
        if "DROP" not in kinds:
            continue
        fam = ("DROP", goal["predicate"], "d2-12")
        if fam not in fams:
            continue
        if len(fams[fam]) >= n_per_family:
            continue
        ep = {"episode_id": f"p2-{fam[0]}-{fam[1]}-{seed}-{len(fams[fam]):04d}",
              "family": list(fam), "layout_hash": lh,
              "layout_tables": tables, "task": task, "d_star": d0,
              "opt_kinds": kinds}
        fams[fam].append(ep)
    for f, v in fams.items():
        if len(v) < n_per_family:
            raise RuntimeError(f"famille {f}: {len(v)}/{n_per_family} tâches "
                               f"(attempts={attempts})")
    return fams


def split_family(episodes, props=(0.6, 0.2, 0.2)):
    """Splits disjoints (ordre déterministe par episode_id)."""
    eps = sorted(episodes, key=lambda e: e["episode_id"])
    n = len(eps)
    n_tr = int(props[0] * n)
    n_dev = int(props[1] * n)
    return {"train": eps[:n_tr], "dev": eps[n_tr:n_tr + n_dev],
            "test": eps[n_tr + n_dev:]}


def build_dataset(n_per_family=60, out_path: str | None = None) -> dict:
    fams = generate_family_tasks(n_per_family)
    ds = {"schema": "ucm-p2-dataset/0.1", "gen_seed": GEN_SEED,
          "d_band": list(D_BAND), "families": {}}
    for f, eps in fams.items():
        ds["families"]["-".join(f)] = {"n": len(eps),
                                       **{k: [e["episode_id"] for e in v]
                                          for k, v in split_family(eps).items()}}
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        json.dump({"dataset": ds, "episodes": {"-".join(f): v
                                               for f, v in fams.items()}},
                  open(out_path, "w"), sort_keys=True, default=str)
    return {"meta": ds, "episodes": fams}
