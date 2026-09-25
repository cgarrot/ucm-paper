"""Mix oracle/récupération à volume contrôlé (chapitre S2b→P2, axe données).

'Récupération' = couple dont l'état résulte d'une DÉVIATION suivie de la
re-solve oracle depuis l'état dévié: on prend un couple oracle, on exécute
UNE action NON-optimale, on émet le couple (état dévié, but) supervisé par
l'oracle depuis cet état (si encore atteignable). Le volume est contrôlé par
la fraction f_recovery: f×n couples remplacés par des récupérations, budget
total n IDENTIQUE entre cellules (égalité de budget de la factorielle).
"""
from __future__ import annotations

import json
import random

from ucm.env.siw import SIW, SIWLayout
from ucm.env.siw_oracle import SIWOracle


def _state_of(state_key) -> "object":
    from ucm.v1.data_adapter import _state_of as _so
    return _so(state_key)


def _spec_of(entry: dict) -> dict:
    from ucm.v1.data_adapter import _spec_of as _spo
    return _spo(entry)


def build_recovery_mix(couple_lines: list[str], store: dict,
                       f_recovery: float = 0.5, seed: int = 0) -> tuple[list[str], dict]:
    """Retourne (lignes mixtes 0.8, stats). Volume total IDENTIQUE: les
    couples 'récupération' REMPLACENT des couples oracle (pas d'ajout).
    f_recovery=0 → lignes oracle pures (identiques en ordre)."""
    if not 0.0 <= f_recovery <= 1.0:
        raise ValueError(f"f_recovery={f_recovery} hors [0,1]")
    rng = random.Random(seed)
    couples = [json.loads(l) for l in couple_lines if l.strip()]
    n = len(couples)
    n_rec = round(f_recovery * n)
    stats = {"n_total": n, "n_oracle": n - n_rec, "n_recovery": n_rec,
             "f_recovery": f_recovery, "unreachable_after_deviation": 0,
             "recovery_d_hist": {}}

    # indices à remplacer (déterministes: sample seedé)
    replace_idx = set(rng.sample(range(n), n_rec)) if n_rec else set()

    oracle_cache: dict = {}
    out = []
    for i, c in enumerate(couples):
        if i not in replace_idx:
            out.append(json.dumps(c, sort_keys=True))
            continue
        # dévier: reconstruire l'état, choisir une action NON-optimale
        lay = SIWLayout(_spec_of(store[c["layout_hash"]]))
        goal = {"predicate": c["goal"]["predicate"], "args": c["goal"]["args"]}
        key = (c["layout_hash"], goal["predicate"], json.dumps(goal["args"], sort_keys=True))
        o = oracle_cache.get(key)
        if o is None:
            o = SIWOracle(lay, goal)
            oracle_cache[key] = o
        st = _state_of(c["state_key"])
        env = SIW(lay)
        env.reset({"init": {"view": st.view, "filled": sorted(st.filled),
                            "chosen": st.chosen, "dialog_open": st.dialog_open,
                            "submitted": sorted(st.submitted)},
                   "goal": goal})
        obs = env.observe()
        optimal = set(o.optimal_actions(st))
        non_opt = [k for k, cand in enumerate(obs["candidates"])
                   if k not in optimal]
        if not non_opt:
            # aucun écart possible (état terminal) → garder l'oracle (compte)
            stats["unreachable_after_deviation"] += 1
            out.append(json.dumps(c, sort_keys=True))
            continue
        k = rng.choice(sorted(non_opt))
        res = env.execute(obs["candidates"][k])
        st2 = env.state
        if not o.reachable(st2):
            stats["unreachable_after_deviation"] += 1
            out.append(json.dumps(c, sort_keys=True))  # garde l'oracle à la place
            continue
        d2 = o.d_star(st2)
        opt2 = sorted(o.optimal_actions(st2))
        stats["recovery_d_hist"][str(d2)] = stats["recovery_d_hist"].get(str(d2), 0) + 1
        rc = dict(c)
        rc["state_key"] = [st2.view, sorted(st2.filled), sorted(st2.chosen.items()),
                           st2.dialog_open, sorted(st2.submitted)]
        rc["d_star"] = d2
        rc["optimal_semantic"] = [
            f"{o.candidates[j]['action']}:{o.candidates[j]['arg']}" for j in opt2]
        rc["rstar_executed"] = None
        rc["rstar_executed_semantic"] = None
        rc["source"] = "recovery"
        out.append(json.dumps(rc, sort_keys=True))

    stats["n_recovery_effective"] = sum(1 for l in out if '"source": "recovery"' in l)
    return out, stats


# ---------------------------------------------------------------------------
# TGK (monde TinyGraphKey — verdict S2b strate G4 d*=13-24)
# ---------------------------------------------------------------------------

def build_recovery_mix_tgk(records: list[dict], f_recovery: float = 0.5,
                           seed: int = 0) -> tuple[list[dict], dict]:
    """Mêmes sémantiques que build_recovery_mix, sur des RECORDS TGK
    (policy_input/supervision): déviation d'UNE action non-optimale via
    TinyGraphKey.execute, re-solve LayoutOracle depuis l'état dévié.
    Volume contrôlé: f REMPLACE (budget total identique)."""
    if not 0.0 <= f_recovery <= 1.0:
        raise ValueError(f"f_recovery={f_recovery} hors [0,1]")
    import random
    rng = random.Random(seed)
    n = len(records)
    n_rec = round(f_recovery * n)
    replace_idx = set(rng.sample(range(n), n_rec)) if n_rec else set()
    stats = {"n_total": n, "n_oracle": n - n_rec, "n_recovery": n_rec,
             "f_recovery": f_recovery, "unreachable_after_deviation": 0,
             "n_recovery_effective": 0}
    from ucm.env.tinygraph import TinyGraphKey
    from ucm.env.oracle import LayoutOracle
    from ucm.eval.gate5_report import rebuild_layout, task_from_record

    out = []
    oracle_cache = {}
    for i, r in enumerate(records):
        if i not in replace_idx:
            out.append(r)
            continue
        try:
            lay = rebuild_layout([r])
            task = task_from_record([r])
            env = TinyGraphKey(lay)
            obs = env.reset(task)
            goal = task["goal"]
            key = (repr(lay), json.dumps(goal, sort_keys=True))
            o = oracle_cache.get(key)
            if o is None:
                o = LayoutOracle(lay, goal)
                oracle_cache[key] = o
            optimal = set(o.optimal_actions(env.state))
            non_opt = [k for k in range(len(obs["candidates"]))
                       if k not in optimal]
            if not non_opt:
                stats["unreachable_after_deviation"] += 1
                out.append(r)
                continue
            k = rng.choice(sorted(non_opt))
            env.execute(obs["candidates"][k])
            st2 = env.state
            if not o.reachable(st2):
                stats["unreachable_after_deviation"] += 1
                out.append(r)
                continue
            sol = o.solve(st2)
            r2 = {"policy_input": env.observe(),
                  "supervision": {"optimal_actions": sol["optimal_actions"],
                                  "d_star": sol["d_star"], "reachable": True},
                  "provenance": dict(r.get("provenance", {}),
                                     source="recovery")}
            out.append(r2)
            stats["n_recovery_effective"] += 1
        except Exception:
            # fail-safe: garde l'oracle (compté) — jamais de perte de volume
            stats["unreachable_after_deviation"] += 1
            out.append(r)
    return out, stats
