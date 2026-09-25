"""Pilote DEV P2 (lead 18:43-4): p_ref et SD inter-seed sur familles tenues à
l'écart (cc3f3eb v02, bucket TGK d*2-12).

But: les chiffres qui manquent pour geler Δ_min AVANT puissance (§5.3).
  1. familles: signatures [kind, pred, bucket] held du manifeste v02, bucket
     d2-12, 3-5 familles sélectionnées (documentées dans l'artefact)
  2. tâches: épisodes du split train classés par signature (kind ∈ kinds du
     plan optimal via LayoutOracle, prédicat du but, bucket d*0)
  3. entraînement: GNNB d=144 FRAIS (B144-class) par seed (frais — le pilote
     mesure la DIFFICULTÉ des familles, pas le transfert), entraîné sur les
     records des épisodes des familles
  4. éval: closed-loop sur des tâches des MÊMES familles, DISJOINTES des
     tâches d'entraînement (split par épisode)
  5. p_ref = succès moyen par famille; SD inter-seed = écart-type des succès
     entre seeds (la variance de RÉFÉRENCE qui borne la détectabilité)
Persistance O_EXCL. HOLD scellés inchangé — DEV uniquement.
"""
from __future__ import annotations

import json
import os
import time

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_M0 = os.path.join(_REPO, "artifacts/data/m0-transitions.jsonl")
_HELD = os.path.join(_REPO, "artifacts/dev-p2-held-back-families-v02.json")
_OUT = os.path.join(_REPO, "artifacts/p2-pilot")


def held_families_d2_12(n_families: int = 4, min_eps: int = 13) -> list[list]:
    """Signatures d2-12 tenues à l'écart — le cartésien inclut des familles
    VIDES (ex: DROP+REACH: poser un objet ne mène jamais à une pièce); la
    sélection retient les n_families PREMIÈRES NON VIDES (comptées au
    classement, pas supposées)."""
    d = json.load(open(_HELD, encoding="utf-8"))
    sigs = [s for s in d["held_back_COMPLETE"]["tgk"]["signatures"]
            if s[2] == "d2-12"]
    # classement sur TOUTES les candidates puis sélection non vide
    fams = classify_episodes(sigs, max_eps_per_family=200)
    sel = [list(s) for s in sigs if len(fams[tuple(s)]) >= min_eps][:n_families]
    if len(sel) < 3:
        raise RuntimeError(f"familles d2-12 non vides insuffisantes: {len(sel)}")
    return sel


def _episode_records(path=_M0, split="train"):
    """Groupe les records par épisode (ordre fichier déterministe)."""
    eps, order = {}, []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r["provenance"]["split"] != split:
            continue
        ref = r["provenance"]["episode_ref"]
        if ref not in eps:
            eps[ref] = []
            order.append(ref)
        eps[ref].append(r)
    return [(ref, eps[ref]) for ref in order]


def classify_episodes(seeds_families: list[list], max_eps_per_family: int = 60):
    """Classe les épisodes train par signature (kind du plan optimal, prédicat,
    bucket d*0 ∈ [2,12]). Retourne {sig_key: [(ref, records, d0, kinds)]}."""
    from collections import defaultdict
    from ucm.env.tinygraph import TinyGraphKey
    from ucm.env.oracle import LayoutOracle
    from ucm.eval.gate5_report import rebuild_layout, task_from_record

    fams = {tuple(s): [] for s in seeds_families}
    # pré-condition rapide: prédicat + d* par épisode
    for ref, recs in _episode_records():
        goal = recs[0]["policy_input"]["goal"]
        pred = goal["predicate"]
        want = [s for s in seeds_families if s[1] == pred]
        if not want:
            continue
        try:
            lay = rebuild_layout(recs)
            task = task_from_record(recs)
            env = TinyGraphKey(lay)
            env.reset(task)
        except Exception:
            continue
        o = LayoutOracle(lay, task["goal"])
        st0 = env.state
        if not o.reachable(st0):
            continue
        d0 = o.d_star(st0)
        if not (2 <= d0 <= 12):
            continue
        # kinds du plan optimal (BFS arrière du natif: actions optimales enchaînées)
        kinds = _optimal_plan_kinds(o, env, st0, max_steps=24)
        for s in want:
            if s[0] in kinds:
                fams[tuple(s)].append((ref, recs, d0, kinds))
                break
    out = {}
    for sig, lst in fams.items():
        lst.sort(key=lambda x: x[0])
        out[sig] = lst[:max_eps_per_family]
    return out


def _optimal_plan_kinds(oracle, env, st0, max_steps=24):
    """Kinds présents dans UN plan optimal (greedy par actions optimales)."""
    kinds = set()
    st = st0
    env2 = env
    for _ in range(max_steps):
        if oracle.d_star(st) == 0:
            kinds.add("STOP")
            return kinds
        opts = oracle.optimal_actions(st)
        if not opts:
            return kinds
        ai = sorted(opts)[0]
        act = oracle.candidates[ai]
        kinds.add(act["action"])
        from ucm.env.oracle import _successor
        st2 = _successor(oracle.layout, st, act)
        if st2 is None:
            return kinds
        st = st2
    return kinds


def run_pilot(n_families=4, seeds=(0, 1, 2), updates=300, n_eval_per_family=8,
              out_dir=_OUT, ts: str | None = None, run: bool = True) -> dict:
    if not run:
        return {"status": "plan", "families": held_families_d2_12(n_families),
                "seeds": list(seeds), "updates": updates}
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    os.makedirs(out_dir, exist_ok=True)
    ts = ts or time.strftime("%Y%m%dT%H%M%S")

    fams_sigs = held_families_d2_12(n_families, min_eps=n_eval_per_family + 5)
    fams = classify_episodes(fams_sigs)

    # split train/eval par épisode (disjoint, déterministe par ref triée)
    plan = {}
    for sig in fams_sigs:
        lst = fams[tuple(sig)]
        cut = len(lst) - n_eval_per_family
        plan["-".join(sig)] = {"train": lst[:cut], "eval": lst[cut:],
                               "n_train_eps": cut, "n_eval_eps": n_eval_per_family}

    from ucm.model.gnn_b import GNNB
    from ucm.v1.runner import finetune
    from ucm.v1.transfer import CoverageTracker, FinetuneConfig
    from ucm.eval.factorial_2x2 import _evaluate_tgk
    from ucm.eval.gate5_report import rebuild_layout, task_from_record
    from ucm.env.tinygraph import TinyGraphKey

    results = {"ts": ts, "families_selection": [
        {"signature": s, "n_train_eps": plan["-".join(s)]["n_train_eps"]}
        for s in fams_sigs],
        "seeds": list(seeds), "updates": updates, "per_family": {}}

    for fam_key, sp in plan.items():
        # records d'entraînement (tous les steps des épisodes train)
        train_records = [r for _, recs, _, _ in sp["train"] for r in recs]
        # épisodes d'éval au format gate5 (ref, hash, d0, recs)
        eval_eps = []
        for ref, recs, d0, _ in sp["eval"]:
            eval_eps.append((ref, recs[0]["provenance"]["layout_hash"], d0, recs))
        per_seed = []
        for seed in seeds:
            mx.random.seed(seed)
            model = GNNB(d=144)
            finetune(model, train_records,
                     FinetuneConfig(updates=updates, seed=seed),
                     CoverageTracker(), f"p2pilot-{fam_key}", len(train_records),
                     trainable="all", world="tgk")
            res = _evaluate_tgk(model, eval_eps, seed=7000 + seed,
                                arm=f"p2pilot-{fam_key}")
            succ = sum(1 for r in res if r.success) / len(res)
            per_seed.append(succ)
        n = len(per_seed)
        mean = sum(per_seed) / n
        sd = (sum((x - mean) ** 2 for x in per_seed) / n) ** 0.5
        results["per_family"][fam_key] = {
            "p_ref": round(mean, 4), "sd_inter_seed": round(sd, 4),
            "per_seed": [round(x, 4) for x in per_seed],
            "n_train_records": len(train_records), "n_eval": len(eval_eps)}
        print(f"[p2-pilot] {fam_key}: p_ref={mean:.3f} sd={sd:.3f} "
              f"per_seed={[round(x,2) for x in per_seed]}")

    apath = os.path.join(out_dir, f"p2-pilot-{ts}.json")
    _fd = os.open(apath, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(_fd, "w") as fh:
        json.dump(results, fh, sort_keys=True, indent=1, default=str)
    results["persisted"] = apath
    return results


if __name__ == "__main__":
    import sys
    r = run_pilot(run="--plan" not in sys.argv)
    print(json.dumps(r, indent=1, default=str)[:1500])
