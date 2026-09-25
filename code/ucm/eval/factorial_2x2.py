"""Harnais factorielle 2×2 lisible — chapitre S2b→P2 (lead 12:35).

    (B144 / récurrent)  ×  (données oracle / données+récupération @f)

BudgÉTÉ ÉGALEMENT entre cellules: MÊMES n couples (la récupération REMPLACE,
jamais ajoute — recovery_mix), MÊMES updates, MÊME batch. Persistance O_EXCL
dès le premier commit (leçon run v9: un run sans dump est un run perdu).
E2E sur le chemin qui exécute (tests/test_factorial_2x2.py — données DEV
réelles, updates=1, aucune fixture parallèle).

Aucun scellé ici: DEV uniquement (layouts d'inventaire + couples v08-style).
"""
from __future__ import annotations

import json
import os
import time

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_OUT_DIR = os.path.join(_REPO, "artifacts/s2b-factorial")

CELLS = ("b144", "rec")          # bras modèle
DATA = ("oracle", "recovery")    # bras données


def build_eval_episodes(store: dict, n: int, seed: int,
                        d0_band: tuple = (1, 4)) -> list[dict]:
    """Épisodes d'éval DEV (jamais scellés): init échantillonné + d* oracle."""
    import random
    from ucm.env.siw import SIWLayout, sample_task
    from ucm.env.siw_oracle import SIWOracle
    from ucm.v1.data_adapter import _spec_of
    rng = random.Random(seed)
    lays = [SIWLayout(_spec_of(e)) for e in store.values()]
    eps = []
    tries = 0
    while len(eps) < n and tries < 100 * n:
        tries += 1
        lay = rng.choice(lays)
        st, goal = sample_task(rng, lay)
        o = SIWOracle(lay, goal)
        if not o.reachable(st):
            continue
        d = o.d_star(st)
        if not (d0_band[0] <= d <= d0_band[1]):
            continue
        eps.append({"episode_id": f"dev-eval-{seed}-{len(eps):04d}",
                    "layout_hash": lay.layout_hash(),
                    "task": {"goal": goal, "init": {
                        "view": st.view, "filled": sorted(st.filled),
                        "chosen": st.chosen, "dialog_open": st.dialog_open,
                        "submitted": sorted(st.submitted)}},
                    "d_star": d})
    if len(eps) < n:
        raise RuntimeError(f"eval episodes: {len(eps)}/{n}")
    # layout_spec résolu depuis le store (les épisodes portent des specs
    # sérialisables — audit bug 4)
    from ucm.v1.data_adapter import _spec_of
    for e in eps:
        e["layout_spec"] = _spec_of(store[e["layout_hash"]])
    return eps


def run_cell(model_arm: str, data_arm: str, couple_lines: list[str], store: dict,
             updates: int, seed: int, eval_episodes: list,
             f_recovery: float = 0.5, T: int = 32, eval_seed: int = 9090,
             world: str = "siw", canon_entry: dict | None = None):
    """UNE cellule de la factorielle. Retourne le dict de résultats complet
    (brut par épisode + métriques) — le harnais persiste, jamais l'appelant."""
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    from ucm.data.recovery_mix import build_recovery_mix
    from ucm.v1.data_adapter import _couples_core
    from ucm.v1.runner import finetune
    from ucm.v1.transfer import CoverageTracker, FinetuneConfig
    from ucm.v1.runner import evaluate as _evaluate

    # données: recovery REMPLACE à fraction f (budget total identique)
    if world == "tgk":
        records = list(couple_lines)  # couple_lines = RECORDS TGK déjà formés
        if data_arm == "recovery":
            from ucm.data.recovery_mix import build_recovery_mix_tgk
            records, mix_stats = build_recovery_mix_tgk(
                records, f_recovery=f_recovery, seed=seed)
        else:
            mix_stats = {"f_recovery": 0.0, "note": "oracle pur"}
    else:
        if data_arm == "recovery":
            lines, mix_stats = build_recovery_mix(couple_lines, store,
                                                  f_recovery=f_recovery, seed=seed)
        else:
            lines, mix_stats = couple_lines, {"f_recovery": 0.0, "note": "oracle pur"}
        records = _couples_core(store, lines)

    # DÉTERMINISME CELLULE (v10): seed AVANT construction du modèle — l'init
    # ne dépend plus de l'état RNG ambiant (série et parallèle identiques).
    import mlx.core as _mx0
    _mx0.random.seed(seed)

    # CANON OBLIGATOIRE en TGK (lead 14:05): les DEUX bras partent du canon
    # B144 — jamais de base fraîche aléatoire pour le verdict S2b.
    canon_sha_expected = None
    if world == "tgk":
        if canon_entry is None:
            raise RuntimeError("world='tgk' exige canon_entry (path+sha256)")
        base, canon_sha_expected = load_canon_gnnb(canon_entry["path"],
                                                   canon_entry["sha256"])
        from ucm.model.recurrent_block import SIWRecModel
        if model_arm == "rec":
            model = SIWRecModel(base, T=T)
        else:
            model = base
        # GARDE (14:05-2): la base doit ÊTRE le canon — pas seulement rester gelée
        actual_sha = (model.base_params_sha256() if model_arm == "rec"
                      else SIWRecModel(model, T=1).base_params_sha256())
        if actual_sha != canon_sha_expected:
            raise RuntimeError("CANON NOT LOADED — base sha != canon sha")
    else:
        if model_arm == "rec":
            from ucm.model.recurrent_block import SIWRecModel
            from ucm.model.siw_model import make_siw_model
            model = SIWRecModel(make_siw_model(), T=T)
        else:
            from ucm.model.siw_model import make_siw_model
            model = make_siw_model()
        if model_arm == "rec":
            import mlx.core as _mx
            _mx.eval(model.parameters())
    base_sha_before = (model.base_params_sha256() if model_arm == "rec"
                       else canon_sha_expected)
    canon_sha_before = canon_sha_expected
    refine_sha_before = None
    if model_arm == "rec":
        import hashlib as _hl0
        import numpy as _np0
        import mlx.nn as _nn0
        parts0 = [k + "|" + _hl0.sha256(_np0.asarray(p.tolist(), dtype=_np0.float32).tobytes()).hexdigest()
                  for k, p in _nn0.utils.tree_flatten(model.refine.parameters())]
        refine_sha_before = _hl0.sha256("\n".join(parts0).encode()).hexdigest()

    # budget ÉGAL: mêmes updates, même config — seule l'init diffère (seed)
    mx.random.seed(seed)
    cfg = FinetuneConfig(updates=updates, seed=seed)
    cov = CoverageTracker()
    trainable = "refine" if model_arm == "rec" else "all"   # GEL base (12:36)
    train_meta = finetune(model, records, cfg, cov,
                          f"{model_arm}-{data_arm}", len(records),
                          trainable=trainable, world=world)

    # GARDE NO-OP (14:55): le refine DOIT avoir changé pendant l'entraînement
    # (l'optimiseur subtree a tourné). Un no-op silencieux = run invalide.
    refine_sha_after = None
    if model_arm == "rec":
        import hashlib as _hl
        import numpy as _np
        import mlx.nn as _nnq
        parts = [k + "|" + _hl.sha256(_np.asarray(p.tolist(), dtype=_np.float32).tobytes()).hexdigest()
                 for k, p in _nnq.utils.tree_flatten(model.refine.parameters())]
        refine_sha_after = _hl.sha256("\n".join(parts).encode()).hexdigest()
        if updates > 0 and refine_sha_after == refine_sha_before:
            raise RuntimeError("REFINE NEVER TRAINED — optimizer no-op (bug 14:55)")

    # GEL VÉRIFIABLE: sha base avant == après (artefact, pas promesse)
    base_sha_after = model.base_params_sha256() if model_arm == "rec" else None
    if model_arm == "rec" and base_sha_before != base_sha_after:
        raise RuntimeError("CIV base DRIFTED during training — freeze broken")
    if model_arm == "rec" and canon_sha_expected is not None \
            and base_sha_after != canon_sha_expected:
        raise RuntimeError("CIV base drifted FROM CANON — frozen ≠ canon")

    # T EFFECTIF PROUVÉ MÉCANIQUEMENT (compteur instrumenté, pas déclaré)
    T_effective = None
    if model_arm == "rec":
        T_effective = model.refine.n_iters
        if T_effective != T:
            raise RuntimeError(f"T_effectif={T_effective} != T déclaré={T}")

    # éval closed-loop DEV (sondes câblées sur les bruts)
    if world == "tgk":
        res = _evaluate_tgk(model, eval_episodes, eval_seed,
                            f"{model_arm}-{data_arm}")
    else:
        res = _evaluate(model, eval_episodes, eval_seed, f"{model_arm}-{data_arm}",
                        len(records))

    # CHECKPOINTS O_EXCL par cellule (lead 13:31-2: sondes post-hoc + P1)
    ckpt_dir = os.environ.get("UCM_S2B_CKPT_DIR") or os.path.join(
        os.environ.get("UCM_RUN_DIR", _OUT_DIR), "ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{model_arm}-{data_arm}-s{seed}.npz")
    if os.path.exists(ckpt_path):
        raise RuntimeError(f"cell ckpt exists (O_EXCL): {ckpt_path}")
    import mlx.core as _mx
    import mlx.nn as _nn
    _mx.savez(ckpt_path, **dict(_nn.utils.tree_flatten(model.parameters())))

    raw = []
    for r in res:
        raw.append({"cell": f"{model_arm}-{data_arm}", "success": bool(r.success),
                    "outcome": r.outcome, "length": r.length,
                    "n_invalid": r.n_invalid, "d_star": r.d_star,
                    "L_star": r.L_star,
                    "goal_reached_without_stop": bool(r.goal_reached_without_stop),
                    "episode_id": r.episode_id})
    succ = sum(1 for r in raw if r["success"])
    return {"cell": f"{model_arm}-{data_arm}",
            "model_arm": model_arm, "data_arm": data_arm,
            "T": T if model_arm == "rec" else 0,
            "T_effective_proven": T_effective,          # compteur mécanique
            "base_frozen_sha256": base_sha_after,        # gel vérifié
            "base_frozen_verified": True if model_arm == "rec" else None,
            "canon_sha256": canon_sha_before,            # la base EST le canon
            "base_is_canon_verified": (canon_sha_before is not None),
            "refine_sha256_before": refine_sha_before,
            "refine_sha256_after": refine_sha_after,
            "refine_actually_trained": (refine_sha_after is not None
                                        and refine_sha_after != refine_sha_before),
            "n_couples": len(records), "n_eval": len(raw),
            "success_rate": succ / max(len(raw), 1),
            "mix_stats": mix_stats, "train": train_meta, "raw": raw}


def run_factorial(couple_lines: list, store: dict | None, updates: int,
                  seed: int = 11, n_eval: int = 40, f_recovery: float = 0.5,
                  T: int = 32, out_dir: str = _OUT_DIR,
                  ts: str | None = None, run: bool = True,
                  world: str = "siw", d_band: tuple = (13, 24),
                  workers: int = 0,
                  canon_freeze: str | None = None) -> dict:
    """Les 4 cellules + sondes + persistance O_EXCL. world='tgk': couple_lines
    = records TGK (store ignoré), éval = strate test_g4 d*∈d_band."""
    if not run:
        return {"status": "plan", "cells": [f"{m}-{d}" for m in CELLS for d in DATA],
                "world": world}
    os.makedirs(out_dir, exist_ok=True)
    ts = ts or time.strftime("%Y%m%dT%H%M%S")
    if world == "tgk":
        eval_eps = load_tgk_eval_episodes(n=n_eval, d_band=d_band)
    else:
        eval_eps = build_eval_episodes(store, n_eval, seed=seed + 1)

    # RESUME (lead 13:31-1): artefact existant → charger + VÉRIFIER + sauter;
    # corrompu → erreur explicite, JAMAIS réécrire.
    def _verify_cell_artifact(path: str, m: str, d: str) -> dict:
        try:
            art = json.load(open(path, encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise RuntimeError(f"cell artefact CORROMPU (jamais réécrit): {path}: {e}")
        req = {"cell", "model_arm", "data_arm", "n_couples", "n_eval",
               "success_rate", "raw"}
        missing = req - set(art)
        if missing:
            raise RuntimeError(f"cell artefact incomplet {path}: manque {sorted(missing)}")
        if art["cell"] != f"{m}-{d}" or art["model_arm"] != m or art["data_arm"] != d:
            raise RuntimeError(f"cell artefact mismatch {path}: {art['cell']}")
        if len(art["raw"]) != art["n_eval"]:
            raise RuntimeError(f"cell artefact {path}: raw {len(art['raw'])} != n_eval {art['n_eval']}")
        if m == "rec" and (art.get("base_frozen_verified") is not True
                           or art.get("T_effective_proven") != T):
            raise RuntimeError(f"cell artefact rec {path}: gardes absentes/obsolètes")
        return art

    work = []
    cells = {}
    for m in CELLS:
        for d in DATA:
            cpath = os.path.join(out_dir, f"cell-{m}-{d}-{ts}.json")
            if os.path.exists(cpath):
                cells[f"{m}-{d}"] = _verify_cell_artifact(cpath, m, d)
            else:
                work.append((m, d))

    def _canon_for(seed_):
        if world != "tgk":
            return None
        cks = canon_checkpoints_from_freeze(canon_freeze)
        return cks[seed_ % len(cks)]

    def _run_and_persist(m, d):
        r = run_cell(m, d, couple_lines, store, updates, seed,
                     eval_eps, f_recovery=f_recovery, T=T, world=world,
                     canon_entry=_canon_for(seed))
        cpath = os.path.join(out_dir, f"cell-{m}-{d}-{ts}.json")
        _fd = os.open(cpath, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        with os.fdopen(_fd, "w") as fh:
            json.dump(r, fh, sort_keys=True, default=str)
        return r

    if workers and work:
        # PARALLÈLE (lead 13:31-3): 2 process pour rec (borne RSS) + pool
        # partagé — cellules indépendantes, seeds contrôlent TOUT (v10).
        import concurrent.futures
        rec_w = [(m, d) for m, d in work if m == "rec"]
        base_w = [(m, d) for m, d in work if m != "rec"]
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {}
            _ce = _canon_for(seed)
            for m, d in rec_w[:2]:      # ≤2 process rec simultanés (RSS)
                futs[pool.submit(_run_cell_worker, (m, d, couple_lines, store,
                                                    updates, seed, eval_eps,
                                                    f_recovery, T, world,
                                                    out_dir, ts, _ce))] = (m, d)
            for m, d in rec_w[2:] + base_w:
                futs[pool.submit(_run_cell_worker, (m, d, couple_lines, store,
                                                    updates, seed, eval_eps,
                                                    f_recovery, T, world,
                                                    out_dir, ts, _ce))] = (m, d)
            for fut in concurrent.futures.as_completed(futs):
                m, d = futs[fut]
                cells[f"{m}-{d}"] = fut.result()
    else:
        for m, d in work:
            cells[f"{m}-{d}"] = _run_and_persist(m, d)

    # Sondes gratuites (champ récepteur + log-ratio plafond-robuste)
    from ucm.eval.probes_s2b import log_ratio_failures, receptive_field_probe
    probes = {"receptive_field": {
        c: receptive_field_probe(r["raw"]) for c, r in cells.items()},
        "log_ratio_vs_b144_oracle": {
            c: log_ratio_failures(r["raw"], cells["b144-oracle"]["raw"])
            for c, r in cells.items()}}

    summary = {"status": "complete", "ts": ts, "T": T, "f_recovery": f_recovery,
               "updates": updates, "seed": seed, "n_eval": n_eval, "world": world,
               "d_band": d_band if world == "tgk" else None,
               "canon_sha256": cells.get("b144-oracle", {}).get("canon_sha256"),
               "cells": {c: {"success_rate": r["success_rate"],
                             "n_couples": r["n_couples"]} for c, r in cells.items()},
               "probes": probes}
    spath = os.path.join(out_dir, f"factorial-summary-{ts}.json")
    _fd = os.open(spath, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(_fd, "w") as fh:
        json.dump(summary, fh, sort_keys=True, indent=1, default=str)
    return {"summary": summary, "cells": cells, "out_dir": out_dir, "ts": ts}


# ---------------------------------------------------------------------------
# TGK (verdict S2b: strate G4 d*=13-24)
# ---------------------------------------------------------------------------

_TGK_TRANSITIONS = os.path.join(_REPO, "artifacts/data/m0-transitions.jsonl")


def load_tgk_train_records(n: int, seed: int = 0) -> list[dict]:
    """Records TGK split=train (déterministes: ordre fichier, n premiers)."""
    out = []
    for line in open(_TGK_TRANSITIONS, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r["provenance"]["split"] == "train":
            out.append(r)
            if len(out) >= n:
                break
    if len(out) < n:
        raise RuntimeError(f"tgk train records: {len(out)}/{n}")
    return out


def load_tgk_eval_episodes(n: int | None = None,
                           d_band: tuple = (13, 24),
                           path: str = _TGK_TRANSITIONS) -> list[tuple]:
    """Épisodes TGK test_g4, strate d*0 ∈ [13,24] — format gate5
    (episode_ref, layout_hash, d_star, records), step-0 seulement."""
    from ucm.eval.gate5_report import load_canon_episodes
    eps = load_canon_episodes(path, split="test_g4")
    sel = [e for e in eps
           if e[2] is not None and d_band[0] <= e[2] <= d_band[1]]
    if n is not None:
        sel = sel[:n]
    if not sel:
        raise RuntimeError("test_g4 d*∈%s: aucun épisode" % (d_band,))
    return sel


def _evaluate_tgk(model, episodes, seed: int, arm: str) -> list:
    """Closed-loop TGK (gate5 pattern): TinyGraphKey + ModelPolicy + §9.4."""
    from ucm.env.tinygraph import TinyGraphKey
    from ucm.eval.rollout import ModelPolicy, run_episode
    from ucm.eval.gate5_report import rebuild_layout, task_from_record
    lay_cache = {}
    results = []
    for (ref, lay_h, d0, recs) in episodes:
        if lay_h not in lay_cache:
            lay_cache[lay_h] = rebuild_layout(recs)
        env = TinyGraphKey(lay_cache[lay_h])
        env.reset(task_from_record(recs))
        pol = ModelPolicy(model, name=f"{arm}", seed=seed)
        r = run_episode(env, pol, ref, seed, arm, d_star=d0)
        r.layout_id = lay_h
        results.append(r)
    return results


def _run_cell_worker(args):
    """Pool worker: une cellule, persistée par le worker (O_EXCL), seed-contrôlée."""
    (m, d, couple_lines, store, updates, seed, eval_eps,
     f_recovery, T, world, out_dir, ts, canon_entry) = args
    r = run_cell(m, d, couple_lines, store, updates, seed,
                 eval_eps, f_recovery=f_recovery, T=T, world=world,
                 canon_entry=canon_entry)
    cpath = os.path.join(out_dir, f"cell-{m}-{d}-{ts}.json")
    _fd = os.open(cpath, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(_fd, "w") as fh:
        json.dump(r, fh, sort_keys=True, default=str)
    return r


# ---------------------------------------------------------------------------
# CANON B144 (lead 14:05 — les DEUX bras partent du canon, jamais du scratch)
# ---------------------------------------------------------------------------

def canon_checkpoints_from_freeze(freeze_path: str | None = None) -> list[dict]:
    """Entrées canon du freeze v10 (arms.pretrained_TGK.checkpoints)."""
    fp = freeze_path or os.path.join(_REPO, "artifacts/freeze-v1bis-official-v10.json")
    fm = json.load(open(fp, encoding="utf-8"))
    cks = fm["arms"]["pretrained_TGK"]["checkpoints"]
    cand = [{"path": c["path"], "sha256": c["sha256"], "seed": c["seed"],
             "role": c.get("role")} for c in cks if c.get("role") == "canon"]
    # FILTRE DIMENSION RÉELLE (découverte 14:2x): s5-s9 du freeze sont d=192
    # (génération différente) — ne garder QUE les canon GNNB(d=144) natifs,
    # sinon le loader fail-closed de toute façon (shapes). Le lead: s0-s4.
    import mlx.core as _mxp
    out = []
    for c in cand:
        try:
            probe = dict(_mxp.load(c["path"])).get("node_enc.layers.0.bias")
            if probe is not None and probe.shape[-1] == 144:
                out.append(c)
        except Exception:
            continue  # illisible → exclu (fail-closed au chargement si requis)
    if not out:
        raise RuntimeError("aucun checkpoint canon d=144 natif (s0-s4) dans le freeze")
    return out


def load_canon_gnnb(ckpt_path: str, expected_sha: str):
    """Charge le canon B144 dans un GNNB(d=144) natif — sha RE-VÉRIFIÉ avant
    chargement, correspondance 100% noms+shapes EXIGÉE (canon natif: tout
    doit transférer). Retourne (model, canon_params_sha256)."""
    import hashlib
    import mlx.core as mx
    import mlx.nn as nn
    from ucm.model.gnn_b import GNNB
    actual = hashlib.sha256(open(ckpt_path, "rb").read()).hexdigest()
    if actual != expected_sha:
        raise RuntimeError(f"canon SHA drift: {ckpt_path}")
    base = GNNB(d=144)
    source = dict(mx.load(ckpt_path))
    own = dict(nn.utils.tree_flatten(base.parameters()))
    if set(source) != set(own):
        missing = set(own) - set(source)
        extra = set(source) - set(own)
        raise RuntimeError(f"canon non-natif (noms≠): missing={sorted(missing)[:3]} extra={sorted(extra)[:3]}")
    bad = [n for n, a in source.items() if tuple(own[n].shape) != tuple(a.shape)]
    if bad:
        raise RuntimeError(f"canon shapes incompatibles: {bad[:5]}")
    base.update(nn.utils.tree_unflatten(sorted(source.items())))
    mx.eval(base.parameters())
    from ucm.model.recurrent_block import SIWRecModel
    canon_sha = SIWRecModel(base, T=1).base_params_sha256()
    return base, canon_sha
