"""S2b ANNEXE (lead 17:54-1) — éval G1′ + bandes secondaires depuis les
checkpoints run3, SANS ré-entraînement.

Bandes: test_g4 d*≤8 (rétention), [16,24], [13,24] + test_g1 (G1′ rétention).
Référence rétention: canon B144 zéro-shot sur les MÊMES bandes — prédicat
pinné: perte ≤ 2pp (comparée au prédicat ≤2pp du protocole).
Publication annexe O_EXCL — le KILL primaire [13,15] est déjà rendu.
"""
from __future__ import annotations

import json
import os
import time

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_RUN3 = os.path.join(_REPO, "artifacts/s2b-factorial/ckpts-run3")
_OUT = os.path.join(_REPO, "artifacts/s2b-factorial/annex")

CELLS = ("b144-oracle", "b144-recovery", "rec-oracle", "rec-recovery")
SEEDS = (100, 101, 102, 103, 104)
T_RUN3 = 32


def _bands():
    # G4 est une strate CONSTRUITE dure (d*0≥13): la rétention d*≤8 vit dans
    # test_g1 (600 épisodes d*≤8). Bandes: G1′ complet + sous-bande rétention
    # d*≤8 (prédicat ≤2pp) + bandes secondaires G4.
    return {
        "g1_prime": ("test_g1", None, None),   # rétention G1 complète (632 eps)
        "g1_d0_8": ("test_g1", 0, 8),          # sous-bande rétention ≤2pp (600 eps)
        "g4_16_24": ("test_g4", 16, 24),       # annexe secondaire (25 eps)
        "g4_13_24": ("test_g4", 13, 24),       # bande large (103 eps)
    }


def load_episodes(split: str, d_lo: int | None, d_hi: int | None,
                  path: str | None = None):
    from ucm.eval.gate5_report import load_canon_episodes
    p = path or os.path.join(_REPO, "artifacts/data/m0-transitions.jsonl")
    eps = load_canon_episodes(p, split=split)
    if d_lo is not None:
        eps = [e for e in eps if e[2] is not None and d_lo <= e[2] <= d_hi]
    if not eps:
        raise RuntimeError(f"bande vide: {split} [{d_lo},{d_hi}]")
    return eps


def load_cell_model(cell: str, seed: int):
    """Reconstruit le modèle d'une cellule depuis son npz run3.
    rec = SIWRecModel(canon, T) chargé (base+refine); b144 = GNNB chargé."""
    import mlx.core as mx
    import mlx.nn as nn
    from ucm.model.gnn_b import GNNB
    from ucm.model.recurrent_block import SIWRecModel
    from ucm.eval.factorial_2x2 import load_canon_gnnb, canon_checkpoints_from_freeze

    path = os.path.join(_RUN3, f"{cell}-s{seed}.npz")
    if not os.path.exists(path):
        raise RuntimeError(f"ckpt run3 manquant: {path}")
    w = dict(mx.load(path))
    if cell.startswith("rec"):
        cks = canon_checkpoints_from_freeze()
        base, _ = load_canon_gnnb(cks[(seed - 100) % len(cks)]["path"],
                                  cks[(seed - 100) % len(cks)]["sha256"])
        m = SIWRecModel(base, T=T_RUN3)
    else:
        m = GNNB(d=144)
    own = dict(nn.utils.tree_flatten(m.parameters()))
    if set(own) != set(w):
        raise RuntimeError(f"{cell}-s{seed}: clés npz ≠ modèle ({len(w)} vs {len(own)})")
    m.update(nn.utils.tree_unflatten(sorted(w.items())))
    mx.eval(m.parameters())
    return m


def eval_band(model, episodes, seed: int) -> dict:
    from ucm.eval.factorial_2x2 import _evaluate_tgk
    res = _evaluate_tgk(model, episodes, seed=9000 + seed,
                        arm=f"annex-s{seed}")
    n = len(res)
    succ = sum(1 for r in res if r.success)
    return {"n": n, "success_rate": succ / max(n, 1)}


def run_annex(out_dir: str = _OUT, ts: str | None = None,
              run: bool = True) -> dict:
    if not run:
        return {"status": "plan", "bands": list(_bands()),
                "cells": list(CELLS), "seeds": list(SEEDS)}
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    os.makedirs(out_dir, exist_ok=True)
    ts = ts or time.strftime("%Y%m%dT%H%M%S")

    # épisodes par bande (une seule lecture du fichier par bande)
    band_eps = {name: load_episodes(*args) for name, args in _bands().items()}

    # référence rétention: canon zéro-shot sur CHAQUE bande
    from ucm.eval.factorial_2x2 import load_canon_gnnb, canon_checkpoints_from_freeze
    cks = canon_checkpoints_from_freeze()
    canon = {}
    for name, eps in band_eps.items():
        base, _ = load_canon_gnnb(cks[0]["path"], cks[0]["sha256"])
        canon[name] = eval_band(base, eps, seed=0)

    results = {"ts": ts, "bands": {}, "canon_zero_shot": canon,
               "retention_predicate_pp": 2.0}
    for name, eps in band_eps.items():
        per_cell = {}
        for cell in CELLS:
            rates = []
            for seed in SEEDS:
                m = load_cell_model(cell, seed)
                r = eval_band(m, eps, seed=seed)
                rates.append(r["success_rate"])
            per_cell[cell] = {
                "mean_success_rate": sum(rates) / len(rates),
                "per_seed": rates, "n_eval_per_seed": len(eps)}
        # prédicat rétention: perte vs canon zéro-shot ≤ 2pp (par cellule)
        retention = {
            cell: {"diff_pp": round(100 * (canon[name]["success_rate"]
                                           - per_cell[cell]["mean_success_rate"]), 2),
                   "retained": (canon[name]["success_rate"]
                                - per_cell[cell]["mean_success_rate"]) <= 0.02}
            for cell in CELLS}
        results["bands"][name] = {"episodes": len(eps), "cells": per_cell,
                                  "retention_vs_canon": retention}
        print(f"[annex] {name}: {len(eps)} eps — " +
              " ".join(f"{c}={per_cell[c]['mean_success_rate']:.3f}"
                       for c in CELLS))

    apath = os.path.join(out_dir, f"s2b-annex-{ts}.json")
    _fd = os.open(apath, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(_fd, "w") as fh:
        json.dump(results, fh, sort_keys=True, indent=1, default=str)
    results["persisted"] = apath
    return results


if __name__ == "__main__":
    import sys
    r = run_annex(run="--plan" not in sys.argv)
    print(json.dumps(r, indent=1, default=str)[:2000])
