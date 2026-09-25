"""Ablation CONTRÔLÉE 4 bras (lead 12:15+12:25+12:44) — via hooks P2 labels_fn.

Design gelé v02 (c0ea1df): 4 bras × 12 seeds appariées, DROP-HAVE 200 test,
permutation PAR EXEMPLE pour C, retrait-à-l'inférence.
"""
from __future__ import annotations
import argparse, json, random, subprocess, sys, time
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# Hooks P2 (bda1d56)
from ucm.p2.runner import train_cell, eval_cell, load_dataset
from ucm.p2.model_p2 import P2Model
from ucm.data.writer import seal_manifest

ARMS = {
    "A_imitation": {"lambda_eff": 0.0, "head_on": False},
    "B_correct":   {"lambda_eff": 0.5, "head_on": True,  "mode": "identity"},
    "C_shuffled":  {"lambda_eff": 0.5, "head_on": True,  "mode": "shuffle_example"},
    "D_simple":    {"lambda_eff": 0.5, "head_on": True,  "mode": "simple"},
}
N_SEEDS = 12
UPDATES = 2000


def make_labels_fn(arm_name: str, seed: int):
    """Construit le labels_fn selon le bras (design v02 gelé)."""
    rng = random.Random(seed * 10000 + hash(arm_name) % 1000)
    mode = ARMS[arm_name].get("mode", "identity")
    
    if mode == "identity":
        return lambda eff, rec: eff  # B: labels exacts
    
    elif mode == "shuffle_example":
        def shuffle_fn(eff, rec):
            # PERMUTATION PAR EXEMPLE (design v02 §2): casse l'association
            # exemple→effet — JAMAIS permutation globale des classes
            permuted = list(eff)
            rng.shuffle(permuted)  # rng propre à CE bras/seed
            return permuted
        return shuffle_fn
    
    elif mode == "simple":
        def simple_fn(eff, rec):
            # Cible simple: validité/terminaison seulement (pas effets spécifiques)
            # classe 0=none → 0; toute classe non-nulle → 1 (binaire validité)
            return [0 if x == 0 else 1 for x in eff]
        return simple_fn
    
    return None  # A: pas utilisé (head_on=False)


def run_ablation():
    producer = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    
    # Charge le dataset P2 (DROP-HAVE)
    dataset = load_dataset()
    # Extraire les records d'entraînement et épisodes de test
    # Le runner P2 utilise son propre format — voir load_dataset
    
    results = {}
    for arm_name, arm_cfg in ARMS.items():
        arm_seeds = []
        for seed in range(N_SEEDS):
            t0 = time.time()
            labels_fn = make_labels_fn(arm_name, seed)
            
            # Construit le modèle
            model = P2Model(
                d=144,
                effect_lambda=arm_cfg["lambda_eff"],
                head_enabled=arm_cfg["head_on"]
            )
            
            # Entraîne via le hook
            # Le runner attend des records au format P2 — utilise le dataset
            # train_cell(model, records, UPDATES, seed, labels_fn=labels_fn)
            # Pour l'instant: mesure juste le temps et les métadonnées
            # (l'entraînement réel nécessite les records P2 formatés)
            
            arm_seeds.append({
                "seed": seed,
                "labels_fn_mode": arm_cfg.get("mode", "none"),
                "lambda_eff": arm_cfg["lambda_eff"],
                "head_on": arm_cfg["head_on"],
                "wall_s": round(time.time() - t0, 1),
                "status": "INFRASTRUCTURE_READY",
            })
        results[arm_name] = arm_seeds
        print(f"  {arm_name}: {len(arm_seeds)} seeds configurés")
    
    # Vérifie que les labels_fn produisent les bons effets
    test_eff = [0, 3, 0, 5, 1, 0, 2]
    test_rec = {"policy_input": {"candidates": [{}] * len(test_eff)}}
    
    fn_b = make_labels_fn("B_correct", 0)
    fn_c = make_labels_fn("C_shuffled", 0)
    fn_d = make_labels_fn("D_simple", 0)
    
    out_b = list(fn_b(test_eff, test_rec))
    out_c = list(fn_c(test_eff, test_rec))
    out_d = list(fn_d(test_eff, test_rec))
    
    print(f"\nVérification labels_fn:")
    print(f"  input:  {test_eff}")
    print(f"  B (identity):   {out_b} — identique ✓")
    print(f"  C (shuffle):    {out_c} — permuté par exemple ✓")
    print(f"  D (simple):     {out_d} — binarisé validité ✓")
    
    assert out_b == test_eff, "B doit être identité"
    assert sorted(out_c) == sorted(test_eff), "C doit conserver le multiset"
    assert out_c != test_eff, "C doit différer de l'input"
    assert all(x in (0, 1) for x in out_d), "D doit être binaire"
    print("  Tous les invariants ✓")
    
    art = seal_manifest({
        "schema": "ucm-ablation-effect-head-controlled/0.1",
        "what": "ablation contrôlée 4 bras — infrastructure prête, labels_fn vérifiés",
        "producer": {"code_commit": producer, "script": "scripts/dev_ablation_controlled.py"},
        "design_frozen": "c0ea1df (3a574de9) — v02 enrichi revue externe",
        "hooks": "bda1d56 — labels_fn injectable dans train_cell",
        "arms_configured": {arm: {"lambda": cfg["lambda_eff"], "head": cfg["head_on"],
                                   "mode": cfg.get("mode", "none")}
                            for arm, cfg in ARMS.items()},
        "n_seeds": N_SEEDS,
        "labels_fn_verified": {
            "B_identity": True, "C_shuffle_example": True, "D_simple": True,
            "C_preserves_multiset": True, "C_breaks_association": True,
        },
        "status": "INFRASTRUCTURE_READY — entraînement réel nécessite records P2 formatés (dataset P2 → train_cell)",
        "hold": "report-only, aucun entraînement exécuté",
    })
    Path("artifacts/dev-ablation-controlled-infra-v01.json").write_text(json.dumps(art, sort_keys=True, indent=2))
    print(f"\nscellé: {art['manifest_sha256'][:12]}")


if __name__ == "__main__":
    run_ablation()
