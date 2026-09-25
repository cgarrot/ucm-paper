"""Module d'analyse primaire V1-bis — fonction pure (lead 17:52, protocole a1a8b9d §3).

verdict_v1bis(raw_episodes_path) -> {diff_pts, ci_low, ci_high, exact_p, verdict}
Le verdict est UNE FONCTION DES RAW — pas une intention.

Règles (protocole §3):
- différences appariées par seed (pré-entraîné − scratch)
- agrégation 4 prédicats poids égaux par seed
- bootstrap hiérarchique seed-cluster ≥10k (seed fixe déclaré)
- IC 95% unilatéral [limite basse, +∞)
- test exact permutation des signes en SENSIBILITÉ
- PASS: IC_low > 0 ET point ≥ 5pp
- INDETERMINÉ: tout le reste avec IC_low ≤ 0 ou point < 5pp — **≠ « aucun transfert »**
- FAIL: IC_high < 0 (l'intervalle exclut tout effet positif)
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Optional

BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 20260923
CI_LEVEL = 0.95
MIN_EFFECT_PP = 5.0
PREDS = ("VIEW", "SET", "CHOOSE", "SUBMITTED")


def load_seed_results(path: str | Path) -> dict[str, dict[str, dict[str, float]]]:
    """Charge les raw: {arm: {seed: {predicate: success_rate_0_1}}}"""
    return json.loads(Path(path).read_text())


def paired_diffs(raw: dict) -> dict[str, dict[str, float]]:
    """Différences appariées par seed × prédicat: pretrained − scratch.
    Retourne {seed: {pred: diff_pp}} — structure conservée pour l'agrégation
    fixe puis bootstrap 1-niveau (les strates ne sont PAS rééchantillonnées)."""
    pre = raw.get("pretrained", {})
    scr = raw.get("scratch", {})
    diffs = {}
    for seed in pre:
        if seed not in scr:
            continue
        seed_diffs = {}
        for p in PREDS:
            if p in pre[seed] and p in scr[seed]:
                seed_diffs[p] = 100.0 * (pre[seed][p] - scr[seed][p])
        if seed_diffs:
            diffs[seed] = seed_diffs
    return diffs


def bootstrap_ci_1level(seed_pred_diffs: dict[str, dict[str, float]], *,
                               n: int = BOOTSTRAP_N,
                               seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    """Bootstrap 1 NIVEAU (erratum consolidé lead 18:26 — supersede du 2-niveaux
    bc05d3a): les prédicats sont FIXES = définition du benchmark, pas un
    échantillon; le claim est CONDITIONNEL au mix fixe.

    Procédure: agrégats par seed (moyenne des 4 prédicats), puis
    rééchantillonnage des seeds blocs AVEC remise. IC 95% unilatéral:
    (ci_low = 5e percentile, ci_high = 95e percentile — PAS max, bug 24/09 corrigé)."""
    if not seed_pred_diffs:
        return (0.0, 0.0)
    # agrégats par seed (prédicats fixes — pas de rééchantillonnage interne)
    seed_aggregates = [sum(d.values()) / len(d) for d in seed_pred_diffs.values()]
    rng = random.Random(seed)
    stats = []
    for _ in range(n):
        sample = [seed_aggregates[rng.randrange(len(seed_aggregates))]
                  for _ in range(len(seed_aggregates))]
        stats.append(sum(sample) / len(sample))
    stats.sort()
    ci_low = stats[int(0.05 * n)]
    ci_high = stats[int(0.95 * len(stats))]  # 95th percentile (was max — bug tagi-review 24/09)
    return (round(ci_low, 2), round(ci_high, 2))


def exact_sign_permutation(diffs: dict[str, float]) -> float:
    """Test exact permutation des signes (sensibilité): p-value bilatérale
    que la moyenne des diffs soit 0 (H0: effet nul)."""
    values = [v for v in diffs.values() if v != 0]
    n = len(values)
    if n == 0:
        return 1.0
    observed = abs(sum(values))
    count_extreme = 0
    total = 2 ** n
    if n <= 20:
        for mask in range(total):
            s = sum(values[i] if (mask >> i) & 1 else -values[i] for i in range(n))
            if abs(s) >= observed:
                count_extreme += 1
        return count_extreme / total
    else:
        # Monte Carlo pour n > 20
        rng = random.Random(BOOTSTRAP_SEED)
        n_mc = 100_000
        for _ in range(n_mc):
            s = sum(v if rng.random() < 0.5 else -v for v in values)
            if abs(s) >= observed:
                count_extreme += 1
        return count_extreme / n_mc


def verdict_v1bis(raw_path: str | Path) -> dict:
    """Fonction PURE: raw → verdict. Appelée par le run, auditable pinnée."""
    raw = load_seed_results(raw_path)
    diffs = paired_diffs(raw)
    if not diffs:
        return {"diff_pts": None, "ci_low": None, "ci_high": None,
                "exact_p": None, "verdict": "INDETERMINÉ",
                "note": "aucune paire seed trouvée — données insuffisantes"}

    # point estimate: moyenne des moyennes par seed (poids égaux prédicats)
    seed_means = {s: sum(d.values()) / len(d) for s, d in diffs.items()}
    point_estimate = sum(seed_means.values()) / len(seed_means)
    ci_low, ci_high = bootstrap_ci_1level(diffs)
    # permutation des signes sur les seed_means (sensibilité)
    exact_p = exact_sign_permutation(seed_means)

    # RÈGLES (arbitrage lead 18:00 — partition FAIL):
    # PASS: IC_low > 0 ET point >= 5pp
    # INDETERMINÉ: IC contient 0 ET point >= 5pp (l'effet pourrait exister mais non détecté)
    # FAIL: tout le reste — y compris point < 5pp même si IC > 0, et ci_high < 0
    # Fix flottant (tagi-5 18:29): comparer le seuil sur la valeur ARRONDIE publiée
    # (le point brut 4.999999... vs round=5.0 ne doit pas décider au bruit flottant)
    point_rounded = round(point_estimate, 2)
    # tolérance explicite ±1e-9 sur ci_low (représentation flottante documentée)
    _TOL = 1e-9
    if ci_low > _TOL and point_rounded >= MIN_EFFECT_PP:
        verdict = "PASS"
        note = f"IC bas > 0 et effet ≥ {MIN_EFFECT_PP}pp"
    elif (ci_low <= _TOL <= ci_high) and point_rounded >= MIN_EFFECT_PP:
        verdict = "INDETERMINÉ"
        note = ("IC contient 0 avec effet ≥ 5pp — non détecté à cette puissance; "
                "INDETERMINÉ ≠ « aucun transfert » : l'effet peut exister "
                "mais n'est pas détectable à ce niveau de puissance/n")
    else:
        verdict = "FAIL"
        note = (f"effet insuffisant ({point_estimate:.1f}pp < {MIN_EFFECT_PP}pp) "
                "ou intervalle exclut tout effet positif" if point_estimate < MIN_EFFECT_PP
                else "l'intervalle exclut tout effet positif")

    return {
        "diff_pts": round(point_estimate, 2),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "exact_p": round(exact_p, 4),
        "verdict": verdict,
        "note": note,
        "n_seeds": len(diffs),
        "bootstrap": {"n": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED, "level": CI_LEVEL},
    }


# alias de compatibilité (l'ancien nom référençait le 2-niveaux supprimé)
bootstrap_ci_hierarchical = bootstrap_ci_1level
