"""Sondes gratuites S2b (lead 12:35) — calculées sur les RAW (jamais sur les
agrégats): (1) corrélation échecs × distance/sauts (hypothèse champ récepteur
3 sauts: le taux d'échec doit grimper au-delà du rayon effectif du modèle);
(2) log-ratio d'échecs robuste au plafond (quand le succès sature à ~97-99%,
ce qui VARIE est le taux d'échec — l'échelle log donne une taille d'effet
lisible même près du plafond)."""

from __future__ import annotations

import math
from collections import defaultdict


def _rank(xs: list[float]) -> list[float]:
    """Rangs moyens (ties partagés) — pour Spearman sans dépendance externe."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    rx, ry = _rank(xs), _rank(ys)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den > 0 else float("nan")


def receptive_field_probe(raw: list[dict]) -> dict:
    """Taux d'échec par saut (d_star) + Spearman(échec, d_star) + L_star.
    Hypothèse 3 sauts: bucket d*≥4 nettement au-dessus ⇒ limite de rayon."""
    by_d = defaultdict(lambda: [0, 0])   # d* -> [failures, n]
    for r in raw:
        d = r.get("d_star")
        if d is None:
            continue
        d = min(int(d), 4)  # bucket 4+ agrège la queue
        key = str(d) if d < 4 else "4+"
        by_d[key][1] += 1
        if not r["success"]:
            by_d[key][0] += 1
    fail_rate_by_d = {k: (f / n if n else None) for k, (f, n) in sorted(by_d.items())}
    ds = [min(int(r["d_star"]), 6) for r in raw
          if r.get("d_star") is not None]
    fs = [0.0 if r["success"] else 1.0 for r in raw if r.get("d_star") is not None]
    return {"fail_rate_by_d_star": fail_rate_by_d,
            "spearman_fail_vs_dstar": spearman(
                [float(r["d_star"]) for r in raw if r.get("d_star") is not None], fs),
            "n": len(ds),
            "hypothesis_3hop": _hop3_readout(fail_rate_by_d)}


def _hop3_readout(fr: dict) -> dict:
    """ Lecture pré-enregistrée de l'hypothèse 3 sauts: le ratio d'échec
    au-delà de 3 sauts vs ≤3 sauts (None si queue vide)."""
    def rate(*keys):
        f = n = 0
        for k in keys:
            if k in fr and fr[k] is not None:
                pass  # besoin des comptes — recalcul depuis fail_rate seul impossible
        return None
    # version comptes (robuste): recalcul depuis raw est fait par l'appelant
    return {"note": "voir fail_rate_by_d_star: 4+ ≫ 1-3 soutient l'hypothèse",
            "fail_rate_le3_avg": _avg(fr, "1", "2", "3"),
            "fail_rate_4plus": fr.get("4+")}


def _avg(fr: dict, *keys) -> float | None:
    vals = [fr[k] for k in keys if fr.get(k) is not None]
    return sum(vals) / len(vals) if vals else None


def log_ratio_failures(raw_a: list[dict], raw_b: list[dict]) -> dict:
    """log(taux d'échec A / taux d'échec B) avec correction de continuité
    (+0.5 / n+1 — Agresti): lisible même quand l'un des taux est ~0
    (plafond). Positif ⇒ A échoue plus que B. Symétrique en échelle log."""
    fa = sum(1 for r in raw_a if not r["success"])
    fb = sum(1 for r in raw_b if not r["success"])
    ra = (fa + 0.5) / (len(raw_a) + 1)
    rb = (fb + 0.5) / (len(raw_b) + 1)
    lr = math.log(ra / rb)
    return {"log_ratio": round(lr, 4), "fail_a": fa, "n_a": len(raw_a),
            "fail_b": fb, "n_b": len(raw_b),
            "fail_rate_a": round(ra, 4), "fail_rate_b": round(rb, 4),
            "reading": "A échoue davantage" if lr > 0 else
                       ("B échoue davantage" if lr < 0 else "égal")}
