"""Permutation round-robin des épisodes R* — helper PARTAGÉ (lead 12:37).

Utilisé par: scripts/dev_dimensioning_rstar.py (métriques) ET la matérialisation
train DEV — UNE SEULE définition de l'ordre, jamais de divergence possible.
Ordre intra-épisode préservé; MÊME permutation pour train et métriques.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Hashable, Sequence

PREDS_CYCLE = ("VIEW", "SET", "CHOOSE", "SUBMITTED")


def round_robin_episodes(by_ep: dict, episode_ids: Sequence[Hashable]) -> list:
    """Permutation DÉTERMINISTE round-robin des épisodes ENTIERS entre les 4
    prédicats. ``by_ep`` = {episode_id: [couples ordonnés par depth]};
    le prédicat d'un épisode = celui de son premier couple. Retourne la liste
    ordonnée des episode_ids. Ordre INTRA-épisode inchangé."""
    by_pred = defaultdict(list)
    for eid in sorted(episode_ids):
        by_pred[by_ep[eid][0]["goal"]["predicate"]].append(eid)
    order, idx = [], {p: 0 for p in PREDS_CYCLE}
    while any(idx[p] < len(by_pred[p]) for p in PREDS_CYCLE):
        for p in PREDS_CYCLE:
            if idx[p] < len(by_pred[p]):
                order.append(by_pred[p][idx[p]])
                idx[p] += 1
    return order


def permuted_stream(by_ep: dict, episode_ids: Sequence[Hashable]) -> list:
    """Flux complet permuté: concatène les couples des épisodes dans l'ordre
    round-robin (ordre intra-épisode préservé). Pour digest FULL."""
    return [c for eid in round_robin_episodes(by_ep, episode_ids) for c in by_ep[eid]]


def write_rr_couples_file(
    couples: list,
    by_ep: dict,
    out_path: str,
    *,
    expected_preds: frozenset = frozenset(PREDS_CYCLE),
) -> dict:
    """Matérialisation PHYSIQUE du flux permuté (lead 12:39: writer RR).

    Écrit le fichier couples en ordre round-robin (épisodes entiers inter-
    prédicats, ordre intra-épisode préservé) — BINAIRE, SHA incrémental dans
    le flux. Assert set(prédicats) == expected_preds (renforcement 12:39).
    Retourne {path, episodes, records, sha256_write_stream, pred_counts}."""
    import hashlib as _hl
    import json as _json
    from pathlib import Path as _P

    ep_order = round_robin_episodes(by_ep, sorted(by_ep))
    preds = {by_ep[eid][0]["goal"]["predicate"] for eid in ep_order}
    assert preds == expected_preds, f"prédicats incomplets: {sorted(preds)} ≠ {sorted(expected_preds)}"
    out = _P(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    from ucm.data.rstar_order import serialize_rr_records as _ser
    permuted = [c for eid in ep_order for c in by_ep[eid]]
    data = _ser(permuted)
    hasher = _hl.sha256(data)
    out.write_bytes(data)
    n = len(permuted)
    from collections import Counter
    pred_counts = Counter(by_ep[eid][0]["goal"]["predicate"] for eid in ep_order)
    return {"path": str(out), "episodes": len(ep_order), "records": n,
            "sha256_write_stream": hasher.hexdigest(), "pred_counts": dict(pred_counts)}


def serialize_rr_records(couples: list) -> bytes:
    """Sérialisation CANONIQUE JSONL (lead 12:50:28): une ligne par couple,
    json.dumps(c, sort_keys=True) + "\n", encodage UTF-8. Ces octets EXACTS
    sont utilisés par le pilote (hashes stream/préfixe) ET le writer — le
    hash du tableau JSON est PROSCRIT pour attester le fichier/train."""
    import json as _json
    return b"".join((_json.dumps(c, sort_keys=True) + "\n").encode("utf-8") for c in couples)
