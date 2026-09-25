"""Fusion scellée des logs d'interactions par-worker — prototype B3+B4.

Design approuvé (tagi-1 21:20, review WS-B 21:18) :
- v1 séquentiel INTACT (InteractionsLog de ucm/v1/transfer.py, run protégé) ;
- v2 parallèle : chaque worker possède sa chaîne sha256 (même règle de seal
  que v1 : seal_i = sha256(canonical(entry_i + prev))[:16], genesis par
  worker) ; la fusion est POST-RUN uniquement.

Contrat d'entrée (writers tagi-3) : worker log = liste d'entrées
  {worker_id, seq, arm, k, motive, target, t, seal}
(« t » = métadonnée humaine, JAMAIS une clé d'ordre — B2).

Critères d'acceptation (B1–B4) :
- B3 : le seal de fusion lie le ROSTER [(worker_id, count, head_seal,
  entries_sha256)] — un worker manquant/silencé fait échouer la vérification ;
- B4 : fusion idempotente — permutations de scheduling → octets identiques,
  rejeu → même sha ;
- ordre total canonique : (arm, k, worker_id, seq) — jamais l'horloge ;
- tamper (entrée, roster, séquence) → échec explicite.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping

from .schema import SchemaError, canonical_json, sha256_hex
from .writer import seal_manifest, verify_manifest

MERGE_SCHEMA = "ucm-interactions-merge/0.2"
GENESIS = "genesis"


def entry_seal(entry: Mapping[str, Any], prev_seal: str) -> str:
    """Même règle que v1 (ucm/v1/transfer.py) — chaîne par worker."""
    payload = json.dumps({**{k: v for k, v in entry.items() if k != "seal"},
                          "prev": prev_seal}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


_REQUIRED_FIELDS = frozenset({"worker_id", "seq", "arm", "k", "motive", "target", "t", "seal"})


def _validate_entry_fields(entry: Mapping[str, Any], where: str) -> None:
    """N1 (lead 16:11): fail-closed — une entrée sans champ requis est une
    erreur, JAMAIS un default silencieux (arm=''/k=0)."""
    missing = _REQUIRED_FIELDS - set(entry)
    if missing:
        raise SchemaError(f"{where}: champs requis manquants {sorted(missing)} — fail-closed (N1)")


def verify_worker_chain(entries: Iterable[Mapping[str, Any]], worker_id: str) -> dict:
    """Vérifie la chaîne d'un worker. Retourne {count, head_seal, entries_sha256}.
    Toute rupture → SchemaError."""
    prev = GENESIS
    n = 0
    canonical_lines: list[str] = []
    for e in entries:
        _validate_entry_fields(e, f"worker {worker_id!r} seq {e.get('seq', '?')}")
        if e.get("worker_id") != worker_id:
            raise SchemaError(f"chaîne worker {worker_id!r}: entrée étrangère {e.get('worker_id')!r}")
        if e.get("seq") != n:
            raise SchemaError(f"chaîne worker {worker_id!r}: seq {e.get('seq')} ≠ {n} (trou/réordonnancement)")
        expected = entry_seal(e, prev)
        if e.get("seal") != expected:
            raise SchemaError(f"chaîne worker {worker_id!r}: seal ROMPU à seq {n} — tampering détecté")
        prev = expected
        canonical_lines.append(canonical_json({k: v for k, v in e.items() if k != "seal"}))
        n += 1
    return {"worker_id": worker_id, "count": n, "head_seal": prev,
            "entries_sha256": sha256_hex(canonical_lines) if canonical_lines else sha256_hex([])}


def _merged_order_key(e: Mapping[str, Any]) -> tuple:
    """B2 : ordre total canonique par CONTENU — jamais l'horloge (« t »)."""
    return (str(e.get("arm", "")), int(e.get("k", 0)), str(e.get("worker_id", "")), int(e.get("seq", 0)))


def merge_worker_logs(worker_logs: Mapping[str, list[Mapping[str, Any]]]) -> dict:
    """Fusion post-run scellée. Entrées: {worker_id: [entries]}.

    CONTRAT NORMATIF (gel v5, lead 16:11):
    - chaque entrée DOIT contenir {worker_id, seq, arm, k, motive, target, t, seal}
      — champ manquant = SchemaError (fail-closed, jamais de default silencieux);
    - les logs worker sont fournis ORDONNÉS PAR seq (N2): une permutation
      intra-worker casse la chaîne (prev_seal) et est détectée comme échec
      de séquence — cohérent avec la chaîne v1;
    - vérifie chaque chaîne (B1);
    - roster lié au seal (B3);
    - ordre canonique déterministe (B2): (arm, k, worker_id, seq) — jamais horloge;
    - artifact idempotent (B4): permutations d'insertion → octets identiques.
    """
    roster = []
    all_entries: list[tuple[tuple, Mapping[str, Any]]] = []
    for worker_id in sorted(worker_logs):  # tri ← B4 (ordre d'insertion sans effet)
        info = verify_worker_chain(worker_logs[worker_id], worker_id)
        roster.append(info)
        for e in worker_logs[worker_id]:
            all_entries.append((_merged_order_key(e), e))
    all_entries.sort(key=lambda x: x[0])  # B2 : ordre total canonique
    merged = [{k: v for k, v in e.items() if k != "seal"} for _, e in all_entries]
    merged_sequence_sha256 = sha256_hex([canonical_json(e) for e in merged])
    artifact = {
        "schema": MERGE_SCHEMA,
        "order_rule": "(arm, k, worker_id, seq) — contenu, jamais horloge",
        "roster": roster,  # B3 : lié au seal — worker manquant = échec
        "merged_count": len(merged),
        "merged_sequence_sha256": merged_sequence_sha256,
    }
    return seal_manifest(artifact)


def merge_worker_logs_bytes(worker_logs: Mapping[str, list[Mapping[str, Any]]]) -> str:
    """Forme sérialisée octet-stable de la fusion (pour comparaison inter-machines)."""
    return canonical_json(merge_worker_logs(worker_logs))


def verify_merge(merge: Mapping[str, Any], worker_logs: Mapping[str, list[Mapping[str, Any]]]) -> bool:
    """Re-vérifie une fusion contre les logs fournis. Tout écart → SchemaError.

    Couvre : worker MANQUANT du roster (B3), entrée tamperée (chaîne B1),
    séquence fusionnée altérée, seal de fusion invalide.
    """
    if not verify_manifest(merge):
        raise SchemaError("merge: seal invalide")
    roster_have = {r["worker_id"]: r for r in merge.get("roster", [])}
    for worker_id in worker_logs:
        if worker_id not in roster_have:
            raise SchemaError(f"merge: worker {worker_id!r} ABSENT du roster scellé (B3)")
    for r in merge.get("roster", []):
        if r["worker_id"] not in worker_logs:
            raise SchemaError(f"merge: roster déclare {r['worker_id']!r} mais log absent (B3)")
    replay = merge_worker_logs(worker_logs)
    if canonical_json(replay) != canonical_json(merge):
        raise SchemaError("merge: rejeu ≠ fusion publiée (tamper entrée/roster/séquence)")
    return True
