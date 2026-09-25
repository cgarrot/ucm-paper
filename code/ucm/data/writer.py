"""Écriture JSONL des transitions UCM — WS-B.

- Une ligne = une transition validée (``schema.validate_record``).
- Déduplication sémantique : clé = (layout_hash, state_goal_hash, action
  sémantique (action,arg)). Un doublon ne duplique pas la ligne : à la
  fermeture, les visites sont fusionnées dans ``provenance.visits`` (la
  provenance des visites n'est jamais perdue, spec §5.2) — alimente
  l'exposition à l'optimiseur (§7.2). Absence de ``visits`` ⇒ exposé une fois.
  La fusion est idempotente : fermer/réouvrir/fermer sans nouveau doublon ne
  change pas le fichier.
- Cohérence : deux occurrences d'une même clé doivent partager le même
  ``policy_input`` sémantique et la même supervision sémantique, sinon
  ``InconsistentDuplicateError`` (jamais de correction silencieuse).
- Contenu addressable : les grosses observations peuvent vivre hors-ligne via
  ``write_blob`` (sha256 complet, 64 hex) ; le record garde la référence.
- ``close()`` produit un manifest de fichier (lignes, uniques, sha256 du
  contenu) — scellé par :func:`seal_manifest`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Iterator, Mapping, Optional

from .schema import (
    Record,
    SchemaError,
    canonical_json,
    sha256_bytes,
    sha256_hex,
    validate_record,
)


class InconsistentDuplicateError(SchemaError):
    """Même clé de dédup mais corps incohérent — corruption de données à investiguer."""


DedupKey = tuple[str, str, tuple[str, Optional[str]]]


@dataclass
class WriteStats:
    path: str
    lines_written: int = 0
    duplicates_merged: int = 0
    unique_records: int = 0
    content_sha256: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "lines_written": self.lines_written,
            "duplicates_merged": self.duplicates_merged,
            "unique_records": self.unique_records,
            "content_sha256": self.content_sha256,
        }


def record_dedup_key(record: Record) -> DedupKey:
    action = record.policy_input.candidates[record.execution.action_ref]
    return (
        record.provenance.layout_hash,
        record.provenance.state_goal_hash,
        action.semantic(),
    )


def _policy_signature(record: Record) -> str:
    pi = record.policy_input
    return sha256_hex(
        {
            "goal": pi.goal.to_dict(),
            "entities": sorted((e.to_dict() for e in pi.entities), key=canonical_json),
            "relations": sorted((r.to_dict() for r in pi.relations), key=canonical_json),
            "candidate_semantics": sorted(map(repr, pi.action_semantics)),
        }
    )


def _supervision_signature(record: Record) -> str:
    cand = record.policy_input.candidates
    opt_semantics = sorted(map(repr, (cand[i].semantic() for i in record.supervision.optimal_actions)))
    return sha256_hex({"optimal": opt_semantics, "d_star": record.supervision.d_star,
                       "reachable": record.supervision.reachable})


def _execution_signature(record: Record) -> str:
    """P4b (audit tagi-5) : observable_result/next_state_hash divergents sur une
    même clé sémantique = incohérence, jamais une fusion silencieuse."""
    return sha256_hex({"observable_result": record.execution.observable_result,
                       "next_state_hash": record.execution.next_state_hash})


class JsonlRecordWriter:
    """Writer JSONL avec dédup sémantique et fusion de provenance.

    >>> with JsonlRecordWriter(path) as w:
    ...     w.write(record)
    ...     stats = w.stats
    """

    def __init__(self, path: str | os.PathLike[str], append: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._index: dict[DedupKey, dict[str, Any]] = {}
        self._dirty_keys: set[DedupKey] = set()
        self.stats = WriteStats(path=str(self.path))
        mode = "a" if (append and self.path.exists()) else "w"
        if mode == "a":
            self._rebuild_index()
        self._fh: IO[str] = self.path.open(mode, encoding="utf-8")

    # -- index ---------------------------------------------------------------

    def _rebuild_index(self) -> None:
        for line_no, raw in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not raw.strip():
                continue
            try:
                record = validate_record(json.loads(raw))
            except Exception as exc:  # pragma: no cover
                raise SchemaError(f"{self.path}:{line_no}: ligne existante invalide: {exc}") from exc
            key = record_dedup_key(record)
            if key in self._index:
                # P6 (audit tagi-5) : doublon PRÉEXISTANT dans le fichier (écrit
                # hors writer/dédup) — détecté, visites fusionnées, une seule
                # ligne survit à la matérialisation.
                visits_dup = list(record.provenance.extra.get("visits") or [])
                if not visits_dup:
                    visits_dup = [self._visit_of(record)]
                self._index[key]["visits"].extend(visits_dup)
                self._dirty_keys.add(key)
                continue
            visits = list(record.provenance.extra.get("visits") or [])
            if not visits:
                # ligne écrite sans fusion : occurrence unique → visite synthétisée
                visits = [self._visit_of(record)]
            self._index[key] = {
                "policy": _policy_signature(record),
                "supervision": _supervision_signature(record),
                "execution": _execution_signature(record),
                "visits": visits,
            }
            self.stats.unique_records += 1

    # -- écriture --------------------------------------------------------------

    def write(self, obj: Mapping[str, Any] | Record) -> bool:
        """Écrit (ou fusionne) une transition. Retourne True si ligne écrite."""
        record = obj if isinstance(obj, Record) else validate_record(dict(obj))
        key = record_dedup_key(record)
        policy_sig = _policy_signature(record)
        supervision_sig = _supervision_signature(record)
        execution_sig = _execution_signature(record)
        if key in self._index:
            existing = self._index[key]
            if existing["policy"] != policy_sig or existing["supervision"] != supervision_sig:
                raise InconsistentDuplicateError(
                    f"doublon incohérent pour {key}: policy/supervision divergent entre occurrences"
                )
            if existing["execution"] != execution_sig:
                raise InconsistentDuplicateError(
                    f"doublon incohérent pour {key}: execution divergent "
                    f"(observable_result/next_state_hash) — P4b"
                )
            existing.setdefault("visits", []).append(self._visit_of(record))
            self.stats.duplicates_merged += 1
            self._dirty_keys.add(key)
            return False
        self._fh.write(record.to_jsonl() + "\n")
        self._fh.flush()
        self._index[key] = {
            "policy": policy_sig,
            "supervision": supervision_sig,
            "execution": execution_sig,
            "visits": [self._visit_of(record)],
        }
        self.stats.lines_written += 1
        self.stats.unique_records += 1
        return True

    @staticmethod
    def _visit_of(record: Record) -> dict[str, Any]:
        """P4/M3 (audit tagi-5) : la visite embarque split ET versions — une
        occurrence fusionnée reste localisable par pool (read_all(split=…)
        passe par provenance, mais les visites gardent la traçabilité complète)."""
        visit = {
            "source": record.provenance.source,
            "split": record.provenance.split,
            "generator_version": record.provenance.generator_version,
            "oracle_version": record.provenance.oracle_version,
        }
        for k, v in record.provenance.extra.items():
            if k != "visits":
                visit[k] = v
        return visit

    # -- cycle de vie -----------------------------------------------------------

    def flush(self) -> None:
        self._fh.flush()

    def close(self) -> WriteStats:
        """Ferme le fichier et matérialise les fusions de visites.

        Passe unique de réécriture streaming (fichier temp + remplacement
        atomique) : pour chaque clé ayant fusionné des doublons, la ligne
        reçoit ``provenance.visits`` = liste complète des occurrences. Les
        lignes sans doublon restent octet-identiques.
        """
        if self._fh.closed:
            return self.stats  # idempotent
        self._fh.flush()
        self._fh.close()
        if self._dirty_keys:
            self._materialize_visits()
        if self.stats.content_sha256 is None:
            self.stats.content_sha256 = sha256_bytes(self.path.read_bytes())
        return self.stats

    def _materialize_visits(self) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        written: set[DedupKey] = set()
        with self.path.open("r", encoding="utf-8") as src, tmp.open("w", encoding="utf-8") as dst:
            for raw in src:
                if not raw.strip():
                    continue
                record = validate_record(json.loads(raw))
                key = record_dedup_key(record)
                if key in self._dirty_keys:
                    if key in written:
                        continue  # P6 : doublon préexistant — une seule ligne fusionnée survit
                    written.add(key)
                    visits = self._index[key]["visits"]
                    obj = json.loads(raw)
                    obj["provenance"]["visits"] = visits
                    dst.write(canonical_json(obj) + "\n")
                else:
                    dst.write(raw if raw.endswith("\n") else raw + "\n")
        tmp.replace(self.path)
        self._dirty_keys.clear()

    def __enter__(self) -> "JsonlRecordWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Manifests scellés
# ---------------------------------------------------------------------------


def seal_manifest(
    manifest: Mapping[str, Any],
    hash_fields: tuple[str, ...] = ("manifest_sha256",),
    exclude: tuple[str, ...] = ("timing",),
) -> dict:
    """Ajoute ``manifest_sha256`` = sha256 du contenu hors champs de scellage.

    ``exclude`` (m3, audit tagi-5) : champs NON scellés — timings/mesures
    non reproductibles (``timing`` par défaut) : ils ne doivent jamais faire
    échouer une vérification byte-identical inter-machines.

    Le manifest scellé est reproductible : mêmes données ⇒ même hash ⇒ gel
    vérifiable avant entraînement confirmatoire (§7.3).
    """
    core = {k: v for k, v in manifest.items() if k not in hash_fields and k not in exclude}
    sealed = dict(manifest)
    sealed["manifest_sha256"] = sha256_hex(core)
    return sealed


def verify_manifest(
    manifest: Mapping[str, Any],
    hash_fields: tuple[str, ...] = ("manifest_sha256",),
    exclude: tuple[str, ...] = ("timing",),
) -> bool:
    core = {k: v for k, v in manifest.items() if k not in hash_fields and k not in exclude}
    expected = manifest.get("manifest_sha256")
    return isinstance(expected, str) and expected == sha256_hex(core)


# ---------------------------------------------------------------------------
# Contenu addressable
# ---------------------------------------------------------------------------


def write_blob(store_dir: str | os.PathLike[str], obj: Any) -> str:
    """Écrit un objet JSON content-addressed (sha256 complet). Retourne le hash."""
    store = Path(store_dir)
    store.mkdir(parents=True, exist_ok=True)
    content = canonical_json(obj)
    digest = sha256_hex(content)
    target = store / f"{digest}.json"
    if not target.exists():
        target.write_text(content, encoding="utf-8")
    elif target.read_text(encoding="utf-8") != content:  # pragma: no cover
        raise SchemaError(f"collision de content-addressing sur {digest}")
    return digest


def write_jsonl(records: Iterator[Record] | list[Record], path: str | os.PathLike[str]) -> WriteStats:
    with JsonlRecordWriter(path, append=False) as w:
        for record in records:
            w.write(record)
        return w.stats
