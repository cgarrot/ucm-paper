"""Lecture JSONL des transitions UCM — WS-B.

- Validation systématique à la lecture (défaut) : un fichier corrompu lève
  ``SchemaError`` avec ligne — jamais de troncature silencieuse (§5.3).
- Itération en flux : ``iter_records`` consomme le fichier ligne à ligne.
- Résolution du contenu addressable : ``load_blob`` pour les observations
  hors-ligne référencées par hash.
- Filtres : split, layout, source — utiles pour construire les vues d'entraînement.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Optional

from .schema import PROVENANCE_SOURCES, Record, SchemaError, SPLIT_ENUM, canonical_json, is_sha256, record_from_dict, sha256_hex


def iter_records(
    path: str | Path,
    *,
    require_k: bool = True,
    split: Optional[str] = None,
    layout_hash: Optional[str] = None,
    source: Optional[str] = None,
) -> Iterator[Record]:
    """Itère les records d'un fichier JSONL (toujours validés), avec filtres.

    m4/P8a (audit tagi-5) : filtres split/source illégaux (fantôme/typo)
    lèvent ValueError plutôt que de rendre silencieusement zéro résultat.
    """
    if split is not None and split not in SPLIT_ENUM:
        raise ValueError(f"split inconnu {split!r} — enum {list(SPLIT_ENUM)} (filtre fantôme interdit)")
    if source is not None and source not in PROVENANCE_SOURCES:
        raise ValueError(
            f"source inconnue {source!r} — enum {list(PROVENANCE_SOURCES)} (filtre fantôme interdit)"
        )
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, 1):
            if not raw.strip():
                continue
            try:
                record = record_from_dict(json.loads(raw), require_k=require_k)
            except SchemaError as exc:
                raise SchemaError(f"{path}:{line_no}: {exc}") from exc
            except json.JSONDecodeError as exc:
                raise SchemaError(f"{path}:{line_no}: JSON invalide: {exc}") from exc
            except (TypeError, KeyError, ValueError) as exc:
                # P11c (audit tagi-5) : toute erreur de type/structure est convertie
                # en SchemaError annotée ligne — discipline d'erreur unique.
                raise SchemaError(f"{path}:{line_no}: structure invalide ({type(exc).__name__}): {exc}") from exc
            if split is not None and record.provenance.split != split:
                continue
            if layout_hash is not None and record.provenance.layout_hash != layout_hash:
                continue
            if source is not None and record.provenance.source != source:
                continue
            yield record


def read_all(path: str | Path, **kwargs: Any) -> list[Record]:
    return list(iter_records(path, **kwargs))


def load_blob(store_dir: str | Path, content_hash: str) -> Any:
    """Charge un blob content-addressed écrit par ``writer.write_blob``.

    M4/P7 (audit tagi-5) : le contenu re-haché doit correspondre au nom —
    toute divergence (fichier corrompu/renommé) est une erreur explicite.
    """
    if not is_sha256(content_hash):
        raise SchemaError(f"content hash sha256 (64 hex) requis, reçu {content_hash!r}")
    target = Path(store_dir) / f"{content_hash}.json"
    if not target.exists():
        raise FileNotFoundError(f"blob {content_hash} absent de {store_dir}")
    content = target.read_text(encoding="utf-8")
    actual = sha256_hex(content)
    if actual != content_hash:
        raise SchemaError(
            f"blob {content_hash}: contenu corrompu — sha256 réel {actual} ≠ nom (M4)"
        )
    return json.loads(content)


def count_lines(path: str | Path) -> int:
    return sum(1 for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip())

def verify_jsonl(path: str | Path) -> dict:
    """m5/P8b (audit tagi-5) : vérifie l'intégrité d'un fichier JSONL.

    Re-hashe le fichier et valide chaque ligne — retourne
    {path, sha256, lines, valid, errors} (errors vide si sain).
    """
    import hashlib as _h
    from pathlib import Path as _P
    p = _P(path)
    digest = _h.sha256(p.read_bytes()).hexdigest()
    errors: list[str] = []
    n = 0
    for line_no, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        n += 1
        try:
            record_from_dict(json.loads(raw))
        except (SchemaError, ValueError, TypeError) as exc:
            errors.append(f"{line_no}: {exc}")
    return {"path": str(p), "sha256": digest, "lines": n, "valid": not errors, "errors": errors[:20]}
