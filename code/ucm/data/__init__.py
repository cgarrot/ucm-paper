"""WS-B — données UCM : schéma, I/O JSONL, splits scellés, inventaire.

Façade stable pour WS-C (et tout consommateur) :

    from ucm.data import Record, iter_records, validate_record

    records = list(iter_records("artifacts/data/m0-transitions.jsonl", split="train"))
    pi = records[0].policy_input          # canaux strictement séparés (§5.1)
    x = pi.entities, pi.relations, pi.goal, pi.candidates

Modules :
- schema    : validation stricte des 4 canaux + anti-fuite §5.1 + hashing canonique
- writer    : JSONL + dédup sémantique (visites conservées) + manifests scellés
- reader    : lecture validée en flux + filtres split/layout + blobs addressables
- splits    : isomorphisme exact (WL + canonical labeling), pools, G2, manifest
- inventory : comptages §7.2, faisabilité G4, composants G2 — avant tout gel
- pipeline  : orchestration M0 (generate → splits → records → inventaire)
- fixtures  : records d'exemple sur le VRAI env+oracle (conformance permanente)
"""

from .schema import (  # noqa: F401
    SCHEMA_VERSION,
    Action,
    Entity,
    Goal,
    PolicyInput,
    Provenance,
    Record,
    Relation,
    SchemaError,
    canonical_json,
    forbidden_key_violation,
    make_state_goal_hash,
    policy_input_from_obs,
    record_from_dict,
    record_from_jsonl,
    sha256_hex,
    validate_policy_input,
    validate_record,
)
from .reader import count_lines, iter_records, load_blob, read_all  # noqa: F401
from .writer import (  # noqa: F401
    JsonlRecordWriter,
    seal_manifest,
    verify_manifest,
    write_blob,
    write_jsonl,
)
