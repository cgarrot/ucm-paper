"""ucm.web — compilateur page HTML -> policy_input UCM.

v1 (sonde, commit 90a303a): ucm/web/compiler_probe.py — LECTURE SEULE,
conservé pour la reproductibilité de la baseline 5.86%.
v2: compiler_v2.py + vocabulary.py + make_snapshots.py — couvre les LIENS,
traite vocabulaire/ids dupliqués/représentation des liens.
"""
from ucm.web.compiler_v2 import (build_report, compile_page, compile_pages,
                                 recompute_v1_on_pages, resolve_href,
                                 write_artifacts)
from ucm.web.vocabulary import (ACTIONS, ENTITY_TYPES, PREDICATES,
                                PolicyInputError, ensure_valid,
                                validate_policy_input)

__all__ = [
    "compile_page", "compile_pages", "build_report", "recompute_v1_on_pages",
    "resolve_href", "write_artifacts", "validate_policy_input",
    "ensure_valid", "PolicyInputError", "ENTITY_TYPES", "PREDICATES",
    "ACTIONS",
]
