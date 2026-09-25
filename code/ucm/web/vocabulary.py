"""Vocabulaire web/2.0 FERMÉ du compilateur web v2 — bloqueur n°1 de la sonde.

La sonde v1 (commit 90a303a) a documenté: "vocabulaire P2 fermé ('view' hors
vocab TGK)". Le compilateur web possède donc SON PROPRE vocabulaire fermé,
indépendant du vocabulaire TGK de ucm/p2/ (réservé à l'équipe tagi, non
modifié). Toute sortie du compilateur v2 est validée contre ce vocabulaire
avant d'être retournée: la sortie est STRICTEMENT typée JSON.

Types d'entités (extensions vs TGK en MAJUSCULES logique):
    view, form, field, button, select, option, LINK   (link = nouveauté v2)

Prédicats de relations:
    on_view, part_of, option_of, submits              (alignés siw.py §observe)

Actions candidates:
    TYPE, CLICK, SELECT, NAVIGATE, STOP
    NAVIGATE × lien = la nouveauté v2 (les liens étaient le trou de couverture
    v1: 476/512 actionnables non couverts sur w3schools-forms).

Typage strict par action (arg -> type d'entité obligatoire):
    TYPE -> field ; CLICK -> button ; SELECT -> option ;
    NAVIGATE -> link ; STOP -> None.
"""
from __future__ import annotations

SCHEMA_VERSION = "web/2.0"

ENTITY_TYPES = frozenset({"view", "form", "field", "button", "select",
                          "option", "link"})
PREDICATES = frozenset({"on_view", "part_of", "option_of", "submits"})
ACTIONS = frozenset({"TYPE", "CLICK", "SELECT", "NAVIGATE", "STOP"})
GOAL_PREDICATES = frozenset({"VIEW"})

# arg d'un candidat doit référencer une entité de ce type (STOP: None).
ACTION_ARG_TYPE = {
    "TYPE": "field",
    "CLICK": "button",
    "SELECT": "option",
    "NAVIGATE": "link",
    "STOP": None,
}

_TOP_KEYS = {"schema_version", "entities", "relations", "goal", "candidates"}
_ENTITY_KEYS = {"id", "type", "attrs", "view"}          # 'view' optionnel
_SCALAR = (str, bool, int, type(None))                  # attrs strictement typés


class PolicyInputError(ValueError):
    """Violation du schéma strict web/2.0."""


def validate_policy_input(pi: dict) -> list[str]:
    """Valide un policy_input contre le vocabulaire fermé web/2.0.

    Retourne la liste des violations (vide = valide). Ne lève jamais:
    utilisée en introspection; `ensure_valid` lève.
    """
    errs: list[str] = []
    bad = lambda m: errs.append(m)  # noqa: E731

    if not isinstance(pi, dict) or set(pi) != _TOP_KEYS:
        bad(f"top-level keys != {_sorted(_TOP_KEYS)}: {sorted(pi) if isinstance(pi, dict) else type(pi)}")
        return errs
    if pi["schema_version"] != SCHEMA_VERSION:
        bad(f"schema_version != {SCHEMA_VERSION!r}: {pi['schema_version']!r}")

    ids: set[str] = set()
    types: dict[str, str] = {}
    if not isinstance(pi["entities"], list):
        bad("entities: pas une liste")
        return errs
    for i, e in enumerate(pi["entities"]):
        if not isinstance(e, dict) or not set(e) <= _ENTITY_KEYS or not {"id", "type", "attrs"} <= set(e):
            bad(f"entity[{i}]: clés invalides {sorted(e) if isinstance(e, dict) else type(e)}")
            continue
        if not isinstance(e["id"], str) or not e["id"]:
            bad(f"entity[{i}].id: pas une chaîne non vide")
            continue
        if e["id"] in ids:
            bad(f"entity id dupliqué (dédup manquante): {e['id']!r}")
            continue
        ids.add(e["id"])
        if e["type"] not in ENTITY_TYPES:
            bad(f"entity {e['id']!r}: type hors vocabulaire {e['type']!r}")
            continue
        types[e["id"]] = e["type"]
        if not isinstance(e["attrs"], dict):
            bad(f"entity {e['id']!r}.attrs: pas un dict")
            continue
        for k, v in e["attrs"].items():
            if not isinstance(k, str):
                bad(f"entity {e['id']!r}.attrs: clé non-str {k!r}")
            if not isinstance(v, _SCALAR) or isinstance(v, float):
                bad(f"entity {e['id']!r}.attrs[{k!r}]: valeur non typée str|bool|int|None: {v!r}")
        if "view" in e and not (isinstance(e["view"], str) or e["view"] is None):
            bad(f"entity {e['id']!r}.view: pas str|None")

    if not isinstance(pi["relations"], list):
        bad("relations: pas une liste")
        return errs
    for i, r in enumerate(pi["relations"]):
        if not isinstance(r, dict) or set(r) != {"subj", "pred", "obj"}:
            bad(f"relation[{i}]: clés != subj/pred/obj")
            continue
        if r["pred"] not in PREDICATES:
            bad(f"relation[{i}]: prédicat hors vocabulaire {r['pred']!r}")
        for side in ("subj", "obj"):
            if r[side] not in ids:
                bad(f"relation[{i}].{side}: référence inconnue {r[side]!r}")

    g = pi["goal"]
    if (not isinstance(g, dict) or set(g) != {"predicate", "args"}
            or g.get("predicate") not in GOAL_PREDICATES
            or not isinstance(g.get("args"), dict)
            or not all(isinstance(k, str) and isinstance(v, str)
                       for k, v in g.get("args", {}).items())):
        bad(f"goal mal formé (placeholder attendu VIEW/{{view}}): {g!r}")

    if not isinstance(pi["candidates"], list):
        bad("candidates: pas une liste")
        return errs
    n_stop = 0
    for i, c in enumerate(pi["candidates"]):
        if not isinstance(c, dict) or set(c) != {"action", "arg"}:
            bad(f"candidate[{i}]: clés != action/arg")
            continue
        a = c["action"]
        if a not in ACTIONS:
            bad(f"candidate[{i}]: action hors vocabulaire {a!r}")
            continue
        want = ACTION_ARG_TYPE[a]
        if a == "STOP":
            n_stop += 1
            if c["arg"] is not None:
                bad(f"candidate[{i}] STOP: arg != None")
            continue
        if not isinstance(c["arg"], str) or c["arg"] not in types:
            bad(f"candidate[{i}] {a}: arg inconnu {c['arg']!r}")
        elif types.get(c["arg"]) != want:
            bad(f"candidate[{i}] {a}: arg {c['arg']!r} de type "
                f"{types.get(c['arg'])!r}, {want!r} attendu")
    if n_stop != 1:
        bad(f"candidats STOP: {n_stop} trouvé(s), 1 attendu")
    elif pi["candidates"] and pi["candidates"][-1]["action"] != "STOP":
        bad("STOP n'est pas le dernier candidat")
    return errs


def ensure_valid(pi: dict) -> dict:
    """Valide et retourne pi; lève PolicyInputError si violation."""
    errs = validate_policy_input(pi)
    if errs:
        raise PolicyInputError(
            f"policy_input invalide ({len(errs)} violation(s)): "
            + "; ".join(errs[:8]))
    return pi


def _sorted(fs: frozenset) -> list[str]:
    return sorted(fs)
