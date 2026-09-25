"""SIW — couche données V1 (M-V1a, spec §12.6 révisée).

ÉTAT : préparation. L'env+oracle SIW (WS-A) et la spéc normative arrivent ;
ce module pose les fondations env-indépendantes et sera ajusté AU COMMIT env
(conformance permanente, discipline V0 : les fixtures vivront sur le VRAI env).

Format candidat (confirmé tagi-1 17:48) : dicts {"action", "arg"} UNAIRES
comme V0 — TYPE copie la valeur depuis le but par référence (unaire sur le
champ cible), SELECT pointe l'OPTION (pas le select), précondition « select
non choisi ». STOP inchangé. Le schéma 0.2 reste structurellement identique :
seuls les REGISTRES s'étendent (mécanisme documenté §6 schema.py).

Isomorphisme (§12.6) : graphe de widgets TYPÉS + machine à états — jamais les
labels (générés aléatoirement par layout). Moteur : canonical_certificate_typed.
"""

from __future__ import annotations

from typing import Optional

from .schema import (
    SchemaError,
    register_action_type,
    register_entity_type,
    register_entity_top_keys,
    register_goal_predicate,
    register_relation_type,
    register_virtual_entity,
)
from .splits import canonical_certificate_typed, typed_isomorphic, wl_hash_typed

# ---------------------------------------------------------------------------
# Registres SIW (DRAFT — ajustés au commit env WS-A ; toute divergence tracée)
# ---------------------------------------------------------------------------

# Types d'entités widgets (§12.6) et leurs attributs observables allowlistés.
# Les LABELS sont des attributs observables (vocabulaire fermé + alias) mais
# N'ENTENT JAMAIS dans l'isomorphisme.
SIW_ENTITY_TYPES: dict[str, frozenset[str]] = {
    "view": frozenset(),
    "button": frozenset({"onclick_kind", "label"}),
    "field": frozenset({"filled", "label"}),
    "select": frozenset({"chosen_option_ref", "label"}),
    "option": frozenset({"label"}),
    "form": frozenset({"status", "label"}),
    "dialog": frozenset({"open", "label"}),
    "agent": frozenset(),  # virtuel (current_view/filled/chosen)
}

SIW_WIDGET_TYPES = ("view", "button", "field", "select", "option", "form", "dialog")
# onclick_kind = attribut STATIQUE de la FSM (§3.2) — entre dans l'isomorphisme
# (passe B tagi-5, MAJEUR D/E) via couleur composite type×kind des boutons.
SIW_BUTTON_KINDS = ("none", "submit", "confirm", "dismiss")

# 9 prédicats de relation (env ebaba96) — layout ET état mélangés dans l'obs ;
# SIW_LAYOUT_PREDS = structure du layout, SIW_STATE_PREDS = état courant.
SIW_RELATION_PREDS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "nav_edge": (("view",), ("view",)),
    "on_view": (("button", "field", "select", "form", "dialog"), ("view",)),
    "part_of": (("button", "field", "select"), ("form",)),
    "option_of": (("option",), ("select",)),
    "submits": (("button",), ("form",)),
    "in_dialog": (("button", "field", "select", "form"), ("dialog",)),
    "current_view": (("agent",), ("view",)),
    "filled": (("field",), ("agent",)),
    "chosen": (("option",), ("select",)),
}
SIW_LAYOUT_PREDS = ("nav_edge", "on_view", "part_of", "option_of", "submits", "in_dialog")
SIW_STATE_PREDS = ("current_view", "filled", "chosen")

# Buts SIW (env ebaba96)
SIW_GOALS: dict[str, tuple[str, ...]] = {
    "VIEW": ("view",),
    "SET": ("field",),
    "CHOOSE": ("select", "option"),
    "SUBMITTED": ("form",),
}

SIW_ACTIONS = ("NAVIGATE", "CLICK", "TYPE", "SELECT", "STOP")
SIW_ACTION_ARGS: dict[str, object] = {
    "NAVIGATE": ("view",),
    "CLICK": ("button",),
    "TYPE": ("field",),
    "SELECT": ("option",),
    "STOP": None,
}
SIW_K_BOUNDS = (30, 60)


def register_siw() -> None:
    """Applique les registres SIW (idempotent), conformes à l'env ebaba96."""
    for etype, attrs in SIW_ENTITY_TYPES.items():
        register_entity_type(etype, attrs)
    # clé top-level structurelle "view" pour les widgets non-option
    register_entity_top_keys("button", ("view",))
    register_entity_top_keys("field", ("view",))
    register_entity_top_keys("select", ("view",))
    register_entity_top_keys("form", ("view",))
    register_entity_top_keys("dialog", ("view",))
    register_virtual_entity("agent", "agent")
    for pred, (subj, obj) in SIW_RELATION_PREDS.items():
        register_relation_type(pred, subj, obj)
    for pred, args in SIW_GOALS.items():
        register_goal_predicate(pred, args)
    for action, arg_types in SIW_ACTION_ARGS.items():
        register_action_type(action, arg_types)


def siw_policy_input_from_obs(obs, require_k: bool = False) -> dict:
    """Extrait + valide policy_input d'une obs SIW (require_k=False : K=V+B+F+O+1,
    bornes 30-60 garanties par le générateur, vérifiées par siw_check_k)."""
    register_siw()
    from .schema import policy_input_from_obs as _base
    pi = _base(obs, require_k=require_k)
    siw_check_k(pi)
    siw_validate_structure(pi)
    return pi


def siw_check_k(pi: dict) -> int:
    """K = V+B+F+O+1 doit être dans les bornes 30-60 (§12.6 many-candidates)."""
    k = len(pi["candidates"])
    lo, hi = SIW_K_BOUNDS
    if not (lo <= k <= hi):
        raise SchemaError(f"SIW: K={k} hors bornes {SIW_K_BOUNDS} (§12.6)")
    return k


def siw_validate_structure(pi: dict) -> None:
    """Structure SIW : nav_edge symétrique, une current_view, un agent, chaque
    widget non-option sur une vue, options rattachées à leur select."""
    rels = pi["relations"]
    ents = {e["id"]: e for e in pi["entities"]}
    by_pred: dict[str, list] = {}
    for r in rels:
        by_pred.setdefault(r["pred"], []).append(r)
    nav = by_pred.get("nav_edge", [])
    directed = {(r["subj"], r["obj"]) for r in nav}
    if len(directed) != len(nav):
        raise SchemaError("SIW nav_edge: doublon directionnel")
    for r in nav:
        if (r["obj"], r["subj"]) not in directed:
            raise SchemaError(f"SIW nav_edge: sens manquant ({r['obj']}→{r['subj']})")
    if len(by_pred.get("current_view", [])) != 1:
        raise SchemaError("SIW: exactement une relation current_view requise")
    for e in pi["entities"]:
        if e["type"] == "option":
            continue
        if e["type"] != "view" and not any(
            (r["pred"] == "on_view" or r["pred"] == "in_dialog") and r["subj"] == e["id"]
            for r in rels
        ):
            raise SchemaError(f"SIW: widget {e['id']!r} sans vue ni dialogue (on_view/in_dialog requis)")
    for r in by_pred.get("option_of", []):
        if ents.get(r["obj"], {}).get("type") != "select":
            raise SchemaError("SIW option_of: obj doit être un select")


def siw_certificate(
    n_widgets: int,
    typed_edges: list[tuple[int, int, int]],
    widget_types_as_colors: list[int],
) -> str:
    """Hash INVARIANT du graphe de widgets typé (préfiltre, jamais canonique :
    les orbites massives de distracteurs rendent le certificat exact
    infaisable — voir wl_hash_typed). Exactitude par :func:`typed_isomorphic`."""
    return wl_hash_typed(n_widgets, typed_edges, widget_types_as_colors)


# ---------------------------------------------------------------------------
# Inventaire de faisabilité (§12.6 / consigne tagi-1 17:44) — AVANT gel
# ---------------------------------------------------------------------------
# À livrer dès l'env : couples (état-UI, but) uniques, couverture aux budgets
# k = 0 / 100 / 500 / 2000 / 10000, existence de la difficulté cible,
# manifest additif scellé (canon V0 9a19d8f4 jamais muté).


SIW_INVENTORY_BUDGETS = (0, 100, 500, 2000, 10000)


# ---------------------------------------------------------------------------
# Fixtures SIW — conformance permanente sur le VRAI env+oracle (discipline V0)
# ---------------------------------------------------------------------------


def make_siw_record(
    *,
    action_ref: int = 0,
    task: Optional[dict] = None,
    layout=None,
    split: str = "train",
    source: str = "oracle",
    provenance_extra: Optional[dict] = None,
    generator_version: str = "siw-fixtures-0.1",
    oracle_version: str = "siw-oracle-0.1",
    optimal_actions: Optional[list[int]] = None,
    d_star: Optional[int] = None,
    reachable: Optional[bool] = None,
) -> dict:
    """Une transition SIW réelle : reset → obs → solve → execute → record §3.3."""
    import random as _random

    from ..env.siw import SIW, generate_siw_layout
    from ..env.siw_oracle import solve as _solve
    from .schema import make_state_goal_hash, record_from_dict

    register_siw()
    if layout is None:
        layout = generate_siw_layout(_random.Random(7), n_views=4, with_dialog=True)
    if task is None:
        from ..env.siw import sample_task
        rng = _random.Random(3)
        state, goal = sample_task(rng, layout)
        task = {"init": {
            "view": state.view, "filled": sorted(state.filled),
            "chosen": dict(state.chosen), "dialog_open": state.dialog_open,
            "submitted": sorted(state.submitted),
        }, "goal": goal}
    env = SIW(layout)
    obs = env.reset(task)
    policy_input = siw_policy_input_from_obs(obs)

    state_before = env.state.copy()
    s_hash = env.state_hash()
    l_hash = layout.layout_hash()
    sg_hash = make_state_goal_hash(l_hash, s_hash, env.goal)
    supervision = _solve(layout, state_before, env.goal)

    result = env.execute(env.candidates()[action_ref])
    reachable_final = supervision["reachable"] if reachable is None else reachable
    d_final = (supervision["d_star"] if supervision["reachable"] else None) if d_star is None else d_star
    opt_final = list(optimal_actions) if optimal_actions is not None else list(supervision["optimal_actions"])
    if reachable_final is False and optimal_actions is None:
        opt_final = []
    record = {
        "schema_version": "0.2",
        "policy_input": policy_input,
        "execution": {
            "action_ref": action_ref,
            "observable_result": "valid" if result["valid"] else "invalid",
            "next_state_hash": env.state_hash(),
        },
        "supervision": {"optimal_actions": opt_final, "d_star": d_final, "reachable": reachable_final},
        "provenance": {
            "layout_hash": l_hash,
            "state_goal_hash": sg_hash,
            "split": split,
            "source": source,
            "generator_version": generator_version,
            "oracle_version": oracle_version,
        },
    }
    if provenance_extra:
        record["provenance"].update(provenance_extra)
    return record_from_dict(record, require_k=False).to_dict()


def _siw_node_color(entity: dict) -> int:
    """Couleur composite : type × onclick_kind pour les boutons (FSM §3.2).
    labels et attributs d'ÉTAT exclus (§12.6)."""
    type_index = {t: i for i, t in enumerate(SIW_WIDGET_TYPES)}
    if entity["type"] == "button":
        kind = entity.get("attrs", {}).get("onclick_kind", "none")
        ki = SIW_BUTTON_KINDS.index(kind) if kind in SIW_BUTTON_KINDS else len(SIW_BUTTON_KINDS)
        return len(SIW_WIDGET_TYPES) + ki  # couleurs composites bouton×kind
    return type_index.get(entity["type"], len(type_index) + len(SIW_BUTTON_KINDS))


def _siw_typed_graph_from_obs(pi: dict):
    """(n, edges, node_colors) du LAYOUT : arêtes structurelles uniquement
    (nav_edge/on_view/part_of/option_of/submits/in_dialog) + couleurs
    type×onclick_kind — relations d'ÉTAT et labels exclus (§12.6)."""
    register_siw()
    pred_index = {p: i for i, p in enumerate(SIW_LAYOUT_PREDS)}
    nodes = [e["id"] for e in pi["entities"]]
    idx = {eid: i for i, eid in enumerate(nodes)}
    node_colors = [_siw_node_color(e) for e in pi["entities"]]
    edges = []
    for r in pi["relations"]:
        if r["pred"] in pred_index:
            edges.append((idx[r["subj"]], idx[r["obj"]], pred_index[r["pred"]]))
    return len(nodes), edges, node_colors


def siw_layout_certificate_from_obs(pi: dict) -> str:
    """Hash invariant du layout SIW (préfiltre). Exactitude : siw_layout_isomorphic."""
    return wl_hash_typed(*_siw_typed_graph_from_obs(pi))


def siw_layout_isomorphic(pi_a: dict, pi_b: dict) -> bool:
    """Isomorphisme EXACT de layouts SIW (multigraphes typés, labels exclus)."""
    return typed_isomorphic(_siw_typed_graph_from_obs(pi_a), _siw_typed_graph_from_obs(pi_b))
