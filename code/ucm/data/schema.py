"""UCM data schema — WS-B (tagi-2).

Record contract: PLAN.md §3.3 (figé), dérivé de la spec §5.1/§5.2.
Une ligne JSONL = une transition. Conforme à l'observation ``TinyGraphKey.observe()``
(WS-A, ucm/env/tinygraph.py) :

- ``goal``      : ``{"predicate", "args"}`` — REACH{room} / HAVE{object} / AT{object,room}
- ``entities``  : ``[{"id", "type", "attrs"}]`` — types room/agent/key/parcel/door
- ``relations`` : ``[{"subj", "pred", "obj"}]`` — adjacent (les deux sens),
  connects, unlocks, at, held
- ``candidates``: ``[{"action", "arg"}]`` — K = R+6, ``arg=null`` ssi STOP,
  ordre canonique env (randomisation en aval)

L'obs de l'env porte aussi un ``schema_version`` top-level : il est retiré au
moment d'embarquer l'obs dans le record (redondant avec celui de la ligne) —
voir :func:`policy_input_from_obs`.

Canaux (spec §5.1) — séparation stricte :
- ``policy_input``  : but, entités/attributs observables, relations, candidats
                      syntaxiques. Allowlist stricte, aucun champ interdit (§5.1).
- ``execution``     : action exécutée (indice de candidat), résultat observable,
                      hash de l'état suivant.
- ``supervision``   : A*, d*, reachable. Jamais une entrée modèle.
- ``provenance``    : layout canonique, hash état-but, split, source, versions.

Conventions V0 :
- Distinction absent/inconnu/false (§5.1) : champ structurellement absent =
  OMITTÉ ; inconnu explicite = ``null`` ; ``false`` = booléen signifiant.
- Références de candidats : entiers, indices dans ``policy_input.candidates``
  (permutation-safe) — ``execution.action_ref`` et
  ``supervision.optimal_actions``.
- Hashs env : ``layout_hash`` = ``env.layout.layout_hash()`` (16 hex) ;
  ``execution.next_state_hash`` = ``env.state_hash()`` de l'état suivant (16 hex).
  ``provenance.state_goal_hash`` : :func:`make_state_goal_hash`. Le contenu
  addressable (blobs, manifests) utilise le sha256 complet (64 hex).

Toute extension d'allowlist passe par les registres documentés
(``register_entity_type``, ``register_relation_type``) et doit être tracée
dans PLAN.md §6 « Déviations ».
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional

SCHEMA_VERSION = "0.2"

# ---------------------------------------------------------------------------
# Helpers : canonical JSON + hashing
# ---------------------------------------------------------------------------

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_HEX16_RE = re.compile(r"^[0-9a-f]{16}$")


def canonical_json(obj: Any) -> str:
    """Sérialisation canonique : clés triées, séparateurs compacts, ensure_ascii."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(obj: Any) -> str:
    """SHA-256 (64 hex) du JSON canonique de ``obj`` (ou de la chaîne)."""
    if isinstance(obj, str):
        payload = obj
    else:
        payload = canonical_json(obj)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(_SHA256_RE.match(value))


def is_hex16(value: Any) -> bool:
    """Hash court env : layout_hash(), state_hash() — 16 hex (64 bits)."""
    return isinstance(value, str) and bool(_HEX16_RE.match(value))


def hex16(obj: Any) -> str:
    """Troncature 64 bits du sha256 canonique — convention env (16 hex)."""
    return sha256_hex(obj)[:16]


def make_state_goal_hash(layout_hash: str, state_hash: str, goal: Mapping[str, Any]) -> str:
    """Hash canonique (layout, état physique, but) — contrat tagi-1.

    ``state_hash`` = ``env.state_hash()`` (16 hex, état seul lié au layout).
    Le hash combine layout + état + but ; 64 bits suffisent au volume V0
    (<10^6 couples, collision attendue ~3e-8).
    """
    if not (is_hex16(layout_hash) or is_sha256(layout_hash)):
        raise SchemaError(f"layout_hash: 16 ou 64 hex requis, reçu {layout_hash!r}")
    if not (is_hex16(state_hash) or is_sha256(state_hash)):
        raise SchemaError(f"state_hash: 16 ou 64 hex requis, reçu {state_hash!r}")
    return hex16([layout_hash, state_hash, dict(goal)])


# ---------------------------------------------------------------------------
# Erreurs
# ---------------------------------------------------------------------------


class SchemaError(ValueError):
    """Violation du contrat de schéma (validation stricte, jamais de correction silencieuse)."""


# ---------------------------------------------------------------------------
# Anti-fuite : champs interdits dans policy_input (spec §5.1)
# ---------------------------------------------------------------------------

# Tokens exacts interdits (après normalisation lowercase / '-'->'_').
_FORBIDDEN_TOKENS = frozenset(
    {
        "reward",
        "progress",
        "distance",
        "distance_to_goal",
        "d_star",
        "dstar",
        "success",
        "terminal",
        "plan",
        "next_observation",
        "next_state",
        "next_raw_observation",
        "next_raw_observation_ref",
        "expert",
        "optimal",
        "optimal_actions",
        "optimal_action_refs",
        "optimal_mask",
        "mask",
        "timestep",
        "time_step",
        "step_count",
        "budget",
        "budget_counter",
        "episode",
        "episode_number",
        "episode_id",
        "split",
        "generator",
        "generator_id",
        "generator_version",
        "domain",
        "domain_id",
        "source",
        "oracle",
        "oracle_version",
        "layout_hash",
        "state_goal_hash",
        "seed",
        # NB: "label" (singulier) retiré — attribut UI observable légitime en
        # SIW (§12.6, labels textuels arbitraires) ; "labels"/"y_true" (pluriel,
        # cibles BC) restent interdits.
        "labels",
        "y_true",
    }
)

# Sous-chaînes interdites (robustesse aux variantes) : toute clé normalisée
# contenant l'un de ces fragments est rejetée.
_FORBIDDEN_SUBSTRINGS = (
    "reward",
    "progress",
    "distance",
    "d_star",
    "dstar",
    "success",
    "terminal",
    "optimal",
    "expert",
    "timestep",
    "time_step",
    "budget",
    "episode",
    "split",
    "generator",
    "domain",
    "oracle",
    "layout_hash",
    "state_goal_hash",
    "seed",
    "mask",
    "next_",
)


def _norm_key(key: str) -> str:
    return key.strip().lower().replace("-", "_")


def forbidden_key_violation(obj: Any, _path: str = "policy_input") -> list[str]:
    """Retourne la liste des chemins de clés interdites trouvés dans ``obj``.

    Scan récursif clés/valeurs. Utilisé par la validation stricte et par les
    tests anti-fuite des trois WS (§3.5).
    """
    violations: list[str] = []
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            if isinstance(key, str):
                norm = _norm_key(key)
                if norm in _FORBIDDEN_TOKENS or any(s in norm for s in _FORBIDDEN_SUBSTRINGS):
                    violations.append(f"{_path}.{key}")
            violations.extend(forbidden_key_violation(value, f"{_path}.{key}"))
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            violations.extend(forbidden_key_violation(value, f"{_path}[{i}]"))
    return violations


# ---------------------------------------------------------------------------
# Buts (format env : {"predicate", "args"} ; spec §4.2)
# ---------------------------------------------------------------------------

GOAL_PREDICATES = ("REACH", "HAVE", "AT")
_GOAL_ARGS: dict[str, frozenset[str]] = {
    "REACH": frozenset({"room"}),
    "HAVE": frozenset({"object"}),
    "AT": frozenset({"object", "room"}),
}


def register_goal_predicate(predicate: str, arg_names: Iterable[str]) -> None:
    """Extension des prédicats de but (SIW §12.6 : VIEW/SET/CHOOSE/SUBMITTED)."""
    global GOAL_PREDICATES
    if predicate not in GOAL_PREDICATES:
        GOAL_PREDICATES = GOAL_PREDICATES + (predicate,)
    _GOAL_ARGS[predicate] = frozenset(arg_names)


@dataclass(frozen=True)
class Goal:
    predicate: str
    args: Mapping[str, str]

    def __post_init__(self) -> None:
        validate_goal(self.to_dict())

    def to_dict(self) -> dict:
        return {"predicate": self.predicate, "args": dict(self.args)}


def validate_goal(obj: Any) -> None:
    if not isinstance(obj, dict):
        raise SchemaError(f"goal: dict requis, reçu {type(obj).__name__}")
    if set(obj) != {"predicate", "args"}:
        raise SchemaError(f"goal: clés exactes ['args','predicate'] attendues, reçu {sorted(obj)}")
    predicate = obj["predicate"]
    if predicate not in GOAL_PREDICATES:
        raise SchemaError(f"goal.predicate: {predicate!r} ∉ {GOAL_PREDICATES}")
    args = obj["args"]
    if not isinstance(args, dict):
        raise SchemaError("goal.args: dict requis")
    expected = _GOAL_ARGS[predicate]
    if set(args) != set(expected):
        raise SchemaError(f"goal {predicate}: args exactes {sorted(expected)} attendues, reçu {sorted(args)}")
    for k in sorted(args):
        v = args[k]
        if not isinstance(v, str) or not v:
            raise SchemaError(f"goal.args.{k}: référence non-vide requise, reçu {v!r}")


# ---------------------------------------------------------------------------
# Entités et relations (policy_input, format env)
# ---------------------------------------------------------------------------

# Allowlist V0 des types d'entités -> clés autorisées dans "attrs".
ENTITY_TYPES: dict[str, frozenset[str]] = {
    "room": frozenset(),
    "agent": frozenset(),
    "key": frozenset(),
    "parcel": frozenset(),
    "door": frozenset({"locked"}),
}

# Allowlist V0 des prédicats de relation -> (types subj autorisés, types obj).
RELATION_PREDS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "adjacent": (("room",), ("room",)),
    "connects": (("door",), ("room",)),
    "unlocks": (("key",), ("door",)),
    "at": (("agent", "key", "parcel"), ("room",)),
    "held": (("key", "parcel"), ("agent",)),
}


def register_entity_type(entity_type: str, allowed_attrs: Iterable[str]) -> None:
    """Extension d'allowlist (tracée dans PLAN.md §6)."""
    ENTITY_TYPES[entity_type] = frozenset(allowed_attrs)


def register_relation_type(pred: str, subj_types: tuple[str, ...], obj_types: tuple[str, ...]) -> None:
    RELATION_PREDS[pred] = (subj_types, obj_types)


# Clés top-level optionnelles par type d'entité (SIW : "view" structurel).
ENTITY_TOP_KEYS: dict[str, frozenset[str]] = {}
# Cardinalités d'entités requises PAR DOMAINE (I1, tranchage tagi-1 18:29) :
# la signature = un type d'entité propre au monde ; présence de la signature
# ⇒ application de ses cardinalités. Mondes mélangés ⇒ erreur explicite.
WORLD_REQUIRED_ENTITIES: dict[str, frozenset[str]] = {
    "room": frozenset({"agent", "key", "parcel", "door"}),  # TinyGraphKey (§4.1)
    "view": frozenset(),  # SIW (§12.6) : aucune cardinalité (agent virtuel)
}


def register_world(signature_entity_type: str, required_entity_types: Iterable[str]) -> None:
    """Extension par domaine (I1) — remplacer/ajouter une signature de monde."""
    WORLD_REQUIRED_ENTITIES[signature_entity_type] = frozenset(required_entity_types)
# Entités virtuelles référencées par les relations sans être listées
# (SIW : "agent" dans current_view/filled/chosen).
VIRTUAL_ENTITIES: dict[str, str] = {}


def register_entity_top_keys(entity_type: str, keys: Iterable[str]) -> None:
    ENTITY_TOP_KEYS[entity_type] = frozenset(keys)


def register_virtual_entity(entity_id: str, entity_type: str) -> None:
    VIRTUAL_ENTITIES[entity_id] = entity_type


@dataclass(frozen=True)
class Entity:
    id: str
    type: str
    attrs: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"id": self.id, "type": self.type, "attrs": dict(self.attrs)}


@dataclass(frozen=True)
class Relation:
    subj: str
    pred: str
    obj: str

    def to_dict(self) -> dict:
        return {"subj": self.subj, "pred": self.pred, "obj": self.obj}


def validate_entity(obj: Any) -> Entity:
    if not isinstance(obj, dict):
        raise SchemaError(f"entity: dict requis, reçu {type(obj).__name__}")
    allowed_keys = {"id", "type", "attrs"} | ENTITY_TOP_KEYS.get(obj.get("type", ""), frozenset())
    extra_keys = set(obj) - allowed_keys
    if extra_keys:
        raise SchemaError(
            f"entity: clés non autorisées {sorted(extra_keys)} — attendues {sorted(allowed_keys)}"
        )
    missing_keys = {"id", "type", "attrs"} - set(obj)
    if missing_keys:
        raise SchemaError(f"entity: champs requis manquants {sorted(missing_keys)}")
    entity_type = obj["type"]
    if entity_type not in ENTITY_TYPES:
        raise SchemaError(f"entity.type: {entity_type!r} ∉ {sorted(ENTITY_TYPES)}")
    entity_id = obj["id"]
    if not isinstance(entity_id, str) or not entity_id:
        raise SchemaError(f"entity.id: non-vide requis, reçu {entity_id!r}")
    attrs = obj["attrs"]
    if not isinstance(attrs, dict):
        raise SchemaError(f"entity {entity_id!r}.attrs: dict requis")
    allowed = ENTITY_TYPES[entity_type]
    extra = set(attrs) - set(allowed)
    if extra:
        raise SchemaError(
            f"entity {entity_id!r} ({entity_type}): attrs non autorisés {sorted(extra)} (allowlist {sorted(allowed)})"
        )
    viol = forbidden_key_violation(obj)
    if viol:
        raise SchemaError(f"entity {entity_id!r}: clés interdites {viol}")
    top_view = obj.get("view")
    if "view" in ENTITY_TOP_KEYS.get(entity_type, frozenset()) and "view" in obj:
        # None = explicitement sans vue (ex. widgets de dialogue) — absent≠inconnu
        if top_view is not None and (not isinstance(top_view, str) or not top_view):
            raise SchemaError(f"entity {entity_id!r}.view: référence de vue str non-vide ou null requise")
    if entity_type == "door":
        if "locked" not in attrs:
            raise SchemaError(f"entity {entity_id!r}.attrs: 'locked' requis (absent ≠ false, §5.1)")
        if not isinstance(attrs["locked"], bool):
            raise SchemaError(f"entity {entity_id!r}.attrs.locked: bool requis")
    return Entity(id=entity_id, type=entity_type, attrs=dict(attrs))


def validate_relation(obj: Any, entity_types: Mapping[str, str]) -> Relation:
    if not isinstance(obj, dict):
        raise SchemaError(f"relation: dict requis, reçu {type(obj).__name__}")
    if set(obj) != {"subj", "pred", "obj"}:
        raise SchemaError(f"relation: clés exactes ['obj','pred','subj'] attendues, reçu {sorted(obj)}")
    pred = obj["pred"]
    if pred not in RELATION_PREDS:
        raise SchemaError(f"relation.pred: {pred!r} ∉ {sorted(RELATION_PREDS)}")
    subj, ob = obj["subj"], obj["obj"]
    for name, ref, expected_types in (("subj", subj, None), ("obj", ob, None)):
        if not isinstance(ref, str) or not ref:
            raise SchemaError(f"relation.{name}: référence non-vide requise, reçu {ref!r}")
    subj_types, obj_types = RELATION_PREDS[pred]
    actual_subj = entity_types.get(subj, VIRTUAL_ENTITIES.get(subj))
    actual_obj = entity_types.get(ob, VIRTUAL_ENTITIES.get(ob))
    if actual_subj is None or actual_obj is None:
        raise SchemaError(f"relation {pred}: référence inconnue ({subj!r} → {ob!r})")
    if actual_subj not in subj_types:
        raise SchemaError(f"relation {pred}: subj {subj!r} de type {actual_subj!r}, ∈ {list(subj_types)} attendu")
    if actual_obj not in obj_types:
        raise SchemaError(f"relation {pred}: obj {ob!r} de type {actual_obj!r}, ∈ {list(obj_types)} attendu")
    return Relation(subj=subj, pred=pred, obj=ob)


def _validate_relation_structure(relations: tuple[Relation, ...], entity_types: Mapping[str, str]) -> None:
    """Cohérence structurelle de l'observation (positions, porte, topologie).

    - adjacent : chaque arête non orientée présente dans les DEUX sens, sans doublon.
    - connects : exactement 2 (une par extrémité de l'unique porte).
    - unlocks : exactement 1 (clé → porte).
    - at/held : l'agent a exactement un « at » ; chaque objet exactement un
      « at » XOR un « held » (vers l'agent) — absent/inconnu/false distincts.
    """
    by_pred: dict[str, list[Relation]] = {}
    for r in relations:
        by_pred.setdefault(r.pred, []).append(r)

    adj = by_pred.get("adjacent", [])
    directed = {(r.subj, r.obj) for r in adj}
    if len(directed) != len(adj):
        raise SchemaError("relations adjacent: doublon (subj,obj)")
    for r in adj:
        if (r.obj, r.subj) not in directed:
            raise SchemaError(f"relations adjacent: sens manquant ({r.obj}→{r.subj})")
        if r.subj == r.obj:
            raise SchemaError("relations adjacent: self-loop")

    # M1 (audit tagi-5 vague 2) : topologie connexe (§4.1 « graphe connecté »)
    rooms = [e for e, t in entity_types.items() if t == "room"]
    neighbors: dict[str, set[str]] = {r: set() for r in rooms}
    for r in adj:
        neighbors[r.subj].add(r.obj)
        neighbors[r.obj].add(r.subj)
    if rooms:
        seen = {rooms[0]}
        stack = [rooms[0]]
        while stack:
            cur = stack.pop()
            for nb in neighbors[cur]:
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        if len(seen) != len(rooms):
            unreachable = sorted(set(rooms) - seen)
            raise SchemaError(f"relations adjacent: topologie non connexe — pièces isolées {unreachable}")

    connects = by_pred.get("connects", [])
    doors = [e for e, t in entity_types.items() if t == "door"]
    if len(doors) != 1:
        raise SchemaError(f"policy_input: exactement une porte requise (§4.1), trouvé {len(doors)}")
    door = doors[0]
    endpoints = {r.obj for r in connects}
    if len(connects) != 2 or len(endpoints) != 2:
        raise SchemaError("relations connects: exactement 2 extrémités distinctes requises")
    if any(r.subj != door for r in connects):
        raise SchemaError("relations connects: subj doit être la porte")
    # P1c (audit tagi-5 vague 2) : la porte porte sur une ARÊTE — ses deux
    # extrémités doivent être reliées par une relation adjacent (§4.1).
    end_list = sorted(endpoints)
    if not any(r.subj == end_list[0] and r.obj == end_list[1] for r in adj) and not any(
        r.subj == end_list[1] and r.obj == end_list[0] for r in adj
    ):
        raise SchemaError(
            f"relations connects: la porte relie {end_list} qui ne sont PAS adjacentes (§4.1) — la porte porte sur une arête"
        )

    unlocks = by_pred.get("unlocks", [])
    if len(unlocks) != 1:
        raise SchemaError("relations unlocks: exactement 1 (clé → porte) requise")

    agents = [e for e, t in entity_types.items() if t == "agent"]
    if len(agents) != 1:
        raise SchemaError(f"policy_input: exactement un agent requis, trouvé {len(agents)}")
    agent = agents[0]
    ats = by_pred.get("at", [])
    helds = by_pred.get("held", [])
    agent_at = [r for r in ats if r.subj == agent]
    if len(agent_at) != 1:
        raise SchemaError("relations at: l'agent doit avoir exactement une position")
    for obj in (e for e, t in entity_types.items() if t in ("key", "parcel")):
        n_at = sum(1 for r in ats if r.subj == obj)
        n_held = sum(1 for r in helds if r.subj == obj)
        if n_at + n_held != 1:
            raise SchemaError(
                f"relations: objet {obj!r} doit avoir exactement un 'at' ou un 'held' "
                f"(reçu at={n_at}, held={n_held}) — posé/porté sans ambiguïté"
            )
        for r in helds:
            if r.obj != agent:
                raise SchemaError("relations held: obj doit être l'agent")


# ---------------------------------------------------------------------------
# Candidats (spec §4.3, §5.3) — format env {"action","arg"}
# ---------------------------------------------------------------------------

# Liste MUTABLE : l'extension (register_action_type) doit être visible par
# toutes les références importées (SIW §12.6) — pas de rebinding.
ACTION_TYPES = ["MOVE", "PICK", "DROP", "UNLOCK", "STOP"]
_ARG_ENTITY_TYPES: dict[str, object] = {
    "MOVE": "room",
    "PICK": ("key", "parcel"),
    "DROP": ("key", "parcel"),
    "UNLOCK": "door",
    "STOP": None,
}


def register_action_type(action: str, arg_entity_types: object) -> None:
    """Extension du registre d'actions (SIW §12.6 / M-V1a) — documentée PLAN §6.

    ``arg_entity_types`` : type d'entité unique (str) ou tuple de types pour
    l'argument ``arg`` ; ``None`` si l'action n'en prend pas (STOP).
    """
    if action not in ACTION_TYPES:
        ACTION_TYPES.append(action)
    _ARG_ENTITY_TYPES[action] = arg_entity_types


@dataclass(frozen=True)
class Action:
    action: str
    arg: Optional[str] = None

    def __post_init__(self) -> None:
        if self.action not in ACTION_TYPES:
            raise SchemaError(f"action.action: {self.action!r} ∉ {ACTION_TYPES}")
        if self.action == "STOP":
            if self.arg is not None:
                raise SchemaError("action STOP: arg doit être null")
        elif not isinstance(self.arg, str) or not self.arg:
            raise SchemaError(f"action {self.action}: arg non-vide requis")

    def semantic(self) -> tuple[str, Optional[str]]:
        """Identité sémantique indépendante de l'ordre des candidats."""
        return (self.action, self.arg)

    def to_dict(self) -> dict:
        return {"action": self.action, "arg": self.arg}


def validate_candidates(obj: Any, entity_types: Mapping[str, str], require_k: bool = True) -> tuple[Action, ...]:
    """Valide la liste des candidats : énumération goal-blind K = R + 6 (§4.3)."""
    if not isinstance(obj, list) or not obj:
        raise SchemaError("candidates: liste non vide requise")
    actions: list[Action] = []
    seen_semantic: set[tuple[str, Optional[str]]] = set()
    n_rooms = sum(1 for t in entity_types.values() if t == "room")
    for i, cand in enumerate(obj):
        if not isinstance(cand, dict) or set(cand) != {"action", "arg"}:
            raise SchemaError(f"candidates[{i}]: clés exactes ['action','arg'] attendues (contrat tagi-1)")
        action = Action(action=cand["action"], arg=cand["arg"])
        expected_type = _ARG_ENTITY_TYPES[action.action]
        if expected_type is not None:
            if action.arg not in entity_types:
                raise SchemaError(f"candidates[{i}]: arg inconnu {action.arg!r}")
            actual = entity_types[action.arg]
            ok = actual in expected_type if isinstance(expected_type, tuple) else actual == expected_type
            if not ok:
                raise SchemaError(
                    f"candidates[{i}] {action.action}: {action.arg!r} de type {actual!r}, {expected_type!r} attendu"
                )
        if action.semantic() in seen_semantic:
            raise SchemaError(f"candidates: doublon sémantique {action.semantic()}")
        seen_semantic.add(action.semantic())
        actions.append(action)
    if require_k:
        expected_k = n_rooms + 6
        if len(actions) != expected_k:
            raise SchemaError(f"candidates: K = R+6 = {expected_k} attendu, reçu {len(actions)}")
    return tuple(actions)


# ---------------------------------------------------------------------------
# Canal policy_input
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyInput:
    goal: Goal
    entities: tuple[Entity, ...]
    relations: tuple[Relation, ...]
    candidates: tuple[Action, ...]

    def to_dict(self) -> dict:
        return {
            "goal": self.goal.to_dict(),
            "entities": [e.to_dict() for e in self.entities],
            "relations": [r.to_dict() for r in self.relations],
            "candidates": [a.to_dict() for a in self.candidates],
        }

    @property
    def entity_types(self) -> dict[str, str]:
        return {e.id: e.type for e in self.entities}

    @property
    def action_semantics(self) -> tuple[tuple[str, Optional[str]], ...]:
        return tuple(a.semantic() for a in self.candidates)


def validate_policy_input(obj: Any, require_k: bool = True) -> PolicyInput:
    """Validation stricte : allowlist exacte + anti-fuite §5.1 + cohérence structurelle."""
    if not isinstance(obj, dict):
        raise SchemaError(f"policy_input: dict requis, reçu {type(obj).__name__}")
    expected = {"goal", "entities", "relations", "candidates"}
    if set(obj) != expected:
        raise SchemaError(f"policy_input: clés exactes {sorted(expected)} attendues, reçu {sorted(obj)}")
    viol = forbidden_key_violation(obj)
    if viol:
        raise SchemaError(f"policy_input: champs interdits (§5.1) {viol}")

    validate_goal(obj["goal"])

    # P11c (audit tagi-5) : erreurs de type propres (jamais de TypeError brut)
    if not isinstance(obj["entities"], list):
        raise SchemaError(f"policy_input.entities: liste requise, reçu {type(obj['entities']).__name__}")
    if not isinstance(obj["relations"], list):
        raise SchemaError(f"policy_input.relations: liste requise, reçu {type(obj['relations']).__name__}")

    entities: list[Entity] = []
    entity_types: dict[str, str] = {}
    for ent_obj in obj["entities"]:
        ent = validate_entity(ent_obj)
        if ent.id in entity_types:
            raise SchemaError(f"entity.id dupliqué: {ent.id!r}")
        entity_types[ent.id] = ent.type
        entities.append(ent)
    n_by_type: dict[str, int] = {}
    for t in entity_types.values():
        n_by_type[t] = n_by_type.get(t, 0) + 1
    # I1 : cardinalités par domaine (registre, tranchage tagi-1 18:29)
    signatures = [sig for sig in WORLD_REQUIRED_ENTITIES if n_by_type.get(sig, 0) > 0]
    if len(signatures) > 1:
        raise SchemaError(f"policy_input: mondes mélangés {signatures} — un seul domaine par observation")
    if signatures:
        for required in sorted(WORLD_REQUIRED_ENTITIES[signatures[0]]):
            if n_by_type.get(required, 0) != 1:
                raise SchemaError(
                    f"policy_input: exactement une entité {required!r} requise "
                    f"(domaine {signatures[0]!r}), trouvé {n_by_type.get(required, 0)}"
                )
        if signatures[0] == "room" and n_by_type.get("room", 0) < 2:
            raise SchemaError("policy_input: au moins 2 pièces requises (§4.1)")

    relations = tuple(validate_relation(r, entity_types) for r in obj["relations"])
    # Structure V0 (adjacent symétrique/connects/unlocks/at-held) : uniquement
    # pour le monde TinyGraphKey (signature §4.1). Les autres mondes (SIW)
    # valident leur structure dans leur module dédié (siw_validate_structure).
    if any(t == "door" for t in entity_types.values()):
        _validate_relation_structure(relations, entity_types)
    candidates = validate_candidates(obj["candidates"], entity_types, require_k=require_k)

    # le but référence des entités existantes du bon type (spec §4.2)
    goal = obj["goal"]
    for arg, types in (("room", ("room",)), ("object", ("key", "parcel"))):
        if arg in goal["args"]:
            actual = entity_types.get(goal["args"][arg])
            if actual not in types:
                raise SchemaError(f"goal.args.{arg}: {goal['args'][arg]!r} de type {actual!r}, {list(types)} attendu")

    return PolicyInput(
        goal=Goal(predicate=goal["predicate"], args=dict(goal["args"])),
        entities=tuple(entities),
        relations=relations,
        candidates=candidates,
    )


def policy_input_from_obs(obs: Mapping[str, Any], require_k: bool = True) -> dict:
    """Extrait policy_input d'une observation env (retire son schema_version).

    P3 (audit tagi-5) : aucune clé top-level inconnue n'est ignorée
    silencieusement (§5.3 erreur visible) — l'obs doit contenir exactement
    {schema_version, goal, entities, relations, candidates}.
    """
    expected_obs = {"schema_version", "goal", "entities", "relations", "candidates"}
    extra = set(obs) - expected_obs
    if extra:
        raise SchemaError(f"obs: clés top-level inconnues {sorted(extra)} — drop silencieux interdit (§5.3)")
    pi = {k: obs[k] for k in ("goal", "entities", "relations", "candidates")}
    validate_policy_input(pi, require_k=require_k)
    return pi


# ---------------------------------------------------------------------------
# Canaux execution / supervision / provenance
# ---------------------------------------------------------------------------

OBSERVABLE_RESULTS = ("valid", "invalid")
PROVENANCE_SOURCES = ("oracle", "perturbation", "dagger")
# m7 (audit tagi-5 vague 2) : enum fermé des splits — toute nouvelle cellule
# doit être déclarée ici ET dans le manifest de splits avant d'apparaître.
SPLIT_ENUM = ("train", "val", "test_g1", "test_g2", "test_g3", "test_g4")


@dataclass(frozen=True)
class Execution:
    action_ref: int
    observable_result: str
    next_state_hash: str


@dataclass(frozen=True)
class Supervision:
    optimal_actions: tuple[int, ...]
    d_star: Optional[int]
    reachable: bool


@dataclass(frozen=True)
class Provenance:
    layout_hash: str
    state_goal_hash: str
    split: str
    source: str
    generator_version: str
    oracle_version: str
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "layout_hash": self.layout_hash,
            "state_goal_hash": self.state_goal_hash,
            "split": self.split,
            "source": self.source,
            "generator_version": self.generator_version,
            "oracle_version": self.oracle_version,
        }
        d.update(self.extra)
        return d


def validate_execution(obj: Any, n_candidates: int) -> Execution:
    if not isinstance(obj, dict):
        raise SchemaError(f"execution: dict requis, reçu {type(obj).__name__}")
    if set(obj) != {"action_ref", "observable_result", "next_state_hash"}:
        raise SchemaError(f"execution: clés exactes ['action_ref','next_state_hash','observable_result'] attendues, reçu {sorted(obj)}")
    action_ref = obj["action_ref"]
    if not isinstance(action_ref, int) or isinstance(action_ref, bool) or not (0 <= action_ref < n_candidates):
        raise SchemaError(f"execution.action_ref: indice de candidat ∈ [0,{n_candidates}) requis, reçu {action_ref!r}")
    result = obj["observable_result"]
    if result not in OBSERVABLE_RESULTS:
        raise SchemaError(f"execution.observable_result: {result!r} ∉ {OBSERVABLE_RESULTS}")
    if not is_hex16(obj["next_state_hash"]):
        raise SchemaError("execution.next_state_hash: hex16 (env.state_hash()) requis")
    return Execution(action_ref=action_ref, observable_result=result, next_state_hash=obj["next_state_hash"])


def validate_supervision(obj: Any, n_candidates: int) -> Supervision:
    if not isinstance(obj, dict):
        raise SchemaError(f"supervision: dict requis, reçu {type(obj).__name__}")
    if set(obj) != {"optimal_actions", "d_star", "reachable"}:
        raise SchemaError(f"supervision: clés exactes ['d_star','optimal_actions','reachable'] attendues, reçu {sorted(obj)}")
    reachable = obj["reachable"]
    if not isinstance(reachable, bool):
        raise SchemaError("supervision.reachable: bool requis")
    optimal = obj["optimal_actions"]
    if not isinstance(optimal, list) or any((not isinstance(i, int) or isinstance(i, bool)) for i in optimal):
        raise SchemaError("supervision.optimal_actions: liste d'indices entiers requis")
    if len(set(optimal)) != len(optimal):
        raise SchemaError("supervision.optimal_actions: indices dupliqués")
    if any(not (0 <= i < n_candidates) for i in optimal):
        raise SchemaError(f"supervision.optimal_actions: indices ∈ [0,{n_candidates}) requis")
    d_star = obj["d_star"]
    if not reachable:
        if optimal:
            raise SchemaError("supervision unreachable: aucune action experte ne doit être fabriquée (§4.6)")
        if d_star is not None:
            raise SchemaError("supervision unreachable: d_star doit être null")
    else:
        if not isinstance(d_star, int) or isinstance(d_star, bool) or d_star < 0:
            raise SchemaError("supervision reachable: d_star entier ≥ 0 requis")
        if not optimal:
            raise SchemaError("supervision reachable: A* non vide requis")
    return Supervision(optimal_actions=tuple(optimal), d_star=d_star, reachable=reachable)


def validate_provenance(obj: Any) -> Provenance:
    if not isinstance(obj, dict):
        raise SchemaError(f"provenance: dict requis, reçu {type(obj).__name__}")
    required = {"layout_hash", "state_goal_hash", "split", "source", "generator_version", "oracle_version"}
    missing = required - set(obj)
    if missing:
        raise SchemaError(f"provenance: champs requis manquants {sorted(missing)}")
    for h in ("layout_hash", "state_goal_hash"):
        if not is_hex16(obj[h]):
            raise SchemaError(f"provenance.{h}: hex16 (convention env, 16 hex) requis")
    if not isinstance(obj["split"], str) or not obj["split"]:
        raise SchemaError("provenance.split: str non vide requis")
    if obj["split"] not in SPLIT_ENUM:
        raise SchemaError(f"provenance.split: {obj['split']!r} ∉ {list(SPLIT_ENUM)} (enum fermé m7)")
    if obj["source"] not in PROVENANCE_SOURCES:
        raise SchemaError(f"provenance.source: {obj['source']!r} ∉ {PROVENANCE_SOURCES}")
    for v in ("generator_version", "oracle_version"):
        if not isinstance(obj[v], str) or not obj[v]:
            raise SchemaError(f"provenance.{v}: version non vide requise")
    extra = {k: v for k, v in obj.items() if k not in required}
    # NB : episode/step sont légitimes en provenance (§5.1 « numéro de pas ») ;
    # le canal provenance n'entre jamais dans le modèle, pas de scan interdit ici.
    return Provenance(
        layout_hash=obj["layout_hash"],
        state_goal_hash=obj["state_goal_hash"],
        split=obj["split"],
        source=obj["source"],
        generator_version=obj["generator_version"],
        oracle_version=obj["oracle_version"],
        extra=extra,
    )


# ---------------------------------------------------------------------------
# Record complet (PLAN §3.3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Record:
    schema_version: str
    policy_input: PolicyInput
    execution: Execution
    supervision: Supervision
    provenance: Provenance

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "policy_input": self.policy_input.to_dict(),
            "execution": {
                "action_ref": self.execution.action_ref,
                "observable_result": self.execution.observable_result,
                "next_state_hash": self.execution.next_state_hash,
            },
            "supervision": {
                "optimal_actions": list(self.supervision.optimal_actions),
                "d_star": self.supervision.d_star,
                "reachable": self.supervision.reachable,
            },
            "provenance": self.provenance.to_dict(),
        }

    def to_jsonl(self) -> str:
        return canonical_json(self.to_dict())


def validate_record(obj: Any, require_k: bool = True) -> Record:
    """Validation stricte d'une transition complète + cohérence inter-canaux."""
    if not isinstance(obj, dict):
        raise SchemaError(f"record: dict requis, reçu {type(obj).__name__}")
    expected = {"schema_version", "policy_input", "execution", "supervision", "provenance"}
    if set(obj) != expected:
        raise SchemaError(f"record: clés exactes {sorted(expected)} attendues, reçu {sorted(obj)}")
    if obj["schema_version"] != SCHEMA_VERSION:
        raise SchemaError(f"record.schema_version: {SCHEMA_VERSION!r} requis, reçu {obj['schema_version']!r}")

    policy_input = validate_policy_input(obj["policy_input"], require_k=require_k)
    n_cand = len(policy_input.candidates)
    stop_indices = [i for i, a in enumerate(policy_input.candidates) if a.action == "STOP"]

    execution = validate_execution(obj["execution"], n_cand)
    supervision = validate_supervision(obj["supervision"], n_cand)
    # P2 (audit tagi-5 vague 2) : d*=0 ⇒ A* = {STOP} EXACTEMENT (§4.6) —
    # l'égalité d'ensemble, pas la seule appartenance.
    if supervision.reachable and supervision.d_star == 0:
        if set(supervision.optimal_actions) != set(stop_indices):
            raise SchemaError(
                f"supervision d*=0: A* doit être exactement {{STOP}} (§4.6), reçu {sorted(supervision.optimal_actions)}"
            )
    if supervision.reachable and supervision.d_star > 0 and any(i in stop_indices for i in supervision.optimal_actions):
        raise SchemaError("supervision d*>0: STOP ∉ A* (toute action de A* fait décroître d*, §4.6)")
    provenance = validate_provenance(obj["provenance"])
    return Record(
        schema_version=SCHEMA_VERSION,
        policy_input=policy_input,
        execution=execution,
        supervision=supervision,
        provenance=provenance,
    )


def record_from_dict(obj: Mapping[str, Any], require_k: bool = True) -> Record:
    return validate_record(obj, require_k=require_k)


def record_from_jsonl(line: str, require_k: bool = True) -> Record:
    return record_from_dict(json.loads(line), require_k=require_k)
