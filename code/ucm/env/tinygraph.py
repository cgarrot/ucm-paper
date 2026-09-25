"""TinyGraphKey — environnement V0 (spec §4).

Monde déterministe pleinement observable : graphe de R pièces (4..8 au train),
un agent, deux objets transportables (clé, colis), une arête porte verrouillable.
Buts REACH(room) / HAVE(object) / AT(object, room). Actions complètes MOVE/PICK/
DROP/UNLOCK/STOP, invalides payantes (no-op + 1 pas), STOP vérifié par
l'évaluateur uniquement. Horizon 64 décisions.

Contrat d'information : `observe()`/`candidates()` ne renvoient QUE policy_input.
Aucun champ interdit (spec §5.1) : pas de d*, reward, succès oracle, plan,
prochaine observation, timestep, numéro d'épisode ni identifiant de générateur.
"""

from __future__ import annotations

import hashlib
import json
from typing import Optional

ACTION_TYPES = ("MOVE", "PICK", "DROP", "UNLOCK", "STOP")
GOAL_PREDICATES = ("REACH", "HAVE", "AT")
HORIZON_DEFAULT = 64

# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #


class LayoutError(ValueError):
    pass


class Layout:
    """Graphe de pièces + porte. Les IDs sont des références de liaison locales."""

    def __init__(self, rooms: list[str], edges: list[tuple[str, str]],
                 door_edge: int, meta: Optional[dict] = None):
        self.rooms = list(rooms)
        self.edges = [(a, b) for a, b in edges]
        self.door_edge = door_edge
        self.meta = dict(meta or {})

        if len(set(self.rooms)) != len(self.rooms):
            raise LayoutError("room ids must be unique")
        if not (2 <= len(self.rooms) <= 64):
            raise LayoutError("2..64 rooms expected")
        room_set = set(self.rooms)
        for a, b in self.edges:
            if a not in room_set or b not in room_set:
                raise LayoutError("edge references unknown room")
            if a == b:
                raise LayoutError("no self-loop")
        norm = {frozenset(e) for e in self.edges}
        if len(norm) != len(self.edges):
            raise LayoutError("no duplicate edge")
        if not (0 <= door_edge < len(self.edges)):
            raise LayoutError("door_edge out of range")
        if not self._is_connected():
            raise LayoutError("graph must be connected")

        self._adj: dict[str, set[str]] = {r: set() for r in self.rooms}
        for a, b in self.edges:
            self._adj[a].add(b)
            self._adj[b].add(a)
        self.door_rooms = tuple(self.edges[door_edge])

    def _is_connected(self) -> bool:
        if not self.edges:
            return len(self.rooms) == 1
        seen = {self.rooms[0]}
        stack = [self.rooms[0]]
        while stack:
            cur = stack.pop()
            for a, b in self.edges:
                if a == cur and b not in seen:
                    seen.add(b)
                    stack.append(b)
                elif b == cur and a not in seen:
                    seen.add(a)
                    stack.append(a)
        return len(seen) == len(self.rooms)

    def adjacent(self, a: str, b: str) -> bool:
        return b in self._adj[a]

    def degree(self, room: str) -> int:
        return len(self._adj[room])

    def is_bridge(self, edge_index: int) -> bool:
        """Une arête est un pont si sa suppression déconnecte le graphe."""
        if not (0 <= edge_index < len(self.edges)):
            raise LayoutError("edge_index out of range")  # jamais False silencieux
        edges_wo = [e for i, e in enumerate(self.edges) if i != edge_index]
        seen = {self.rooms[0]}
        stack = [self.rooms[0]]
        while stack:
            cur = stack.pop()
            for a, b in edges_wo:
                if a == cur and b not in seen:
                    seen.add(b)
                    stack.append(b)
                elif b == cur and a not in seen:
                    seen.add(a)
                    stack.append(a)
        return len(seen) != len(self.rooms)

    def canonical(self) -> str:
        payload = {
            "rooms": sorted(self.rooms),
            "edges": sorted(sorted(e) for e in self.edges),
            "door": sorted(self.edges[self.door_edge]),
        }
        return json.dumps(payload, sort_keys=True)

    def layout_hash(self) -> str:
        return hashlib.sha256(self.canonical().encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# État physique + tâche
# --------------------------------------------------------------------------- #


class PhysicalState:
    __slots__ = ("agent", "carried", "key_room", "parcel_room", "door_locked")

    def __init__(self, agent: str, carried: Optional[str],
                 key_room: Optional[str], parcel_room: Optional[str],
                 door_locked: bool):
        self.agent = agent
        self.carried = carried
        self.key_room = key_room
        self.parcel_room = parcel_room
        self.door_locked = door_locked

    def key(self):
        return (self.agent, self.carried, self.key_room,
                self.parcel_room, self.door_locked)

    def __eq__(self, other):
        return isinstance(other, PhysicalState) and self.key() == other.key()

    def __hash__(self):
        return hash(self.key())

    def copy(self) -> "PhysicalState":
        return PhysicalState(*self.key())


def goal_satisfied(state: PhysicalState, goal: dict) -> bool:
    """Vérité de l'évaluateur. `goal` : {"predicate", "args"}. Stricte :
    toute référence d'objet inconnue lève (pas d'alias silencieux)."""
    pred, args = goal["predicate"], goal.get("args", {})
    if pred == "REACH":
        return state.agent == args["room"]
    if pred == "HAVE":
        if args["object"] not in ("key", "parcel"):
            raise ValueError(f"unknown object {args['object']!r}")
        return state.carried == args["object"]
    if pred == "AT":
        if args["object"] == "key":
            loc = state.key_room
        elif args["object"] == "parcel":
            loc = state.parcel_room
        else:
            raise ValueError(f"unknown object {args['object']!r}")
        return loc == args["room"]  # porté ⇒ loc None ≠ room (spec §4.2)
    raise ValueError(f"unknown predicate {pred}")


def task_goal(pred: str, **args) -> dict:
    if pred not in GOAL_PREDICATES:
        raise ValueError(f"unknown predicate {pred}")
    return {"predicate": pred, "args": args}


# --------------------------------------------------------------------------- #
# Environnement
# --------------------------------------------------------------------------- #

OBJ_ROOM = {"key": "key_room", "parcel": "parcel_room"}


class TinyGraphKey:
    """MDP déterministe. Chaque décision (STOP compris) coûte un pas."""

    def __init__(self, layout: Layout, horizon: int = HORIZON_DEFAULT):
        self.layout = layout
        self.horizon = horizon
        self.door_id = "door0"
        self.goal: Optional[dict] = None
        self.state: Optional[PhysicalState] = None
        self.step_count = 0
        self.terminal = False
        self.last_result: Optional[str] = None

    # -- cycle de vie -------------------------------------------------------- #

    def reset(self, task: dict) -> dict:
        init, goal = task["init"], task["goal"]
        self._validate_goal(goal)
        carried = init.get("carried")  # None | "key" | "parcel"
        self.goal = goal
        self.state = PhysicalState(
            agent=init["agent"],
            carried=carried,
            key_room=None if carried == "key" else init["key"],
            parcel_room=None if carried == "parcel" else init["parcel"],
            door_locked=init.get("door_locked", True),
        )
        self.step_count = 0
        self.terminal = False
        self.last_result = None
        self._check_state()
        return self.observe()

    def _validate_goal(self, goal: dict):
        """Validation stricte du but : prédicat connu, clés d'args exactes,
        références ∈ entités du layout (spec §4.5 : types respectés)."""
        if not isinstance(goal, dict) or set(goal) != {"predicate", "args"}:
            raise LayoutError("goal: clés exactes ['predicate','args'] requises")
        pred, args = goal["predicate"], goal["args"]
        expected = {"REACH": {"room"}, "HAVE": {"object"},
                    "AT": {"object", "room"}}.get(pred)
        if expected is None:
            raise LayoutError(f"unknown predicate {pred!r}")
        if not isinstance(args, dict) or set(args) != expected:
            raise LayoutError(f"goal args: clés exactes {sorted(expected)} requises")
        if "object" in args and args["object"] not in ("key", "parcel"):
            raise LayoutError(f"unknown object {args['object']!r}")
        for ref in (args.get("room"),):
            if ref is not None and ref not in self.layout._adj:
                raise LayoutError(f"unknown room {ref!r}")

    def _check_state(self):
        s = self.state
        lay = self.layout
        for r in (s.agent, s.key_room, s.parcel_room):
            if r is not None and r not in lay._adj:
                raise LayoutError("state references unknown room")
        if s.carried not in (None, "key", "parcel"):
            raise LayoutError("carried must be None|key|parcel")
        if (s.carried == "key") != (s.key_room is None):
            raise LayoutError("inconsistent key carrying")
        if (s.carried == "parcel") != (s.parcel_room is None):
            raise LayoutError("inconsistent parcel carrying")

    # -- observation (policy_input UNIQUEMENT) -------------------------------- #

    def observe(self) -> dict:
        s, lay = self.state, self.layout
        entities = [{"id": r, "type": "room", "attrs": {}} for r in lay.rooms]
        entities.append({"id": "agent", "type": "agent", "attrs": {}})
        entities.append({"id": "key", "type": "key", "attrs": {}})
        entities.append({"id": "parcel", "type": "parcel", "attrs": {}})
        entities.append({"id": self.door_id, "type": "door",
                         "attrs": {"locked": bool(s.door_locked)}})

        relations = []
        for i, (a, b) in enumerate(lay.edges):
            relations.append({"subj": a, "pred": "adjacent", "obj": b})
            relations.append({"subj": b, "pred": "adjacent", "obj": a})
        for r in lay.door_rooms:
            relations.append({"subj": self.door_id, "pred": "connects", "obj": r})
        relations.append({"subj": "key", "pred": "unlocks", "obj": self.door_id})
        relations.append({"subj": "agent", "pred": "at", "obj": s.agent})
        for obj in ("key", "parcel"):
            loc = getattr(s, OBJ_ROOM[obj])
            if loc is not None:
                relations.append({"subj": obj, "pred": "at", "obj": loc})
            else:
                relations.append({"subj": obj, "pred": "held", "obj": "agent"})

        return {
            "schema_version": "0.2",
            "entities": entities,
            "relations": relations,
            "goal": {"predicate": self.goal["predicate"],
                     "args": dict(self.goal.get("args", {}))},
            "candidates": self.candidates(),
        }

    def candidates(self) -> list[dict]:
        """Énumération syntaxique goal-blind. K = R + 6. Ordre canonique stable ;
        la randomisation d'ordre (train ET test) se fait en aval, avec liaison
        des labels."""
        out = [{"action": "MOVE", "arg": r} for r in self.layout.rooms]
        out += [{"action": "PICK", "arg": "key"}, {"action": "PICK", "arg": "parcel"},
                {"action": "DROP", "arg": "key"}, {"action": "DROP", "arg": "parcel"},
                {"action": "UNLOCK", "arg": self.door_id},
                {"action": "STOP", "arg": None}]
        return out

    # -- dynamique ------------------------------------------------------------ #

    def _passable(self, a: str, b: str) -> bool:
        lay = self.layout
        if not lay.adjacent(a, b):
            return False
        if frozenset((a, b)) == frozenset(lay.door_rooms) and self.state.door_locked:
            return False
        return True

    def _is_valid(self, action: dict) -> bool:
        s, lay = self.state, self.layout
        kind, arg = action["action"], action["arg"]
        if kind == "MOVE":
            return arg != s.agent and self._passable(s.agent, arg)
        if kind == "PICK":
            return (arg in ("key", "parcel") and s.carried is None
                    and getattr(s, OBJ_ROOM[arg]) == s.agent)
        if kind == "DROP":
            return arg in ("key", "parcel") and s.carried == arg
        if kind == "UNLOCK":
            return (arg == self.door_id and s.door_locked
                    and s.agent in lay.door_rooms and s.carried == "key")
        if kind == "STOP":
            return True
        return False

    def execute(self, action: dict) -> dict:
        if self.terminal:
            raise RuntimeError("episode is terminal")
        if action not in self.candidates():
            raise ValueError(f"action not in candidate list: {action}")
        s = self.state
        kind, arg = action["action"], action["arg"]
        valid = self._is_valid(action)

        if kind == "STOP":
            self.step_count += 1  # chaque décision, STOP inclus, coûte un pas (§4.4)
            self.terminal = True
            self.last_result = "success" if goal_satisfied(s, self.goal) \
                else "premature_stop"
            # Pas de contrôle d'horizon ici : un STOP en décision H est
            # évalué (succès/prématuré), jamais timeout.
            return self._step_result(valid=True, terminal=True,
                                     result=self.last_result)

        if valid:
            if kind == "MOVE":
                s.agent = arg
                if s.carried is not None:
                    pass  # l'objet porté reste porté (position None)
            elif kind == "PICK":
                s.carried = arg
                setattr(s, OBJ_ROOM[arg], None)
            elif kind == "DROP":
                setattr(s, OBJ_ROOM[arg], s.agent)
                s.carried = None
            elif kind == "UNLOCK":
                s.door_locked = False  # définitif ; la clé reste portée

        self.step_count += 1
        if self.step_count >= self.horizon:
            self.terminal = True
            self.last_result = "timeout"
            return self._step_result(valid=valid, terminal=True,
                                     result="timeout")
        return self._step_result(valid=valid, terminal=False,
                                 result="ok" if valid else "invalid")

    def _step_result(self, valid: bool, terminal: bool, result: str) -> dict:
        return {"valid": valid, "terminal": terminal, "result": result,
                "step": self.step_count}

    # -- côté évaluateur UNIQUEMENT (jamais policy_input) --------------------- #

    def goal_satisfied(self) -> bool:
        return goal_satisfied(self.state, self.goal)

    def state_hash(self) -> str:
        payload = json.dumps(
            {"layout": self.layout.layout_hash(), "state": list(self.state.key())},
            sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]
