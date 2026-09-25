"""Programme DSL: TGK (TinyGraphKey) — le natif devient UNE INSTANCE.

Fidélité exacte à ucm/env/tinygraph.py (sémantique verbatim):
  MOVE r: garde = r≠agent ∧ passable(agent,r)  [passable = adjacent ∧ (¬paire-porte ∨ ¬verrouillée)]
  PICK o: garde = carried=None ∧ room(o)=agent
  DROP o: garde = carried=o
  UNLOCK d: garde = verrouillée ∧ agent∈door_rooms ∧ carried=key
  STOP: valide toujours; succès = but
  Buts: REACH r | HAVE o | AT o r
Layout tables: rooms (liste), adj (paires), door_rooms (paire).
"""
from __future__ import annotations

TGK_DSL_PROGRAM = {
    "schema": "ucm-dsl/1.0",
    "world": "tgk",
    "state": ["agent", "carried", "key_room", "parcel_room", "door_locked"],
    "reset": {
        "agent": ["task", "init", "agent"],
        "carried": ["task", "init", "carried"],
        "key_room": ["if", ["eq", ["task", "init", "carried"], ["const", "key"]],
                     ["const", None], ["task", "init", "key"]],
        "parcel_room": ["if", ["eq", ["task", "init", "carried"], ["const", "parcel"]],
                        ["const", None], ["task", "init", "parcel"]],
        "door_locked": ["if", ["is_none", ["task", "init", "door_locked"]],
                        ["const", True], ["task", "init", "door_locked"]],
    },
    "state_check": {
        "agent_known": ["mem1", "rooms", ["field", "agent"]],
        "key_consistent": ["eq", ["eq", ["field", "carried"], ["const", "key"]],
                           ["is_none", ["field", "key_room"]]],
        "parcel_consistent": ["eq", ["eq", ["field", "carried"], ["const", "parcel"]],
                              ["is_none", ["field", "parcel_room"]]],
    },
    "actions": [
        {"name": "MOVE",
         "arg_source": ["each", "rooms"],
         "guard": [["not", ["eq", ["field", "agent"], ["param", "arg"]]],
                   ["or",
                    ["not", ["mem2", "door_pair",
                             ["field", "agent"], ["param", "arg"]]],
                    ["not", ["field", "door_locked"]]],
                   ["mem2", "adj", ["field", "agent"], ["param", "arg"]]],
         "effects": [["set", "agent", ["param", "arg"]]]},
        {"name": "PICK",
         "arg_source": ["const_list", ["key", "parcel"]],
         "guard": [["is_none", ["field", "carried"]],
                   ["eq", ["field", ["if", ["eq", ["param", "arg"], ["const", "key"]],
                                     ["const", "key_room"], ["const", "parcel_room"]]],
                    ["field", "agent"]]],
         "effects": [["set", "carried", ["param", "arg"]],
                     ["set", ["if", ["eq", ["param", "arg"], ["const", "key"]],
                              ["const", "key_room"], ["const", "parcel_room"]],
                      ["const", None]]]},
        {"name": "DROP",
         "arg_source": ["const_list", ["key", "parcel"]],
         "guard": [["eq", ["field", "carried"], ["param", "arg"]]],
         "effects": [["set", ["if", ["eq", ["param", "arg"], ["const", "key"]],
                              ["const", "key_room"], ["const", "parcel_room"]],
                      ["field", "agent"]],
                     ["set", "carried", ["const", None]]]},
        {"name": "UNLOCK",
         "arg_source": ["const_list", ["door0"]],
         "guard": [["field", "door_locked"],
                   ["mem1", "door_room_list", ["field", "agent"]],
                   ["eq", ["field", "carried"], ["const", "key"]]],
         "effects": [["set", "door_locked", ["const", False]]]},
    ],
    "goal": ["or",
             ["and", ["eq", ["task", "goal", "predicate"], ["const", "REACH"]],
              ["eq", ["field", "agent"], ["task", "goal", "args", "room"]]],
             ["and", ["eq", ["task", "goal", "predicate"], ["const", "HAVE"]],
              ["eq", ["field", "carried"], ["task", "goal", "args", "object"]]],
             ["and", ["eq", ["task", "goal", "predicate"], ["const", "AT"]],
              ["eq", ["field",
                      ["if", ["eq", ["task", "goal", "args", "object"], ["const", "key"]],
                       ["const", "key_room"], ["const", "parcel_room"]]],
               ["task", "goal", "args", "room"]]]],
}


def layout_from_native(layout) -> dict:
    """Tables DSL depuis un Layout natif (rooms/adj/door_rooms)."""
    adj = sorted({tuple(sorted((a, b))) for a, b in layout.edges})
    door = list(layout.door_rooms)   # TUPLE (x, y) — UNE paire, pas deux
    return {
        "rooms": list(layout.rooms),
        "adj": [list(p) for p in adj],
        "door_rooms": [door],          # tableau de paires (mem2)
        "door_pair": [door],           # idem, pour la garde MOVE
        "door_room_list": door,        # liste PLATE (mem1: agent ∈ pièces de la porte)
    }
