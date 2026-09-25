"""Tensorizer P2 — tensorize_obs avec vocabs étendus (types ctx_*, relation
observed_effect). Copie fidèle de ucm.model.tensorize.tensorize_obs, seules
les vocabs changent (le tronc P2 est frais — pas de compat canon)."""
from __future__ import annotations

import numpy as np

from ucm.model.fixtures import ACTION_TYPES
from ucm.model.tensorize import N_CAP, K_CAP
from ucm.p2.model_p2 import P2_ENTITY_TYPES, P2_REL_PREDS, D_IN_P2

_GOAL_PREDS = ["REACH", "HAVE", "AT"]


def _vocab_index(vocab, value, what):
    try:
        return vocab.index(value)
    except ValueError:
        raise ValueError(f"unknown {what}: {value!r} (closed vocab P2)") from None


def tensorize_obs_p2(obs: dict) -> dict:
    entities = obs["entities"]
    if not (1 <= len(entities) <= N_CAP):
        raise ValueError(f"entity count {len(entities)} out of capacity")
    ent_index = {}
    nodes = np.zeros((len(entities), D_IN_P2), dtype=np.float32)
    t_block = len(P2_ENTITY_TYPES)
    for i, e in enumerate(entities):
        if e["id"] in ent_index:
            raise ValueError(f"duplicate entity id {e['id']!r}")
        ent_index[e["id"]] = i
        nodes[i, _vocab_index(P2_ENTITY_TYPES, e["type"], "entity type")] = 1.0
        if e["type"] == "door":
            nodes[i, t_block + (1 if e.get("attrs", {}).get("locked", False) else 0)] = 1.0

    edges = np.zeros((len(obs["relations"]), 2), dtype=np.int32)
    edge_types = np.zeros(len(obs["relations"]), dtype=np.int32)
    for i, r in enumerate(obs["relations"]):
        try:
            edges[i, 0] = ent_index[r["subj"]]
            edges[i, 1] = ent_index[r["obj"]]
        except KeyError as k:
            raise ValueError(f"relation {r!r} references unknown entity {k}") from None
        edge_types[i] = _vocab_index(P2_REL_PREDS, r["pred"], "relation pred")

    goal_refs = _goal_refs_p2(obs["goal"], ent_index)

    cands = obs["candidates"]
    if not (1 <= len(cands) <= K_CAP):
        raise ValueError(f"candidate count {len(cands)} out of capacity")
    cand_types = np.zeros(len(cands), dtype=np.int32)
    cand_args = np.full((len(cands), 2), -1, dtype=np.int32)
    for i, c in enumerate(cands):
        cand_types[i] = _vocab_index(ACTION_TYPES, c["action"], "action type")
        if c["arg"] is not None:
            if c["arg"] not in ent_index:
                raise ValueError(f"candidate {c!r} references unknown entity")
            cand_args[i, 0] = ent_index[c["arg"]]

    return {"nodes": nodes, "edges": edges, "edge_types": edge_types,
            "goal_pred": np.int32(_vocab_index(_GOAL_PREDS, obs["goal"]["predicate"],
                                               "goal predicate")),
            "goal_refs": goal_refs,
            "cand_types": cand_types, "cand_args": cand_args, "labels": None}


def _goal_refs_p2(goal, ent_index):
    """CONVENTION NATIVE EXACTE (bug 23:44 discriminant 2): slot 0 = OBJET
    (HAVE/AT), slot 1 = PIÈCE (REACH/AT) — la version P2 inversait les slots
    (room→0) et retournait une forme (1,2): le goal_vector liait les
    mauvaises représentations (mismatch train/... non — mismatch avec la
    convention du tronc, generalization fermée). Forme (2,) comme le natif."""
    import numpy as np
    refs = [-1, -1]
    args = goal.get("args", {})
    for key, slot in (("object", 0), ("room", 1)):
        v = args.get(key)
        if v is not None:
            if v not in ent_index:
                raise ValueError(f"goal arg {key!r} unknown entity {v!r}")
            refs[slot] = ent_index[v]
    return np.array(refs, dtype=np.int32)
