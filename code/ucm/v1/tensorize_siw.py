"""SIW observation → tensors (GNNB batch format), FREEZE V1a §1.

Closed-vocab one-hots per entity type block; label strings DROPPED (§12.6:
binding by entity reference, labels are arbitrary aliases — embedding them
would be a shortcut). Caps raised for SIW sizes (declared, visible errors).
"""

from __future__ import annotations

import numpy as np

from ucm.model.siw_model import (D_IN_SIW, SIW_ACTION_TYPES, SIW_DIALOG_STATES,
                                 SIW_ENTITY_TYPES, SIW_FIELD_STATES,
                                 SIW_FORM_STATUS, SIW_GOAL_PREDICATES,
                                 SIW_ONCLICK_KINDS, SIW_RELATION_PREDS,
                                 SIW_SELECT_STATES)

N_CAP = 128
K_CAP = 96
MIN_S2CAP = 4

_VI = lambda vocab, v, what: (vocab.index(v) if v in vocab else
                              (_ for _ in ()).throw(ValueError(f"unknown {what} {v!r}")))


def _node_features(e: dict) -> np.ndarray:
    f = np.zeros(D_IN_SIW, dtype=np.float32)
    t = _VI(SIW_ENTITY_TYPES, e["type"], "entity type")
    f[t] = 1.0
    a = e.get("attrs", {})
    base = len(SIW_ENTITY_TYPES)
    if e["type"] == "button":
        f[base + _VI(SIW_ONCLICK_KINDS, a.get("onclick_kind", "none"), "onclick kind")] = 1.0
    elif e["type"] == "field":
        f[base + len(SIW_ONCLICK_KINDS)
          + _VI(SIW_FIELD_STATES, "filled" if a.get("filled") else "empty", "field state")] = 1.0
    elif e["type"] == "select":
        f[base + len(SIW_ONCLICK_KINDS) + len(SIW_FIELD_STATES)
          + _VI(SIW_SELECT_STATES, "chosen" if a.get("chosen_option_ref") else "unchosen",
                "select state")] = 1.0
    elif e["type"] == "form":
        f[base + len(SIW_ONCLICK_KINDS) + len(SIW_FIELD_STATES) + len(SIW_SELECT_STATES)
          + _VI(SIW_FORM_STATUS, a.get("status", "draft"), "form status")] = 1.0
    elif e["type"] == "dialog":
        f[base + len(SIW_ONCLICK_KINDS) + len(SIW_FIELD_STATES) + len(SIW_SELECT_STATES)
          + len(SIW_FORM_STATUS)
          + _VI(SIW_DIALOG_STATES, "open" if a.get("open") else "closed", "dialog state")] = 1.0
    return f


def tensorize_siw_obs(obs: dict) -> dict:
    entities = obs["entities"]
    if not (1 <= len(entities) <= N_CAP):
        raise ValueError(f"SIW entity count {len(entities)} out of [{1},{N_CAP}]")
    idx: dict[str, int] = {}
    # materialize the virtual agent node (referenced by relations, unlisted)
    if not any(e["id"] == "agent" for e in entities):
        entities = list(entities) + [{"id": "agent", "type": "agent", "attrs": {}}]
    nodes = np.zeros((len(entities), D_IN_SIW), dtype=np.float32)
    for i, e in enumerate(entities):
        if e["id"] in idx:
            raise ValueError(f"duplicate entity {e['id']!r}")
        idx[e["id"]] = i
        nodes[i] = _node_features(e)   # label attr intentionally unused
    edges = np.zeros((len(obs["relations"]), 2), dtype=np.int32)
    etypes = np.zeros(len(obs["relations"]), dtype=np.int32)
    for i, r in enumerate(obs["relations"]):
        try:
            edges[i] = (idx[r["subj"]], idx[r["obj"]])
        except KeyError as k:
            raise ValueError(f"SIW relation {r!r} unknown entity {k}") from None
        etypes[i] = _VI(SIW_RELATION_PREDS, r["pred"], "relation pred")
    goal = obs["goal"]
    refs = [-1, -1]
    for slot, v in enumerate([a for a in goal.get("args", {}).values() if isinstance(a, str)][:2]):
        if v not in idx:
            raise ValueError(f"goal arg references unknown entity {v!r}")
        refs[slot] = idx[v]
    cands = obs["candidates"]
    if not (1 <= len(cands) <= K_CAP):
        raise ValueError(f"SIW candidate count {len(cands)} out of [{1},{K_CAP}]")
    ctypes = np.zeros(len(cands), dtype=np.int32)
    cargs = np.full((len(cands), 2), -1, dtype=np.int32)
    for i, c in enumerate(cands):
        ctypes[i] = _VI(SIW_ACTION_TYPES, c["action"], "action type")
        if c.get("arg") is not None:
            if c["arg"] not in idx:
                raise ValueError(f"candidate {c!r} unknown entity")
            cargs[i, 0] = idx[c["arg"]]
    return {"nodes": nodes, "edges": edges, "edge_types": etypes,
            "goal_pred": np.int32(_VI(SIW_GOAL_PREDICATES, goal["predicate"], "predicate")),
            "goal_refs": np.array(refs, dtype=np.int32),
            "cand_types": ctypes, "cand_args": cargs, "labels": None}


def labels_from_supervision_siw(obs: dict, optimal_actions) -> np.ndarray:
    K = len(obs["candidates"])
    lab = np.zeros(K, dtype=np.float32)
    if not optimal_actions:
        raise ValueError("empty supervision")
    if all(isinstance(a, int) for a in optimal_actions):
        for i in optimal_actions:
            if not (0 <= i < K):
                raise ValueError(f"optimal index {i} out of [0,{K})")
            lab[i] = 1.0
    else:
        opt = {(a["action"], a.get("arg")) for a in optimal_actions}
        for i, c in enumerate(obs["candidates"]):
            if (c["action"], c.get("arg")) in opt:
                lab[i] = 1.0
    if lab.sum() == 0:
        raise ValueError("optimal actions not among candidates")
    return lab


def collate_siw(examples: list[dict]) -> dict:
    """Same batch layout as ucm.model.tensorize.collate, with D_IN_SIW."""
    from ucm.model.tensorize import collate
    return collate(examples, d_in=D_IN_SIW)
