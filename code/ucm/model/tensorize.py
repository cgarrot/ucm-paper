"""Canonical policy_input → tensors for Model A (PLAN §3.4; spec §5.3).

Consumes the WS-A canonical observation format (M0 landed):

    goal:       {"predicate", "args": {...}}    REACH{room} HAVE{object} AT{object,room}
    entities:   [{"id", "type": room|agent|key|parcel|door, "attrs"}]  door: {"locked": bool}
    relations:  [{"subj", "pred": adjacent|connects|unlocks|at|held, "obj"}]
    candidates: [{"action": MOVE|PICK|DROP|UNLOCK|STOP, "arg": id|None}]   K = R+6
    supervision.optimal_actions: list[int] (indices into candidates — canonical order)

Tensor contract (PLAN §3.4):
    nodes      [N, d_in] float32 — closed-vocab one-hots (types + observable
               attributes); goal role tags are added INSIDE the model
    edges      [E, 2] int32 — localized indices (never raw ids) + edge_types [E]
    goal       predicate_id + ref_indices [2] (object ref then room ref; -1 unused)
    candidates type_ids [K] + arg_node_indices [K,2] (-1 absent) + pad_mask [K]
    labels     0/1 [K] — loss channel only
    output     logits [K]

Rules (§5.3): ids bind, never embed; this module sees policy_input ONLY;
coherent permutations permute tensors; capacity overflow is a visible error.
"""

from __future__ import annotations

import numpy as np

from ucm.model.fixtures import ACTION_TYPES, ENTITY_TYPES, RELATION_PREDS

D_IN = len(ENTITY_TYPES) + 2          # + door-locked {no, yes} block = 7
N_CAP = 64                            # spec §5.3 harness caps (visible errors)
K_CAP = 32
MIN_S2CAP = 4


def _vocab_index(vocab: list[str], value: str, what: str) -> int:
    try:
        return vocab.index(value)
    except ValueError:
        raise ValueError(f"unknown {what} '{value}' (closed vocabulary: {vocab})") from None


def _goal_refs(goal: dict, ent_index: dict) -> np.ndarray:
    """Reference node indices [2]: slot 0 = requested object (HAVE/AT),
    slot 1 = requested room (REACH/AT). -1 when unused."""
    pred, args = goal["predicate"], goal.get("args", {})
    refs = [-1, -1]
    for key, slot in (("object", 0), ("room", 1)):
        if key in args:
            rid = args[key]
            if rid not in ent_index:
                raise ValueError(f"goal arg '{key}' references unknown entity {rid!r}")
            refs[slot] = ent_index[rid]
    if refs == [-1, -1]:
        raise ValueError(f"goal {goal!r} carries no reference")
    return np.array(refs, dtype=np.int32)


def tensorize_obs(obs: dict) -> dict:
    """Single canonical policy_input → per-episode numpy tensors (deterministic)."""
    entities = obs["entities"]
    if not (1 <= len(entities) <= N_CAP):
        raise ValueError(f"entity count {len(entities)} out of declared capacity [1,{N_CAP}]")
    ent_index: dict[str, int] = {}
    nodes = np.zeros((len(entities), D_IN), dtype=np.float32)
    t_block = len(ENTITY_TYPES)
    lock_block = t_block
    for i, e in enumerate(entities):
        if e["id"] in ent_index:
            raise ValueError(f"duplicate entity id {e['id']!r}")
        ent_index[e["id"]] = i
        nodes[i, _vocab_index(ENTITY_TYPES, e["type"], "entity type")] = 1.0
        if e["type"] == "door":
            locked = e.get("attrs", {}).get("locked", False)
            nodes[i, lock_block + (1 if locked else 0)] = 1.0

    edges = np.zeros((len(obs["relations"]), 2), dtype=np.int32)
    edge_types = np.zeros(len(obs["relations"]), dtype=np.int32)
    for i, r in enumerate(obs["relations"]):
        try:
            edges[i, 0] = ent_index[r["subj"]]
            edges[i, 1] = ent_index[r["obj"]]
        except KeyError as k:
            raise ValueError(f"relation {r!r} references unknown entity {k}") from None
        edge_types[i] = _vocab_index(RELATION_PREDS, r["pred"], "relation pred")

    goal_refs = _goal_refs(obs["goal"], ent_index)

    cands = obs["candidates"]
    if not (1 <= len(cands) <= K_CAP):
        raise ValueError(f"candidate count {len(cands)} out of declared capacity [1,{K_CAP}]")
    cand_types = np.zeros(len(cands), dtype=np.int32)
    cand_args = np.full((len(cands), 2), -1, dtype=np.int32)
    for i, c in enumerate(cands):
        ctype = _vocab_index(ACTION_TYPES, c["action"], "action type")
        cand_types[i] = ctype
        if c["arg"] is not None:
            if c["arg"] not in ent_index:
                raise ValueError(f"candidate {c!r} references unknown entity")
            if c["action"] == "STOP":
                raise ValueError("STOP candidate must have a null argument")
            cand_args[i, 0] = ent_index[c["arg"]]

    return {"nodes": nodes, "edges": edges, "edge_types": edge_types,
            "goal_pred": np.int32(_vocab_index(["REACH", "HAVE", "AT"], obs["goal"]["predicate"],
                                               "goal predicate")),
            "goal_refs": goal_refs,
            "cand_types": cand_types, "cand_args": cand_args, "labels": None}


def labels_from_supervision(obs: dict, optimal_actions) -> np.ndarray:
    """0/1 labels aligned with obs['candidates'] order. Accepts the canonical
    index list (supervision.optimal_actions) or explicit {"action","arg"} dicts
    (matched by reference). Empty/unbound sets are visible errors (§4.6: never
    supervise unreachable states)."""
    K = len(obs["candidates"])
    lab = np.zeros(K, dtype=np.float32)
    if not optimal_actions:
        raise ValueError("optimal action set is empty (unreachable?) — cannot supervise")
    if all(isinstance(a, int) for a in optimal_actions):
        for i in optimal_actions:
            if not (0 <= i < K):
                raise ValueError(f"optimal action index {i} out of range [0,{K})")
            lab[i] = 1.0
    else:
        opt = {(a["action"], a["arg"]) if isinstance(a, dict) else tuple(a)
               for a in optimal_actions}
        for i, c in enumerate(obs["candidates"]):
            if (c["action"], c["arg"]) in opt:
                lab[i] = 1.0
    if lab.sum() == 0:
        raise ValueError("optimal actions not found among candidates — binding error")
    return lab


def randomize_obs_order(obs: dict, rng) -> dict:
    """Return a copy of obs with entity order, relation order and candidate
    order shuffled (ids and content unchanged). PLAN §WS-C (audit tagi-5 T8):
    order randomization must be exercised in train AND test so that list rank
    carries no learnable signal; binding follows the ids, tensorization follows
    the order."""
    import random as _random
    if not isinstance(rng, _random.Random):
        raise TypeError("rng must be random.Random (seeded)")
    entities = list(obs["entities"])
    rng.shuffle(entities)
    relations = list(obs["relations"])
    rng.shuffle(relations)
    candidates = list(obs["candidates"])
    rng.shuffle(candidates)
    return {"goal": obs["goal"], "entities": entities,
            "relations": relations, "candidates": candidates}


def permute_example(ex: dict, rng) -> dict:
    """Randomize order of an ALREADY-TENSORIZED example (numpy-level, no dict
    round-trip): node permutation π + relation (edge) order shuffle + candidate
    order shuffle with labels permuted together. Equivalent to tensorizing
    randomize_obs_order(obs) — see test_g5 equivalence. Keeps per-epoch
    tensorization-free augmentation fast."""
    n = ex["nodes"].shape[0]
    k = len(ex["cand_types"])
    e = len(ex["edges"])
    pi = np.arange(n)
    rng.shuffle(pi)          # old position → new position
    inv = np.empty(n, dtype=np.int64)
    inv[pi] = np.arange(n)   # new position → old position

    out = dict(ex)
    out["nodes"] = ex["nodes"][pi]            # old row pi[i] → position i
    order = np.arange(e)
    rng.shuffle(order)                       # relation order shuffle
    out["edges"] = inv[ex["edges"][order]].astype(np.int32)  # values remapped old→new
    out["edge_types"] = ex["edge_types"][order]

    out["goal_refs"] = np.where(ex["goal_refs"] >= 0,
                                inv[ex["goal_refs"]], -1).astype(np.int32)
    cand_order = np.arange(k)
    rng.shuffle(cand_order)                  # candidate order shuffle
    out["cand_types"] = ex["cand_types"][cand_order]
    out["cand_args"] = np.where(ex["cand_args"][cand_order] >= 0,
                                inv[np.where(ex["cand_args"][cand_order] >= 0,
                                             ex["cand_args"][cand_order], 0)],
                                -1).astype(np.int32)
    if ex.get("labels") is not None:
        out["labels"] = ex["labels"][cand_order]
    return out


_SLOT_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}


def _slot_tables(edges: np.ndarray, n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-node padded gather table over the 2E message slots + mask.
    Slot 2e: receiver = subj(e), sense 0. Slot 2e+1: receiver = obj(e), sense 1.
    Content-keyed cache (lead 21:26 microbench: ~2.2× on collation batch64) —
    pure function of (edges, n_nodes), safe across examples and steps."""
    key = (n_nodes, edges.tobytes())
    hit = _SLOT_CACHE.get(key)
    if hit is not None:
        return hit
    n_slots = 2 * len(edges)
    recv = np.empty(n_slots, dtype=np.int32)
    for e in range(len(edges)):
        recv[2 * e] = edges[e, 0]
        recv[2 * e + 1] = edges[e, 1]
    counts = np.bincount(recv, minlength=n_nodes) if n_slots else np.zeros(n_nodes, dtype=int)
    s2cap = max(int(counts.max()) if counts.size and counts.max() else 0, MIN_S2CAP)
    idx = np.zeros((n_nodes, s2cap), dtype=np.int32)
    mask = np.zeros((n_nodes, s2cap), dtype=np.float32)
    cursor = np.zeros(n_nodes, dtype=np.int64)
    for s in range(n_slots):
        r = int(recv[s])
        j = int(cursor[r])
        idx[r, j] = s
        mask[r, j] = 1.0
        cursor[r] += 1
    _SLOT_CACHE[key] = (idx, mask)
    return idx, mask


def collate(examples: list[dict], d_in: int | None = None) -> dict:
    """Batch of tensorized examples (+ labels or None) → padded MLX arrays.
    Dynamic padding to per-batch maxima; every padded entry is masked.
    d_in defaults to the V0 closed vocab (D_IN); SIW passes D_IN_SIW."""
    import mlx.core as mx  # imported late: tensorize stays numpy-pure/testable

    b = len(examples)
    d_in = d_in or D_IN
    n = max(ex["nodes"].shape[0] for ex in examples)
    e_slots = max(2 * len(ex["edges"]) for ex in examples)
    k = max(ex["cand_types"].shape[0] for ex in examples)
    # slot tables computed ONCE per example (lead 21:24: was twice — the max()
    # scan recomputed every table before the assignment loop rebuilt them)
    tables = [_slot_tables(ex["edges"], ex["nodes"].shape[0]) for ex in examples]
    s2cap = max(t0[0].shape[1] for t0 in tables)

    nodes = np.zeros((b, n, d_in), dtype=np.float32)
    node_mask = np.zeros((b, n), dtype=np.float32)
    msg_other = np.zeros((b, e_slots), dtype=np.int32)
    msg_type = np.zeros((b, e_slots), dtype=np.int32)
    msg_sense = np.zeros((b, e_slots), dtype=np.int32)
    msg_mask = np.zeros((b, e_slots), dtype=np.float32)
    slot_idx = np.zeros((b, n, s2cap), dtype=np.int32)
    slot_mask = np.zeros((b, n, s2cap), dtype=np.float32)
    goal_pred = np.zeros(b, dtype=np.int32)
    goal_refs = np.full((b, 2), -1, dtype=np.int32)
    goal_ref_present = np.zeros((b, 2), dtype=np.float32)
    cand_types = np.zeros((b, k), dtype=np.int32)
    cand_args = np.zeros((b, k, 2), dtype=np.int32)
    arg_present = np.zeros((b, k, 2), dtype=np.float32)
    cand_mask = np.zeros((b, k), dtype=np.float32)
    labels = np.zeros((b, k), dtype=np.float32)
    has_labels = all(ex.get("labels") is not None for ex in examples)

    for i, ex in enumerate(examples):
        ni = ex["nodes"].shape[0]
        nodes[i, :ni] = ex["nodes"]
        node_mask[i, :ni] = 1.0
        edges, et = ex["edges"], ex["edge_types"]
        for j in range(len(edges)):
            msg_other[i, 2 * j] = edges[j, 1]
            msg_type[i, 2 * j] = et[j]
            msg_sense[i, 2 * j] = 0
            msg_other[i, 2 * j + 1] = edges[j, 0]
            msg_type[i, 2 * j + 1] = et[j]
            msg_sense[i, 2 * j + 1] = 1
        msg_mask[i, :2 * len(edges)] = 1.0
        sidx, smask = tables[i]
        slot_idx[i, :ni, :sidx.shape[1]] = sidx
        slot_mask[i, :ni, :smask.shape[1]] = smask
        goal_pred[i] = ex["goal_pred"]
        goal_refs[i] = ex["goal_refs"]
        goal_ref_present[i] = (ex["goal_refs"] >= 0).astype(np.float32)
        ki = len(ex["cand_types"])
        cand_types[i, :ki] = ex["cand_types"]
        arg_present[i, :ki] = (ex["cand_args"] >= 0).astype(np.float32)
        cand_args[i, :ki] = np.where(ex["cand_args"] >= 0, ex["cand_args"], 0)
        cand_mask[i, :ki] = 1.0
        if ex.get("labels") is not None:
            labels[i, :ki] = ex["labels"]

    return {
        "nodes": mx.array(nodes), "node_mask": mx.array(node_mask),
        "msg_other": mx.array(msg_other), "msg_type": mx.array(msg_type),
        "msg_sense": mx.array(msg_sense), "msg_mask": mx.array(msg_mask),
        "slot_idx": mx.array(slot_idx), "slot_mask": mx.array(slot_mask),
        "goal_pred": mx.array(goal_pred), "goal_refs": mx.array(goal_refs),
        "goal_ref_present": mx.array(goal_ref_present),
        "cand_types": mx.array(cand_types), "cand_args": mx.array(cand_args),
        "arg_present": mx.array(arg_present), "cand_mask": mx.array(cand_mask),
        "labels": mx.array(labels) if has_labels else None,
    }


def tensorize_batch(records: list[dict]) -> dict:
    """Canonical records (policy_input + supervision) → collated batch."""
    exs = []
    for rec in records:
        ex = tensorize_obs(rec["policy_input"])
        ex["labels"] = labels_from_supervision(rec["policy_input"],
                                               rec["supervision"]["optimal_actions"])
        exs.append(ex)
    return collate(exs)
