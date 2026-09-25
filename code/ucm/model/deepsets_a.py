"""Model A — Deep Sets local (spec §6.1), MLX implementation.

Normative architecture (spec §6.1, verbatim mapping):

    1. Closed-vocab one-hot entity inputs (types + observable attributes) —
       tensorize.py output, d_in=9.
    2. Goal role tags computed INSIDE the model from goal refs
       (requested object / requested room) — explicit computation on allowed
       inputs, not a progress annotation; the adapter stays goal-independent.
    3. Shared two-layer node encoder.
    4. Per-entity pooling of incident relations with edge type, sense
       (direction) and representation of the other endpoint; local update
       contextualizes each entity (agent/objects localizable via AT/HELD).
    5. Global pool of contextualized entities; goal vector built from its
       predicate and the referenced contextualized representations.
    6. Shared candidate-scorer MLP over [action type, contextualized arg
       representations, global context, goal]; STOP has its own type and null
       (zeroed) arguments.
    7. No positional embedding on arbitrary list rank.

Parameter budget: spec target 0.3–0.7M. Exact count printed by
``DeepSetsA.print_param_count()`` and asserted by tests.

All parameters float32 (FP32 first, spec §8.2). Every padded entry is masked;
padding is information-free by construction.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ucm.model.fixtures import ACTION_TYPES, ENTITY_TYPES, GOAL_PREDICATES, RELATION_PREDS
from ucm.model.tensorize import D_IN

NEG_INF = -1e9  # finite large negative for stable masked ops in fp32


class MLP(nn.Module):
    def __init__(self, dims: list[int]):
        super().__init__()
        self.layers = [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]

    def __call__(self, x: mx.array) -> mx.array:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = nn.gelu(x)
        return x


def _onehot(idx: mx.array, n: int) -> mx.array:
    return (mx.arange(n) == idx[..., None]).astype(mx.float32)


class DeepSetsA(nn.Module):
    """Deep Sets local encoder + joint candidate scorer (spec §6.1)."""

    def __init__(self, d: int = 128, local_update: bool = True):
        super().__init__()
        self.local_update = local_update  # False → §6.1 ablation: global pooling only
        self.d = d
        self.n_etype = len(RELATION_PREDS)
        self.n_pred = len(GOAL_PREDICATES)
        self.n_atype = len(ACTION_TYPES)
        self.n_role_tags = 2  # goal ref0 / goal ref1 (computed inside the model)

        # (3) shared node encoder — two layers
        self.node_enc = MLP([D_IN + self.n_role_tags, d, d])
        self.node_enc_ln = nn.LayerNorm(d)

        # (4) incident-relation message + local update
        self.edge_type_emb = nn.Embedding(self.n_etype, d)
        self.sense_emb = nn.Embedding(2, d)
        self.edge_mlp = MLP([3 * d, d, d])
        self.update_mlp = MLP([2 * d, 2 * d, d])
        self.update_ln = nn.LayerNorm(d)

        # (5) goal vector from predicate + referenced representations
        self.goal_mlp = MLP([self.n_pred + 2 * d, d, d])

        # (6) shared candidate scorer (STOP: own type, null args → zeroed)
        self.score_mlp = MLP([self.n_atype + 2 * d + 2 * d, 2 * d, 1])

    # -- pieces ------------------------------------------------------------
    def encode(self, batch: dict) -> mx.array:
        """nodes [B,N,d_in] + goal role tags → contextualized entities u [B,N,d]."""
        nodes, refs = batch["nodes"], batch["goal_refs"]  # [B,N,9], [B,2]
        b, n, _ = nodes.shape
        safe_refs = mx.maximum(refs, 0)  # clamp -1 → 0, masked below
        # role tags: [B,N,2] one-hot bindings of goal refs (explicit, allowed input)
        tags = (mx.broadcast_to(mx.expand_dims(safe_refs, 1), (b, n, 2)) ==
                mx.broadcast_to(mx.arange(n).reshape(1, n, 1), (b, n, 2))).astype(mx.float32)
        tags = tags * mx.broadcast_to(mx.expand_dims((refs >= 0).astype(mx.float32), 1), (b, n, 2))

        h = self.node_enc(mx.concatenate([nodes, tags], axis=-1))  # [B,N,d]
        h = self.node_enc_ln(h)

        # message slots: other-endpoint representation + edge type + sense
        h_other = mx.take_along_axis(h, mx.expand_dims(batch["msg_other"], -1), axis=1)  # [B,S2,d]
        m = self.edge_mlp(mx.concatenate(
            [self.edge_type_emb(batch["msg_type"]),
             self.sense_emb(batch["msg_sense"]),
             h_other], axis=-1))  # [B,S2,d]
        m = m * mx.expand_dims(batch["msg_mask"], -1)

        # per-entity aggregation through padded slot tables (sum — order invariant)
        b, s2, _ = m.shape
        # flat gather: slot j of batch i lives at i*S2 + j in the flattened axis
        flat = m.reshape(b * s2, self.d)
        flat_idx = batch["slot_idx"] + (mx.arange(b) * s2).reshape(b, 1, 1)  # [B,N,S2cap]
        agg = mx.take(flat, flat_idx, axis=0)  # [B,N,S2cap,d]
        agg = (agg * mx.expand_dims(batch["slot_mask"], -1)).sum(axis=2)  # [B,N,d]
        if not self.local_update:
            # §6.1 ablation: replace per-entity incident pooling by the GLOBAL
            # pooled context (no localized relational information)
            w = mx.expand_dims(batch["node_mask"], -1)
            gmean = (h * w).sum(axis=1, keepdims=True) / mx.maximum(
                w.sum(axis=1, keepdims=True), 1.0)
            agg = mx.broadcast_to(gmean, h.shape)

        u = self.update_mlp(mx.concatenate([h, agg], axis=-1))
        u = self.update_ln(u)
        return u

    def global_pool(self, u: mx.array, node_mask: mx.array) -> mx.array:
        w = mx.expand_dims(node_mask, -1)
        return (u * w).sum(axis=1) / mx.maximum(w.sum(axis=1), 1.0)  # [B,d] masked mean

    def goal_vector(self, u: mx.array, batch: dict) -> mx.array:
        refs, present = batch["goal_refs"], batch["goal_ref_present"]  # [B,2], [B,2]
        safe = mx.maximum(refs, 0)
        uref = mx.take_along_axis(u, mx.expand_dims(safe, -1), axis=1)  # [B,2,d]
        uref = uref * mx.expand_dims(present, -1)  # zero missing refs
        pred_1h = _onehot(batch["goal_pred"], self.n_pred)  # [B,3]
        return self.goal_mlp(mx.concatenate([pred_1h, uref.reshape(-1, 2 * self.d)], axis=-1))

    def score_candidates(self, u: mx.array, g_ctx: mx.array, goal_vec: mx.array, batch: dict) -> mx.array:
        args = batch["cand_args"]  # [B,K,2] (clamped ≥0; -1s flagged by arg_present)
        uargs = mx.take_along_axis(
            mx.broadcast_to(mx.expand_dims(u, 1), (u.shape[0], args.shape[1], u.shape[1], u.shape[2])),
            mx.expand_dims(args, -1), axis=2)  # [B,K,2,d]
        uargs = uargs * mx.expand_dims(batch["arg_present"], -1)  # null args → zeros
        t1h = _onehot(batch["cand_types"], self.n_atype)  # [B,K,5]
        ctx = mx.broadcast_to(mx.expand_dims(mx.concatenate([g_ctx, goal_vec], axis=-1), 1),
                              (u.shape[0], args.shape[1], 2 * self.d))
        x = mx.concatenate([t1h, uargs.reshape(u.shape[0], args.shape[1], 2 * self.d), ctx], axis=-1)
        logits = self.score_mlp(x).squeeze(-1)  # [B,K]
        return logits + (1.0 - batch["cand_mask"]) * NEG_INF  # padding masked (visible, huge negative)

    # -- full forward --------------------------------------------------------
    def __call__(self, batch: dict) -> mx.array:
        u = self.encode(batch)
        g_ctx = self.global_pool(u, batch["node_mask"])
        goal_vec = self.goal_vector(u, batch)
        return self.score_candidates(u, g_ctx, goal_vec, batch)

    def param_count(self) -> int:
        return sum(int(p.size) for _, p in nn.utils.tree_flatten(self.parameters()))

    def print_param_count(self) -> int:
        n = self.param_count()
        print(f"DeepSetsA parameters: {n:,} ({n / 1e6:.3f}M) — spec target 0.3–0.7M")
        return n


def predict_single(model: DeepSetsA, obs: dict) -> mx.array:
    """Convenience: one policy_input observation → logits over its candidates."""
    from ucm.model.tensorize import collate, tensorize_obs
    ex = tensorize_obs(obs)
    ex["labels"] = None
    batch = collate([ex])
    return model(batch)[0]
