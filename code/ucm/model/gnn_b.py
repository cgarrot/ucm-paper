"""Model B — GNN relationnel (spec §6.2), motivated by the audited GATE-1
failure of A (depth: exact 1-hop ties + d*≥3 misrankings; WS-C audit M1).

Normative architecture (spec §6.2, verbatim mapping):
    - d = 192, target 1.1–1.5M parameters (exact count printed)
    - goal role tags added BEFORE message passing (tags and goal binding also
      removed in the no-goal ablation)
    - THREE DISTINCT relational message-passing blocks, each with its own
      message MLP (3d→d→d) over [edge type, sense, other endpoint] and local
      update MLP (2d→d→d), with residual connections and layer norms
    - pooling and scorer comparable to A (§6.1): masked global pool, goal
      vector from predicate + referenced contextualized representations,
      shared candidate scorer over [action type, argument representations,
      global context, goal]; STOP own type + null args
    - no positional embedding on arbitrary list rank

GATE-5 criteria (pre-specified by lead before any run):
    (1) failure stratum (exact ties + d*≥3 of the M1 diagnostic): B ≥ A+5pts,
        paired bootstrap IC95 > 0
    (2) d*≤2 stratum: no loss > 2 points
    (3) measured cost ≤ 2× A
    (4) dedicated tie-separability test: same-1-hop-signature pairs must yield
        distinct representations (A provably cannot — its u-rows are
        bit-identical there); if B cannot either, that is an observation-
        scheme limit (major discovery), documented.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ucm.model.deepsets_a import MLP, NEG_INF, _onehot
from ucm.model.fixtures import ACTION_TYPES, GOAL_PREDICATES, RELATION_PREDS
from ucm.model.tensorize import D_IN


class MessageBlock(nn.Module):
    """One relational message-passing round (distinct weights per block)."""

    def __init__(self, d: int, n_etype: int):
        super().__init__()
        self.d = d
        self.edge_type_emb = nn.Embedding(n_etype, d)
        self.sense_emb = nn.Embedding(2, d)
        self.edge_mlp = MLP([3 * d, d, d])     # message: type + sense + other
        self.update_mlp = MLP([2 * d, d, d])   # local update: h_i + aggregated

    def aggregate(self, h: mx.array, batch: dict) -> mx.array:
        h_other = mx.take_along_axis(h, mx.expand_dims(batch["msg_other"], -1), axis=1)
        m = self.edge_mlp(mx.concatenate(
            [self.edge_type_emb(batch["msg_type"]),
             self.sense_emb(batch["msg_sense"]),
             h_other], axis=-1))
        m = m * mx.expand_dims(batch["msg_mask"], -1)
        b, s2, _ = m.shape
        flat = m.reshape(b * s2, self.d)
        flat_idx = batch["slot_idx"] + (mx.arange(b) * s2).reshape(b, 1, 1)
        agg = mx.take(flat, flat_idx, axis=0)
        return (agg * mx.expand_dims(batch["slot_mask"], -1)).sum(axis=2)

    def __call__(self, h: mx.array, batch: dict) -> mx.array:
        agg = self.aggregate(h, batch)
        return self.update_mlp(mx.concatenate([h, agg], axis=-1))


class GNNB(nn.Module):
    """GNN B: 3 distinct message blocks with residual + norm (§6.2)."""

    def __init__(self, d: int = 192, n_blocks: int = 3):
        super().__init__()
        self.d = d
        self.n_etype = len(RELATION_PREDS)
        self.n_pred = len(GOAL_PREDICATES)
        self.n_atype = len(ACTION_TYPES)

        # shared node encoder + goal role tags BEFORE message passing (§6.2)
        self.node_enc = MLP([D_IN + 2, d, d])
        self.node_enc_ln = nn.LayerNorm(d)
        self.blocks = [MessageBlock(d, self.n_etype) for _ in range(n_blocks)]
        self.block_lns = [nn.LayerNorm(d) for _ in range(n_blocks)]

        self.goal_mlp = MLP([self.n_pred + 2 * d, d, d])
        self.score_mlp = MLP([self.n_atype + 2 * d + 2 * d, 2 * d, 1])

    def encode(self, batch: dict) -> mx.array:
        nodes, refs = batch["nodes"], batch["goal_refs"]
        b, n, _ = nodes.shape
        safe_refs = mx.maximum(refs, 0)
        tags = (mx.broadcast_to(mx.expand_dims(safe_refs, 1), (b, n, 2)) ==
                mx.broadcast_to(mx.arange(n).reshape(1, n, 1), (b, n, 2))).astype(mx.float32)
        tags = tags * mx.broadcast_to(mx.expand_dims((refs >= 0).astype(mx.float32), 1), (b, n, 2))

        h = self.node_enc_ln(self.node_enc(mx.concatenate([nodes, tags], axis=-1)))
        for blk, ln in zip(self.blocks, self.block_lns):
            h = ln(h + blk(h, batch))    # residual + norm (§6.2)
        return h

    def global_pool(self, u: mx.array, node_mask: mx.array) -> mx.array:
        w = mx.expand_dims(node_mask, -1)
        return (u * w).sum(axis=1) / mx.maximum(w.sum(axis=1), 1.0)

    def goal_vector(self, u: mx.array, batch: dict) -> mx.array:
        refs, present = batch["goal_refs"], batch["goal_ref_present"]
        safe = mx.maximum(refs, 0)
        uref = mx.take_along_axis(u, mx.expand_dims(safe, -1), axis=1)
        uref = uref * mx.expand_dims(present, -1)
        pred_1h = _onehot(batch["goal_pred"], self.n_pred)
        return self.goal_mlp(mx.concatenate([pred_1h, uref.reshape(-1, 2 * self.d)], axis=-1))

    def score_candidates(self, u: mx.array, g_ctx: mx.array, goal_vec: mx.array,
                         batch: dict) -> mx.array:
        args = batch["cand_args"]
        uargs = mx.take_along_axis(
            mx.broadcast_to(mx.expand_dims(u, 1),
                            (u.shape[0], args.shape[1], u.shape[1], u.shape[2])),
            mx.expand_dims(args, -1), axis=2)
        uargs = uargs * mx.expand_dims(batch["arg_present"], -1)
        t1h = _onehot(batch["cand_types"], self.n_atype)
        ctx = mx.broadcast_to(mx.expand_dims(mx.concatenate([g_ctx, goal_vec], axis=-1), 1),
                              (u.shape[0], args.shape[1], 2 * self.d))
        x = mx.concatenate([t1h, uargs.reshape(u.shape[0], args.shape[1], 2 * self.d), ctx], axis=-1)
        logits = self.score_mlp(x).squeeze(-1)
        return logits + (1.0 - batch["cand_mask"]) * NEG_INF

    def __call__(self, batch: dict) -> mx.array:
        u = self.encode(batch)
        g_ctx = self.global_pool(u, batch["node_mask"])
        goal_vec = self.goal_vector(u, batch)
        return self.score_candidates(u, g_ctx, goal_vec, batch)

    def param_count(self) -> int:
        return sum(int(p.size) for _, p in nn.utils.tree_flatten(self.parameters()))

    def print_param_count(self) -> int:
        n = self.param_count()
        print(f"GNNB parameters: {n:,} ({n / 1e6:.3f}M) — spec target 1.1–1.5M")
        return n


def predict_single_b(model: GNNB, obs: dict) -> mx.array:
    from ucm.model.tensorize import collate, tensorize_obs
    ex = tensorize_obs(obs)
    ex["labels"] = None
    return model(collate([ex]))[0]
