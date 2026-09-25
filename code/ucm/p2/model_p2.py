"""P2-1 (B/C) — Modèle P2: B144-class FRAIS à vocabulaire étendu + tête
de prédiction d'effet en perte auxiliaire conjointe.

- P2GNNB: GNNB(d) avec node_enc reconstruit sur D_IN_P2 (types ctx_*).
- EffectHead: pour chaque candidat, prédit le changement d'état TYPÉ de
  l'exécuter: classes {none, agent_moved, carried_key, carried_parcel,
  dropped_key, dropped_parcel, door_unlocked}.
- Perte conjointe: L = L_policy + λ·L_effect (λ=0.5 par défaut).
- Garde HEAD-NEVER-TRAINED: si head_enabled=False, les paramètres de la
  tête ne reçoivent AUCUN gradient (sha avant/après identiques — le même
  mécanisme qui a attrapé le no-op du refine).
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ucm.model.gnn_b import GNNB
from ucm.model.fixtures import ENTITY_TYPES

CTX_ENTITY_TYPES = [f"ctx_{k.lower()}" for k in ("MOVE", "PICK", "DROP", "UNLOCK")]
P2_ENTITY_TYPES = ENTITY_TYPES + CTX_ENTITY_TYPES
P2_REL_PREDS = ["adjacent", "connects", "unlocks", "at", "held", "observed_effect"]
D_IN_P2 = len(P2_ENTITY_TYPES) + 2   # + door-locked block (même convention)

EFFECT_CLASSES = ("none", "agent_moved", "carried_key", "carried_parcel",
                  "dropped_key", "dropped_parcel", "door_unlocked")
EFFECT_IX = {c: i for i, c in enumerate(EFFECT_CLASSES)}


def effect_label(kind: str, arg, valid: bool) -> int:
    """Classe typée de l'exécution de (kind, arg) — dérivée du DSL."""
    if not valid:
        return EFFECT_IX["none"]
    if kind == "MOVE":
        return EFFECT_IX["agent_moved"]
    if kind == "PICK":
        return EFFECT_IX["carried_key" if arg == "key" else "carried_parcel"]
    if kind == "DROP":
        return EFFECT_IX["dropped_key" if arg == "key" else "dropped_parcel"]
    if kind == "UNLOCK":
        return EFFECT_IX["door_unlocked"]
    return EFFECT_IX["none"]


class P2GNNB(GNNB):
    """B144-class à vocabulaire étendu (contexte). Mêmes blocs/tête que GNNB.
    CAUSE NaN 20:36 (racine réelle): GNNB fixe n_etype=len(RELATION_PREDS)
    natif (5) — l'arête observed_effect (index 5) sortait de l'embedding
    edge_type → lookup hors-bornes → NaN en backward. On reconstruit AUSSI
    les edge_type_emb sur le vocabulaire P2 (6)."""
    def __init__(self, d: int = 144, n_blocks: int = 3):
        super().__init__(d=d, n_blocks=n_blocks)
        self.n_etype = len(P2_REL_PREDS)
        self.node_enc = nnMLP([D_IN_P2 + 2, d, d])
        for blk in self.blocks:
            blk.edge_type_emb = nn.Embedding(self.n_etype, d)


def nnMLP(dims):
    from ucm.model.deepsets_a import MLP
    return MLP(dims)


class EffectHead(nn.Module):
    """(s,a) → changement typé: MLP sur [repr arg du candidat ; contexte]."""

    def __init__(self, d: int):
        super().__init__()
        self.n_classes = len(EFFECT_CLASSES)
        self.mlp = nnMLP([2 * d, d, self.n_classes])

    def __call__(self, u, g_ctx, batch):
        # u: (B, N, d); args: (B, K, 2) — on prend le 1er argument
        args = batch["cand_args"][..., :1]                    # (B, K, 1)
        u4 = mx.broadcast_to(mx.expand_dims(u, 1),
                             (u.shape[0], args.shape[1], u.shape[1], u.shape[2]))
        uargs = mx.take_along_axis(u4, mx.expand_dims(args, -1), axis=2)
        uargs = uargs * mx.expand_dims(batch["arg_present"][..., :1], -1)
        b, k, d = uargs.shape[0], uargs.shape[1], u.shape[2]
        uargs = uargs.reshape(b, k, d)
        ctx = mx.broadcast_to(mx.expand_dims(g_ctx, 1), (b, k, d))
        x = mx.concatenate([uargs, ctx], axis=-1)
        return self.mlp(x)   # (B, K, n_classes) — logits croisés


class P2Model(nn.Module):
    """Tronc GNNB-class + tête d'effet. loss = policy + λ·effect."""

    def __init__(self, d: int = 144, effect_lambda: float = 0.5,
                 head_enabled: bool = True):
        super().__init__()
        self.base = P2GNNB(d=d)
        self.head = EffectHead(d)
        self.effect_lambda = effect_lambda
        self.head_enabled = head_enabled

    def __call__(self, batch):
        u = self.base.encode(batch)
        g_ctx = self.base.global_pool(u, batch["node_mask"])
        goal_vec = self.base.goal_vector(u, batch)
        logits = self.base.score_candidates(u, g_ctx, goal_vec, batch)
        effect_logits = self.head(u, g_ctx, batch) if self.head_enabled else None
        return logits, effect_logits

    def params_sha(self, module) -> str:
        import hashlib
        import numpy as np
        parts = [k + "|" + hashlib.sha256(
            np.asarray(p.tolist(), dtype=np.float32).tobytes()).hexdigest()
                 for k, p in nn.utils.tree_flatten(module.parameters())]
        return hashlib.sha256("\n".join(parts).encode()).hexdigest()
