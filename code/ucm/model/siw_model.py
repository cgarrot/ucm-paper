"""SIW-adapted Model B (FREEZE V1a §1): frozen B144 core, fresh SIW vocab.

Transferable (same names & shapes as canon GNNB d=144):
    blocks.{0,1,2}.{edge_mlp, update_mlp, sense_emb}, score_mlp
Fresh by necessity (shape differs with the SIW closed vocab):
    node_enc (D_in_SIW ≠ D_in_TGK), goal_mlp (4 SIW predicates ≠ 3 TGK),
    blocks.*.edge_type_emb (9 SIW relation preds ≠ 5 TGK)
The loaded/skipped manifest is published by ucm.v1.transfer.build_arm.

Label strings in SIW observations are NEVER embedded (§12.6: binding by
entity reference; labels are arbitrary aliases) — tensorize_siw drops them.
"""

from __future__ import annotations

import mlx.nn as nn

from ucm.model.gnn_b import GNNB

SIW_ENTITY_TYPES = ["view", "button", "field", "select", "option", "form", "dialog",
                       "agent"]  # agent = noeud virtuel materialisé par le consommateur
# (les relations current_view/filled/chosen référencent "agent" sans entité listée)
# real closed vocab from the delivered env (ucm/env/siw.py:98-100, layouts scellés)
SIW_ONCLICK_KINDS = ["none", "submit", "confirm", "dismiss"]
SIW_FIELD_STATES = ["empty", "filled"]
SIW_SELECT_STATES = ["unchosen", "chosen"]
SIW_FORM_STATUS = ["draft", "complete", "submitted"]
SIW_DIALOG_STATES = ["closed", "open"]
SIW_RELATION_PREDS = ["nav_edge", "on_view", "part_of", "option_of", "submits",
                      "in_dialog", "current_view", "filled", "chosen"]
SIW_GOAL_PREDICATES = ["VIEW", "SET", "CHOOSE", "SUBMITTED"]
SIW_ACTION_TYPES = ["NAVIGATE", "CLICK", "TYPE", "SELECT", "STOP"]

# closed attribute one-hot blocks per entity type (sum = D_IN_SIW - len(types))
D_IN_SIW = (len(SIW_ENTITY_TYPES) + len(SIW_ONCLICK_KINDS) + len(SIW_FIELD_STATES)
            + len(SIW_SELECT_STATES) + len(SIW_FORM_STATUS) + len(SIW_DIALOG_STATES))


def make_siw_model(d: int = 144) -> GNNB:
    """Fresh SIW-adapted model. Same class as canon (GNNB) so that the
    transferable parameter NAMES match exactly; the vocab-dependent shapes
    are set by construction from the SIW constants above."""
    m = GNNB(d=d)
    # re-shape ONLY the vocab-dependent submodules (fresh weights); score_mlp
    # keeps the canon shape (5 SIW action types = 5 TGK) → TRANSFERS.
    import mlx.core as mx
    m.n_etype = len(SIW_RELATION_PREDS)
    m.n_pred = len(SIW_GOAL_PREDICATES)
    m.n_atype = len(SIW_ACTION_TYPES)
    for blk in m.blocks:
        blk.edge_type_emb = nn.Embedding(m.n_etype, d)
    m.goal_mlp = MLP_SIW((m.n_pred + 2 * d, d, d))
    m.node_enc = MLP_SIW((D_IN_SIW + 2, d, d))
    mx.eval(m.parameters())
    return m


def MLP_SIW(dims):
    from ucm.model.deepsets_a import MLP
    return MLP(list(dims))
