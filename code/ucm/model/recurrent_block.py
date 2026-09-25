"""CIV minimal — bloc récurrent au-dessus de B144 (chapitre S2b→P2, lead 12:35).

Conception (choix retenu: GRU):
    h = B144.encode(batch)                     # poids B144 inchangés/transférables
    pour t in 1..T:  h ← LN(GRU(h, MP_partagé(h)))   # UN bloc partagé, T paramétrable
    logits = B144.tête(pool/goal/score)(h)

- Constant en paramètres quand T croît (poids partagés) — le champ récepteur
  grandit avec T, pas le budget de paramètres (hypothèse 3 sauts à tester).
- Entraînable en DEV (même boucle que B144 — le modèle expose __call__(batch)).
- Architecturé pour la factorielle: make_siw_rec_model(T) — le bras 'B144' de
  la factorielle est make_siw_model() SANS le bloc (T=0 ≡ B144 pur).
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ucm.model.gnn_b import MessageBlock


class GRUCell(nn.Module):
    """Cellule GRU standard sur d (portes z, r + candidate)."""

    def __init__(self, d: int):
        super().__init__()
        self.d = d
        self.wz = nn.Linear(2 * d, d)
        self.wr = nn.Linear(2 * d, d)
        self.wh = nn.Linear(2 * d, d)

    def __call__(self, h: mx.array, m: mx.array) -> mx.array:
        x = mx.concatenate([h, m], axis=-1)
        z = mx.sigmoid(self.wz(x))
        r = mx.sigmoid(self.wr(x))
        rh = mx.concatenate([r * h, m], axis=-1)
        cand = mx.tanh(self.wh(rh))
        return (1.0 - z) * h + z * cand


class RecurrentRefine(nn.Module):
    """Raffinement récurrent PARTAGÉ au-dessus des états de B144.

    T itérations d'un seul MessageBlock (poids partagés) + GRUCell + LayerNorm.
    T paramétrable — défaut 32. param_count CONSTANT en T."""

    def __init__(self, d: int, n_etype: int, T: int = 32):
        super().__init__()
        self.d, self.T = d, T
        self.mp = MessageBlock(d, n_etype)      # UN bloc, partagé
        self.gru = GRUCell(d)
        self.ln = nn.LayerNorm(d)

    # Instrumentation NORMATIVE (tagi-5 12:36): compteur d'itérations
    # RÉELLEMENT exécutées — lu par l'artefact, pas cru sur parole.
    n_iters: int = 0

    def __call__(self, h: mx.array, batch: dict) -> mx.array:
        self.n_iters = 0
        for _ in range(self.T):        # BOUCLE SÉQUENTIELLE (pas de
            m = self.mp(h, batch)      # vectorisation T — le p95 se
            h = self.ln(self.gru(h, m))  # mesure sur cette boucle)
            self.n_iters += 1
        return h


class SIWRecModel(nn.Module):
    """B144 + RecurrentRefine. Le classifieur/tête de B144 est réutilisé tel
    quel sur les états raffinés — seuls les poids du bloc récurrent sont
    nouveaux (le canon reste chargeable dans .base)."""

    def __init__(self, base, T: int = 32):
        super().__init__()
        self.base = base                          # GNNB (B144, transférable)
        self.refine = RecurrentRefine(base.d, base.n_etype, T=T)

    def __call__(self, batch: dict) -> mx.array:
        u = self.base.encode(batch)
        u = self.refine(u, batch)
        g_ctx = self.base.global_pool(u, batch["node_mask"])
        goal_vec = self.base.goal_vector(u, batch)
        return self.base.score_candidates(u, g_ctx, goal_vec, batch)

    # GEL VÉRIFIABLE (tagi-5 12:36): sha des paramètres BASE — avant/après
    # entraînement doivent être IDENTIQUES (contre-factuel: unfreeze ⇒ drift).
    def base_params_sha256(self) -> str:
        import hashlib
        import numpy as _np
        parts = []
        for k, p in nn.utils.tree_flatten(self.base.parameters()):
            a = _np.asarray(p.tolist(), dtype=_np.float32)
            parts.append(k + "|" + hashlib.sha256(a.tobytes()).hexdigest())
        blob = "\n".join(parts).encode()
        return hashlib.sha256(blob).hexdigest()

    def param_count(self) -> int:
        return sum(int(p.size) for _, p in nn.utils.tree_flatten(self.parameters()))


def make_siw_rec_model(d: int = 144, T: int = 32) -> SIWRecModel:
    """Bras 'récurrent' de la factorielle — T=0 ≡ B144 pur (bras 'B144')."""
    from ucm.model.siw_model import make_siw_model
    if T <= 0:
        raise ValueError("T<=0: use make_siw_model() directly (B144 arm)")
    m = SIWRecModel(make_siw_model(d), T=T)
    mx.eval(m.parameters())
    return m
