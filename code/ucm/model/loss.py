"""Set-valued behavioural cloning loss (spec §8.1):

    L = -log Σ_{a ∈ A*} p(a)        with p = softmax over the candidate set

Numerically stable via logsumexp. ONLY padding is masked in the distribution
(§8.1): the A* mask enters the loss numerator, never the runtime candidate
enumeration. Invalid physical actions remain fully in the softmax denominator.

STOP is trained as a complete action (spec §8.1). No progress/value/reconstruction
terms in the main arm.
"""

from __future__ import annotations

import mlx.core as mx

NEG_INF = -1e9  # finite large negative (fp32-stable) for masked reductions


def set_bc_loss(logits: mx.array, labels: mx.array, pad_mask: mx.array) -> mx.array:
    """Mean over batch of -log Σ_{a∈A*} p(a).

    logits   [B, K] float32 — raw scores, padding already huge-negative in model
                          output but re-masked here defensively (same constant)
    labels   [B, K] float32 0/1 — 1 iff candidate ∈ A* (loss-only channel)
    pad_mask [B, K] float32 1/0 — 1 = real candidate, 0 = padding

    Rows with an empty A* are a data error upstream (unreachable states must
    never reach BC); a large-but-finite loss is returned and the caller asserts.
    """
    masked = logits + (1.0 - pad_mask) * NEG_INF
    num = mx.logsumexp(masked + (1.0 - labels) * NEG_INF, axis=-1)  # log Σ p-numerator terms
    den = mx.logsumexp(masked, axis=-1)                             # log Σ p over real candidates
    return mx.mean(den - num)


def optimal_action_rate(logits: mx.array, labels: mx.array, pad_mask: mx.array) -> mx.array:
    """Fraction of examples where argmax over real candidates ∈ A* (diagnostic,
    stratification by |A*| happens in eval reporting, not here)."""
    masked = logits + (1.0 - pad_mask) * NEG_INF
    pick = mx.argmax(masked, axis=-1)
    hit = mx.take_along_axis(labels, mx.expand_dims(pick, -1), axis=-1).squeeze(-1)
    return mx.mean(hit)


def prob_mass_on_optimal(logits: mx.array, labels: mx.array, pad_mask: mx.array) -> mx.array:
    """Mean probability mass Σ_{a∈A*} p(a) (diagnostic only, spec §9.2)."""
    masked = logits + (1.0 - pad_mask) * NEG_INF
    logp = masked - mx.logsumexp(masked, axis=-1, keepdims=True)
    p = mx.exp(logp) * pad_mask          # real probabilities under padding-only mask
    return mx.mean(mx.sum(p * labels, axis=-1))
