# 08 — Roadmap and Open Locks

*Where the project stands at snapshot `635ba3c` (2026-09-25, evening) and what would falsify the next steps.*

## 1. State summary

| Piece | State |
|---|---|
| Vision stack (voice → comprehension → executor → measured result) | defined; executor built |
| V0 TGK generalization / composition | **established** (97.5 % / 99.4 %) |
| V0 cost / latency | established (~1 ms, 1.921× vs 1-hop gate) |
| Verified DSL layer | established (differential ≥10k states, 3 mutations detected) |
| V1-bis transfer TGK→SIW | **not demonstrated** (saturated regime; bound ≤ +0.73 pp); joint cell table published |
| S2b recurrence | **KILL** (−7.7 pp vs +15.8); retention passes |
| P2 context | **not testable as designed**; presence-distraction measured |
| Auxiliary-target line (v01 + v02) | **not established**: +34 pp not replicated; v02 non-confirmatory; validity target outside the canon |
| S5 product line | internal criteria met in synthetic demo (44/44, 32/32; stub) |
| Web compiler v2 | **established on the probe**: 100 % (504/504) live, v1 parity 5.86 % verified, 21 tests |
| Browser end-to-end | **mechanics proven**; calibrated refusal on compiled pages (executor not trained on them) |
| Strict inter-signature composition | **0 %** on one exploratory point; intra-band GO |

## 2. The three locks (priority order)

### Lock 1 — Strict composition

**Question.** Can the controller act correctly on task signatures whose optimal plans were never in training, when those signatures are known-optimal and demonstrated?

**Protocol sketch.** Pre-register families by optimal-plan signature; verify known-optimality with the oracle; train only within a band; evaluate held-back signatures with raw-first publication; keep H1/H2 probes as diagnostics (depth vs signature; goal→type shortcut) with structural-sampling checks **before** freezing (the N=0 stratum A lesson).

**Falsifier.** If held-back-signature success remains at 0 % with adequate power across several signatures while intra-band stays ~97 %, composition is a genuine wall for one-pass, fixed-depth policies and the answer is search or structure, not scale.

**Existing signal.** Zero-shot closed loop on a held-back family is 58.3 % without the shortcut, so there is something to build on.

### Lock 2 — Depth via short search (not size)

**Question.** Does a correct short-search arm (fixed beam or verified lookahead over the same policy) break the ~12–15 depth wall?

**Protocol sketch.** Fix the broken beam; compare reactive vs short search at equal wall-clock budget buckets; report success by d* band, regret, and any-time trade-off; keep the DAgger guard (no "success by stopping earlier"); use the S2b deep strata and the 25→45 % fine-tune baseline as references.

**Falsifier.** No gain over the reactive policy at equal budget on d* ∈ [13,24].

### Lock 3 — Web executor and its corpus (compiler side done)

**Question.** Once an executor is trained on compiled pages, does it act correctly on real web tasks — and does the calibrated refusal survive on web goals?

**State.** The compiler side is **done for static pages** on the probe: compiler v2 reaches **100 % (504/504)** live under the v1 actionability definition frozen verbatim (v1 parity 5.86 % reproduced by replay), with a closed web vocabulary, LINK action, deterministic id de-dup, and 21 tests. The browser end-to-end assembly proves compile → decide → execute with traced shims; the executor currently **refuses** (calibrated STOP) because its training world is SIW, not compiled pages.

**Needed pieces.** (1) an **execution corpus of compiled pages** (planned 300–400 pages; the corpus is both training set and benchmark); (2) the consumer's tensorizer (web vocabulary, D_IN) and a LINK action in the policy's action space; (3) re-measure refusal calibration on web tasks after training; (4) extend coverage beyond static HTML (JavaScript/SPA, iframes/shadow DOM) with an explicit boundary; (5) the in-context line stays gated behind a value-of-information test (≥2× Δ_min) if it is revisited at all.

**Falsifier.** With a corpus and tensorizer in place, web-task execution stays near zero outside refusal, or refusal calibration on web tasks cannot be recovered to ≥90 % recall at ≤5 % false refusals.

## 3. Secondary work items

1. **Auxiliary-target line (effect head): close it or re-open with fresh data.** v02 executed the planned arms at 4 000 updates but re-used v01 seeds/data and closed **non-confirmatory**; the +34 pp never replicated and the validity target is outside the canon. Options: drop the head (the decision path does not need it), or run a *fresh-data* prospective design at planned power (4 families, 800 tests/family, new seeds) if the auxiliary-signal question still matters.
2. **In-context re-entry gate:** latent per-episode effects, discriminability shown before training, value-of-information ≥ 2× Δ_min, and a shuffled-target control that cannot be vacuous. The web switch rule re-opened this line, but only behind the gate.
3. **Attribution-scale transfer:** the V1-bis instruments are sound; a harder target family (or a shorter budget regime with headroom, e.g. k ≤ 32 on a genuinely difficult family) is the way to make any transfer claim falsifiable. Keep the joint cell table updated so dependency clusters are never averaged away.
4. **Unreachable-goal generator parity:** the S5 impossible-goal construction is now part of the DSL toolkit; keep it as a first-class generator with oracle verification.
5. **Product:** train the executor on the compiled-page corpus; replace the stub with real voice and the real comprehension model; measure the economic comparison instead of modelling it.

## 4. What would make the paper obsolete (in a good way)

- A strictly compositional successor that still runs at ~1 ms and ~1 M params.
- An external replication of GATE-2/GATE-3 on independently generated TGK/SIW canons (R3).
- A published 2026–2027 result on compact relational controllers with search that reaches deep plans without scaling parameters.
- A working DOM→policy compiler with ≥90 % coverage and closed-loop execution measured on real pages.

## 5. Non-goals (unchanged)

No claim of general intelligence; no multi-domain framework before the locks; no RL/value/reconstruction terms in the main arm; no text/pixels as policy input; no deployment without measured abstention; no post-hoc threshold movement.
