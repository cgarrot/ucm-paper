# 08 — Roadmap and Open Locks

*Where the project stands at snapshot `a47f7b7` (2026-09-25) and what would falsify the next steps.*

## 1. State summary

| Piece | State |
|---|---|
| Vision stack (voice → comprehension → executor → measured result) | defined; executor built |
| V0 TGK generalization / composition | **established** (97.5 % / 99.4 %) |
| V0 cost / latency | established (~1 ms, 1.921× vs 1-hop gate) |
| Verified DSL layer | established (differential ≥10k states, 3 mutations detected) |
| V1-bis transfer TGK→SIW | **not demonstrated** (saturated regime; bound ≤ +0.73 pp) |
| S2b recurrence | **KILL** (−7.7 pp vs +15.8); retention passes |
| P2 context | **not testable as designed**; presence-distraction measured |
| P2 effect head | **not replicated**; INDETERMINATE at executed power |
| S5 product line | internal criteria met in synthetic demo (44/44, 32/32; stub) |
| Real-software compiler | **5.86 %** probe; layer missing |
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

### Lock 3 — Real-software compiler

**Question.** Can DOM→policy_input compilation reach ≥90 % actionable-element coverage with a vocabulary and tensorizer that match real pages?

**Needed pieces (identified by the probe).** A LINK action signature (hyperlinks are new signatures to learn, not to "compile away"); a web vocabulary + tensorizer (D_IN beyond the closed P2 set); DOM id de-duplication; a live page-agent path (the probe is static); and a decision on the in-context line (re-opened by the switch rule) with an information-value gate (≥2× Δ_min) this time.

**Falsifier.** Coverage stays <90 % with a correct compiler, or coverage ≥90 % but closed-loop execution does not transfer.

## 3. Secondary work items

1. **Effect head decision** (three options on the table): re-run the v02 design at planned power (4 families, 800/family, 12 paired seeds), standardize the simple validity/termination auxiliary target, or drop the head and reallocate to the locks. Recommendation in the record: do not publish +34 pp; if kept, use the v02 design.
2. **In-context re-entry gate:** latent per-episode effects, discriminability shown before training, value-of-information ≥ 2× Δ_min, and a shuffled-target control that cannot be vacuous.
3. **Attribution-scale transfer:** the V1-bis instruments are sound; a harder target family (or a shorter budget regime with headroom, e.g. k ≤ 32 on a genuinely difficult family) is the way to make any transfer claim falsifiable.
4. **Unreachable-goal generator parity:** the S5 impossible-goal construction is now part of the DSL toolkit; keep it as a first-class generator with oracle verification.
5. **Product:** replace the stub with real voice and the real comprehension model; measure the economic comparison instead of modelling it.

## 4. What would make the paper obsolete (in a good way)

- A strictly compositional successor that still runs at ~1 ms and ~1 M params.
- An external replication of GATE-2/GATE-3 on independently generated TGK/SIW canons (R3).
- A published 2026–2027 result on compact relational controllers with search that reaches deep plans without scaling parameters.
- A working DOM→policy compiler with ≥90 % coverage and closed-loop execution measured on real pages.

## 5. Non-goals (unchanged)

No claim of general intelligence; no multi-domain framework before the locks; no RL/value/reconstruction terms in the main arm; no text/pixels as policy input; no deployment without measured abstention; no post-hoc threshold movement.
