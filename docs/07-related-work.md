# 07 — Related Work and Positioning (2026)

*Condensed from the project's web watch (`archive/review/REPORT.md`, `FOURTH-REVIEW.md`, `FIFTH-REVIEW.md`), which surveyed the 2025–2026 literature across six themes. Status tags from the original: [V] verified full text, [A] abstract, [S] snippet.*

## 1. The category UCM lives in

**System One Models are now a category.** TypeSafe's announcement (Sept. 2026) [V] formalizes non-autoregressive typed probabilistic decisions with parallel sampling (70–500 ms per call) and calibration via RLCD. **cua-s1-forms** (trycua, Sept. 2026) [V] is the closest published artifact: **706 048 parameters**, a 2-layer byte-level Transformer encoder scoring form-field options one-pass, trained on 10 000 synthetic episodes with signature-disjoint splits and a shuffled-context control (37 % vs 99.95 % top-1). It also *hard-codes* execution order in downstream code — the model only scores.

**Positioning:** UCM shares the thesis (small, typed, one-pass, ~0.7 M params) and independently converged on the evaluation practices (signature-disjoint splits, shuffled controls). It differs in kind: *relational graph observations*, *goal-conditioned multi-step control with an exact oracle*, and *transfer between worlds*. No paper found in the survey combines those three at this scale.

## 2. Relational policies for planning

- **Per-Domain Generalizing Policies** (DFKI/Sarre 2026) [V]: distilling Q* from an exact planner into a relational GNN. Documented trap: vanilla Q* regression does **not** separate optimal from non-optimal actions without a margin regularizer. Their Q policy is one forward pass, 3–18× cheaper than V+successors. UCM trains a set-valued log-likelihood over candidates instead, and evaluates on world families rather than fixed PDDL domains.
- **Learning to Search and Searching to Learn** (RWTH/Geffner 2026) [V]: Q-GNN guides a best-first search; the search data retrains Q; strong size generalization (30 → 488 blocks zero-shot). Lesson recorded by UCM's watch: when the model is exact, best-first beats real-time for data generation; and search↔learn loops are the current best answer to depth.
- **Neural Value Iteration** (2026) [S] and **efficient lookahead encoding** [S]: related answers to depth, not directly comparable at UCM's scale.

## 3. Meta-learning and composition in randomized worlds

- **AnyMDP** (NeurIPS 2025) [A]: in-context RL meta-trained in randomized MDPs; no relational structure or shared genres.
- **RACES** (2026) [V]: typed composition of verifiable environments (domain/codomain signatures, SEQUENTIAL/PARALLEL/SORT/SELECT); 50 base environments ≈ 300 individual ones for LLM reasoning. The clearest 2026 evidence that *typed world composition* buys diversity cheaply.
- **Agent-World / AgentMercury / C-World** [A/S]: large-scale environment synthesis for general agents (LLM-centric).

**Positioning:** the niche "procedural control worlds with typed interaction genres for a compact relational controller" remains empty. UCM's P2 attempted it; its in-context arm failed identifiability (a documented design failure), and strict composition remains the open lock.

## 4. Evaluation integrity literature

- **Preregistration for Experiments with AI Agents** (MIT, 2026) [V]: extends preregistration to agent experiments; catalogues researcher degrees of freedom and "invisible specification search". UCM's gates, sealed canons, frozen thresholds and raw-first publication are directly aligned; the watch judged UCM "ahead of this literature" in instrument discipline.
- **Automated Benchmark Auditing** (2026) [V]: >25.7 % of audited tasks across 168 benchmarks have critical problems; filtering shifts rankings by ~+10 %. Validates UCM's construct-validity obsession (and its R* repair).
- **Monitoring Web Agents Without Internal Signals** (2026) [V]: propagating a terminal failure label to all prefixes is temporally inexact supervision — the exact mirror of UCM's V1 support defect. Independent corroboration that the V1 diagnosis was a real construct-validity bug class.
- **When Agents Do Not Stop** (2026) [V]: infinite agentic loops as a first-class failure; UCM's absorbing-loop dominance (55 % of failures) matches this taxonomy for compact controllers.

## 5. Things UCM explicitly is not

- Not a latent world-model planner (no imagination, no probes, no judge): **state + goal + candidates → action scores**.
- Not autoregressive, not language-conditioned, no pretrained embeddings; labels are never embedded.
- Not integrated with a constraint/judge layer; the judge/constraint design document (`archive/docs/UCM-JUDGE-CONSTRAINT-LAYER.md`) is explicitly non-normative, and the recommendation is a *sidecar* (option 0), never in the fast path.
- Not a claim that the external judge/planner proposals (LeJudge/Jev) should be fused into the controller; their published numbers are not transposable to TGK/SIW.

## 6. The open frontier after this report

| Frontier | State of the art | UCM's gap |
|---|---|---|
| Depth via search | RWTH/Geffner: search↔learn loops, huge size generalization | UCM's short-search arm **not tested** (beam broken; greedy = reactive); reactive fine-tune generalizes 25 → 45 % on deep strata |
| Strict composition | RACES: typed environment composition with LLM | UCM: 0 % on one held-out signature (exploratory), 97 % intra-band |
| Compact typed controllers | cua-s1-forms: 706 k params, forms only, order hard-coded | UCM: relational, goal-conditioned, multi-step, abstention |
| Real software bridge | GUI/VLM agents (autoregressive, >0.8 B) | UCM: 5.86 % coverage static probe; compiler layer missing |
| Pre-registration | early literature | UCM: a working exemplar with 13 errata as the cost of honesty |
