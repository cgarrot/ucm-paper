# 06 — Reproducibility and Research Integrity

*The instrument is part of the result. This document records how claims were protected from the researchers, what broke, and how it was repaired. Sources: `code/ucm/v1/atomic_publish.py`, `episode_budget.py`, `sealed_reader.py`, `freeze_v1bis.py`, `archive/docs/ERRATUM-*.md`, `archive/docs/INCIDENT-*.md`.*

## 1. The principle

A four-day program at this scope is only auditable if the machine enforces the promises. The project therefore treats the following as **testable instruments**, not conventions:

| Instrument | Invariant | Test |
|---|---|---|
| One-read registry | a sealed artifact is opened exactly once; intent logged before open | `test_confirm_v3.py`, `test_main_ingestion_fd.py` |
| Episode budgets | k = complete episodes with terminal STOP; prefix-nested, hash-published | `test_episode_budget.py` |
| Freeze manifest | code/input drift aborts **before** any read | `test_freeze_v1bis.py` |
| Atomic publish | never overwrite; crash at any point leaves a consistent state | `test_atomic_publish.py` |
| Journals | append-only; invalid tail refused, never truncated | `test_atomic_publish.py` |
| Interaction log | counts everything or fails; SHA-chained | `test_interactions_merge.py` |
| Anti-leak | supervision/provenance cannot change tensors | `test_model.py` |
| Terminal STOP invariant | every optimal walk ends at `d* = 0` with `{STOP}` | centralized `walk_optimal_plan` + fail-closed raise |

## 2. Fail-closed by construction

- Critical invariants **raise** (`RuntimeError`), they are not `assert` statements — `python -O` strips asserts (lesson recorded after a critical check was found strippable).
- A freeze collision (`EEXIST`) is an error, not a retry; no artifact is ever overwritten.
- The publication pointer is a hardlink of the manifest: if the bundle is incomplete, the pointer cannot exist; if the pointer exists, the bundle verified.
- Journal tails that fail validation are refused; a human must investigate.
- Stage-B evaluation refuses to run without the sealed test and an explicit GO token.

## 3. Errata ledger (13)

| # | Erratum | What was wrong | Correction |
|---|---|---|---|
| E1 | Protocol V1-bis values | adaptation updates/batch omitted | 2000 / 64 fixed before run |
| E2 | Bootstrap definition | "hierarchical" = 2 levels judged anti-conservative | superseded by E3 |
| E3 | SD / bootstrap / estimand | 10.9 pp SD underivable | SD re-derived 12.61 pp (n−1), ≈75 % power at 10 seeds, 1-level block bootstrap, eval = same 600 episodes |
| E4 | GATE-2/M3 failure count | 83 vs 79 discrepancy | stochastic replay (RNG/harness differ), official 79/3160 |
| E5 | Recurrent profile context | "full" 11.79 ms vs model-only 14.08 ms confusion | gate measure = conservative model-only; ratios and RSS published |
| E6 | Null-source records | provenance/label distribution error | corrected append-only |
| E7 | D9 append-only | journal tail semantics | clarified: refuse, never truncate |
| E8 | Distribution ESS/κ audit | sampling-efficiency estimate | corrected |
| E9 | S2b report | KILL scope over-general; G1′ not run; data effect over-interpreted | scope narrowed to "refine without identity path over frozen head"; retention annex completed; data effect p=0.094 not promoted |
| E10 | P2-1 requalification | in-context claimed tested | non-identifiable by construction; "reads context" withdrawn; discriminant addendum (presence ≠ content) |
| E11 | Cycle over-claims | 4 over-statements (incl. VISION 4/4, §13.4 GO) | VISION corrected then re-earned on correct base; §13.4 "not tested"; strict-composition factors not disentangled |
| E12 | Cycle epistemic | exploratory test cited as pre-registered | intra-band GO pre-registered, strict KILL exploratory; post-freeze tests never cite the frozen design |
| E13 | P2 closed-loop ablation | +34 pp presented as discovery; dimensioning gap | not replicated; INDETERMINATE; n_test 12 vs design 800 documented |

## 4. Incidents (5)

| # | Incident | Impact | Closure |
|---|---|---|---|
| I1 | V1 sealed-read incident | one-read violated (11 opens), provenance hash duplicated, raw not persisted | campaign reclassified EXPLORATORY; V1-bis instrumented |
| I2–I4 | Manifest read v1–v3 | read-protocol violations during manifest checks | strict one-read instruments, intent-before-open logging |
| I5 | Checkpoint reconciliation | s5–s9 artifact dirs are width 192, not 144 | runtime shape filter verified no run ever loaded them; incident closed **without re-execution** |

## 5. Seven interceptions before contamination (V0)

1. Distribution imbalance: a category generated at 0 % — caught by per-cell re-measurement after any distribution fix.
2. Test-split zeros: zero-count episodes polluting test cells.
3. Seal on a stale generator revision.
4. Mapping-harness index bug: invariance proven, closed loop at 3 %.
5. Verdict harness CI sign inversion (A−B vs B−A) — twice, on different artifacts.
6. Frozen decisions not implemented (cost statistic, tie criterion).
7. In-sample/one-seed/asymmetric "4/4" gate reading → resolved toward the harder bar.

Cumulative cost ≈1 hour; each interception is documented in `archive/docs/PLAN.md` §6 (lessons registry, 23 entries).

## 6. What a reproducer needs

| Item | Location |
|---|---|
| Code (frozen) | `code/` (snapshot of working repo commit `a47f7b7`) |
| Curated result JSONs | `results/json/` (33 files) |
| Original French reports/protocols/errata | `archive/docs/`, `archive/spec/` |
| Full artifacts (451 JSON, 500 NPZ, 1.7 GB) | working repository (private) |
| Environment | Python 3.12.13, MLX 0.32.2, numpy 2.5.3, macOS 26.6.2, Apple M5 32 GB; CPU-only equivalence tests; optional RTX 3070 second backend documented but not required |
| Determinism | canon byte-identical on M5 / Debian XMG / auditor replay; seeds and RNG schemes fixed per campaign (10k/20k/30k/40k + seed for eval) |

## 7. Known gaps (declared)

- Raw per-episode rows for V1 (first campaign) were not persisted — hence the reclassification.
- The P2 ablation executed at n_test=12/family, far below design; conclusions are labelled INDETERMINATE.
- The web probe is static (no live page agent), 4 pages, no LINK action tests.
- The product-line numbers use a declared stub, not a real comprehension model.
- The recurrent KILL is conservative by design (frozen base asymmetry) and its scope is explicitly narrowed.
