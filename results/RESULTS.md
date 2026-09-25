# Results — curated artifacts

This directory contains the **curated result artifacts** cited by the paper and the ledger. Every file is copied verbatim from the working repository's `artifacts/` tree (no re-computation). Full raw data (451 JSON, 500 checkpoints, ~1.7 GB) lives in the working repository.

## Files and the claims they support

| File | Supports |
|---|---|
| `json/v0-gate2-confirmation.json` | Paper §8.1, Ledger §1.1 — 97.50 % generalization, CI, per-goal |
| `json/v0-gate3-zeroshot-official.json` | Paper §8.1, Ledger §1.1 — 99.35 % composition, bands, dedup disclosure |
| `json/v0-gate5-arch-diagnostic.json` | Paper §8.2, Ledger §1.2 — A vs B diagnostic (train) |
| `json/v0-gate5-val-stratum.json` | Paper §8.2, Ledger §1.2 — promotion on layout-disjoint val (+62.2 pp) |
| `json/v0-cost-4cells.json` | Paper §8.3, Ledger §1.2 — cost protocol cells |
| `json/v0-m3-controls.json` | Paper §8.2, Ledger §1.3 — interface controls, permutation test |
| `json/v0-m3-failures-baselines.json` | Paper §8.2, Ledger §1.3 — taxonomy, G3/G4 depth, GATE-4 baselines |
| `json/v0-at-stratification.json` | Ledger §1.3 — AT per-depth stratification |
| `json/v1-exploratory-transfer.json` | Paper §8.4, Ledger §2 — exploratory transfer, controls, AULC |
| `json/v1-null-label-distribution.json` | Ledger §2 — null control label distribution |
| `json/v1bis-stage-b-metrics.json` | Paper §8.5, Ledger §2 — 120 cells, per-predicate rates, verdicts |
| `json/v1bis-verdict-k64.json`, `-k128`, `-k256` | Paper §8.5 — verdicts per budget |
| `json/v1bis-raw-sample-200.jsonl` | Paper §8.5 — raw schema sample (first 200 of 72 000 rows) |
| `json/v1bis-freeze-v10.json` | Paper §7.2 — freeze manifest of the confirmatory campaign |
| `json/s2b-run3-s100.json` … `-s104.json` | Paper §8.6, Ledger §3 — factorial run3 per-seed cells |
| `json/s2b-retention-annex.json` | Paper §8.6 — retention bands from checkpoints (no retraining) |
| `json/p2-q4-effect-head.json` | Paper §8.7 — Q4 discriminant (native vs P2 vs run4 arms) |
| `json/p2-ablation-controlled-full.json` | Paper §8.7 — 4 arms × 12 seeds training convergence |
| `json/p2-ablation-controlled-rapid-v01.json` | Ledger §4 — rapid pilot of the ablation |
| `json/p2-ablation-closed-loop.json` | Paper §8.7 — closed-loop verdict (A/B/C/D) |
| `json/p2-ablation-closed-loop-erratum.json` | Paper §8.7 — post-hoc statistics and design-deviation record |
| `json/p2-h1-h2-probes.json` | Paper §8.7 — H1 strata, H2 shortcut and zero-shot rate |
| `json/s5-full-budget.json` | Paper §8.8 — 44/44, p95 1.146 ms, 1 120×, false refusals 0/40 |
| `json/s5-refusal-calibration-v2.json` | Paper §8.8 — calibration design and results |
| `json/s5-escalation-recall.json` | Paper §8.8 — first remediation (recall 1.0, false 0.675) |
| `json/s5-real-demo-simulated.json` | Paper §8.8 — 32/32 demo with labelled stub |
| `json/web-compiler-probe.json` | Paper §8.9 — 5.86 % coverage, per-page decomposition |

## Provenance rule

Every claim line in `docs/05-results-ledger.md` cites the file here, the world, and the model. If a claim cannot cite an artifact, it does not exist as a claim. This mirrors the project's internal rule ("each verdict line cites artifact + world + model, mechanically verified").

## Raw sample schema (`v1bis-raw-sample-200.jsonl`)

Each line is one episode evaluation row (14 fields): arm, seed, k, episode_id, predicate, layout hash, success, outcome, length, `n_invalid`, `d_star`, `L_star`, and two provenance fields. The full 72 000-row file was published in the working repository before any aggregation.
