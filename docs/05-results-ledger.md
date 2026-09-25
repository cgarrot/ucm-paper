# 05 — Results Ledger

*Every quantitative claim of the project, with artifact, world, model and epistemic status. This ledger is the single source of truth for the paper's numbers. Statuses: **[E]** established (pre-registered + sealed + audited), **[X]** exploratory, **[N]** not established / failed / inconclusive.*

## 1. V0 — TinyGraphKey

### 1.1 Generalization

| Claim | Value | Artifact | Status |
|---|---|---|---|
| GATE-2 closed-loop success, B144, 5 seeds | **0.9750**, CI [0.9648 ; 0.9850] | `results/json/v0-gate2-confirmation.json` | [E] |
| Episodes / clusters | 3 160 episodes, 630 layout clusters, 10 000-bootstrap | idem | [E] |
| Per-goal success | REACH 0.9876 (n=1050) · HAVE 0.9972 (n=1060) · AT 0.9400 (n=1050) | idem | [E] |
| Mean regret on successes | 0.0558 | idem | [E] |
| Invalid-step rate (paid) | 0.0996 | idem | [E] |
| Premature STOP rate | 0.00032 | idem | [E] |
| GATE-3 zero-shot composition (reserved cell) | **0.99352**, CI [0.98596 ; 0.99878] | `results/json/v0-gate3-zeroshot-official.json` | [E] |
| GATE-3 volume | 494 episodes × 5 seeds = 2 470 reads, 102 layouts, 2 dedup skips disclosed | idem | [E] |
| GATE-3 by band (d* 3…10) | 1.000 / 1.000 / 0.9986 / 1.000 / 0.9959 / 0.9185 / 0.9692 / 0.900 | idem | [E] |
| GATE-3 invalid rate / regret | 0.0046 / 0.0228 | idem | [E] |

### 1.2 Architecture election and cost

| Claim | Value | Artifact | Status |
|---|---|---|---|
| GATE-1 Model A overfit OA (non-trivial) | **0.9658** vs bar 0.99 → FAIL | `archive/docs/REPORT-V0.md` | [N] |
| A bit-exact tie on symmetric pair | logits 0.410357 == 0.410357 | `code/tests/test_model.py::TestTieSeparability` | [E] |
| GATE-5 diagnostic (train), B−A | +9.51 pp, CI [5.98 ; 13.47], 305 pairs, 5 seeds | `results/json/v0-gate5-arch-diagnostic.json` | [E] |
| GATE-5 promotion (val, layout-disjoint), B144−A | **+62.20 pp** (0.8813 vs 0.2593), CI [54.63 ; 68.90], 455 pairs, 210 clusters | `results/json/v0-gate5-val-stratum.json` | [E] |
| Safe stratum d*≤2 (A → B144) | 0.8414 → **1.000** | idem | [E] |
| Official sustained cost ratio B144/A | **1.921** (IQR 0.309, corroborated 1.913) | `archive/docs/PLAN.md` §6 #12–13 | [E] |
| Cost ratio B160 / B192 | 2.035 (5/5 blocks >2) / 3.05, 2.31, ~2.15 | idem | [N] |
| Widths / params | A 366 337 · B144 694 513 · B160 856 161 · B192 1 230 145 | `results/json/v0-gate5-*.json` | [E] |

### 1.3 Controls, baselines, failures, depth

| Claim | Value | Artifact | Status |
|---|---|---|---|
| Control: no goal | success **0.0** (28 % goal-reached-without-STOP) | `results/json/v0-m3-controls.json` | [E] |
| Control: candidates zeroed | **0.0** (always STOP) | idem | [E] |
| Control: relations removed | **0.0** (always STOP) | idem | [E] |
| Control: counterfactual goal | 0.9604 | idem | [E] |
| Metamorphic permutation | outcomes identical 200/200 | idem | [E] |
| GATE-4 baselines on test_g1 | model 0.9750 · local heuristic 0.8259 · random-valid 0.6060 · random-syntactic 0.0459 | `results/json/v0-m3-failures-baselines.json` | [E] |
| Failure taxonomy (pre-specified) | 46 loops (55 %) · 36 repeated invalid (43 %) · 1 other; total 83 inspected / official 79 of 3 160 | idem | [E] |
| G3 (326 eps, 9–12 rooms) | 0.9448 overall; bands: d*2 1.00 → d*12 0.25 | idem | [E] |
| G4 (103 eps, d*13–24) | 0.3689; d*13 0.463, d*14 0.545, d*15 0.400, d*16 0.000, d*17 0.091, ≥18 0.000 | idem | [E] |
| Effective planning depth | ~12–15 actions (mechanistic: 3 message rounds + greedy) | `archive/docs/REPORT-V0.md` | [E] |
| AT stratification | 1.000 at d*3–5 (0 failures), 0.940 at 6–8, 0.679 at 9–11 | `results/json/v0-at-stratification.json` | [E] |

### 1.4 Runtime

| Metric | Value | Status |
|---|---|---|
| Batch-1 p95, quiet CPU | 0.52–0.66 ms | [E] |
| Interleaved sustained p95 | ~1.04 ms | [E] |
| Margin under 20 ms envelope | 16–38× | [E] |
| Accelerator budget consumed | ≈4 h 50 of 24 h | [E] |
| Official train+eval wall | 86.3 min | [E] |
| Peak RSS train / batch-1 inference | 749 MB / 67.6 MB | [E] |
| Canon reproducibility | byte-identical on 3 machines | [E] |

## 2. V1 / V1-bis — transfer TGK → SIW

| Claim | Value | Artifact | Status |
|---|---|---|---|
| V1 exploratory primary (k*=500) pretrained−scratch | +12.57 pp, CI [2.56 ; 22.35] | `results/json/v1-exploratory-transfer.json` | [X] |
| V1 control−scratch / control−pretrained | +13.37 [3.77 ; 22.76] / +0.81 [−13.32 ; 13.88] | idem | [X] |
| V1 AULC log k | scratch 0.199 · pretrained 0.253 · control 0.287 | idem | [X] |
| V1 instrument violations | 11 opens (one-read not held), duplicated provenance hash, no raw/checkpoints | `archive/docs/INCIDENT-2026-09-22-sealed-read.md` | [N] |
| V1-bis arm means k=64/128/256 | scratch 97.10/98.18/99.18 · pretrained 97.90/98.08/99.10 · control-validity 96.22/96.98/98.95 · control-null 96.37/98.17/99.34 | `results/json/v1bis-stage-b-metrics.json` | [E] |
| V1-bis verdicts | k=64 +0.80 FAIL · k=128 −0.10 FAIL · **k=256 −0.08 FAIL (primary)** | `results/json/v1bis-verdict-k*.json` | [E] |
| Saturation margins by k | 2.90 / 1.82 / 0.82 pp | `archive/docs/REPORT-V1BIS-FINAL-v2.md` | [E] |
| Corrected upper bound at k=256 | ≤ **+0.73 pp** (95th pct) | idem | [E] |
| k=64 secondary | one-sided p ≈ 0.028 (Bonferroni ≈ 0.085), −28 % relative failures (126 vs 174) | idem | [X] |
| STOP artifact elimination | `goal_reached_without_stop = 0 / 72 000` (V1: 29–44 %) | `results/json/v1bis-stage-b-metrics.json` | [E] |
| Campaign volume | 120 cells, 4 workers, 3 h 37 wall, 14.5 h CPU-cumulated, 72 000 raws | `archive/docs/REPORT-V1BIS-FINAL.md` | [E] |
| Verdict | **non-demonstration in a saturated regime**, not a refutation | idem | [E] |

## 3. S2b — recurrence (2×2)

| Claim | Value | Artifact | Status |
|---|---|---|---|
| Per-seed B144-oracle (s100…s104) | 55.1 / 47.4 / 47.4 / 61.5 / 52.6 → mean **52.8** | `results/json/s2b-run3-s100..104.json` | [E] |
| Per-seed CIV(T=32)-oracle | 51.3 / 47.4 / 42.3 / 28.2 / 56.4 → mean **45.1** | idem | [E] |
| Difference | −3.8 / 0.0 / −5.1 / −33.3 / +3.8 → mean **−7.7 pp** vs frozen **+15.8** | idem | [N] KILL |
| Retention d*≤8 (annex) | 97.9–99.4 %, ≤2 pp everywhere → PASS | `results/json/s2b-retention-annex.json` | [E] |
| Deep bands [13,24] / [16,24] | B144-oracle 45.2 / 21.6 · CIV-oracle 38.6 / 18.4 · canon zero-shot 25.2 / 8.0 | idem | [X] |
| Data-source effect (oracle>recovery) | +6.4 pp, one-sided p = 0.094 (n=5) → **not promoted** | `archive/docs/ERRATUM-REPORT-S2B-2026-09-24.md` | [N] |
| Real MDE | ≈18.5 pp (seed SD 5.9 pp) > frozen 15.8 pp | idem | [E] |
| Recurrent profile | p95 14.08 ms model-only; per-update p50 11.9×, sustained ≈10.1×; RSS 9.7 GB vs 1.4 GB | `archive/docs/ERRATUM-2026-09-24-recurrent-profile-context.md` | [E] |

## 4. P2 — context and auxiliary head

| Claim | Value | Artifact | Status |
|---|---|---|---|
| Q4 free contrast (DROP-HAVE, 8 seeds) | native GNNB 0.3125 = P2 λ=0 0.3125 · run4 without context D0 0.65625 · D1-mixed 0.55208 · D1-shuffled 0.27083 | `results/json/p2-q4-effect-head.json` | [X] |
| Initial reading "+34 pp effect head" | **not replicated** by controlled ablation | `archive/docs/ERRATUM-CYCLE-OVERCLAIMS-2026-09-25.md` | [N] |
| Controlled ablation (12 seeds, 2000 updates) | A 35.4 % · B 41.0 % · C 29.9 % · **D 52.8 %** | `results/json/p2-ablation-closed-loop.json` | [N] INDETERMINATE |
| B−A / B−C / B−D | +5.6 pp (CI ∋ 0) / +11.1 pp / **−11.8 pp** | idem | [N] |
| Head auxiliary only | policy logits identical by construction; head never trained when disabled | `code/ucm/p2/model_p2.py` | [E] |
| Effect learning: content matters | effect loss B 0.096 vs C 0.129 | `results/json/p2-ablation-controlled-full.json` | [X] |
| In-context identifiability | **not testable by construction** (constant semantics; history from another episode) | `archive/docs/ERRATUM-P2-1-REQUALIFICATION-2026-09-25.md` | [N] |
| Measured context effect (DROP-HAVE) | −18.8 pp (8/8 seeds); p pre-specified 0.031; post-hoc 0.055/0.11 Holm-killed | `results/json/p2-ablation-closed-loop-erratum.json` | [X] |
| Discriminant addendum | D1-mixed weights: 55.2 % without context vs 46.9 % with; shuffled 27–28 % both ways | `archive/docs/ERRATUM-CYCLE-OVERCLAIMS-2026-09-25.md` | [X] |
| H1 stratum B (AT d*≤12 with door) | 0.80 (20/25) | `results/json/p2-h1-h2-probes.json` | [E] |
| H1 stratum A (AT d*20–24 without door) | **N = 0** — structurally absent from the sampler | idem | [N] |
| H2 shortcut (initial argmax = DROP) | 0.0833 < 0.40 threshold → shortcut does not exist | idem | [E] |
| H2 zero-shot closed loop on DROP-HAVE | 0.5833 (7/12) → real zero-shot planning | idem | [X] |
| Dimensioning gap | executed n_test = 12/family vs frozen 800/family; 2000 vs 4000 updates | `results/json/p2-ablation-closed-loop-erratum.json` | [N] |
| Ablation v02 (72 cells: 2 families × A/B/D × 12 seeds, 4000 updates) | per-arm means over both families: A 62.2 % · B 66.3 % · D 71.5 % → **D−A = +9.4 pp** | `results/json/p2-ablation-v02-confirmation.json` | [X] |
| v02 frozen convention (bootstrap seed-cluster signed sum) | CI excludes 0, point ≥5 pp → level-1 GO *under that convention* | `results/json/p2-ablation-v02-verdict-erratum.json` | [X] |
| v02 sensitivity | paired t and z conventions FAIL (CI-low −0.5 / −0.0); D−B +5.2 pp non-significant after Holm; B>C not measurable (arm C removed) | idem | [N] |
| v02 final closure | **NON-CONFIRMATORY** (re-uses v01 seeds/data); D−A exploratory (p = 0.047 one-sided permutation, carried by saturated DROP-AT; DROP-HAVE p = 0.0625); **validity does not enter the canon** | `docs/ERRATUM-ABLATION-V02-VERDICT-2026-09-25.md` (archive) | [N] |

## 5. S5 — product line and abstention

| Claim | Value | Artifact | Status |
|---|---|---|---|
| Execution success (full budget) | **44/44 = 1.0** (bar ≥0.97) | `results/json/s5-full-budget.json` | [E]* |
| p95 per decision | **1.146 ms** (bar ≤5 ms) | idem | [E]* |
| Cost ratio vs Jev stub | **1 120×** (147 decisions, 157.5 ms UCM, JEV_MS=1200) | idem | [E]* |
| Abstention recall | **40/40 = 1.0** (bar ≥0.90) | idem | [E]* |
| False refusals | **0/40 = 0.0** (bar ≤5 %) | idem | [E]* |
| Calibration v2 design | 80+80 discriminant pairs, 874 records (15.47 %), 4000 updates, frozen before training | `results/json/s5-refusal-calibration-v2.json` | [E] |
| First remediation (context) | recall 1.0 but false escalations 0.675 | `results/json/s5-escalation-recall.json` | [N] |
| Real demo (simulated Jev) | 32/32; ratio 505.9–581×; Jev stub 0.026 ms/req; voice layer not exercised | `results/json/s5-real-demo-simulated.json` | [X] |
| Required formulation | "internal criteria met in a synthetic demo with simulated Jev; real voice/Jev and real economic benefit remain to be established" | `archive/docs/passation-*` (working repo) | [E] |

\* Established as an internal, synthetic demonstration with a labelled stub; not an end-to-end product measurement.

## 6. The software bridge

| Claim | Value | Artifact | Status |
|---|---|---|---|
| v1 probe baseline (frozen actionability definition) | **5.86 %** (30/512 actionable) vs bar ≥90 % | `results/json/web-compiler-probe.json` | [N] historical |
| Compiler v2 on calibrated snapshots | **100 % per page** (4/4, 1/1, 5/5, 494/494 = 504/504); v1 parity reproduced exactly (512/30/5.86 %) | `results/json/web-compiler-v2-report.json` | [E] |
| v2 blockers treated | closed web vocabulary + LINK action; deterministic DOM-id de-dup (`fname_2`, `lname_2/_3`); href resolution 8 kinds | idem | [E] |
| v2 live validation | real refetch of the 4 pages; **100 % everywhere, 504/504 live**; HTML persisted as evidence; GO ≥95 % PASS | `results/json/web-compiler-v2-live.json` | [E] |
| v2 tests | 21 tests (id collisions, href kinds, form nesting, v1 parity, vocabulary rejection, hermetic regeneration, artifact reproducibility) | `code/tests/test_web_compiler.py` | [E] |
| E2E smoke 1 (real page) | 31 actionables, 32 candidates, correct STOP (VIEW already satisfied), decision 11.233 ms | `results/json/web-e2e-smoke1.json` | [X] |
| E2E smoke 2 (real httpbin form) | 4 actionables, 3 fields compiled (`custtel`, `custemail`, `comments`), injected SET goal → **STOP = calibrated refusal** (training world ≠ compiled pages) | `results/json/web-e2e-smoke2.json` | [X] |
| E2E mechanics | compile → decide → execute through a real browser daemon + extension; shims traced (link→button, SELECT via option click, NAVIGATE via tool) | idem + working repo `ucm/web/e2e_bridge.py` | [X] |
| Web executor decision | **not established** — requires an execution corpus of compiled pages (planned 300–400; both training set and benchmark) | — | [N] |
| Daemon quirk | `navigate` returns HTTP 500 while navigating; traced for the daemon side (audit note b) | working repo `artifacts/web-probe/e2e/` | [E] |

## 7. Instrument-level facts

| Fact | Value |
|---|---|
| Commits / days | 435 / 4 (2026-09-22→25) |
| Python LOC / tests | 37 007 / 581 (21 web-compiler tests) |
| Artifacts (working repo) | 451+ JSON, 500 NPZ, ~1.7 GB; +72 ablation-v02 cells, +web-probe-v2/live-v2/e2e |
| Errata / incidents | 14 / 5 (+1 audit note for the browser daemon quirk) |
| V0 interceptions before contamination | 7 (≈1 h total cost) |
| Sealed-campaign freezes | v4 → v10 |
| Raw rows published before aggregation (V1-bis) | 72 000 |
| Joint cell table (V1-bis) | 122 rows; 10 adaptations share 5 canonical sources (dependency clusters exposed) |
| DSL closure coverage | ≥10 000 states/world; 3/3 mutations detected |
| Anti-leak tests | supervision/provenance cannot change tensors (bit-identity) |
| Web compiler parity | v1 definition replayed on snapshots: 512/30/5.86 % reproduced exactly |
