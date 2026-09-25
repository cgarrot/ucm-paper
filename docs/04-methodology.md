# 04 — Experimental Methodology

*Gates, sealing, pre-registration, attribution controls, statistics and artifact discipline. Sources: `code/ucm/eval/`, `code/ucm/v1/`, `archive/docs/PROTOCOL-*.md`, `archive/docs/PLAN.md`, `archive/docs/ERRATUM-*.md`.*

## 1. The gate chain

Each gate has a question, a frozen threshold, and the power to stop the iteration. Thresholds are decided **before** the corresponding measurement; moving one after seeing results requires an erratum and a new campaign.

| Gate | Question | Frozen threshold | Outcome |
|---|---|---|---|
| GATE-0 | is the canon correct, sealed, reproducible? | byte-identical across machines | PASS (sha `9a19d8f4`) |
| GATE-1 | can the model fit at all (in-sample)? | non-trivial optimal-action rate ≥ 0.99 | **A FAILED 0.9658** → Model B |
| GATE-5 | which architecture, at what cost? | +5 pts depth stratum, ≤2 pts safe-stratum loss, cost ≤2×, tie separability | **B144 elected 4/4** |
| GATE-2 | closed-loop generalization on new instances | success ≥ 95 %, CI-low ≥ 90 %, per-goal ≥ 85 % | PASS 97.50 % |
| GATE-3 | zero-shot composition on a reserved cell | success ≥ 80 % | PASS 99.35 % |
| M3 | interface controls; failure taxonomy | controls collapse; taxonomy pre-specified | PASS (0.0 / 0.0 / 0.0) |
| GATE-4 | baselines off-distribution | report, compare | model 97.5 % vs heuristic 82.6 % |

Post-V0 experiments each froze their own protocol: **P2** (families, Δ_min windows, oracle gate before training, identifiability contract), **S2b** (2×2, threshold +15.8 pp, retention ≤2 pp), **V1-bis** (estimand, 4 arms, 10 seeds, k-plan, PASS = one-sided CI > 0 ∧ point ≥5 pp), **S5** (exec ≥97 %, false refusal ≤5 %, p95 ≤5 ms, ratio ≥10×, recall ≥90 %).

## 2. Sealing and the one-read invariant

- **Canon**: dataset generation produces a sealed split manifest (layout-grouped, hash-bound). Test splits refuse to load unless explicitly allowed, and never in training code paths.
- **One-read registry** (`sealed_reader.py`): a sealed artifact is read through a registry keyed by realpath, opened **once** per process/campaign, with intent logged *before* the open. Multi-k runs read the data file exactly once and derive prefix slices from the in-memory store.
- **Freeze manifests**: bind code files, input SHAs, budget plan, execution order, and seeds; verification aborts **before any read** if code or inputs drifted. The confirmatory V1-bis campaign used freeze v10 (rebuilds v7→v8→v9→v10 are part of the record).
- **Stage-B**: the locked final evaluation (120 cells), raw per-episode rows persisted before aggregation, verdicts recomputed from raw rows by a second party.

## 3. Episode-level budgets (R\* contract)

Adaptation budgets are **k complete episodes**, not k records: each episode is a physically reachable optimal-plan trajectory with a terminal STOP at `d* = 0` (schema 0.8). Budgets are nested prefix slices with published per-prefix hashes; the same slice is used across arms, so arm comparisons are paired by construction. Guards: contiguity, sequential depth, strictly decreasing `d*`, exact length `d*₀ + 1`; non-canonical files (CRLF, blank lines, missing final newline) are detected and refused before training.

## 4. Attribution controls

Four arms differing **only in supervision source**:

| Arm | Supervision | Purpose |
|---|---|---|
| `scratch` | none (random init) | floor |
| `pretrained` | canonical source-world weights, then target supervision | the effect |
| `control-validity` | one uniformly drawn **physically valid** action per record | removes decisional info, keeps validity structure |
| `control-null` | uniform over **all** candidates | removes validity structure too |

Same architecture, optimizer, updates, batch, checkpoint rule (`final_at_fixed_budget`, no target selection), nested data slice, and a **shared seed-matched fresh initialization** across arms before weight loading (to avoid confounding transfer with init luck). Published interpretation contract: control ≈ pretrained ⇒ optimization/exposure gain; control ≈ scratch ⇒ decisional gain; overlapping CIs ⇒ **no attribution**.

## 5. Statistics

- **Hierarchical paired bootstrap** where warranted (seeds, then layouts/tasks), 95 % CIs; one-sided thresholds for confirmatory claims.
- **Interface controls** use episode counts, not per-step rates, to avoid pseudo-replication.
- **Power**: design SDs must be derivable; an underivable SD (10.9 pp) was revoked and re-derived (12.61 pp n−1), accepting lower nominal power (≈75 % at 10 seeds for 10 pp). For S2b, inter-seed variance at the real base gave MDE ≈18.5 pp vs the frozen 15.8 pp — a documented threshold-design failure, not a re-run.
- **Failure-rate secondary**: log failure-ratio with Agresti +0.5 correction, robust to ceiling effects (implemented in `ucm.eval.probes_s2b`).
- **No post-hoc selection**: arms chosen on validation only; test cells read once; subgroup analyses are labelled exploratory; daft "stop earlier" wins are guarded against (DAgger guard).

## 6. Artifact and publication discipline

- **Immutable run artifacts**: `artifacts/<timestamp>-<name>/` with `config.json`, `metrics.jsonl`, `final.json`, `checkpoint.npz`; publication refuses to overwrite (O_EXCL) and fails closed on collisions.
- **Atomic bundles**: tmp file → fsync → hardlink to final name (no-replace) → fsync parent; a bundle manifest carries SHA-256 of every member plus the bundle path; the pointer is a hardlink of the manifest and the only publication signal.
- **Journals**: append-only framed entries (`len sha256 json`), one writer at a time (flock + pid + process start-time); an invalid tail is **refused**, never silently truncated; stale-lock recovery requires an explicit confirm plus death proof.
- **Interaction logs**: SHA-256-chained; a log counts only if it counts every recorded interaction or fails.
- **Raw-first**: evaluation stores per-episode rows (14 fields) before any aggregate; 72 000 rows for V1-bis.

## 7. What this discipline caught (selected)

| Catch | Cost |
|---|---|
| Distribution imbalance (0 % category) found by per-cell re-measurement | ~1 h |
| Mapping-harness index bug: invariance proven but wrong indexing → 3 % closed loop | ~1 h |
| Verdict harness sign inversion (A−B vs B−A), twice | ~1 h |
| Test-split zero pollution in generation | caught at generation |
| Stale-generator seal (seal on the wrong generator revision) | caught at gate |
| Frozen decisions not implemented (cost/tie criteria) | caught at audit |
| STOP-terminal regression in P2 records | caught by centralized invariant |
| Checkpoint width mismatch (s5–s9 = d192) | caught by runtime shape filter; no run ever loaded them |

## 8. Pre-registration template used for each campaign

1. **Question** (one sentence; what would falsify it).
2. **Estimand** (population, unit, aggregation, weights).
3. **Arms** (differ only in the manipulated variable; everything else fixed).
4. **Data** (source, splits, sizes, seeds; disjointness guarantees).
5. **Instrument guards** (shas, one-read, freeze, O_EXCL).
6. **Frozen thresholds** with power justification at the real base.
7. **Verdict rule** (PASS / INDETERMINATE / FAIL and their interpretations).
8. **Post-hoc rules** (what may be reported as exploratory; what may never be claimed).
9. **Publication plan** (raw-first, artifact names, erratum handling).
