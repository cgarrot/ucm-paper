# UCM — Universal Control Model

**A ~0.7 M-parameter relational controller that executes structured tasks fast, cheaply, and knows when to abstain.**

This repository is the research-readable companion to the UCM project: a scientific paper, detailed architecture and environment documentation, curated result artifacts, and a frozen source-code snapshot. It is the "paper repo" — the working repository (full history, all 1.7 GB of artifacts) is separate.

> **Status (2026-09-25, private / pre-publication).** The paper reports both what is **established** (pre-registered, GO) and what is **not established** (failed or inconclusive experiments, errata). We consider the negative results and the research-integrity machinery first-class contributions. Nothing here is a claim of general intelligence; UCM is an execution layer, not a brain.

---

## What UCM is

The project starts from a product observation: a large model can *understand* a request but cannot *execute* complex software tasks step by step without a large model call at every decision — too slow, too expensive. UCM is the missing piece: a tiny non-autoregressive policy that takes a **structured description of the world** (entities, relations, goal, candidate actions) and directly **scores complete typed actions** (one forward pass, no token generation), with an explicit "I don't know / I'm blocked" behavior (premature STOP → escalation).

```
VOICE ──► transcription ──► big model: UNDERSTAND + DECOMPOSE
                                 │ structured goal (interaction genres)
                                 ▼
                          UCM : EXECUTE — the small model (~1 ms/decision)
                                 │ adapts, recovers, escalates when stuck
                                 ▼
                          MEASURED RESULT ──► on failure: targeted re-decomposition
```

UCM is evaluated in two exact, procedurally generated worlds with closed action vocabularies and exact oracles:

| World | Description | Goals |
|---|---|---|
| **TinyGraphKey (TGK)** | rooms graph, one agent, a key, a parcel, one lockable door; `MOVE / PICK / DROP / UNLOCK / STOP` | `REACH(room)`, `HAVE(object)`, `AT(object, room)` |
| **SIW** (Synthetic Interaction World) | software-like views, buttons, fields, selects, options, forms, dialogs; `NAVIGATE / CLICK / TYPE / SELECT / STOP`; labels drawn from a closed vocabulary deliberately uncorrelated with roles | `VIEW`, `SET`, `CHOOSE`, `SUBMITTED` |

Both worlds are re-expressed as instances of a **verified DSL interpreter**, so one generic oracle/BFS and one generator pipeline serve both — with differential and mutation testing proving the DSL equals the native engines state-by-state.

## Headline results

### Established (pre-registered, sealed, independently audited)

| Claim | Result | Source |
|---|---|---|
| Closed-loop generalization on unseen TGK instances | **97.50 %** success, CI<sub>95</sub> [96.48 ; 98.50], 3 160 held-out episodes, 630 layout clusters, 5 seeds | `results/json/v0-gate2-confirmation.json` |
| Zero-shot compositional recombination | **99.35 %** success, CI [98.60 ; 99.88], 494 episodes / 102 layouts on a reserved cell (`AT(key, junction)`) | `results/json/v0-gate3-zeroshot-official.json` |
| Architecture election (relational GNN vs 1-hop Deep Sets) | **+62.2 pp** on layout-disjoint val stratum (88.1 % vs 25.9 %, CI [54.6 ; 68.9]); safe stratum never degraded; sustained cost ratio **1.921×** (≤2× gate) | `results/json/v0-gate5-val-stratum.json` |
| Interface controls collapse without goal / candidates structure / relations | **0.0** success for all three; counterfactual goal 0.960; permutation invariance 200/200 | `results/json/v0-m3-controls.json` |
| Cost and latency | **~1.04 ms** p95 per decision (sustained, interleaved CPU), 16–38× margin under the 20 ms envelope; 694 513 parameters | `docs/05-results-ledger.md` |
| Product-line demo (synthetic, simulated comprehension layer) | execution **44/44** and **32/32** on two held-out suites; p95 **1.146 ms**; **1120×** cheaper than a 1 200 ms/decision stub; refusal recall **40/40**, false refusals **0/40** | `results/json/s5-full-budget.json`, `results/json/s5-real-demo-simulated.json` |
| Verified DSL layer | differential equality over ≥10 000 states per world; 3 injected interpreter mutations detected | `docs/03-verified-dsl.md` |
| V1-bis instrument repair | the "goal reached without STOP" pathology is **eliminated (0 / 72 000 episodes)**, vs 29–44 % in V1 | `results/json/v1bis-stage-b-metrics.json` |

### Not established (honest ledger)

| Question | Verdict |
|---|---|
| Does TGK pretraining transfer ≥5 pp to SIW after supervised adaptation? | **Not demonstrated** in the tested regime (k = 64/128/256 episodes; SIW-small is ~97–99 % saturated, so the maximum possible effect was ≤ +0.73 pp at k = 256 — an instrument limitation, not evidence of absence) |
| Does a recurrent refinement block (T = 32) beat the one-pass model on deep plans? | **KILL** (−7.7 pp vs the frozen +15.8 pp threshold; conservative test, base frozen) |
| Does short in-context history improve decisions? | **Not testable by construction** in the executed design; what was measured is a *presence* distraction (−8.3 pp). Content reading not established |
| Does the auxiliary effect head double the family success (initial +34 pp)? | **Not replicated** by the controlled ablation (B−A = +5.6 pp, CI ∋ 0; a *simple* validity/termination target did better); verdict INDETERMINATE |
| Strict inter-signature compositional transfer | **0.0 %** on one held-out signature (exploratory, factors not disentangled); intra-band transfer GO (97.0 %) |
| Web DOM → policy compiler on 4 real pages | **5.86 % coverage** (forms: 100 % of their fields/buttons; denominator dominated by `<a>` links); switch rule triggered: the compiler needs its own vocabulary, tensorizer and DOM id de-duplication |

## The model in 30 seconds

- **Elected architecture: GNN-B144** — a 3-round relational message-passing network, hidden width 144, **694 513 parameters**, FP32, MLX.
- Nodes = entities (one-hot typed), edges = typed relations; a masked global mean pools a context vector; the goal is a typed predicate over entity references; a shared scorer scores **every candidate action** in one parallel pass.
- **Set-valued behavioural cloning loss**: `L = −log Σ_{a ∈ A*} softmax_a(candidates)` — only padding is masked; invalid-but-syntactically-listed actions remain in the denominator; STOP is an explicit action trained like any other.
- No labels, no raw ids, no text embeddings, no autoregression. The interface is the contract.
- Variants: **Deep Sets A** (1-hop, 366 337 params — failed GATE-1 at 96.58 % with a bit-exactly proven 1-hop tie), **B160/B192** (cost-gate failures, kept as a capacity/cost frontier), **SIW-adapted B144** (698 401 params), **CIV recurrent refine** (272 448 extra params shared across T iterations), **P2 model + effect head** (≈738 008 params).

## Repository map

```
paper/          The scientific paper (Markdown; PDF via tools/build_pdf.sh)
  PAPER.md      Main paper — abstract, methods, results, discussion, appendices
  figures/      Paper figures (SVG, generated by tools/make_figures.py)
docs/           Deep-dive documentation (English)
  01-architecture.md      Model architectures, tensor contract, loss
  02-environments.md      TGK and SIW normative specifications
  03-verified-dsl.md      The generic interpreter, differential + mutation testing
  04-methodology.md       Gates, pre-registration, sealed canons, attribution controls
  05-results-ledger.md    Every quantitative claim with artifact + world + model
  06-reproducibility.md   Instrument discipline: errata, incidents, atomic publication
  07-related-work.md      2026 positioning (System One Models, planning GNNs, preregistration)
  08-roadmap.md           What is next and what would falsify it
results/        Curated result artifacts (JSON) + claim mapping (RESULTS.md)
code/           Frozen source snapshot (ucm/, tests/, scripts/, configs/)
tools/          Figure/PDF build scripts
archive/        Original project documents (French) for provenance
```

## Reproduce

The paper repo is self-contained for reading and figure generation. Trained checkpoints and raw logs live in the working repository.

```bash
# Figures (from curated result JSONs)
python3 -m venv .venv && .venv/bin/pip install matplotlib
.venv/bin/python tools/make_figures.py     # writes paper/figures/*.svg

# Paper PDF (uses headless Chrome)
bash tools/build_pdf.sh                    # writes paper/PAPER.pdf

# Tests of the snapshot (requires mlx, numpy — see code/README.md)
cd code && python -m pytest -q             # 560 tests
```

## Snapshot provenance

- Working repository snapshot: commit **`a47f7b7c36daa87a33fb3aa12506799d27024187`** (2026-09-25 15:41), 423 commits, 2026-09-22 → 2026-09-25.
- Code: 35 205 lines of Python, 560 tests, MLX 0.32.2 / Python 3.12.13 / Apple M5 32 GB (CPU + GPU).
- Artifacts referenced: 451 JSON files (~1.7 GB) in the working repo; only curated, small result JSONs are mirrored here under `results/json/`.
- The `archive/` directory preserves the original French research documents (vision, spec, protocols, reports, errata, incidents, external reviews) unchanged, for provenance and auditability.

## Citation and license

Citation metadata: [`CITATION.cff`](CITATION.cff). This repository is private and pre-publication; licensing will be decided before any public release. Contact the author before redistributing.
