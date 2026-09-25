# 01 — Architecture

*UCM model family: tensor contract, Deep Sets A, GNN B (elected B144), recurrent CIV, SIW adaptation, P2 model with effect head. Source: `code/ucm/model/`, `code/ucm/p2/model_p2.py`.*

## 1. The interface contract

Everything starts from one discipline: the policy sees **only** `policy_input`. Supervision (optimal actions) and provenance (hashes, ids) travel in separate channels and are proven incapable of changing tensors (`tests/test_model.py`, anti-leak tests). The contract:

```
observe(state)      -> policy_input   # entities, relations, goal, candidates
candidates()        -> goal-blind, state-blind syntactic enumeration
policy(x, goal, C)  -> score for every candidate (one parallel pass)
execute(action)     -> transition; invalid = paid no-op; STOP terminal
goal_satisfied(s)   -> independent evaluator, never a policy input
```

Forbidden in `policy_input`: `d_star`, reward, plan, next observation, timestep, episode id, or any field encoding distance to the goal. Candidate enumeration is **goal-blind** (the list does not shrink with the goal) and **state-blind** in SIW (all syntactically known widgets, visible or not).

## 2. Tensor contract (`code/ucm/model/tensorize.py`)

| Concept | Tensor | Shape / encoding |
|---|---|---|
| Entities | `nodes` | `[B, N, D_IN]` one-hot type + attribute blocks; `N_CAP = 64` |
| Relations | `edges`, `edge_types`, `edge_senses` | localized node indices; a relation becomes two directed messages (sense 0/1) |
| Incident slots | `slot_idx [B,N,S]`, `slot_mask` | cached gather tables over message slots; `MIN_S2CAP = 4` |
| Goal | `goal_pred`, `goal_refs [2]` | predicate one-hot + two reference slots (object, room; −1 unused) |
| Candidates | `cand_types`, `cand_args [B,K,2]` | action-type index + argument node indices; `K_CAP = 32` TGK / 96 SIW |
| Labels | `labels [B,K]` | 1 for optimal set `A*`, 0 otherwise (supervision channel only) |

Rules enforced by tests: raw ids are never embedded; padding is masked everywhere; order is randomized coherently (`randomize_obs_order`, `permute_example` permutes entities/relations/candidates and labels together); STOP is a candidate with null args.

`D_IN` per world: TGK **7** (`room, agent, key, parcel, door` + 2 door-state slots); SIW **21** (8 entity types + 4 onclick kinds + 2 field states + 2 select states + 3 form statuses + 2 dialog states); P2 **11** (TGK + 4 context kinds).

## 3. Model A — Deep Sets (1-hop), 366 337 parameters

```
node_enc   : MLP([D_IN+2, 128, 128]) + LayerNorm
edge_mlp   : MLP([3d, d, d]) over [type_emb, sense_emb, h_other]   # 1-hop only
update_mlp : MLP([2d, 2d, d]) + LayerNorm
pool       : masked mean -> g_ctx
goal_mlp   : MLP([n_pred+2d, d, d])
score_mlp  : MLP([n_atype+2d+2d, 2d, 1])   # shared candidate scorer
```

Purpose: a deliberately weak relational encoder. Documented failure: on a symmetric pair with identical 1-hop signatures the logits are **bit-exactly tied** (`0.410357 == 0.410357`), and GATE-1 overfit stalls at 96.58 % against a 99 % bar. This is what forces Model B.

## 4. Model B — relational GNN, 3 rounds

```
node_enc : MLP([D_IN+2, d, d]) + LayerNorm     # includes goal role tags
for each of 3 blocks (own type/sense embeddings):
    h = ln(h + block(h))                       # residual + norm per round
pool -> g_ctx ; goal_mlp ; score_mlp           # same heads as A
```

Width election (frozen before runs, 4 criteria):

| Model | Params | Depth stratum (val, layout-disjoint) | Cost (official interleaved) | Verdict |
|---|---:|---|---|---|
| A (1-hop) | 366 337 | 25.9 % | 1.00× | capacity fail |
| **B144** | **694 513** | **88.1 %** (+62.2 pp, CI [54.6 ; 68.9]) | **1.921×** | **elected 4/4** |
| B160 | 856 161 | 91.2 % | 2.035× (5/5 blocks >2) | cost fail |
| B192 | 1 230 145 | 89.5 % | 3.05 / 2.31 / ~2.15× | cost fail |

Safety stratum `d* ≤ 2`: A 84.1 % → B144 **100 %**. Tie separability passes for B, fails bit-exactly for A.

## 5. Training objective (set-valued BC)

```python
L = -logsumexp over A* of log_softmax(candidate logits)
# padding masked out; INVALID but enumerated actions stay in the denominator
```

Diagnostics: `optimal_action_rate`, `prob_mass_on_optimal`. Optimizer defaults: AdamW lr 3e-4, wd 1e-4, clip 1.0, FP32, batch 64, ≤10 epochs / 10 000 updates, best-validation checkpoint. `--overfit` diagnostic: 100 episodes, 5 000 updates, target ≥99 % OA.

## 6. SIW adaptation

`make_siw_model(d=144)` reshapes only vocabulary-dependent modules (`node_enc`, `goal_mlp`, `edge_type_emb`), keeps the scorer (5 actions = 5 actions) → **698 401 parameters**. Transfer loads only name-matching, shape-compatible tensors and publishes a loaded/skipped manifest.

## 7. CIV — recurrent refinement (S2b)

- One shared `MessageBlock` + `GRUCell` + LayerNorm, iterated `T = 32` times at decision time; **272 448 parameters, constant in T**; `n_iters` mechanically instrumented (`T_effective` proof).
- Wraps a frozen base model; base SHA verified before/after; heads reused.
- Verdict: KILL vs one-pass B144 on deep plans (§8.6 of the paper). Runtime: p95 14.08 ms model-only (PASS ≤20 ms), ~10–12× per-update wall, RSS 9.7 GB vs 1.4 GB.

## 8. P2 model — context entities and effect head

- `P2_ENTITY_TYPES = TGK + [ctx_move, ctx_pick, ctx_drop, ctx_unlock]`; `P2_REL_PREDS = TGK + [observed_effect]`; `D_IN_P2 = 11`.
- Context is not a raw history: observed effects become typed nodes with `observed_effect` edges to canonical targets; self-loops added on context nodes (a P2GNNB divergence: no incoming message ⇒ empty aggregate ⇒ NaN).
- `EffectHead = MLP([2d, d, 7])` over `[first candidate-arg rep ; global context]`; `loss = policy_BC + λ · effect_CE`, λ = 0.5.
- `head_enabled=False` ⇒ optimizer never sees head parameters (SHA-guarded) ⇒ **policy logits identical by construction**, which makes "the head is purely auxiliary" a mechanical statement, not an empirical one.
- Approximate size at d=144: 695 233 trunk + 42 775 head ≈ **738 008**.

## 9. Anti-leak and invariance guarantees (tested)

- Changing supervision or provenance bits cannot change the tensors (bit-identity test).
- Permuting ids/order yields identical distributions over outcomes (metamorphic tests; 200/200 episode permutations in M3).
- Labels are never embedded; roles are never inferred from labels (SIW labels are drawn independently of role).
- `D_IN` mismatches between tensorizer and model are regression-tested (a real collate bug was caught this way).

## 10. Checkpoints in the record

| Artifact | Model | Notes |
|---|---|---|
| `20260922-155847-gate2c-B144-s4` | B144 | GATE-2 seed 4 cell, 694 513 params, val OA 0.9985 |
| `gate3-zeroshot/gate3-official.json` | B144 ×5 | GATE-3 aggregate |
| `gate5-report/*` | A/B144/B160/B192 | election, promotion, cost protocols |
| `s5-demo/s5-full-ckpt-20260925T121754.npz` | SIW B144 | product-line executor |
| V0 `s5`–`s9` artifact dirs | **d=192, not 144** | never loaded by any run (runtime shape filter); incident closed without re-execution |

## 11. The web layer (compiler v2 + end-to-end bridge)

*Source: `code/ucm/web/` (`vocabulary.py`, `compiler_v2.py`, `make_snapshots.py`, `e2e_bridge.py`), `code/tests/test_web_compiler.py` (21 tests).*

```
HTML / live DOM ──> compiler_v2.parse_dom ──> entities + relations + candidates
                        │                        (strict web/2.0 schema)
                        ├─ id allocator: deterministic suffix _2, _3 … (renames traced)
                        ├─ href resolver: relative | site-root | absolute | scheme |
                        │                  anchor | same_page | other_scheme | unresolvable
                        └─ validate_policy_input (closed vocab; raises on violation)
                                   │
        e2e_bridge: shim link→button ──> SIW tensorizer ──> s5-full policy ──> candidate
                                   │                                      │
                        WebBridge snapshot (live DOM)          native tools click/fill/navigate
```

Design decisions:

- **Closed web vocabulary** `web/2.0`: entity types `view, form, field, button, select, option, link`; predicates `on_view, part_of, option_of, submits`; actions `TYPE, CLICK, SELECT, NAVIGATE, STOP` with strict argument typing (`TYPE→field`, `CLICK→button`, `SELECT→option`, `NAVIGATE→link`, `STOP→None`). The vocabulary is independent of the TGK/SIW vocabularies: the compiler owns its schema, the model consumer owns tensorization.
- **Actionability definition frozen from v1** (`compiler_probe.py`, commit `90a303a`): text-like inputs, textareas, buttons/submit, selects, `<a href>` — so v1↔v2 parity is mechanically verifiable (v1 replayed on the snapshots reproduces 512 actionable / 30 covered / 5.86 % exactly).
- **Deterministic de-duplication** of DOM ids (first occurrence keeps the bare id; later ones `_2`, `_3`; renames in stats).
- **Links as first-class entities** with resolved href kind and `NAVIGATE` candidates — the v1 coverage hole (476/512 actionables on the tutorial page).
- **Declared non-coverage**: JavaScript/SPA rendering, iframes/shadow DOM, disabled/hidden state, event handlers beyond `submits`, ARIA, and tensorization.
- **End-to-end shims are explicit and traced in artifacts**: `link → button` (closed SIW vocabulary), `SELECT` executed as a click on the `<option>`, `NAVIGATE` executed via the daemon's navigate tool; the voice layer is labelled transcripts (Muse external, not exercised) and Jev is a labelled stub.

The end-to-end run's result — a calibrated STOP/refusal on a real web form — is the abstention channel doing its job under distribution shift; the missing piece is an execution corpus of compiled pages (both training set and benchmark), not a better refusal.
