# 03 — The Verified DSL Layer

*UCM re-expresses both native engines as data programs for a generic interpreter, then proves equivalence by differential testing and mutation testing. Source: `code/ucm/dsl/` (`core.py`, `tgk_program.py`, `siw_program.py`, `bfs.py`, `differential.py`, `differential_siw.py`, `records_util.py`, `closure_evidence.py`).*

## 1. Why a DSL at all

The project needed one generator/oracle pipeline for two worlds, usable as the *source of truth* for held-back task families, auxiliary labels, and unreachable-goal construction. Rewriting two oracles in parallel invites divergence; a generic interpreter with the worlds as data, plus a proof obligation against the originals, gives one verified path.

```
   native TGK  ──┐                        ┌── differential ──> equality proof
                 ├──> DSL programs ──> interpreter ──> BFS oracle
   native SIW  ──┘        (data)       (reset-only)      │
                                                         └── records (supervision)
```

## 2. The interpreter (`core.py`)

Programs are JSON expression trees (`DSL_SCHEMA = "ucm-dsl/1.0"`). **No Python `eval`.**

```
expr   := ["const",v] | ["field",f] | ["param",p] | ["task",k…] | ["layout",k]
        | ["and",…] | ["or",…] | ["not",e] | ["eq",e1,e2] | ["is_none",e]
        | ["if",c,a,b] | ["mem1",table,e] | ["mem2",table,e1,e2] | ["smem",field,e]
        | ["tget",table,key] | ["dget",field,key] | ["d_has",field,key]
        | ["s_all_in",field,list] | ["d_all_has",field,list]
effect := ["set",field,expr] | ["s_add",field,expr] | ["s_rm",field,expr] | ["d_set",field,k,v]
```

Key design decisions:

- **Immutable canonical state.** Fields sorted; sets stored as sorted lists; canonical form is comparable across implementations.
- **Reset-only construction.** `reset(task)` is the only entry point; there is no state injection. This kills an entire class of "construct the state then compare" shortcuts and makes differential testing meaningful.
- **Guards and effect groups.** `_guard_ok` decides validity; `_apply` applies per-action `when` conditions and per-group effects. A false `when` is a **valid no-op** (cost paid, no change) — exactly the SIW semantics.
- **One successor function** shared by BFS and execution; `execute` handles STOP, timeout and goal check.
- **The interpreter knows no world.** TGK and SIW are two program instances (`tgk_program.py`, `siw_program.py`) plus layout adapters (`layout_from_native`, `layout_tables_from_siw`).

## 3. Generic oracle (`bfs.py`)

`DSLBFS.explore` computes the forward closure from the reset-built initial state while recording the transition graph (cap 200 000 states), then `_goal_dist` runs a reverse BFS from all goal states. `solve` returns `{reachable, d_star, optimal_actions, goal_dist, n_states}`; `d* = 0 ⇒ optimal = {STOP}`. A regression note in the module records the original bug — *forward depth is not goal distance* — and the fix.

## 4. Differential equivalence

For every state of a forward closure (states reconstructed **only by reset**), compare DSL vs native:

1. candidate list (as actions, not indices);
2. per-candidate validity;
3. canonical successor state;
4. goal truth.

Then per task: `d*` and the optimal-action set, again action-wise. The harness returns counters and `equal = not mismatches`, and coverage reporting is explicit about how many states were seen.

**Coverage in the record:** ≥10 000 states per world, sourced from `test_g1` canon layouts (TGK) and the DEV inventory (SIW).

## 5. Mutation testing

A differential that never fires is worthless. The closure-evidence harness injects three semantic bugs into the interpreter and requires ≥1 divergence per mutation:

| Mutation | Change | Detected on |
|---|---|---|
| `M1_negations_ignored` | `not` → True | TGK |
| `M2_s_add_noop` | set-add never modifies state | SIW (only `s_add` user) |
| `M3_eq_inverted` | `eq` → `!=` | TGK |

Evidence artifacts: `artifacts/dsl-evidence/dsl-closure-evidence-*.json` in the working repo (layout selection, aggregate match counts, mutation detection).

## 6. The centralized terminal invariant

`records_util.walk_optimal_plan` walks the optimal plan from reset and emits a record for **every** state, including the terminal `d* = 0` state whose optimal set is exactly `{STOP}`. `assert_terminal_stop_invariant` fails closed with a `RuntimeError` (never an `assert` statement, which `-O` strips) if the last record is not terminal. This invariant had already regressed once in the P2 records path; centralizing it means "the bug cannot come back" by construction. It is the reason the V1-bis STOP pathology was mechanically eliminated (0 / 72 000).

## 7. What the DSL is used for downstream

- **Task-family generation (P2):** held-back families are selected by optimal-plan *signature* — `(kind, goal predicate, d* bucket)` — computed through the DSL, not patched onto native data.
- **Unreachable-goal construction (S5):** impossible tasks are produced by DSL constraints (e.g. target widget inside a dialog that can never open) and verified unreachable by the oracle before use.
- **Auxiliary labels (P2):** effect classes for the effect head are derived from DSL transitions (`effect_label(kind, arg, valid)`), so labels and dynamics cannot drift apart.
- **Recovery data (S2b):** the recovery arm deviates one non-optimal action then re-solves with the same oracle — a DSL-level data synthesis knob.

## 8. Tests of record

| Test | Guarantee |
|---|---|
| `test_dsl_jalon1.py` | interpreter generic / reset-only / schema guard / STOP semantics; BFS `d*` and optimal sets vs native; forward-depth regression; TGK differential equality; injected mutation detected |
| `test_dsl_jalons23.py` | SIW differential equality; reset-only; click effect groups per kind; closure ≥10k; three mutations; M2 requires SIW; P2 pilot completeness |
| `closure_evidence.py` | reproducible evidence artifact for closure + mutations |

## 9. Honest limits

- The DSL covers the *enumerated* state space of the two toy worlds; it is not a general planning language.
- Mutation testing demonstrates detection of three specific bug classes, not semantic completeness of the interpreter.
- Differential equality is established on the covered closure (≥10k states), not on an unbounded state space (states are finite but the harness samples layouts).
