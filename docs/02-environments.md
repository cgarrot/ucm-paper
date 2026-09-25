# 02 — Environments

*Normative specifications of TinyGraphKey (TGK) and SIW. Source: `code/ucm/env/` (`tinygraph.py`, `siw.py`, `oracle.py`, `siw_oracle.py`), `archive/docs/SIW-SPEC.md`, `code/ucm/data/generate*.py`.*

Both worlds share one interface: `reset() / observe() / candidates() / execute(action) / goal_satisfied(state)`, deterministic dynamics, a finite horizon (64), and an exact oracle used only for supervision and evaluation.

```
        TGK                                   SIW
  rooms + door + key + parcel        views + widgets + forms + dialogs
  MOVE PICK DROP UNLOCK STOP         NAVIGATE CLICK TYPE SELECT STOP
  REACH / HAVE / AT                   VIEW / SET / CHOOSE / SUBMITTED
```

## 1. TinyGraphKey (relational world)

### 1.1 Objects and state

- connected undirected room graph, no multi-edges; `R = 4…8` at train (up to 64 supported);
- exactly one door edge with `locked ∈ {true,false}`;
- one agent, one key, one parcel; each object is in a room or carried (at most one);
- a static relation key↔door;
- no capacity, orientation, hidden metric distance, or randomness.

`PhysicalState = (agent_room, carried, key_room, parcel_room, door_locked)`; `state_hash` binds layout + state; `layout_hash` is a truncated SHA-256 of a canonical layout certificate.

### 1.2 Observation schema (policy input only)

```json
{"schema_version": "0.2",
 "entities":  [{"id": "...", "type": "room|agent|key|parcel|door", "attrs": {"locked": false}}],
 "relations": [{"subj": "...", "pred": "adjacent|connects|unlocks|at|held", "obj": "..."}],
 "goal":      {"predicate": "REACH|HAVE|AT", "args": {"room": "...", ...}},
 "candidates":[{"action": "...", "arg": "..."}]}
```

### 1.3 Dynamics (all candidates always enumerated)

| Action | Valid iff | Effect |
|---|---|---|
| `MOVE(r)` | `r` adjacent; door unlocked if the edge carries it | agent moves; carried object stays carried |
| `PICK(o)` | `o` in current room and hand empty | `o` carried |
| `DROP(o)` | `o` carried | `o` placed in current room |
| `UNLOCK(d)` | agent at a door endpoint, door locked, matching key carried | door unlocked permanently |
| `STOP` | always selectable | terminates; success iff goal true on native state |

Every decision costs one step, **including invalid actions (paid no-ops) and STOP**. Horizon 64; timeout is failure even if the goal was reached without STOP. `K = R + 6` candidates, goal-blind.

### 1.4 Goals

`REACH(room)` (agent in room), `HAVE(object)` (carried), `AT(object, room)` (**placed**, i.e. not carried — a held object has location `None ≠ room`). Goal types are balanced under split restrictions.

### 1.5 Oracle

`LayoutOracle(layout, goal)` enumerates all physical states (`2·(R² + 2R)`) and runs multi-source backward BFS from every goal state. Outputs `{d_star, L_star, optimal_actions, reachable}`; `d* = 0 ⇒ optimal = {STOP}`. The oracle is checked by exhaustive Bellman equivalence and a differential against the environment's successor.

## 2. SIW — Synthetic Interaction World

Specification of record: `archive/docs/SIW-SPEC.md` (French, normative for the V1 campaign). Summary:

### 2.1 Entities (closed registry)

| Entity | Attributes | Notes |
|---|---|---|
| `view` | — | 2–5 views, connected nav graph |
| `button` | `onclick_kind ∈ {submit, confirm, dismiss, none}` | `none` = distractor |
| `field` | `filled` | 2–4 per form |
| `select` | `chosen_option_ref` (nullable) | ≤1 per form, 2–4 options |
| `option` | — | belongs to exactly one select |
| `form` | `status ∈ {draft, complete, submitted}` | **derived**: complete ⟺ all fields filled ∧ select chosen |
| `dialog` | `open` bool | 0–1 initially open (30 % of layouts) |

### 2.2 Relations

`on_view`, `nav_edge`, `part_of`, `option_of`, `in_dialog`, `submits`, `current_view`, `filled`, `chosen`. Attributes are state; relations are structure.

### 2.3 Labels

Drawn from a closed ~60-word vocabulary with alias pairs, **uniformly at random independent of role** ("a submit button may be labelled cancel"). Binding is by reference only; no pretrained embeddings; isomorphic layouts may have completely different labels. A permutation test guarantees labels (and label–candidate mapping) carry no role signal.

### 2.4 Actions

| Action | Validity | Notes |
|---|---|---|
| `NAVIGATE(v)` | `v` adjacent to current view | navigating to the current view is invalid |
| `CLICK(b)` | `b` visible | see effect table |
| `TYPE(f)` | `f` visible and empty | unary, goal-blind (fix for `SUBMITTED` solvability) |
| `SELECT(o)` | visible, parent select unchosen | irreversible (the single reachability trap: 100 % of unreachable couples stem from `select_irreversible`) |
| `STOP` | always | terminal |

Visibility: `visible(w) = on_view(current_view) ∧ (¬in_dialog ∨ dialog.open)`. Candidates are the complete **syntactic, goal-blind and state-blind** enumeration; `K = V + B + F + O + 1`, target 30–60 enforced by adding bounded distractor widgets.

**Effect groups for CLICK:** `submit` → submitted (and closes open dialogs) *iff* the form is complete, otherwise **valid no-op**; `confirm`/`dismiss` are equivalent in V1 and close the dialog if open; `none` is always valid and never has an effect.

**Invalid vs valid no-op:** invalid = no-op + cost + `invalid` flag (non-adjacent navigate, invisible click/type/select, type on filled field, re-select); valid no-op = cost + `valid` flag, zero effect (distractors, premature submit, confirm without dialog). The model must learn both classes; rates are reported separately.

### 2.5 Goals

`VIEW(view)` (current view), `SET(field)` (filled), `CHOOSE(option)` (chosen), `SUBMITTED(form)` (submitted).

### 2.6 Oracle

Full FSM enumeration: `views × filled ⊆ fields × chosen × dialog × submitted ⊆ forms`; successor/predecessor graph cached per layout (LRU), backward multi-source BFS from goal states. Valid no-ops are self-loops that are never optimal. Same output contract as TGK; validated by exhaustive Bellman equivalence, differential against the environment, and targeted counterexamples (including door/chain edge cases in TGK).

## 3. Generators and canons

- TGK generation produced the **canon M0**: 13 758 transitions / 2 599 episodes, pools train 200 / val 50 / test 127+163+81, plus the additive G2 extension (102 layouts). All splits are layout-grouped (no layout crosses splits) and sealed with manifests.
- SIW generation uses new seeds per campaign; `SIW-small` is the V1/V1-bis profile (K bounded, one form), `standard` allows more widgets. V1-bis sealed test2 = 600 episodes (150 per predicate), seed 20261003, SHA published before the run. Generation seeds are never reused between DEV and sealed campaigns.
- All generation is deterministic from a seed and reproduces byte-identically across machines (verified on M5 / XMG / auditor replay).

## 4. What the two worlds are for

| Property | TGK | SIW |
|---|---|---|
| Relations with arguments | rooms, door, key | widgets, forms, dialogs |
| Resources / irreversible actions | key carried; door unlock | select irreversible; submit closes dialogs |
| Non-productive actions | invalid paid no-ops | **both** invalid and valid no-ops |
| Arbitrary labels | no | yes (role-independent) |
| Candidate count | R + 6 (≈10–14) | 30–60 |
| Transfer role | source world | target world |

The shared-interface bet: if one ~0.7 M policy core learns both, the interface — not the domain — is doing the work.
