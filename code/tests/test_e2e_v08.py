"""E2E schema 0.8 v3 (lead 12:42 — previous 0784ff3 was still not full E2E).

FULL pipeline: writer → file → registry.read_once → **couples_lines_to_records**
(the actual adapter used by training) → validate_extracted(records_from_lines,
adapter_records, store) — a SHARED validator that both the main test and the
MUTANT tests call. Mutants corrupt the JSONL bytes, then the SAME validator
must raise AssertionError.

Checks on adapter-derived records (not manual resets):
  - optimal_semantic == oracle set-valued per record (via adapter records' policy_input+supervision)
  - d_star matches oracle AND is strictly decreasing within episode
  - terminal d*=0: {STOP:None} unique, rstar=None, last by depth as-read
  - rstar∈optimales, executing leads to next full state_key
  - depth sequential as-read (not sorted), episode groups contiguous
  - len(recs) == first['d_star'] + 1 (NOT tautological terminal count)

DEV only, no test2/seed 03/04. Band (2,4) prospective.
"""

import builtins
import hashlib
import json
import os

import pytest

from tests.test_confirm_v3 import _REPO, _LAYOUT_STORE


def _generate_v08(path, seed=42, n=4, band=(2, 4), n_layouts=6):
    from ucm.data.siw_pipeline import generate_adaptation_episodes_rstar
    from ucm.env.siw import SIWLayout
    from ucm.v1.data_adapter import _spec_of
    store = json.load(open(_LAYOUT_STORE))
    layouts = [SIWLayout(_spec_of(e)) for e in list(store.values())[:n_layouts]]
    eps, meta = generate_adaptation_episodes_rstar(layouts, seed=seed,
                                                     n_episodes=n, d0_band=band)
    with open(path, "w") as fh:
        for e in eps:
            fh.write(json.dumps(e, sort_keys=True, default=str) + "\n")
    return eps, meta


def _rebuild_init(state_key):
    view, filled, chosen, dialog, submitted = state_key
    return {"view": view, "chosen": dict(chosen), "dialog_open": dialog,
            "filled": list(filled), "submitted": list(submitted)}


def _to_action(semantic):
    if semantic is None:
        return None
    action, _, arg = semantic.partition(":")
    return (action, None if arg == "None" else arg)


def _norm_key(key_tuple):
    """SIWState.key() → comparable list matching JSON state_key format."""
    return [key_tuple[0], sorted(key_tuple[1]),
            [list(p) for p in key_tuple[2]], key_tuple[3], sorted(key_tuple[4])]


def _norm_json_key(state_key):
    return [state_key[0], sorted(state_key[1]),
            [list(p) if isinstance(p, (tuple, list)) else p for p in state_key[2]],
            state_key[3], sorted(state_key[4])]


def validate_e2e(lines, adapter_records, store):
    """SHARED validator: both main test and mutants call THIS function.
    lines = raw JSONL lines from registry; adapter_records = from
    couples_lines_to_records(lines, store). Raises AssertionError on any
    cross-check failure."""
    from ucm.env.siw import SIW, SIWLayout
    from ucm.env.siw_oracle import SIWOracle
    from ucm.v1.data_adapter import _spec_of

    raw = [json.loads(l) for l in lines if l.strip()]
    assert len(raw) == len(adapter_records), \
        f"adapter record count {len(adapter_records)} != raw {len(raw)}"

    # group by episode AS READ; assert CONTIGUITY (no interleaving)
    by_ep = {}
    _seen_eps = set()
    _current_ep = None
    for r in raw:
        if r["episode_id"] != _current_ep:
            assert r["episode_id"] not in _seen_eps, \
                f"episode {r['episode_id']} re-appears (non-contiguous)"
            _seen_eps.add(r["episode_id"])
            _current_ep = r["episode_id"]
        by_ep.setdefault(r["episode_id"], []).append(r)

    for ep_id, recs in by_ep.items():
        # depth sequential AS READ (not sorted)
        depths = [r["depth"] for r in recs]
        assert depths == list(range(len(recs))), \
            f"{ep_id}: depths {depths} not sequential 0..{len(recs)-1}"

        first = recs[0]
        lay = SIWLayout(_spec_of(store[first["layout_hash"]]))
        oracle = SIWOracle(lay, first["goal"])

        # episode length: d*(first) + 1 (NOT terminal count)
        assert len(recs) == first["d_star"] + 1, \
            f"{ep_id}: {len(recs)} records != d*({first['d_star']})+1"

        for i, rec in enumerate(recs):
            # find corresponding adapter record by zipping (same order)
            adapter_idx = sum(len(v) for k, v in by_ep.items()
                              if list(by_ep).index(k) < list(by_ep).index(ep_id)) + i
            adapter_rec = adapter_records[adapter_idx]

            # rebuild env
            env = SIW(lay)
            obs = env.reset({"init": _rebuild_init(rec["state_key"]),
                             "goal": first["goal"]})
            sol = oracle.solve(env.state)
            cands = env.candidates()

            # d_star matches oracle AND strictly decreasing
            assert rec["d_star"] == sol["d_star"], \
                f"{ep_id}[{i}]: d* {rec['d_star']} != oracle {sol['d_star']}"
            if i > 0:
                assert rec["d_star"] < recs[i-1]["d_star"], \
                    f"{ep_id}[{i}]: d* not decreasing"

            # optimal_semantic == oracle set-valued
            oracle_sem = {f"{c['action']}:{c.get('arg')}" for c in
                          (cands[j] for j in sol["optimal_actions"])}
            declared_sem = set(rec["optimal_semantic"])
            assert declared_sem == oracle_sem, \
                f"{ep_id}[{i}]: declared {declared_sem} != oracle {oracle_sem}"

            # adapter-derived supervision matches oracle (INDICES + d_star + policy_input)
            adapter_optimal = set(adapter_rec["supervision"]["optimal_actions"])
            oracle_indices = set(sol["optimal_actions"])
            assert adapter_optimal == oracle_indices, \
                f"{ep_id}[{i}]: adapter opt {adapter_optimal} != oracle idx {oracle_indices}"
            assert adapter_rec["supervision"]["d_star"] == sol["d_star"], \
                f"{ep_id}[{i}]: adapter d* {adapter_rec['supervision']['d_star']} != oracle {sol['d_star']}"
            # adapter policy_input == the FULL observed obs (all keys incl.
            # schema_version and any future additions) — lead 13:02
            _pi = adapter_rec["policy_input"]
            assert _pi == obs, \
                f"{ep_id}[{i}]: adapter policy_input != obs (keys diff: " \
                f"{set(_pi) ^ set(obs)})"
            # adapter supervision.reachable == oracle reachable
            assert adapter_rec["supervision"]["reachable"] == sol["reachable"], \
                f"{ep_id}[{i}]: adapter reachable != oracle"
            # provenance layout_hash preserved
            assert adapter_rec["provenance"]["layout_hash"] == rec["layout_hash"], \
                f"{ep_id}[{i}]: adapter provenance layout != raw"
            # zip identity: adapter state_goal_hash == recomputed from raw
            from ucm.v1.data_adapter import _state_of, _state_goal_hash
            raw_state = _state_of(rec["state_key"])
            expected_sgh = _state_goal_hash(rec["layout_hash"], raw_state, first["goal"])
            assert adapter_rec["provenance"]["state_goal_hash"] == expected_sgh, \
                f"{ep_id}[{i}]: adapter sgh {adapter_rec['provenance']['state_goal_hash']} != recomputed {expected_sgh}"

            if rec["d_star"] == 0:
                assert oracle_sem == {"STOP:None"}, \
                    f"{ep_id}[{i}]: d*=0 optimales {oracle_sem}"
                assert rec["rstar_executed_semantic"] is None
                assert rec["rstar_executed"] is None
                assert i == len(recs) - 1
            else:
                rstar_sem = rec["rstar_executed_semantic"]
                assert rstar_sem in declared_sem
                act = _to_action(rstar_sem)
                found = [j for j, c in enumerate(cands)
                         if (c["action"], c.get("arg")) == act]
                assert found, f"{ep_id}[{i}]: rstar {rstar_sem} not in cands"
                assert rec["rstar_executed"] in found
                env.execute(cands[rec["rstar_executed"]])
                if i + 1 < len(recs):
                    expected = _norm_json_key(recs[i+1]["state_key"])
                    actual = _norm_key(env.state.key())
                    assert actual == expected, \
                        f"{ep_id}[{i}]: after exec {actual} != next {expected}"


class TestE2EV08Full:
    def test_full_e2e_with_adapter(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _generate_v08("v08.jsonl", seed=42, n=4, band=(2, 4))
        store = json.load(open(_LAYOUT_STORE))

        from ucm.v1.sealed_reader import SealedOpenRegistry
        from ucm.v1.data_adapter import couples_lines_to_records
        reg = SealedOpenRegistry()
        fmt, lines, sha = reg.read_once("v08.jsonl")
        assert fmt == "couples"

        # THE ADAPTER: this is what training actually consumes
        adapter_records = couples_lines_to_records(lines, _LAYOUT_STORE)
        assert len(adapter_records) > 0

        # validate: shared validator on raw lines + adapter records
        validate_e2e(lines, adapter_records, store)

    def test_mutant_label_fails_same_validator(self, tmp_path, monkeypatch):
        """Corrupt optimal_semantic in the JSONL → same validate_e2e raises."""
        monkeypatch.chdir(tmp_path)
        _generate_v08("v08.jsonl", seed=42, n=4, band=(2, 4))
        lines = [l for l in open("v08.jsonl") if l.strip()]
        # find a d*>0 record and corrupt its optimal_semantic
        for i, l in enumerate(lines):
            r = json.loads(l)
            if r["d_star"] > 0 and r["optimal_semantic"]:
                r["optimal_semantic"] = ["STOP:None"] + r["optimal_semantic"][1:]
                lines[i] = json.dumps(r, sort_keys=True, default=str)
                break
        with open("v08-mut.jsonl", "w") as fh:
            fh.write("\n".join(lines) + "\n")
        store = json.load(open(_LAYOUT_STORE))
        from ucm.v1.data_adapter import couples_lines_to_records
        mut_lines = [l for l in open("v08-mut.jsonl") if l.strip()]
        adapter_recs = couples_lines_to_records(mut_lines, _LAYOUT_STORE)
        with pytest.raises(AssertionError, match="declared.*oracle"):
            validate_e2e(mut_lines, adapter_recs, store)

    def test_mutant_transition_fails_same_validator(self, tmp_path, monkeypatch):
        """TRUE transition mutant: swap rstar_executed to a DIFFERENT optimal
        action (same d*, same optimal set, VALID state — only the EXECUTED
        path changes). The adapter passes (d*/labels unchanged). The shared
        validator catches the 'after exec' state mismatch because the next
        record's state_key corresponds to the ORIGINAL action, not the swap."""
        monkeypatch.chdir(tmp_path)
        _generate_v08("v08.jsonl", seed=42, n=4, band=(2, 4))
        lines = [l for l in open("v08.jsonl") if l.strip()]
        raw = [json.loads(l) for l in lines]
        by_ep = {}
        for r in raw:
            by_ep.setdefault(r["episode_id"], []).append(r)

        # Find a mid-trajectory record with ≥2 optimales (swap candidates)
        target_ep, target_idx, swap_to = None, None, None
        for ep_id, recs in by_ep.items():
            if len(recs) < 3:
                continue
            first = recs[0]
            for i, r in enumerate(recs[:-1]):  # not terminal
                if r["d_star"] > 0 and len(r["optimal_semantic"]) >= 2:
                    # find the OTHER optimal ≠ the executed one
                    others = [s2 for s2 in r["optimal_semantic"]
                              if s2 != r["rstar_executed_semantic"]]
                    if others:
                        target_ep, target_idx = ep_id, i
                        swap_to = others[0]
                        break
            if target_ep:
                break
        if target_ep is None:
            pytest.fail("no multi-optimal mid-trajectory record — band (2,4) must produce one")

        # Mutate: swap rstar to a DIFFERENT optimal (d* + labels + state_key unchanged)
        for li, l in enumerate(lines):
            r = json.loads(l)
            if r["episode_id"] == target_ep and r["depth"] == target_idx:
                r["rstar_executed_semantic"] = swap_to
                # recompute the index for the swapped semantic
                from ucm.env.siw import SIW, SIWLayout
                from ucm.env.siw_oracle import SIWOracle
                from ucm.v1.data_adapter import _spec_of
                store = json.load(open(_LAYOUT_STORE))
                lay = SIWLayout(_spec_of(store[r["layout_hash"]]))
                env = SIW(lay)
                env.reset({"init": _rebuild_init(r["state_key"]), "goal": r["goal"]})
                cands = env.candidates()
                action, _, arg = swap_to.partition(":")
                arg = None if arg == "None" else arg
                for j, c in enumerate(cands):
                    if (c["action"], c.get("arg")) == (action, arg):
                        r["rstar_executed"] = j
                        break
                lines[li] = json.dumps(r, sort_keys=True, default=str)
                break
        with open("v08-mut2.jsonl", "w") as fh:
            fh.write("\n".join(lines) + "\n")
        store = json.load(open(_LAYOUT_STORE))
        from ucm.v1.data_adapter import couples_lines_to_records
        mut_lines = [l for l in open("v08-mut2.jsonl") if l.strip()]
        adapter_recs = couples_lines_to_records(mut_lines, _LAYOUT_STORE)
        # Adapter passes (d*, labels, state all valid). The shared validator
        # must catch the transition mismatch: after exec → different state
        # than the one recorded in the NEXT record's state_key.
        with pytest.raises(AssertionError, match="after exec"):
            validate_e2e(mut_lines, adapter_recs, store)

    def test_mutant_state_dstar_fails_same_validator(self, tmp_path, monkeypatch):
        """d* mutant: replace a mid-trajectory record's state_key with the
        FIRST record's (valid state, same layout/goal — adapter passes, but
        d* at that depth ≠ oracle d* → validator catches)."""
        monkeypatch.chdir(tmp_path)
        _generate_v08("v08.jsonl", seed=42, n=4, band=(2, 4))
        lines = [l for l in open("v08.jsonl") if l.strip()]
        raw = [json.loads(l) for l in lines]
        by_ep = {}
        for r in raw:
            by_ep.setdefault(r["episode_id"], []).append(r)
        target = None
        for ep_id, recs in by_ep.items():
            if len(recs) >= 3:
                target = recs[1]
                break
        if target is None:
            pytest.fail("no multi-step trajectory")
        for i, l in enumerate(lines):
            r = json.loads(l)
            if r["episode_id"] == target["episode_id"] and r["depth"] == target["depth"]:
                r["state_key"] = by_ep[target["episode_id"]][0]["state_key"]
                lines[i] = json.dumps(r, sort_keys=True, default=str)
                break
        with open("v08-mut3.jsonl", "w") as fh:
            fh.write("\n".join(lines) + "\n")
        store = json.load(open(_LAYOUT_STORE))
        from ucm.v1.data_adapter import couples_lines_to_records
        mut_lines = [l for l in open("v08-mut3.jsonl") if l.strip()]
        adapter_recs = couples_lines_to_records(mut_lines, _LAYOUT_STORE)
        with pytest.raises(AssertionError):  # either d* or after-exec — both prove detection
            validate_e2e(mut_lines, adapter_recs, store)

    def test_hash_mismatch_after_exactly_one_open(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _generate_v08("v08.jsonl", seed=42, n=4, band=(2, 4))
        from ucm.v1.sealed_reader import SealedOpenRegistry
        reg = SealedOpenRegistry()
        opens = []
        real_open = builtins.open
        def spy(f, *a, **k):
            opens.append(str(f)); return real_open(f, *a, **k)
        monkeypatch.setattr(builtins, "open", spy)
        try:
            with pytest.raises(RuntimeError, match="MISMATCH"):
                reg.read_once("v08.jsonl", expected_hash="0" * 64)
        finally:
            monkeypatch.setattr(builtins, "open", real_open)
        assert len([o for o in opens if "v08.jsonl" in o]) == 1
