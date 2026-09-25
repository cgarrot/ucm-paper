"""Tests for episode-level budget selection (V1-bis opt-in, DEV only).

Covers all 13:05 audit points:
  - complete episodes: d*→0 terminal, depths sequential, d* strictly decreasing,
    len == d*+1 — truncated episodes → ValueError
  - prefix hashes match independent recomputation
  - materialize: adapter receives ALL records of k episodes (NO [:K_STAR])
  - finetune spy: opt-in path passes exactly N-episode records to finetune
  - N > 500 records: proves no [:500] slicing
  - interleaved → ValueError
  - non-canonical file (CRLF/blank/no-final-newline) → hash mismatch detected
  - multi-k without re-opening the data file
"""

import builtins
import hashlib
import json
import os

import pytest

from tests.test_confirm_v3 import _REPO, _LAYOUT_STORE
from tests.test_e2e_v08 import _generate_v08


class TestEpisodeCompleteness:
    def _mk_lines(self, *records):
        return [json.dumps(r) for r in records]

    def test_truncated_episode_fails(self):
        """d* never reaches 0 → truncated → ValueError."""
        from ucm.v1.episode_budget import select_episodes
        lines = self._mk_lines(
            {"episode_id": "A", "depth": 0, "d_star": 3, "state_key": ["v0", [], [], False, []]},
            {"episode_id": "A", "depth": 1, "d_star": 2, "state_key": ["v1", [], [], False, []]},
            # d* goes 3→2→1 but never reaches 0 (STOP missing)
            {"episode_id": "A", "depth": 2, "d_star": 1, "state_key": ["v2", [], [], False, []]},
        )
        with pytest.raises(ValueError, match="truncated"):
            select_episodes(lines, 1)

    def test_non_sequential_depths_fails(self):
        from ucm.v1.episode_budget import select_episodes
        lines = self._mk_lines(
            {"episode_id": "A", "depth": 0, "d_star": 1},
            {"episode_id": "A", "depth": 2, "d_star": 0},  # skips depth 1!
        )
        with pytest.raises(ValueError, match="not sequential"):
            select_episodes(lines, 1)

    def test_d_star_not_decreasing_fails(self):
        from ucm.v1.episode_budget import select_episodes
        lines = self._mk_lines(
            {"episode_id": "A", "depth": 0, "d_star": 2},
            {"episode_id": "A", "depth": 1, "d_star": 2},  # not decreasing
            {"episode_id": "A", "depth": 2, "d_star": 0},
        )
        with pytest.raises(ValueError, match="not strictly decreasing"):
            select_episodes(lines, 1)

    def test_wrong_length_fails(self):
        from ucm.v1.episode_budget import select_episodes
        lines = self._mk_lines(
            {"episode_id": "A", "depth": 0, "d_star": 3},
            {"episode_id": "A", "depth": 1, "d_star": 2},
            {"episode_id": "A", "depth": 2, "d_star": 1},
            {"episode_id": "A", "depth": 3, "d_star": 0},
            # d*+1 = 4 records, but we add a 5th phantom
            {"episode_id": "A", "depth": 4, "d_star": 0},  # d*=0 at depth 4 with first d*=3 → 5≠4
        )
        # Actually: first d*=3, so len should be 4, but we have 5 records
        # d* goes 3→2→1→0→0 (0 not strictly decreasing from 0)
        with pytest.raises(ValueError):
            select_episodes(lines, 1)


class TestSelectEpisodes:
    def test_select_k_episodes_all_records(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _generate_v08("v08.jsonl", seed=42, n=4, band=(2, 4))
        lines = [l for l in open("v08.jsonl") if l.strip()]
        from ucm.v1.episode_budget import select_episodes

        sel, meta = select_episodes(lines, 2)
        assert meta["n_episodes"] == 2
        groups = {}
        current = None
        for l in lines:
            r = json.loads(l)
            if r["episode_id"] != current:
                groups[r["episode_id"]] = []
                current = r["episode_id"]
            groups[r["episode_id"]].append(l)
        ep_ids = list(groups.keys())
        expected = sum(len(groups[e]) for e in ep_ids[:2])
        assert meta["n_records"] == expected == len(sel)

    def test_prefix_hashes_match_recomputation(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _generate_v08("v08.jsonl", seed=42, n=4, band=(2, 4))
        lines = [l for l in open("v08.jsonl") if l.strip()]
        from ucm.v1.episode_budget import select_episodes
        _, meta = select_episodes(lines, 3)
        groups = {}
        current = None
        for l in lines:
            r = json.loads(l)
            if r["episode_id"] != current:
                groups[r["episode_id"]] = []
                current = r["episode_id"]
            groups[r["episode_id"]].append(l)
        ep_ids = list(groups.keys())
        for k in (1, 2, 3):
            running = []
            for e in ep_ids[:k]:
                running.extend(groups[e])
            canon = "\n".join(r.rstrip("\n") for r in running) + "\n"
            expected = hashlib.sha256(canon.encode()).hexdigest()
            assert meta["prefix_sha256"][k] == expected

    def test_interleaved_fails_closed(self):
        from ucm.v1.episode_budget import select_episodes
        lines = [
            json.dumps({"episode_id": "A", "depth": 0, "d_star": 1}),
            json.dumps({"episode_id": "B", "depth": 0, "d_star": 0}),
            json.dumps({"episode_id": "A", "depth": 1, "d_star": 0}),  # re-appears
        ]
        with pytest.raises(ValueError, match="non-contiguous"):
            select_episodes([l for l in lines], 2)


class TestFinetuneSpy:
    def test_finetune_receives_exact_n_episode_records(self, tmp_path, monkeypatch):
        """Spy on finetune: the opt-in path passes EXACTLY the records from
        N complete episodes (verified by count and metadata)."""
        monkeypatch.chdir(tmp_path)
        _generate_v08("v08.jsonl", seed=42, n=4, band=(2, 4))
        lines = [l for l in open("v08.jsonl") if l.strip()]
        from ucm.v1.episode_budget import materialize_episodes
        from ucm.v1.runner import finetune

        received = []
        real_finetune = finetune
        def finetune_spy(model, records, cfg, cov, arm, k):
            received.append(len(records))
            return real_finetune(model, records, cfg, cov, arm, k)
        # patch in runner module namespace (where it's called)
        import ucm.v1.runner as _runner_mod
        monkeypatch.setattr(_runner_mod, "finetune", finetune_spy)

        records, meta = materialize_episodes(lines, 2, _LAYOUT_STORE)
        # simulate the opt-in call: finetune(model, records, ...)
        from ucm.v1.transfer import FinetuneConfig, CoverageTracker
        import mlx.core as mx
        mx.set_default_device(mx.cpu)
        from ucm.model.siw_model import make_siw_model
        m = make_siw_model()
        cfg = FinetuneConfig(updates=1)
        cov = CoverageTracker()
        _runner_mod.finetune(m, records, cfg, cov, "test", 2)
        assert received == [meta["n_materialized"]]
        # n_materialized > 0 and == sum of 2 episodes' records
        groups = {}
        current = None
        for l in lines:
            r = json.loads(l)
            if r["episode_id"] != current:
                groups[r["episode_id"]] = []
                current = r["episode_id"]
            groups[r["episode_id"]].append(l)
        ep_ids = list(groups.keys())
        expected = sum(len(groups[e]) for e in ep_ids[:2])
        assert received[0] == expected


class TestNonCanonicalFile:
    def test_crlf_detected(self, tmp_path):
        """CRLF line endings produce different bytes → hash mismatch detected."""
        from ucm.v1.episode_budget import verify_full_file_sha
        lines_lf = ['{"a":1}\n', '{"a":2}\n']
        lines_crlf = ['{"a":1}\r\n', '{"a":2}\r\n']
        # SHA of the actual CRLF file
        crlf_bytes = "".join(lines_crlf).encode()
        crlf_sha = hashlib.sha256(crlf_bytes).hexdigest()
        # verify against LF reconstruction → mismatch
        assert not verify_full_file_sha(crlf_sha, lines_crlf)

    def test_blank_lines_detected(self, tmp_path):
        from ucm.v1.episode_budget import verify_full_file_sha
        lines_with_blank = ['{"a":1}\n', '\n', '{"a":2}\n']
        raw = "".join(lines_with_blank).encode()
        raw_sha = hashlib.sha256(raw).hexdigest()
        # blank line stripped in canonical → different hash
        assert not verify_full_file_sha(raw_sha, lines_with_blank)

    def test_no_final_newline_detected(self, tmp_path):
        from ucm.v1.episode_budget import verify_full_file_sha
        lines_no_nl = ['{"a":1}\n', '{"a":2}']  # no trailing \n
        raw = "".join(lines_no_nl).encode()
        raw_sha = hashlib.sha256(raw).hexdigest()
        # canonical adds \n → different hash
        assert not verify_full_file_sha(raw_sha, lines_no_nl)


class TestMultiKNoReopen:
    def test_multi_k_no_data_file_reopen(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _generate_v08("v08.jsonl", seed=42, n=4, band=(2, 4))
        lines = [l for l in open("v08.jsonl") if l.strip()]
        from ucm.v1.episode_budget import materialize_episodes

        opens = []
        real_open = builtins.open
        def spy(f, *a, **k):
            opens.append(str(f))
            return real_open(f, *a, **k)
        monkeypatch.setattr(builtins, "open", spy)
        try:
            for k in (1, 2, 3):
                recs, meta = materialize_episodes(lines, k, _LAYOUT_STORE)
        finally:
            monkeypatch.setattr(builtins, "open", real_open)
        # v08.jsonl NOT re-opened (layout_store opens are expected)
        v08_opens = [o for o in opens if "v08.jsonl" in o]
        assert len(v08_opens) == 0


class TestFromReadOnce:
    def test_valid_canonical_passes(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _generate_v08("v08.jsonl", seed=42, n=4, band=(2, 4))
        from ucm.v1.sealed_reader import SealedOpenRegistry
        from ucm.v1.episode_budget import from_read_once
        reg = SealedOpenRegistry()
        fmt, lines, sha = reg.read_once("v08.jsonl")
        verified = from_read_once(lines, sha, sha)
        assert verified == lines

    def test_wrong_expected_sha_fails_before_adapter(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _generate_v08("v08.jsonl", seed=42, n=4, band=(2, 4))
        from ucm.v1.sealed_reader import SealedOpenRegistry
        from ucm.v1.episode_budget import from_read_once
        reg = SealedOpenRegistry()
        fmt, lines, sha = reg.read_once("v08.jsonl")
        with pytest.raises(ValueError, match="SHA"):
            from_read_once(lines, sha, "0" * 64)

    def test_crlf_fails_before_adapter(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _generate_v08("v08.jsonl", seed=42, n=4, band=(2, 4))
        # re-write with CRLF
        raw = open("v08.jsonl", "rb").read().replace(b"\n", b"\r\n")
        open("v08-crlf.jsonl", "wb").write(raw)
        from ucm.v1.sealed_reader import SealedOpenRegistry
        from ucm.v1.episode_budget import from_read_once
        reg = SealedOpenRegistry()
        fmt, lines, sha = reg.read_once("v08-crlf.jsonl")
        with pytest.raises(ValueError, match="non-canonical"):
            from_read_once(lines, sha, sha)


def sha_of(lines):
    stripped = [l.rstrip("\r\n") for l in lines if l.strip()]
    canon = ("\n".join(stripped) + "\n") if stripped else ""
    return hashlib.sha256(canon.encode()).hexdigest()


class TestFinetuneEpisodeBudget:
    def test_production_entry_point_calls_finetune_no_slicing(self, tmp_path, monkeypatch):
        """Spy on finetune: finetune_episode_budget (the PRODUCTION opt-in)
        calls finetune with EXACTLY k episodes' records — no slicing."""
        monkeypatch.chdir(tmp_path)
        _generate_v08("v08.jsonl", seed=42, n=4, band=(2, 4))
        lines = [l for l in open("v08.jsonl") if l.strip()]

        from ucm.v1 import runner as _runner_mod
        received = []
        real_finetune = _runner_mod.finetune
        def spy(model, records, cfg, cov, arm, k):
            received.append(len(records))
            return {"updates": 0, "wall_s": 0.0}
        monkeypatch.setattr(_runner_mod, "finetune", spy)
        # production code calls _runner.finetune via _runner module reference

        from ucm.v1.episode_budget import finetune_episode_budget, select_episodes
        import mlx.core as mx
        mx.set_default_device(mx.cpu)
        from ucm.model.siw_model import make_siw_model
        from ucm.v1.transfer import FinetuneConfig, CoverageTracker
        m = make_siw_model()
        cfg = FinetuneConfig(updates=0)
        cov = CoverageTracker()
        result = finetune_episode_budget(m, sha_of(lines), sha_of(lines), lines, 2, cfg, cov, "test", _LAYOUT_STORE)
        assert len(received) == 1
        _, meta = select_episodes(lines, 2)
        assert received[0] == meta["n_records"]
        assert "episode_budget_metadata" in result

    def test_gt500_records_no_slicing(self, tmp_path, monkeypatch):
        """Clone episodes (unique IDs) to exceed 500 records, verify the
        production entry point passes ALL of them (no [:500])."""
        monkeypatch.chdir(tmp_path)
        _generate_v08("v08.jsonl", seed=42, n=4, band=(2, 4))
        lines = [l for l in open("v08.jsonl") if l.strip()]
        # clone episodes with unique IDs until > 500 records
        raw = [json.loads(l) for l in lines]
        # group by episode
        groups = {}
        for r in raw:
            groups.setdefault(r["episode_id"], []).append(r)
        ep_list = list(groups.values())
        # replicate episodes with new IDs until total > 500
        clones = []
        idx = 0
        while sum(len(g) for g in clones) + len(raw) <= 800:
            src = ep_list[idx % len(ep_list)]
            clone_ep = f"clone-{idx}"
            episode_clone = []
            for r in src:
                c = dict(r)
                c["episode_id"] = clone_ep
                episode_clone.append(c)
            clones.append(episode_clone)  # one LIST per episode
            idx += 1
        # flatten clones into a single list of records
        clone_records = [r for ep in clones for r in ep]
        all_raw = raw + clone_records
        all_lines = [json.dumps(r, sort_keys=True, default=str) for r in all_raw]
        # verify total > 500
        assert len(all_lines) > 500, f"clone expansion insufficient: {len(all_lines)}"
        total = len(all_lines)

        from ucm.v1 import runner as _runner_mod
        received = []
        def spy(model, records, cfg, cov, arm, k):
            received.append(len(records))
            return {"updates": 0}
        # patch runner module (the production reference)
        from ucm.v1 import runner as _runner_mod
        monkeypatch.setattr(_runner_mod, "finetune", spy)

        from ucm.v1.episode_budget import finetune_episode_budget
        import mlx.core as mx
        mx.set_default_device(mx.cpu)
        from ucm.model.siw_model import make_siw_model
        from ucm.v1.transfer import FinetuneConfig, CoverageTracker
        m = make_siw_model()
        cfg = FinetuneConfig(updates=0)
        cov = CoverageTracker()
        # select ALL episodes (should be >500 records total)
        n_eps = len({r["episode_id"] for r in all_raw})
        finetune_episode_budget(m, sha_of(all_lines), sha_of(all_lines), all_lines, n_eps, cfg, cov, "test", _LAYOUT_STORE)
        assert received[0] > 500, f"expected >500 records, got {received[0]}"
        assert received[0] == total, f"finetune received {received[0]} != {total}"


class TestEmptyFileSha:
    def test_empty_lines_sha_of_empty_string(self):
        from ucm.v1.episode_budget import verify_full_file_sha
        assert verify_full_file_sha(hashlib.sha256(b"").hexdigest(), [])


class TestTerminalStopSemantics:
    def test_terminal_stop_unique_and_none(self, tmp_path, monkeypatch):
        """Terminal record: optimal_semantic == {STOP:None} EXACTLY,
        rstar_executed_semantic is None, rstar_executed is None."""
        monkeypatch.chdir(tmp_path)
        _generate_v08("v08.jsonl", seed=42, n=4, band=(2, 4))
        lines = [l for l in open("v08.jsonl") if l.strip()]
        from ucm.v1.episode_budget import _episode_groups
        groups = _episode_groups(lines)
        for ep, recs in groups.items():
            parsed = [json.loads(l) for l in recs]
            terminal = parsed[-1]
            assert terminal["d_star"] == 0
            assert set(terminal["optimal_semantic"]) == {"STOP:None"}, \
                f"{ep}: terminal opt_sem {terminal['optimal_semantic']}"
            assert terminal["rstar_executed_semantic"] is None
            assert terminal["rstar_executed"] is None


class TestHashRequired:
    def test_wrong_sha_refused_before_spy(self, tmp_path, monkeypatch):
        """Production entry point with WRONG expected_sha → ValueError BEFORE
        finetune is called (spy proves finetune never runs)."""
        monkeypatch.chdir(tmp_path)
        _generate_v08("v08.jsonl", seed=42, n=4, band=(2, 4))
        lines = [l for l in open("v08.jsonl") if l.strip()]
        from ucm.v1 import runner as _runner_mod
        received = []
        def spy(model, records, cfg, cov, arm, k):
            received.append(len(records))
            return {"updates": 0}
        monkeypatch.setattr(_runner_mod, "finetune", spy)
        from ucm.v1.episode_budget import finetune_episode_budget
        import mlx.core as mx
        mx.set_default_device(mx.cpu)
        from ucm.model.siw_model import make_siw_model
        from ucm.v1.transfer import FinetuneConfig, CoverageTracker
        m = make_siw_model()
        cfg = FinetuneConfig(updates=0)
        cov = CoverageTracker()
        with pytest.raises(ValueError, match="SHA"):
            finetune_episode_budget(m, sha_of(lines), "0" * 64, lines, 2, cfg, cov, "t", _LAYOUT_STORE)
        assert received == [], f"finetune called despite wrong SHA: {received}"

    def test_crlf_refused_before_spy(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _generate_v08("v08.jsonl", seed=42, n=4, band=(2, 4))
        raw = open("v08.jsonl", "rb").read().replace(b"\n", b"\r\n")
        open("v08-crlf.jsonl", "wb").write(raw)
        lines_crlf = [l for l in open("v08-crlf.jsonl") if l.strip()]
        from ucm.v1 import runner as _runner_mod
        received = []
        def spy(model, records, cfg, cov, arm, k):
            received.append(len(records))
            return {"updates": 0}
        monkeypatch.setattr(_runner_mod, "finetune", spy)
        from ucm.v1.episode_budget import finetune_episode_budget
        import mlx.core as mx
        mx.set_default_device(mx.cpu)
        from ucm.model.siw_model import make_siw_model
        from ucm.v1.transfer import FinetuneConfig, CoverageTracker
        m = make_siw_model()
        cfg = FinetuneConfig(updates=0)
        cov = CoverageTracker()
        # reader_sha = sha of CRLF bytes (what read_once would compute)
        crlf_sha = hashlib.sha256(raw).hexdigest()
        with pytest.raises(ValueError, match="non-canonical"):
            finetune_episode_budget(m, crlf_sha, crlf_sha, lines_crlf, 2, cfg, cov, "t", _LAYOUT_STORE)
        assert received == []

    def test_from_read_once_chain_to_budget(self, tmp_path, monkeypatch):
        """Full chain: read_once → from_read_once (verify) →
        finetune_episode_budget (with the SAME hashes) → spy sees k episodes."""
        monkeypatch.chdir(tmp_path)
        _generate_v08("v08.jsonl", seed=42, n=4, band=(2, 4))
        from ucm.v1.sealed_reader import SealedOpenRegistry
        from ucm.v1.episode_budget import from_read_once, finetune_episode_budget
        reg = SealedOpenRegistry()
        fmt, lines, sha = reg.read_once("v08.jsonl")
        verified = from_read_once(lines, sha, sha)
        assert verified is not None

        from ucm.v1 import runner as _runner_mod
        received = []
        def spy(model, records, cfg, cov, arm, k):
            received.append(len(records))
            return {"updates": 0}
        monkeypatch.setattr(_runner_mod, "finetune", spy)
        import mlx.core as mx
        mx.set_default_device(mx.cpu)
        from ucm.model.siw_model import make_siw_model
        from ucm.v1.transfer import FinetuneConfig, CoverageTracker
        m = make_siw_model()
        cfg = FinetuneConfig(updates=0)
        cov = CoverageTracker()
        result = finetune_episode_budget(m, sha, sha, verified, 2, cfg, cov, "t", _LAYOUT_STORE)
        assert len(received) == 1 and received[0] > 0
        assert "episode_budget_metadata" in result


class TestPrefixHashesMatchFileBytes:  # R2, lead 13:18
    def test_prefix_hashes_match_actual_file_bytes(self, tmp_path, monkeypatch):
        """For each k, prefix_hashes(lines, [k])[k] must equal sha256 of the
        ACTUAL bytes written to the file for the first k episodes."""
        monkeypatch.chdir(tmp_path)
        _generate_v08("v08.jsonl", seed=42, n=4, band=(2, 4))
        from ucm.v1.episode_budget import prefix_hashes

        # Read as text lines (same as read_once would produce)
        lines = [l.rstrip("\n") for l in open("v08.jsonl") if l.strip()]
        mem_hashes = prefix_hashes(lines, [1, 2, 3])

        # Compute from ACTUAL file bytes: group raw lines by episode
        raw = open("v08.jsonl", "rb").read()
        raw_lines = [l for l in raw.split(b"\n") if l.strip()]
        groups = {}
        current = None
        for l in raw_lines:
            r = json.loads(l)
            if r["episode_id"] != current:
                groups[r["episode_id"]] = []
                current = r["episode_id"]
            groups[r["episode_id"]].append(l)
        ep_ids = list(groups.keys())
        assert len(ep_ids) >= 3
        for k in (1, 2, 3):
            prefix_bytes = b"\n".join(
                l for ep in ep_ids[:k] for l in groups[ep]) + b"\n"
            file_sha = hashlib.sha256(prefix_bytes).hexdigest()
            assert mem_hashes[k] == file_sha, \
                f"k={k}: mem {mem_hashes[k][:16]}… != file {file_sha[:16]}…"

    def test_interleaved_prefix_fails(self):
        from ucm.v1.episode_budget import prefix_hashes
        lines = [
            json.dumps({"episode_id": "A", "depth": 0, "d_star": 1}),
            json.dumps({"episode_id": "B", "depth": 0, "d_star": 0}),
            json.dumps({"episode_id": "A", "depth": 1, "d_star": 0}),
        ]
        with pytest.raises(ValueError, match="non-contiguous"):
            prefix_hashes([l for l in lines], [2])


class TestWriterRRPrefixHashes:  # R2 writer, lead 13:22
    def test_prefix_hashes_match_writer_rr_bytes(self, tmp_path, monkeypatch):
        """Full chain: generate → group by episode → write_rr_couples_file
        (physical RR write) → read_once (registry: 1 FD) → prefix_hashes ==
        SHA of the actual writer bytes for every prefix k and full file.
        NOTE: the bytes-oracle re-opens rr.jsonl (2nd open) — DEV fixture
        only, NOT a sealed one-read test."""
        monkeypatch.chdir(tmp_path)
        from ucm.data.siw_pipeline import generate_adaptation_episodes_rstar
        from ucm.data.rstar_order import write_rr_couples_file
        from ucm.env.siw import SIWLayout
        from ucm.v1.data_adapter import _spec_of
        from ucm.v1.sealed_reader import SealedOpenRegistry
        from ucm.v1.episode_budget import prefix_hashes

        # 1) Generate DEV data
        store = json.load(open(_LAYOUT_STORE))
        layouts = [SIWLayout(_spec_of(e)) for e in list(store.values())[:6]]
        eps, meta = generate_adaptation_episodes_rstar(layouts, seed=42,
                                                        n_episodes=4, d0_band=(2, 4))
        # group by episode (as the writer expects)
        by_ep = {}
        for e in eps:
            by_ep.setdefault(e["episode_id"], []).append(e)

        # 2) WRITE via the physical RR writer
        wmeta = write_rr_couples_file(eps, by_ep, "rr.jsonl")
        writer_sha = wmeta["sha256_write_stream"]

        # 3) READ via registry (1 FD)
        reg = SealedOpenRegistry()
        fmt, lines, reader_sha = reg.read_once("rr.jsonl")
        assert reader_sha == writer_sha, \
            f"reader {reader_sha[:16]}… != writer {writer_sha[:16]}…"

        # 4) Compute prefix hashes from memory
        n_eps = len(by_ep)
        ks = list(range(1, n_eps + 1))
        mem_hashes = prefix_hashes(lines, ks)

        # 5) Verify against ACTUAL FILE BYTES (grouped by episode from file)
        raw_lines = [l for l in open("rr.jsonl", "rb").read().split(b"\n") if l.strip()]
        file_groups = {}
        current = None
        for l in raw_lines:
            r = json.loads(l)
            if r["episode_id"] != current:
                file_groups[r["episode_id"]] = []
                current = r["episode_id"]
            file_groups[r["episode_id"]].append(l)
        file_ep_ids = list(file_groups.keys())
        assert set(file_ep_ids) == set(by_ep.keys()), "episode set mismatch"

        for k in ks:
            prefix_bytes = b"\n".join(
                l for ep in file_ep_ids[:k] for l in file_groups[ep]) + b"\n"
            file_sha = hashlib.sha256(prefix_bytes).hexdigest()
            assert mem_hashes[k] == file_sha, \
                f"k={k}: mem {mem_hashes[k][:16]}… != writer file {file_sha[:16]}…"

        # 6) Full file hash matches
        full_mem = prefix_hashes(lines, [n_eps])[n_eps]
        assert full_mem == writer_sha, \
            f"full: mem {full_mem[:16]}… != writer {writer_sha[:16]}…"

        # 7) RR ORDER: episodes interleaved by predicate (writer's design)
        # verify at least 2 different predicates appear in the first few eps
        first_preds = set()
        for ep_id in file_ep_ids[:4]:
            recs = file_groups[ep_id]
            if recs:
                first_preds.add(json.loads(recs[0])["goal"]["predicate"])
        assert len(first_preds) >= 2, \
            f"RR order: only {len(first_preds)} predicates in first 4 episodes"

    def test_writer_interleaved_fails(self):
        from ucm.v1.episode_budget import prefix_hashes
        lines = [
            json.dumps({"episode_id": "A", "depth": 0, "d_star": 1}),
            json.dumps({"episode_id": "B", "depth": 0, "d_star": 0}),
            json.dumps({"episode_id": "A", "depth": 1, "d_star": 0}),
        ]
        with pytest.raises(ValueError, match="non-contiguous"):
            prefix_hashes([l for l in lines], [2])


class TestPreloadedStore:  # R3, lead 13:27
    def test_preload_once_reuse_multi_k_no_store_reopen(self, tmp_path, monkeypatch):
        """Pre-load store ONCE → materialize for k=1,2,3 WITHOUT re-opening
        the store file (spy: store file read-opens == 1 from preload only)."""
        import builtins
        monkeypatch.chdir(tmp_path)
        _generate_v08("v08.jsonl", seed=42, n=4, band=(2, 4))
        lines = [l for l in open("v08.jsonl") if l.strip()]

        # Copy layout store to a local path so we can spy on it
        import shutil
        shutil.copy(_LAYOUT_STORE, "store.json")
        store_sha = hashlib.sha256(open("store.json", "rb").read()).hexdigest()

        from ucm.v1.episode_budget import preload_store, materialize_episodes_preloaded
        import ucm.v1.episode_budget as _eb

        opens = []
        real_open = builtins.open
        def spy(f, *a, **k):
            opens.append(os.path.realpath(str(f)))
            return real_open(f, *a, **k)
        monkeypatch.setattr(builtins, "open", spy)

        # Pre-load ONCE
        store_mapping, sha = preload_store("store.json", store_sha)
        assert sha == store_sha

        # Multi-k: all use the SAME pre-loaded mapping
        data_sha = _eb.sha_of(lines) if hasattr(_eb, 'sha_of') else hashlib.sha256(
            ("\n".join(l.rstrip("\r\n") for l in lines if l.strip()) + "\n").encode()).hexdigest()
        for k in (1, 2, 3):
            records, meta = materialize_episodes_preloaded(
                lines, k, store_mapping, data_sha, data_sha)
            assert len(records) > 0

        monkeypatch.setattr(builtins, "open", real_open)
        # store.json: exactly 1 read-open (from preload_store), 0 from multi-k
        store_opens = [o for o in opens if o.endswith("store.json")]
        assert len(store_opens) == 1, \
            f"store.json opened {len(store_opens)}× (expect 1 from preload only): {store_opens}"
        # v08.jsonl: 0 opens (lines already in memory)
        v08_opens = [o for o in opens if o.endswith("v08.jsonl")]
        assert len(v08_opens) == 0

    def test_wrong_store_sha_fails(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        import shutil
        shutil.copy(_LAYOUT_STORE, "store.json")
        from ucm.v1.episode_budget import preload_store
        with pytest.raises(Exception, match="MISMATCH"):
            preload_store("store.json", "0" * 64)

    def test_records_identical_path_vs_preloaded(self, tmp_path, monkeypatch):
        """Records from pre-loaded store == records from path-based store."""
        monkeypatch.chdir(tmp_path)
        _generate_v08("v08.jsonl", seed=42, n=4, band=(2, 4))
        lines = [l for l in open("v08.jsonl") if l.strip()]
        import shutil
        shutil.copy(_LAYOUT_STORE, "store.json")
        store_sha = hashlib.sha256(open("store.json", "rb").read()).hexdigest()

        from ucm.v1.episode_budget import preload_store, materialize_episodes_preloaded
        from ucm.v1.episode_budget import materialize_episodes
        

        store_mapping, _ = preload_store("store.json", store_sha)
        data_sha = sha_of(lines)  # local helper

        records_pre, meta_pre = materialize_episodes_preloaded(
            lines, 2, store_mapping, data_sha, data_sha)
        records_path, meta_path = materialize_episodes(lines, 2, _LAYOUT_STORE)

        assert len(records_pre) == len(records_path)
        for rp, rn in zip(records_pre, records_path):
            assert rp["supervision"] == rn["supervision"]
            assert rp["policy_input"]["goal"] == rn["policy_input"]["goal"]
            assert rp["provenance"]["layout_hash"] == rn["provenance"]["layout_hash"]


class TestProductionPreloaded:  # R3 production, lead 13:32
    def test_preloaded_entry_store_opened_once_multi_k(self, tmp_path, monkeypatch):
        """Production preloaded: store opened EXACTLY 1× across k=1,2,3,
        data file 0× (already in memory), finetune receives k-episode records."""
        import builtins, shutil
        monkeypatch.chdir(tmp_path)
        _generate_v08("v08.jsonl", seed=42, n=4, band=(2, 4))
        lines = [l for l in open("v08.jsonl") if l.strip()]
        shutil.copy(_LAYOUT_STORE, "store.json")
        store_sha = hashlib.sha256(open("store.json", "rb").read()).hexdigest()
        data_sha = sha_of(lines)

        from ucm.v1.episode_budget import preload_store, finetune_episode_budget_preloaded
        from ucm.v1 import runner as _runner_mod

        received = []
        def spy_finetune(model, records, cfg, cov, arm, k):
            received.append(len(records))
            return {"updates": 0}

        opens = []
        real_open = builtins.open
        def spy_open(f, *a, **k):
            opens.append(os.path.realpath(str(f)))
            return real_open(f, *a, **k)

        monkeypatch.setattr(_runner_mod, "finetune", spy_finetune)
        monkeypatch.setattr(builtins, "open", spy_open)

        store_mapping, _ = preload_store("store.json", store_sha)
        import mlx.core as mx
        mx.set_default_device(mx.cpu)
        from ucm.model.siw_model import make_siw_model
        from ucm.v1.transfer import FinetuneConfig, CoverageTracker
        m = make_siw_model()
        cfg = FinetuneConfig(updates=0)
        cov = CoverageTracker()

        for k in (1, 2, 3):
            finetune_episode_budget_preloaded(m, data_sha, data_sha, lines, k,
                                               cfg, cov, "test", store_mapping, store_sha)

        monkeypatch.setattr(builtins, "open", real_open)
        # Store: exactly 1 open (from preload_store), 0 from multi-k
        store_opens = [o for o in opens if o.endswith("store.json")]
        assert len(store_opens) == 1, f"store.json: {len(store_opens)} opens (expect 1)"
        # Data file: 0 opens
        v08_opens = [o for o in opens if o.endswith("v08.jsonl")]
        assert len(v08_opens) == 0
        # finetune called 3 times with increasing records
        assert len(received) == 3
        assert received[0] < received[1] < received[2]
