"""V1-bis opt-in: episode-level budget selection on schema 0.8 DEV data.

NEW module — does NOT touch runner_confirm.main or any sealed/test2 path.

Semantics: the budget k selects k COMPLETE EPISODES (all their records,
including the terminal STOP), not k individual records. Selection is
round-robin over the writer's episode ordering (already deterministic).

API:
    select_episodes(lines, k) → (selected_lines, metadata)
      - lines: raw JSONL lines from a single read
      - k: number of EPISODES to select (0 ≤ k ≤ total_episodes)
      - selected_lines: the exact JSONL lines for those episodes, in order
      - metadata: {n_episodes, n_records, episode_ids, per_episode_record_counts,
                   prefix_sha256: {1..k → sha256 of concatenated lines}}

    prefix_hashes(lines, ks) → {k: sha256}
      - SHA-256 of the JSONL bytes of the first k episodes' lines concatenated
      - matches writer's serialize_rr_records for the same prefix

    materialize_episodes(lines, k, layout_store) → (records, metadata)
      - select k episodes → pass ALL their lines to couples_lines_to_records
      - records are in the ORIGINAL order (no re-slicing)
      - metadata carries episode identities for the training manifest

Discipline: single read (the caller does read_once), no re-open for multi-k.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict


def _episode_groups(lines: list[str], require_complete: bool = True
                    ) -> "OrderedDict[str, list[str]]":
    """Group contiguous lines by episode_id. Asserting:
    - contiguity (no re-appearance)
    - completeness (if require_complete): d* strictly decreasing to 0,
      last record d*=0 (STOP terminal), depths sequential 0..n-1."""
    groups: OrderedDict[str, list[str]] = OrderedDict()
    current = None
    for l in lines:
        if not l.strip():
            continue
        r = json.loads(l)
        ep = r["episode_id"]
        if ep != current:
            if ep in groups:
                raise ValueError(f"episode {ep!r} re-appears (non-contiguous)")
            groups[ep] = []
            current = ep
        groups[ep].append(l)
    if require_complete:
        for ep, recs in groups.items():
            parsed = [json.loads(l) for l in recs]
            depths = [r["depth"] for r in parsed]
            if depths != list(range(len(depths))):
                raise ValueError(f"episode {ep!r}: depths {depths} not sequential")
            dstars = [r["d_star"] for r in parsed]
            if dstars[-1] != 0:
                raise ValueError(f"episode {ep!r}: truncated (last d*={dstars[-1]} ≠ 0)")
            if any(dstars[i] <= dstars[i+1] for i in range(len(dstars)-1)):
                raise ValueError(f"episode {ep!r}: d* not strictly decreasing")
            if len(parsed) != parsed[0]["d_star"] + 1:
                raise ValueError(f"episode {ep!r}: {len(parsed)} records ≠ d*({parsed[0]['d_star']})+1")
    return groups


def select_episodes(lines: list[str], k: int) -> tuple[list[str], dict]:
    """Select the first k COMPLETE episodes from the stream. Returns
    (selected_lines, metadata). Zero re-opens: operates on lines in memory."""
    groups = _episode_groups(lines)
    ep_ids = list(groups.keys())
    if not (0 <= k <= len(ep_ids)):
        raise ValueError(f"k={k} out of [0, {len(ep_ids)}]")
    selected = []
    counts = {}
    for ep in ep_ids[:k]:
        selected.extend(groups[ep])
        counts[ep] = len(groups[ep])
    # prefix hashes: sha256 of the JSONL bytes for prefixes 1..k
    prefix_sha = {}
    running = []
    for ep in ep_ids[:k]:
        running.extend(groups[ep])
        canon = "\n".join(r.rstrip("\n") for r in running) + "\n"
        prefix_sha[len(prefix_sha) + 1] = hashlib.sha256(canon.encode()).hexdigest()
    meta = {
        "n_episodes": k,
        "n_records": len(selected),
        "episode_ids": ep_ids[:k],
        "per_episode_record_counts": counts,
        "prefix_sha256": prefix_sha,
    }
    return selected, meta


def prefix_hashes(lines: list[str], ks: list[int]) -> dict[int, str]:
    """SHA-256 of the JSONL bytes of the first k episodes, for each k in ks.
    Single pass — no re-open for multi-k."""
    groups = _episode_groups(lines)
    ep_ids = list(groups.keys())
    out = {}
    for k in sorted(ks):
        if not (0 <= k <= len(ep_ids)):
            raise ValueError(f"k={k} out of [0, {len(ep_ids)}]")
        selected = []
        for ep in ep_ids[:k]:
            selected.extend(groups[ep])
        canon = ("\n".join(r.rstrip("\n") for r in selected) + "\n") if selected else ""
        out[k] = hashlib.sha256(canon.encode()).hexdigest()
    return out


def materialize_episodes(lines: list[str], k: int, layout_store: str) -> tuple[list[dict], dict]:
    """Select k episodes → materialize ALL their records via the adapter.
    Returns (records, metadata). The records list is exactly what finetune
    should receive — no re-slicing with records[:K_STAR]."""
    selected, meta = select_episodes(lines, k)
    from ucm.v1.data_adapter import couples_lines_to_records
    records = couples_lines_to_records(selected, layout_store)
    meta["n_materialized"] = len(records)
    meta["adapter_layout_store"] = layout_store
    return records, meta


def verify_full_file_sha(reader_sha: str, lines: list[str]) -> bool:
    """Verify that the canonical reconstruction of lines matches the SHA the
    sealed_reader computed on the actual file bytes. If the file had CRLF,
    blank lines, or a missing final newline, the reconstruction will NOT match
    → the caller must reject non-canonical files before trusting prefix hashes.

    NOTE: read_once computes SHA on raw bytes (including any \r\n etc).
    Our prefix hashes operate on canonical "\n".join(...) + "\n" bytes.
    These will differ for non-canonical files — that IS the detection."""
    import hashlib as _h
    stripped = [l.rstrip("\r\n") for l in lines if l.strip()]
    canon = ("\n".join(stripped) + "\n") if stripped else ""
    canon_sha = _h.sha256(canon.encode()).hexdigest()
    return reader_sha == canon_sha


def from_read_once(lines: list[str], reader_sha: str, expected_sha: str) -> list[str]:
    """Validate SHA + canonical, return the VERIFIED lines (not records —
    the caller passes them to finetune_episode_budget for per-k selection).
    Raises before any adapter/train on any violation."""
    if reader_sha != expected_sha:
        raise ValueError(f"reader SHA {reader_sha[:12]}… != expected {expected_sha[:12]}…")
    if not verify_full_file_sha(reader_sha, lines):
        raise ValueError("non-canonical file (CRLF/blank-lines/no-final-newline) — "
                         "prefix hashes would be unreliable")
    return lines


# 3) Add the production entry point: select + materialize + finetune
def finetune_episode_budget(model, reader_sha: str, expected_sha: str,
                             lines: list[str], k: int,
                             cfg, cov, arm: str, layout_store: str):
    """Production opt-in entry point (v2 — lead 13:14: NO hash bypass).

    REQUIRES reader_sha and expected_sha — validates canonical + SHA BEFORE
    any select/adapter/finetune. The old signature without hashes is REMOVED
    to prevent bypassing verification via direct lines.

    This is what the V1-bis runner should call instead of
    finetune(model, records[:K_STAR], ...)."""
    # 1) SHA check: reader must match expected published hash
    if reader_sha != expected_sha:
        raise ValueError(f"reader SHA {reader_sha[:12]}… != expected "
                         f"{expected_sha[:12]}… — refusing to train")
    # 2) Canonical check: no CRLF/blank/no-final-newline
    if not verify_full_file_sha(reader_sha, lines):
        raise ValueError("non-canonical file — prefix hashes unreliable, "
                         "refusing to train")
    # 3) NOW select + materialize + finetune (hashes already verified)
    selected_lines, meta = select_episodes(lines, k)
    from ucm.v1.data_adapter import couples_lines_to_records
    records = couples_lines_to_records(selected_lines, layout_store)
    assert len(records) == meta["n_records"], \
        f"adapter returned {len(records)} != expected {meta['n_records']}"
    import ucm.v1.runner as _runner
    result = _runner.finetune(model, records, cfg, cov, arm, k)
    result["episode_budget_metadata"] = meta
    return result


# ---------------------------------------------------------------------------
# R3 (lead 13:27): pre-loaded store — ONE read, reused across ALL k values.
# ---------------------------------------------------------------------------

# Crash 3 fix (lead 22:22): module-level store cache — exactly 1 read_once
# PER PROCESS. Pool chunking calls _worker_cell several times per worker
# process; the 2nd read_once of the same file violates one-read-per-file
# (registry refuses) → crash. Cache hit skips the read entirely.
_STORE_CACHE: dict = {}  # (path, sha) -> (mapping, resolved_sha)


def preload_store(store_path: str, expected_sha: str) -> tuple[dict, str]:
    """Read the layout store ONCE per process (SealedOpenRegistry discipline),
    validate SHA, return (store_mapping, sha). Cached per (path, sha) — all
    subsequent calls in the same process reuse it without re-opening.

    The old path-based layout_store param in couples_lines_to_records remains
    the default; this is the OPT-IN for multi-k efficiency."""
    cache_key = (store_path, expected_sha)
    if cache_key in _STORE_CACHE:
        return _STORE_CACHE[cache_key]
    from ucm.v1.sealed_reader import sealed_registry
    reg = sealed_registry()
    fmt, lines, sha = reg.read_once(store_path, expected_hash=expected_sha)
    # NOTE: the store file is a SINGLE JSON OBJECT (not JSONL). If it ever
    # becomes multi-line JSONL, this parse will fail — by design (fail-closed).
    raw = "\n".join(lines) if lines else "{}"
    store = json.loads(raw)
    if not isinstance(store, dict):
        raise ValueError(f"store is not a JSON object (got {type(store).__name__}) — "
                         "single-object store only")
    # v02 wrapper unwrap (crash GO-run 19:58): the v1bis-gen store ships as
    # {schema, manifest_sha256, store: {...}, ...}. A bare layout map has
    # layout-hash keys (16 hex). Detect the wrapper by its two signature keys
    # and unwrap — fail-closed if the inner store is not a dict.
    if "store" in store and "manifest_sha256" in store:
        inner = store["store"]
        if not isinstance(inner, dict):
            raise ValueError("v02 wrapper detected but inner store is not a dict")
        store = inner
    _STORE_CACHE[cache_key] = (store, sha)
    return store, sha


def materialize_episodes_preloaded(lines: list[str], k: int,
                                     store_mapping: dict,
                                     reader_sha: str, expected_sha: str) -> tuple[list[dict], dict]:
    """Like materialize_episodes but takes a PRE-LOADED store mapping
    (no file re-open per k). Validates SHA + canonical first."""
    if reader_sha != expected_sha:
        raise ValueError(f"reader SHA mismatch — refusing")
    if not verify_full_file_sha(reader_sha, lines):
        raise ValueError("non-canonical — refusing")
    selected, meta = select_episodes(lines, k)
    records = _couples_from_store(selected, store_mapping)
    meta["n_materialized"] = len(records)
    return records, meta


def _couples_from_store(lines: list[str], store: dict) -> list[dict]:
    """In-memory variant of couples_lines_to_records that takes the store
    as a dict instead of a path — no file open."""
    from ucm.v1.data_adapter import couples_from_store_dict
    return couples_from_store_dict(lines, store)


def finetune_episode_budget_preloaded(model, reader_sha: str, expected_sha: str,
                                       lines: list[str], k: int,
                                       cfg, cov, arm: str,
                                       store_mapping: dict, store_sha: str):
    """Production entry (R3 preloaded): validates data SHA + canonical, then
    calls _couples_core with the PRE-LOADED store — the store file was opened
    exactly ONCE (by preload_store) and is reused for ALL k values."""
    if reader_sha != expected_sha:
        raise ValueError(f"reader SHA mismatch — refusing to train")
    if not verify_full_file_sha(reader_sha, lines):
        raise ValueError("non-canonical — refusing")
    selected, meta = select_episodes(lines, k)
    from ucm.v1.data_adapter import _couples_core
    records = _couples_core(store_mapping, selected)
    assert len(records) == meta["n_records"]
    import ucm.v1.runner as _runner
    result = _runner.finetune(model, records, cfg, cov, arm, k)
    result["episode_budget_metadata"] = meta
    return result


def sha_of(lines: list[str]) -> str:
    """SHA-256 of the canonical JSONL bytes (same normalization as prefix_hashes)."""
    stripped = [l.rstrip("\r\n") for l in lines if l.strip()]
    canon = ("\n".join(stripped) + "\n") if stripped else ""
    return hashlib.sha256(canon.encode()).hexdigest()
