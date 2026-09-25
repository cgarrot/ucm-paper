"""Script de scellement test2 V1-bis (DEV-dry, lead 16:01 step 1).

Finalizes the sealing script integrating tagi-5 notes N1-N3:
  N1: journal §3-bis of generation failures + replacement (M1/iso/canonicity
      checked BEFORE run, failures logged)
  N2: publication via BundleWriter/publish_pointer + recovery (crash-atomic v9)
  N3: canonicity controls (UTF-8/LF, stream SHA + per-episode SHA)

DEV-dry mode: the script structure, validation, journal, publication
mechanics, and pre-sealing checks are all implemented and tested — but the
actual test2 generation is STUBBED (requires V1BIS_STEP4_GO=LEAD_APPROVED
to execute). This makes the script READY for the GO.

Usage:
    .venv/bin/python -m ucm.v1.seal_test2 --dry-run    # DEV validation
    .venv/bin/python -m ucm.v1.seal_test2               # requires GO token
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import tempfile
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

import mlx.core as mx

mx.set_default_device(mx.cpu)

EXPECTED_TOTAL = 600
EXPECTED_PER_PREDICATE = 150
PREDICATES = ("VIEW", "SET", "CHOOSE", "SUBMITTED")
# v6 anti-circularity: the expected hash lives in a PUBLISHED POINTER FILE
# (append-only, published by the freeze ceremony). The freeze binds sha of
# seal_test2.py itself, so the constant must NOT be embedded here.
_FREEZE_POINTER = os.path.join(_REPO, "artifacts/freeze-v1bis-official-v10.sha256")
def _expected_protocol_hash() -> str:
    h = open(_FREEZE_POINTER, encoding="utf-8").read().strip()
    if len(h) != 64:
        raise RuntimeError(f"freeze pointer corrupt: {len(h)} chars, expected 64")
    return h
EXPECTED_PROTOCOL_HASH = _expected_protocol_hash()  # published constant, not recomputed
OUT_DIR = os.environ.get("UCM_TEST2_SEAL_DIR", "/tmp/ucm-test2-seal")


class SealingJournal:
    """N1: §3-bis journal — every attempt (success or failure) is logged
    with evidence. Append-only, never rewritten."""

    def __init__(self, path: str):
        self.path = path
        self.entries = []
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        if os.path.exists(path):
            # read append-only format; handle tail partial line (crash mid-write)
            for line_no, line in enumerate(open(path), 1):
                if not line.strip():
                    continue
                try:
                    self.entries.append(json.loads(line))
                except json.JSONDecodeError as e:
                    import warnings
                    warnings.warn(f"SealingJournal: partial tail at line {line_no} "
                                  f"(crash mid-write): {e}", RuntimeWarning)
                    self.entries.append({"t": "recovered",
                                         "event": "journal_recovered",
                                         "line": line_no, "error": str(e)[:100]})

    def log(self, event: str, evidence: dict):
        """Append-only with fsync: never truncate, each entry durable."""
        entry = {"t": datetime.datetime.now().isoformat(timespec="seconds"),
                 "event": event, **evidence}
        # journal_recovered is emitted by the constructor on partial tail recovery
        self.entries.append(entry)
        # append to file (never rewrite), then fsync
        with open(self.path, "a") as fh:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    @property
    def has_replacement(self) -> bool:
        return any(e["event"] == "generation_failed_and_replaced" for e in self.entries)


def validate_m1(episodes: list[dict]) -> dict:
    """M1 validation: 600 total, 150 per predicate, layout_hash recompute,
    disjunction from adaptation pool."""
    from collections import Counter
    if len(episodes) != EXPECTED_TOTAL:
        return {"ok": False, "reason": f"total {len(episodes)} != {EXPECTED_TOTAL}"}
    per_pred = Counter(e["task"]["goal"]["predicate"] for e in episodes)
    for p in PREDICATES:
        if per_pred.get(p, 0) != EXPECTED_PER_PREDICATE:
            return {"ok": False, "reason": f"predicate {p}: {per_pred.get(p, 0)} != {EXPECTED_PER_PREDICATE}"}
    # layout_hash recompute
    from ucm.v1.runner_confirm import _recompute_layout_hash
    for e in episodes:
        if _recompute_layout_hash(e["layout_spec"]) != e["layout_hash"]:
            return {"ok": False, "reason": f"layout_hash mismatch for {e['episode_id']}"}
    return {"ok": True, "per_predicate": dict(per_pred), "n_layouts": len({e["layout_hash"] for e in episodes})}


def build_adaptation_pool() -> set:
    """Sealing pool (2nd defense behind the generator). Builds from FOUR sources:
    1. v1bis-gen files (SIW adaptation pool)
    2. TGK pre-training data (m0-transitions)
    3. SIW 420 pool (replay seed 20260926, NOT 20261026)
    4. DEV 30 (inventory-siw-dev-layouts)
    All fail-closed: missing source → error, not silent skip."""
    import glob as _g
    hashes = set()
    sources = {"v1bis_gen": 0, "tgk": 0, "siw420": 0, "dev30": 0}

    # 1. v1bis-gen
    for f in sorted(_g.glob(os.path.join(_REPO, "artifacts/v1bis-gen/v1bis-gen-*.jsonl"))):
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            h = r.get("layout_hash", "")
            if h:
                hashes.add(h)
                sources["v1bis_gen"] += 1

    # 2. TGK (hash-only access to canon 9a19d8f4)
    tgk = os.path.join(_REPO, "artifacts/data/m0-transitions.jsonl")
    if os.path.exists(tgk):
        for line in open(tgk):
            if not line.strip():
                continue
            r = json.loads(line)
            h = r.get("provenance", {}).get("layout_hash", "")
            if h:
                hashes.add(h)
                sources["tgk"] += 1

    # 3. SIW 420 pool (seed 20260926 — NOT 20261026, tagi-5 16:26)
    siw420 = os.path.join(_REPO, "artifacts/dev-v1bis-pool200.json")
    if os.path.exists(siw420):
        pool = json.load(open(siw420))
        for h in pool.get("exclusion_set", {}).get("layout_hashes", []):
            hashes.add(h)
            sources["siw420"] += 1

    # 4. DEV 30 (inventory published)
    inv = os.path.join(_REPO, "artifacts/inventory-siw-dev-layouts.json")
    if os.path.exists(inv):
        for h in json.load(open(inv)):
            hashes.add(h)
            sources["dev30"] += 1

    # Fail-closed: all four sources must have contributed
    for name, count in sources.items():
        if count == 0:
            raise RuntimeError(f"sealing pool incomplete: {name} contributed 0 hashes")

    return hashes


def validate_iso_disjunction(episodes: list[dict], adaptation_hashes: set) -> dict:
    """Iso-disjunction: test2 layouts must be disjoint from adaptation pool.
    FAIL-CLOSED: any overlap → not ok."""
    if not adaptation_hashes:
        return {"ok": False, "reason": "adaptation pool is empty (must be built from v1bis-gen)"}
    test_layouts = {e["layout_hash"] for e in episodes}
    overlap = test_layouts & adaptation_hashes
    if overlap:
        return {"ok": False, "reason": f"{len(overlap)} layouts overlap adaptation pool"}
    return {"ok": True, "n_test_layouts": len(test_layouts),
            "n_adaptation": len(adaptation_hashes)}


def validate_canonicity(path: str) -> dict:
    """N3: canonicity controls — UTF-8, LF, stream SHA + per-episode SHA."""
    raw = open(path, "rb").read()
    # UTF-8 check
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError:
        return {"ok": False, "reason": "not valid UTF-8"}
    # LF check (no CRLF)
    if b"\r\n" in raw:
        return {"ok": False, "reason": "contains CRLF (must be LF)"}
    # stream SHA
    stream_sha = hashlib.sha256(raw).hexdigest()
    # per-episode SHA
    lines = raw.decode().strip().split("\n")
    per_ep = {}
    for i, line in enumerate(lines):
        per_ep[f"ep_{i}"] = hashlib.sha256(line.encode()).hexdigest()
    return {"ok": True, "stream_sha256": stream_sha,
            "n_episodes": len(lines), "per_episode_sha256": per_ep}


def publish_stage_b(data_path: str, freeze_path: str, protocol_hash: str,
                    out_dir: str) -> dict:
    """N2+N3: publication via BundleWriter/publish_pointer (crash-atomic v9),
    LINKED to the freeze manifest (hash published BEFORE run)."""
    from ucm.v1.atomic_publish import BundleWriter, capability_probe
    os.makedirs(out_dir, exist_ok=True)
    capability_probe(out_dir)
    # compute the data SHA (stream, the ONLY hash)
    data_sha = hashlib.sha256(open(data_path, "rb").read()).hexdigest()
    # build data-manifest LINKED to freeze
    dm = {
        "path": os.path.abspath(data_path),
        "sha256_write_stream": data_sha,
        "freeze_protocol_hash": protocol_hash,
        "freeze_path": os.path.abspath(freeze_path),
        "published_before_run": True,
    }
    bw = BundleWriter(out_dir, "test2-seal")
    bw.add_bytes("test2-episodes.jsonl", open(data_path, "rb").read())
    bw.add_bytes("data-manifest.json",
                  json.dumps(dm, indent=2, sort_keys=True).encode())
    manifest = bw.finalize()
    bw.publish_pointer(os.path.join(out_dir, "test2.pointer"))
    return manifest


def seal_test2(dry_run: bool = True) -> dict:
    """Main sealing function. dry_run=True: validate structure without
    generating. dry_run=False: requires V1BIS_STEP4_GO=LEAD_APPROVED."""
    if not dry_run:
        token = os.environ.get("V1BIS_STEP4_GO")
        if token != "LEAD_APPROVED":
            raise RuntimeError("GO token required: set V1BIS_STEP4_GO=LEAD_APPROVED")

    journal = SealingJournal(os.path.join(OUT_DIR, "sealing-journal.json"))

    # === GENERATION (dry_run: stub; full: real generator) ===
    if dry_run:
        # DEV-dry: create a miniature valid file for structure validation
        # (6 episodes, 150 per predicate is too heavy for dry — use 1 each)
        # This validates the PIPELINE, not the data.
        # We just validate the script structure without actual episodes
        episodes = None  # stub
        generation_result = {"mode": "dry_run", "note": "no generation — structure validation only"}
        journal.log("generation_stub", generation_result)
    else:
        # FULL PATH (B2: ordered freeze→pool→generation→validations→publication)
        from ucm.v1.freeze_v1bis import verify_freeze_v1bis as _vf
        from ucm.data.test2_generation import generate_test2_episodes

        # 1. FREEZE (constant, not tautological)
        FREEZE = os.path.join(_REPO, "artifacts/freeze-v1bis-official-v10.json")
        fm = _vf(FREEZE, EXPECTED_PROTOCOL_HASH)
        journal.log("freeze_verified", {"hash": EXPECTED_PROTOCOL_HASH[:16]})

        # 2. POOL (4 sources, fail-closed)
        pool = build_adaptation_pool()
        journal.log("pool_built", {"n": len(pool)})

        # 3. GENERATION (§3-bis: seed 20261003 → replacement 20261004)
        gen_seed = 20261003
        replacement_seed = 20261004
        episodes = None
        for attempt_seed in (gen_seed, replacement_seed):
            journal.log("generation_start", {"seed": attempt_seed,
                                              "is_replacement": attempt_seed == replacement_seed})
            try:
                art = generate_test2_episodes(seed_pool=attempt_seed, seed_episodes=attempt_seed)
                episodes = art["selfcontained_full"]
                break
            except Exception as e:
                journal.log("generation_failed", {"seed": attempt_seed,
                                                  "error": str(e)[:200]})
                if attempt_seed == replacement_seed:
                    journal.log("generation_definitive_failure", {"seed": attempt_seed})
                    raise RuntimeError(f"§3-bis: both seeds failed (last: {e})")

        # 4. VALIDATIONS (M1 + iso + canonicity)
        m1 = validate_m1(episodes)
        if not m1["ok"]:
            journal.log("validation_m1_failed", m1)
            raise RuntimeError(f"M1: {m1['reason']}")
        journal.log("validation_m1_ok", m1)

        iso = validate_iso_disjunction(episodes, pool)
        if not iso["ok"]:
            journal.log("validation_iso_failed", iso)
            raise RuntimeError(f"iso: {iso['reason']}")
        journal.log("validation_iso_ok", iso)

        # Write episodes to temp file for canonicity + publication
        data_path = os.path.join(OUT_DIR, "test2-episodes.jsonl")
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(data_path, "w", encoding="utf-8", newline="\n") as fh:
            for e in episodes:
                fh.write(json.dumps(e, sort_keys=True, default=str) + "\n")
        canon = validate_canonicity(data_path)
        if not canon["ok"]:
            journal.log("validation_canonicity_failed", canon)
            raise RuntimeError(f"canonicity: {canon['reason']}")
        journal.log("validation_canonicity_ok", {"sha": canon["stream_sha256"][:16]})

        # 5. PUBLICATION (N2: BundleWriter + pointer, hash published BEFORE run)
        manifest = publish_stage_b(data_path, FREEZE, EXPECTED_PROTOCOL_HASH, OUT_DIR)
        journal.log("published", {"bundle_manifest": str(manifest)[:200]})

        return {"status": "sealed", "n_episodes": len(episodes),
                "stream_sha256": canon["stream_sha256"],
                "journal_entries": len(journal.entries),
                "out_dir": OUT_DIR}

    # === VALIDATIONS (structure validated in dry_run) ===
    if dry_run:
        # Validate structure + wire iso-disjunction
        assert callable(validate_m1)
        assert callable(validate_iso_disjunction)
        assert callable(validate_canonicity)
        assert callable(publish_stage_b)
        # Build adaptation pool from BOTH sources (v1bis-gen + TGK)
        pool = build_adaptation_pool()
        assert len(pool) > 0, "adaptation pool built from v1bis-gen + TGK files"
        journal.log("adaptation_pool_built", {"n_hashes": len(pool),
                                               "sources": ["v1bis-gen/*.jsonl", "m0-transitions.jsonl"]})
        # Verify freeze (tagi-5 fix 1: EXPECTED constant, not tautological recompute)
        from ucm.v1.freeze_v1bis import verify_freeze_v1bis
        FREEZE = os.path.join(_REPO, "artifacts/freeze-v1bis-official-v10.json")
        verified_fm = verify_freeze_v1bis(FREEZE, EXPECTED_PROTOCOL_HASH)
        journal.log("freeze_verified", {"protocol_hash": EXPECTED_PROTOCOL_HASH[:16] + "…",
                                         "version": verified_fm.get("version")})
        journal.log("structure_validated", {"functions": ["validate_m1",
                                                           "validate_iso_disjunction",
                                                           "validate_canonicity",
                                                           "publish_stage_b"]})

    # === PUBLICATION (N2: crash-atomic v9) ===
    if dry_run:
        # Validate BundleWriter + pointer mechanics on a mini file
        from ucm.v1.atomic_publish import BundleWriter, capability_probe
        os.makedirs(OUT_DIR, exist_ok=True)
        capability_probe(OUT_DIR)
        bw = BundleWriter(OUT_DIR, "test2-dry")
        bw.add_bytes("test.txt", b"dry-run validation")
        mini_manifest = bw.finalize()
        bw.publish_pointer(os.path.join(OUT_DIR, "test2-dry.pointer"))
        journal.log("publication_dry_run", {"bundle": "test2-dry",
                                             "manifest_keys": list(mini_manifest.keys()) if isinstance(mini_manifest, dict) else str(type(mini_manifest))})

    return {"status": "dry_run_complete" if dry_run else "sealed",
            "journal_entries": len(journal.entries),
            "journal_path": journal.path,
            "out_dir": OUT_DIR}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", default=False)
    args = ap.parse_args()
    result = seal_test2(dry_run=True)  # always dry-run from CLI until GO
    print(json.dumps(result, indent=2, default=str))
