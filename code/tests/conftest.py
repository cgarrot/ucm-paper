"""Shared fixtures.

v8 is built and bound: REAL freeze verification is the DEFAULT.
UCM_IDENTITY_FREEZE_VERIFY=1 opts out (legacy crutch, do not use for the run).
"""
import os
import json
import pytest


@pytest.fixture(autouse=True)
def _identity_freeze_verify(tmp_path, monkeypatch):
    # Persistence safety (lead 06:53): tests NEVER write run artifacts into
    # the real artifacts/v1bis-run — per-test tmp dir + unique O_EXCL-safe ts.
    import tempfile
    monkeypatch.setenv("UCM_RUN_DIR", tempfile.mkdtemp(prefix="ucm-test-run-"))
    # Crash-3 cache isolation: each test starts with an empty store cache
    # (fresh-process semantics — open counts must not leak across tests).
    import ucm.v1.episode_budget as _eb
    _eb._STORE_CACHE.clear()
    if not os.environ.get("UCM_IDENTITY_FREEZE_VERIFY"):  # v8 built: REAL verify is the default now
        yield
        return
    import ucm.v1.freeze_v1bis as _fz
    import ucm.v1.runner_official as _ro
    import ucm.v1.runner_parallel as _rp
    import hashlib
    def _iv(p, h):
        # Hash-checking identity: refuses wrong protocol hash (that behavior
        # is real and tested), but skips the code-drift pin (v8 scope).
        import hashlib as _hl
        fm = json.load(open(p))
        canon = _hl.sha256(json.dumps(fm, sort_keys=True, default=str).encode()).hexdigest()
        if canon != h:
            import sys
            sys.exit(f"freeze-v1bis: protocol hash mismatch (test identity) {canon[:12]}… != {h[:12]}…")
        return fm
    _reals = [(_fz, "verify_freeze_v1bis", _fz.verify_freeze_v1bis),
              (_ro, "verify_freeze_v1bis", _ro.verify_freeze_v1bis),
              (_rp, "verify_freeze_v1bis", _rp.verify_freeze_v1bis)]
    for mod, name, _ in _reals:
        setattr(mod, name, _iv)
    yield
    for mod, name, real in _reals:
        setattr(mod, name, real)
