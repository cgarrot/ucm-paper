"""One-read STATIC sealed-file reader (lead finding 21:35) — single streaming
open per sealed file: format detection + parsing + content hash in ONE pass.

Discipline (for the confirmation runner):
    - log INTENTION BEFORE the open (sealed opening declared up front)
    - ONE open (streaming; second open of the same path in the same process →
      RuntimeError via SealedOpenRegistry)
    - log hash/count AFTER the read
    - never re-open: consumers get the parsed data + hash; adapter_manifest
      reuses the recorded hash instead of touching the file again.

Test2 (confirmation) is NEVER opened by anything here until its protocol is
frozen; this module is exercised on DEV files only.
"""

from __future__ import annotations

import hashlib
import json


class SealedOpenRegistry:
    """Process-wide one-open guard for sealed artifacts (empty allowlist of
    re-opens): the first open() of a registered path passes, any subsequent
    open() raises RuntimeError — including probes and hash recomputations."""

    def __init__(self):
        self.opened: dict[str, int] = {}
        self.records: dict[str, dict] = {}

    @staticmethod
    def _key(path: str) -> str:
        """Canonical key: os.path.realpath — relative/absolute/symlink of the
        same file count as ONE read (10:58)."""
        import os as _os
        return _os.path.realpath(path)

    def read_once(self, path: str, expected_hash: str | None = None
                  ) -> tuple[str, list[str], str]:
        """Single streaming read → (format_hint, lines, FULL sha256).
        expected_hash: if provided (full hex), mismatch → RuntimeError (no
        silent corruption; lead 21:44 (6))."""
        key = self._key(path)
        if key in self.opened:
            raise RuntimeError(
                f"sealed file {path} (→{key}) ALREADY OPENED "
                f"({self.opened[key]} reads) — one-read discipline violated")
        self.opened[key] = 1
        h = hashlib.sha256()
        lines = []
        first = None
        with open(path, "rb") as fh:
            for raw in fh:
                h.update(raw)
                line = raw.decode().rstrip("\n")
                if line.strip():
                    lines.append(line)
                    if first is None:
                        first = line
        full = h.hexdigest()
        if expected_hash is not None and full != expected_hash:
            raise RuntimeError(f"sealed hash MISMATCH for {path}: read {full}, "
                               f"expected {expected_hash}")
        # schema-based dispatch (10:26): parse the first line as JSON and
        # look at keys/schema — no byte-prefix peek; works for both v1
        # couples ({goal,layout_hash,state_key}) and v2 (schema-first,
        # sort_keys => first key alphabetical, NOT predictable)
        try:
            first_obj = json.loads(first) if first else {}
        except Exception:
            first_obj = {}
        keys = set(first_obj.keys()) if isinstance(first_obj, dict) else set()
        if isinstance(first_obj, dict) and "schema" in first_obj:
            schema = str(first_obj["schema"])
            fmt = "couples" if "couple" in schema or "adaptation" in schema \
                else "episodes" if "episode" in schema else "records"
        elif {"goal", "layout_hash", "state_key"} <= keys:
            fmt = "couples"
        elif {"episode_ref", "layout_hash", "init", "goal"} <= keys or \
                {"episode_id", "layout_spec", "task"} <= keys:
            fmt = "episodes"
        else:
            fmt = "records"
        self.records[key] = {"path": path, "n_lines": len(lines),
                              "sha256_full": full, "short": full[:16], "format": fmt}
        return fmt, lines, full

    def recorded_hash(self, path: str) -> str:
        """FULL sha256 from the SINGLE read — never re-opens the file."""
        key = self._key(path)
        if key not in self.records:
            raise RuntimeError(f"{path} was never read; refusing to open now")
        return self.records[key]["sha256_full"]


_REGISTRY: SealedOpenRegistry | None = None


def sealed_registry() -> SealedOpenRegistry:
    """PROCESS-WIDE singleton (lead 21:44 (6)): per-instance guards allowed
    separate registries to bypass one-read; the singleton closes that hole."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = SealedOpenRegistry()
    return _REGISTRY
