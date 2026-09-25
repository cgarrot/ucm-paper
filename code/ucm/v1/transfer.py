"""V1 transfer harness (§12) — target-agnostic core, frozen per FREEZE.md.

Components:
    - UniqueCoupleBudgets: nested, arm-identical budget selection over unique
      (layout, state, goal) couples, ranked by content hash (seed 0).
    - Arms: scratch | pretrained-TGK (frozen core from canon B144 checkpoints)
      | control-nontarget (canon data with non-informative supervision).
    - finetune: fixed 2000 updates, batch 64, checkpoint at fixed budget,
      no selection on target performance.
    - CoverageTracker: unique couples SEEN by the optimizer per arm/budget
      (published next to closed-loop success; a near-zero-coverage negative
      at k*=500 is uninterpretable and must be reported as such).
    - InteractionsLog: every target-side interaction counted (§12 freeze).

The SIW environment/fixtures are NOT required to build or test this module:
a mock target (same interface: typed entities/relations/goal/candidates)
exercises the machinery end-to-end in tests. Architecture is frozen BEFORE
any target inspection (FREEZE.md §1).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Budgets: unique couples, nested, identical across arms
# ---------------------------------------------------------------------------

def couple_hash(couple: dict) -> str:
    """Content hash of a unique (layout, state, goal) couple."""
    payload = json.dumps(couple, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass
class UniqueCoupleBudgets:
    """Budget bookkeeping over unique couples. Selection order = hash of the
    COUPLE PROJECTION (default: the object itself) with seed 0 — deterministic,
    documented; nested cumulative slices IDENTICAL across arms. The carried
    objects may be full records (projector extracts the couple identity)."""

    couples: list
    seed: int = 0
    projector: object = None  # callable(obj) → couple dict for hashing/ordering

    def __post_init__(self):
        proj = self.projector or (lambda c: c)
        # dedup by couple hash (aliases/permutations create no new example)
        seen: dict[str, dict] = {}
        for c in self.couples:
            seen.setdefault(couple_hash(proj(c)), c)
        self._unique = sorted(seen.values(), key=lambda c: couple_hash(proj(c)))
        self._hashes = [couple_hash(proj(c)) for c in self._unique]

    @property
    def n_unique(self) -> int:
        return len(self._unique)

    def slice(self, k: int) -> list[dict]:
        """First k couples in the frozen hash order (nested by construction)."""
        return self._unique[:k]

    def budget_of_couple(self, couple: dict) -> int:
        return self._hashes.index(couple_hash(couple)) + 1


K_GRID = (0, 100, 500, 2000, 10000)
K_STAR = 500  # primary point (frozen)


# ---------------------------------------------------------------------------
# Coverage tracker
# ---------------------------------------------------------------------------

class CoverageTracker:
    """Counts UNIQUE couples actually exposed to the optimizer per arm/budget.
    Published next to closed-loop success (§12.5)."""

    def __init__(self):
        self._seen: dict[str, set[str]] = {}

    def expose(self, arm: str, k: int, couple: dict) -> None:
        self._seen.setdefault(f"{arm}@k={k}", set()).add(couple_hash(couple))

    def coverage(self) -> dict:
        return {key: len(s) for key, s in sorted(self._seen.items())}

    def as_dict(self) -> dict:
        return {"coverage_unique_couples": self.coverage()}


# ---------------------------------------------------------------------------
# Interactions log (§12: every target interaction counted)
# ---------------------------------------------------------------------------

class InteractionsLog:
    """Tamper-evident (mineur b): each entry carries a sha256 chain seal —
    entry_i.seal = sha256(entry_{i-1}.seal + canonical json of entry_i)."""

    def __init__(self, path: str = "artifacts/v1/interactions-log.json"):
        self.path = path
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        self.entries: list = []
        if os.path.exists(path):
            self.entries = json.load(open(path))
            self.verify()

    def _seal_of(self, entry: dict, prev_seal: str) -> str:
        import hashlib
        payload = json.dumps({**entry, "prev": prev_seal}, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def log(self, motive: str, arm: str = "-", target: str = "SIW-dev") -> None:
        prev = self.entries[-1]["seal"] if self.entries else "genesis"
        entry = {"i": len(self.entries), "t": time.strftime("%H:%M:%S"),
                 "motive": motive, "arm": arm, "target": target}
        entry["seal"] = self._seal_of(entry, prev)
        self.entries.append(entry)
        with open(self.path, "w") as fh:
            json.dump(self.entries, fh, indent=1)

    def verify(self) -> bool:
        prev = "genesis"
        for e in self.entries:
            if self._seal_of({k: v for k, v in e.items() if k != "seal"}, prev) != e.get("seal"):
                raise RuntimeError("InteractionsLog seal BROKEN — tampering detected")
            prev = e["seal"]
        return True

    @property
    def count(self) -> int:
        return len(self.entries)


# ---------------------------------------------------------------------------
# Fine-tune regime (frozen §3)
# ---------------------------------------------------------------------------

@dataclass
class FinetuneConfig:
    updates: int = 2_000
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    clip_norm: float = 1.0
    # checkpoint at FIXED budget; NO selection on target performance (§12.4)
    checkpoint_policy: str = "final_at_fixed_budget"
    seed: int = 0


# ---------------------------------------------------------------------------
# Arms factory (frozen §2): all arms share the B144-for-SIW architecture;
# only the initialization of the TRANSFERRED CORE differs.
# ---------------------------------------------------------------------------

def build_arm(arm: str, siw_model_factory, canon_checkpoint: str | None,
              control_checkpoint: str | None = None, fresh_init=None,
              seed: int | None = None):
    """siw_model_factory() → fresh SIW-adapted model (new vocab embeddings,
    B144 core shape). Arm semantics (FREEZE.md §2):
      scratch       → nothing loaded
      pretrained-TGK→ message-passing blocks + scorer head loaded from canon B144
      control-nontarget → same but from the non-informative-source checkpoint
    """
    import mlx.core as mx
    import mlx.nn as nn

    if seed is not None:
        mx.random.seed(seed)  # deterministic factory (audit B2)
    model = siw_model_factory()
    if fresh_init is not None:
        # SHARED fresh init across arms (freeze §12.4: 'mêmes nouveaux
        # embeddings/têtes initialisés') — the pretrained-vs-scratch contrast
        # must not confound transfer with init luck (audit B2)
        model.update(nn.utils.tree_unflatten(list(fresh_init)))
    if arm == "scratch":
        mx.eval(model.parameters())
        model.transfer_manifest = {"arm": "scratch", "loaded": [], "skipped": "n/a (no source)"}
        return model
    ckpt = canon_checkpoint if arm == "pretrained-TGK" else control_checkpoint
    if ckpt is None:
        raise ValueError(f"arm {arm!r} requires its source checkpoint")
    source = dict(mx.load(ckpt))
    own = dict(nn.utils.tree_flatten(model.parameters()))
    # FILTERED transfer (audit M1): keep only names present in the target with
    # compatible shapes; publish the loaded/skipped manifest (auditability).
    filtered, loaded, skipped = [], [], []
    for name, arr in source.items():
        if name in own and tuple(own[name].shape) == tuple(arr.shape):
            filtered.append((name, arr))
            loaded.append(name)
        else:
            reason = "shape" if name in own else "name"
            skipped.append({"name": name, "reason": reason,
                            "src_shape": list(arr.shape),
                            "dst_shape": list(own[name].shape) if name in own else None})
    tree = nn.utils.tree_unflatten(sorted(filtered, key=lambda kv: kv[0]))
    model.update(tree)
    mx.eval(model.parameters())
    model.transfer_manifest = {"arm": arm, "source": ckpt,
                               "loaded": sorted(loaded), "n_loaded": len(loaded),
                               "skipped": skipped, "n_skipped": len(skipped)}
    return model


ARM_NAMES = ("scratch", "pretrained-TGK", "control-nontarget")
