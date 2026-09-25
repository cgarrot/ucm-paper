#!/usr/bin/env python3
"""Build the OFFICIAL V1-bis freeze (v7+) — reproducible, no heredocs.

Reads inputs from committed artifacts (gen-gate v02 couples, store v2,
checkpoint dirs), builds the freeze with the FULL BOUND_CODE closure
(35 modules incl. the official-path six), writes it O_EXCL to a new path
plus the append-only .sha256 pointer, and prints the pasted-checklist
verification (module list + count + canonical hash) for the announcement.

Usage: .venv/bin/python scripts/build_freeze_v1bis_official.py <out.json>
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ucm.v1.freeze_v1bis import build_freeze_v1bis_official, BOUND_CODE

REPO = Path(__file__).resolve().parents[1]


def _sha(p: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    out = sys.argv[1]
    if os.path.exists(out):
        raise SystemExit(f"refusing to overwrite: {out}")

    # inputs from committed artifacts (single sources of truth)
    gate = json.loads((REPO / "artifacts/dev-v1bis-gen-gate-v02.json").read_text())
    couples = [{"path": str(REPO / f["path"]),
                "sha256": f["sha256_full"],
                "gen_seed": f["seed"],
                "status": ("replacement_of_%d" % f["replacement_of"])
                if f.get("is_replacement") else "original"}
               for f in gate["files"]]
    assert len(couples) == 10, f"expected 10 couples files, got {len(couples)}"

    store_path = str(REPO / "artifacts/v1bis-gen-store-v02.json")
    store_sha = _sha(store_path)

    def _arm(kind: str, seeds) -> list[dict]:
        pref = {"canon": "gate2c-B144", "validity": "v1-control",
                "null": "v1-null"}[kind]
        role = {"canon": "canon", "validity": "control", "null": "control"}[kind]
        out_l = []
        for s in seeds:
            cands = sorted((REPO / "artifacts").glob(f"????????-??????-{pref}-s{s}"))
            if not cands:
                raise SystemExit(f"no checkpoint dir for {pref} s{s}")
            d = cands[-1]
            fin = json.loads((d / "final.json").read_text())
            out_l.append({"path": str(d / "checkpoint.npz"),
                          "sha256": _sha(str(d / "checkpoint.npz")),
                          "checkpoint_policy": "best_validation",
                          "source_run_updates_run": fin["updates_run"],
                          "loaded_checkpoint_update": fin["best"]["update"],
                          "role": role, "seed": s})
        return out_l

    seeds = list(range(10))
    canon = _arm("canon", seeds)
    validity = _arm("validity", seeds)
    null = _arm("null", seeds)

    canon_hash = build_freeze_v1bis_official(
        out,
        couples_files=couples, store_path=store_path, store_sha=store_sha,
        k_plan=[64, 128, 256], seeds=seeds, updates=2000, eval_seed=50000,
        canon_ckpts=canon, control_validity_ckpts=validity,
        control_null_ckpts=null)

    # append-only pointer
    ptr = out.replace(".json", ".sha256")
    if os.path.exists(ptr):
        raise SystemExit(f"pointer exists: {ptr}")
    Path(ptr).write_text(canon_hash + "\n")

    # pasted checklist for the announcement
    fm = json.loads(Path(out).read_text())
    mods = sorted(fm["code"])
    need = ["runner_official", "runner_parallel", "seal_test2",
            "test2_generation", "interactions_merge", "verdict_v1bis"]
    print(f"CHECKLIST v-next | version: {fm.get('version')} | modules: {len(mods)}")
    print("  nouveaux:", {n: any(n in m for m in mods) for n in need})
    print(f"  canon: {canon_hash[:16]}… | pointer: {ptr}")


if __name__ == "__main__":
    main()
