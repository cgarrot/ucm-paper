"""Freeze-manifest DEV for runner_v1bis (lead 14:05) — Stage A binding.

Binds BEFORE any run: SHA-256 of all code modules + input SHAs + k-plan +
train seed. The manifest is COMMITTED as freeze-v1bis.json; its canonical
SHA is published as --protocol-hash. At startup, drift in ANY bound SHA
⇒ abort BEFORE any data file read.

Adapted from runner_confirm's approach but SEPARATE paths — no touching
runner_confirm.
"""

from __future__ import annotations

import hashlib
import json
import os

# Code modules bound by the freeze (runner + episode_budget + adapter + publish + reader + generators)
BOUND_CODE = [
    "ucm/v1/runner_v1bis.py",
    "ucm/v1/episode_budget.py",
    "ucm/v1/data_adapter.py",
    "ucm/v1/atomic_publish.py",
    "ucm/v1/sealed_reader.py",
    "ucm/data/siw_pipeline.py",
    "ucm/data/rstar_order.py",
    "ucm/env/siw.py",          # tagi-5 finding 2
    "ucm/env/siw_oracle.py",
    "ucm/model/siw_model.py",
    "ucm/v1/transfer.py",      # finetune/build_arm semantics
    "ucm/v1/runner.py",        # finetune imported from here (14:18 fix 3)
    "ucm/model/gnn_b.py",
    "ucm/model/deepsets_a.py",
    # full finetune closure (14:25 fix B)
    "ucm/v1/tensorize_siw.py",  # imported by runner.py's finetune path
    "ucm/model/loss.py",       # set_bc_loss used in finetune
    "ucm/model/train.py",      # _grad_global_norm used in finetune
    "ucm/model/tensorize.py",  # collate used by tensorize_siw
    "ucm/model/fixtures.py",   # vocab constants used by tensorize
    # eval closure (14:27)
    "ucm/eval/rollout.py",     # run_episode, ModelPolicy
    "ucm/eval/metrics.py",     # summarize, paired_hierarchical_bootstrap
    "ucm/v1/policy_siw.py",    # SIWModelPolicy
    # TGK fixtures deps — declared, not deferred (14:27)
    "ucm/data/generate.py",
    "ucm/env/oracle.py",
    "ucm/env/tinygraph.py",
    # residual closure (14:31)
    "ucm/data/schema.py",        # imported by generate.py
    "ucm/eval/baselines.py",     # imported by control_source valid_actions
    "ucm/v1/control_source.py",  # control arm labels
    "ucm/v1/freeze_v1bis.py",    # self-bound (14:31)
    # official-path closure (revue 4 item 4, lead 18:45 — v5/v6 l'avaient omis)
    "ucm/v1/runner_official.py",
    "ucm/v1/runner_parallel.py",
    "ucm/v1/seal_test2.py",
    "ucm/data/test2_generation.py",
    "ucm/data/interactions_merge.py",
    "ucm/data/verdict_v1bis.py",
]

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _rp(rel: str) -> str:
    return rel if os.path.isabs(rel) else os.path.join(_REPO_ROOT, rel)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_freeze_v1bis(freeze_path: str, *, couples_path: str, couples_sha: str,
                        store_path: str, store_sha: str,
                        k_plan: list[int], seed: int, updates: int) -> str:
    """Build + write freeze-v1bis.json. Returns the canonical SHA to publish
    as --protocol-hash. Validates: source files exist, SHAs are hex64,
    k_plan is positive ints, code modules readable."""
    for v, name in ((couples_sha, "couples_sha"), (store_sha, "store_sha")):
        if not (isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v)):
            raise SystemExit(f"freeze-v1bis: {name} not sha256 hex64: {v!r}")
    if not all(isinstance(k, int) and k > 0 for k in k_plan):
        raise SystemExit(f"freeze-v1bis: k_plan must be positive ints: {k_plan}")
    fm = {
        "version": 1,
        "code": {rel: _sha256_file(_rp(rel)) for rel in BOUND_CODE},
        "inputs": {
            "couples": {"path": os.path.abspath(couples_path), "sha256": couples_sha},
            "store": {"path": os.path.abspath(store_path), "sha256": store_sha},
        },
        "protocol": {
            "k_plan": sorted(k_plan),
            "k_plan_exec_order": list(k_plan),  # exact execution order (14:18 fix 2)
            "seed": seed,
            "updates": updates,
            "runner": "v1bis-dev",
        },
    }
    # O_EXCL: never overwrite a published freeze
    fd = os.open(freeze_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as fh:
        json.dump(fm, fh, indent=2, sort_keys=True)
    return hashlib.sha256(json.dumps(fm, sort_keys=True, default=str).encode()).hexdigest()


def verify_freeze_v1bis(freeze_path: str, protocol_hash: str, *,
                        couples_sha: str | None = None,
                        store_sha: str | None = None,
                        k_plan: list[int] | None = None,
                        seed: int | None = None,
                        updates: int | None = None,
                        # official schema (v2) cross-checks
                        couples_files: list[dict] | None = None,
                        arms: dict | None = None) -> dict:
    """Verify at startup: canonical SHA matches, code SHAs match current
    files (drift ⇒ abort), AND input/protocol args cross-checked against
    the manifest (tagi-5 finding 1). Returns the freeze manifest."""
    if not os.path.exists(freeze_path):
        raise SystemExit(f"freeze-v1bis: manifest missing: {freeze_path}")
    fm = json.load(open(freeze_path))
    canon = hashlib.sha256(json.dumps(fm, sort_keys=True, default=str).encode()).hexdigest()
    if canon != protocol_hash:
        raise SystemExit(f"freeze-v1bis: protocol hash mismatch — canonical "
                         f"{canon[:12]}… != published {protocol_hash[:12]}…")
    for rel, declared in fm.get("code", {}).items():
        p = _rp(rel)
        if not os.path.exists(p):
            raise SystemExit(f"freeze-v1bis: code drift in {rel} (MISSING)")
        actual = _sha256_file(p)
        if actual != declared:
            raise SystemExit(f"freeze-v1bis: code drift in {rel} "
                             f"({actual[:12]}… != {declared[:12]}…)")
    # Cross-check inputs and protocol against caller args (finding 1)
    if couples_sha is not None:
        fm_cs = fm["inputs"]["couples"]["sha256"]
        if fm_cs != couples_sha:
            raise SystemExit(f"freeze-v1bis: couples SHA disagrees — manifest "
                             f"{fm_cs[:12]}… vs run {couples_sha[:12]}…")
    if store_sha is not None:
        fm_ss = fm["inputs"]["store"]["sha256"]
        if fm_ss != store_sha:
            raise SystemExit(f"freeze-v1bis: store SHA disagrees — manifest "
                             f"{fm_ss[:12]}… vs run {store_sha[:12]}…")
    if k_plan is not None:
        fm_kp = fm["protocol"]["k_plan"]
        fm_kp_order = fm["protocol"].get("k_plan_exec_order", fm_kp)
        if fm_kp_order != list(k_plan):
            raise SystemExit(f"freeze-v1bis: k_plan EXECUTION ORDER disagrees — "
                             f"manifest {fm_kp_order} vs run {list(k_plan)}")
    if seed is not None and fm["protocol"]["seed"] != seed:
        raise SystemExit(f"freeze-v1bis: seed disagrees — manifest "
                         f"{fm['protocol']['seed']} vs run {seed}")
    if updates is not None and fm["protocol"]["updates"] != updates:
        raise SystemExit(f"freeze-v1bis: updates disagrees — manifest "
                         f"{fm['protocol']['updates']} vs run {updates}")

    # Official schema (v2) cross-checks (lead 15:34)
    if fm.get("version", 1) >= 2:
        # 10 couples_files: path + sha256
        if couples_files is not None:
            fm_cf = fm["inputs"].get("couples_files", [])
            if len(fm_cf) != len(couples_files):
                raise SystemExit(f"freeze-v1bis: couples_files count — manifest "
                                 f"{len(fm_cf)} vs run {len(couples_files)}")
            for i, (m, r) in enumerate(zip(fm_cf, couples_files)):
                if m.get("sha256") != r.get("sha256"):
                    raise SystemExit(f"freeze-v1bis: couples_files[{i}].sha256 — "
                                     f"manifest {m.get('sha256', '?')[:12]}… vs "
                                     f"run {r.get('sha256', '?')[:12]}…")
        # arms checkpoints: count + all SHAs present + valid
        if arms is not None:
            for arm_name, arm_run in arms.items():
                fm_arm = fm["arms"].get(arm_name)
                if fm_arm is None:
                    raise SystemExit(f"freeze-v1bis: arm {arm_name!r} not in manifest")
                fm_ck = fm_arm.get("checkpoints", [])
                run_ck = arm_run.get("checkpoints", [])
                if len(fm_ck) != len(run_ck):
                    raise SystemExit(f"freeze-v1bis: arm {arm_name} checkpoint count — "
                                     f"manifest {len(fm_ck)} vs run {len(run_ck)}")
                for j, (m, r) in enumerate(zip(fm_ck, run_ck)):
                    if m.get("sha256") != r.get("sha256"):
                        raise SystemExit(f"freeze-v1bis: arm {arm_name}[{j}].sha256 — "
                                         f"manifest {m.get('sha256', '?')[:12]}… vs "
                                         f"run {r.get('sha256', '?')[:12]}…")
    return fm


# ---------------------------------------------------------------------------
# Official V1-bis freeze (lead 14:53): 4 arms × 10 seeds, 10 couples files
# ---------------------------------------------------------------------------

def build_freeze_v1bis_official(freeze_path: str, *,
                                 couples_files: list[dict],  # [{path, sha256, gen_seed, status}]
                                 store_path: str, store_sha: str,
                                 k_plan: list[int],
                                 seeds: list[int], updates: int,
                                 eval_seed: int,
                                 canon_ckpts: list[dict],   # 10 entries G2-style
                                 control_validity_ckpts: list[dict],  # 10 entries
                                 control_null_ckpts: list[dict]  # 10 entries
                                 ) -> str:
    """Build freeze for the OFFICIAL V1-bis run: 4 arms × 10 seeds.
    All arms share the SAME target data; difference = initialization only.

    STRICT OFFICIAL MODE (lead 15:39/15:43): REFUSES any deviation from the
    pre-registered protocol — test fixtures can NEVER feed the official path."""
    # STRICT GUARDS
    if len(couples_files) != 10:
        raise SystemExit(f"OFFICIAL: couples_files must be exactly 10, got {len(couples_files)}")
    if sorted(k_plan) != [64, 128, 256]:
        raise SystemExit(f"OFFICIAL: k_plan must be [64,128,256] episodes, got {k_plan}")
    if sorted(seeds) != list(range(10)):
        raise SystemExit(f"OFFICIAL: seeds must be 0-9, got {seeds}")
    if updates != 2000:
        raise SystemExit(f"OFFICIAL: updates must be 2000, got {updates}")
    for i, cf in enumerate(couples_files):
        for field in ("path", "sha256", "gen_seed", "status"):
            if field not in cf:
                raise SystemExit(f"OFFICIAL: couples_files[{i}] missing {field}")
        if not isinstance(cf["gen_seed"], int):
            raise SystemExit(f"OFFICIAL: couples_files[{i}].gen_seed must be int")
    # validate SHAs hex64
    for v, name in [(store_sha, "store_sha")] +                    [(cf["sha256"], f"couples[{i}].sha256") for i, cf in enumerate(couples_files)]:
        if not (isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v)):
            raise SystemExit(f"freeze-v1bis-official: {name} not sha256 hex64")
    # validate arm checkpoint lists
    for arm_name, ckpts in [("canon", canon_ckpts),
                             ("control_validity", control_validity_ckpts),
                             ("control_null", control_null_ckpts)]:
        if len(ckpts) != len(seeds):
            raise SystemExit(f"freeze-v1bis-official: {arm_name} has {len(ckpts)} "
                             f"checkpoints, expected {len(seeds)} (one per seed)")
        for e in ckpts:
            for field in ("path", "sha256", "checkpoint_policy",
                          "source_run_updates_run", "loaded_checkpoint_update",
                          "role", "seed"):
                if field not in e:
                    raise SystemExit(f"freeze-v1bis-official: {arm_name} entry missing {field}")
            if not os.path.isabs(e["path"]):
                raise SystemExit(f"freeze-v1bis-official: {arm_name} relative path forbidden")
    fm = {
        "version": 2,
        "code": {rel: _sha256_file(_rp(rel)) for rel in BOUND_CODE},
        "inputs": {
            "couples_files": couples_files,
            "store": {"path": os.path.abspath(store_path), "sha256": store_sha},
            "eval_config": {"stage": "B-placeholder", "note": "hash test2 published before run"},
        },
        "arms": {
            "scratch": {"init": "fresh"},
            "pretrained_TGK": {"checkpoints": canon_ckpts},
            "control_validity": {"checkpoints": control_validity_ckpts,
                                  "note": "reuses v1-control (uniform among valid)"},
            "control_null": {"checkpoints": control_null_ckpts,
                              "note": "uniform among ALL candidates (lead 14:53 ruling)"},
        },
        "protocol": {
            "k_plan": sorted(k_plan),
            "k_plan_exec_order": list(k_plan),
            "seeds": seeds,
            "updates": updates,
            "batch": 64,  # erratum 20df839: binding auto-contenu
            "eval_seed": eval_seed,
        },
        "asymmetry_declaration": (
            "canon s0-s4 were gated on test_g1 (one-read consumed in V0); "
            "canon s5-s9 validated on VALIDATION split only; "
            "test_g1 NEVER re-opened; checkpoint_policy identical (best_validation)"),
    }
    fd = os.open(freeze_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "w") as fh:
        json.dump(fm, fh, indent=2, sort_keys=True)
    return hashlib.sha256(json.dumps(fm, sort_keys=True, default=str).encode()).hexdigest()
