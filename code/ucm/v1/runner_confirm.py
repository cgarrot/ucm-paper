"""CONFIRMATION runner v4 — validation par PRÉDICATS, manifeste source
pré-open, binding 2 étapes (NO-GO 21:51/23:17, plan v4.9 M1+G2+G3+G4).

Stage A (pre-generation freeze) — --protocol-hash binds:
    code sha256 (runner_confirm.py) + source manifest sha256 (10 checkpoints
    canon/control: paths, sha64, updates) + couples file + exclusion store.
Stage B (dataset stage) — --test2-hash: sha64 of the generated sealed test2,
published BEFORE the run, verified at read_once (mismatch → abort; the
pre-generation protocol cannot know it).

Pre-open guarantee (G2): the SOURCE manifest is fully verified (files exist,
sha64 match, updates proven from final.json) BEFORE the test2 open — a missing
or corrupt checkpoint aborts WITHOUT consuming the unique read (spy-tested).

Validation M1: exactly 600 episodes; exactly 150 per predicate
(VIEW/SET/CHOOSE/SUBMITTED) via task.goal.predicate; layout_hash RECOMPUTED
from layout_spec (SIWLayout) must match; layouts disjoint from the adaptation
pool; NO per-layout quota (variable counts are the rule; a uniform 4/layout
file that also balances predicates is licit and must pass).

G3: raw goal_type = task.goal.predicate (injected at parse time).

All v3/v4.6 guarantees retained: failfast before any side effect, staging
private 0700 + atomic publish, no numbers before seal, raw-before-derived,
CI recomputed from raw, train_seed/eval_seed separated, zero tmp files.
"""

from __future__ import annotations

import argparse
import fcntl
import glob
import hashlib
import inspect
import json
import os

import mlx.core as mx
import mlx.nn as nn

mx.set_default_device(mx.cpu)

from ucm.eval import metrics as M
from ucm.eval.rollout import EpisodeResult
from ucm.model.siw_model import make_siw_model
from ucm.v1.data_adapter import couples_lines_to_records
from ucm.v1.runner import _build_env, evaluate, finetune
from ucm.v1.sealed_reader import sealed_registry
from ucm.v1.transfer import CoverageTracker, FinetuneConfig, InteractionsLog, build_arm

K_STAR = 500
CONFIRM_ARMS = ("scratch", "pretrained-TGK", "control-nontarget")
EXPECTED_TOTAL = 600
EXPECTED_PER_PREDICATE = 150
PREDICATES = ("VIEW", "SET", "CHOOSE", "SUBMITTED")


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _code_sha() -> str:
    """Self-hash of this module (G4: code changes break the protocol)."""
    with open(__file__, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _verify_source_manifest(manifest_path: str) -> dict:
    """G2 v2: verify ALL source checkpoints BEFORE any sealed open, with
    EXPLICIT best-vs-final attribution (lead 01:31): each entry declares
    checkpoint_policy, source_run_updates_run and loaded_checkpoint_update;
    we verify against final.json (updates_run, best.update, checkpoint_policy)
    and the path suffix (.npz = selected checkpoint; -final.npz = final)."""
    man = json.load(open(manifest_path))
    entries = man.get("checkpoints", man if isinstance(man, list) else [])
    if len(entries) != 10:
        raise SystemExit(f"source manifest: expected 10 checkpoints, got {len(entries)}")
    binding: dict[str, str] = {}
    for e in entries:
        p = e["path"]
        if not os.path.isabs(p):
            raise SystemExit(f"source manifest: RELATIVE path forbidden "
                             f"(producer must commit absolute read-only): {p!r}")
        for field in ("sha256", "checkpoint_policy", "source_run_updates_run",
                      "loaded_checkpoint_update", "role", "seed"):
            if field not in e:
                raise SystemExit(f"source manifest entry missing {field}: {p}")
        if not os.path.exists(p):
            raise SystemExit(f"source checkpoint MISSING: {p} (abort before any read)")
        if _sha256_file(p) != e["sha256"]:
            raise SystemExit(f"source checkpoint sha MISMATCH: {p}")
        fin = json.load(open(os.path.join(os.path.dirname(p), "final.json")))
        if fin.get("updates_run") != e["source_run_updates_run"]:
            raise SystemExit(f"{p}: updates_run {fin.get('updates_run')} != "
                             f"declared {e['source_run_updates_run']}")
        if fin.get("checkpoint_policy") != e["checkpoint_policy"]:
            raise SystemExit(f"{p}: checkpoint_policy {fin.get('checkpoint_policy')} "
                             f"!= declared {e['checkpoint_policy']}")
        if e["checkpoint_policy"] == "best_validation":
            # checkpoint.npz holds the BEST-validation params → attribution is
            # best.update, NOT updates_run
            if p.endswith("-final.npz"):
                raise SystemExit(f"{p}: best_validation policy selects checkpoint.npz, "
                                 f"not -final.npz")
            if fin.get("best", {}).get("update") != e["loaded_checkpoint_update"]:
                raise SystemExit(f"{p}: loaded ckpt = best.update "
                                 f"{fin.get('best', {}).get('update')} != declared "
                                 f"{e['loaded_checkpoint_update']}")
        elif e["checkpoint_policy"] == "final":
            if not p.endswith("-final.npz"):
                raise SystemExit(f"{p}: final policy selects -final.npz")
            if fin.get("updates_run") != e["loaded_checkpoint_update"]:
                raise SystemExit(f"{p}: loaded ckpt = updates_run mismatch")
        role, seed_s = e.get("role"), e.get("seed")
        if role not in ("canon", "control") or seed_s not in range(5):
            raise SystemExit(f"{p}: invalid role/seed binding "
                             f"({role!r}, {seed_s!r})")
        key = f"{role}|s{seed_s}"
        if key in binding:
            raise SystemExit(f"duplicate checkpoint binding: {key}")
        binding[key] = p
    if len(binding) != 10:
        raise SystemExit(f"source manifest: {len(binding)}/10 role×seed bindings")
    return binding


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_BOUND_CODE = ("ucm/v1/runner_confirm.py", "ucm/v1/runner.py", "ucm/v1/transfer.py",
               "ucm/v1/policy_siw.py", "ucm/v1/tensorize_siw.py", "ucm/v1/data_adapter.py",
               "ucm/v1/sealed_reader.py", "ucm/model/siw_model.py", "ucm/model/tensorize.py",
               "ucm/model/loss.py", "ucm/model/gnn_b.py",
               "ucm/eval/metrics.py", "ucm/eval/rollout.py",
               "ucm/env/siw.py", "ucm/env/siw_oracle.py", "ucm/data/siw_pipeline.py")


def _rp(rel: str) -> str:
    """Repo-root-anchored path (tests run from tmp dirs)."""
    return rel if os.path.isabs(rel) else os.path.join(_REPO_ROOT, rel)


def _freeze_manifest_sha(fm_path: str) -> tuple[str, dict]:
    """G4 v2 (lead 01:32): the COMMITTED freeze-manifest.json binds EVERYTHING
    (code snapshot SHAs of all runner/generator/analysis modules, source
    manifest + sha, couples CONTENT sha64, exclusion store sha, generation
    params+seeds+distribution, test2 data-manifest method+path pattern,
    protocol constants). We verify each bound code SHA against the CURRENT
    files (any drift → abort) and return the canonical sha of the manifest."""
    fm = json.load(open(fm_path))
    for rel, declared in fm.get("code", {}).items():
        p = _rp(rel)
        if not os.path.exists(p):
            raise SystemExit(f"freeze manifest: code drift in {rel} (MISSING) — re-freeze")
        actual = _sha256_file(p)
        if actual != declared:
            raise SystemExit(f"freeze manifest: code drift in {rel} "
                             f"({actual[:12]}… != {declared[:12]}…) — re-freeze required")
    for key, spec in fm.get("inputs", {}).items():
        if key == "couples":
            # SEALED: NEVER opened here (one-read incident 02:31) — frozen sha
            # binds the protocol; compared by STRING in _failfast; the ONLY
            # open/hash happens in reg.read_once AFTER the intent log.
            continue
        if not os.path.exists(spec["path"]):
            raise SystemExit(f"freeze manifest: {key} input MISSING ({spec['path']})")
        if _sha256_file(spec["path"]) != spec["sha256"]:
            raise SystemExit(f"freeze manifest: {key} content changed ({spec['path']})")
    canon = json.dumps(fm, sort_keys=True, default=str)
    return hashlib.sha256(canon.encode()).hexdigest(), fm


_HEX64 = __import__("re").compile(r"^[0-9a-f]{64}$")


def _require_hex64(value: str, what: str) -> None:
    """B3bis: expected hashes must be lowercase hex64 — 64 arbitrary chars
    (e.g. 'Z'*64) are NOT valid sha256; validating BEFORE the open prevents
    burning the unique read on a garbage hash (lead 06:02)."""
    if not isinstance(value, str) or not _HEX64.match(value):
        raise SystemExit(f"failfast: {what} is not a valid sha256 hex64: {value!r}")


def _failfast(args) -> str:
    """Stage A binding v2 — freeze-manifest anchored, ALL validation BEFORE
    any side effect. --protocol-hash must equal the canonical sha of the
    COMMITTED freeze manifest (after live code/input verification)."""
    missing = [f for f in ("protocol_hash", "couples_hash", "test2_hash",
                           "couples_file", "test2_file", "freeze_manifest",
                           "out_dir") if not getattr(args, f, None)]
    if missing:
        raise SystemExit(f"failfast: missing required args {missing}")
    # B3bis: ALL expected hashes validated as hex64 BEFORE any file open
    _require_hex64(args.protocol_hash, "--protocol-hash")
    _require_hex64(args.couples_hash, "--couples-hash")
    _require_hex64(args.test2_hash, "--test2-hash")
    # cheap existence guards FIRST — a second process (staging exists) must
    # refuse WITHOUT reading any sealed CONTENT (spy-tested)
    if os.path.exists(args.out_dir) or os.path.exists(args.out_dir + ".staging"):
        raise SystemExit(f"failfast: output/staging for {args.out_dir} EXISTS — "
                         f"exclusive runs only, no silent replay")
    if not os.path.exists(args.freeze_manifest):
        raise SystemExit(f"failfast: freeze manifest missing: {args.freeze_manifest}")
    fm_sha, fm = _freeze_manifest_sha(args.freeze_manifest)
    if fm_sha != args.protocol_hash:
        raise SystemExit(f"failfast: protocol hash mismatch — freeze manifest "
                         f"canonical {fm_sha} != published {args.protocol_hash}")
    # CLI invariants must AGREE with the frozen manifest
    if fm["protocol"]["k_star"] != K_STAR or fm["protocol"]["updates"] != args.updates:
        raise SystemExit("failfast: CLI disagrees with frozen protocol constants")
    if os.path.realpath(fm["inputs"]["couples"]["path"]) != os.path.realpath(args.couples_file):
        raise SystemExit("failfast: couples path disagrees with freeze manifest")
    _require_hex64(fm["inputs"]["couples"]["sha256"], "freeze-manifest couples sha")
    if fm["inputs"]["couples"]["sha256"] != args.couples_hash:
        raise SystemExit("failfast: couples CONTENT sha disagrees with freeze manifest")
    dm_spec = fm["generation"]["test2_data_manifest"]
    dm_path = dm_spec["manifest_path"]
    if not os.path.exists(dm_path):
        raise SystemExit(f"failfast: data manifest missing: {dm_path}")
    dm = json.load(open(dm_path))
    # B3 hardening (lead 05:10): STRICT path match (convention: the freeze
    # stores the path EXACTLY as the writer returned it; the CLI must pass
    # the same string) and STRICT method key — no fallbacks.
    dm_declared_path = dm.get("path")
    if dm_declared_path is None:
        raise SystemExit("failfast: data manifest missing 'path' (writer contract)")
    _norm = lambda p: os.path.normpath(os.path.abspath(p))
    if _norm(dm_declared_path) != _norm(args.test2_file):
        raise SystemExit(f"failfast: data manifest path {dm_declared_path!r} != "
                         f"CLI --test2-file {args.test2_file!r}")
    method = dm_spec["method"]
    if method not in dm:
        raise SystemExit(f"failfast: data manifest lacks method key {method!r}")
    attested = dm[method]
    _require_hex64(attested, "data-manifest attested hash")
    if attested != args.test2_hash:
        raise SystemExit(f"failfast: --test2-hash not attested by data manifest "
                         f"({attested} != {args.test2_hash})")
    if os.path.realpath(fm["inputs"]["exclusion_store"]["path"]) != os.path.realpath(args.layout_store):
        raise SystemExit("failfast: layout store disagrees with freeze manifest")
    # G2 v3 (02:32): the 10 VERIFIED paths are BOUND to the run — the cell
    # loop uses THESE EXACT paths (role+seed → path), never a glob.
    ckpt_paths = _verify_source_manifest(fm["inputs"]["source_manifest"]["path"])
    if len(ckpt_paths) != 10:
        raise SystemExit(f"failfast: expected 10 bound checkpoints, got {len(ckpt_paths)}")
    if _sha256_file(fm["inputs"]["source_manifest"]["path"]) != \
            fm["inputs"]["source_manifest"]["sha256"]:
        raise SystemExit("failfast: source manifest content drift")
    args._bound_ckpts = ckpt_paths
    return fm_sha


def _recompute_layout_hash(layout_spec: dict) -> str:
    """Writer rule: SIWLayout(spec).layout_hash()."""
    from ucm.env.siw import SIWLayout
    spec = dict(layout_spec)
    if isinstance(spec.get("widgets"), dict):
        spec["widgets"] = list(spec["widgets"].values())
    return SIWLayout(spec).layout_hash()


def _parse_test2_lines(lines: list[str], adaptation_layouts: set) -> list[dict]:
    """M1: predicate-balanced validation, NO layout quota; G3 goal_type."""
    eps = []
    per_pred = {p: 0 for p in PREDICATES}
    layouts_used = set()
    for line in lines:
        e = json.loads(line)
        need = {"episode_id", "layout_spec", "task", "d_star", "layout_hash"}
        if not need <= set(e):
            raise RuntimeError(f"test2 line not in writer contract: keys={sorted(e)}")
        # layout_hash must be RECOMPUTABLE from the embedded spec
        if _recompute_layout_hash(e["layout_spec"]) != e["layout_hash"]:
            raise RuntimeError(f"layout_hash mismatch for {e['episode_id']}")
        pred = e["task"]["goal"]["predicate"]
        if pred not in per_pred:
            raise RuntimeError(f"unknown predicate {pred}")
        per_pred[pred] += 1
        layouts_used.add(e["layout_hash"])
        eps.append({**e, "goal_type": pred})   # G3: inject for the raw
    if len(eps) != EXPECTED_TOTAL:
        raise RuntimeError(f"test2: expected {EXPECTED_TOTAL} episodes, got {len(eps)}")
    for p, n in per_pred.items():
        if n != EXPECTED_PER_PREDICATE:
            raise RuntimeError(f"test2 predicate {p}: {n} != {EXPECTED_PER_PREDICATE}")
    overlap = layouts_used & adaptation_layouts
    if overlap:
        raise RuntimeError(f"test2 layouts OVERLAP the adaptation pool: {len(overlap)}")
    # NOTE (nuance lead 23:19): NO per-layout quota — variable counts are the
    # rule; uniform 4/layout is licite if predicates balance.
    return eps


def _results_from_raw(raw_path: str) -> dict[str, list[EpisodeResult]]:
    by_arm: dict[str, list[EpisodeResult]] = {}
    for line in open(raw_path):
        r = json.loads(line)
        er = EpisodeResult(
            episode_id=r["episode_id"], seed=r["train_seed"],
            policy_name=f"{r['arm']}@k={r['k']}", success=r["success"],
            outcome=r["outcome"], length=r["length"], n_invalid=r["n_invalid"],
            goal_reached_without_stop=r["goal_reached_without_stop"],
            d_star=r["d_star"], L_star=r["L_star"], layout_id=r["layout_hash"],
            goal_type=r.get("goal_type", ""))
        by_arm.setdefault(r["arm"], []).append(er)
    return by_arm


def main(args):
    protocol_hash = _failfast(args)
    staging = args.out_dir.rstrip("/") + ".staging"
    os.makedirs(staging, exist_ok=False)
    os.chmod(staging, 0o700)
    lock_fh = open(os.path.join(staging, ".lock"), "w")
    fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ilog = InteractionsLog(os.path.join(staging, "interactions-log.json"))
    reg = sealed_registry()
    _buffered = []

    ilog.log(f"CONFIRM intent: adaptation couples ONE READ ({args.couples_file})",
             arm="-", target="SIW-adaptation")
    _fsync(ilog)
    fmt, lines, full = reg.read_once(args.couples_file, expected_hash=args.couples_hash)
    assert fmt == "couples", f"unexpected couples format {fmt}"
    records = couples_lines_to_records(lines, args.layout_store)
    adaptation_layouts = {r["provenance"]["layout_hash"] for r in records}
    ilog.log(f"couples read: {len(records)} records sha256={full}", arm="-")

    ilog.log(f"CONFIRM intent: SEALED TEST2 ONE READ ({args.test2_file})",
             arm="all", target="SIW-test2")
    _fsync(ilog)
    fmt2, lines2, full2 = reg.read_once(args.test2_file, expected_hash=args.test2_hash)
    test_eps = _parse_test2_lines(lines2, adaptation_layouts)
    ilog.log(f"test2 read: {len(test_eps)} episodes sha256={full2} "
             f"(predicates 150×4, {len({e['layout_hash'] for e in test_eps})} layouts)",
             arm="all")

    cfg = FinetuneConfig(updates=args.updates)
    cov = CoverageTracker()
    raw_path = os.path.join(staging, "confirm-episodes-raw.jsonl")
    raw_fh = open(raw_path, "w")
    ckpt_manifest = {}
    for seed in range(5):
        mx.random.seed(9000 + seed)
        base_tree = nn.utils.tree_flatten(make_siw_model().parameters())
        cck = args._bound_ckpts[f"canon|s{seed}"]   # manifest-bound EXACT path
        kck = args._bound_ckpts[f"control|s{seed}"]  # (02:32: no glob at run)
        for arm in CONFIRM_ARMS:
            model = build_arm(arm, make_siw_model, cck, kck, fresh_init=base_tree)
            cfg.seed = seed
            finetune(model, records[:K_STAR], cfg, cov, arm, K_STAR)
            eval_seed = 50_000 + seed
            mx.random.seed(eval_seed)
            res = evaluate(model, test_eps, seed, arm, K_STAR)
            for r in res:
                raw_fh.write(json.dumps({
                    "arm": arm, "train_seed": seed, "eval_seed": eval_seed,
                    "k": K_STAR, "episode_id": r.episode_id,
                    "layout_hash": r.layout_id, "goal_type": r.goal_type,
                    "success": r.success, "outcome": r.outcome,
                    "length": r.length, "n_invalid": r.n_invalid,
                    "d_star": r.d_star, "L_star": r.L_star,
                    "goal_reached_without_stop": r.goal_reached_without_stop},
                    default=str) + "\n")
            ck = os.path.join(staging, f"cell-{arm}-s{seed}-k{K_STAR}.npz")
            params = nn.utils.tree_flatten(model.parameters())
            mx.savez(ck, **{k2: v for k2, v in params})
            ckpt_manifest[f"{arm}|s{seed}"] = {
                "path": os.path.basename(ck), "sha256": _sha256_file(ck),
                "updates": cfg.updates}
            _buffered.append(f"{arm} s{seed} k=500: "
                             f"{M.summarize(res)['success_rate']:.4f}")
    raw_fh.flush()
    os.fsync(raw_fh.fileno())
    raw_fh.close()

    from_raw = _results_from_raw(raw_path)
    primary_raw = M.paired_hierarchical_bootstrap(
        from_raw["pretrained-TGK"], from_raw["scratch"], n_boot=5000, seed=0)
    attribution_raw = {
        "control_vs_scratch": M.paired_hierarchical_bootstrap(
            from_raw["control-nontarget"], from_raw["scratch"], n_boot=5000, seed=0),
        "control_vs_pretrained": M.paired_hierarchical_bootstrap(
            from_raw["control-nontarget"], from_raw["pretrained-TGK"],
            n_boot=5000, seed=0),
    }
    per_seed = {}
    for arm, rs in from_raw.items():
        for er in rs:
            per_seed.setdefault(f"{arm}|s{er.seed}", []).append(er.success)
    per_seed = {k: sum(v) / len(v) for k, v in sorted(per_seed.items())}

    artifact = {
        "gate": "V1 CONFIRMATION (primary only, k*=500) — v4",
        "protocol_hash": protocol_hash,
        "primary_from_raw": {"diff_pts": primary_raw["point_diff"] * 100,
                              "ci": [primary_raw["ci_low"], primary_raw["ci_high"]],
                              "PASS": (primary_raw["point_diff"] >= 0.05
                                       and primary_raw["ci_low"] > 0),
                              "raw": primary_raw},
        "attribution_from_raw": attribution_raw,
        "per_seed_from_raw": per_seed,
        "sealed": {"couples_sha256": full, "test2_sha256": full2},
        "raw_episodes_file": "confirm-episodes-raw.jsonl",
        "raw_sha256": _sha256_file(raw_path),
        "checkpoints_manifest": ckpt_manifest,
        "coverage": cov.as_dict(),
        "interactions_count": ilog.count,
        "arms_from_raw": {a: M.summarize(r) for a, r in from_raw.items()},
        "note": "AULC / 80% threshold: ABSENT by design (primary-only confirmation)",
    }
    with open(os.path.join(staging, "confirm-report.json"), "w") as fh:
        json.dump(artifact, fh, indent=2, default=str)
    assert len(artifact["raw_sha256"]) == 64
    assert all(len(v["sha256"]) == 64 for v in ckpt_manifest.values())
    assert "ci" in artifact["primary_from_raw"]
    os.rename(staging, args.out_dir)      # ATOMIC PUBLISH
    for line in _buffered:
        print(line, flush=True)
    print(json.dumps(artifact["primary_from_raw"], indent=1, default=str))


def _ck_path(pattern, seed):
    g = sorted(glob.glob(pattern.format(seed=seed)))[-1]
    return g if g.endswith(".npz") else g + "/checkpoint.npz"


def _fsync(ilog):
    with open(ilog.path, "rb") as f:
        os.fsync(f.fileno())


def build_freeze_manifest(freeze_path: str, *, couples_path, couples_sha,
                          layout_store, source_manifest_path, updates,
                          seed_gen, seed_ep, test2_manifest_pattern) -> str:
    """Paths are stored ABSOLUTE at freeze time — the snapshot works
    from any worktree; read-only consumption verified by G2 before
    the test2 open."""
    # (07:19 NO-GO) The source manifest is a PUBLISHED/COMMITTED input — we
    # NEVER rewrite it. The PRODUCER must write absolute read-only paths; we
    # enforce that here and fail on any relative path.
    """FREEZE HELPER (stage A): writes freeze-manifest.json binding code
    snapshot SHAs, input contents, generation params, and protocol constants;
    returns its canonical sha (the value to publish as --protocol-hash).

    Full read-only validation BEFORE any write (08:32): an invalid source
    manifest aborts the freeze with ZERO side effects — seeds are never
    consumed by a freeze on an invalid manifest."""
    source_manifest_path = os.path.abspath(source_manifest_path)
    for e in json.load(open(source_manifest_path)).get("checkpoints", []):
        if not os.path.isabs(e.get("path", "")):
            raise SystemExit(f"build_freeze_manifest: source manifest entry has "
                             f"a RELATIVE path (producer must commit absolute): "
                             f"{e.get('path')!r}")
    _verify_source_manifest(source_manifest_path)
    fm = {
        "version": 2,
        "code": {rel: _sha256_file(_rp(rel)) for rel in _BOUND_CODE},
        "inputs": {
            "couples": {"path": couples_path, "sha256": couples_sha},
            "exclusion_store": {"path": layout_store,
                                  "sha256": _sha256_file(layout_store)},
            "source_manifest": {"path": source_manifest_path,
                                  "sha256": _sha256_file(source_manifest_path)},
        },
        "generation": {
            "builder": "ucm.data.siw_pipeline.build_siw_test_episodes",
            "builder_sha": _sha256_file(_rp("ucm/data/siw_pipeline.py")),
            "seed_gen": seed_gen, "seed_episodes": seed_ep,
            "distribution": "600 total = 150 per predicate (VIEW/SET/CHOOSE/"
                            "SUBMITTED), pool 170 layouts, rng.randrange with "
                            "replacement, variable layouts-per-episode",
            "test2_data_manifest": {"method": "sha256_write_stream",
                                     "manifest_path": test2_manifest_pattern},
        },
        "protocol": {"k_star": K_STAR, "arms": list(CONFIRM_ARMS),
                      "seeds": list(range(5)), "updates": updates,
                      "init_seed_base": 9000, "eval_seed_base": 50000,
                      "expected_total": EXPECTED_TOTAL,
                      "expected_per_predicate": EXPECTED_PER_PREDICATE},
    }
    # O_EXCL (08:34): a published freeze manifest must NEVER be silently
    # overwritten — creation exclusive, abort BEFORE any write if it exists
    try:
        fd = os.open(freeze_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        raise SystemExit(f"build_freeze_manifest: {freeze_path} ALREADY EXISTS — "
                         f"published freeze, no silent re-freeze (inspect/resolve manually)")
    with os.fdopen(fd, "w") as fh:
        json.dump(fm, fh, indent=2, sort_keys=True)
    return hashlib.sha256(json.dumps(fm, sort_keys=True, default=str).encode()).hexdigest()


def compute_protocol_hash(args_ns) -> str:
    """Backward-compatible alias: canonical sha of the freeze manifest."""
    return _freeze_manifest_sha(args_ns.freeze_manifest)[0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--couples-file", required=True)
    ap.add_argument("--layout-store", default="artifacts/inventory-siw-dev-layouts.json")
    ap.add_argument("--canon-ckpt-pattern", default="artifacts/*gate2c-B144-s{seed}/checkpoint.npz")
    ap.add_argument("--control-ckpt-pattern", default="artifacts/*v1-control-s{seed}/checkpoint.npz")
    ap.add_argument("--test2-file", required=True)
    ap.add_argument("--freeze-manifest", required=True,
                    help="COMMITTED freeze-manifest.json (stage-A anchor: code+"
                         "inputs+generation+protocol; its canonical sha = "
                         "--protocol-hash)")
    ap.add_argument("--source-manifest", default=None,
                    help="DEPRECATED: inferred from the freeze manifest")
    ap.add_argument("--protocol-hash", required=True,
                    help="stage-A published protocol hash (code+manifest+couples+store)")
    ap.add_argument("--couples-hash", required=True)
    ap.add_argument("--test2-hash", required=True,
                    help="stage-B dataset hash, published AFTER generation, "
                         "BEFORE the run")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--updates", type=int, default=2000)
    main(ap.parse_args())
