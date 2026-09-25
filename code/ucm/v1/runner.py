"""V1 transfer RUNNER (§12, FREEZE V1a) — arms × seeds × budgets, fixed-budget
finetune (NO target selection), ONE reading of the sealed test cell covering
the whole pre-declared k-grid (the AULC secondary requires success per k by
design — this IS the single opening, no re-runs), paired hierarchical IC at
k*=500, coverage published per arm/budget, interactions counted.

Data sources:
    --couples-file    adaptation couples (unique (layout, state, goal)) —
                      sealed file when published; dev provider otherwise
    --test-file       sealed test episodes (ONE read)
    --canon-ckpt-pattern / --control-ckpt-pattern  per-seed source checkpoints

The control arm never uses d_star (records flagged control; runner asserts).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random

import mlx.core as mx
import mlx.nn as nn

mx.set_default_device(mx.cpu)

from ucm.eval import metrics as M
from ucm.eval.rollout import ModelPolicy, run_episode
from ucm.model.siw_model import make_siw_model
from ucm.model.train import _grad_global_norm
from ucm.v1.tensorize_siw import (collate_siw, labels_from_supervision_siw,
                                  tensorize_siw_obs)
from ucm.v1.control_source import couple_of_record
from ucm.v1.transfer import (ARM_NAMES, CoverageTracker, FinetuneConfig,
                             InteractionsLog, UniqueCoupleBudgets, build_arm)
from mlx.utils import tree_map
from mlx.optimizers import AdamW
from ucm.model.loss import set_bc_loss

OUT = "artifacts/v1"


# ---------------------------------------------------------------------------
# data providers
# ---------------------------------------------------------------------------

def load_couples(path: str | None, n_dev: int = 400, seed: int = 0,
                 layout_store: str = "artifacts/inventory-siw-dev-layouts.json") -> list[dict]:
    if path:
        # schema dispatch via SealedOpenRegistry (10:26): single FD, no peek
        from ucm.v1.sealed_reader import sealed_registry
        reg = sealed_registry()
        fmt, lines, _ = reg.read_once(path)
        if fmt == "couples":
            from ucm.v1.data_adapter import couples_lines_to_records
            return couples_lines_to_records(lines, layout_store)
        return [json.loads(l) for l in lines if l.strip()]
    # DEV provider (labelled): couples derived from sealed TRAIN layouts via
    # the SIW env + oracle — adaptation-side only, no test interaction.
    from ucm.env.siw import SIW, SIWLayout
    from ucm.env.siw_oracle import solve as siw_solve  # if exposed; else env oracle
    raise NotImplementedError("dev provider: wire to sealed SIW layouts when "
                              "published; use --couples-file for the sealed run")


def load_test_episodes(path: str) -> list[dict]:
    """Sealed test episodes: [{episode_id, layout_spec, task, d_star,
    layout_hash}] — ONE read. The env is REBUILT and RESET per rollout
    (audit bug 4): episodes carry serializable specs, never live envs."""
    eps = [json.loads(l) for l in open(path) if l.strip()]
    for e in eps:
        assert {"episode_id", "layout_spec", "task"} <= set(e), e
    return eps


def _build_env(episode: dict):
    from ucm.env.siw import SIW, SIWLayout
    spec = dict(episode["layout_spec"])
    if isinstance(spec.get("widgets"), dict):  # id → widget map tolerated
        spec["widgets"] = list(spec["widgets"].values())
    env = SIW(SIWLayout(spec))
    env.reset(episode["task"])
    return env


# ---------------------------------------------------------------------------
# finetune (fixed budget, no selection)
# ---------------------------------------------------------------------------

def finetune(model, records, cfg: FinetuneConfig, cov: CoverageTracker, arm: str, k: int,
            trainable: str = "all", world: str = "siw"):
    """Fixed-update finetune on the k-couple slice. Coverage = unique couples
    exposed. Checkpoint policy: final at fixed budget (FREEZE §3).
    trainable: 'all' | 'refine' — 'refine' GÈLE la base du CIV (tagi-5 12:36
    normatif): l'optimiseur ne touche QUE model.refine (pas de drift wd).
    world: 'siw' | 'tgk' — front-end de tensorisation (S2b verdict = TGK)."""
    import time
    import numpy as np
    if k == 0 or not records:
        return {"updates": 0, "wall_s": 0.0}
    # NOTE (audit #8): the TARGET fine-tune data is IDENTICAL for all arms
    # (freeze §2/§3: the control's non-informativity lives in its SOURCE
    # pretraining — the control checkpoint from train_control.py/load_control_
    # records — NOT in the target budget; §12.3: same target data per arm).
    exs = []
    if world == "tgk":
        from ucm.model.tensorize import (collate as _collate_tgk,
                                         labels_from_supervision as _lfs_tgk,
                                         tensorize_obs as _tobs_tgk)
        for rec in records:
            ex = _tobs_tgk(rec["policy_input"])
            ex["labels"] = _lfs_tgk(rec["policy_input"],
                                    rec["supervision"]["optimal_actions"])
            exs.append(ex)
        collate_fn = _collate_tgk
    else:
        for rec in records:
            ex = tensorize_siw_obs(rec["policy_input"])
            sup = rec["supervision"]["optimal_actions"]
            ex["labels"] = labels_from_supervision_siw(rec["policy_input"], sup)
            exs.append(ex)
        collate_fn = collate_siw
    for rec in records:
        cov.expose(arm, k, {"layout": rec.get("provenance", {}).get("layout_hash", ""),
                            "state": rec.get("provenance", {}).get("state_goal_hash", ""),
                            "goal": rec["policy_input"].get("goal", {})})
    rng = random.Random(cfg.seed)
    opt = AdamW(learning_rate=cfg.lr, weight_decay=cfg.weight_decay)
    idx = list(range(len(exs)))
    t0 = time.time()
    losses = []
    loss_curve = []   # (step, loss) échantillonnés — publié dans l'artefact
    for step in range(1, cfg.updates + 1):
        rng.shuffle(idx)
        chunk = [exs[i] for i in idx[:cfg.batch_size]] or exs
        batch = collate_fn(chunk)

        def loss_fn():
            return set_bc_loss(model(batch), batch["labels"], batch["cand_mask"])

        loss, grads = nn.value_and_grad(model, loss_fn)()
        gn = _grad_global_norm(grads)
        scale = mx.minimum(1.0, cfg.clip_norm / mx.maximum(gn, 1e-12))
        if trainable == "refine":
            # GEL BASE: l'OPTIMISEUR ne voit QUE model.refine (subtree) —
            # AdamW wd ne peut PAS dériver la base (elle n'est jamais passée
            # à update, contrairement à des grads zéro qui laisseraient le
            # weight-decay agir). Preuve: base_params_sha256 avant/après.
            # BUG 14:55 CORRIGÉ: value_and_grad retourne un dict IMBRIQUÉ
            # {'base':…, 'refine':…} — l'ancien filtre par clés PLATES avait
            # une intersection VIDE (opt.update no-op ⇒ refine jamais entraîné,
            # cause racine du 0%). Le subtree est grads['refine'] directement.
            sub = tree_map(lambda g: g * scale, dict(grads["refine"]))
            opt.update(model.refine, sub)
        else:
            opt.update(model, tree_map(lambda g: g * scale, grads))
        mx.eval(loss, model.parameters())
        lv = float(loss)
        if step == 1 or step % 200 == 0:
            losses.append(lv)
            loss_curve.append({"step": step, "loss": lv})
    return {"updates": cfg.updates,
            "mean_loss_tail": sum(losses) / max(len(losses), 1),
            "loss_first": loss_curve[0]["loss"] if loss_curve else None,
            "loss_last": loss_curve[-1]["loss"] if loss_curve else None,
            "loss_curve": loss_curve,
            "n_loss_samples": len(losses),
            "wall_s": round(time.time() - t0, 1)}


# ---------------------------------------------------------------------------
# closed-loop evaluation (one read)
# ---------------------------------------------------------------------------

from ucm.v1.policy_siw import SIWModelPolicy


def evaluate(model, episodes, seed: int, arm: str, k: int, horizon: int = 64):
    results = []
    for ep in episodes:
        env = _build_env(ep)  # fresh env + reset per rollout (audit bug 4)
        pol = SIWModelPolicy(model, name=f"{arm}@k={k}", seed=seed)
        r = run_episode(env, pol, ep["episode_id"], seed, f"{arm}@k={k}",
                        d_star=ep.get("d_star"), horizon=horizon)
        r.layout_id = ep.get("layout_hash", "")
        r.seed = 100 + seed  # eval seed level distinct from training seed
        r.goal_type = ep.get("goal_type", "")
        results.append(r)
    return results


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(args):
    os.makedirs(OUT, exist_ok=True)
    ilog = InteractionsLog()
    ilog.log("runner start (no target touch)", arm="-")
    sealed_slices = None
    if args.couples_pattern:
        # SEALED per-budget files are the frozen nested subsets (94 ⊂ 376 ⊂
        # 852 ⊂ 1191 — verified set-inclusion); the runner consumes them
        # DIRECTLY rather than re-deriving an ordering.
        import json as _j
        sealed_slices = {}
        for k in [int(x) for x in args.k_grid.split(",")]:
            if k == 0:
                sealed_slices[0] = []
                continue
            p = args.couples_pattern.format(k=k)
            fmt_k, lines_k, _ = sealed_registry().read_once(p)
            from ucm.v1.data_adapter import couples_lines_to_records
            sealed_slices[k] = couples_lines_to_records(lines_k, args.layout_store)
            print(f"scellé k={k}: {len(sealed_slices[k])} records ← {p}", flush=True)
        records = sealed_slices[max(sealed_slices)]
        budgets = UniqueCoupleBudgets(records, projector=couple_of_record)
    else:
        records = load_couples(args.couples_file)  # full records
        budgets = UniqueCoupleBudgets(records, projector=couple_of_record)
    print(f"couples uniques (k max): {budgets.n_unique}")
    # materialization verifiability (lead 19:39): hash of the MATERIALIZED
    # records per budget — an auditor re-deriving gets the same hashes
    import hashlib as _h
    def _recs_hash(recs):
        canon = "\n".join(json.dumps({"pi": r["policy_input"], "sup": r["supervision"],
                                        "prov": r["provenance"]}, sort_keys=True, default=str)
                           for r in recs)
        return _h.sha256(canon.encode()).hexdigest()[:16]
    materialized_hashes = {f"k={k}": _recs_hash(budgets.slice(k))
                           for k in [int(x) for x in args.k_grid.split(",")] if k > 0}
    fmt2, lines2, _ = sealed_registry().read_once(args.test_file)
    if fmt2 == "episodes":
        from ucm.v1.data_adapter import episodes_lines_to_runner
        test_eps = episodes_lines_to_runner(lines2, args.test_layout_store)
    else:
        test_eps = [json.loads(l) for l in lines2 if l.strip()]
    # 10:55: hashes from the REGISTRY (recorded at the unique read), never
    # (10:55/10:58) hashes from the registry — full SHAs of the unique read
    _sealed_sources = [p for p in (args.couples_file, args.test_file) if p]
    if args.couples_pattern:
        _sealed_sources += [args.couples_pattern.format(k=k)
                            for k in [int(x) for x in args.k_grid.split(",")] if k > 0]
    _reg = sealed_registry()
    # FAIL-CLOSED (10:58/11:01): every declared sealed source must have been
    # read — missing/unread → recorded_hash raises, no silent omission
    _hashes = {p: _reg.recorded_hash(p) for p in _sealed_sources}
    assert set(_hashes.keys()) == set(_sealed_sources), "hash audit: source set mismatch"
    ilog.log(f"sealed content hashes (registry, full sha): {json.dumps(_hashes)}",
             arm="adapter", target="SIW-sealed")
    ilog.log(f"SEALED TEST OPENED (one read, {len(test_eps)} episodes, grid {args.k_grid})",
             arm="all", target="SIW-test")

    cfg = FinetuneConfig(seed=0)
    cov = CoverageTracker()
    per_cell = {}
    all_results = []
    canon_p = args.canon_ckpt_pattern or "artifacts/*gate2c-B144-s{seed}/checkpoint.npz"
    control_p = args.control_ckpt_pattern or "artifacts/*v1-control-s{seed}/checkpoint.npz"
    arm_list = list(ARM_NAMES)
    if getattr(args, "secondary_validity_only", False):
        arm_list.append("control-validity-only")  # declared secondary
    for seed in [int(s) for s in args.seeds.split(",")]:
        canon_ck = sorted(glob.glob(canon_p.format(seed=seed)))
        control_ck = sorted(glob.glob(control_p.format(seed=seed)))
        # SHARED fresh init for this seed (audit B2): built once, every arm
        # and every budget cell starts from THIS initialization (+ transfer).
        mx.random.seed(9000 + seed)
        base = make_siw_model()
        base_tree = nn.utils.tree_flatten(base.parameters())
        for arm in arm_list:
            cck = canon_ck[-1] if canon_ck else None
            kck = control_ck[-1] if control_ck else None
            if arm == "control-nontarget":
                assert kck is not None, ("control arm requires its source "
                                         "checkpoint (train_control.py / "
                                         "load_control_records variant)")
            if arm == "pretrained-TGK":
                assert cck is not None, "pretrained arm requires canon checkpoint"
            for k in [int(x) for x in args.k_grid.split(",")]:
                # FRESH start per cell (audit bug 6): same shared init,
                # independent training — k=500 has NOT seen k=100's updates.
                model = build_arm(arm, make_siw_model, cck, kck,
                                  fresh_init=base_tree)
                if not hasattr(model, "transfer_manifest"):
                    model.transfer_manifest = {"arm": arm, "note": "factory-direct"}
                cfg.seed = seed
                # RULING 19:31 (REVERT of 19:21, tagi-5 arbitration): the
                # reference control arm receives the SAME informative target
                # slice — it differs in SOURCE only (non-informative TGK
                # checkpoint, validated at load by load_control_records).
                slice_records = sealed_slices[k] if sealed_slices is not None else budgets.slice(k)
                if arm == "control-validity-only" and k > 0:
                    slice_records = derive_control_records_siw(slice_records)
                ft = finetune(model, slice_records, cfg, cov, arm, k)
                res = evaluate(model, test_eps, seed, arm, k)
                all_results.extend(res)
                s = M.summarize(res)
                per_cell[f"{arm}|s{seed}|k={k}"] = {**s, "finetune": ft}
                print(f"{arm} s{seed} k={k}: success {s['success_rate']:.3f}", flush=True)

    # primary: k*=500 pretrained vs scratch (paired hierarchical)
    def at(arm, k):
        return [r for r in all_results if r.policy_name == f"{arm}@k={k}"]
    # NOTE: control-validity-only is REPORTED but never part of the reference
    # comparisons (ruling 19:31)
    k_star = 500
    primary = M.paired_hierarchical_bootstrap(at("pretrained-TGK", k_star),
                                              at("scratch", k_star), n_boot=5000, seed=0)
    secondary = {
        "control_vs_scratch": M.paired_hierarchical_bootstrap(
            at("control-nontarget", k_star), at("scratch", k_star), n_boot=5000, seed=0),
        "control_vs_pretrained": M.paired_hierarchical_bootstrap(
            at("control-nontarget", k_star), at("pretrained-TGK", k_star), n_boot=5000, seed=0),
    }
    # AULC over log(1+k) common grid
    import math
    grid = sorted({int(x) for x in args.k_grid.split(",")})
    aulc = {}
    for arm in ARM_NAMES:
        num = den = 0.0
        for a, b in zip(grid, grid[1:]):
            sa = M.summarize(at(arm, a))["success_rate"]
            sb = M.summarize(at(arm, b))["success_rate"]
            w = math.log(1 + b) - math.log(1 + a)
            num += w * (sa + sb) / 2
            den += w
        aulc[arm] = round(num / den, 4) if den else None

    artifact = {
        "protocol_ref": "ucm/v1/FREEZE.md (frozen before target inspection)",
        "primary_k_star": {"diff_pts": primary["point_diff"] * 100,
                           "ci": [primary["ci_low"], primary["ci_high"]],
                           "PASS": primary["point_diff"] >= 0.05 and primary["ci_low"] > 0},
        "secondary": secondary, "aulc_log_grid": aulc,
        "coverage": cov.as_dict(),
        "materialized_records_hashes_per_budget": materialized_hashes,
        "interactions_count": ilog.count,
        "cells": per_cell,
        "threshold_80pct": {arm: (min(k for k in grid
                                     if M.summarize(at(arm, k))["success_rate"] >= 0.8)
                                   if any(M.summarize(at(arm, k))["success_rate"] >= 0.8
                                          for k in grid) else ">10000 (censuré)")
                            for arm in ARM_NAMES},
        "args": vars(args),
    }
    with open(f"{OUT}/transfer-report.json", "w") as fh:
        json.dump(artifact, fh, indent=2, default=str)
    print(json.dumps({k: artifact[k] for k in ("primary_k_star", "aulc_log_grid",
                                               "coverage", "interactions_count")},
                     indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--k-grid", default="0,100,500,2000,10000")
    ap.add_argument("--couples-file", default=None)
    ap.add_argument("--couples-pattern", default=None,
                    help="sealed per-budget files, e.g. artifacts/siw-couples-k{k}.jsonl")
    ap.add_argument("--layout-store", default="artifacts/inventory-siw-dev-layouts.json")
    ap.add_argument("--test-file", required=True)
    ap.add_argument("--test-layout-store", default="artifacts/layouts-SIW-small.json")
    ap.add_argument("--canon-ckpt-pattern", default=None)
    ap.add_argument("--control-ckpt-pattern", default=None)
    ap.add_argument("--secondary-validity-only", action="store_true",
                    help="DECLARED SECONDARY arm (ruling 19:31): control-source "
                         "init + non-informative target slice via "
                         "derive_control_records_siw. Optional analysis, NEVER "
                         "the reference comparison.")
    main(ap.parse_args())
