"""Training pipeline for Model A (spec §8.2, PLAN WS-C).

Configuration normative for V0 (spec §8.2):
    AdamW, lr 3e-4, weight decay 1e-4, batch 64, grad-norm clip 1.0, FP32,
    BC capped at 10 epochs OR 10k updates (first limit reached);
    validation every 500 updates and at the end;
    selection at best validation metric, then length/cost as tie-break (later,
    closed-loop); three exploratory seeds supported via --seed.

Overfit diagnostic (GATE-1 / M1): 100 episodes, up to 5k updates, NOT subject
to the 10-epoch cap (spec §8.2). Target ≥99 % optimal actions.

Every run writes artifacts/<timestamp>-<name>/ with config, seeds, metrics,
timings and checkpoint (spec §15.2) — nothing lives only in a terminal.

Data: JSONL records (schema §3.3/§5.2: policy_input + supervision) via --data,
or --synthetic N for dev fixtures (ucm.model.fixtures — dev scaffolding only).
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import os
import platform
import random
import subprocess
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from ucm.model.deepsets_a import DeepSetsA
from ucm.model.gnn_b import GNNB
from ucm.model.loss import optimal_action_rate, prob_mass_on_optimal, set_bc_loss
from ucm.model.tensorize import (collate, labels_from_supervision, permute_example,
                                 tensorize_obs)
from ucm.model import fixtures as F
from mlx.utils import tree_map


@dataclasses.dataclass
class TrainConfig:
    name: str = "modelA"
    data: str = ""  # JSONL path (WS-B records); "" with synthetic/episodes → dev fixtures
    synthetic: int = 0  # number of synthetic single-state records
    episodes: int = 0  # number of FULL episodes (trajectories) from the real env
    synthetic_seed: int = 0
    val_fraction: float = 0.15  # only used when a single pool is given
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    clip_norm: float = 1.0
    max_updates: int = 10_000
    max_epochs: int = 10
    eval_every: int = 500
    seed: int = 0
    overfit: bool = False  # diagnostic mode: ≤5k updates, epoch cap lifted
    overfit_updates: int = 5_000
    device: str = "cpu"  # "cpu" | "gpu"
    arch: str = "A"  # "A" (DeepSets d=128) | "B" (GNN d=192)
    train_split: str | None = None  # canon provenance.split for training pool
    val_split: str | None = None    # canon provenance.split for validation pool
    d_model: int = 128


def _git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _episode_records(n_episodes: int, seed: int) -> tuple[list[dict], dict]:
    """Full A*-trajectory records from the REAL env/oracle (WS-A generate
    module). Mixture: band d*=2..12 with d*=0 allowed (§4.5). Returns records
    + d* histogram (published in the artifact)."""
    from ucm.data import generate as gen
    rng = random.Random(seed)
    stats = gen.GenerationStats()
    records: list[dict] = []
    hist: dict[str, int] = {}
    guard = 0
    while stats.episodes < n_episodes and guard < n_episodes * 100:
        guard += 1
        lay = gen.generate_layout(rng, rng.randint(4, 8), 0.4)
        recs = gen.generate_episode(rng, lay, d_star_band=(2, 12), stats=stats,
                                    allow_zero=False, oracle_cache={},
                                    g2_mode="exclude", episode_ref=f"ep{stats.episodes}")
        if recs:
            records.extend(recs)
            d0 = recs[0]["supervision"]["d_star"]
            hist[str(d0)] = hist.get(str(d0), 0) + 1
    if stats.episodes < n_episodes:
        raise RuntimeError(f"episode generation stalled at {stats.episodes}/{n_episodes}")
    return records, hist


def load_records(path: str, split: str | None = None,
                 allow_test: bool = False) -> list[dict]:
    """Read WS-B JSONL records; validate supervision reachability + caps.
    Visible errors, no silent filtering of unsupported lines (spec §9.4).
    split: if set, keep ONLY provenance.split == split. Training splits never
    see test_* cells (audit tagi-5 M7): a split starting with 'test' requires
    allow_test=True explicitly (evaluation-only readers)."""
    records = []
    with open(path) as fh:
        for ln, line in enumerate(fh, 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("schema_version") != "0.2":
                raise ValueError(f"{path}:{ln}: schema_version != 0.2")
            rec_split = rec.get("provenance", {}).get("split")
            if split is not None:
                if rec_split != split:
                    if rec_split and rec_split.startswith("test") and not allow_test \
                            and not split.startswith("test"):
                        continue  # never silently mix test cells into training
                    continue
            elif rec_split and rec_split.startswith("test") and not allow_test:
                continue  # default: training loaders exclude test cells
            sup = rec["supervision"]
            if not sup.get("reachable", False) or not sup.get("optimal_actions"):
                raise ValueError(f"{path}:{ln}: unreachable/empty supervision must not enter BC")
            tensorize_obs(rec["policy_input"])  # validation: caps + vocab + bindings
            records.append(rec)
    if not records:
        raise ValueError(f"{path}: no records")
    return records


def _examples_from_records(records: list[dict]) -> list[dict]:
    exs = []
    for rec in records:
        ex = tensorize_obs(rec["policy_input"])
        ex["labels"] = labels_from_supervision(rec["policy_input"], rec["supervision"]["optimal_actions"])
        exs.append(ex)
    return exs


def _batches(exs: list[dict], cfg: TrainConfig, rng: random.Random):
    """Epoch-based shuffling (without replacement) → index batches. Per example
    per epoch: coherent ORDER randomization (entities + relations + candidates,
    labels permuted together) — PLAN §WS-C, audit tagi-5 T8: list rank must
    carry no learnable signal, in train as in test."""
    n = len(exs)
    idx = list(range(n))
    while True:
        rng.shuffle(idx)
        for s in range(0, n, cfg.batch_size):
            chunk = [exs[i] for i in idx[s:s + cfg.batch_size]]
            yield [permute_example(dict(ex), rng) for ex in chunk]


def _grad_global_norm(grads: dict) -> mx.array:
    leaves = [g for _, g in nn.utils.tree_flatten(grads)]
    mx.eval(*leaves)
    total = mx.zeros(())
    for g in leaves:
        total = total + (g.astype(mx.float32) ** 2).sum()
    return mx.sqrt(total)


def evaluate(model: DeepSetsA, exs: list[dict], batch_size: int = 128) -> dict:
    """Diagnostic metrics over a labelled example pool (batched, mx.eval'ed)."""
    losses, hits, masses, denom = [], 0, 0.0, 0
    for s in range(0, len(exs), batch_size):
        batch = collate(exs[s:s + batch_size])
        logits = model(batch)
        loss = set_bc_loss(logits, batch["labels"], batch["cand_mask"])
        hit = optimal_action_rate(logits, batch["labels"], batch["cand_mask"])
        mass = prob_mass_on_optimal(logits, batch["labels"], batch["cand_mask"])
        mx.eval(loss, hit, mass)
        bs = batch["labels"].shape[0]
        losses.append(float(loss) * bs)
        hits += int(hit * bs)
        masses += float(mass) * bs
        denom += bs
    return {"loss": sum(losses) / denom, "optimal_action_rate": hits / denom,
            "mean_p_on_optimal": masses / denom, "n_examples": denom}


def train(cfg: TrainConfig) -> dict:
    t_start = time.time()
    # deterministic seeding (python, numpy; MLX RNG seeded for param init)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    try:
        mx.random.seed(cfg.seed)
    except Exception:
        pass
    if cfg.device == "gpu":
        mx.set_default_device(mx.gpu)
    else:
        mx.set_default_device(mx.cpu)

    # --- data --------------------------------------------------------------
    val_exs: list = []  # canon-val pool may be loaded below (M7: never clobbered)
    if cfg.data:
        records = load_records(cfg.data, split=cfg.train_split)
        src = {"kind": "jsonl", "path": cfg.data, "n_records": len(records),
               "train_split": cfg.train_split, "val_split": cfg.val_split}
        if cfg.val_split:
            val_records = load_records(cfg.data, split=cfg.val_split)
            val_exs = _examples_from_records(val_records)  # canon pools are
            # layout-disjoint by construction (WS-B GATE-0) — no leakage
    elif cfg.episodes > 0:
        records, ep_stats = _episode_records(cfg.episodes, cfg.synthetic_seed)
        src = {"kind": "episodes-dev", "n_episodes": cfg.episodes,
               "n_records": len(records), "d_star_hist": ep_stats, "fixture_seed": cfg.synthetic_seed}
    else:
        records = F.make_records(random.Random(cfg.synthetic_seed), cfg.synthetic)
        src = {"kind": "synthetic-dev", "n_records": len(records),
               "fixture_seed": cfg.synthetic_seed}
    rng = random.Random(cfg.seed)
    idx = list(range(len(records)))
    rng.shuffle(idx)
    if not cfg.overfit and not cfg.val_split:
        # layout-grouped holdout (audit tagi-5 M7): episodes of one layout stay
        # on one side — group by provenance.layout_hash
        n_val = max(1, int(len(records) * cfg.val_fraction)) if not cfg.overfit else 0
        by_lay: dict[str, list[int]] = {}
        for i, rec in enumerate(records):
            by_lay.setdefault(rec.get("provenance", {}).get("layout_hash", f"r{i}"), []).append(i)
        val_idx: set[int] = set()
        lay_keys = sorted(by_lay)
        rng.shuffle(lay_keys)
        taken = 0
        for k in lay_keys:
            if taken >= n_val:
                break
            val_idx.update(by_lay[k])
            taken += len(by_lay[k])
        val_exs = _examples_from_records([records[i] for i in sorted(val_idx)])
        train_exs = _examples_from_records([records[i] for i in idx if i not in val_idx])
    elif cfg.val_split:
        # canon val pool already loaded; training pool = the filtered records
        train_exs = _examples_from_records(records)
    if cfg.overfit:  # diagnostic overfits the FULL given pool (typically 100 episodes)
        train_exs = _examples_from_records(records)

    # --- model / optimizer ---------------------------------------------------
    if cfg.arch == "B":
        if cfg.d_model == 128:  # default (A's width) → §6.2 normative B width
            cfg.d_model = 192
        assert cfg.d_model in (144, 160, 192), "B width: 192 (§6.2), 160 (V0b) or 144 (V0c, lead 15:09)"
        # config must not lie (audit m9): d_model stays as constructed
    model = (DeepSetsA(d=cfg.d_model) if cfg.arch == "A" else GNNB(d=cfg.d_model))
    n_params = model.print_param_count()
    mx.eval(model.parameters())
    optimizer = optim.AdamW(learning_rate=cfg.lr, weight_decay=cfg.weight_decay)

    run_dir = os.path.join("artifacts", f"{_dt.datetime.now():%Y%m%d-%H%M%S}-{cfg.name}")
    os.makedirs(run_dir, exist_ok=True)
    artifact = {
        "config": dataclasses.asdict(cfg),
        "data": src,
        "n_params": n_params,
        "seed": cfg.seed,
        "git_rev": _git_rev(),
        "runtime": {"python": sys.version.split()[0], "mlx": mx.__version__,
                    "numpy": np.__version__, "os": platform.platform(),
                    "device": str(mx.default_device())},
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(run_dir, "config.json"), "w") as fh:
        json.dump(artifact, fh, indent=2)

    # --- loop ----------------------------------------------------------------
    max_updates = min(cfg.overfit_updates, cfg.max_updates) if cfg.overfit else cfg.max_updates
    max_epochs = sys.maxsize if cfg.overfit else cfg.max_epochs
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    best = {"metric": -1.0}
    batch_iter = _batches(train_exs, cfg, rng)
    n_train = len(train_exs)
    updates_per_epoch = max(1, (n_train + cfg.batch_size - 1) // cfg.batch_size)
    t0 = time.time()
    stop_reason = "max_updates"
    with open(metrics_path, "w") as mfh:
        for update in range(1, max_updates + 1):
            batch = collate(next(batch_iter))

            def loss_fn():
                return set_bc_loss(model(batch), batch["labels"], batch["cand_mask"])

            loss, grads = nn.value_and_grad(model, loss_fn)()
            gn = _grad_global_norm(grads)
            scale = mx.minimum(1.0, cfg.clip_norm / mx.maximum(gn, 1e-12))
            grads = tree_map(lambda g: g * scale, grads)
            optimizer.update(model, grads)
            mx.eval(loss, model.parameters())
            if update % cfg.eval_every == 0 or update == max_updates:
                val_pool = val_exs if val_exs else train_exs
                ev = evaluate(model, val_pool)
                sel = ev["optimal_action_rate"]  # selection metric (validation)
                if sel > best["metric"]:
                    best = {"metric": sel, "update": update, **ev}
                    # M6 (audit tagi-5): checkpoint the SELECTED model (best
                    # validation), not the last one
                    _params = nn.utils.tree_flatten(model.parameters())
                    mx.eval(*[p for _, p in _params])
                    mx.savez(os.path.join(run_dir, "checkpoint.npz"),
                             **{k: v for k, v in _params})
                rec = {"update": update, "epoch": update / updates_per_epoch,
                       "train_loss": float(loss), "grad_norm": float(gn),
                       "lr": cfg.lr, "eval_on": "val" if val_exs else "train",
                       **ev, "elapsed_s": round(time.time() - t0, 2)}
                mfh.write(json.dumps(rec) + "\n")
                mfh.flush()
                print(f"[{cfg.name}] upd {update:>5} loss {float(loss):.4f} "
                      f"OA-rate {ev['optimal_action_rate']:.4f} p(A*) {ev['mean_p_on_optimal']:.4f}", flush=True)
            if update // updates_per_epoch >= max_epochs:
                stop_reason = "max_epochs"
                break

    wall = time.time() - t_start
    final = evaluate(model, train_exs)
    out = {
        **artifact,
        "final_train": final,
        "best": best,
        "updates_run": update,
        "stop_reason": stop_reason,
        "checkpoint_policy": "best_validation" if best.get("update") else "final",
        "wall_time_s": round(wall, 2),
        "throughput_updates_per_s": round(update / wall, 2),
        "total_run_s": round(time.time() - t_start, 2),
    }
    with open(os.path.join(run_dir, "final.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    params = nn.utils.tree_flatten(model.parameters())
    # M6 (audit tagi-5): the checkpoint on disk is the SELECTED model — the
    # best-validation save from the loop is authoritative; keep the final
    # parameters as a separate artifact for diagnostics only.
    mx.savez(os.path.join(run_dir, "checkpoint-final.npz"),
             **{k: v for k, v in params})
    if best.get("update") is None:
        # no improvement event (e.g. eval never beat the initial -1): final IS the selection
        import shutil
        shutil.copy(os.path.join(run_dir, "checkpoint-final.npz"),
                    os.path.join(run_dir, "checkpoint.npz"))
    print(json.dumps({"run_dir": run_dir, "final_train_OA": final["optimal_action_rate"],
                      "best_val_OA": best["metric"], "updates": update, "wall_s": round(wall, 1)}))
    return out


def main():
    ap = argparse.ArgumentParser(description="Train Model A (UCM WS-C)")
    ap.add_argument("--name", default="modelA")
    ap.add_argument("--data", default="", help="JSONL records path (schema 0.2)")
    ap.add_argument("--synthetic", type=int, default=0, help="dev single-state records")
    ap.add_argument("--episodes", type=int, default=0, help="full episodes from the real env")
    ap.add_argument("--synthetic-seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--clip-norm", type=float, default=1.0)
    ap.add_argument("--max-updates", type=int, default=10_000)
    ap.add_argument("--max-epochs", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overfit", action="store_true")
    ap.add_argument("--overfit-updates", type=int, default=5_000)
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--arch", default="A", choices=["A", "B"])
    ap.add_argument("--train-split", default=None)
    ap.add_argument("--val-split", default=None)
    args = ap.parse_args()
    cfg = TrainConfig(**vars(args))
    if not cfg.data and cfg.synthetic <= 0 and cfg.episodes <= 0:
        ap.error("provide --data, --episodes N or --synthetic N")
    train(cfg)


if __name__ == "__main__":
    main()
