"""Latency profiling for the decision path (spec §10.2, PLAN M1 DoD).

MLX is lazily evaluated: a timing of graph CONSTRUCTION is not an inference
latency. Rules enforced here (spec §10.2):

    - every timed section is forced with mx.eval() and measured AFTER completion
    - 50 warm-up iterations, then ≥1000 measured decisions
    - batch 1 (single decision), p50/p95/p99 reported
    - stages measured separately: tensorization (adapter), model forward,
      selection; plus the full pipeline
    - padding regime declared (dynamic per-episode here: N, K, E exact)
    - machine/OS/MLX/precision recorded with every report

Objectives (spec §10.3, proposed): p95 model ≤20 ms batch 1; p95 full decision
with adapter ≤50 ms on the declared profile.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import statistics
import sys
import time

import mlx.core as mx
import numpy as np

from ucm.model.tensorize import collate, tensorize_obs


def _percentiles(xs: list[float]) -> dict:
    xs = sorted(xs)
    n = len(xs)

    def pct(p):
        return xs[min(n - 1, int(p / 100 * n))]
    return {"p50_ms": round(pct(50) * 1e3, 3), "p95_ms": round(pct(95) * 1e3, 3),
            "p99_ms": round(pct(99) * 1e3, 3), "mean_ms": round(statistics.fmean(xs) * 1e3, 3),
            "n": n}


def bench_decisions(observations: list[dict], model, warmup: int = 50,
                    n_measured: int = 1000, seed: int = 0,
                    device_note: str = "") -> dict:
    """Per-decision batch-1 latency over cycled observations.

    Stages: tensorize+collate (adapter), model forward (incl. mx.eval), argmax
    selection; full = all three. The same pre-tensorized arrays are reused for
    the model-only stage; the full pipeline re-tensorizes each decision.
    """
    import random as _random
    rng = _random.Random(seed)

    pre = []
    for obs in observations:
        ex = tensorize_obs(obs)
        ex["labels"] = None
        pre.append(collate([ex]))

    def forward(batch):
        lg = model(batch)[0]
        mx.eval(lg)
        return int(mx.argmax(lg).item())

    # warm-up (compilation + caches)
    for _ in range(warmup):
        forward(pre[rng.randrange(len(pre))])
    mx.eval(model.parameters())

    # model-only stage
    t_model = []
    for i in range(n_measured):
        batch = pre[i % len(pre)]
        mx.eval()  # drain
        t0 = time.perf_counter()
        forward(batch)
        t_model.append(time.perf_counter() - t0)

    # full pipeline stage (tensorize + forward + select)
    t_full, t_tens, t_sel = [], [], []
    for i in range(n_measured):
        obs = observations[i % len(observations)]
        mx.eval()
        t0 = time.perf_counter()
        ex = tensorize_obs(obs)
        ex["labels"] = None
        batch = collate([ex])
        t1 = time.perf_counter()
        lg = model(batch)[0]
        mx.eval(lg)
        t2 = time.perf_counter()
        pick = int(mx.argmax(lg).item())
        t3 = time.perf_counter()
        t_tens.append(t1 - t0)
        t_sel.append(t3 - t2)
        t_full.append(t3 - t0)
    report = {
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "runtime": {"python": sys.version.split()[0], "mlx": mx.__version__,
                    "numpy": np.__version__, "os": platform.platform(),
                    "device": device_note or str(mx.default_device()),
                    "precision": "fp32"},
        "padding_regime": "dynamic per episode (N,K,E exact, slot capacity padded)",
        "n_observations": len(observations),
        "model_only": _percentiles(t_model),
        "tensorize": _percentiles(t_tens),
        "selection": _percentiles(t_sel),
        "full_decision": _percentiles(t_full),
        "gates": {"p95_model_le_20ms": _percentiles(t_model)["p95_ms"] <= 20.0,
                  "p95_full_le_50ms": _percentiles(t_full)["p95_ms"] <= 50.0},
    }
    return report


def bench_and_save(observations: list[dict], model, run_dir: str, **kw) -> dict:
    rep = bench_decisions(observations, model, **kw)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "latency.json"), "w") as fh:
        json.dump(rep, fh, indent=2)
    return rep
