#!/usr/bin/env python3
"""Profil borné S2b : RecurrentRefine (SIWRecModel, T=32) vs B144 — DEV-only, report-only.

Mesures :
  A) PER-UPDATE entraînement (phases warm-up 50 / mesuré 200 / soutenu 200, batch 64, CPU) :
     kernel + full-update (mirroir de la boucle finetune), RSS, throttling (médiane soutenu/mesuré).
  B) LATENCE §10.2 STRICT (batch-1) : warm-up 50, ≥1000 décisions mesurées, drain `mx.eval`
     avant chaque chrono, segments séparés (tensorize / modèle / sélection / full), **CPU et GPU
     côte à côte**, p50/p95/p99 ; gate S2b : **p95 modèle ≤ 20 ms** (et full ≤ 50 ms, seuils V0
     publiés côte à côte sans attribution §10.3).

Chaque régime tourne dans SON processus (RSS propre) ; charge (loadavg) échantillonnée par phase.
Fixture : couples DEV générés à la volée (seed 20260929, N=256) — AUCUN scellé/test2.
Usage : python scripts/profile_recurrent_s2b.py --report reports/recurrent-s2b-profile.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_REPO = Path(__file__).resolve().parents[1]
_DEV_STORE = _REPO / "artifacts" / "inventory-siw-dev-layouts.json"

PHASES = {"warmup": 50, "measured": 200, "sustained": 200}


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rss_mb() -> float:
    import resource
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round((int(v) if sys.platform == "darwin" else int(v) * 1024) / 1e6, 1)


def _generate_couples(path: str, n: int, seed: int, n_layouts: int = 8):
    from ucm.data.siw_pipeline import generate_adaptation_episodes_rstar
    from ucm.env.siw import SIWLayout
    from ucm.v1.data_adapter import _spec_of
    store = json.load(open(_DEV_STORE))
    layouts = [SIWLayout(_spec_of(e)) for e in list(store.values())[:n_layouts]]
    eps, meta = generate_adaptation_episodes_rstar(layouts, seed=seed, n_episodes=n, d0_band=(2, 4))
    with open(path, "w") as fh:
        for e in eps:
            fh.write(json.dumps(e, sort_keys=True, default=str) + "\n")
    return meta


def _pct(xs: list[float]) -> dict:
    s = sorted(xs)
    n = len(s)

    def p(pc):
        return s[min(n - 1, int(pc / 100 * n))]
    return {"p50_ms": round(p(50) * 1e3, 3), "p95_ms": round(p(95) * 1e3, 3),
            "p99_ms": round(p(99) * 1e3, 3), "mean_ms": round(statistics.fmean(s) * 1e3, 3), "n": n}


# ---------------------------------------------------------------------------------------------
# A) per-update entraînement (boucle miroir de ucm/v1/runner.py::finetune)
# ---------------------------------------------------------------------------------------------
def _child_train(model_arm: str, workdir: str, k: int, out_json: str) -> int:
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    from mlx.optimizers import AdamW
    from mlx.utils import tree_map
    import mlx.nn as nn
    from ucm.model.loss import set_bc_loss
    from ucm.model.train import _grad_global_norm
    from ucm.v1.data_adapter import _couples_core
    from ucm.v1.tensorize_siw import (collate_siw, labels_from_supervision_siw,
                                        tensorize_siw_obs)
    from ucm.v1.transfer import FinetuneConfig

    store = json.load(open(os.path.join(workdir, "store.json")))
    lines = [l for l in open(os.path.join(workdir, "couples.jsonl")) if l.strip()][:k]
    records = _couples_core(store, lines)
    exs = []
    for rec in records:
        ex = tensorize_siw_obs(rec["policy_input"])
        ex["labels"] = labels_from_supervision_siw(rec["policy_input"],
                                                     rec["supervision"]["optimal_actions"])
        exs.append(ex)

    if model_arm == "rec":
        from ucm.model.recurrent_block import make_siw_rec_model
        model = make_siw_rec_model(T=32)
        n_params = model.param_count()
    else:
        from ucm.model.siw_model import make_siw_model
        model = make_siw_model()
        n_params = 698_401
    mx.eval(model.parameters())

    cfg = FinetuneConfig(updates=1, seed=0)     # batch 64 par défaut ; updates gérés par phases
    opt = AdamW(learning_rate=cfg.lr, weight_decay=cfg.weight_decay)
    rng = random.Random(cfg.seed)
    idx = list(range(len(exs)))
    times = {ph: {"kernel": [], "full": [], "pre": []} for ph in PHASES}
    loads = {}
    for phase, n_updates in PHASES.items():
        loads[phase] = [round(x, 2) for x in os.getloadavg()]
        for _ in range(n_updates):
            t_full0 = time.perf_counter()
            rng.shuffle(idx)
            chunk = [exs[i] for i in idx[:cfg.batch_size]] or exs
            batch = collate_siw(chunk)
            t_pre1 = time.perf_counter()
            mx.eval()  # drain
            t_k0 = time.perf_counter()

            def loss_fn():
                return set_bc_loss(model(batch), batch["labels"], batch["cand_mask"])

            loss, grads = nn.value_and_grad(model, loss_fn)()
            gn = _grad_global_norm(grads)
            scale = mx.minimum(1.0, cfg.clip_norm / mx.maximum(gn, 1e-12))
            opt.update(model, tree_map(lambda g: g * scale, grads))
            mx.eval(loss, model.parameters())
            t_k1 = time.perf_counter()
            times[phase]["full"].append(t_k1 - t_full0)
            times[phase]["pre"].append(t_pre1 - t_full0)
            times[phase]["kernel"].append(t_k1 - t_k0)

    med = {ph: statistics.median(times[ph]["full"]) for ph in times}
    med_k = {ph: statistics.median(times[ph]["kernel"]) for ph in times}
    res = {
        "model_arm": model_arm, "n_params": n_params, "k_episodes": k,
        "n_records": len(records), "batch_size": 64, "device": "cpu",
        "kernel": {ph: _pct(times[ph]["kernel"]) for ph in times},
        "full_update": {ph: _pct(times[ph]["full"]) for ph in times},
        "shuffle_collate": {ph: _pct(times[ph]["pre"]) for ph in times},
        "updates_per_s": {"kernel_measured": round(1 / med_k["measured"], 2),
                           "full_measured": round(1 / med["measured"], 2),
                           "full_sustained": round(1 / med["sustained"], 2)},
        "throttle_ratio_full": round(med["sustained"] / med["measured"], 4),
        "rss_mb": _rss_mb(),
        "load_samples": loads,
    }
    Path(out_json).write_text(json.dumps(res, indent=2))
    print(f"[train {model_arm}] full p50 mesuré={res['full_update']['measured']['p50_ms']}ms "
          f"kern p50={res['kernel']['measured']['p50_ms']}ms rss={res['rss_mb']}MB", flush=True)
    return 0


# ---------------------------------------------------------------------------------------------
# B) latence §10.2 strict (batch-1) — CPU et GPU
# ---------------------------------------------------------------------------------------------
def _child_latency(model_arm: str, device: str, workdir: str, out_json: str) -> int:
    import mlx.core as mx
    mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)
    from ucm.v1.data_adapter import _couples_core
    from ucm.v1.tensorize_siw import collate_siw, tensorize_siw_obs

    store = json.load(open(os.path.join(workdir, "store.json")))
    lines = [l for l in open(os.path.join(workdir, "couples.jsonl")) if l.strip()][:32]
    records = _couples_core(store, lines)
    observations = [r["policy_input"] for r in records]

    if model_arm == "rec":
        from ucm.model.recurrent_block import make_siw_rec_model
        model = make_siw_rec_model(T=32)
    else:
        from ucm.model.siw_model import make_siw_model
        model = make_siw_model()
    mx.eval(model.parameters())

    pre = []
    for obs in observations:
        ex = tensorize_siw_obs(obs)
        ex["labels"] = None
        pre.append(collate_siw([ex]))

    def forward(batch):
        lg = model(batch)[0]
        mx.eval(lg)
        return int(mx.argmax(lg).item())

    rng = random.Random(0)
    for _ in range(50):                      # warm-up (50)
        forward(pre[rng.randrange(len(pre))])
    mx.eval(model.parameters())

    t_model = []
    for i in range(1000):                    # ≥1000 décisions mesurées
        batch = pre[i % len(pre)]
        mx.eval()                            # drain
        t0 = time.perf_counter()
        forward(batch)
        t_model.append(time.perf_counter() - t0)

    t_full, t_tens, t_sel = [], [], []
    for i in range(1000):
        obs = observations[i % len(observations)]
        mx.eval()
        t0 = time.perf_counter()
        ex = tensorize_siw_obs(obs)
        ex["labels"] = None
        batch = collate_siw([ex])
        t1 = time.perf_counter()
        lg = model(batch)[0]
        mx.eval(lg)
        t2 = time.perf_counter()
        _ = int(mx.argmax(lg).item())
        t3 = time.perf_counter()
        t_tens.append(t1 - t0); t_sel.append(t3 - t2); t_full.append(t3 - t0)

    n_iters = getattr(getattr(model, "refine", None), "n_iters", None)
    res = {
        "model_arm": model_arm, "device": device,
        "runtime": {"python": sys.version.split()[0], "mlx": mx.__version__},
        "padding_regime": "par batch (collate_siw — graphes SIW hétérogènes)",
        "n_observations": len(observations),
        "refine_iters_executed": n_iters,
        "model_only": _pct(t_model), "tensorize": _pct(t_tens),
        "selection": _pct(t_sel), "full_decision": _pct(t_full),
        "gates": {"p95_model_le_20ms": _pct(t_model)["p95_ms"] <= 20.0,
                   "p95_full_le_50ms": _pct(t_full)["p95_ms"] <= 50.0},
        "rss_mb": _rss_mb(),
        "load_samples": [round(x, 2) for x in os.getloadavg()],
    }
    Path(out_json).write_text(json.dumps(res, indent=2))
    print(f"[lat {model_arm} {device}] model p50={res['model_only']['p50_ms']}ms "
          f"p95={res['model_only']['p95_ms']}ms (gate={res['gates']['p95_model_le_20ms']}) "
          f"full p95={res['full_decision']['p95_ms']}ms iters={n_iters} rss={res['rss_mb']}MB",
          flush=True)
    return 0


def main() -> int:
    ap_ = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--child", nargs="+", metavar="ARG")
    ap_.add_argument("--episodes", type=int, default=256)
    ap_.add_argument("--seed", type=int, default=20260929)
    ap_.add_argument("--k-train", type=int, default=256)
    ap_.add_argument("--report", default=None)
    args = ap_.parse_args()
    if args.child:
        kind = args.child[0]
        if kind == "train":            # train <arm> <workdir> <out>
            return _child_train(args.child[1], args.child[2], args.k_train, args.child[3])
        if kind == "lat":              # lat <arm> <device> <workdir> <out>
            return _child_latency(args.child[1], args.child[2], args.child[3], args.child[4])
        raise SystemExit(f"kind inconnu {kind!r}")

    workdir = tempfile.mkdtemp(prefix="recurrent-s2b-")   # tmp DEV uniquement
    t0 = time.perf_counter()
    _generate_couples(os.path.join(workdir, "couples.jsonl"), n=args.episodes, seed=args.seed)
    shutil.copy(_DEV_STORE, os.path.join(workdir, "store.json"))
    gen_s = time.perf_counter() - t0
    couples_sha = _sha256_file(os.path.join(workdir, "couples.jsonl"))
    print(f"[fixture] {args.episodes} ép. seed={args.seed} gen={gen_s:.2f}s sha={couples_sha[:16]}",
          flush=True)

    runs = {}
    for arm in ("b144", "rec"):
        out = os.path.join(workdir, f"train-{arm}.json")
        cmd = [sys.executable, str(Path(__file__).resolve()), "--child", "train", arm, workdir, out,
               "--k-train", str(args.k_train)]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            raise SystemExit(f"train {arm} échec:\n{p.stderr[-1500:]}")
        runs[f"train_{arm}"] = json.loads(Path(out).read_text())
        print(p.stdout.strip(), flush=True)

    for arm in ("b144", "rec"):
        for dev in ("cpu", "gpu"):
            out = os.path.join(workdir, f"lat-{arm}-{dev}.json")
            cmd = [sys.executable, str(Path(__file__).resolve()), "--child", "lat", arm, dev, workdir, out]
            p = subprocess.run(cmd, capture_output=True, text=True)
            if p.returncode != 0:
                raise SystemExit(f"lat {arm}/{dev} échec:\n{p.stderr[-1500:]}")
            runs[f"lat_{arm}_{dev}"] = json.loads(Path(out).read_text())
            print(p.stdout.strip(), flush=True)

    rec_t, b144_t = runs["train_rec"], runs["train_b144"]
    report = {
        "kind": "recurrent-s2b-profile",
        "deliverable": "a2a1af8 (RecurrentRefine T=32 + factiorelle 2x2)",
        "fixture": {"episodes": args.episodes, "seed": args.seed, "gen_wall_s": round(gen_s, 2),
                    "couples_sha256": couples_sha,
                    "note": "DEV généré à la volée — aucun test2/scellé"},
        "training": {"phases": PHASES, "batch_size": 64, "k_train": args.k_train,
                      "b144": b144_t, "rec": rec_t,
                      "ratio_full_p50_rec_over_b144": round(
                          rec_t["full_update"]["measured"]["p50_ms"] /
                          b144_t["full_update"]["measured"]["p50_ms"], 3),
                      "ratio_kernel_p50_rec_over_b144": round(
                          rec_t["kernel"]["measured"]["p50_ms"] /
                          b144_t["kernel"]["measured"]["p50_ms"], 3)},
        "latency": {k: v for k, v in runs.items() if k.startswith("lat_")},
        "gate_s2b_p95_model_le_20ms": {
            "rec_cpu": runs["lat_rec_cpu"]["gates"]["p95_model_le_20ms"],
            "rec_gpu": runs["lat_rec_gpu"]["gates"]["p95_model_le_20ms"],
            "b144_cpu": runs["lat_b144_cpu"]["gates"]["p95_model_le_20ms"],
            "b144_gpu": runs["lat_b144_gpu"]["gates"]["p95_model_le_20ms"],
        },
        "limits": [
            "modèle rec fraîche init (pas de pré-entraînement) — latence/temps indépendants de l'état",
            "CPU pour l'entraînement (régime officiel) ; latence CPU+GPU publiés côte à côte (§10.3)",
            "fixture DEV bornée (256 ép.) ; machine partagée (charge échantillonnée par phase)",
            "report-only, aucune optimisation, aucun seuil figé",
        ],
    }
    print(f"[ratio] per-update rec/b144 (full p50) = "
          f"{report['training']['ratio_full_p50_rec_over_b144']} | "
          f"latence rec CPU p95={runs['lat_rec_cpu']['model_only']['p95_ms']}ms "
          f"GPU p95={runs['lat_rec_gpu']['model_only']['p95_ms']}ms", flush=True)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"rapport → {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
