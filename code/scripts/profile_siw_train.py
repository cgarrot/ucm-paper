#!/usr/bin/env python3
"""M-V1b tagi-4 — profil comparatif CPU/GPU de l'entraînement SIW BATCHÉ (§10.2).

Usage:
  python scripts/profile_siw_train.py --device both            # profil officiel
  python scripts/profile_siw_train.py --device both --smoke    # validation (secondes)

Protocole (spec §10.2, régime entraînement) :
  - warm-up (compilation/caches) PUIS segment mesuré ≥200 updates PUIS segment
    SOUTENU (second segment, détection de throttling par comparaison des
    médianes) ;
  - DEUX mesures publiées côte à côte (précision tagi-1 21:24) :
      (a) « kernel » — noyau MLX/optimiseur seul (t0 après shuffle+collate+
          drain) : le débit du calcul pur ;
      (b) « full_update » — wall-clock complet d'un update (shuffle +
          collate_siw + drain + noyau + mx.eval), la mesure end-to-end ;
    + « shuffle_collate » (segment pré-noyau) et « init » (chargement données,
    init modèle/optimiseur) HORS boucle, séparés ;
  - chaque mesure forcée avec mx.eval() (MLX paresseux) et chronométrée APRÈS
    complétion ;
  - deux régimes (CPU, GPU) publiés côte à côte, AUCUNE attribution
    d'architecture (§10.3) ;
  - machine/OS/runtime/précision/batch/mémoire/power déclarés avec chaque
    rapport ; énergie non mesurée (pas de compteur système) → « non_mesuree ».

Note mesure (tagi-1 21:24) : collate_siw batch64 (N44/E52/K43) ≈7,8 ms/batch
CPU en DEV — le speedup end-to-end GPU est donc plus faible que le speedup
noyau ; c'est pourquoi les deux sont publiés.

La boucle d'entraînement est un MIRROIR EXACT de ucm/v1/runner.py::finetune
(mêmes opérations : shuffle, chunk, collate_siw, set_bc_loss, value_and_grad,
clip global, AdamW, mx.eval) — seule la chronométrage par update est ajoutée.

Nombres OFFICIELS : à relever sur machine calme (après verdict M-V1b), hors
contention — un profil mesuré sous contention n'est pas un profil.
Dépend du module runner ucm/v1/data_adapter.py (adaptateur scellés→records).

Sortie : <out>/cpu.json + <out>/gpu.json + <out>/summary.json + table stdout.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob as _glob
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.optimizers import AdamW
from mlx.utils import tree_map

from ucm.model.loss import set_bc_loss
from ucm.model.siw_model import make_siw_model
from ucm.model.train import _grad_global_norm
from ucm.v1.tensorize_siw import (collate_siw, labels_from_supervision_siw,
                                  tensorize_siw_obs)

DEFAULTS = {
    "couples_file": "artifacts/siw-couples-k2000.jsonl",
    "layout_store": "artifacts/inventory-siw-dev-layouts.json",
    "batch_size": 64, "warmup": 50, "measured": 200, "sustained": 200,
    "lr": 3e-4, "weight_decay": 1e-4, "clip_norm": 1.0, "seed": 0,
}
SMOKE = {"batch_size": 16, "warmup": 3, "measured": 10, "sustained": 10, "max_records": 64}
THROTTLE_FLAG_RATIO = 1.10  # médiane soutenu / médiane mesuré au-delà → suspect


def _git_rev() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _max_rss_bytes() -> int:
    import resource
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(v) if sys.platform == "darwin" else int(v) * 1024  # macOS: octets, Linux: Ko


def _power_source() -> str:
    try:
        if sys.platform == "darwin":
            out = subprocess.run(["pmset", "-g", "batt"], capture_output=True,
                                 text=True, timeout=5).stdout
            if "AC Power" in out:
                return "AC"
            if "Battery Power" in out:
                return "battery"
        else:
            for p in sorted(_glob.glob("/sys/class/power_supply/A*/online")) + \
                     sorted(_glob.glob("/sys/class/power_supply/ADP*/online")):
                return "AC" if open(p).read().strip() == "1" else "battery"
    except Exception:
        pass
    return "unknown"


def _top_cpu_processes(n: int = 5) -> list[dict]:
    """Top consommateurs CPU au moment du rapport (conditions de fenêtre)."""
    try:
        out = subprocess.run(["ps", "aux"], capture_output=True, text=True, timeout=5).stdout
        rows = []
        for line in out.splitlines()[1:]:
            parts = line.split(None, 10)
            if len(parts) < 11:
                continue
            try:
                cpu = float(parts[2])
            except ValueError:
                continue
            rows.append({"pid": parts[1], "cpu_pct": cpu, "command": parts[10][:80]})
        rows.sort(key=lambda r: -r["cpu_pct"])
        return rows[:n]
    except Exception:
        return []


def _pct(xs: list[float]) -> dict:
    s = sorted(xs)
    n = len(s)

    def p(pc):
        return s[min(n - 1, int(pc / 100 * n))]
    return {"p50_ms": round(p(50) * 1e3, 3), "p95_ms": round(p(95) * 1e3, 3),
            "p99_ms": round(p(99) * 1e3, 3), "mean_ms": round(statistics.fmean(s) * 1e3, 3),
            "n": n}


def load_examples(couples_file: str, layout_store: str, max_records: int = 0):
    from ucm.v1.data_adapter import couples_to_records
    records = couples_to_records(couples_file, layout_store)
    if max_records:
        records = records[:max_records]
    exs = []
    for rec in records:
        ex = tensorize_siw_obs(rec["policy_input"])
        ex["labels"] = labels_from_supervision_siw(
            rec["policy_input"], rec["supervision"]["optimal_actions"])
        exs.append(ex)
    return exs, len(records)


def run_regime(device: str, exs: list, cfg: dict) -> dict:
    """Boucle = miroir exact de runner.finetune, + chronométrage par update.

    Mesures par update : kernel (noyau seul), full (wall complet), shuffle_collate
    (segment pré-noyau). Init (modèle/optimiseur) chronométrée hors boucle.
    """
    mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    try:
        mx.random.seed(cfg["seed"])
    except Exception:
        pass

    t_init0 = time.perf_counter()
    model = make_siw_model()
    n_params = model.print_param_count()
    mx.eval(model.parameters())
    opt = AdamW(learning_rate=cfg["lr"], weight_decay=cfg["weight_decay"])
    model_init_s = time.perf_counter() - t_init0

    rng = random.Random(cfg["seed"])
    idx = list(range(len(exs)))
    phases = ("warmup", "measured", "sustained")
    times: dict[str, dict[str, list[float]]] = {
        ph: {"kernel": [], "full": [], "shuffle_collate": []} for ph in phases}

    for phase in phases:
        for _ in range(cfg[phase]):
            t_full0 = time.perf_counter()
            rng.shuffle(idx)
            chunk = [exs[i] for i in idx[:cfg["batch_size"]]] or exs
            batch = collate_siw(chunk)
            t_pre1 = time.perf_counter()
            mx.eval()  # drain (MLX paresseux)
            t_k0 = time.perf_counter()

            def loss_fn():
                return set_bc_loss(model(batch), batch["labels"], batch["cand_mask"])

            loss, grads = nn.value_and_grad(model, loss_fn)()
            gn = _grad_global_norm(grads)
            scale = mx.minimum(1.0, cfg["clip_norm"] / mx.maximum(gn, 1e-12))
            opt.update(model, tree_map(lambda g: g * scale, grads))
            mx.eval(loss, model.parameters())
            t_k1 = time.perf_counter()
            times[phase]["full"].append(t_k1 - t_full0)
            times[phase]["shuffle_collate"].append(t_pre1 - t_full0)
            times[phase]["kernel"].append(t_k1 - t_k0)

    med = {ph: statistics.median(times[ph]["full"]) for ph in phases if times[ph]["full"]}
    med_k = {ph: statistics.median(times[ph]["kernel"]) for ph in phases if times[ph]["kernel"]}
    ratio = med["sustained"] / med["measured"] if med.get("measured") and med.get("sustained") else None
    ratio_k = med_k["sustained"] / med_k["measured"] if med_k.get("measured") and med_k.get("sustained") else None
    return {
        "times": times,
        "n_params": n_params,
        "model_init_s": model_init_s,
        "throttle_ratio": ratio,
        "throttle_ratio_kernel": ratio_k,
    }


def report(device: str, cfg: dict, data_meta: dict, run: dict, data_load_s: float) -> dict:
    times = run["times"]
    med = {ph: statistics.median(times[ph]["full"]) for ph in times if times[ph]["full"]}
    med_k = {ph: statistics.median(times[ph]["kernel"]) for ph in times if times[ph]["kernel"]}
    ratio = run["throttle_ratio"]
    return {
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "git_rev": _git_rev(),
        "device": device,
        "runtime": {"python": sys.version.split()[0], "mlx": mx.__version__,
                    "numpy": np.__version__, "os": platform.platform(),
                    "precision": "fp32", "declared_device": str(mx.default_device())},
        "machine": {"max_rss_mb": round(_max_rss_bytes() / 1e6, 1),
                    "power_source": _power_source(), "energy": "non_mesuree"},
        "host_conditions": {
            "load_avg_1_5_15": [round(x, 2) for x in os.getloadavg()],
            "top_cpu_processes": _top_cpu_processes(5),
            "note": "fenêtre officielle = charge stable < ~1,5 sur plusieurs échantillons ; "
                    "un profil mesuré sous contention n'est pas un profil (§10.2)",
        },
        "protocol": {k: cfg[k] for k in ("warmup", "measured", "sustained", "batch_size",
                                         "lr", "weight_decay", "clip_norm", "seed")},
        "data": data_meta,
        "model": {"arch": "GNNB-SIW", "n_params": run["n_params"]},
        "padding_regime": "par batch (collate_siw — graphes SIW hétérogènes)",
        "init": {"data_load_s": round(data_load_s, 2),
                 "model_init_s": round(run["model_init_s"], 3)},
        "kernel": {ph: _pct(times[ph]["kernel"]) for ph in times},
        "shuffle_collate": {ph: _pct(times[ph]["shuffle_collate"]) for ph in times},
        "full_update": {ph: _pct(times[ph]["full"]) for ph in times},
        "warmup": _pct(times["warmup"]["full"]),
        "measured": _pct(times["measured"]["full"]),
        "sustained": _pct(times["sustained"]["full"]),
        "updates_per_s": {
            "kernel_measured": round(1 / med_k["measured"], 2) if "measured" in med_k else None,
            "full_measured": round(1 / med["measured"], 2) if "measured" in med else None,
            "full_sustained": round(1 / med["sustained"], 2) if "sustained" in med else None,
        },
        "throttling_check": {
            "median_ratio_sustained_over_measured": round(ratio, 4) if ratio else None,
            "median_ratio_kernel": round(run["throttle_ratio_kernel"], 4) if run["throttle_ratio_kernel"] else None,
            "flag_throttle_suspect": bool(ratio and ratio > THROTTLE_FLAG_RATIO),
            "method": f"médiane(soutenu)/médiane(mesuré) > {THROTTLE_FLAG_RATIO} ⇒ suspect ; "
                      "machine calme exigée (§10.2) — un profil sous contention n'est pas un profil",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="both", choices=["cpu", "gpu", "both"])
    ap.add_argument("--couples-file", default=DEFAULTS["couples_file"])
    ap.add_argument("--layout-store", default=DEFAULTS["layout_store"])
    ap.add_argument("--max-records", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    ap.add_argument("--warmup", type=int, default=DEFAULTS["warmup"])
    ap.add_argument("--measured", type=int, default=DEFAULTS["measured"])
    ap.add_argument("--sustained", type=int, default=DEFAULTS["sustained"])
    ap.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    ap.add_argument("--weight-decay", type=float, default=DEFAULTS["weight_decay"])
    ap.add_argument("--clip-norm", type=float, default=DEFAULTS["clip_norm"])
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    ap.add_argument("--smoke", action="store_true",
                    help="validation brève (secondes) : warmup=3, mesuré=10, soutenu=10, batch=16, 64 records")
    ap.add_argument("--out", default=None)
    ap.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.smoke:
        for k, v in SMOKE.items():
            setattr(args, k.replace("-", "_"), v)

    out = Path(args.out or f"artifacts/{_dt.datetime.now():%Y%m%d-%H%M%S}-siw-train-profile")
    out.mkdir(parents=True, exist_ok=True)

    if args.device == "both" and not args._child:
        # chaque régime dans SON processus : RSS propre + isolation GPU/CPU
        passthrough = [a for a in sys.argv[1:] if not a.startswith("--device") and a != "both"]
        codes = []
        for dev in ("cpu", "gpu"):
            cmd = [sys.executable, str(Path(__file__).resolve()), *passthrough,
                   "--device", dev, "--out", str(out), "--_child"]
            codes.append(subprocess.run(cmd).returncode)
        if any(codes):
            print(f"⚠ régime(s) en échec: {codes}", file=sys.stderr)
        reports = {}
        for dev in ("cpu", "gpu"):
            p = out / f"{dev}.json"
            if p.exists():
                reports[dev] = json.loads(p.read_text())
        if len(reports) == 2:
            (out / "summary.json").write_text(json.dumps(
                {"created": _dt.datetime.now().isoformat(timespec="seconds"),
                 "protocol_note": "deux régimes publiés côte à côte, aucune attribution d'architecture (§10.3)",
                 "smoke": bool(args.smoke), "cpu": reports["cpu"], "gpu": reports["gpu"]},
                indent=2), encoding="utf-8")
            print(f"\n== RÉSUMÉ ({'SMOKE — non officiel' if args.smoke else 'profil'}) ==")
            for dev in ("cpu", "gpu"):
                r = reports[dev]
                print(f"{dev:4s} kernel p50={r['kernel']['measured']['p50_ms']}ms "
                      f"| full p50={r['full_update']['measured']['p50_ms']}ms "
                      f"p95={r['full_update']['measured']['p95_ms']}ms "
                      f"| {r['updates_per_s']['full_measured']} upd/s full "
                      f"({r['updates_per_s']['kernel_measured']} upd/s kernel) "
                      f"| throttle={r['throttling_check']['flag_throttle_suspect']} "
                      f"(ratio {r['throttling_check']['median_ratio_sustained_over_measured']})")
            print(f"→ {out}/summary.json")
        return 0 if not any(codes) else 1

    # --- un seul régime, dans ce processus ---
    cfg = {k: getattr(args, k) for k in ("batch_size", "warmup", "measured", "sustained",
                                         "lr", "weight_decay", "clip_norm", "seed")}
    t_load0 = time.perf_counter()
    exs, n_records = load_examples(args.couples_file, args.layout_store, args.max_records)
    data_load_s = time.perf_counter() - t_load0
    data_meta = {"couples_file": args.couples_file, "layout_store": args.layout_store,
                 "n_records": n_records, "max_records_cap": args.max_records}
    run = run_regime(args.device, exs, cfg)
    rep = report(args.device, cfg, data_meta, run, data_load_s)
    (out / f"{args.device}.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"[{args.device}] kernel p50={rep['kernel']['measured']['p50_ms']}ms "
          f"full p50={rep['full_update']['measured']['p50_ms']}ms "
          f"upd/s full={rep['updates_per_s']['full_measured']} "
          f"rss={rep['machine']['max_rss_mb']}MB power={rep['machine']['power_source']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
