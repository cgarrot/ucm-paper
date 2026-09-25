"""Q4 (lead 05:59-0): départager la lenteur P2GNNB du reste + « lit le contexte ?».

(1) natif GNNB vs P2GNNB(λ=0) sur DROP-HAVE × 8 seeds — même données, même
    éval, seule l'architecture/tensorisation diffère;
(2) rééval des ckpts-run4 SANS contexte à la décision (D0-style obs pour
    les cellules D1-mixed/D1-shuffled) — discriminant gratuit « lit le
    contexte »: si SANS-ctx ≈ AVEC-ctx ⇒ le modèle ignore le contexte.
Persistance O_EXCL. Aucune interprétation — chiffres.
"""
from __future__ import annotations

import json
import os
import time

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_OUT = os.path.join(_REPO, "artifacts/p2-run/q4")
FAM = "DROP-HAVE-d2-12"


def _native_8seeds(records_by_seed, sp):
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    from ucm.model.gnn_b import GNNB
    from ucm.v1.runner import finetune
    from ucm.v1.transfer import CoverageTracker, FinetuneConfig
    from ucm.p2.runner import _evaluate_tgk_native_records
    out = []
    for seed in range(8):
        records = records_by_seed[seed]
        mx.random.seed(seed)
        m = GNNB(d=144)
        finetune(m, records, FinetuneConfig(updates=4000, seed=seed),
                 CoverageTracker(), f"q4-nat-{seed}", len(records),
                 trainable="all", world="tgk")
        s = _evaluate_tgk_native_records(m, sp)
        out.append(s)
    return out


def _p2_8seeds(records_by_seed, sp):
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    from ucm.p2.model_p2 import P2Model
    from ucm.p2.runner import train_cell, _evaluate_p2_simple
    out = []
    for seed in range(8):
        records = records_by_seed[seed]
        mx.random.seed(seed)
        m = P2Model(d=144, head_enabled=False)   # λ=0
        train_cell(m, records, 4000, seed)
        s = _evaluate_p2_simple(m, sp)
        out.append(s)
    return out


def _records_d0(seed):
    from ucm.p2.runner import load_dataset, build_arm_records
    _, sp = build_arm_records(load_dataset()["episodes"], FAM, "D0", seed)
    recs, _ = build_arm_records(load_dataset()["episodes"], FAM, "D0", seed)
    return recs, sp


def run_q4(out_dir=_OUT):
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    from ucm.p2.runner import load_dataset, build_arm_records
    _, sp = build_arm_records(load_dataset()["episodes"], FAM, "D0", 0)
    recs_by_seed = {}
    for seed in range(8):
        r, _ = build_arm_records(load_dataset()["episodes"], FAM, "D0", seed)
        recs_by_seed[seed] = r
    print("[q4] natif GNNB × 8 seeds…", flush=True)
    nat = _native_8seeds(recs_by_seed, sp)
    print("natif:", nat, flush=True)
    print("[q4] P2GNNB λ=0 × 8 seeds…", flush=True)
    p2 = _p2_8seeds(recs_by_seed, sp)
    print("p2:", p2, flush=True)

    # (2) ckpts-run4 SANS contexte
    from ucm.p2.runner import _evaluate_ckpt_without_context
    noctx = {}
    for arm in ("D0", "D1-mixed", "D1-shuffled"):
        rates = []
        for seed in range(8):
            ck = os.path.join(_REPO, "artifacts/p2-run/ckpts-run4",
                              f"{FAM}-{arm}-s{seed}.npz")
            rates.append(_evaluate_ckpt_without_context(ck, sp))
        noctx[arm] = rates
        print(f"sans-ctx {arm}:", rates, flush=True)

    art = {"ts": ts, "family": FAM, "updates": 4000,
           "native_gnnb_8seeds": nat, "p2gnnb_lambda0_8seeds": p2,
           "mean_native": sum(nat) / 8, "mean_p2": sum(p2) / 8,
           "run4_without_context": noctx,
           "mean_run4_noctx": {a: sum(v) / 8 for a, v in noctx.items()}}
    apath = os.path.join(out_dir, f"q4-{ts}.json")
    _fd = os.open(apath, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(_fd, "w") as fh:
        json.dump(art, fh, indent=1, sort_keys=True)
    art["persisted"] = apath
    return art


if __name__ == "__main__":
    print(json.dumps(run_q4(), indent=1, default=str))
