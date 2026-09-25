"""Coût intercalé — outil documenté (hors CI), deux variantes.

VARIANTE OFFICIELLE (défaut, ``--variant official``) — protocole FIGÉ #12
(tagi-1 13:31), statistique de décision des critères coût GATE-5 :
    - device CPU (meilleur régime des deux bras selon les tables publiées) ;
    - les DEUX bras compilés (``mx.compile``), batch 1, tenseurs représentatifs
      (K médian du diagnostic scellé, mêmes tenseurs pour les deux bras) ;
    - 6 blocs ; dans chaque bloc, alternance A,B,A,B,A,B = 6 segments de
      ``--decisions`` décisions, warm-up ``--warmup`` avant CHAQUE segment ;
    - p95 par segment ; p95 d'un bras dans un bloc = médiane de ses 3 segments ;
    - ratio du bloc = p95(B) / p95(A) ;
    - STATISTIQUE DE DÉCISION : médiane des ratios des blocs 2 à 6
      (bloc 1 = rampe thermique, publié mais écarté) ;
    - VERDICT : ≤ 2.00 PASS, > 2.00 FAIL — binaire, sans zone grise ;
      dispersion (min/max/IQR) publiée.

VARIANTE SECONDAIRE (``--variant secondary``) — corroboration NON-MARKING :
    5 répétitions intercalées A,B,A,B,... dans une même session (mêmes état
    thermique/machine pour les deux bras), warm-up partagé 300 puis 50 entre
    les mesures, ≥1000 décisions par répétition, médiane des p95 par bras,
    ratio = médiane(p95_B)/médiane(p95_A). Publiée à titre de corroboration ;
    elle ne remplace jamais la statistique officielle #12.

EXIGENCE DE FENÊTRE CALME : mesure machine-dépendante. À n'exécuter que sur une
machine au repos (aucun autre calcul lourd), après sondage du load ; publier
``load_start``/``load_end`` avec le résultat. C'est pourquoi l'outil n'est PAS
branché en CI : la périodicité tuerait la mesure.

Usage :
    .venv/bin/python -m ucm.eval.cost_interleaved \
        --run-a artifacts/<A-s4> --run-b artifacts/<B-s4> \
        --out /tmp/cost-interleaved.json [--variant official|secondary]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from ucm.eval.gate5_report import CANON, bench_decisions_compiled, load_canon_episodes, load_model
from ucm.model.tensorize import collate, tensorize_obs


def _median_k_batches(n_tensors: int = 8) -> tuple[list, int]:
    """Tenseurs représentatifs : K médian du diagnostic (100 premiers épisodes train)."""
    eps = load_canon_episodes(CANON, split="train", limit=100)
    obs_all = [rec["policy_input"] for (_, _, _, rs) in eps for rec in rs]
    ks = [len(o["candidates"]) for o in obs_all]
    med_k = int(np.median(ks))
    reps = [o for o in obs_all if len(o["candidates"]) == med_k][:n_tensors]
    batches = []
    for o in reps:
        ex = tensorize_obs(o)
        ex["labels"] = None
        batches.append(collate([ex]))
    return batches, med_k


def _set_device(device: str) -> None:
    import mlx.core as mx
    mx.set_default_device(mx.cpu if device == "cpu" else mx.gpu)


def measure_official(run_a: str, run_b: str, *, blocks: int = 6, segments: int = 3,
                     decisions: int = 1000, warmup: int = 50, device: str = "cpu",
                     n_tensors: int = 8) -> dict:
    """Protocole #12 figé — statistique OFFICIELLE (voir docstring module)."""
    import mlx.core as mx

    _set_device(device)
    t_start = time.time()
    load_start = os.getloadavg()
    mA, _ = load_model(run_a)
    mB, _ = load_model(run_b)
    batches, med_k = _median_k_batches(n_tensors)
    keys = [k for k in batches[0] if k != "labels"]
    cfA = mx.compile(lambda *a: mA(dict(zip(keys, a)))[0])
    cfB = mx.compile(lambda *a: mB(dict(zip(keys, a)))[0])

    def segment(cf) -> float:
        for i in range(warmup):
            mx.eval(cf(*[batches[i % len(batches)][k] for k in keys]))
        t = []
        for i in range(decisions):
            args = [batches[i % len(batches)][k] for k in keys]
            mx.eval()
            t0 = time.perf_counter()
            mx.eval(cf(*args))
            t.append(time.perf_counter() - t0)
        t.sort()
        return round(t[min(decisions - 1, int(0.95 * decisions))] * 1e3, 4)

    block_rows = []
    for b in range(1, blocks + 1):
        segA, segB, order = [], [], []
        for _ in range(segments):
            pA = segment(cfA)
            segA.append(pA)
            order.append(["A", pA])
            pB = segment(cfB)
            segB.append(pB)
            order.append(["B", pB])
        mA95, mB95 = statistics.median(segA), statistics.median(segB)
        block_rows.append({"block": b, "segments": order, "p95_A": mA95,
                           "p95_B": mB95, "ratio": round(mB95 / mA95, 4)})

    ratios = [bl["ratio"] for bl in block_rows[1:]]
    med = statistics.median(ratios)
    q1, q3 = np.percentile(ratios, [25, 75])
    return {
        "variant": "official (frozen #12)",
        "protocol": ("tagi-1 13:31 frozen: CPU, both mx.compile, batch1, median-K tensors, "
                     f"{blocks} blocks x (A,B)x{segments} x {decisions} decisions, "
                     f"warmup {warmup}/segment, block p95 = median of {segments} segments, "
                     "decision = median ratio blocks 2-6"),
        "checkpoints": {"A": run_a, "B": run_b},
        "device": device,
        "tensors": {"median_K": med_k, "n_representative": len(batches)},
        "blocks": block_rows,
        "decision": {"statistic": "median of block ratios 2..6", "value": round(med, 4),
                     "min": min(ratios), "max": max(ratios), "q1": round(float(q1), 4),
                     "q3": round(float(q3), 4), "iqr": round(float(q3 - q1), 4),
                     "threshold": 2.00, "verdict": "PASS" if med <= 2.0 else "FAIL"},
        "environment": {"load_start": load_start, "load_end": os.getloadavg(),
                        "wall_s": round(time.time() - t_start, 1),
                        "started": time.strftime("%H:%M:%S")},
    }


def measure_secondary(run_a: str, run_b: str, *, reps: int = 5, decisions: int = 1000,
                      warmup: int = 50, device: str = "cpu", n_tensors: int = 8) -> dict:
    """Variante 5 répétitions intercalées — corroboration NON-MARKING."""
    _set_device(device)
    t_start = time.time()
    load_start = os.getloadavg()
    mA, _ = load_model(run_a)
    mB, _ = load_model(run_b)
    # obs_list attendu par bench_decisions_compiled : observations brutes (K médian)
    eps = load_canon_episodes(CANON, split="train", limit=100)
    obs_all = [rec["policy_input"] for (_, _, _, rs) in eps for rec in rs]
    ks = [len(o["candidates"]) for o in obs_all]
    med_k = int(np.median(ks))
    obs_list = [o for o in obs_all if len(o["candidates"]) == med_k][:n_tensors]
    bench_decisions_compiled(mA, obs_list, warmup=300, n_measured=100)  # shared warm-up
    bench_decisions_compiled(mB, obs_list, warmup=50, n_measured=100)
    cells = {"A": [], "B": []}
    for _ in range(reps):
        cells["A"].append(bench_decisions_compiled(mA, obs_list, warmup=warmup,
                                                   n_measured=decisions)["p95_ms"])
        cells["B"].append(bench_decisions_compiled(mB, obs_list, warmup=warmup,
                                                   n_measured=decisions)["p95_ms"])
    medA, medB = statistics.median(cells["A"]), statistics.median(cells["B"])
    return {
        "variant": "secondary (5 interleaved reps, NON-MARKING)",
        "protocol": (f"A,B alternating in one calm session, both compiled, {reps} interleaved reps, "
                     f">={decisions} decisions, median p95 per arm"),
        "checkpoints": {"A": run_a, "B": run_b},
        "device": device,
        "tensors": {"median_K": med_k, "n_representative": len(obs_list)},
        "p95_A_ms_reps": cells["A"], "p95_B_ms_reps": cells["B"],
        "median_p95_A_ms": medA, "median_p95_B_ms": medB,
        "ratio": medB / medA, "PASS": medB / medA <= 2.0,
        "note": "corroboration non-marquante — la statistique officielle est la variante #12",
        "environment": {"load_start": load_start, "load_end": os.getloadavg(),
                        "wall_s": round(time.time() - t_start, 1),
                        "started": time.strftime("%H:%M:%S")},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-a", required=True)
    ap.add_argument("--run-b", required=True)
    ap.add_argument("--out", default="/tmp/cost-interleaved.json")
    ap.add_argument("--variant", default="official", choices=["official", "secondary"])
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--segments", type=int, default=3)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--decisions", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--tensors", type=int, default=8)
    args = ap.parse_args(argv)
    if args.variant == "official":
        res = measure_official(args.run_a, args.run_b, blocks=args.blocks,
                               segments=args.segments, decisions=args.decisions,
                               warmup=args.warmup, device=args.device,
                               n_tensors=args.tensors)
    else:
        res = measure_secondary(args.run_a, args.run_b, reps=args.reps,
                                decisions=args.decisions, warmup=args.warmup,
                                device=args.device, n_tensors=args.tensors)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    if args.variant == "official":
        d = res["decision"]
        print(f"[official #12] median ratio blocs 2-6 = {d['value']} "
              f"(min {d['min']} / max {d['max']} / IQR {d['iqr']}) -> {d['verdict']}")
    else:
        print(f"[secondary] ratio median p95 = {res['ratio']:.3f} -> "
              f"{'PASS' if res['PASS'] else 'FAIL'} (non-marking)")
    print(f"load_start={res['environment']['load_start']} load_end={res['environment']['load_end']} "
          f"wall={res['environment']['wall_s']}s -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
