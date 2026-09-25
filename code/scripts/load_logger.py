#!/usr/bin/env python3
"""Journal de charge de la fenêtre étape 4 (OPS) — échantillonne loadavg + process cibles.

Écrit un JSONL (une ligne par échantillon) : horodatage, loadavg 1/5/15, et les process dont
la commande matche un motif (pid, %cpu, %mem, rss_kb). Sert de preuve « charge journalisée
pendant tout le run » (exigence lead 19:19:20). Lecture seule, aucun effet sur le run.

Usage :
  python scripts/load_logger.py --out reports/step4-load-<date>.jsonl --interval 30
  (Ctrl-C / kill pour arrêter ; --duration 0 = infini)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime


def _sample(pattern: re.Pattern) -> dict:
    procs = []
    try:
        out = subprocess.run(["ps", "aux"], capture_output=True, text=True, timeout=10).stdout
        for line in out.splitlines()[1:]:
            parts = line.split(None, 10)
            if len(parts) < 11:
                continue
            try:
                cpu = float(parts[2]); mem = float(parts[3]); rss = int(parts[5])
            except ValueError:
                continue
            if pattern.search(parts[10]):
                procs.append({"pid": parts[1], "cpu_pct": cpu, "mem_pct": mem,
                              "rss_kb": rss, "command": parts[10][:100]})
    except Exception:
        pass
    return {
        "t": datetime.now().isoformat(timespec="seconds"),
        "epoch": round(time.time(), 3),
        "load_1_5_15": [round(x, 2) for x in os.getloadavg()],
        "target_processes": procs,
    }


def main() -> int:
    ap_ = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--out", required=True)
    ap_.add_argument("--interval", type=float, default=30.0)
    ap_.add_argument("--duration", type=float, default=0.0, help="0 = infini")
    ap_.add_argument("--pattern", default=r"runner_official|runner_v1bis|python")
    args = ap_.parse_args()
    pattern = re.compile(args.pattern)
    t_end = (time.time() + args.duration) if args.duration > 0 else None
    with open(args.out, "a", encoding="utf-8") as fh:
        while True:
            fh.write(json.dumps(_sample(pattern)) + "\n")
            fh.flush()
            if t_end is not None and time.time() >= t_end:
                break
            time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    sys.exit(main())
