"""Train the CONTROL source checkpoints (FREEZE §2, validated 17:48): canon
B144 architecture pretrained on the canon TGK train pool with NON-INFORMATIVE
supervision (ucm.v1.control_source rule). One checkpoint per seed, same
regime as the canon gate2c runs. One-time setup (~1 min/seed).
"""

from __future__ import annotations

import argparse
import glob

import mlx.core as mx

mx.set_default_device(mx.cpu)

from ucm.model.train import TrainConfig, load_records, train as train_model
from ucm.v1.control_source import load_control_records, make_control_records
from ucm.v1.transfer import couple_hash

import json
import os
import random


def main(seeds: list[int]):
    os.makedirs("artifacts/v1", exist_ok=True)
    path = "artifacts/v1/control-source-records.jsonl"
    # LOAD-OR-REGENERATE + VALIDATION (audit 19:32: the declared variant rule
    # must be EXERCISED in production, not test-only): if the artifact exists
    # it is loaded THROUGH load_control_records (raises on any violation);
    # regeneration writes then immediately re-loads via the same validator.
    if os.path.exists(path):
        control = load_control_records(path)
        print(f"control records: LOADED+VALIDATED {len(control)} ← {path}")
    else:
        records = load_records("artifacts/data/m0-transitions.jsonl", split="train")
        control = make_control_records(records)
        with open(path, "w") as fh:
            for r in control:
                fh.write(json.dumps(r) + "\n")
        control = load_control_records(path)  # validate what we just wrote
        print(f"control records: GENERATED+VALIDATED {len(control)} → {path}")
    for s in seeds:
        existing = sorted(glob.glob(f"artifacts/*v1-control-s{s}"))
        if existing and os.path.exists(existing[-1] + "/checkpoint.npz"):
            print(f"seed {s}: exists → {existing[-1]}")
            continue
        cfg = TrainConfig(name=f"v1-control-s{s}", data=path, arch="B", d_model=144,
                          train_split=None, val_split=None, val_fraction=0.0,
                          max_updates=850, max_epochs=10, eval_every=200, seed=s)
        train_model(cfg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4")
    args = ap.parse_args()
    main([int(s) for s in args.seeds.split(",")])
