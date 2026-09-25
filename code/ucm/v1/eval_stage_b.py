"""ÉVAL STAGE-B — LA lecture unique du test2 scellé (GO étape 4, 11:27).

Protocole (lead 11:27, pré-enregistré):
  1. ONE-READ des 600 épisodes scellés via SealedOpenRegistry, hash ANCRÉ
     depuis le POINTEUR publié (artifacts/test2-sealed/test2.pointer) —
     l'ancre n'est JAMAIS recalculée localement.
  2. Par checkpoint (arm, train_seed, k) des 120 cellules officielles
     (k ∈ freeze k_plan — les fichiers k1/k2 de pollution test sont ignorés):
     npz → modèle → SIWModelPolicy → run_episode §9.4 (horizon 64, STOP
     natif) sur les 600 épisodes, mx.random.seed(eval_seed + train_seed).
  3. RAW PAR ÉPISODE PERSISTÉ D'ABORD: JSONL append (schéma exact du lead),
     AVANT toute agrégation. Reprise: une cellule avec 600 lignes est sautée.
  4. Métriques par cellule + verdict_v1bis(raw) — verdict par k (les trois
     publiés côte à côte, aucun choix de k n'est pré-enregistré dans le
     freeze: pas de cherry-picking possible).
  5. Publication chemin neuf O_EXCL + bundle/pointer v9.

Le test2 n'est JAMAIS ouvert par les tests (données synthétiques) — la
lecture unique appartient à CE module, process dédié, token GO requis.
"""
from __future__ import annotations

import json
import os
import time

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_FREEZE = os.path.join(_REPO, "artifacts/freeze-v1bis-official-v10.json")
_POINTER = os.path.join(_REPO, "artifacts/test2-sealed/test2.pointer")
_EPISODES = os.path.join(_REPO, "artifacts/test2-sealed/test2-episodes.jsonl")
_CKPT_DIR = os.path.join(_REPO, "artifacts/v1bis-run/cells")
_OUT_DIR = os.path.join(_REPO, "artifacts/v1bis-eval")

RAW_FIELDS = ("arm", "train_seed", "eval_seed", "k", "episode_id",
              "layout_hash", "goal_type", "success", "outcome", "length",
              "n_invalid", "goal_reached_without_stop", "d_star", "L_star")


def pointer_sha(pointer_path: str = _POINTER) -> str:
    """ANCRE: sha du membre test2-episodes.jsonl LU depuis le pointeur
    publié. Jamais recalculé — c'est l'ancre qui vérifie le fichier."""
    ptr = json.load(open(pointer_path, encoding="utf-8"))
    for m in ptr["members"]:
        if m["name"] == "test2-episodes.jsonl":
            return m["sha256"]
    raise RuntimeError("pointer has no test2-episodes.jsonl member")


def official_cells(freeze: dict) -> list[tuple[str, int, int]]:
    """(arm, seed, k) des cellules OFFICIELLES — uniquement les k du freeze
    (la pollution k1/k2 des tests est exclue par construction)."""
    arms = sorted(freeze["arms"].keys())
    seeds = freeze["protocol"]["seeds"]
    k_plan = freeze["protocol"]["k_plan_exec_order"]
    return [(a, s, k) for a in arms for s in seeds for k in k_plan]


def evaluate(freeze_path: str = _FREEZE, pointer_path: str = _POINTER,
             episodes_path: str = _EPISODES, ckpt_dir: str = _CKPT_DIR,
             out_dir: str = _OUT_DIR, ts: str | None = None,
             run: bool = True, expected_n: int = 600) -> dict:
    """Éval Stage-B complète. run=False → plan seul (aucune lecture)."""
    import hashlib
    fm = json.load(open(freeze_path, encoding="utf-8"))
    eval_seed = fm["protocol"]["eval_seed"]
    cells = official_cells(fm)

    # tous les checkpoints existent (fail-closed avant TOUTE lecture)
    missing = [f"{a}-s{s}-k{k}.npz" for a, s, k in cells
               if not os.path.exists(os.path.join(ckpt_dir, f"{a}-s{s}-k{k}.npz"))]
    if missing:
        raise RuntimeError(f"missing checkpoints: {missing[:5]}… ({len(missing)})")

    if not run:
        return {"status": "plan", "n_cells": len(cells), "eval_seed": eval_seed}

    # token GO — c'est LA lecture unique
    if os.environ.get("V1BIS_STEP4_GO") != "LEAD_APPROVED":
        raise RuntimeError("STAGE-B EVAL LOCKED: V1BIS_STEP4_GO=LEAD_APPROVED required")

    os.makedirs(out_dir, exist_ok=True)
    ts = ts or time.strftime("%Y%m%dT%H%M%S")
    raw_path = os.path.join(out_dir, f"stage-b-raw-{ts}.jsonl")

    # ---- 1. ONE-READ (process-dedicated registry) ----
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    from ucm.v1.sealed_reader import SealedOpenRegistry
    sha_anchor = pointer_sha(pointer_path)
    reg = SealedOpenRegistry()
    fmt, lines, reader_sha = reg.read_once(episodes_path, expected_hash=sha_anchor)
    if reader_sha != sha_anchor:
        raise RuntimeError("sealed episodes sha != pointer anchor")
    episodes = [json.loads(l) for l in lines if l.strip()]
    if len(episodes) != expected_n:
        raise RuntimeError(f"expected {expected_n} sealed episodes, got {len(episodes)}")
    print(f"[stage-b] ONE-READ ok: 600 episodes, anchor {sha_anchor[:12]}…")

    # ---- 3. raw append-first (reprise: cellule complète ⇒ sautée) ----
    done = {}
    if os.path.exists(raw_path):
        for line in open(raw_path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                key = (r["arm"], r["train_seed"], r["k"])
                done[key] = done.get(key, 0) + 1

    import mlx.nn as nn
    from ucm.model.siw_model import make_siw_model
    from ucm.v1.policy_siw import SIWModelPolicy
    from ucm.eval.rollout import run_episode as _run_episode
    from ucm.v1.runner import _build_env

    todo = [(a, s, k) for a, s, k in cells if done.get((a, s, k), 0) < len(episodes)]
    print(f"[stage-b] {len(cells) - len(todo)}/{len(cells)} cells done, {len(todo)} to go")

    t0 = time.time()
    with open(raw_path, "a", encoding="utf-8") as raw_fh:
        for ci, (arm, seed, k) in enumerate(todo):
            model = make_siw_model()
            w = mx.load(os.path.join(ckpt_dir, f"{arm}-s{seed}-k{k}.npz"))
            model.update(nn.utils.tree_unflatten(list(w.items())))
            mx.eval(model.parameters())
            pol = SIWModelPolicy(model, name=f"{arm}-s{seed}-k{k}",
                                 seed=eval_seed + seed)
            mx.random.seed(eval_seed + seed)
            for e in episodes:
                env = _build_env(e)
                res = _run_episode(env, pol, e["episode_id"], eval_seed,
                                   pol.name, d_star=e.get("d_star"))
                rec = {
                    "arm": arm, "train_seed": seed, "eval_seed": eval_seed,
                    "k": k, "episode_id": e["episode_id"],
                    "layout_hash": e["layout_hash"],
                    "goal_type": e["task"]["goal"]["predicate"],
                    "success": bool(res.success), "outcome": res.outcome,
                    "length": res.length, "n_invalid": res.n_invalid,
                    "goal_reached_without_stop": bool(res.goal_reached_without_stop),
                    "d_star": e.get("d_star"),
                    "L_star": (e.get("d_star") + 1) if e.get("d_star") is not None else None,
                }
                raw_fh.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
            raw_fh.flush()
            print(f"[stage-b] {ci+1}/{len(todo)} {arm}-s{seed}-k{k} "
                  f"({time.time()-t0:.0f}s)")

    return _aggregate_and_publish(raw_path, fm, ts, out_dir, freeze_path)


def _aggregate_and_publish(raw_path: str, fm: dict, ts: str,
                           out_dir: str, freeze_path: str) -> dict:
    """4+5: métriques par cellule, verdicts par k, publication O_EXCL."""
    import hashlib
    from ucm.data.verdict_v1bis import verdict_v1bis

    # relecture du raw LOCAL (ce n'est pas le scellé — le raw nous appartient)
    per_cell = {}   # (arm, seed, k) -> {pred: [success…]}
    for line in open(raw_path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        key = (r["arm"], r["train_seed"], r["k"])
        per_cell.setdefault(key, {}).setdefault(r["goal_type"], []).append(
            1.0 if r["success"] else 0.0)

    cells_meta = {}
    for (arm, seed, k), preds in sorted(per_cell.items()):
        cells_meta[f"{arm}|s{seed}|k={k}"] = {
            p: {"rate": sum(v) / len(v), "n": len(v)}
            for p, v in sorted(preds.items())
        }

    # verdict par k (les TROIS publiés — le freeze ne pré-enregistre pas de k)
    verdicts = {}
    for k in fm["protocol"]["k_plan_exec_order"]:
        raw_v = {"pretrained": {}, "scratch": {}}
        for (arm, seed, kk), preds in per_cell.items():
            if kk != k:
                continue
            if arm == "pretrained_TGK":
                raw_v["pretrained"][str(seed)] = {
                    p: sum(v) / len(v) for p, v in preds.items()}
            elif arm == "scratch":
                raw_v["scratch"][str(seed)] = {
                    p: sum(v) / len(v) for p, v in preds.items()}
        vpath = os.path.join(out_dir, f"stage-b-verdict-raw-k{k}-{ts}.json")
        _fd = os.open(vpath, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        with os.fdopen(_fd, "w") as fh:
            json.dump(raw_v, fh, sort_keys=True, indent=1)
        verdicts[f"k={k}"] = verdict_v1bis(vpath)

    # publication O_EXCL: métriques + dump
    metrics = {"n_cells": len(per_cell), "cells": cells_meta,
               "verdicts": verdicts, "ts": ts}
    mpath = os.path.join(out_dir, f"stage-b-metrics-{ts}.json")
    _fd = os.open(mpath, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(_fd, "w") as fh:
        json.dump(metrics, fh, sort_keys=True, indent=1, default=str)

    # bundle/pointer v9 (crash-atomic) — raw + metrics + verdicts
    from ucm.v1.atomic_publish import BundleWriter, capability_probe
    capability_probe(out_dir)
    bw = BundleWriter(out_dir, "stage-b-eval")
    bw.add_bytes("stage-b-raw.jsonl", open(raw_path, "rb").read())
    bw.add_bytes("stage-b-metrics.json", open(mpath, "rb").read())
    for k in fm["protocol"]["k_plan_exec_order"]:
        vp = os.path.join(out_dir, f"stage-b-verdict-raw-k{k}-{ts}.json")
        bw.add_bytes(f"stage-b-verdict-raw-k{k}.json", open(vp, "rb").read())
    manifest = bw.finalize()
    bw.publish_pointer(os.path.join(out_dir, f"stage-b-eval-{ts}.pointer"))

    return {"status": "official_complete", "n_cells": len(per_cell),
            "verdicts": verdicts, "raw": raw_path, "metrics": mpath,
            "bundle": manifest}


if __name__ == "__main__":
    import sys
    r = evaluate(run="--plan" not in sys.argv)
    print(json.dumps(r, indent=2, default=str))
