"""Pipeline M0 WS-B — génération → splits → records scellés → inventaire.

Orchestration (aucune modification de ucm/data/generate.py, WS-A) :

1. **Layouts** : ``generate_layout`` seedé jusqu'à remplir les pools (les
   groupes isomorphes sont dédupliqués par le manifest ; on régénère tant
   qu'un pool est sous-dimensionné — jamais de remplissage silencieux).
2. **Manifest de splits** : ``make_split_manifest`` (groupes atomiques, zéro
   isomorphisme inter-pools, scellé par seed).
3. **Épisodes par pool** : ``generate_episode`` (WS-A) avec, pour train/val :
   rejet explicite des tâches réservées G2 « AT(clé, jonction deg≥3) » (§7.4)
   et assignation ``provenance.split`` = pool du layout.
4. **Écriture** : JSONL via ``JsonlRecordWriter`` (dédup sémantique + visites).
5. **Inventaire** : ``build_inventory`` → ``artifacts/inventory-M0.json`` +
   sonde G4 (bande d* 13–24, §4.5) — verdict publié, jamais inventé.

Usage :
    .venv/bin/python -m ucm.data.pipeline --seed 1007 --out-artifacts artifacts \
        [--train 200] [--val 50] [--test-min 100] [--episodes-per-layout 3] \
        [--g4-probe 300]
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Optional

from ..env.oracle import LayoutOracle
from ..env.tinygraph import Layout
from . import generate as gen
from .inventory import build_inventory, write_inventory
from .reader import iter_records
from .schema import record_from_dict
from .splits import (
    g2_components_check,
    is_g2_reserved,
    layout_info,
    layout_to_dict,
    make_split_manifest,
    write_manifest,
)
from .writer import JsonlRecordWriter

PIPELINE_VERSION = "0.1.0"


def _fill_pools(seed: int, train_n: int, val_n: int, min_test: int,
                rooms_range: tuple[int, int], bridge_fraction: float,
                max_rounds: int = 40):
    """Génère des layouts jusqu'à ce que le manifest puisse être scellé.

    Retourne (manifest_result, layouts). Chaque round ajoute des layouts
    frais (seed dérivé) ; les groupes isomorphes réduisent le besoin.
    """
    rng = random.Random(seed)
    layouts: list[Layout] = []
    seen: set[str] = set()
    result = None
    for round_no in range(max_rounds):
        batch = [
            gen.generate_layout(rng, rng.randint(*rooms_range), bridge_fraction)
            for _ in range(max(train_n, val_n, min_test) // 2 + 10)
        ]
        # layouts 9-12 pièces (cellules G3/G4) — même flux seedé
        batch += [
            gen.generate_layout(rng, rng.randint(9, 12), bridge_fraction)
            for _ in range(min_test // 3 + 10)
        ]
        for layout in batch:
            h = layout.layout_hash()
            if h not in seen:
                seen.add(h)
                layouts.append(layout)
        try:
            result = make_split_manifest(
                layouts, seed=seed, train_n=train_n, val_n=val_n, min_test=min_test,
                min_test_g34=min_test,
            )
            return result, layouts
        except ValueError:
            continue
    raise RuntimeError(
        f"pools non remplis après {max_rounds} rounds : "
        f"{len(layouts)} layouts distincts — étendre la génération"
    )


POOL_G2_MODE = {
    "train": "exclude", "val": "exclude", "test_g1": "exclude",
    "test_g2": "require", "test_g3": "exclude", "test_g4": "exclude",
}
POOL_BAND = {"test_g4": (13, 24)}  # les autres : bande train (2, 12)


def generate_pool_records(
    layouts_by_pool: dict[str, list[Layout]],
    *,
    seed: int,
    episodes_per_layout: dict[str, int],
    d_star_band: tuple[int, int] = (2, 12),
    g2_filter_pools: tuple[str, ...] = ("train", "val"),
) -> tuple[list[dict], dict]:
    """Génère les épisodes POOL PAR POOL — miroir exact de ``generate()`` 0.2.0.

    Contrat tagi-1 (re-scellement GATE-0) :
    - UN SEUL appel de boucle par pool avec le TOTAL d'épisodes
      (epl[pool] × len(layouts)) — jamais par-layout ;
    - quota d*=0 pool-level : max(1, round(total × 0.10)), premier arrivé ;
    - cycle forcé REACH/HAVE/AT (équilibrage §4.2) ; « AT » si g2-require ;
    - g2_mode par cellule : exclude (train/val/g1/g3/g4), require (test_g2) ;
    - test_g4 vise la bande d* 13–24 ;
    - episode_ref = f"{pool}-{seed}-{i:05d}".
    """
    rng = random.Random(seed)
    stats = gen.GenerationStats()
    oracle_cache: dict = {}
    records: list[dict] = []
    for pool, layouts in sorted(layouts_by_pool.items()):
        epl = episodes_per_layout.get(pool, 0)
        if epl <= 0 or not layouts:
            continue
        total = epl * len(layouts)
        g2_mode = POOL_G2_MODE.get(pool, "exclude")
        band = POOL_BAND.get(pool, d_star_band)
        zero_quota = max(1, round(total * gen.D_STAR_ZERO_FRACTION)) if total >= 1 else 0
        zero_done = 0
        pred_cycle = ["REACH", "HAVE", "AT"]
        for i in range(total):
            layout = layouts[rng.randrange(len(layouts))]
            # garde POOL (re-scellement final, tagi-5 §7.4) : d*=0 autorisé
            # UNIQUEMENT train/val (val 10.00% documenté, choix tagi-5) ;
            # TOUTE cellule test_* = 0 zéro initial — G-STOP est une cellule
            # séparée par design (pas de victoires gratuites en G1/G3).
            allow_zero = pool in ("train", "val") and zero_done < zero_quota
            forced = pred_cycle[i % 3] if g2_mode != "require" else "AT"
            ep_ref = f"{pool}-{seed}-{i:05d}"
            recs = gen.generate_episode(
                rng, layout, band, stats, False, oracle_cache,
                g2_mode=g2_mode, episode_ref=ep_ref,
                forced_pred=forced, allow_zero_now=allow_zero,
            )
            if recs is None:
                continue
            if len(recs) == 1:  # épisode INITIALEMENT satisfait (d0=0)
                zero_done += 1
            for r in recs:
                r["provenance"]["split"] = pool
            records.extend(recs)
    stats_summary = stats.summary()
    stats_summary["layouts_in_generation"] = sum(len(v) for v in layouts_by_pool.values())
    stats_summary["bridge_doors"] = sum(
        1 for pool_layouts in layouts_by_pool.values() for l in pool_layouts
        if l.meta.get("door_on_bridge")
    )
    stats_summary["pipeline_version"] = PIPELINE_VERSION
    return records, stats_summary


def probe_g4_feasibility(seed: int, layouts_n: int = 30, goals_per_layout: int = 3,
                         rooms_range: tuple[int, int] = (4, 8)) -> dict:
    """Sonde §4.5 : la bande d* 13–24 EXISTE-t-elle pour ces tailles ?

    Exact : pour (layout, but) échantillonnés, parcourt TOUTE la table de
    distances de l'oracle (API publique enumerate_states/reachable/d_star) et
    retient max d*. Compter un but comme « bande atteignable » si max d* ∈ 13–24.
    """
    from ..env.oracle import enumerate_states
    rng = random.Random(seed + 1)
    goals_with_band = 0
    goals_sampled = 0
    global_max = 0
    max_d_hist: dict[str, int] = {}
    unreachable_goals = 0
    for _ in range(layouts_n):
        layout = gen.generate_layout(rng, rng.randint(*rooms_range), 0.5)
        states = enumerate_states(layout)
        for _ in range(goals_per_layout):
            goal = gen._sample_goal(rng, layout)
            oracle = LayoutOracle(layout, goal)
            reachable_states = [st for st in states if oracle.reachable(st)]
            if not reachable_states:
                unreachable_goals += 1
                continue
            max_d = max(oracle.d_star(st) for st in reachable_states)
            goals_sampled += 1
            global_max = max(global_max, max_d)
            max_d_hist[str(max_d // 5 * 5)] = max_d_hist.get(str(max_d // 5 * 5), 0) + 1
            if 13 <= max_d <= 24:
                goals_with_band += 1
    return {
        "layouts_sampled": layouts_n,
        "goals_sampled": goals_sampled,
        "goals_with_band_13_24": goals_with_band,
        "goals_unreachable_everywhere": unreachable_goals,
        "global_max_d_star": global_max,
        "max_d_hist_bucket5": dict(sorted(max_d_hist.items())),
        "verdict": (
            f"EXISTE : {goals_with_band}/{goals_sampled} buts atteignent la bande 13–24 "
            f"(max global = {global_max})" if goals_with_band > 0
            else f"VIDE : 0/{goals_sampled} — max global = {global_max} ; revoir le générateur "
                 f"ou la bande avant d'ouvrir G4 (spec §4.5)"
        ),
    }


def run_m0_pipeline(
    *,
    seed: int,
    out_dir: str | Path,
    train_n: int = 200,
    val_n: int = 50,
    min_test: int = 100,
    episodes_per_layout: Optional[dict[str, int]] = None,
    g4_probe_attempts: int = 300,
    run_config: Optional[dict] = None,
) -> dict:
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    episodes_per_layout = episodes_per_layout or {
        "train": 3, "val": 2, "test_g1": 5, "test_g2": 5, "test_g3": 2, "test_g4": 2,
    }
    # §15.2 (reliquat lead 12:17) : la config EXACTE du run est publiée —
    # aucune inférence possible (divergence min_test 100 vs 155 observée).
    config = {
        "seed": seed,
        "train_n": train_n,
        "val_n": val_n,
        "min_test": min_test,
        "min_test_g34": min_test,
        "episodes_per_layout": dict(episodes_per_layout),
        "g4_probe_attempts": g4_probe_attempts,
        "bridge_fraction": 0.5,
        "rooms_train": [4, 8],
        "rooms_g34": [9, 12],
        "pool_g2_mode": dict(POOL_G2_MODE),
        "pool_band": {k: list(v) for k, v in POOL_BAND.items()},
        "pipeline_version": PIPELINE_VERSION,
        "generator_version": gen.GENERATOR_VERSION,
        "oracle_version": gen.ORACLE_VERSION,
        "zero_quota_rule": "pool-level max(1, round(total*0.10)), pools train+val only",
    }
    if run_config:
        config.update(run_config)
    (out / "run-config.json").write_text(json.dumps(config, sort_keys=True, indent=2), encoding="utf-8")

    # 1-2. layouts + manifest scellé
    result, layouts = _fill_pools(seed, train_n, val_n, min_test, (4, 8), 0.5)
    manifest_path = write_manifest(out / "split-manifest-M0.json", result.manifest)

    # 3. épisodes par pool
    layouts_by_pool: dict[str, list[LayoutInfo]] = {"train": result.train, "val": result.val}
    for info in result.test:
        pool = result.pool_of[info.layout_hash]
        layouts_by_pool.setdefault(pool, []).append(info)
    layouts_by_pool = {
        pool: [i.layout for i in infos] for pool, infos in layouts_by_pool.items()
        if episodes_per_layout.get(pool, 0) > 0
    }
    records, gen_stats = generate_pool_records(
        layouts_by_pool, seed=seed + 2, episodes_per_layout=episodes_per_layout
    )

    # 4. écriture JSONL (validation stricte + dédup par le writer)
    jsonl_path = out / "data" / "m0-transitions.jsonl"
    with JsonlRecordWriter(jsonl_path, append=False) as w:
        for r in records:
            w.write(record_from_dict(r))
        stats_obj = w.close()
    write_stats = stats_obj.to_dict()

    # tâches (layout, but) pour le contrôle G2 train/val
    tasks = [
        (record_from_dict(r).policy_input, r["policy_input"]["goal"], r["provenance"]["layout_hash"])
        for r in records
    ]
    layouts_by_hash = {l.layout_hash(): l for l in layouts}
    # tâches par ÉPISODE (premier record de chaque episode_ref) — comptage
    # m1 exact pour les composants G2
    train_val_tasks = []
    seen_episodes: set[str] = set()
    for r in records:
        pool = r["provenance"]["split"]
        if pool not in ("train", "val"):
            continue
        ep_ref = r["provenance"].get("episode_ref", "")
        if ep_ref in seen_episodes:
            continue
        seen_episodes.add(ep_ref)
        layout = layouts_by_hash[r["provenance"]["layout_hash"]]
        train_val_tasks.append((layout, r["policy_input"]["goal"]))
    g2_check = g2_components_check(train_val_tasks)

    # 5. sonde G4 (tailles train 4-8, puis tailles G3 9-12) + inventaire
    g4_probe = {
        "rooms_4_8": probe_g4_feasibility(seed, layouts_n=g4_probe_attempts, rooms_range=(4, 8)),
        "rooms_9_12": probe_g4_feasibility(seed, layouts_n=g4_probe_attempts, rooms_range=(9, 12)),
    }
    inv_records = list(iter_records(jsonl_path))
    layouts_map = {h: layout_to_dict(l) for h, l in layouts_by_hash.items()}
    (out / "layouts-M0.json").write_text(json.dumps(layouts_map, sort_keys=True, indent=2))
    inventory = build_inventory(
        inv_records,
        layouts=layouts_by_hash,
        tasks=train_val_tasks,
        generator_stats=gen_stats,
        timing={"seconds_total": round(time.time() - t0, 2)},
    )
    inventory["g4_probe"] = g4_probe
    inventory["g2"]["verdict"] = (
        "OK — composants présents, réservation intacte" if g2_check["present"]
        else "INSUFFISANT — voir components_missing/reserved_leak (spec §7.4, avant gel)"
    )
    inventory["g2"]["reserved_leak"] = g2_check["reserved_leak"]
    inventory["g2"]["components"] = g2_check["counts"]
    from .writer import seal_manifest
    inventory = seal_manifest(inventory)
    inv_path = write_inventory(inventory, out / "inventory-M0.json")

    return {
        "manifest": manifest_path,
        "transitions": jsonl_path,
        "inventory": inv_path,
        "write_stats": write_stats,
        "gen_stats": gen_stats,
        "g4_probe": g4_probe,
        "g2": g2_check,
        "seconds": round(time.time() - t0, 2),
    }


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=1007)
    ap.add_argument("--out-artifacts", default="artifacts")
    ap.add_argument("--train", type=int, default=200)
    ap.add_argument("--val", type=int, default=50)
    ap.add_argument("--test-min", type=int, default=100)
    ap.add_argument("--episodes-per-layout", default="train:3,val:2,test_g1:5",
                    help="ex. train:3,val:2,test_g1:5")
    ap.add_argument("--g4-probe", type=int, default=300)
    args = ap.parse_args(argv)

    epl = {}
    for part in args.episodes_per_layout.split(","):
        pool, n = part.split(":")
        epl[pool] = int(n)

    summary = run_m0_pipeline(
        seed=args.seed,
        out_dir=args.out_artifacts,
        train_n=args.train,
        val_n=args.val,
        min_test=args.test_min,
        episodes_per_layout=epl,
        g4_probe_attempts=args.g4_probe,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
