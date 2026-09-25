"""Inventaire de faisabilité WS-B (spec §7.2, §4.5, §7.4) — AVANT tout gel.

Compte, à partir des transitions JSONL (+ layouts optionnels) :
- layouts distincts et groupes isomorphes (si layouts fournis) ;
- transitions brutes, couples uniques (layout, état, but), couples-action uniques ;
- expositions à l'optimiseur (visites dédupliquées) ;
- actions optimales annotées et cardinalité de A* (histogramme) ;
- histogramme d* : bandes 0 / 1 / 2–12 / 13–24 / >24 / unreachable — la bande
  13–24 répond à la question de faisabilité G4 (§4.5 : inventaire AVANT de
  créer/ouvrir le test) ;
- composants G2 présents au train (§7.4) et disponibilité de jonctions.

L'artefact publié est scellé (manifest_sha256) — ``artifacts/inventory-M0.json``.

CLI :
    .venv/bin/python -m ucm.data.inventory --jsonl t.jsonl \
        [--layouts layouts.json] [--tasks tasks.json] [--out artifacts/inventory-M0.json]

    layouts.json : {layout_hash: {"rooms": [...], "edges": [[a,b],...], "door_edge": i}}
    tasks.json   : [{"layout_hash": ..., "goal": {...}}, ...]  (tâches train/val)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from ..env.tinygraph import Layout
from .reader import iter_records
from .schema import Record
from .splits import (
    g2_components_check,
    group_isomorphic,
    layout_from_dict,
    layout_info,
    is_g2_reserved,
)
from .writer import seal_manifest, verify_manifest

INVENTORY_SCHEMA = "ucm-inventory/0.1"


def d_star_band(d_star: Optional[int]) -> str:
    if d_star is None:
        return "unreachable"
    if d_star == 0:
        return "0"
    if d_star == 1:
        return "1"
    if 2 <= d_star <= 12:
        return "2-12"
    if 13 <= d_star <= 24:
        return "13-24"
    return ">24"


def build_inventory(
    records: Iterable[Record],
    *,
    layouts: Optional[Mapping[str, Layout]] = None,
    tasks: Optional[Iterable[tuple[Layout, dict]]] = None,
    generator_stats: Optional[Mapping[str, Any]] = None,
    timing: Optional[Mapping[str, Any]] = None,
) -> dict:
    """Construit l'inventaire scellé. ``records`` : Record déjà validés (reader).

    ``timing`` (m3) : mesures non reproductibles — stockées hors empreinte
    (clé ``timing``, exclue du scellage) pour ne pas casser la vérification
    byte-identical inter-machines.
    """
    n_records = 0
    couples: set[tuple[str, str]] = set()
    couple_actions: set[tuple[str, str, tuple[Any, ...]]] = set()
    exposures = 0
    d_hist: dict[str, int] = {}
    a_star_hist: dict[int, int] = {}
    optimal_annotated = 0
    layout_hashes: set[str] = set()
    episodes: set[Any] = set()
    splits_seen: dict[str, int] = {}
    sources_seen: dict[str, int] = {}
    ep_first: dict[Any, tuple[str, Optional[int], str]] = {}
    episode_order: list[Any] = []
    episode_refs_lines: set[Any] = set()

    for record in records:
        n_records += 1
        prov = record.provenance
        layout_hashes.add(prov.layout_hash)
        couples.add((prov.layout_hash, prov.state_goal_hash))
        action = record.policy_input.candidates[record.execution.action_ref]
        couple_actions.add((prov.layout_hash, prov.state_goal_hash, action.semantic()))
        visits = prov.extra.get("visits")
        exposures += len(visits) if visits else 1
        band = d_star_band(record.supervision.d_star if record.supervision.reachable else None)
        d_hist[band] = d_hist.get(band, 0) + 1
        if record.supervision.reachable:
            optimal_annotated += 1
            k = len(record.supervision.optimal_actions)
            a_star_hist[k] = a_star_hist.get(k, 0) + 1
        # episodes_distinct (passe finale tagi-5) : nombre d'episode_ref
        # DISTINCTS — le générateur émet « episode_ref », jamais « episode ».
        # Reliquat 12:16 : l'union inclut les episode_ref des VISITES fusionnées
        # (premier record d'un épisode mergé dans une ligne antérieure).
        ep_ref = prov.extra.get("episode_ref")
        if ep_ref is not None:
            episodes.add(ep_ref)
            episode_refs_lines.add(ep_ref)
        for visit in (prov.extra.get("visits") or []):
            if isinstance(visit, dict) and isinstance(visit.get("episode_ref"), str):
                episodes.add(visit["episode_ref"])
        splits_seen[prov.split] = splits_seen.get(prov.split, 0) + 1
        sources_seen[prov.source] = sources_seen.get(prov.source, 0) + 1
        # métriques par ÉPISODE (m1, re-scellement GATE-0) : premier record de
        # l'épisode (episode_ref) ⇒ d* initial, prédicat initial
        if ep_ref is not None and ep_ref not in ep_first:
            ep_first[ep_ref] = (prov.split, record.supervision.d_star,
                                record.policy_input.goal.predicate)
            episode_order.append(ep_ref)

    inv: dict[str, Any] = {
        "schema": INVENTORY_SCHEMA,
        "layouts_distinct": len(layout_hashes),
        "transitions_raw": n_records,
        "unique_couples_layout_state_goal": len(couples),
        "unique_couple_actions": len(couple_actions),
        "optimizer_exposures": exposures,
        "episodes_distinct": len(episodes),
        "episodes_distinct_lines": None,  # rempli après la boucle (reliquat tagi-5)
        "optimal_actions_annotated": optimal_annotated,
        "a_star_cardinality_hist": {str(k): a_star_hist[k] for k in sorted(a_star_hist)},
        "d_star_hist": {b: d_hist.get(b, 0) for b in ("0", "1", "2-12", "13-24", ">24", "unreachable")},
        "splits": dict(sorted(splits_seen.items())),
        "sources": dict(sorted(sources_seen.items())),
        # §4.5 : faisabilité G4 — la bande 13-24 doit être démontrée avant gel
        "g4_feasibility": {
            "band_13_24_couples": d_hist.get("13-24", 0),
            "verdict": "sufficient" if d_hist.get("13-24", 0) > 0 else "EMPTY — revoir le générateur avant test G4 (spec §4.5)",
        },
    }

    inv["episodes_distinct_lines"] = len(episode_refs_lines)

    # m1 (re-scellement GATE-0) : zéros INITIAUX et prédicats par pool,
    # mesurés par ÉPISODE (premier record) — pas par transition.
    if ep_first:
        per_pool: dict[str, dict[str, Any]] = {}
        for ep_ref in episode_order:
            split, d0, pred = ep_first[ep_ref]
            m = per_pool.setdefault(split, {"episodes": 0, "zero_initial": 0, "predicates": {}})
            m["episodes"] += 1
            if d0 == 0:
                m["zero_initial"] += 1
            m["predicates"][pred] = m["predicates"].get(pred, 0) + 1
        for split, m in per_pool.items():
            n = m["episodes"]
            m["zero_initial_pct"] = round(100.0 * m["zero_initial"] / n, 2) if n else 0.0
            m["predicate_pct"] = {
                p: round(100.0 * c / n, 2) for p, c in sorted(m["predicates"].items())
            } if n else {}
            del m["predicates"]
            pp = m["predicate_pct"]
            if split == "test_g2":
                m["predicate_balance"] = "by_design AT-only (g2 require)" if pp.get("AT") == 100.0 else "UNEXPECTED"
            elif split == "test_g4":
                # la bande 13-24 est physiquement dominée par AT (logistique
                # clé+colis) : REACH/HAVE dépassent rarement 12 — fallback du
                # générateur documenté, compté ; écart 30-37% structurel.
                m["predicate_balance"] = (
                    "structural_AT_dominance (documenté, fallback compté)" if pp.get("AT", 0) >= 60 else "UNEXPECTED"
                )
            else:
                m["predicate_balance"] = "balanced" if all(30 <= v <= 37 for v in pp.values()) else "OFF_TARGET"
        inv["episodes_metrics_per_pool"] = dict(sorted(per_pool.items()))
        zero_total = sum(m["zero_initial"] for m in per_pool.values())
        ep_total = sum(m["episodes"] for m in per_pool.values())
        inv["zero_initial_pct_global"] = round(100.0 * zero_total / ep_total, 2) if ep_total else 0.0

    if layouts:
        infos = [layout_info(l) for l in layouts.values()]
        groups = group_isomorphic(infos)
        inv["isomorphism_groups"] = len(groups)
        inv["groups_detail"] = [
            {"certificate": g[0].certificate, "n_rooms": g[0].n_rooms, "members": [m.layout_hash for m in g]}
            for g in groups
        ]
        junction_test = [
            info.layout_hash
            for info in infos
            if info.has_junction
        ]
        inv["junction_layouts"] = len(junction_test)

    if tasks is not None:
        task_list = list(tasks)
        g2 = g2_components_check(task_list)
        inv["g2"] = {
            "train_task_count": len(task_list),
            "components": g2["counts"],
            "components_missing": g2["missing"],
            "present": g2["present"],
            "reserved_leak": g2["reserved_leak"],
            "verdict": (
                "OK — composants présents, réservation intacte" if g2["present"]
                else "INSUFFISANT — voir components_missing/reserved_leak (spec §7.4, avant gel)"
            ),
        }
        inv["g2_reserved_tasks_available"] = sum(
            1 for layout, goal in task_list if is_g2_reserved(layout, goal)
        )

    if generator_stats:
        inv["generator"] = dict(generator_stats)  # {generated, accepted, rejected, reasons...}
    if timing:
        # m3 : hors empreinte — clé "timing" exclue du scellage
        inv["timing"] = dict(timing)

    return seal_manifest(inv)


def write_inventory(inventory: dict, path: str | Path) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(inventory, sort_keys=True, indent=2), encoding="utf-8")
    return str(p)


def _load_layouts(path: str | Path) -> dict[str, Layout]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {h: layout_from_dict(d) for h, d in raw.items()}


def _load_tasks(path: str | Path, layouts: Mapping[str, Layout]) -> list[tuple[Layout, dict]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [(layouts[t["layout_hash"]], t["goal"]) for t in raw]


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Inventaire de faisabilité UCM (WS-B)")
    parser.add_argument("--jsonl", required=True, nargs="+", help="fichiers transitions JSONL")
    parser.add_argument("--layouts", help="JSON {layout_hash: layout_dict}")
    parser.add_argument("--tasks", help="JSON [{layout_hash, goal}] (tâches train/val)")
    parser.add_argument("--out", default="artifacts/inventory-M0.json")
    parser.add_argument("--generator-stats", help="JSON stats générateur (optionnel)")
    parser.add_argument("--timing", help="JSON timings non reproductibles (optionnel, hors empreinte)")
    args = parser.parse_args(argv)

    records = [r for path in args.jsonl for r in iter_records(path)]
    layouts = _load_layouts(args.layouts) if args.layouts else None
    tasks = _load_tasks(args.tasks, layouts) if (args.tasks and layouts) else None
    gen = json.loads(Path(args.generator_stats).read_text()) if args.generator_stats else None
    timing = json.loads(Path(args.timing).read_text()) if args.timing else None

    inv = build_inventory(records, layouts=layouts, tasks=tasks, generator_stats=gen, timing=timing)
    out = write_inventory(inv, args.out)
    print(f"inventaire → {out}  (manifest_sha256={inv['manifest_sha256'][:16]}…)")
    print(json.dumps({k: v for k, v in inv.items() if k in ("transitions_raw", "unique_couples_layout_state_goal", "d_star_hist", "g4_feasibility")}, indent=2))
    if not verify_manifest(inv):
        print("ERREUR: manifest incohérent", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
