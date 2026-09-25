"""Pipeline SIW (M-V1a) — inventaire de faisabilité AVANT gel (§4.5 SIW).

Compte, par layout et globalement :
- états-UI énumérables (siw_oracle.enumerate_states), par layout ;
- couples (layout, état, but) UNIQUES et leur répartition d* (tables d'oracle) ;
- couverture EMPIRIQUE aux budgets k = 0/100/500/2000/10000 (tirages
  sample_task — même distribution que le générateur d'épisodes) ;
- distribution des prédicats de but (théorique sur couples énumérés +
  empirique sur tirages) avec tolérance 18-32% (asymétrie VIEW documentée
  tagi-1 18:24 : buts VIEW intrinsèquement plus courts) ;
- existence de bandes d* : min/max observés → verdict de resserrage de bande
  AVANT gel (§4.5 : publier ce qui existe, ne pas inventer) ;
- groupes d'isomorphie de layouts (wl_hash_typed préfiltre + typed_isomorphic
  exact — labels exclus par construction).

Artéfact scellé (timing hors empreinte) : artifacts/inventory-siw-dev.json.
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any, Iterable, Optional

from ..env.siw import SIWLayout, sample_task
from ..env.siw_oracle import SIWOracle, enumerate_states
from .schema import sha256_hex
from .siw import (
    SIW_BUTTON_KINDS,
    SIW_LAYOUT_PREDS,
    SIW_WIDGET_TYPES,
    register_siw,
    siw_layout_isomorphic,
)
from .splits import wl_hash_typed
from .writer import seal_manifest, verify_manifest

SIW_INVENTORY_SCHEMA = "ucm-siw-inventory/0.1"
PRED_TOLERANCE = (18.0, 32.0)


def _typed_graph_from_layout(layout: SIWLayout):
    """(n, edges, node_colors) du layout SIW — couleurs type×onclick_kind
    (FSM §3.2, passe B tagi-5), labels exclus par construction."""
    register_siw()
    type_index = {t: i for i, t in enumerate(SIW_WIDGET_TYPES)}
    nodes: list[str] = list(layout.views) + list(layout.widgets.keys())
    idx = {eid: i for i, eid in enumerate(nodes)}
    node_colors = [type_index["view"]] * len(layout.views)
    for w in layout.widgets.values():
        if w["type"] == "button":
            kind = w.get("kind", "none")
            ki = SIW_BUTTON_KINDS.index(kind) if kind in SIW_BUTTON_KINDS else len(SIW_BUTTON_KINDS)
            node_colors.append(len(SIW_WIDGET_TYPES) + ki)
        else:
            node_colors.append(type_index.get(w["type"], len(type_index) + len(SIW_BUTTON_KINDS)))
    pred_index = {p: i for i, p in enumerate(SIW_LAYOUT_PREDS)}
    edges: list[tuple[int, int, int]] = []
    for a, b in layout.nav_edges:
        edges.append((idx[a], idx[b], pred_index["nav_edge"]))
    for w in layout.widgets.values():
        wid = idx[w["id"]]
        if w["type"] != "option" and w.get("view"):
            edges.append((wid, idx[w["view"]], pred_index["on_view"]))
        if w.get("form"):
            edges.append((wid, idx[w["form"]], pred_index["part_of"]))
        if w.get("select"):
            edges.append((wid, idx[w["select"]], pred_index["option_of"]))
        if w.get("submit_for"):
            edges.append((wid, idx[w["submit_for"]], pred_index["submits"]))
        if w.get("in_dialog"):
            edges.append((wid, idx[w["in_dialog"]], pred_index["in_dialog"]))
    return len(nodes), edges, node_colors


def _layout_iso_hash(layout: SIWLayout) -> str:
    """(tagi-5 re-audit, mineur 22:05) — IMPORTANT, à lire littéralement :
    _layout_iso_hash == wl_hash_typed(*_typed_graph_from_layout(layout)).
    C'est le PRÉFILTRE WL typé (16 hex, kind-aware, invariant de renumérotation) —
    ce N'est PAS une preuve isomorphique indépendante (il peut fusionner des
    graphes non-isomorphes). La GARANTIE de disjonction des pools repose sur
    ``typed_isomorphic`` (exact) ; ce hash ne sert qu'au bucketing/traçabilité."""
    return wl_hash_typed(*_typed_graph_from_layout(layout))


def group_siw_layouts(layouts: Iterable[SIWLayout]) -> list[list[SIWLayout]]:
    """Groupes d'isomorphie exacts (préfiltre WL + typed_isomorphic)."""
    buckets: dict[str, list[SIWLayout]] = {}
    for lay in layouts:
        buckets.setdefault(_layout_iso_hash(lay), []).append(lay)
    groups: list[list[SIWLayout]] = []
    for bucket in buckets.values():
        remaining = list(bucket)
        while remaining:
            head = remaining[0]
            g = [head]
            rest = []
            for other in remaining[1:]:
                g1 = _typed_graph_from_layout(head)
                g2 = _typed_graph_from_layout(other)
                from .splits import typed_isomorphic
                if typed_isomorphic(g1, g2):
                    g.append(other)
                else:
                    rest.append(other)
            groups.append(g)
            remaining = rest
    return groups


def _goal_key(goal: dict) -> tuple:
    return (goal["predicate"], tuple(sorted(goal["args"].items())))


def build_siw_inventory(
    layouts: list[SIWLayout],
    *,
    seed: int,
    goals_per_layout: int = 10,
    budgets: tuple[int, ...] = (0, 100, 500, 2000, 10000),
    d_band_proposed: tuple[int, int] = (2, 8),
    generation_params: Optional[dict] = None,
    covered_sets_dir: Optional[str] = None,
) -> dict:
    t0 = time.time()
    rng = random.Random(seed)
    register_siw()

    groups = group_siw_layouts(layouts)
    couples: set[tuple] = set()
    d_hist: dict[str, int] = {}
    d_max_global = 0
    d_min_global = 10**9
    unreachable_couples = 0
    unreachable_select_irrev = 0
    unreachable_other = 0
    states_total = 0
    pred_theoretical: dict[str, int] = {}
    layout_details: list[dict] = []

    for layout in layouts:
        l_hash = layout.layout_hash()
        states = enumerate_states(layout)
        states_total += len(states)
        # buts échantillonnés du même générateur que les épisodes
        goals: list[dict] = []
        seen_goals: set[tuple] = set()
        attempts = 0
        while len(goals) < goals_per_layout and attempts < goals_per_layout * 40:
            attempts += 1
            _, goal = sample_task(rng, layout)
            gk = _goal_key(goal)
            if gk in seen_goals:
                continue
            seen_goals.add(gk)
            goals.append(goal)
        d_max_lay = 0
        for goal in goals:
            oracle = SIWOracle(layout, goal)
            for st in states:
                key = (l_hash, st.key(), _goal_key(goal))
                if key in couples:
                    continue
                couples.add(key)
                pred_theoretical[goal["predicate"]] = pred_theoretical.get(goal["predicate"], 0) + 1
                if oracle.reachable(st):
                    d = oracle.d_star(st)
                    d_hist[str(d)] = d_hist.get(str(d), 0) + 1
                    d_max_global = max(d_max_global, d)
                    d_min_global = min(d_min_global, d)
                    d_max_lay = max(d_max_lay, d)
                else:
                    unreachable_couples += 1
                    d_hist["unreachable"] = d_hist.get("unreachable", 0) + 1
                    # propriété SELECT-irréversible (tagi-5, registre) :
                    # mauvais choix ⇒ CHOOSE(autre option) unreachable
                    if goal["predicate"] == "CHOOSE":
                        cur = st.chosen.get(goal["args"]["select"])
                        if cur is not None and cur != goal["args"]["option"]:
                            unreachable_select_irrev += 1
                        else:
                            unreachable_other += 1
                    else:
                        unreachable_other += 1
        layout_details.append({
            "layout_hash": l_hash,
            "n_states": len(states),
            "n_goals": len(goals),
            "d_max": d_max_lay,
            "k_candidates": len(layout.views)
            + sum(1 for w in layout.widgets.values() if w["type"] == "button")
            + sum(1 for w in layout.widgets.values() if w["type"] == "field")
            + sum(1 for w in layout.widgets.values() if w["type"] == "option") + 1,
        })

    # couverture empirique aux budgets (même distribution que le générateur)
    coverage: dict[int, int] = {}
    covered: set[tuple] = set()
    draws = 0
    subset_hashes: dict[int, str] = {}
    for k in budgets:
        while draws < k:
            layout = layouts[rng.randrange(len(layouts))]
            state, goal = sample_task(rng, layout)
            covered.add((layout.layout_hash(), state.key(), _goal_key(goal)))
            draws += 1
        coverage[k] = len(covered)
        # hash du sous-ensemble couvert à ce budget — SUR LA FORME SÉRIALISÉE
        # CANONIQUE (mêmes lignes que le fichier dumpé) : lien mécanique
        # fichier↔scellé, imbrication vérifiable (k croissant ⊂ sur-ensemble).
        def _ser(x):
            if isinstance(x, frozenset):
                return sorted(x)
            if isinstance(x, (list, tuple)):
                return [_ser(v) for v in x]
            return x

        import json as _json
        lines_k = sorted(
            _json.dumps({
                "layout_hash": layout_hash,
                "state_key": _ser(state_key),
                "goal": {"predicate": goal_key[0], "args": dict(goal_key[1])},
            }, sort_keys=True)
            for layout_hash, state_key, goal_key in covered
        )
        subset_hashes[k] = sha256_hex(lines_k)
        if covered_sets_dir:
            from pathlib import Path as _P
            d = _P(covered_sets_dir)
            d.mkdir(parents=True, exist_ok=True)
            (d / f"siw-couples-k{k}.jsonl").write_text(
                "\n".join(lines_k) + ("\n" if lines_k else ""), encoding="utf-8")
    pred_empirical: dict[str, int] = {}
    pred_in_band: dict[str, int] = {}
    rng2 = random.Random(seed + 1)
    for _ in range(4000):
        layout = layouts[rng2.randrange(len(layouts))]
        state, goal = sample_task(rng2, layout)
        pred_empirical[goal["predicate"]] = pred_empirical.get(goal["predicate"], 0) + 1
        # asymétrie POST-filtrage bande (là où elle se manifeste — tagi-1 18:24)
        oracle = SIWOracle(layout, goal)
        if oracle.reachable(state) and d_band_proposed[0] <= oracle.d_star(state) <= d_band_proposed[1]:
            pred_in_band[goal["predicate"]] = pred_in_band.get(goal["predicate"], 0) + 1

    n_couples = len(couples)
    n_emp = sum(pred_empirical.values())
    n_band = sum(pred_in_band.values())
    pred_pct = {p: round(100 * c / max(1, n_emp), 2) for p, c in sorted(pred_empirical.items())}
    pred_band_pct = {p: round(100 * c / max(1, n_band), 2) for p, c in sorted(pred_in_band.items())}
    pred_ok = {p: (PRED_TOLERANCE[0] <= v <= PRED_TOLERANCE[1]) for p, v in pred_pct.items()}
    pred_band_ok = {p: (PRED_TOLERANCE[0] <= v <= PRED_TOLERANCE[1]) for p, v in pred_band_pct.items()}

    in_band = sum(v for k, v in d_hist.items() if k != "unreachable" and d_band_proposed[0] <= int(k) <= d_band_proposed[1])
    reachable_couples = n_couples - unreachable_couples

    inv = {
        "schema": SIW_INVENTORY_SCHEMA,
        "seed": seed,
        "generation_params": generation_params or {},  # re-dérivation mécanique
        "layouts": len(layouts),
        "iso_groups": len(groups),
        "states_total": states_total,
        "couples_unique": n_couples,
        "couples_reachable": reachable_couples,
        "couples_unreachable": unreachable_couples,
        "unreachable_breakdown": {
            "select_irreversible": unreachable_select_irrev,
            "other": unreachable_other,
            "note": "propriété SELECT-irréversible (tagi-5, registre #16): "
                    "choisir une option rend CHOOSE(autre) unreachable — "
                    "difficulté structurelle SIW, publiée (§4.5)",
        },
        "d_star_hist": {k: d_hist[k] for k in sorted(d_hist, key=lambda x: int(x) if x != "unreachable" else 10**9)},
        "d_min_global": d_min_global if d_min_global < 10**9 else None,
        "d_max_global": d_max_global,
        "band_verdict": {
            "proposed": list(d_band_proposed),
            "in_band_couples": in_band,
            "in_band_pct": round(100 * in_band / max(1, reachable_couples), 2),
            "verdict": (
                f"Bande {d_band_proposed} couvre {round(100 * in_band / max(1, reachable_couples), 1)}% des couples "
                f"atteignables (max observé {d_max_global}) — resserrage AVANT gel conforme §4.5 "
                f"(publié, pas inventé)" if d_max_global <= d_band_proposed[1] + 2
                else f"max {d_max_global} dépasse la bande proposée — revoir avant gel"
            ),
        },
        "predicates": {
            "empirical_pct_raw": pred_pct,
            "tolerance_18_32_raw": pred_ok,
            "empirical_pct_in_band": pred_band_pct,
            "tolerance_18_32_in_band": pred_band_ok,
            "asymmetry_note": "VIEW sous-représenté APRÈS filtrage de bande (buts intrinsèquement plus courts — tagi-1 18:24), documenté, non corrigé",
        },
        "coverage_budgets": {str(k): coverage[k] for k in budgets},
        "coverage_subset_hashes": {str(k): subset_hashes[k] for k in budgets},
        "coverage_note": "couverture = couples uniques couverts par k tirages sample_task (distribution générateur)",
        "layout_details": layout_details,
    }
    sealed = seal_manifest(inv)
    sealed["timing"] = {"seconds_total": round(time.time() - t0, 2)}
    return sealed


def write_siw_inventory(inventory: dict, path: str | Path) -> str:
    import json

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(inventory, sort_keys=True, indent=2), encoding="utf-8")
    return str(p)


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    import json

    from ..env.siw import generate_siw_layout

    ap = argparse.ArgumentParser(description="Inventaire SIW de faisabilité (AVANT gel)")
    ap.add_argument("--layouts", type=int, default=20)
    ap.add_argument("--views-min", type=int, default=2)
    ap.add_argument("--views-max", type=int, default=5)
    ap.add_argument("--goals-per-layout", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260924)
    ap.add_argument("--profile", default="small", choices=["small", "standard"])
    ap.add_argument("--out", default="artifacts/inventory-siw-dev.json")
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)
    layouts = [
        generate_siw_layout(rng, rng.randint(args.views_min, args.views_max),
                            with_dialog=rng.random() < 0.5, profile=args.profile)
        for _ in range(args.layouts)
    ]
    gen_params = {"layouts": args.layouts, "views_range": [args.views_min, args.views_max],
                  "profile": args.profile, "goals_per_layout": args.goals_per_layout,
                  "with_dialog": "p=0.5", "generator": "generate_siw_layout"}
    inv = build_siw_inventory(layouts, seed=args.seed, goals_per_layout=args.goals_per_layout,
                              generation_params=gen_params)
    # layouts publiés (auditabilité/re-dérivation mécanique — tagi-5)
    import json as _json
    lay_out = Path(args.out).with_name(Path(args.out).stem + "-layouts.json")
    lay_out.write_text(_json.dumps(
        {l.layout_hash(): {"views": l.views, "widgets": l.widgets,
                           "nav_edges": [list(e) for e in l.nav_edges], "labels": l.labels}
         for l in layouts}, sort_keys=True), encoding="utf-8")
    out = write_siw_inventory(inv, args.out)
    print(f"inventaire SIW → {out} (scellé {inv['manifest_sha256'][:16]}…)")
    print(json.dumps({k: inv[k] for k in ("layouts", "iso_groups", "couples_unique",
                                          "d_max_global", "band_verdict", "predicates",
                                          "coverage_budgets")}, indent=2, ensure_ascii=False)[:1500])
    return 0 if verify_manifest(inv) else 1


if __name__ == "__main__":
    raise SystemExit(main())


# ---------------------------------------------------------------------------
# Manifest de splits SIW (additif, scellé) — décisions tagi-1 18:31
# ---------------------------------------------------------------------------

SIW_SPLIT_SCHEMA = "ucm-siw-split-manifest/0.1"


def make_siw_split_manifest(
    layouts: list[SIWLayout],
    *,
    seed: int,
    train_n: int = 200,
    val_n: int = 50,
    min_test: int = 100,
) -> dict:
    """Pools SIW iso-disjoints (groupes atomiques), scellés, additifs.

    Trois entrées de décision DISTINCTES (tagi-1 18:31) :
    1. bande d* cible = 2-8 (§4.5 : max observé 8, publié) ;
    2. asymétrie NATURELLE conservée au train/val (VIEW 7.8%, SUBMITTED 39.1%
       in-band — structurelle, analogue AT-G4 V0, sur-correction refusée) ;
    3. cellules test STRATIFIÉES par prédicat dans la bande (~25%×4 parmi les
       couples in-band) — lisibilité par prédicat sans IC anémiques, sans
       toucher au train.
    """
    seen: set[str] = set()
    uniq = []
    for lay in layouts:
        h = lay.layout_hash()
        if h not in seen:
            seen.add(h)
            uniq.append(lay)
    groups = group_siw_layouts(uniq)
    rng = random.Random(seed)
    rng.shuffle(groups)
    train, val, test = [], [], []
    for g in groups:
        if len(train) < train_n:
            train.extend(g)
        elif len(val) < val_n:
            val.extend(g)
        else:
            test.extend(g)
    if len(train) < train_n:
        raise ValueError(f"SIW train insuffisant: {len(train)} < {train_n}")
    if len(val) < val_n:
        raise ValueError(f"SIW val insuffisant: {len(val)} < {val_n}")
    if len(test) < min_test:
        raise ValueError(f"SIW test insuffisant: {len(test)} < {min_test}")

    manifest = {
        "schema": SIW_SPLIT_SCHEMA,
        "seed": seed,
        "sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "iso_groups": len(groups),
        "pools": {
            "train": sorted(l.layout_hash() for l in train),
            "val": sorted(l.layout_hash() for l in val),
            "test": sorted(l.layout_hash() for l in test),
        },
        "iso_group_hashes": sorted(sorted(l.layout_hash() for l in g) for g in groups),
        "decisions": {
            "band_d_star": {"target": [2, 8], "source": "inventory-siw-dev (max observé 8, §4.5 publié)"},
            "train_asymmetry": {
                "policy": "naturelle conservée (sur-correction refusée)",
                "in_band_pct_measured": {"VIEW": 7.83, "SET": 27.18, "CHOOSE": 25.94, "SUBMITTED": 39.05},
                "justification": "asymétrie structurelle SIW (VIEW courts, SUBMITTED = chaîne complète) — analogue dominance AT-G4 V0",
            },
            "test_stratification": {
                "policy": "échantillonnage équilibré ~25%×4 par prédicat parmi les couples in-band",
                "rationale": "IC par prédicat lisibles sans distorsion du train (tagi-1 18:31)",
                "applies_to": "cellules test uniquement",
            },
            "zero_quota": "pool-level, train uniquement (leçons V0)",
        },
        "zero_iso_check": "groupes isomorphes atomiques (wl_hash_typed + typed_isomorphic exact, labels exclus)",
    }
    return seal_manifest(manifest)


def verify_siw_split_manifest(manifest: dict, layouts: list[SIWLayout]) -> dict:
    """Anti-contamination SIW : re-groupe et vérifie disjonction+tailles+scellage."""
    groups = group_siw_layouts(layouts)
    pool_of: dict[str, str] = {}
    for pool in ("train", "val", "test"):
        for h in manifest["pools"][pool]:
            pool_of[h] = pool
    errors: list[str] = []
    if not verify_manifest(manifest):
        errors.append("scellage invalide")
    by_hash = {l.layout_hash(): l for l in layouts}
    for g in groups:
        pools_in_group = {pool_of.get(l.layout_hash()) for l in g} - {None}
        if len(pools_in_group) > 1:
            errors.append(f"CONTAMINATION: groupe isomorphe éclaté {pools_in_group} ({[l.layout_hash() for l in g]})")
    for h in pool_of:
        if h not in by_hash:
            errors.append(f"layout_hash du manifest absent: {h}")
    for pool in ("train", "val", "test"):
        if len(manifest["pools"][pool]) < manifest["sizes"][pool]:
            errors.append(f"{pool}: taille manifest {manifest['sizes'][pool]} > layouts fournis")
    return {"valid": not errors, "errors": errors, "pools": {p: len(v) for p, v in manifest["pools"].items()}}


# ---------------------------------------------------------------------------
# M-V1b : épisodes TEST stratifiés ~25%×4 in-band (tagi-1 18:31/19:33)
# ---------------------------------------------------------------------------


def build_siw_test_episodes(
    test_layouts: list[SIWLayout],
    *,
    seed: int,
    n_per_predicate: int = 150,
    band: tuple[int, int] = (2, 8),
    max_attempts_per_ep: int = 400,
    safety_cap_total: int = 200_000,
) -> dict:
    """Épisodes d'évaluation SIW : (init, but) initiaux stratifiés par
    prédicat dans la bande, supervision oracle embarquée (d*, reachable).
    Une lecture : une ligne = un épisode.

    LIKE-FOR-LIKE STRICT (arbitrage lead 21:40 — la branche v2 avec rejet de
    doublons a été RETIRÉE: jamais retenue pour test2, son rejet était de plus
    bogué tautologiquement). Comportement = historique exact (bit-compatible
    cellule initiale scellée), max_attempts_per_ep IGNORÉ (paramètre mort
    historique, documenté), doublons NON rejetés (COMPTÉS en descriptif v4.1).
    Cap de SÉCURITÉ global 200k : non-modifiant, protège de la boucle infinie.
    """
    rng = random.Random(seed)
    from ..env.siw_oracle import SIWOracle
    preds = ("VIEW", "SET", "CHOOSE", "SUBMITTED")
    episodes: list[dict] = []
    rejected_band = rejected_unreachable = 0
    predicate_skips = 0          # v4.1 (tagi-5): compteurs passifs — observation
    duplicate_couples_descriptive = 0  # doublons COMPTÉS, jamais filtrés
    total_attempts = 0  # cap de SÉCURITÉ global (lead 21:42): ne modifie AUCUN
                        # tirage réussi — protège seulement de la boucle infinie
    seen_couples: set = set()
    for pred in preds:
        produced = 0
        while produced < n_per_predicate:
            total_attempts += 1
            if total_attempts > safety_cap_total:
                raise RuntimeError(
                    f"sampler safety-cap {safety_cap_total} tentatives totales atteint "
                    f"({pred} {produced}/{n_per_predicate}) — pool insuffisant, FAIL explicite"
                )
            layout = test_layouts[rng.randrange(len(test_layouts))]
            state, goal = sample_task(rng, layout)
            if goal["predicate"] != pred:
                predicate_skips += 1
                continue  # stratification : tirage jusqu'au prédicat cible
            oracle = SIWOracle(layout, goal)
            if not oracle.reachable(state):
                rejected_unreachable += 1
                continue
            d0 = oracle.d_star(state)
            if not (band[0] <= d0 <= band[1]):
                rejected_band += 1
                continue
            ck = (layout.layout_hash(), state.key(),
                  (goal["predicate"], tuple(sorted(goal["args"].items()))))
            if ck in seen_couples:
                duplicate_couples_descriptive += 1  # v4.1: descriptif, PAS filtré
            else:
                seen_couples.add(ck)
            produced += 1
            episodes.append({
                "episode_ref": f"siwtest-{seed}-{len(episodes):05d}",
                "layout_hash": layout.layout_hash(),
                "init": {"view": state.view, "filled": sorted(state.filled),
                         "chosen": dict(state.chosen),
                         "dialog_open": state.dialog_open,
                         "submitted": sorted(state.submitted)},
                "goal": goal,
                "d_star": d0,
                "reachable": True,
            })
    counts = {p: sum(1 for e in episodes if e["goal"]["predicate"] == p) for p in preds}
    d_hist: dict[str, int] = {}
    for e in episodes:
        d_hist[str(e["d_star"])] = d_hist.get(str(e["d_star"]), 0) + 1
    manifest = {
        "schema": "ucm-siw-test-episodes/0.1",
        "seed": seed,
        "band": list(band),
        "sampler": "legacy like-for-like (v2 retiré — arbitrage 21:40)",
        "n_per_predicate_target": n_per_predicate,
        "counts": counts,
        "pct": {p: round(100 * c / len(episodes), 2) for p, c in counts.items()},
        "d_star_hist": {k: d_hist[k] for k in sorted(d_hist, key=int)},
        "rejections": {"band": rejected_band, "unreachable": rejected_unreachable},
        "passive_counters_v41": {"attempts_total": total_attempts,
                                 "predicate_skips": predicate_skips,
                                 "duplicate_couples_descriptive": duplicate_couples_descriptive,
                                 "max_attempts_per_ep_declared_but_unused": 400},
        "stratification": "échantillonnage équilibré par prédicat in-band (tagi-1 18:31)",
        "test_pool_only": True,
    }
    return seal_manifest(manifest), episodes


def write_siw_test_episodes(episodes: list[dict], manifest: dict, path_prefix: str) -> tuple[str, str]:
    import json

    from pathlib import Path as _P
    p = _P(path_prefix + ".jsonl")
    m = _P(path_prefix + "-manifest.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for e in episodes:
            fh.write(json.dumps(e, sort_keys=True) + "\n")
    m.write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")
    return str(p), str(m)


# ---------------------------------------------------------------------------
# PASS C tagi-5 : matérialisation au format RUNNER + scellage octets
# ---------------------------------------------------------------------------


def _rebuild_siw_layout(spec: dict) -> SIWLayout:
    widgets = spec["widgets"]
    if isinstance(widgets, dict):
        widgets = list(widgets.values())
    return SIWLayout({"views": spec["views"], "widgets": widgets,
                      "nav_edges": [tuple(e) for e in spec["nav_edges"]]})


def _state_from_key(state_key) -> "object":
    from ..env.siw import SIWState
    view, filled, chosen, dialog, submitted = state_key
    return SIWState(view, frozenset(filled), dict(chosen), dialog, frozenset(submitted))


def materialize_couples_runner(*args, **kwargs):
    """DÉPRÉCIÉ (tagi-5 19:51, traçabilité) : chemin unique de vérité =
    ``ucm.v1.data_adapter.couples_to_records`` (indices oracle directs,
    binding vérifié, registre de mapping). Ce shim délègue — les fichiers
    publiés artifacts/siw-couples-runner-k*.jsonl sont produits par l'adaptateur."""
    from ..v1.data_adapter import couples_to_records
    return couples_to_records(*args, **kwargs)


def materialize_test_episodes_runner(episodes_file, layouts_file, *, out_path=None):
    """DÉPRÉCIÉ : délègue à ``ucm.v1.data_adapter`` (chemin unique de vérité).

    Compatibilité d'ancienne signature : écrit ``out_path`` si fourni et
    retourne {path, episodes, sha256} ; sinon retourne la liste de l'adaptateur."""
    import json as _json

    from ..v1.data_adapter import episodes_to_runner
    records = episodes_to_runner(episodes_file, layouts_file)
    if out_path is None:
        return records
    from pathlib import Path as _P
    body = "".join(_json.dumps(r, sort_keys=True) + "\n" for r in records)
    p = _P(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")
    return {"path": str(p), "episodes": len(records), "sha256": sha256_hex(body)}


# ---------------------------------------------------------------------------
# Prototype: batchs de couples à bornes scellées (adaptation parallèle future)
# ---------------------------------------------------------------------------


def split_couples_into_batches(
    couples_file: str,
    *,
    n_batches: int,
    out_prefix: str,
) -> dict:
    """Découpe un fichier couples scellé en N batchs ÉQUITABLES avec bornes
    scellées: chaque batch = tranche contiguë de l'ordre canonique du fichier;
    l'index scellé prouve (a) union == ensemble complet, (b) ordre préservé,
    (c) chaque batch re-hashe à son sha256 publié. Workers parallèles sans
    partage d'état: worker j consomme batch j, couverture agrégée vérifiable.
    """
    import json as _json

    from pathlib import Path as _P

    lines = [l for l in _P(couples_file).read_text(encoding="utf-8").splitlines() if l.strip()]
    if n_batches < 1 or n_batches > len(lines):
        raise ValueError(f"n_batches={n_batches} invalide pour {len(lines)} couples")
    size = -(-len(lines) // n_batches)  # plafond
    batches = [lines[i:i + size] for i in range(0, len(lines), size)]
    out = _P(out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    index = {"schema": "ucm-siw-couples-batches/0.1", "source": str(_P(couples_file)),
             "n_batches": len(batches), "total": len(lines), "batches": []}
    for j, batch in enumerate(batches):
        p = out.parent / f"{out.name}-b{j:02d}.jsonl"
        body = "".join(l + "\n" for l in batch)
        p.write_text(body, encoding="utf-8")
        index["batches"].append({
            "batch": j, "path": p.name, "records": len(batch),
            "first": batch[0][:120], "last": batch[-1][:120],  # bornes lisibles
            "sha256": sha256_hex(body),
        })
    index["union_sha256"] = sha256_hex([b["sha256"] for b in index["batches"]])
    return seal_manifest(index)


def verify_couples_batches(index: dict, batch_dir: str) -> dict:
    """Re-hash chaque batch == sha publié + union + ordre (bornes contiguës)."""
    import hashlib as _h

    from pathlib import Path as _P

    errors: list[str] = []
    prev_last = None
    d = _P(batch_dir)
    for b in index["batches"]:
        f = d / b["path"]
        if not f.exists():
            errors.append(f"batch manquant: {b['path']}")
            continue
        actual = _h.sha256(f.read_bytes()).hexdigest()
        if actual != b["sha256"]:
            errors.append(f"{b['path']}: sha256 ≠ publié")
        lines = [l for l in f.read_text().splitlines() if l.strip()]
        if len(lines) != b["records"]:
            errors.append(f"{b['path']}: {len(lines)} lignes ≠ {b['records']}")
        if prev_last is not None and lines and lines[0] != prev_last:
            if lines[0][:120] != b["first"][:120]:
                errors.append(f"{b['path']}: borne first ≠ index")
        if lines:
            prev_last = lines[-1]
        if lines and lines[-1][:120] != b["last"][:120]:
            errors.append(f"{b['path']}: borne last ≠ index")
    if not verify_manifest(index):
        errors.append("index non scellé/vérifiable")
    return {"valid": not errors, "errors": errors}


# ---------------------------------------------------------------------------
# test2 (v4, arbitrage lead 21:40): pool générateur déterministe accept-ou-FAIL
# ---------------------------------------------------------------------------


def generate_disjoint_pool_v4(
    *,
    seed: int,
    n_candidates: int,
    n_accept: int,
    exclusion_specs: dict[str, dict],
    views_range: tuple[int, int] = (2, 5),
    profile: str = "small",
) -> tuple[list[SIWLayout], dict]:
    """Algorithme pool test2 ARBITRÉ (lead 21:40) — accept-ou-FAIL strict:

    1. générer EXACTEMENT ``n_candidates`` layouts (seed unique, ordre exact:
       randint(views) PUIS with_dialog=rng.random()<0.5, profile) ;
    2. trier par layout_hash (ordre canonique) ;
    3. accepter les ``n_accept`` premiers SANS collision — hash OU wl typé
       kind-aware OU isomorphisme exact — ni avec le corpus d'exclusion
       (``exclusion_specs``: layout store publiés) NI entre nouveaux ;
    4. TOUTE collision résiduelle à l'épuisement des candidats = FAIL
       (RuntimeError) — PAS de blocs supplémentaires, PAS de fallback.

    Retourne (layouts_acceptés, rapport) — rapport sans timing, scellable.
    """
    import random as _r

    from ..env.siw import generate_siw_layout as _gen_layout
    from .splits import typed_isomorphic, wl_hash_typed

    rng = _r.Random(seed)
    cands = []
    for _ in range(n_candidates):
        cands.append(_gen_layout(
            rng, rng.randint(views_range[0], views_range[1]),
            with_dialog=rng.random() < 0.5, profile=profile))
    # exclusion: PRÉCALCUL unique (audit 21:48-5) — certificats + wl + graphes
    excl_certs: set[str] = set()
    excl_wl: set[str] = set()
    excl_graphs: list[tuple] = []
    for name, spec in exclusion_specs.items():
        for h, raw in spec["store"].items():
            lay = _rebuild_siw_layout(raw)
            excl_certs.add(_layout_iso_hash(lay))
            g = _typed_graph_from_layout(lay)
            excl_wl.add(wl_hash_typed(*g))
            excl_graphs.append(g)
    seen_hashes: set[str] = set()
    seen_certs: set[str] = set()
    seen_graphs: list[tuple] = []
    accepted = []
    collisions = {"hash_or_wl_existing": 0, "iso_existing": 0, "iso_new": 0, "dup_hash": 0}
    for lay in sorted(cands, key=lambda l: l.layout_hash()):
        h = lay.layout_hash()
        if h in seen_hashes:
            collisions["dup_hash"] += 1
            continue
        g = _typed_graph_from_layout(lay)
        c = wl_hash_typed(*g)          # précalcul comparé aux tables existantes
        cert = _layout_iso_hash(lay)
        if cert in excl_certs or c in excl_wl:
            collisions["hash_or_wl_existing"] += 1
            continue
        if any(typed_isomorphic(g, eg) for eg in excl_graphs):
            collisions["iso_existing"] += 1
            continue
        if cert in seen_certs:
            collisions["iso_new"] += 1
            continue
        if any(typed_isomorphic(g, ng) for ng in seen_graphs):
            collisions["iso_new"] += 1
            continue
        if len(accepted) == n_accept:
            break
        accepted.append(lay)
        seen_hashes.add(h)
        seen_certs.add(cert)
        seen_graphs.append(g)
    if len(accepted) < n_accept:
        raise RuntimeError(
            f"pool v4 FAIL: {len(accepted)}/{n_accept} acceptés sur {n_candidates} candidats "
            f"(collisions: {collisions}) — arbitrage: PAS de blocs supplémentaires, PAS de fallback"
        )
    report = {
        "algorithm": "v4: N candidats seed unique → tri layout_hash → accept-ou-FAIL",
        "seed": seed, "n_candidates": n_candidates, "n_accept": n_accept,
        "accepted": len(accepted), "collisions": collisions,
        "exclusion_sources": sorted(exclusion_specs),
    }
    return accepted, report


# ---------------------------------------------------------------------------
# test2 (lead 21:42): épisodes AUTO-SUFFISANTS — layout_spec inline, hash
# incrémental PENDANT l'écriture, zéro réouverture avant run.
# ---------------------------------------------------------------------------


def write_selfcontained_episodes(
    episodes: list[dict],
    layouts_by_hash: dict[str, SIWLayout],
    *,
    out_path: str,
) -> dict:
    """Écrit une ligne/épisode AU FORMAT RUNNER AUTO-SUFFISANT:

        {episode_id, layout_spec(complet), task{init,goal}, d_star, layout_hash}

    Le sha256 est calculé INCRÉMENTALEMENT dans le flux d'écriture (aucune
    réouverture du fichier) — le manifeste retourné publie ce hash. Le runner
    vérifie au premier et unique open."""
    import hashlib as _h

    import json as _json

    from pathlib import Path as _P

    hasher = _h.sha256()
    n = 0
    total_bytes = 0
    out = _P(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as fh:
        for e in episodes:
            lay = layouts_by_hash[e["layout_hash"]]
            data = (_json.dumps({
                "episode_id": e["episode_ref"],   # CONTRAT v4.1 (tagi-3): episode_id + task
                "layout_spec": {"views": lay.views, "widgets": lay.widgets,
                                "nav_edges": [list(x) for x in lay.nav_edges],
                                "labels": lay.labels},   # v4.1: fidélité observation
                "task": {"init": e["init"], "goal": e["goal"]},
                "d_star": e["d_star"],
                "layout_hash": e["layout_hash"],
            }, sort_keys=True) + "\n").encode("utf-8")
            fh.write(data)          # BINAIRE: mêmes octets hashés/écrits (leçon #21)
            hasher.update(data)
            total_bytes += len(data)
            n += 1
    return {"path": str(out), "episodes": n, "sha256_write_stream": hasher.hexdigest(),
            "bytes": total_bytes, "bytes_per_episode": total_bytes // max(1, n)}


# ---------------------------------------------------------------------------
# V1-bis (arbitrage lead 10:09, Option B) — support d'adaptation NON dégénéré.
# DEV UNIQUEMENT: aucun seed scellé (20261003/04 intacts), aucun scellé touché.
# ---------------------------------------------------------------------------


def sample_task_degeneracy(layouts: list[SIWLayout], *, seed: int, n_draws: int = 60) -> dict:
    """Quantifie la dégénérescence des états initiaux de sample_task:
    fraction d'états vides (aucun filled/chosen/dialog/submitted) vs
    états porteurs de features."""
    import random as _r

    rng = _r.Random(seed)
    empty = featureful = 0
    per_feature = {"filled>0": 0, "chosen>0": 0, "dialog": 0, "submitted>0": 0}
    for lay in layouts:
        for _ in range(n_draws):
            st, _goal = sample_task(rng, lay)
            if len(st.filled) == 0 and len(st.chosen) == 0 and not st.dialog_open and len(st.submitted) == 0:
                empty += 1
            else:
                featureful += 1
            per_feature["filled>0"] += len(st.filled) > 0
            per_feature["chosen>0"] += len(st.chosen) > 0
            per_feature["dialog"] += st.dialog_open
            per_feature["submitted>0"] += len(st.submitted) > 0
    n = empty + featureful
    return {"draws": n, "empty_states": empty, "empty_pct": round(100 * empty / max(1, n), 1),
            "featureful_states": featureful, "per_feature": per_feature}


def support_inventory_by_depth_feature(
    layouts: list[SIWLayout],
    *,
    seed: int,
    per_layout_goals: int = 4,
    states_sample: int = 40,
) -> dict:
    """Inventaire de support des étas ATTEIGNABLES par prédicat × profondeur ×
    features — la base de l'arbitrage V1-bis (support riche vs dégénéré)."""
    import random as _r

    from ..env.siw_oracle import enumerate_states

    rng = _r.Random(seed)
    out: dict[str, dict] = {}
    for pred_target in ("VIEW", "SET", "CHOOSE", "SUBMITTED"):
        d_hist: dict = {}
        f_hist: dict = {}
        n = 0
        for lay in layouts:
            fields = [w["id"] for w in lay.widgets.values() if w["type"] == "field"]
            selects = [(w["id"], [o["id"] for o in lay.widgets.values() if o.get("select") == w["id"]])
                       for w in lay.widgets.values() if w["type"] == "select"]
            forms = [w["id"] for w in lay.widgets.values() if w["type"] == "form"]
            for _ in range(per_layout_goals):
                goals = []
                if fields: goals.append({"predicate": "SET", "args": {"field": rng.choice(fields)}})
                if selects:
                    s, opts = rng.choice(selects)
                    goals.append({"predicate": "CHOOSE", "args": {"select": s, "option": rng.choice(opts)}})
                if forms: goals.append({"predicate": "SUBMITTED", "args": {"form": rng.choice(forms)}})
                goals.append({"predicate": "VIEW", "args": {"view": rng.choice(lay.views)}})
                goal = next(g for g in goals if g["predicate"] == pred_target)
                o = SIWOracle(lay, goal)
                states = [s for s in enumerate_states(lay) if o.reachable(s)]
                rng.shuffle(states)
                for s in states[:states_sample]:
                    d = o.d_star(s)
                    d_hist[str(min(d, 8))] = d_hist.get(str(min(d, 8)), 0) + 1
                    nf = len(s.filled) + len(s.chosen) + (1 if s.dialog_open else 0) + len(s.submitted)
                    f_hist[str(min(nf, 4))] = f_hist.get(str(min(nf, 4)), 0) + 1
                    n += 1
        out[pred_target] = {"n_states": n,
                            "d_star_hist": dict(sorted(d_hist.items(), key=lambda kv: int(kv[0]))),
                            "features_hist": dict(sorted(f_hist.items(), key=lambda kv: int(kv[0])))}
    return out


def generate_multistep_adaptation_couples(
    layouts: list[SIWLayout],
    *,
    seed: int,
    couples_per_layout: int = 16,
    band: tuple[int, int] = (0, 8),
    extras_recovery_frac: float = 0.25,
    max_trajectory_states: int = 12,
) -> tuple[list[dict], dict]:
    """Couples d'adaptation V1-bis — v4 (contre-examen tagi-ask 10:33).

    RECENTRAGE: le support = départs sample_task (états initiaux VIDES — la
    distribution RÉELLE du test) + trajectoires oracle COMPLÈTES depuis ces
    départs, CHAQUE état intermédiaire inclus, TERMINAL d*=0 ({STOP}) inclus.
    AUCUN filtre de richesse (B3: le filtre évictionnait la masse test-like).
    Extras post-action: UNIQUEMENT en recovery ≤ extras_recovery_frac, sans
    filtre richesse, dans la closure forward. Dédup + round-robin layouts.
    Exception déclarée: profil small (1 form) → un seul but SUBMITTED distinct
    par layout (variation de contexte par vues au départ).
    """
    import random as _r

    from ..env.siw import SIWState, goal_satisfied, sample_task
    from ..env.siw_oracle import SIWOracle, _successor

    rng = _r.Random(seed)
    preds = ("VIEW", "SET", "CHOOSE", "SUBMITTED")
    couples: list[dict] = []
    seen: set = set()
    stats = {"layouts": len(layouts), "per_predicate": {p: 0 for p in preds},
             "d_hist": {}, "feature_type_hist": {"initial_empty": 0, "filled": 0,
                                                 "chosen": 0, "dialog": 0, "submitted": 0},
             "d0_stop_couples": 0, "d_pos_couples": 0, "shortfalls": [],
             "sources": {"departure": 0, "trajectory": 0, "terminal_stop": 0, "extra_recovery": 0},
             "dedup_skipped": 0, "submitted_goal_exception_layouts": 0,
             "closure_sizes": {}}

    def _emit(lay, goal, st, o, source):
        key = (lay.layout_hash(), st.key(),
               (goal["predicate"], tuple(sorted(goal["args"].items()))))
        if key in seen:
            stats["dedup_skipped"] += 1
            return
        seen.add(key)
        opt = [o.candidates[i]["action"] + ":" + str(o.candidates[i]["arg"])
               for i in o.optimal_actions(st)]
        d = o.d_star(st)
        has_stop = any(x.startswith("STOP:") for x in opt)
        if d == 0:
            assert has_stop and len(opt) == 1, "d*=0 sans {STOP} exact !"
            stats["d0_stop_couples"] += 1
            if source == "trajectory":
                stats["sources"]["terminal_stop"] += 1
        else:
            assert not has_stop, f"STOP à d*={d} !"
            stats["d_pos_couples"] += 1
        couples.append({"schema": "ucm-siw-adaptation-couples/0.4",
                        "layout_hash": lay.layout_hash(),
                        "state_key": [st.view, sorted(st.filled), sorted(st.chosen.items()),
                                      st.dialog_open, sorted(st.submitted)],
                        "goal": goal, "d_star": d, "optimal_semantic": opt,
                        "source": source})
        stats["sources"][source] = stats["sources"].get(source, 0) + 1
        stats["per_predicate"][goal["predicate"]] += 1
        stats["d_hist"][str(d)] = stats["d_hist"].get(str(d), 0) + 1
        if len(st.filled) == 0 and len(st.chosen) == 0 and not st.dialog_open and len(st.submitted) == 0:
            stats["feature_type_hist"]["initial_empty"] += 1
        else:
            stats["feature_type_hist"]["filled"] += len(st.filled) > 0
            stats["feature_type_hist"]["chosen"] += len(st.chosen) > 0
            stats["feature_type_hist"]["dialog"] += st.dialog_open
            stats["feature_type_hist"]["submitted"] += len(st.submitted) > 0

    # exception SUBMITTED: profiler le nombre de buts distincts
    for lay in layouts:
        n_forms = sum(1 for w in lay.widgets.values() if w["type"] == "form")
        if n_forms < 2:
            stats["submitted_goal_exception_layouts"] += 1

    # round-robin layouts, buts contrastés (un par prédicat disponible, rotation)
    pred_cycle = {p: 0 for p in preds}
    extras_budget = int(extras_recovery_frac * couples_per_layout * len(layouts))
    extras_used = 0
    for round_ in range(max(1, couples_per_layout // 4)):
        for lay in layouts:
            if len([c for c in couples if c["layout_hash"] == lay.layout_hash()]) >= couples_per_layout:
                continue
            # but CONTRASTÉ: prédicat le moins servi sur ce layout
            lay_preds = {c["goal"]["predicate"] for c in couples if c["layout_hash"] == lay.layout_hash()}
            avail = [p for p in preds if p not in lay_preds]
            if not avail:
                avail = list(preds)
            pred = avail[pred_cycle[avail[0]] % len(avail)]
            pred_cycle[pred] = pred_cycle.get(pred, 0) + 1
            # départ sample_task avec prédicat cible
            for _try in range(30):
                st0, goal = sample_task(rng, lay)
                if goal["predicate"] == pred:
                    break
            else:
                continue
            o = SIWOracle(lay, goal)
            if not o.reachable(st0):
                continue
            if not (band[0] <= o.d_star(st0) <= band[1]):
                continue
            _emit(lay, goal, st0, o, "departure")  # état INITIAL (masse test-like)
            # trajectoire COMPLÈTE jusqu'au terminal (STOP inclus)
            st = st0
            n_tr = 0
            while o.d_star(st) > 0 and n_tr < max_trajectory_states:
                act = o.candidates[rng.choice(o.optimal_actions(st))]
                nxt = _successor(lay, st, act)
                if nxt is None:
                    break
                st = nxt
                n_tr += 1
                _emit(lay, goal, st, o, "trajectory")
            # extra recovery (≤25% global, sans filtre richesse, dans closure)
            if extras_used < extras_budget:
                closure = forward_reachable_closure(lay)
                states_by_key = _closure_states_by_key(lay, closure)
                rich = [s for s in states_by_key.values()
                        if (len(s.filled) + len(s.chosen)) > 0 and o.reachable(s)
                        and band[0] <= o.d_star(s) <= band[1]
                        and (lay.layout_hash(), s.key(),
                             (goal["predicate"], tuple(sorted(goal["args"].items())))) not in seen]
                if rich:
                    stx = rng.choice(rich)
                    _emit(lay, goal, stx, o, "extra_recovery")
                    extras_used += 1

    assert couples, "dataset VIDE"
    assert stats["d0_stop_couples"] > 0 and stats["d_pos_couples"] > 0
    lay_by_hash = {l.layout_hash(): l for l in layouts}
    for c in couples:
        closure = forward_reachable_closure(lay_by_hash[c["layout_hash"]])
        stats["closure_sizes"][lay_by_hash[c["layout_hash"]].layout_hash()[:8]] = len(closure)
        key = (c["state_key"][0], frozenset(c["state_key"][1]), tuple(c["state_key"][2]),
               c["state_key"][3], frozenset(c["state_key"][4]))
        assert key in closure, "couple hors closure !"
    stats["d_hist"] = dict(sorted(stats["d_hist"].items(), key=lambda kv: int(kv[0])))
    stats["initial_mass_pct"] = round(100 * stats["sources"]["departure"] / max(1, len(couples)), 1)
    stats["extras_pct"] = round(100 * stats["sources"]["extra_recovery"] / max(1, len(couples)), 1)
    return couples, stats

_FORWARD_CLOSURE_CACHE: dict[str, frozenset] = {}


def forward_reachable_closure(layout: SIWLayout) -> frozenset:
    """Fermeture forward des états PHYSIQUEMENT atteignables depuis TOUS les
    états initiaux supportés par sample_task (view ∈ views, dialog admissible
    si le layout possède un dialog, filled/chosen/submitted vides), via
    _successor (self-loops/invalides ignorées). Cache par layout_hash."""
    from ..env.siw_oracle import _successor, candidate_actions
    from ..env.siw import SIWState
    from collections import deque

    h = layout.layout_hash()
    cached = _FORWARD_CLOSURE_CACHE.get(h)
    if cached is not None:
        return cached
    has_dialog = any(w["type"] == "dialog" for w in layout.widgets.values())
    initials = [SIWState(v, frozenset(), {}, dlog, frozenset())
                for v in layout.views for dlog in ((False, True) if has_dialog else (False,))]
    acts = candidate_actions(layout)
    seen: set = set()
    dq = deque()
    for s in initials:
        if s.key() not in seen:
            seen.add(s.key())
            dq.append(s)
    while dq:
        st = dq.popleft()
        for a in acts:
            nxt = _successor(layout, st, a)
            if nxt is None:
                continue
            k = nxt.key()
            if k not in seen:
                seen.add(k)
                dq.append(nxt)
    closure = frozenset(seen)
    _FORWARD_CLOSURE_CACHE[h] = closure
    return closure


def _closure_states_by_key(layout: SIWLayout, closure: frozenset) -> dict:
    """États SIWState de la closure, indexés par key()."""
    from ..env.siw import SIWState
    out = {}
    for k in closure:
        view, filled, chosen, dialog, submitted = k
        out[k] = SIWState(view, frozenset(filled), dict(chosen), dialog, frozenset(submitted))
    return out


# CONTRAT mapping optimal_semantic → optimal_actions (tagi-5 5c; v3)
# optimal_semantic[i] = "<ACTION>:<arg>". Mapping vers indices: env.candidates()
# ordre canonique; optimal_actions = indices dont (action,arg) ∈ semantic.
# (ii) équivalence: couples_to_records RECALCULE l'oracle (chemin sûr) —
# recommandation: assert de croissement optimal_semantic == oracle au
# matérialiseur (double vérification, jamais simple consommation).


def semantic_to_indices(optimal_semantic: list[str], candidates: list[dict]) -> list[int]:
    """Mapping canonique — assert bijectif (candidats uniques par (action,arg))."""
    sem_set = set(optimal_semantic)
    idx = [i for i, c in enumerate(candidates)
           if f"{c['action']}:{c['arg']}" in sem_set]
    assert len(idx) == len(sem_set), "mapping sémantique→indices non bijectif !"
    return sorted(idx)


def generate_adaptation_episodes_rstar(
    layouts: list[SIWLayout],
    *,
    seed: int,
    n_episodes: int,
    d0_band: tuple[int, int] = (2, 8),
    max_resample: int = 200,
    max_traj_steps: int = 64,
) -> tuple[list[dict], dict]:
    """R* — v6 (audit lead 10:49, 4 bloqueurs corrigés).

    (1) QUOTA EXACT par prédicat: rotation layouts ÉLIGIBLES même prédicat
        jusqu'au quota — épuisement du pool = RuntimeError FAIL (jamais drop).
    (2) RNG UNIQUE par état: rstar_choice EST l'action exécutée (une seule
        sélection/état: rng.choice(sorted(optimal_actions))); terminal d0
        n'effectue AUCUN choix ({STOP} unique).
    (3) TERMINAL GARANTI: chaque épisode finit d*=0 avec terminal STOP —
        assert ; compte STOP == episodes ; plafond atteint = RuntimeError.
    (4) SÉRIALISATION TRIÉE: optimal_semantic construit depuis indices TRIÉS.
    Recovery: RETIRÉE (hors protocole jusqu'à spécification — audit 10:49).
    Multiplicité préservée; R* canonique sorted-before-choice.
    """
    import random as _r

    from ..env.siw import sample_task
    from ..env.siw_oracle import SIWOracle, _successor

    rng = _r.Random(seed)
    preds = ("VIEW", "SET", "CHOOSE", "SUBMITTED")
    if n_episodes % len(preds) != 0:
        raise ValueError(f"n_episodes={n_episodes} doit être multiple de {len(preds)} (quota exact)")
    quota = n_episodes // len(preds)
    couples: list[dict] = []
    seen_keys: dict = {}
    stats = {"episodes_target": n_episodes, "episodes": 0,
             "per_predicate_episodes": {p: 0 for p in preds},
             "resample_attempts": 0, "impossible_skipped_layouts": 0,
             "multiplicity": {"total": 0, "distinct": 0, "duplicates": 0},
             "d_hist": {}, "d0_stop": 0, "d_pos": 0,
             "trajectory_len_hist": {}, "per_predicate_couples": {p: 0 for p in preds},
             "truncated": 0, "recovery": "RETIRÉE (hors protocole, audit 10:49)"}

    def _emit_state(lay, goal, st, o, source, eid, depth, executed_index=None):
        opt_actions = sorted(o.optimal_actions(st))  # (4) indices triés partout
        d = o.d_star(st)
        opt_sem = [f"{o.candidates[i]['action']}:{o.candidates[i]['arg']}"
                   for i in opt_actions]
        if d == 0:
            assert len(opt_sem) == 1 and opt_sem[0].startswith("STOP"), "d0 sans {STOP} exact"
            stats["d0_stop"] += 1
        else:
            assert not any(x.startswith("STOP") for x in opt_sem), f"STOP à d*={d} !"
            stats["d_pos"] += 1
        key = (lay.layout_hash(), st.key(),
               (goal["predicate"], tuple(sorted(goal["args"].items()))))
        seen_keys[key] = seen_keys.get(key, 0) + 1  # multiplicité préservée
        couples.append({"schema": "ucm-siw-adaptation-couples/0.8",
                        "episode_id": eid, "depth": depth,
                        "layout_hash": lay.layout_hash(),
                        "state_key": [st.view, sorted(st.filled), sorted(st.chosen.items()),
                                      st.dialog_open, sorted(st.submitted)],
                        "goal": goal, "d_star": d,
                        "optimal_semantic": opt_sem,
                        "rstar_executed": executed_index,  # (2) l'ACTION EXÉCUTÉE
                        # 0.8 (tagi-5 11:23): sémantique EXÉCUTÉE publiée —
                        # l'index seul n'est pas dérivable en κ_executed sans
                        # les candidats; on publie la chaîne (action:arg).
                        "rstar_executed_semantic": (
                            f"{o.candidates[executed_index]['action']}:"
                            f"{o.candidates[executed_index]['arg']}"
                        ) if executed_index is not None else None,
                        "source": source})
        stats["per_predicate_couples"][goal["predicate"]] += 1
        stats["d_hist"][str(d)] = stats["d_hist"].get(str(d), 0) + 1
        return opt_actions

    ep_i = 0
    for pred in preds:  # (1) quota EXACT par prédicat
        produced_pred = 0
        layout_cycle = 0
        while produced_pred < quota:
            exhausted_all = True
            for offset in range(len(layouts)):  # rotation layouts éligibles
                lay = layouts[(layout_cycle + offset) % len(layouts)]
                produced = False
                for _attempt in range(max_resample):
                    stats["resample_attempts"] += 1
                    st0, goal = sample_task(rng, lay)
                    if goal["predicate"] != pred:
                        continue
                    o = SIWOracle(lay, goal)
                    if not o.reachable(st0):
                        continue
                    if not (d0_band[0] <= o.d_star(st0) <= d0_band[1]):
                        continue
                    produced = True
                    break
                if produced:
                    exhausted_all = False
                    layout_cycle += offset + 1
                    ep_i += 1
                    stats["episodes"] += 1
                    stats["per_predicate_episodes"][pred] += 1
                    eid = f"rstar-{seed}-{ep_i:05d}"
                    # v7 (M1 tagi-5): UNE SEULE émission par état — le départ est
                    # émis DANS la boucle (source departure à depth 0), jamais avant
                    st = st0
                    depth = 0
                    d0_star = o.d_star(st0)
                    ep_records = 0
                    while o.d_star(st) > 0:
                        if depth >= max_traj_steps:
                            raise RuntimeError(
                                f"épisode {eid}: plafond trajectoire sans terminal — FAIL explicite")
                        opt_actions = sorted(o.optimal_actions(st))
                        choice_idx = rng.choice(opt_actions)  # RNG UNIQUE
                        act = o.candidates[choice_idx]
                        source = "departure" if depth == 0 else "trajectory"
                        _emit_state(lay, goal, st, o, source, eid, depth,
                                    executed_index=choice_idx)
                        ep_records += 1
                        nxt = _successor(lay, st, act)
                        if nxt is None:
                            raise RuntimeError(f"épisode {eid}: optimale sans successeur")
                        st = nxt
                        depth += 1
                    # terminal d0: {STOP} unique, émis UNE fois, AUCUN choix RNG
                    _emit_state(lay, goal, st, o, "terminal", eid, depth, executed_index=None)
                    ep_records += 1
                    # M1 asserts par épisode
                    assert ep_records == d0_star + 1,                         f"{eid}: {ep_records} records ≠ d*(s0)+1 = {d0_star + 1}"
                    stats["trajectory_len_hist"][str(min(depth, 12))] = (
                        stats["trajectory_len_hist"].get(str(min(depth, 12)), 0) + 1)
                    produced_pred += 1
                    if produced_pred >= quota:
                        break
                else:
                    stats["impossible_skipped_layouts"] += 1
            if exhausted_all:  # (1) épuisement → FAIL
                raise RuntimeError(
                    f"R* FAIL: prédicat {pred} — aucun layout éligible "
                    f"({produced_pred}/{quota} épisodes) — pool insuffisant, quota exact exigé")
    # (3) invariants globaux
    assert stats["episodes"] == n_episodes, f"{stats['episodes']} ≠ {n_episodes}"
    assert stats["d0_stop"] == n_episodes, f"STOP {stats['d0_stop']} ≠ épisodes {n_episodes}"
    assert all(v == quota for v in stats["per_predicate_episodes"].values())
    # M1: sources globaux + zéro doublon INTRA-épisode (inter-épisode préservé)
    n_dep = sum(1 for c in couples if c["source"] == "departure")
    n_term = sum(1 for c in couples if c["source"] == "terminal")
    assert n_dep == n_episodes and n_term == n_episodes,         f"departure {n_dep} / terminal {n_term} ≠ {n_episodes} (double émission M1)"
    intra: set = set()
    for c in couples:
        k = (c["episode_id"], c["depth"],
             tuple(map(str, c["state_key"])))
        assert k not in intra, f"doublon intra-épisode M1: {k}"
        intra.add(k)
    stats["multiplicity"] = {"total": len(couples), "distinct": len(seen_keys),
                             "duplicates": len(couples) - len(seen_keys)}
    stats["d_hist"] = dict(sorted(stats["d_hist"].items(), key=lambda kv: int(kv[0])))
    return couples, stats


# ---------------------------------------------------------------------------
# S0 baselines CPU (lead 13:43, décision e97f9a1 §S0) — DEV-only, exploratoire.
# Trois niveaux d'info clairement étiquetés, fixtures DEV R* schema 0.8.
# Aucun seuil, aucun gate.
# ---------------------------------------------------------------------------

def _baseline_random_valid(layout: SIWLayout, state, goal, rng) -> tuple:
    """Niveau 'random valide': choisit une action PHYSIQUEMENT valide au hasard.
    A accès aux vraies transitions pour filtrer (mais pas à l'oracle d*)."""
    from ..env.siw_oracle import _successor, candidate_actions
    acts = candidate_actions(layout)
    valid = [a for a in acts if _successor(layout, state, a) is not None or a["action"] == "STOP"]
    return rng.choice(valid) if valid else {"action": "STOP", "arg": None}


def _baseline_goal_heuristic(layout: SIWLayout, state, goal, rng) -> tuple:
    """Niveau 'informations réellement disponibles': heuristique_goal-blind
    utilisant UNIQUEMENT l'observation (pas d'*). Ex: si goal SET(field) et
    field pas rempli et pas porté → TYPE(field) ou MOVE vers vue du field."""
    from ..env.siw_oracle import candidate_actions
    acts = candidate_actions(layout)
    pred = goal["predicate"]
    args = goal["args"]
    # heuristique simple par prédicat — PAS d'oracle
    if pred == "VIEW":
        target = args.get("view")
        if state.view == target:
            return {"action": "STOP", "arg": None}
        # NAVIGATE vers la vue cible si candidat
        for a in acts:
            if a["action"] == "NAVIGATE" and a["arg"] == target:
                return a
    elif pred == "SET":
        fld = args.get("field")
        if fld in state.filled:
            return {"action": "STOP", "arg": None}
        w = layout.widgets.get(fld, {})
        if w.get("view") == state.view:
            return {"action": "TYPE", "arg": fld}
        for a in acts:
            if a["action"] == "NAVIGATE" and a["arg"] == w.get("view"):
                return a
    elif pred == "CHOOSE":
        sel, opt = args.get("select"), args.get("option")
        if state.chosen.get(sel) == opt:
            return {"action": "STOP", "arg": None}
        w = layout.widgets.get(sel, {})
        if w.get("view") == state.view:
            for a in acts:
                if a["action"] == "SELECT" and a["arg"] == opt:
                    return a
        for a in acts:
            if a["action"] == "NAVIGATE" and a["arg"] == w.get("view"):
                return a
    elif pred == "SUBMITTED":
        form = args.get("form")
        if form in state.submitted:
            return {"action": "STOP", "arg": None}
        # cherche le bouton submit
        for w in layout.widgets.values():
            if w.get("submit_for") == form and w.get("view") == state.view:
                return {"action": "CLICK", "arg": w["id"]}
        for a in acts:
            if a["action"] == "NAVIGATE" and a["arg"] == layout.widgets.get(form, {}).get("view"):
                return a
    # fallback: random valide
    return _baseline_random_valid(layout, state, goal, rng)


def run_s0_baseline_episode(
    layout: SIWLayout,
    init_state,
    goal: dict,
    *,
    level: str,
    seed: int,
    horizon: int = 64,
) -> dict:
    """Un épisode S0 baseline. level ∈ {'oracle', 'random_valid', 'goal_heuristic'}.
    'oracle' = borne sup privilégiée (vraies transitions, d* exact).
    'random_valid' = informations réellement disponibles (transitions connues, pas d').
    'goal_heuristic' = informations réellement disponibles (observation seulement, heuristique).
    Retourne {level, steps, success, goal_ever, stopped, timeout, actions_taken, latency_ms}."""
    import random as _r
    import time as _t
    from ..env.siw import SIW, goal_satisfied
    from ..env.siw_oracle import SIWOracle, candidate_actions

    rng = _r.Random(seed)
    t0 = _t.perf_counter()
    env = SIW(layout, horizon=horizon)
    env.goal = goal
    env.state = init_state.copy()
    env.step_count = 0
    env.terminal = False
    o = SIWOracle(layout, goal) if level == "oracle" else None

    steps = 0
    success = False
    goal_ever = False
    stopped = False
    timeout = False

    while not env.terminal and steps < horizon:
        if level == "oracle":
            if not o.reachable(env.state):
                break
            opt = o.optimal_actions(env.state)
            if not opt:
                break
            act = o.candidates[rng.choice(opt)]
        elif level == "random_valid":
            act = _baseline_random_valid(layout, env.state, goal, rng)
        elif level == "goal_heuristic":
            act = _baseline_goal_heuristic(layout, env.state, goal, rng)
        else:
            raise ValueError(f"niveau inconnu: {level}")

        if act["action"] == "STOP":
            stopped = True
            success = goal_satisfied(env.state, layout, goal)
            env.terminal = True
            break

        if goal_satisfied(env.state, layout, goal):
            goal_ever = True

        env.execute(act)
        steps += 1
        if env.terminal:
            timeout = True
            goal_ever = goal_ever or goal_satisfied(env.state, layout, goal)
            break

    if goal_satisfied(env.state, layout, goal):
        goal_ever = True

    latency_ms = (_t.perf_counter() - t0) * 1000
    return {"level": level, "steps": steps, "success": success,
            "goal_ever": goal_ever, "stopped": stopped, "timeout": timeout,
            "actions_taken": steps, "latency_ms": round(latency_ms, 3)}
