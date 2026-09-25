"""Point d'entrée génération test2 WS-B (lead 16:21) — pièce (a) du scellement.

Génère 600 épisodes (150×4 prédicats, M1) au format auto-suffiant:
{episode_id, layout_spec (views+widgets+nav_edges+labels), task{init,goal}, d_star, layout_hash}.

Pool de layouts FRAIS disjoints: générés par seed paramètre, iso-disjoints
(kind-aware exact) de TOUS corpus existants (TGK canon + SIW pools + v1bis-gen
+ DEV). Seed 20261003 par défaut (JAMAIS consommé avant token lead).

Usage:
    from ucm.data.test2_generation import generate_test2_episodes
    result = generate_test2_episodes(seed_pool=20261003, seed_episodes=20261004)
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Optional

from ..env.siw import SIW, SIWLayout, SIWState, generate_siw_layout, sample_task
from ..env.siw_oracle import SIWOracle
from .siw_pipeline import generate_disjoint_pool_v4, _layout_iso_hash, _typed_graph_from_layout
from .rstar_order import serialize_rr_records
from .writer import seal_manifest


# Corpus d'exclusion (rejeu déterministe — AUCUN manifeste scellé lu)
def _build_exclusion_store() -> dict[str, dict]:
    """Construit le store d'exclusion par REJEU seul (zéro lecture scellée).

    Sources: TGK canon n/a (monde différent); SIW pools (420 via seed 20260926 — CORRIGÉ tagi-5 16:29A);
    DEV (30 via inventory-siw-dev-layouts.json, publié non scellé);
    v1bis-gen (via seeds 20261030-39, fichiers publiés).
    """
    store: dict[str, dict] = {}
    # SIW pools 420 (rejeu déterministe, seed 20260926 — vérifié intersection 420/420)
    rng = random.Random(20260926)  # CORRECTIF audit 16:26 (20261026 était FAUX)
    for _ in range(420):
        lay = generate_siw_layout(rng, rng.randint(2, 5), with_dialog=rng.random() < 0.5,
                                  profile="small")
        store[lay.layout_hash()] = {"views": lay.views, "widgets": lay.widgets,
                                    "nav_edges": [list(e) for e in lay.nav_edges]}
    # DEV 30 (fichier publié non scellé)
    dev_path = Path(__file__).parents[2] / "artifacts" / "inventory-siw-dev-layouts.json"
    if dev_path.exists():
        store.update(json.loads(dev_path.read_text()))
    # v1bis-gen (BLOQUANT 16:36 corrigé): lecture des HASHES des fichiers publiés
    # = source FIABLE PRIMAIRE (les 26 vrais layouts d'adaptation).
    gen_hashes: set[str] = set()
    gen_dir = Path(__file__).parents[2] / "artifacts" / "v1bis-gen"
    if not gen_dir.exists():
        raise RuntimeError(f"v1bis-gen absent: {gen_dir} — fail-closed (les 26 layouts d'adaptation DOIVENT être exclus)")
    for f in sorted(gen_dir.glob("v1bis-gen-*.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                h = rec.get("layout_hash")
                if h:
                    gen_hashes.add(h)
            except json.JSONDecodeError:
                raise RuntimeError(f"ligne corrompue dans {f.name} — fail-closed")
    if not gen_hashes:
        raise RuntimeError(f"aucun layout_hash trouvé dans {gen_dir} — fail-closed")
    # hash-only: les fichiers n'ont pas les specs → hash-only pour ces 26 layouts.
    # Tentative d'iso-exact via re-déivation SUPPLÉMENTAIRE (non-bloquant):
    # hash-only documenté comme risque résiduel assumé (petit corpus, arbitrage lead 16:36).
    # (c) tagi-5 17:57: câble le store v2 (specs COMPLÈTES des 26) →
    # le résiduel hash-only tombe ENTIEREMENT (WL/iso sur les vraies specs)
    store_v2_path = Path(__file__).parents[2] / "artifacts" / "v1bis-gen-store-v02.json"
    if not store_v2_path.exists():
        raise RuntimeError(f"store v2 absent: {store_v2_path} — fail-closed (specs 26 layouts requises)")
    store_v2 = json.loads(store_v2_path.read_text())
    specs_26 = store_v2.get("store", {})
    for h in gen_hashes:
        if h not in store:
            if h in specs_26:
                store[h] = specs_26[h]  # SPECS COMPLÈTES (fin du hash-only)
            else:
                raise RuntimeError(f"layout {h} dans fichiers mais absent du store v2 — fail-closed")
    # NOTE: le replay pool200 de 854caff était le MAUVAIS PROCÉDÉ — supprimé.
    return store


def generate_test2_pool(*, seed_pool: int, n_candidates: int = 900,
                        n_accept: int = 170) -> tuple[list[SIWLayout], dict]:
    """Génère le pool test2: layouts frais iso-disjoints de TOUS corpus existants."""
    exclusion_store = _build_exclusion_store()
    # sépare: layouts complets (SIW pools + DEV, pour iso exact) vs hash-only (v1bis-gen, préfiltre)
    full_store = {h: d for h, d in exclusion_store.items() if d is not None}
    hash_only = {h for h, d in exclusion_store.items() if d is None}
    pool, report = generate_disjoint_pool_v4(
        seed=seed_pool, n_candidates=n_candidates, n_accept=n_accept,
        exclusion_specs={"full_layouts": {"store": full_store}},
    )
    # post-filtre hash: rejette tout candidat dont le hash est dans hash_only
    pool_filtered = [l for l in pool if l.layout_hash() not in hash_only]
    report["hash_only_rejections"] = len(pool) - len(pool_filtered)
    if len(pool_filtered) < n_accept:
        raise RuntimeError(
            f"pool test2 INSUFFISANT après hash-only: {len(pool_filtered)}/{n_accept} "
            f"({report['hash_only_rejections']} rejets hash) — FAIL explicite, pas de <170 silencieux"
        )
    return pool_filtered, report


def generate_test2_episodes(
    *,
    seed_pool: int = 20261003,
    seed_episodes: int = 20261004,
    n_per_predicate: int = 150,
    d0_band: tuple[int, int] = (2, 8),
    n_pool_layouts: int = 170,
    safety_cap_total: int = 200_000,
) -> dict:
    """Point d'entrée génération test2 — 600 épisodes auto-suffiants.

    Retourne un manifest scellé avec:
    - pool info (disjonction, seeds, counts)
    - episodes: liste de dicts {episode_id, layout_spec, task, d_star, layout_hash}
    - sha256 du flux JSONL sérialisé (octets exacts)
    - producteur: code_commit
    """
    import subprocess
    producer = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()

    # Pool frais disjoint
    pool, pool_report = generate_test2_pool(seed_pool=seed_pool, n_accept=n_pool_layouts)
    # ASSERTION bout-en-bout (tagi-5 16:29 A): contre la référence NON-SCELLÉE
    # pool200.dev::exclusion_set (450 = 420∪DEV30) — JAMAIS le manifeste scellé
    pool200_path = Path(__file__).parents[2] / "artifacts" / "dev-v1bis-pool200.json"
    if not pool200_path.exists():
        raise RuntimeError(f"référence non-scellée absente: {pool200_path} — fail-closed (D3)")
    dev_path_fc = Path(__file__).parents[2] / "artifacts" / "inventory-siw-dev-layouts.json"
    if not dev_path_fc.exists():
        raise RuntimeError(f"DEV layouts absents: {dev_path_fc} — fail-closed (D3)")
    ref = json.loads(pool200_path.read_text())
    ref_hashes = set(ref["exclusion_set"]["layout_hashes"])
    dev30 = set(json.loads(dev_path_fc.read_text()).keys())
    # D1: ÉGALITÉ EXACTE — replay_420 doit être EXACTEMENT (ref - dev30)
    replay_420 = set()
    rng_check = random.Random(20260926)
    for _ in range(420):
        lay = generate_siw_layout(rng_check, rng_check.randint(2, 5),
                                  with_dialog=rng_check.random() < 0.5, profile="small")
        replay_420.add(lay.layout_hash())
    expected_420 = ref_hashes - dev30
    if replay_420 != expected_420:
        extra = len(replay_420 - expected_420)
        missing = len(expected_420 - replay_420)
        raise RuntimeError(
            f"D1 ÉGALITÉ EXACTE FAIL: replay≠ref-dev30 (extra={extra}, missing={missing}) — "
            f"inclusion partielle détectée et REJETÉE"
        )

    # Sampler: like-for-like (build_siw_test_episodes legacy + safety cap)
    from .siw_pipeline import build_siw_test_episodes
    manifest_eps, episodes = build_siw_test_episodes(
        pool, seed=seed_episodes, n_per_predicate=n_per_predicate,
        band=d0_band, safety_cap_total=safety_cap_total,
    )

    # Auto-suffisant: layout_spec inline par épisode
    by_hash = {l.layout_hash(): l for l in pool}
    selfcontained = []
    for e in episodes:
        lay = by_hash[e["layout_hash"]]
        selfcontained.append({
            "episode_id": e["episode_ref"],
            "layout_spec": {"views": lay.views, "widgets": lay.widgets,
                            "nav_edges": [list(x) for x in lay.nav_edges],
                            "labels": lay.labels},
            "task": {"init": e["init"], "goal": e["goal"]},
            "d_star": e["d_star"],
            "layout_hash": e["layout_hash"],
        })

    # SHA du flux JSONL (octets exacts via sérialiseur canonique)
    stream_bytes = serialize_rr_records(selfcontained)
    sha_stream = hashlib.sha256(stream_bytes).hexdigest()

    art = seal_manifest({
        "schema": "ucm-siw-test2-generation/0.1",
        "what": "génération test2 — 600 épisodes auto-suffiants (pièce a scellement)",
        "producer": {"code_commit": producer, "module": "ucm/data/test2_generation.py"},
        "pool": {
            "seed": seed_pool, "n_candidates": pool_report.get("n_candidates", 900),
            "accepted": len(pool), "collisions": pool_report.get("collisions", {}),
            "exclusion": "rejeu déterministe — AUCUN manifeste/cellule scellé lu",
        },
        "episodes": {
            "seed": seed_episodes, "n_per_predicate": n_per_predicate,
            "total": len(selfcontained), "band": list(d0_band),
            "sampler": "legacy like-for-like + safety cap 200k",
            "sha256_stream": sha_stream,
            "bytes": len(stream_bytes),
        },
        "selfcontained_full": selfcontained,
        "hold": "GÉNÉRATION RÉELLE UNIQUEMENT SUR TOKEN LEAD — ce module ne fait RIEN à l'import",
    })
    return art


if __name__ == "__main__":
    print("Module import — aucune génération sans token lead explicite")
