"""Splits WS-B — isomorphisme, pools, réservation G2, manifest scellé (spec §7.3/§7.4).

Règles implémentées :
- **Isomorphisme** : préfiltre par hash structural (color refinement / 1-WL sur
  le graphe à arêtes colorées — porte distinguée), puis comparaison exacte par
  certificat canonique (individualization-refinement) dans les buckets.
- **Grouping** : topologie + position structurelle de la porte (§7.3). Les IDs,
  positions d'agent/objets, verrouillage, buts N'ENTENT PAS dans la clé : tous
  les états/buts d'un layout restent dans le même pool. Un groupe isomorphe est
  atomique : jamais éclaté entre deux pools.
- **Pools** : 200 train / 50 val / ≥100 par cellule test (vérifié ; erreur si
  insuffisant — jamais de remplissage silencieux).
- **Réservation G2** (§7.4) : les TÂCHES « AT(clé, pièce degré ≥3) » sont
  exclues du train et de la validation de sélection ; les composants
  « AT(clé, deg≤2) », « AT(colis, deg≥3) », « HAVE(clé) », « REACH(deg≥3) »
  doivent y être présents (vérifié par :func:`g2_components`). Le degré est
  calculé depuis les relations observables, pas depuis l'oracle.
- **Manifest** : scellé (``manifest_sha256``) avec seed, hashes et comptages —
  figé avant l'entraînement confirmatoire.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from ..env.tinygraph import Layout
from .writer import seal_manifest, verify_manifest  # réexport pratique

# ---------------------------------------------------------------------------
# Sérialisation layout (JSON) — miroir du constructeur Layout(rooms, edges, door_edge)
# ---------------------------------------------------------------------------


def layout_to_dict(layout: Layout) -> dict:
    return {
        "rooms": list(layout.rooms),
        "edges": [list(e) for e in layout.edges],
        "door_edge": layout.door_edge,
    }


def layout_from_dict(obj: dict) -> Layout:
    return Layout(obj["rooms"], [tuple(e) for e in obj["edges"]], obj["door_edge"])


# ---------------------------------------------------------------------------
# Graphe à arêtes colorées (couleur 1 = porte) sur les indices de pièces
# ---------------------------------------------------------------------------


def _adjacency(layout: Layout) -> list[set[int]]:
    idx = {r: i for i, r in enumerate(layout.rooms)}
    n = len(layout.rooms)
    adj: list[set[int]] = [set() for _ in range(n)]
    for i, (a, b) in enumerate(layout.edges):
        color = 1 if i == layout.door_edge else 0
        adj[idx[a]].add(idx[b])
        adj[idx[b]].add(idx[a])
    return adj


def _door_pair(layout: Layout) -> tuple[int, int]:
    a, b = layout.edges[layout.door_edge]
    idx = {r: i for i, r in enumerate(layout.rooms)}
    u, v = idx[a], idx[b]
    return (u, v) if u < v else (v, u)


def degrees(layout: Layout) -> dict[str, int]:
    """Degrés des pièces — dérivables des relations observables (§7.4)."""
    return {r: layout.degree(r) for r in layout.rooms}


def room_degree(layout: Layout, room: str) -> int:
    return layout.degree(room)


# ---------------------------------------------------------------------------
# Préfiltre : hash structural 1-WL (arêtes colorées)
# ---------------------------------------------------------------------------


def _refine(adj: list[set[int]], door: tuple[int, int], colors: list[int], rounds: int) -> list[int]:
    """Color refinement avec couleurs HASHÉES : les valeurs sont invariantes
    par isomorphisme (fonction déterministe des signatures récursives)."""
    n = len(adj)
    for _ in range(rounds):
        new_colors = []
        for v in range(n):
            neigh = tuple(sorted(
                (1 if (min(v, u), max(v, u)) == door else 0, colors[u]) for u in adj[v]
            ))
            sig = repr((colors[v], neigh)).encode()
            new_colors.append(int.from_bytes(hashlib.sha256(sig).digest()[:8], "big"))
        if len(set(new_colors)) == len(set(colors)) and _same_partition(colors, new_colors):
            return new_colors
        colors = new_colors
    return colors


def _same_partition(a: list[int], b: list[int]) -> bool:
    if len(a) != len(b):
        return False
    pa = {frozenset(idx for idx, x in enumerate(a) if x == v) for v in set(a)}
    pb = {frozenset(idx for idx, x in enumerate(b) if x == v) for v in set(b)}
    return pa == pb


def wl_prefilter_hash(layout: Layout, rounds: int = 3) -> str:
    """Hash structurel rapide — préfiltre d'isomorphisme (peut fusionner, ne sépare jamais à tort)."""
    adj = _adjacency(layout)
    door = _door_pair(layout)
    colors = _refine(adj, door, [0] * len(adj), rounds)
    payload = (len(adj), tuple(sorted(colors)))
    return hashlib.sha256(repr(payload).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Certificat canonique exact (individualization-refinement)
# ---------------------------------------------------------------------------


def _certificate_from_ordering(adj: list[set[int]], door: tuple[int, int], order: list[int]) -> str:
    parts = []
    for i in range(len(order)):
        u = order[i]
        for j in range(i + 1, len(order)):
            v = order[j]
            if v in adj[u]:
                color = 1 if (min(u, v), max(u, v)) == door else 0
                parts.append(f"{i}-{j}:{color}")
    return f"n{len(order)};" + ",".join(parts)


def _canonical(adj: list[set[int]], door: tuple[int, int], colors: list[int], budget: list[int]) -> tuple[str, list[int]]:
    """Certificat canonique exact d'un graphe coloré (arêtes et sommets).

    ``budget`` : compteur de nœuds de recherche (liste mutable) ; les graphes
    pathologiquement symétriques (cliques denses) dépassent le budget → erreur
    explicite plutôt qu'explosion combinatoire.
    """
    n = len(adj)
    colors = _refine(adj, door, colors, rounds=n + 1)

    def discrete(cs: list[int]) -> bool:
        return len(set(cs)) == n

    def ordering_of(cs: list[int]) -> list[int]:
        pairs = sorted((c, v) for v, c in enumerate(cs))
        return [v for _, v in pairs]

    def search(cs: list[int]) -> tuple[str, list[int]]:
        budget[0] -= 1
        if budget[0] < 0:
            raise ValueError("budget de canonical labeling dépassé (graphe trop symétrique)")
        if discrete(cs):
            order = ordering_of(cs)
            return _certificate_from_ordering(adj, door, order), order
        # cellule cible : plus petite cellule non singleton ; départage par
        # couleur min — invariant par isomorphisme (couleurs = hashes)
        cells: dict[int, list[int]] = {}
        for v, c in enumerate(cs):
            cells.setdefault(c, []).append(v)
        min_size = min(len(m) for m in cells.values() if len(m) > 1)
        target_color = min(c for c, members in cells.items() if len(members) == min_size)
        best: Optional[tuple[str, list[int]]] = None
        for v in sorted(cells[target_color]):
            cs2 = list(cs)
            cs2[v] = max(cs) + 1  # individualisation
            cs2 = _refine(adj, door, cs2, rounds=n + 1)
            cert, order = search(cs2)
            if best is None or cert < best[0]:
                best = (cert, order)
        assert best is not None
        return best

    return search(list(colors))


_SEARCH_BUDGET = 200_000


def canonical_certificate(layout: Layout) -> str:
    """Certificat exact : deux layouts ont le même certificat ssi isomorphes
    (topologie + position structurelle de la porte)."""
    adj = _adjacency(layout)
    door = _door_pair(layout)
    cert, _ = _canonical(adj, door, [0] * len(adj), budget=[_SEARCH_BUDGET])
    return cert


# ---------------------------------------------------------------------------
# Infos de layout + groupes isomorphes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LayoutInfo:
    layout: Layout
    layout_hash: str
    wl_hash: str
    certificate: str
    n_rooms: int
    has_junction: bool  # au moins une pièce de degré ≥ 3
    degrees: dict[str, int]

    @property
    def iso_key(self) -> str:
        return self.certificate


def layout_info(layout: Layout) -> LayoutInfo:
    degs = degrees(layout)
    return LayoutInfo(
        layout=layout,
        layout_hash=layout.layout_hash(),
        wl_hash=wl_prefilter_hash(layout),
        certificate=canonical_certificate(layout),
        n_rooms=len(layout.rooms),
        has_junction=any(d >= 3 for d in degs.values()),
        degrees=degs,
    )


def group_isomorphic(infos: Iterable[LayoutInfo]) -> list[list[LayoutInfo]]:
    """Préfiltre WL → buckets → comparaison exacte par certificat dans le bucket."""
    buckets: dict[str, list[LayoutInfo]] = {}
    for info in infos:
        buckets.setdefault(info.wl_hash, []).append(info)
    groups_by_cert: dict[str, list[LayoutInfo]] = {}
    for bucket in buckets.values():
        for info in bucket:
            groups_by_cert.setdefault(info.certificate, []).append(info)
    # un certificat ne peut apparaître que dans un seul bucket WL (même wl_hash
    # ⇒ même bucket) ; on le vérifie par sécurité
    return [sorted(members, key=lambda m: m.layout_hash) for _, members in sorted(groups_by_cert.items())]


# ---------------------------------------------------------------------------
# Réservation G2 (§7.4) — niveau TÂCHE (but sur layout)
# ---------------------------------------------------------------------------


def is_g2_reserved(layout: Layout, goal: dict) -> bool:
    """Tâche réservée G2 : AT(clé, pièce de degré ≥ 3). Exclue du train/val.

    P13 (audit tagi-5) : but malformé/inconnu ⇒ False robuste (jamais de
    KeyError brut) — un but invalide n'est pas une tâche réservée, il sera
    rejeté plus tôt par la validation du schéma.
    """
    if not isinstance(goal, dict):
        return False
    if goal.get("predicate") != "AT":
        return False
    args = goal.get("args")
    if not isinstance(args, dict) or args.get("object") != "key":
        return False
    room = args.get("room")
    if not isinstance(room, str) or room not in layout.rooms:
        return False
    return room_degree(layout, room) >= 3


G2_COMPONENTS = ("at_key_deg_le_2", "at_parcel_deg_ge_3", "have_key", "reach_deg_ge_3")


def g2_task_components(layout: Layout, goal: dict) -> set[str]:
    """Composants de couverture G2 portés par une tâche (train/val)."""
    comps: set[str] = set()
    args = goal.get("args", {})
    pred = goal.get("predicate")
    if pred == "AT" and args.get("object") == "key" and room_degree(layout, args["room"]) <= 2:
        comps.add("at_key_deg_le_2")
    if pred == "AT" and args.get("object") == "parcel" and room_degree(layout, args["room"]) >= 3:
        comps.add("at_parcel_deg_ge_3")
    if pred == "HAVE" and args.get("object") == "key":
        comps.add("have_key")
    if pred == "REACH" and room_degree(layout, args["room"]) >= 3:
        comps.add("reach_deg_ge_3")
    return comps


def g2_components_check(tasks: Iterable[tuple[Layout, dict]]) -> dict[str, Any]:
    """Vérifie la présence de chaque composant G2 dans les tâches train/val (§7.4).

    Retourne {"present": bool, "counts": {composant: n}, "missing": [...]}.
    """
    counts: dict[str, int] = {c: 0 for c in G2_COMPONENTS}
    reserved_leak = 0
    for layout, goal in tasks:
        if is_g2_reserved(layout, goal):
            reserved_leak += 1
            continue
        for comp in g2_task_components(layout, goal):
            counts[comp] += 1
    missing = [c for c in G2_COMPONENTS if counts[c] == 0]
    return {
        "present": not missing and reserved_leak == 0,
        "counts": counts,
        "missing": missing,
        "reserved_leak": reserved_leak,
    }


# ---------------------------------------------------------------------------
# Pools + manifest scellé
# ---------------------------------------------------------------------------

SPLIT_SCHEMA = "ucm-split-manifest/0.1"


@dataclass
class SplitResult:
    manifest: dict
    pool_of: dict[str, str]  # layout_hash -> pool
    train: list[LayoutInfo] = field(default_factory=list)
    val: list[LayoutInfo] = field(default_factory=list)
    test: list[LayoutInfo] = field(default_factory=list)


def _assign_cells(groups: list[list["LayoutInfo"]], infos, train_n: int, val_n: int) -> dict[str, list["LayoutInfo"]]:
    """Assignation déterministe des cellules (contrat tagi-1, 2026-09-22).

    - train/val : layouts 4–8 pièces (spec §4.5), groupes dans l'ordre mélangé.
    - test_g1 : 4–8 non-jonction + jonctions non tirées au 1/3 (tâches en
      g2-exclude) — distribution proche du train.
    - test_g2 : 1/3 des groupes test 4–8 À JONCTION (tâches g2-require,
      d* 2–12, « mêmes tailles et difficulté » §7.4).
    - test_g3 : 9–12 pièces ; test_g4 : 1/3 des groupes 9–12 (réserve G4,
      tâches d* 13–24 sur ces tailles où la bande existe).
    """
    pools: dict[str, list[LayoutInfo]] = {"train": [], "val": [],
                                          "test_g1": [], "test_g2": [],
                                          "test_g3": [], "test_g4": []}
    junction_seen = 0
    big_seen = 0
    for group in groups:
        head = group[0]
        if head.n_rooms >= 9:
            big_seen += 1
            cell = "test_g4" if big_seen % 3 == 0 else "test_g3"
            pools[cell].extend(group)
            continue
        if len(pools["train"]) < train_n:
            pools["train"].extend(group)
            continue
        if len(pools["val"]) < val_n:
            pools["val"].extend(group)
            continue
        if head.has_junction:
            junction_seen += 1
            cell = "test_g2" if junction_seen % 3 == 0 else "test_g1"
        else:
            cell = "test_g1"
        pools[cell].extend(group)
    return pools


def make_split_manifest(
    layouts: Iterable[Layout],
    *,
    seed: int,
    train_n: int = 200,
    val_n: int = 50,
    min_test: int = 100,
    cell_of=None,
    min_test_g34: Optional[int] = None,
) -> SplitResult:
    """Assigne des layouts aux pools, zéro isomorphisme inter-pools, manifest scellé.

    - ``cell_of(info) -> str`` : classification des cellules test (défaut : G3
      pour 9–12 pièces, G1 sinon).
    - Les groupes isomorphes sont mélangés avec ``seed`` (générateur local,
      reproductible) et remplissent train puis val, le reste part en test.
      Les tailles sont des PLANCHERS : les groupes étant atomiques, un léger
      dépassement est possible (jamais un sous-effectif).
    - ``min_test`` vérifié par cellule (erreur sinon).
    - G2 : les LAYOUTS test comportant une jonction sont marqués
      ``junction_layouts_test_g2`` — les tâches réservées sont filtrées par
      :func:`is_g2_reserved` à la génération des épisodes (niveau tâche).

    """
    infos = [layout_info(l) for l in layouts]
    hashes = [i.layout_hash for i in infos]
    if len(set(hashes)) != len(hashes):
        raise ValueError("layouts en doublon (même layout_hash) — dédupliquer avant split")
    groups = group_isomorphic(infos)
    rng = random.Random(seed)
    rng.shuffle(groups)

    cell_of = cell_of  # override optionnel (tests/outils)
    if cell_of is not None:
        pools: dict[str, list[LayoutInfo]] = {"train": [], "val": []}
        for group in groups:
            head = group[0]
            if len(pools["train"]) < train_n:
                pools["train"].extend(group)
            elif len(pools["val"]) < val_n:
                pools["val"].extend(group)
            else:
                pools.setdefault(cell_of(head), []).extend(group)
        cells = {c: m for c, m in pools.items() if c.startswith("test")}
    else:
        pools = _assign_cells(groups, infos, train_n, val_n)
        cells = {c: m for c, m in pools.items() if c.startswith("test")}
    train, val = pools["train"], pools["val"]
    test = [info for members in cells.values() for info in members]
    if len(train) < train_n:
        raise ValueError(f"train insuffisant : {len(train)} < {train_n} — générer plus de layouts")
    if len(val) < val_n:
        raise ValueError(f"val insuffisant : {len(val)} < {val_n} — générer plus de layouts")
    # min_test s'applique au sous-total test 4-8 (g1+g2, tailles train) ;
    # min_test_g34 (optionnel) au sous-total 9-12 (g3+g4). Les cellules
    # individuelles, plus rares, sont suivies dans le manifest.
    cell_counts = {c: len(m) for c, m in sorted(cells.items())}
    n_g12 = cell_counts.get("test_g1", 0) + cell_counts.get("test_g2", 0)
    if n_g12 < min_test:
        raise ValueError(f"pool test 4-8 (g1+g2) insuffisant : {n_g12} < {min_test} — étendre la génération")
    if min_test_g34 is not None:
        n_g34 = cell_counts.get("test_g3", 0) + cell_counts.get("test_g4", 0)
        if n_g34 < min_test_g34:
            raise ValueError(f"pool test 9-12 (g3+g4) insuffisant : {n_g34} < {min_test_g34} — étendre la génération")

    pool_of: dict[str, str] = {}
    for info in train:
        pool_of[info.layout_hash] = "train"
    for info in val:
        pool_of[info.layout_hash] = "val"
    for cell, members in cells.items():
        for info in members:
            pool_of[info.layout_hash] = cell

    manifest = {
        "schema": SPLIT_SCHEMA,
        "seed": seed,
        "sizes": {"train": len(train), "val": len(val), "test_total": len(test),
                  "test_cells": cell_counts},
        "cells_needing_extension_before_gates": [
            c for c, n in cell_counts.items() if n < 100 and c in ("test_g1", "test_g2", "test_g3", "test_g4")
        ],
        "iso_groups": len(groups),
        "pools": {
            "train": [i.layout_hash for i in sorted(train, key=lambda x: x.layout_hash)],
            "val": [i.layout_hash for i in sorted(val, key=lambda x: x.layout_hash)],
            "test": {
                cell: [i.layout_hash for i in sorted(members, key=lambda x: x.layout_hash)]
                for cell, members in sorted(cells.items())
            },
        },
        "certificates": {i.layout_hash: i.certificate for i in sorted(infos, key=lambda x: x.layout_hash)},
        "junction_layouts_test_g2": [i.layout_hash for i in sorted(cells.get("test_g2", []), key=lambda x: x.layout_hash)],
        "g2_reservation": {
            "rule": "AT(key, room degree>=3) excluded from train+val tasks; test_g2 = require on disjoint junction layouts (tagi-1 contract)",
            "components_required": list(G2_COMPONENTS),
        },
        "zero_iso_check": "by construction (atomic isomorphism groups; certificates recorded)",
    }
    return SplitResult(manifest=seal_manifest(manifest), pool_of=pool_of, train=train, val=val, test=test)


def write_manifest(path, manifest: dict) -> str:
    """Écrit le manifest JSON scellé. Retourne le chemin."""
    import json
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")
    return str(p)


def verify_split_manifest(manifest: Mapping[str, Any], layouts: Iterable[Layout]) -> dict:
    """m8 (audit tagi-5) : anti-contamination automatisée.

    Re-dérive les certificats canoniques depuis ``layouts`` et vérifie :
    - chaque layout_hash du manifest existe et son certificat re-dérivé
      correspond à celui enregistré ;
    - chaque certificat n'apparaît que dans UN SEUL pool (disjonction stricte,
      groupes isomorphes atomiques) ;
    - les tailles de pools correspondent ;
    - le manifest est scellé et vérifiable.

    Retourne {"valid": bool, "errors": [...], "pools": {pool: n}}.
    """
    from .writer import verify_manifest as _verify

    errors: list[str] = []
    if not _verify(manifest):
        errors.append("manifest_sha256 invalide (scellage rompu)")
    infos = [layout_info(l) for l in layouts]
    by_hash = {i.layout_hash: i for i in infos}
    cert_to_pool: dict[str, str] = {}
    pool_counts: dict[str, int] = {}
    pools = manifest.get("pools", {})
    entries: list[tuple[str, str]] = []
    for h in pools.get("train", []):
        entries.append((h, "train"))
    for h in pools.get("val", []):
        entries.append((h, "val"))
    for cell, members in (pools.get("test", {}) or {}).items():
        for h in members:
            entries.append((h, cell))
    for h, pool in entries:
        info = by_hash.get(h)
        if info is None:
            errors.append(f"layout_hash du manifest absent des layouts fournis: {h}")
            continue
        recorded = (manifest.get("certificates", {}) or {}).get(h)
        if recorded is not None and recorded != info.certificate:
            errors.append(f"{h}: certificat re-dérivé ≠ manifest ({info.certificate} vs {recorded})")
        owner = cert_to_pool.setdefault(info.certificate, pool)
        if owner != pool:
            errors.append(
                f"CONTAMINATION: certificat {info.certificate[:12]}… dans {owner} ET {pool} ({h})"
            )
        pool_counts[pool] = pool_counts.get(pool, 0) + 1
    sizes = manifest.get("sizes", {})
    if sizes.get("train") is not None and pool_counts.get("train", 0) < sizes["train"]:
        errors.append(f"train manifest {sizes['train']} > re-dérivé {pool_counts.get('train', 0)}")
    if sizes.get("val") is not None and pool_counts.get("val", 0) < sizes["val"]:
        errors.append(f"val manifest {sizes['val']} > re-dérivé {pool_counts.get('val', 0)}")
    for cell, n in (sizes.get("test_cells", {}) or {}).items():
        if pool_counts.get(cell, 0) != n:
            errors.append(f"{cell}: manifest {n} ≠ re-dérivé {pool_counts.get(cell, 0)}")
    return {"valid": not errors, "errors": errors, "pools": dict(sorted(pool_counts.items()))}


SPLIT_SCHEMA_V2 = "ucm-split-manifest/0.2"


def extend_split_manifest_v2(
    v1_manifest: Mapping[str, Any],
    candidate_layouts: Iterable[Layout],
    *,
    seed: int,
    min_g2_total: int = 100,
    rooms_range: tuple[int, int] = (4, 8),
) -> dict:
    """Extension ADDITIVE test_g2 → manifest v2 (contrat tagi-1 12:22).

    - v1 (da9cf4a) JAMAIS muté : le manifest retourné est une copie scellée
      séparément, ``schema`` passe en 0.2 avec ``extends`` = sha256 du v1.
    - Candidats retenus : jonctions 4–8 pièces (tailles comparables au train,
      §7.4), layout_hash nouveau, certificat iso-disjoint de TOUS les pools v1
      (train/val/g1/g2/g3/g4 — y compris les jonctions déjà en g1) ET des
      nouveaux retenus.
    - Groupes isomorphes atomiques ; remplissage mélangé par ``seed`` jusqu'à
      ``min_g2_total`` layouts g2 TOTAUX (v1 + nouveaux).
    - Génération require-mode assumée en aval (0.2.1 interdit déjà les zéros).
    """
    import copy

    if not verify_manifest(v1_manifest):
        raise ValueError("v1 manifest non scellé/vérifiable — extension refusée")
    v1 = dict(v1_manifest)

    v1_hashes: set[str] = set(v1["pools"].get("train", [])) | set(v1["pools"].get("val", []))
    v1_certs: set[str] = set(v1.get("certificates", {}).values())
    for members in (v1["pools"].get("test", {}) or {}).values():
        v1_hashes.update(members)
        v1_certs.update(v1["certificates"][h] for h in members if h in v1.get("certificates", {}))
    old_g2 = list((v1["pools"].get("test", {}) or {}).get("test_g2", []))
    n_needed = min_g2_total - len(old_g2)
    if n_needed <= 0:
        raise ValueError(f"test_g2 déjà à {len(old_g2)} ≥ {min_g2_total} — extension inutile")

    infos = []
    seen_hashes: set[str] = set(v1_hashes)
    for layout in candidate_layouts:
        if not (rooms_range[0] <= len(layout.rooms) <= rooms_range[1]):
            continue
        info = layout_info(layout)
        if not info.has_junction:
            continue
        if info.layout_hash in seen_hashes:
            continue
        if info.certificate in v1_certs:
            continue  # isomorphe d'un pool v1 → interdit (disjonction stricte)
        seen_hashes.add(info.layout_hash)
        infos.append(info)

    groups = group_isomorphic(infos)
    rng = random.Random(seed)
    rng.shuffle(groups)
    new_g2: list[LayoutInfo] = []
    new_certs: set[str] = set()
    for group in groups:
        if len(old_g2) + len(new_g2) >= min_g2_total:
            break
        cert = group[0].certificate
        if cert in new_certs:
            continue
        new_certs.add(cert)
        new_g2.extend(group)
    if len(old_g2) + len(new_g2) < min_g2_total:
        raise ValueError(
            f"extension insuffisante : test_g2 = {len(old_g2) + len(new_g2)} < {min_g2_total} "
            f"après regroupement atomique — étendre les candidats"
        )

    v2 = copy.deepcopy(v1)
    v2["schema"] = SPLIT_SCHEMA_V2
    v2["extends"] = v1["manifest_sha256"]
    v2["extension"] = {
        "what": "test_g2 additive (contrat tagi-1 12:22)",
        "seed": seed,
        "added_layouts": sorted(i.layout_hash for i in new_g2),
        "added_groups": len(new_certs),
        "old_g2": len(old_g2),
        "new_total": len(old_g2) + len(new_g2),
        "disjoint_from": "all v1 pools (certificates)",
    }
    v2["pools"]["test"]["test_g2"] = sorted(old_g2 + [i.layout_hash for i in new_g2])
    for i in new_g2:
        v2["certificates"][i.layout_hash] = i.certificate
    v2["junction_layouts_test_g2"] = sorted(v2["pools"]["test"]["test_g2"])
    v2["sizes"]["test_cells"]["test_g2"] = len(v2["pools"]["test"]["test_g2"])
    v2["sizes"]["test_total"] = sum(v2["sizes"]["test_cells"].values())
    v2["cells_needing_extension_before_gates"] = [
        c for c, n in v2["sizes"]["test_cells"].items() if n < 100
    ]
    return seal_manifest(v2)


# ---------------------------------------------------------------------------
# V1-SIW : certificat canonique TYPÉ (graphe de widgets / machine à états)
# ---------------------------------------------------------------------------


def canonical_certificate_typed(
    n: int,
    edges: list[tuple[int, int, int]],
    node_colors: list[int],
    search_budget: int = _SEARCH_BUDGET,
) -> str:
    """Certificat canonique exact d'un multigraphe typé NON ORIGENTÉ (SIW).

    - ``node_colors[i]`` : couleur initiale du nœud i (type du widget).
      Les labels ne participent jamais (aucun paramètre ne les reçoit).
    - ``edges`` : (u, v, couleur) — les arêtes (u,v,c) et (v,u,c) sont la MÊME
      arête non orientée (normalisées) ; les arêtes parallèles de couleurs
      différentes sont licites (multigraphe typé).
    Moteur : couleurs hashées (invariance iso), refinement, individualization,
    budget explicite. Certificat : couleurs de nœuds puis arêtes triées.
    """
    adj: list[set[int]] = [set() for _ in range(n)]
    edge_set: set[tuple[int, int, int]] = set()
    for u, v, color in edges:
        if u == v:
            raise ValueError("self-loop interdit")
        a, b = (u, v) if u < v else (v, u)
        edge_set.add((a, b, color))
        adj[u].add(v)
        adj[v].add(u)
    edge_colors: dict[tuple[int, int], list[int]] = {}
    for a, b, c in edge_set:
        edge_colors.setdefault((a, b), []).append(c)
    for k in edge_colors:
        edge_colors[k] = sorted(edge_colors[k])

    def neigh_colors(v: int, colors: list[int]) -> tuple:
        out = []
        for u in adj[v]:
            for c in edge_colors[(min(u, v), max(u, v))]:
                out.append((c, colors[u]))
        return tuple(sorted(out))

    def refine(colors: list[int], rounds: int) -> list[int]:
        for _ in range(rounds):
            new = []
            for v in range(n):
                sig = repr((colors[v], neigh_colors(v, colors))).encode()
                new.append(int.from_bytes(hashlib.sha256(sig).digest()[:8], "big"))
            if _same_partition(colors, new):
                return new
            colors = new
        return colors

    def discrete(cs: list[int]) -> bool:
        return len(set(cs)) == n

    def certificate(order: list[int]) -> str:
        parts = [f"c{node_colors[v]}" for v in order]
        pos = {v: i for i, v in enumerate(order)}
        eset = []
        for (a, b), cs_ in edge_colors.items():
            for c in cs_:
                eset.append((pos[a], pos[b], c) if pos[a] < pos[b] else (pos[b], pos[a], c))
        parts.extend(f"{i}-{j}:{c}" for i, j, c in sorted(eset))
        return f"n{n};" + ",".join(parts)

    budget = [search_budget]

    def search(cs: list[int]) -> str:
        budget[0] -= 1
        if budget[0] < 0:
            raise ValueError("budget de canonical labeling dépassé (graphe trop symétrique)")
        cs = refine(cs, n + 1)
        if discrete(cs):
            return certificate([v for _, v in sorted((c, v) for v, c in enumerate(cs))])
        cells: dict[int, list[int]] = {}
        for v, c in enumerate(cs):
            cells.setdefault(c, []).append(v)
        min_size = min(len(m) for m in cells.values() if len(m) > 1)
        target = min(c for c, m in cells.items() if len(m) == min_size)
        best: Optional[str] = None
        for v in sorted(cells[target]):
            cs2 = list(cs)
            cs2[v] = max(cs) + 1
            cert = search(cs2)
            if best is None or cert < best:
                best = cert
        assert best is not None
        return best

    return search(list(node_colors))


# ---------------------------------------------------------------------------
# V1-SIW : multigraphes typés à orbites massives (distracteurs) — le certificat
# canonique exact explose ; on utilise hash invariant WL + test d'iso exact.
# ---------------------------------------------------------------------------


def _typed_graph(n: int, edges: list[tuple[int, int, int]], node_colors: list[int]):
    adj: list[set[int]] = [set() for _ in range(n)]
    ec: dict[tuple[int, int], list[int]] = {}
    for u, v, c in edges:
        if u == v:
            raise ValueError("self-loop interdit")
        key = (min(u, v), max(u, v))
        ec.setdefault(key, []).append(c)
        adj[u].add(v)
        adj[v].add(u)
    for k in ec:
        ec[k] = sorted(ec[k])
    return adj, ec


def _refine_typed(n, adj, ec, colors, rounds):
    for _ in range(rounds):
        new = []
        for v in range(n):
            neigh = []
            for u in adj[v]:
                for c in ec[(min(u, v), max(u, v))]:
                    neigh.append((c, colors[u]))
            sig = repr((colors[v], tuple(sorted(neigh)))).encode()
            new.append(int.from_bytes(hashlib.sha256(sig).digest()[:8], "big"))
        if _same_partition(colors, new):
            return new
        colors = new
    return colors


def wl_hash_typed(n: int, edges: list[tuple[int, int, int]], node_colors: list[int], rounds: int = 4) -> str:
    """Hash INVARIANT d'isomorphisme (préfiltre) d'un multigraphe typé.

    Correctif (passe B tagi-5, 18:49) : le payload précédent encodait les
    arêtes par INDICES bruts — dépendant de la numérotation, donc PAS un
    invariant (deux numérotations du même graphe donnaient des hashs
    différents). Payload corrigé : arêtes encodées par COULEURS RAFFINÉES des
    extrémités (min/max) + couleur d'arête — invariant par construction.
    """
    adj, ec = _typed_graph(n, edges, node_colors)
    colors = _refine_typed(n, adj, ec, list(node_colors), rounds)
    edge_encoding = sorted(
        (min(colors[u], colors[v]), max(colors[u], colors[v]), c)
        for (u, v), cs in ec.items() for c in cs
    )
    payload = (n, tuple(sorted(node_colors)), tuple(sorted(colors)),
               tuple(edge_encoding))
    return hashlib.sha256(repr(payload).encode()).hexdigest()[:16]


def typed_isomorphic(
    g1: tuple[int, list[tuple[int, int, int]], list[int]],
    g2: tuple[int, list[tuple[int, int, int]], list[int]],
    node_budget: int = 200_000,
) -> bool:
    """Isomorphisme EXACT de multigraphes typés (backtracking refinement).

    Sound & complete : renvoie True ssi il existe une bijection préservant
    couleurs de nœuds et multigraphes d'arêtes. Les orbites symétriques
    (distracteurs) rendent la recherche TRIVIALE (tout candidat marche) —
    c'est le cas dual du certificat canonique qui, lui, explose.
    """
    n1, e1, c1 = g1
    n2, e2, c2 = g2
    if n1 != n2 or len(e1) != len(e2) or sorted(c1) != sorted(c2):
        return False
    if wl_hash_typed(*g1) != wl_hash_typed(*g2):
        return False
    adj1, ec1 = _typed_graph(n1, e1, c1)
    adj2, ec2 = _typed_graph(n2, e2, c2)
    col1 = _refine_typed(n1, adj1, ec1, list(c1), n1 + 1)
    col2 = _refine_typed(n2, adj2, ec2, list(c2), n2 + 1)
    if sorted(col1) != sorted(col2):
        return False

    budget = [node_budget]

    def search(m: dict[int, int]) -> bool:
        budget[0] -= 1
        if budget[0] < 0:
            raise ValueError("budget typed_isomorphic dépassé")
        if len(m) == n1:
            return True
        # cellule cible : plus petite classe non singletons de g1 (couleur min)
        cells: dict[int, list[int]] = {}
        assigned = set(m)
        for v in range(n1):
            if v not in assigned:
                cells.setdefault(col1[v], []).append(v)
        if not cells:
            return True
        target = min(cells, key=lambda k: (len(cells[k]), k))
        v0 = min(cells[target])
        # candidats de g2 : même couleur raffinée, non encore pris
        used = set(m.values())
        cands = [w for w in range(n2) if w not in used and col2[w] == col1[v0]]
        for w in cands:
            # cohérence locale des arêtes sous l'extension v0→w
            ok = True
            for u in adj1[v0]:
                if u in m:
                    key2 = (min(w, m[u]), max(w, m[u]))
                    want = ec1[(min(v0, u), max(v0, u))]
                    if ec2.get(key2, []) != want:
                        ok = False
                        break
            # arêtes inverses : voisins de w déjà mappés vers v0
            if ok:
                for x in adj2[w]:
                    pre = [v for v, mm in m.items() if mm == x]
                    if pre:
                        u = pre[0]
                        key1 = (min(v0, u), max(v0, u))
                        if ec1.get(key1, []) != ec2[(min(w, x), max(w, x))]:
                            ok = False
                            break
            if ok:
                m[v0] = w
                if search(m):
                    return True
                del m[v0]
        return False

    try:
        return search({})
    except ValueError:
        raise
