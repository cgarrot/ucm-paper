#!/usr/bin/env python3
"""M-V1a tagi-4 — régénère les scellés SIW M-V1b et imprime les empreintes.

Usage: python scripts/repro_siw_seals.py <out_dir>   (depuis n'importe où)

Construit depuis les paramètres publiés (run-config-siw-mv1b.json +
generation_params de inventory-siw-dev.json) :
  1. inventaire seed 20260924 (30 layouts, views 2-5, goals 12, profile small,
     dialog p=0.5) + dumps couples k{0,100,500,2000,10000}
  2. split manifest seed 20260926 (420 layouts, ordre de génération)
     + certificates_kind_aware + memberships, puis scellage
  3. épisodes test seed 20260927 (pool test du manifest trié par hash,
     bande 2-8, 150/prédicat)

Validé inter-machines : 9/9 empreintes identiques Mac (Python 3.12.13) ↔
XMG Debian (Python 3.13.5), 22/09/2026 (tagi-4).
"""
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ucm.data.schema import sha256_hex
from ucm.data.siw_pipeline import (
    _layout_iso_hash,
    build_siw_inventory,
    build_siw_test_episodes,
    make_siw_split_manifest,
    write_siw_inventory,
    write_siw_test_episodes,
)
from ucm.data.writer import seal_manifest, verify_manifest
from ucm.env.siw import generate_siw_layout


def gen_layouts(seed: int, n: int, vmin: int = 2, vmax: int = 5, profile: str = "small"):
    rng = random.Random(seed)
    return [
        generate_siw_layout(rng, rng.randint(vmin, vmax), with_dialog=rng.random() < 0.5, profile=profile)
        for _ in range(n)
    ]


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def couples_seal(p: Path) -> str:
    lines = p.read_text(encoding="utf-8").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return sha256_hex(lines)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # 1) inventaire + couples (seed 20260924, params publication)
    layouts_inv = gen_layouts(20260924, 30)
    gen_params = {
        "generator": "generate_siw_layout",
        "goals_per_layout": 12,
        "layouts": 30,
        "profile": "small",
        "views_range": [2, 5],
        "with_dialog": "p=0.5",
    }
    inv = build_siw_inventory(
        layouts_inv, seed=20260924, goals_per_layout=12,
        generation_params=gen_params, covered_sets_dir=str(out),
    )
    write_siw_inventory(inv, out / "inventory-siw-dev.json")

    # dump layouts comme le CLI (auditabilité)
    (out / "inventory-siw-dev-layouts.json").write_text(
        json.dumps(
            {l.layout_hash(): {"views": l.views, "widgets": l.widgets,
                               "nav_edges": [list(e) for e in l.nav_edges], "labels": l.labels}
             for l in layouts_inv},
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    # 2) split manifest (seed 20260926, 420 layouts, ordre de génération)
    # + certificates_kind_aware (par layout) + memberships, puis scellage.
    layouts_split = gen_layouts(20260926, 420)
    manifest = make_siw_split_manifest(layouts_split, seed=20260926, train_n=200, val_n=50, min_test=100)
    manifest["certificates_kind_aware"] = {l.layout_hash(): _layout_iso_hash(l) for l in layouts_split}
    manifest["memberships"] = {h: pool for pool, hs in manifest["pools"].items() for h in hs}
    manifest = seal_manifest(manifest)
    (out / "split-manifest-SIW-v1.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")

    # 3) épisodes test (seed 20260927, pool test du manifest SIW trié par hash)
    test_set = set(manifest["pools"]["test"])
    test_layouts = sorted(
        [l for l in layouts_split if l.layout_hash() in test_set], key=lambda l: l.layout_hash())
    ep_manifest, episodes = build_siw_test_episodes(
        test_layouts, seed=20260927, n_per_predicate=150, band=(2, 8))
    write_siw_test_episodes(episodes, ep_manifest, str(out / "siw-test-episodes"))

    # --- rapport ---
    print("== EMPREINTES GÉNÉRÉES ==")
    for k in (0, 100, 500, 2000, 10000):
        p = out / f"siw-couples-k{k}.jsonl"
        print(f"couples k{k:<6d} bytes={sha_file(p)} seal={couples_seal(p)}")
    print(f"test-episodes.jsonl    bytes={sha_file(out / 'siw-test-episodes.jsonl')}")
    print(f"test-episodes-manifest bytes={sha_file(out / 'siw-test-episodes-manifest.json')} "
          f"seal={ep_manifest['manifest_sha256']} verify={verify_manifest(ep_manifest)}")
    print(f"inventory              bytes={sha_file(out / 'inventory-siw-dev.json')} "
          f"seal={inv['manifest_sha256']} verify={verify_manifest(inv)}")
    print(f"split-manifest         bytes={sha_file(out / 'split-manifest-SIW-v1.json')} "
          f"seal={manifest['manifest_sha256']} verify={verify_manifest(manifest)}")
    print(f"pool test: {len(test_layouts)} layouts (manifest sizes: {manifest['sizes']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
