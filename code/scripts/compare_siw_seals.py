#!/usr/bin/env python3
"""M-V1a tagi-4 — compare les scellés SIW régénérés aux références Mac figées.

Usage: python scripts/compare_siw_seals.py <out_dir>   (depuis n'importe où)

Références = fichiers Mac (artifacts/) : bytes sha256 + scellés embarqués.
L'inventaire est comparé par son SCELLÉ uniquement (timing exclu par design).
Sortie : ligne par fichier + verdict N/N. Exit 0 si tout est identique.

Validé inter-machines : 9/9 identiques Mac ↔ XMG Debian, 22/09/2026 (tagi-4).
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ucm.data.schema import sha256_hex

# (bytes_sha256 | None, seal | None)
REF = {
    "siw-couples-k0.jsonl": (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"),
    "siw-couples-k100.jsonl": (
        "f364993afbe6f7df32b0a39073176d273317a8094cee141537fddabd98338581",
        "40ee70d4197d2270d15f4a728db2414877a93da04999e39c68154dea0c7e1e90"),
    "siw-couples-k500.jsonl": (
        "bfb8d084080194309ec88492c6baa2b316abf17df27faaea02b8871abb81a9df",
        "443f6eb0644c0812684a9d11ae93be8f8c86a2152ad7563e51d6fe76cdaffc64"),
    "siw-couples-k2000.jsonl": (
        "c4a4289ed52032ce16aa4f052c0ea14c339c0dfe67434fe28c47afeae6b9fc33",
        "62eba7e59db4bc88a848c55d225d8cb61f7d79a8f7bd17fb3aed0affc407f96e"),
    "siw-couples-k10000.jsonl": (
        "045352fa9b43a1ecd9c59e782ee930fb06c777b4cafaf5a00ad6dc5ff33f4971",
        "1e6b3d962473fb4dcb28f52fdc6d0023ca0c0024d15f42484b6dffce0e0f6734"),
    "siw-test-episodes.jsonl": (
        "6141938863685185650466cce52c439d515179fce720bece2d6d62b42b8c2d35", None),
    "siw-test-episodes-manifest.json": (
        "8ec8dac8781a78788048353f657ee7be917bf05bf31938d28a8fb7f8fa643115",
        "32515dab681ca1102133650336a2e4f73dd660e38226f59c0538fbbc6205b94f"),
    "split-manifest-SIW-v1.json": (
        "551371db9085e5af4e7f2a58e4ce8800de144b5c00bc3a931f4208f1f14732a2",
        "0b28a93022c5d7ebd302d8f5f4ac6994da61ee1f8569f31ebadd40a4ff49c652"),
    "inventory-siw-dev.json": (
        None,  # timing.seconds_total non reproductible — exclu par design
        "5b0c51a9734d351d490a80865174406b696836accffd233dc96193d3a1b28254"),
}


def seal_of(path: Path) -> str:
    if path.name.startswith("siw-couples"):
        lines = path.read_text(encoding="utf-8").split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        return sha256_hex(lines)
    return json.loads(path.read_text(encoding="utf-8")).get("manifest_sha256", "")


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    out = Path(sys.argv[1])
    ok = 0
    n = 0
    for fname, (fb, fs) in REF.items():
        p = out / fname
        if not p.exists():
            print(f"{fname:34s} ABSENT ✗")
            n += 1
            continue
        b = hashlib.sha256(p.read_bytes()).hexdigest()
        line = f"{fname:34s} bytes={'OK' if fb is None else b == fb}"
        good = fb is None or b == fb
        if fs:
            s = seal_of(p)
            line += f" seal={'OK' if s == fs else 'DIFF'} ({s[:16]}…)"
            good = good and s == fs
        else:
            line += f" seal=n/a (bytes {b[:16]}…)"
        print(line, "✓" if good else "✗")
        ok += good
        n += 1
    print(f"==> {ok}/{n} vérifications identiques")
    return 0 if ok == n else 1


if __name__ == "__main__":
    sys.exit(main())
