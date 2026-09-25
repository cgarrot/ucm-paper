#!/usr/bin/env python3
"""Phase 3 — kill -9 E2E borné (DEV tmp uniquement) pour `ucm/v1/atomic_publish.py`.

≤3 runs courts, un par writer : `publish_no_replace` / `BundleWriter+pointer` / `FramedJournal`.
Après chaque SIGKILL : inspection FS (existence/hash/taille/tmp résiduels), assertions :
aucun publié partiel ; collision ⇒ abort ; re-run idempotent (recovery A/A′/C) ; journal intact.

Usage :
  python scripts/phase3_kill_e2e.py --report reports/atomic-publish-phase3-kill-e2e.json
  (mode interne : --child <scenario> <workdir>)

DEV-only : tout vit sous `tempfile.mkdtemp` (TMPDIR système). Aucun scellé/seed/test2 touché.
Note d'honnêteté : SIGKILL prouve la résistance au crash PROCESSUS (atomicité des noms) ; ne
prouve PAS la durabilité power-loss (non testable localement — cf. design v9).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ucm.v1 import atomic_publish as ap  # noqa: E402

SCENARIOS = ("file", "bundle", "journal")


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------
# mode enfant : démarre la publication, signale la fenêtre, puis écrit lentement (cible du kill)
# ---------------------------------------------------------------------------------------------
def _child(scenario: str, workdir: str) -> int:
    if scenario == "file":
        final = os.path.join(workdir, "out.bin")

        def slow(fd: int) -> None:
            for _ in range(16):
                os.write(fd, b"X" * 262144)
                time.sleep(0.05)

        print("IN_WINDOW", flush=True)
        ap.publish_no_replace(final, writer=slow)
        print("DONE", flush=True)
    elif scenario == "bundle":
        bw = ap.BundleWriter(workdir, "run")

        def slow(fd: int) -> None:
            for _ in range(16):
                os.write(fd, b"Y" * 65536)
                time.sleep(0.05)

        print("IN_WINDOW", flush=True)
        bw.add_writer("report.bin", slow)
        bw.add_bytes("raw.jsonl", b'{"r":1}\n' * 10)
        bw.finalize()
        bw.publish_pointer(os.path.join(workdir, "run.pointer"))
        print("DONE", flush=True)
    elif scenario == "journal":
        j = ap.FramedJournal(os.path.join(workdir, "j.log"))
        j.acquire()
        print("IN_WINDOW", flush=True)
        for i in range(200):
            j.append({"i": i})
            time.sleep(0.03)
        j.release()
        print("DONE", flush=True)
    else:
        print(f"scénario inconnu: {scenario}", file=sys.stderr)
        return 2
    return 0


# ---------------------------------------------------------------------------------------------
# mode parent : 1 run par scénario (kill pendant la fenêtre), inspection + assertions
# ---------------------------------------------------------------------------------------------
def _run_scenario(scenario: str) -> dict:
    workdir = tempfile.mkdtemp(prefix=f"phase3-{scenario}-")
    cmd = [sys.executable, str(Path(__file__).resolve()), "--child", scenario, workdir]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    assert proc.stdout is not None
    marker = proc.stdout.readline().strip()
    if marker != "IN_WINDOW":
        raise RuntimeError(f"{scenario}: pas de marqueur IN_WINDOW (got {marker!r})")
    time.sleep(0.30)                       # kill dans la fenêtre (écriture lente en cours)
    os.kill(proc.pid, signal.SIGKILL)
    proc.wait()
    files = sorted(os.listdir(workdir))
    res: dict = {
        "scenario": scenario,
        "workdir": workdir,
        "injection": "SIGKILL (kill -9) à +0.30 s après le marqueur IN_WINDOW — écriture lente EN COURS (dans la fenêtre de publication)",
        "commands": [f"Popen({cmd!r})", f"readline → IN_WINDOW", "sleep 0.30", f"kill -9 {proc.pid}"],
        "files_after_kill": files,
        "assertions": {},
    }
    a = res["assertions"]

    if scenario == "file":
        final = os.path.join(workdir, "out.bin")
        residues = [f for f in files if ".out.bin.tmp." in f]
        res["tmp_residues"] = [
            {"name": f, "size": os.path.getsize(os.path.join(workdir, f))} for f in residues]
        a["aucun_final_partiel"] = not os.path.exists(final)
        # re-run idempotent (nouvelle tentative propre)
        ap.publish_no_replace(final, data=b"COMPLETE-PAYLOAD")
        res["final_sha256_after_rerun"] = _sha256_file(final)
        a["rerun_publie_complet"] = open(final, "rb").read() == b"COMPLETE-PAYLOAD"
        # collision ⇒ abort (jamais d'écrasement)
        try:
            ap.publish_no_replace(final, data=b"OTHER")
            a["collision_abort"] = False
        except ap.CollisionError:
            a["collision_abort"] = True
        res["residues_listes_non_supprimes"] = len(residues) >= 1  # état D : listés, jamais supprimés auto

    elif scenario == "bundle":
        pointer = os.path.join(workdir, "run.pointer")
        bundles = [d for d in files if ".run.bundle." in d]
        res["bundles"] = bundles
        res["manifest_present"] = any(
            os.path.exists(os.path.join(workdir, b, "manifest.json")) for b in bundles)
        a["aucun_pointeur_publie"] = not os.path.exists(pointer)
        # recovery : fail-closed (bundle incomplet ⇒ RecoveryError, résidus listés) — jamais silencieux
        try:
            out = ap.recover_pointer(pointer, os.path.join(workdir, ".*.bundle.*"))
            a["recovery_fail_closed_ou_publie"] = bool(out)
        except ap.RecoveryError:
            a["recovery_fail_closed_ou_publie"] = True
        # re-run propre : nouveau bundle + pointeur + vérification
        bw = ap.BundleWriter(workdir, "run2")
        bw.add_bytes("raw.jsonl", b'{"ok":1}\n')
        bw.finalize()
        p2 = os.path.join(workdir, "run2.pointer")
        bw.publish_pointer(p2)
        ap.verify_pointer(p2)
        a["rerun_bundle_publie_et_verifie"] = True
        res["run2_pointer_sha256"] = _sha256_file(p2)

    elif scenario == "journal":
        j = ap.FramedJournal(os.path.join(workdir, "j.log"))
        try:
            entries, tail = j.read_all(strict=True)   # ne doit PAS lever : frames complets seulement
            a["journal_intact"] = tail == "ok"
        except ap.JournalTailError:
            entries, tail, a["journal_intact"] = [], "invalid", False
        res["frames_after_kill"] = len(entries)
        info = j.lock_info()
        res["lock_info"] = info
        a["lock_stale_prouve"] = bool(info) and ap.death_proof(info)
        try:
            j.recover_stale_lock(confirm=False)
            a["stale_exige_confirmation"] = False
        except ap.RecoveryError:
            a["stale_exige_confirmation"] = True
        j.recover_stale_lock(confirm=True)
        a["stale_recover_avec_preuve"] = True
        # re-run : ré-acquisition + append + relecture stricte
        j2 = ap.FramedJournal(os.path.join(workdir, "j.log"))
        j2.acquire()
        j2.append({"i": "post-kill"})
        j2.release()
        e2, t2 = j.read_all(strict=True)
        a["journal_valide_apres_recovery"] = t2 == "ok" and len(e2) == len(entries) + 1
        res["frames_after_rerun"] = len(e2)

    res["pass"] = all(a.values())
    return res


def main() -> int:
    ap_ = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--child", nargs=2, metavar=("SCENARIO", "WORKDIR"))
    ap_.add_argument("--report", default=None, help="chemin du rapport JSON (nouveau chemin)")
    args = ap_.parse_args()
    if args.child:
        return _child(args.child[0], args.child[1])
    runs = [_run_scenario(s) for s in SCENARIOS]
    report = {
        "kind": "phase3-kill-e2e",
        "design": "docs/DESIGN-ATOMIC-PUBLISH-V1bis.md (v9, commit dccec02)",
        "module": "ucm/v1/atomic_publish.py",
        "runs": runs,
        "pass_all": all(r["pass"] for r in runs),
        "limits": "SIGKILL = crash PROCESSUS (atomicité des noms) ; durabilité power-loss non "
                  "testable localement (discipline fsync par construction, non mesurée) ; kill "
                  "entre appends journal ⇒ frames complets ; queue partielle couverte en phase 2.",
    }
    for r in runs:
        print(f"[{r['scenario']:7s}] pass={r['pass']} " +
              " ".join(f"{k}={v}" for k, v in r["assertions"].items()))
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"rapport → {args.report}")
    return 0 if report["pass_all"] else 1


if __name__ == "__main__":
    sys.exit(main())
