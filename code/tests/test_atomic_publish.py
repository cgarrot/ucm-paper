"""Phase 2 — harness failpoints DEV-only pour `ucm/v1/atomic_publish.py` (design v9, `dccec02`).

Assertions exigées par la mission : aucun final partiel ; EEXIST ⇒ abort (jamais d'écrasement) ;
recovery idempotent ; queue invalide REFUSÉE. Injections ≤5/writer via les seams `_fsync`/`_link`.
Tout vit sous `tmp_path` pytest (DEV) — aucun artefact historique/scellé touché.
Phase 3 (kill -9) NON incluse.
"""

import hashlib
import json
import os
import subprocess
import sys

import pytest

from ucm.v1 import atomic_publish as ap


# --------------------------------------------------------------------------------------------
# helpers de failpoint
# --------------------------------------------------------------------------------------------
class CountedFail:
    """Remplace un seam : appelle l'original sauf au n-ième appel (raise OSError)."""

    def __init__(self, original, fail_at: int, exc: Exception):
        self._original = original
        self._fail_at = fail_at
        self._exc = exc
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        if self.calls == self._fail_at:
            raise self._exc
        return self._original(*args, **kwargs)


def _residues(directory: str, name: str) -> list[str]:
    return [p for p in os.listdir(directory) if f".{name}.tmp." in p]


# --------------------------------------------------------------------------------------------
# sonde de capacité
# --------------------------------------------------------------------------------------------
def test_capability_probe_ok(tmp_path):
    ap.capability_probe(str(tmp_path))
    # sonde nettoyée (aucun résidu) et parent fsyncé sans erreur
    assert [p for p in os.listdir(tmp_path) if p.startswith(".probe.")] == []


def test_capability_probe_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(ap, "_link", lambda *a, **k: (_ for _ in ()).throw(OSError("EOPNOTSUPP")))
    with pytest.raises(ap.CapabilityError):
        ap.capability_probe(str(tmp_path))


# --------------------------------------------------------------------------------------------
# publish_no_replace — succès, collision, failpoints (3 injections)
# --------------------------------------------------------------------------------------------
def test_publish_ok(tmp_path):
    final = str(tmp_path / "out.json")
    ap.publish_no_replace(final, data=b'{"a": 1}')
    assert open(final, "rb").read() == b'{"a": 1}'
    assert _residues(str(tmp_path), "out.json") == []


def test_publish_collision_no_overwrite(tmp_path):
    final = str(tmp_path / "out.json")
    with open(final, "wb") as fh:
        fh.write(b"PREEXISTING")
    with pytest.raises(ap.CollisionError):
        ap.publish_no_replace(final, data=b"NEW")
    assert open(final, "rb").read() == b"PREEXISTING"
    assert _residues(str(tmp_path), "out.json") == []


def test_failpoint_after_write_before_fsync(tmp_path, monkeypatch):
    final = str(tmp_path / "out.bin")
    monkeypatch.setattr(ap, "_fsync", CountedFail(os.fsync, fail_at=1, exc=OSError("KILL")))
    with pytest.raises(OSError):
        ap.publish_no_replace(final, data=b"x" * 100)
    assert not os.path.exists(final)           # aucun final partiel
    # re-run idempotent (seams restaurés par monkeypatch)
    monkeypatch.undo()
    ap.publish_no_replace(final, data=b"x" * 100)
    assert os.path.getsize(final) == 100


def test_failpoint_after_fsync_before_link(tmp_path, monkeypatch):
    final = str(tmp_path / "out.bin")
    monkeypatch.setattr(ap, "_link", CountedFail(os.link, fail_at=1, exc=OSError("ENOSPC")))
    with pytest.raises(OSError):
        ap.publish_no_replace(final, data=b"data")
    assert not os.path.exists(final)
    monkeypatch.undo()
    ap.publish_no_replace(final, data=b"data")
    assert open(final, "rb").read() == b"data"


def test_failpoint_after_link_before_fsync_parent(tmp_path, monkeypatch):
    final = str(tmp_path / "out.bin")
    payload = b"COMPLETE-PAYLOAD"
    # _fsync appels : 1 = fichier, 2 = fsync dir après link → on casse le 2ᵉ
    monkeypatch.setattr(ap, "_fsync", CountedFail(os.fsync, fail_at=2, exc=OSError("EIO")))
    with pytest.raises(OSError):
        ap.publish_no_replace(final, data=payload)
    # le final existe et est COMPLET (écrit+fsyncé avant le link) — aucun partiel
    assert os.path.exists(final)
    assert open(final, "rb").read() == payload
    monkeypatch.undo()
    with pytest.raises(ap.CollisionError):     # re-run : jamais d'écrasement
        ap.publish_no_replace(final, data=b"OTHER")


# --------------------------------------------------------------------------------------------
# bundle : immuabilité, manifest, pointeur hardlink, collision, tamper
# --------------------------------------------------------------------------------------------
def _make_bundle(tmp_path) -> ap.BundleWriter:
    bw = ap.BundleWriter(str(tmp_path), "run")
    bw.add_bytes("raw.jsonl", b'{"r":1}\n{"r":2}\n')
    bw.add_writer("report.json", lambda fd: os.write(fd, b'{"ok":true}'))
    bw.finalize()
    return bw


def test_bundle_publish_pointer_hardlink_and_verify(tmp_path):
    bw = _make_bundle(tmp_path)
    pointer = str(tmp_path / "run.pointer")
    bw.publish_pointer(pointer)
    manifest = ap.verify_pointer(pointer)
    assert manifest["bundle_path"] == os.path.abspath(bw.dir)      # N3 : localisation
    # pointeur = même inode que le manifest in-bundle (hardlink)
    assert os.stat(pointer).st_ino == os.stat(os.path.join(bw.dir, "manifest.json")).st_ino
    with pytest.raises(ap.CollisionError):                          # no-replace
        bw.publish_pointer(pointer)


def test_bundle_no_write_after_manifest(tmp_path):
    bw = _make_bundle(tmp_path)
    with pytest.raises(ap.PublishError):
        bw.add_bytes("late.txt", b"nope")


def test_bundle_verify_detects_tamper(tmp_path):
    bw = _make_bundle(tmp_path)
    pointer = str(tmp_path / "run.pointer")
    bw.publish_pointer(pointer)
    with open(os.path.join(bw.dir, "raw.jsonl"), "wb") as fh:
        fh.write(b"TAMPERED")
    with pytest.raises(ap.VerificationError):
        ap.verify_pointer(pointer)


# --------------------------------------------------------------------------------------------
# recovery : A (publie), A′ (abort), C (abort)
# --------------------------------------------------------------------------------------------
def test_recover_state_A_single_bundle_publishes(tmp_path):
    bw = _make_bundle(tmp_path)                     # bundle sans pointeur
    pointer = str(tmp_path / "run.pointer")
    res = ap.recover_pointer(pointer, str(tmp_path / ".*.bundle.*"))
    assert res["status"] == "recovered_published"
    assert os.path.exists(pointer)
    ap.verify_pointer(pointer)                       # idempotent/vérifiable


def test_recover_state_A_prime_two_bundles_aborts(tmp_path):
    _make_bundle(tmp_path)
    _make_bundle(tmp_path)                           # 2 bundles valides
    pointer = str(tmp_path / "run.pointer")
    with pytest.raises(ap.RecoveryError) as e:
        ap.recover_pointer(pointer, str(tmp_path / ".*.bundle.*"))
    assert "A′" in str(e.value) or "ambigu" in str(e.value)
    assert not os.path.exists(pointer)               # jamais de choix arbitraire


def test_recover_state_C_pointer_without_bundle_aborts(tmp_path):
    bw = _make_bundle(tmp_path)
    pointer = str(tmp_path / "run.pointer")
    bw.publish_pointer(pointer)
    os.rename(bw.dir, bw.dir + "-moved-hors-protocole")   # bundle disparu (hors protocole)
    with pytest.raises(ap.RecoveryError):
        ap.recover_pointer(pointer, str(tmp_path / ".*.bundle.*"))


# --------------------------------------------------------------------------------------------
# journal : frames, queue invalide refusée, lock mono-writer, stale-lock
# --------------------------------------------------------------------------------------------
def test_journal_append_read_roundtrip(tmp_path):
    j = ap.FramedJournal(str(tmp_path / "j.log"))
    j.acquire()
    try:
        j.append({"i": 0, "motive": "intent"})
        j.append({"i": 1, "motive": "read"})
    finally:
        j.release()
    entries, tail = j.read_all()
    assert [e["i"] for e in entries] == [0, 1] and tail == "ok"


def test_journal_invalid_tail_refused_no_auto_truncate(tmp_path):
    p = str(tmp_path / "j.log")
    j = ap.FramedJournal(p)
    j.acquire()
    try:
        j.append({"i": 0})
    finally:
        j.release()
    raw = open(p, "rb").read()
    with open(p, "wb") as fh:
        fh.write(raw[:-8])                            # tronque le dernier frame (mi-write)
    size_before = os.path.getsize(p)
    with pytest.raises(ap.JournalTailError):
        j.read_all(strict=True)
    entries, tail = j.read_all(strict=False)          # affichage seulement, jamais décision
    assert tail.startswith("invalid") and len(entries) == 0
    assert os.path.getsize(p) == size_before          # AUCUNE troncature auto


def test_journal_lock_mono_writer_busy(tmp_path):
    p = str(tmp_path / "j.log")
    j1 = ap.FramedJournal(p)
    j1.acquire()
    try:
        j2 = ap.FramedJournal(p)
        with pytest.raises(ap.LockBusy):
            j2.acquire()
    finally:
        j1.release()


def test_stale_lock_recovery_requires_confirm_and_death_proof(tmp_path):
    p = str(tmp_path / "j.log")
    lock = p + ".lock"
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()                                        # pid mort
    with open(lock, "w") as fh:
        json.dump({"pid": dead.pid, "start_time": "gone"}, fh)
    j = ap.FramedJournal(p)
    with pytest.raises(ap.RecoveryError):              # sans confirmation ⇒ abort
        j.recover_stale_lock(confirm=False)
    assert j.recover_stale_lock(confirm=True) is True  # preuve de mort + confirm ⇒ rupture explicite
    assert not os.path.exists(lock)


def test_death_proof_alive_owner_is_not_proven_dead(tmp_path):
    st = ap.process_start_time(os.getpid())
    if st is None:
        pytest.skip("ps indisponible pour le start-time")
    assert ap.death_proof({"pid": os.getpid(), "start_time": st}) is False
    # pid réutilisé simulé : start-time enregistré ≠ courant ⇒ preuve de mort de l'ancien détenteur
    assert ap.death_proof({"pid": os.getpid(), "start_time": "WRONG"}) is True
    assert ap.death_proof({"pid": 2 ** 30 + 12345}) is True   # pid inexistant
