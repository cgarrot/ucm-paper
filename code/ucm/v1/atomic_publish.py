"""Publication atomique V1-bis (DEV-only) — implémentation du design v9 (`dccec02`, audité).

Primitives :
- `capability_probe` : sonde comportementale same-FS (link/EEXIST/st_dev) exécutée dans le
  répertoire cible AVANT publication ; échec ⇒ abort fail-closed ;
- `publish_no_replace` : tmp O_EXCL → write/fsync → hardlink no-replace (EEXIST ⇒ abort) →
  fsync parent → unlink → fsync parent ;
- `BundleWriter` : bundle immuable (membres fsyncés ; `manifest.json` fsyncé IN-BUNDLE avec
  `bundle_path` absolu AVANT publish) ; pointeur = HARDLINK du manifest (no-replace) ;
  N1 : le bundle ne se déplace JAMAIS après publication ;
- `verify_pointer` / `recover_pointer` : états A / A′ / B / C / D, fail-closed, jamais d'écrasement ;
- `FramedJournal` : journal framé mono-writer (`flock` + pid + start-time) ; queue invalide
  REFUSÉE sans troncature automatique ; stale-lock recovery avec preuve de mort.

Portée des garanties (design v9) : writers respectant l'API et les permissions FS ; mutations
hors protocole = hors garantie ; consommateurs re-hash/parse fail-closed. N2 : le journal n'est
jamais un signal de publication — seul le pointeur fait autorité.

Ne touche ni `runner_confirm.py` ni aucun scellé. Stdlib uniquement. DEV-only, phases 2-3 encadrées.

Seams de test (`_fsync`, `_link`, `_open`, `_unlink`) : même sémantique que `os.*`, remplacés
uniquement par le harness failpoints (DEV) — aucun effet en production.
"""
from __future__ import annotations

import fcntl
import glob as _glob
import hashlib
import json
import os
import secrets
import subprocess
import time
from typing import Callable, Optional

__all__ = [
    "PublishError", "CollisionError", "CapabilityError", "VerificationError",
    "RecoveryError", "LockBusy", "JournalTailError",
    "capability_probe", "publish_no_replace", "BundleWriter",
    "read_pointer", "verify_pointer", "recover_pointer",
    "process_start_time", "death_proof", "FramedJournal",
]

MANIFEST_NAME = "manifest.json"
BUNDLE_SCHEMA = "ucm-atomic-bundle/0.1"


class PublishError(Exception):
    """Erreur de publication (fail-closed)."""


class CollisionError(PublishError):
    """La destination existe déjà — jamais d'écrasement."""


class CapabilityError(PublishError):
    """La sonde de capacité same-FS a échoué — FS non supporté."""


class VerificationError(PublishError):
    """Re-hash/parse d'un artefact publié en échec."""


class RecoveryError(PublishError):
    """Recovery impossible/ambiguë — inspection manuelle requise."""


class LockBusy(PublishError):
    """Lock mono-writer déjà détenu."""


class JournalTailError(PublishError):
    """Queue de journal invalide/partielle — REFUSÉE (jamais de troncature auto)."""


# --- seams de test (identiques aux os.* ; patchés uniquement par le harness) ---------------
_fsync = os.fsync
_link = os.link
_open = os.open
_unlink = os.unlink


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        n = os.write(fd, view)
        view = view[n:]


def _fsync_dir(path: str) -> None:
    """fsync d'un répertoire (durabilité des entrées)."""
    fd = _open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        _fsync(fd)
    finally:
        os.close(fd)


def _tmp_name(final_path: str) -> str:
    d = os.path.dirname(os.path.abspath(final_path)) or "."
    return os.path.join(d, f".{os.path.basename(final_path)}.tmp.{os.getpid()}.{secrets.token_hex(4)}")


def _cleanup_tmp(tmp: str) -> None:
    """Nettoyage best-effort d'un tmp (jamais d'exception avalée silencieusement côté appelant)."""
    try:
        _unlink(tmp)
    except FileNotFoundError:
        return
    except OSError:
        return
    try:
        _fsync_dir(os.path.dirname(tmp) or ".")
    except OSError:
        pass


def capability_probe(target_dir: str) -> None:
    """Sonde comportementale same-FS — exécutée DANS le répertoire/FS cible de publication.

    Vérifie : hardlink possible (même répertoire), inode+device identiques, no-replace
    effectif (2ᵉ link ⇒ EEXIST). Nettoyage (unlink + fsync parent). Échec ⇒ CapabilityError.
    """
    d = os.path.abspath(target_dir)
    if not os.path.isdir(d):
        raise CapabilityError(f"probe: {d} n'est pas un répertoire")
    a = os.path.join(d, f".probe.{os.getpid()}.{secrets.token_hex(4)}.a")
    b = a + ".b"
    fd = _open(a, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(fd)
    try:
        try:
            _link(a, b)
        except OSError as e:
            raise CapabilityError(f"probe: hardlink indisponible ({e})") from e
        sa, sb = os.stat(a), os.stat(b)
        if (sa.st_ino, sa.st_dev) != (sb.st_ino, sb.st_dev):
            raise CapabilityError("probe: link → inode/device différents")
        try:
            _link(a, b)
        except FileExistsError:
            pass  # no-replace effectif ✅
        else:
            raise CapabilityError("probe: no-replace absent (2ᵉ link accepté)")
    finally:
        for p in (b, a):
            try:
                _unlink(p)
            except OSError:
                pass
        try:
            _fsync_dir(d)
        except OSError:
            pass


def publish_no_replace(final_path: str, data: Optional[bytes] = None, *,
                       writer: Optional[Callable[[int], None]] = None) -> None:
    """Publication atomique no-replace d'UN fichier (design v9 §1).

    tmp O_EXCL → write/fsync → `link(tmp, final)` (EEXIST ⇒ CollisionError) → fsync parent →
    unlink tmp → fsync parent. Un crash avant le link ne laisse AUCUN final ; après le link,
    le final est complet (écrit+fsyncé) ; jamais d'écrasement.
    """
    if (data is None) == (writer is None):
        raise PublishError("publish_no_replace: fournir data OU writer (exactement un)")
    directory = os.path.dirname(os.path.abspath(final_path)) or "."
    tmp = _tmp_name(final_path)
    fd = _open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        if data is not None:
            _write_all(fd, data)
        else:
            writer(fd)  # type: ignore[misc]
        _fsync(fd)
    except BaseException:
        os.close(fd)
        _cleanup_tmp(tmp)
        raise
    os.close(fd)
    try:
        _link(tmp, final_path)
    except FileExistsError:
        _cleanup_tmp(tmp)
        raise CollisionError(f"{final_path} ALREADY EXISTS — no-replace, aucun écrasement")
    except OSError:
        _cleanup_tmp(tmp)
        raise
    _fsync_dir(directory)
    _unlink(tmp)
    _fsync_dir(directory)


class BundleWriter:
    """Bundle immuable + manifest in-bundle + pointeur hardlink (design v9 §2/§3, N1).

    Séquence : membres (fsync chacun) → `manifest.json` (bundle_path absolu, fsync) →
    fsync(bundle) → fsync(parent) → pointeur = hardlink du manifest (no-replace) → fsync(parent).
    Aucune écriture après `finalize()` ; le bundle ne se déplace jamais après publication (N1).
    """

    def __init__(self, parent_dir: str, base_name: str):
        parent = os.path.abspath(parent_dir)
        if not os.path.isdir(parent):
            raise PublishError(f"BundleWriter: {parent} inexistant")
        self.dir = os.path.join(
            parent, f".{base_name}.bundle.{os.getpid()}.{secrets.token_hex(4)}")
        os.makedirs(self.dir, exist_ok=False)
        os.chmod(self.dir, 0o700)
        self._members: list[dict] = []
        self._finalized = False
        self._manifest: Optional[dict] = None
        self._manifest_path = os.path.join(self.dir, MANIFEST_NAME)

    # -- membres ---------------------------------------------------------------------------
    def _add_member(self, name: str, writer: Callable[[int], None]) -> None:
        if self._finalized:
            raise PublishError("bundle finalisé — aucune écriture après le manifest")
        if name == MANIFEST_NAME or "/" in name or name.startswith("."):
            raise PublishError(f"nom de membre invalide: {name!r}")
        path = os.path.join(self.dir, name)
        fd = _open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            writer(fd)
            _fsync(fd)
        finally:
            os.close(fd)
        self._members.append({"name": name, "size": os.path.getsize(path),
                              "sha256": _sha256_file(path)})

    def add_bytes(self, name: str, data: bytes) -> None:
        self._add_member(name, lambda fd: _write_all(fd, data))

    def add_writer(self, name: str, writer: Callable[[int], None]) -> None:
        self._add_member(name, writer)

    # -- finalisation / publication ----------------------------------------------------------
    def finalize(self) -> dict:
        if self._finalized:
            return self._manifest  # type: ignore[return-value]
        manifest = {
            "schema": BUNDLE_SCHEMA,
            "bundle_path": os.path.abspath(self.dir),  # N3 : localisation des membres
            "members": sorted(self._members, key=lambda m: m["name"]),
        }
        data = json.dumps(manifest, sort_keys=True, indent=2).encode()
        fd = _open(self._manifest_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            _write_all(fd, data)
            _fsync(fd)
        finally:
            os.close(fd)
        _fsync_dir(self.dir)
        _fsync_dir(os.path.dirname(self.dir) or ".")
        self._finalized = True
        self._manifest = manifest
        return manifest

    def publish_pointer(self, pointer_path: str) -> None:
        """Pointeur = HARDLINK du manifest in-bundle (no-replace) — SEUL signal de publication."""
        if not self._finalized:
            raise PublishError("publish_pointer: bundle non finalisé")
        try:
            _link(self._manifest_path, pointer_path)
        except FileExistsError:
            raise CollisionError(f"{pointer_path} ALREADY EXISTS — pointeur no-replace")
        _fsync_dir(os.path.dirname(os.path.abspath(pointer_path)) or ".")

    # -- vérification ------------------------------------------------------------------------
    def verify(self) -> None:
        if not self._finalized:
            raise PublishError("verify: bundle non finalisé")
        verify_bundle(self._manifest)  # type: ignore[arg-type]


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_pointer(pointer_path: str) -> dict:
    """Lit le manifest via le pointeur (= manifest, hardlink). Fail-closed si illisible/invalide."""
    try:
        with open(pointer_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    except FileNotFoundError:
        raise RecoveryError(f"pointeur absent: {pointer_path}")
    except (OSError, json.JSONDecodeError) as e:
        raise VerificationError(f"pointeur illisible/invalide: {pointer_path} ({e})") from e
    if manifest.get("schema") != BUNDLE_SCHEMA or "bundle_path" not in manifest:
        raise VerificationError(f"manifest invalide: {pointer_path}")
    return manifest


def verify_bundle(manifest: dict, base_dir: Optional[str] = None) -> None:
    """Re-hash de TOUS les membres vs manifest (consommateur fail-closed)."""
    bundle = base_dir or manifest.get("bundle_path")
    if not bundle or not os.path.isdir(bundle):
        raise VerificationError(f"bundle_path absent/introuvable: {bundle!r}")
    for m in manifest.get("members", []):
        p = os.path.join(bundle, m["name"])
        if not os.path.isfile(p):
            raise VerificationError(f"membre manquant: {m['name']}")
        if os.path.getsize(p) != m["size"] or _sha256_file(p) != m["sha256"]:
            raise VerificationError(f"membre altéré: {m['name']}")


def verify_pointer(pointer_path: str) -> dict:
    manifest = read_pointer(pointer_path)
    verify_bundle(manifest)
    return manifest


def recover_pointer(pointer_path: str, bundle_glob: str) -> dict:
    """Recovery fail-closed des états A / A′ / B / C / D (design v9 §3). Jamais d'écrasement.

    A  (bundle seul, pas de pointeur) : un seul bundle valide ⇒ publie le pointeur (idempotent) ;
    A′ (plusieurs bundles valides)    : ABORT (ambiguïté — jamais « le plus récent ») ;
    B  (pointeur + bundle)            : vérifie les hashes, retourne publié ;
    C  (pointeur sans bundle)         : ABORT (incohérent — inode survit sans bundle_path) ;
    D  (résidus/tmp)                  : listés, JAMAIS supprimés automatiquement.
    """
    if os.path.exists(pointer_path):
        try:
            manifest = verify_pointer(pointer_path)
        except PublishError as e:
            raise RecoveryError(
                f"C/incohérent: pointeur présent mais bundle inexploitable ({e})") from e
        return {"status": "published", "pointer": pointer_path,
                "bundle": manifest["bundle_path"]}
    candidates = sorted(_glob.glob(bundle_glob))
    valid: list[tuple[str, dict]] = []
    invalid: list[str] = []
    for c in candidates:
        mp = os.path.join(c, MANIFEST_NAME)
        try:
            with open(mp, "r", encoding="utf-8") as fh:
                manifest = json.load(fh)
            verify_bundle(manifest, base_dir=c)
            valid.append((c, manifest))
        except (OSError, json.JSONDecodeError, VerificationError):
            invalid.append(c)
    if len(valid) == 1:
        c, _manifest = valid[0]
        try:
            _link(os.path.join(c, MANIFEST_NAME), pointer_path)
        except FileExistsError:
            raise RecoveryError("course pendant recovery — pointeur apparu (inspection)")
        _fsync_dir(os.path.dirname(os.path.abspath(pointer_path)) or ".")
        return {"status": "recovered_published", "pointer": pointer_path, "bundle": c,
                "residues": invalid}
    if len(valid) > 1:
        raise RecoveryError(f"A′ ambigu: {len(valid)} bundles valides — jamais « le plus récent »: {[c for c, _ in valid]}")
    raise RecoveryError(f"aucun bundle valide (résidus D, non supprimés): {invalid}")


# --- journal framé mono-writer --------------------------------------------------------------

def process_start_time(pid: int) -> Optional[str]:
    """Start-time lisible d'un pid (via `ps -o lstart=`), None si indéterminable."""
    try:
        out = subprocess.run(["ps", "-o", "lstart=", "-p", str(pid)],
                             capture_output=True, text=True, timeout=5)
        if out.returncode != 0:
            return None
        s = out.stdout.strip()
        return s or None
    except Exception:
        return None


def death_proof(info: dict) -> bool:
    """Preuve de mort du détenteur (design v9 §4) — fail-closed.

    Vraie si : pid non vivant (`kill(pid,0) → ESRCH`) OU (pid vivant mais start-time
    enregistré ≠ courant ⇒ pid réutilisé par un autre process). Sinon ⇒ False (aucune preuve).
    """
    try:
        pid = int(info["pid"])
    except (KeyError, TypeError, ValueError):
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        pass  # vivant (autre utilisateur)
    except OSError:
        return False
    current = process_start_time(pid)
    recorded = info.get("start_time")
    if not current or not recorded:
        return False  # aucune preuve ⇒ jamais de rupture auto
    return current != recorded


class FramedJournal:
    """Journal append-only framé, mono-writer (`flock` + pid + start-time).

    Frame : `"<len> <sha256_hex> <json>"` + `\\n`, fsync par entrée. La queue partielle/invalide
    est REFUSÉE par `read_all(strict=True)` — jamais de troncature automatique, jamais utilisée
    pour décider. N2 : le journal n'est PAS un signal de publication.
    """

    def __init__(self, path: str):
        self.path = path
        self.lock_path = path + ".lock"
        self._jfd: Optional[int] = None
        self._lfd: Optional[int] = None

    # -- lock --------------------------------------------------------------------------------
    def lock_info(self) -> Optional[dict]:
        try:
            with open(self.lock_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError):
            return None

    def acquire(self) -> None:
        lfd = _open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(lfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            info = self.lock_info()
            os.close(lfd)
            raise LockBusy(f"lock détenu: {info}")
        info = {"pid": os.getpid(), "start_time": process_start_time(os.getpid()),
                "t": round(time.time(), 3)}
        os.ftruncate(lfd, 0)
        os.lseek(lfd, 0, 0)
        _write_all(lfd, json.dumps(info, sort_keys=True).encode())
        _fsync(lfd)
        self._lfd = lfd
        self._jfd = _open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)

    def recover_stale_lock(self, confirm: bool = False) -> bool:
        """Rupture de lock EXPLICITE — exige confirmation ET preuve de mort. Sinon abort."""
        if not confirm:
            raise RecoveryError("recover_stale_lock: confirmation explicite requise")
        info = self.lock_info()
        if not info:
            raise RecoveryError("recover_stale_lock: aucun info de lock — preuve impossible")
        if not death_proof(info):
            raise RecoveryError(f"recover_stale_lock: pas de preuve de mort ({info}) — abort")
        _unlink(self.lock_path)
        _fsync_dir(os.path.dirname(os.path.abspath(self.lock_path)) or ".")
        return True

    # -- écriture / lecture -------------------------------------------------------------------
    def append(self, event: dict) -> None:
        if self._jfd is None:
            raise PublishError("append: journal non acquis (acquire() requis)")
        payload = json.dumps(event, sort_keys=True, separators=(",", ":"))
        raw = payload.encode()
        frame = f"{len(raw)} {hashlib.sha256(raw).hexdigest()} {payload}\n"
        _write_all(self._jfd, frame.encode())
        _fsync(self._jfd)

    def read_all(self, strict: bool = True) -> tuple[list[dict], str]:
        """Retourne (entrées, statut_queue). strict=True : queue invalide ⇒ JournalTailError."""
        if not os.path.exists(self.path):
            return [], "ok"
        with open(self.path, "r", encoding="utf-8") as fh:
            text = fh.read()
        entries: list[dict] = []
        lines = text.split("\n")
        if lines and lines[-1] == "":
            lines.pop()  # fin de fichier propre
        invalid_reason: Optional[str] = None
        for idx, line in enumerate(lines):
            parts = line.split(" ", 2)
            if len(parts) != 3:
                invalid_reason = f"ligne {idx}: format invalide"
                break
            ln_s, sh, payload = parts
            raw = payload.encode()
            try:
                ln = int(ln_s)
            except ValueError:
                invalid_reason = f"ligne {idx}: longueur non entière"
                break
            if ln != len(raw):
                invalid_reason = f"ligne {idx}: longueur {ln} ≠ {len(raw)}"
                break
            if hashlib.sha256(raw).hexdigest() != sh:
                invalid_reason = f"ligne {idx}: sha256 mismatch"
                break
            try:
                entries.append(json.loads(payload))
            except json.JSONDecodeError:
                invalid_reason = f"ligne {idx}: json invalide"
                break
        if invalid_reason is not None:
            if strict:
                raise JournalTailError(f"queue invalide REFUSÉE (aucune troncature auto): {invalid_reason}")
            return entries, f"invalid:{invalid_reason}"
        return entries, "ok"

    def release(self) -> None:
        if self._jfd is not None:
            os.close(self._jfd)
            self._jfd = None
        if self._lfd is not None:
            try:
                fcntl.flock(self._lfd, fcntl.LOCK_UN)
            finally:
                os.close(self._lfd)
                self._lfd = None
