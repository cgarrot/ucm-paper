# DESIGN — Publication atomique V1-bis (DEV-only, chemin opt-in) — v2 (revue lead 13:14)

- **Statut** : note de design, **non normative** avant revue tagi-5 + commit propre. **Ne modifie pas** le chemin historique `ucm/v1/runner_confirm.py`. **N'autorise rien** : HOLD test2/seeds, aucun seal.
- **Contexte** : findings Phase 1 (commit pinné `534545f`) — W1 freeze manifest FAIL (pas de fsync fichier/dir ; partiel + O_EXCL = voie empoisonnée), W2 outputs FAIL power-loss (npz/report non fsync ; pas de fsync dir après rename) + **collision sur dossier vide** (POSIX `rename` remplace un dir vide), W3 journal non atomique (truncate in-place).
- **Objectif** : primitives pour le chemin V1-bis opt-in — (1) jamais de published partiel/corrompu ; (2) **jamais d'écrasement silencieux via l'API de publication (no-replace)** — garanties pour des writers **respectant l'API et les permissions FS** ; **mutations hors protocole = hors garantie** ; consommateurs **re-hash/parse fail-closed** ; (3) durabilité power-loss par discipline fsync fichier+dir ; (4) recovery idempotent, fail-closed ; (5) régimes de crash **distingués** (process kill ≠ power loss).
- **Historique des révisions** : v1 (13:13) — v2 (13:14, revue lead) — v3 (13:16, revue tagi-5 P1–P6) — v4 (13:16, arbitrage lead) — v5 (13:18, micro-notes tagi-5) — v6 (13:19, revue lead) — v7 (13:20, revue lead ; PASS final tagi-5 `fc10e1c3…`) — v8 (13:21, micro-notes tagi-5 ; PASS pré-audit `31ee2e62…`) — **v9 (13:22, passe finale lead) : N1 bundle jamais déplacé après publish ; N2 journal ≠ publication (pointeur seul autorité) ; N3 état C = `bundle_path` absent ; sonde de capacité dans le répertoire/FS cible + nettoyage (unlink + fsync parent) ; preuve de mort (pid+start-time).**

## 1. Primitive `publish_no_replace(final_path, writer)`

1. `tmp = dir/.{name}.tmp.{pid}.{rand}` créé **O_EXCL** (même répertoire/FS que `final`).
2. `writer(fd)` → `flush` → `os.fsync(fd)` → `close`.
3. **Publication no-replace atomique** : `os.link(tmp, final)` — `EEXIST ⇒ abort explicite` (jamais d'écrasement). *(macOS/APFS : hardlink same-FS OK ; `renameat2(RENAME_NOREPLACE)` non portable.)*
4. `os.fsync(parent_dir)` — durabilité de la nouvelle entrée.
5. `os.unlink(tmp)`.
6. **`os.fsync(parent_dir)`** — durabilité du nettoyage (revue lead).

## 2. Bundle de sortie — **bundle immuable unique + manifest-pointeur**

> Le sentinel O_EXCL + `if !exists(final)` était **non conforme** (course check→rename **entre writers, y compris coopérants** ; `rename` remplace un dir vide). Refonte :

1. `bundle = parent/{out}.bundle.{pid}.{rand}` — `makedirs(exist_ok=False)`, 0700, **même FS** que le pointeur (sinon abort).
2. Écrire **TOUS** les membres : raw jsonl, **chaque** `.npz`, report — `flush`+`fsync` par fichier.
3. **`manifest.json` écrit DANS le bundle** : liste exhaustive (membres, tailles, sha256) + métadonnées **+ `bundle_path`** (chemin absolu du bundle — les consommateurs ne lisent que le pointeur, qui est le manifest : il doit permettre de **localiser les membres**) ; `flush`+`fsync`. *(Persisté AVANT publication : la recovery ne dépend JAMAIS d'un manifest en mémoire.)*
4. `os.fsync(bundle_dir)` — bundle durable.
5. `os.fsync(parent)` — nom du bundle durable.
6. **Publication = `publish_no_replace({out}.pointer, manifest_canonique)`** — **le pointeur est le HARDLINK du `manifest.json` in-bundle** (`os.link(bundle/manifest.json, pointer)` : même inode ⇒ octets identiques **par construction**) ; `EEXIST ⇒ abort`. *(Variante copie autorisée : publier une copie et **tester l'égalité bytes pointeur ↔ manifest** — test P6a.)* Les consommateurs ne lisent QUE le pointeur (→ bundle).
7. `os.fsync(parent)` après le link du pointeur.
8. **N1 — le bundle ne se déplace JAMAIS après publication** : aucun rename/déplacement du répertoire bundle post-publish ; il reste adressé par le `bundle_path` du manifest (immuable par discipline).

**Propriétés** : aucun nom « final » en compétition (pas de course sur dir vide) ; `publish_no_replace` **ne remplace pas** l'entrée existante ; bundle **immuable par discipline**. **Portée des garanties** : elles valent pour des writers **respectant l'API** et les permissions FS — un même UID peut muter/`O_TRUNC` l'inode via le hardlink : **mutations hors protocole hors garantie** ; les **consommateurs re-hashent et parsent en fail-closed**.
**Validation post-publication** : re-hash des membres du bundle == `manifest.json` persisté ; échec ⇒ erreur explicite (jamais de réparation silencieuse).
**Variante B (documentée, non préférée)** : rename exclusif macOS validé (sonde comportementale) + fallback **fail-closed** (si la sémantique no-replace ne peut être prouvée → abort).

## 3. États de recovery (explicites, fail-closed, jamais d'écrasement)

Avec le design pointeur : `marker ≡ pointeur`.

| État | Signature FS | Statut | Action |
|---|---|---|---|
| A | bundle seul, **pas de pointeur** | **NON publié** (incomplet possible) | vérifier hashes bundle vs `manifest.json` persisté : OK ⇒ (re-)publier le pointeur *(idempotent)* ; mismatch ⇒ **abort explicite** |
| A′ | **plusieurs bundles** sans pointeur | ambigu | **abort (ambiguïté) — jamais choisir « le plus récent »** ; si **un seul** bundle vérifie ses hashes → publier ; si aucun → abort |
| B | **pointeur + bundle** | publié | vérifier hashes pointés ; **jamais d'écrasement** |
| C | pointeur sans bundle | **incohérent** | abort explicite, inspection manuelle — **N3 : `bundle_path` absent alors que l'inode du pointeur survit** (bundle supprimé/déplacé hors protocole) |
| D | tmp/résidus/sentinels legacy | indéterminé | listés, **non supprimés automatiquement** |

**Cas explicite (revue lead)** : `final` **sans sentinel ni marker** (crash entre suppression et écriture du marker — schéma legacy/rename) = **INCONNU, jamais « publié »** → inspection/vérification avant toute consommation.

## 4. Journal

- **Recommandation primaire (P3) : un fichier par événement** — nom = `{seq}.{hash}.json`, publié via §1 (hardlink no-replace) → **aucune ligne partielle possible** ; **mono-writer imposé** (lock).
- Si log en lignes : append-only **framé** (`"<len> <sha256_hex> <json>"`), `flush`+`fsync` par entrée ; **mono-writer imposé par lock** — c'est **le lock** qui garantit la sérialisation, **pas `O_APPEND`**. *(Qualification : l'assertion `PIPE_BUF` concerne pipes/FIFOs, PAS l'atomicité d'un fichier ordinaire en `O_APPEND` ; aucune atomicité inter-writers n'est supposée du FS.)*
- **Politique de queue fail-closed STRICTE (choix lead)** : le loader **refuse toute queue invalide/partielle ET toute décision** qui en dépendrait ; **aucune troncature automatique** ; **aucune option d'ignorance pour décision** (affichage read-only d'audit possible — jamais silencieux, jamais utilisé pour décider).
- **Mono-writer (mécanique, micro-note tagi-5)** : lock `{log}.lock` via **`flock(LOCK_EX|LOCK_NB)`** ; le lock contient **pid + start-time** ; **stale-lock recovery fail-closed** : jamais de rupture automatique sans preuve de mort du détenteur (vérif pid vivant) ; détenteur mort ⇒ recovery **explicite** (procédure documentée / option `--recover-stale-lock` avec confirmation) ; en cas de doute ⇒ abort.
- **« Preuve de mort » (définition)** : pid non vivant (`kill(pid, 0) → ESRCH`) **ET**, si le pid est réutilisé, start-time enregistré ≠ courant (optionnellement même session/boot) ; **toute incertitude ⇒ abort**.
- **N2 — journal ≠ publication** : le journal n'est **jamais** un signal de publication ; **seul le pointeur fait autorité** (un écrit journal ne vaut pas « publié »).
- Jamais de truncate in-place ; chaîne de scellés par entrée conservée.

## 5. Tests failpoints (DEV-only, tmp DEV uniquement)

Injections (≤5) : après write avant fsync ; après fsync avant link du pointeur ; après link avant fsync parent ; **kill mi-append journal** ; bundle complet sans pointeur (état A).
**Tests P6 (tagi-5)** : (a) pointeur ↔ manifest — comparer **bytes + inode** (inode identique en hardlink ; bytes égaux en variante copie) ; (b) **état A′ avec 2 bundles** → **abort** (jamais « le plus récent »).
Assertions : aucun publié partiel ; `EEXIST ⇒ abort` ; re-run idempotent (état A → publication) ; journal jamais tronqué, queue invalide refusée ; hashes pointés == manifest persisté.
**Régimes distingués** : `SIGKILL` prouve l'atomicité des **noms** (process kill) ; **power loss non testable localement** → durabilité **par construction** (fsync fichier+dir), déclarée « non mesurée » — rien de plus.

## 6. Périmètre, FS & immuabilité

- **FS supportés (P4, arbitrage lead)** : **allowlist POSIX local same-FS (APFS vérifié)** ; **NFS / FS non supporté / cross-FS ⇒ abort fail-closed** (vérification device+type avant toute publication).
- **Sonde de capacité (établie par comportement)** : exécutée **dans le répertoire/FS cible de publication** (pas un tmp arbitraire) — créer un tmp, `os.link(tmp, tmp2)`, vérifier **inode identique**, **`EEXIST` au 2ᵉ link**, **`st_dev` identique** ; **nettoyage** (unlink + **fsync parent**) ; échec ⇒ **abort fail-closed**.
- **« Immuable » = par discipline (P5)** : 0700, **aucune écriture après le manifest** ; durcissement optionnel post-publication (dirs 0500 / fichiers 0444, **avec fsync des métadonnées**).
- **DEV-only** ; ne touche ni `runner_confirm.py` historique, ni scellés, ni seeds réservés/test2.
- **Pas de commit avant** ces précisions + **audit tagi-5**. Phases 2–3 **toujours bloquées**.

---

*Rédigé par tagi-4 (OPS), 2026-09-23, instructions lead 13:13:31 + revues 13:14:48 et 13:16:10 (msgIds `m_mue08o4y_b6f6c064`, `m_mue0abpd_afb57d79`, `m_mue0c33k_7576a79c`).*
