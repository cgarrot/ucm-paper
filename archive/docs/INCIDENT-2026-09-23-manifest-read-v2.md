# INCIDENT 2026-09-23 — lecture du manifest scellé post-HOLD (audit trail append-only)

- **ID** : INCIDENT-2026-09-23-01 (manifest-read post-HOLD)
- **Statut** : HOLD V6/test2 **maintenu** — **aucun GO test2**, aucune génération/scellage/ouverture test2.
- **Nature** : divulgation d'**une lecture** de `artifacts/split-manifest-SIW-v1.json` (artifact V1 scellé) **après le HOLD**.
- **Append-only** : document **séparé** ; le §9 de l'incident 22/09 (committé `dbdbb61`) n'est **pas** modifié. Toute correction future = nouveau document/erratum, jamais d'overwrite.
- **Erratum v2** : référence les fichiers publiés (`artifacts/v1/incident-2026-09-23-manifest-read.json` + `docs/INCIDENT-2026-09-23-manifest-read.md`, commits `5515097` puis corrections partielles `897fe09`) ; **corrige les 5 points** (fenêtre **inférée**, extraction de **hashes** 420, existence **inférée**, sélection **structurelle** effectuée, §6 statut réel) ; **ne réécrit pas** les fichiers publiés.
- **Relais** : tagi-2 (déclarant) → lead 10:29:09 ; recoupement tagi-5 : **existence inférée** (artefact pool200 + sources + mtime post-HOLD) — **non attestée par un log d'ouverture**.
- **Mécanisme lead (ruling tagi-5)** : événement **D9** — sidecar `artifacts/v1/incident-2026-09-23-manifest-read.json` (ce doc = audit trail associé) ; ancien sidecar D1–D8 et §9 incident 22/09 **non modifiés** (immutabilité historique). Ruling : metadata-only = **déviation HOLD stricte**, one-read du test **intact**.

## 1. Fait et statut d'attestation

| Élément | Valeur | Statut |
|---|---|---|
| Ouvertures | **1 FD déclaré** (open unique, lecture seule) | auto-déclaré (tagi-2) ; existence **inférée** (recoupement tagi-5) — **non attestée par un log d'ouverture** |
| Périmètre | **metadata hashes de pools (pool200 DEV)** — pas d'épisodes test | auto-déclaré (non attesté ligne à ligne) |
| Couples/test/layout scellés ouverts | **0** (selon déclarant) | auto-déclaré |
| Horodatage effectif | **inconnu — fenêtre ~10:26 déclarée** (vérif provenance, `ops_note` pool200 v0.3 `b9727cb`) / horodatage exact non attesté ; 10:21:58 = mtime de l'artefact pool200, PAS l'horaire de l'open ; mtime manifest 22/09 sans lien timing FD | à établir par tagi-2 (preuve d'événement horodaté distincte) |
| Extraction | **hashes de métadonnées (420) → `exclusion_set`** ; **aucun contenu test** (épisodes/couples/états) ; zéro issue/outcome | auto-déclaré |

**Attesté** : les sha connus (ci-dessous, pré-HOLD) ; le mtime inchangé (stat, sans ouverture) ; l'existence de l'ouverture est **inférée** (recoupement tagi-5 : artefact pool200 + sources + mtime post-HOLD) — **non attestée par un log d'ouverture**.
**Auto-déclaré (non attesté)** : l'ouverture elle-même, le compteur FD, le périmètre exact, l'extraction effective, l'horodatage — pas de journal par ouverture.

## 2. Provenance (sans réouverture)

- Fichier : `artifacts/split-manifest-SIW-v1.json` — **60 277 octets**, **mtime 2026-09-22 18:51:58** (vérifié par `stat`, **aucune ouverture**).
- sha256 fichier : `551371db9085e5af4e7f2a58e4ce8800de144b5c00bc3a931f4208f1f14732a2`
- scellé `manifest_sha256` : `0b28a93022c5d7ebd302d8f5f4ac6994da61ee1f8569f31ebadd40a4ff49c652`
- **Sources** : `scripts/compare_siw_seals.py` (committé `3b26c01`) + vérification M-V1a tagi-4 du 22/09 (pré-HOLD, déjà divulguée en D6 de l'incident 22/09). **Aucune réouverture du scellé** pour ce document.

## 3. Périmètre d'exposition

- Objet lu : **métadonnées de pools / coordonnées de scellé** (hashes) — pas le contenu des épisodes test, pas de couples, pas de layouts.
- Aucun résultat (issue de tâche, succès) n'est contenu dans ces métadonnées : **sélection structurelle de layouts : OUI** (les 420 hashes ont servi à l'exclusion iso-disjoint du pool DEV) ; **seule la sélection outcome/performance est impossible**.
- **Reproductibilité de l'artefact final** : v0.3 (`b9727cb`) supersede `1a37eddc` avec exclusion dérivée du **REJEU SEED** (zéro lecture scellée) — l'incident **n'entache pas** la reproductibilité de l'artefact final ; l'accès ~10:26 reste divulgué.
- Le manifest reste un **artifact scellé V1** : sa lecture post-HOLD est divulguée comme telle.

## 4. Qualification de politique (metadata vs test)

- À trancher par le lead : les métadonnées de scellé relèvent-elles d'une politique distincte de l'ouverture d'une cellule test ?
- **En l'absence de cette qualification explicite, l'accès est divulgué comme non conforme au sens pré-enregistré** (le HOLD couvrait toute interaction avec les artifacts scellés).
- HOLD test2 intact ; aucun GO test2 ; aucune génération.

## 5. Correctif sans reread

- **Aucune réouverture** du scellé pour auditer ce fait (interdite) : recoupement par déclarations + historique + metadata (`stat`).
- Correctifs futurs (à valider par le lead) : journal par ouverture **y compris metadata** ; politique explicite « metadata vs test » ; garde instrumentée si une lecture metadata est jugée admissible.
- **Règle de pré-intention metadata (D9)** : toute lecture future de métadonnées de scellé (pools/hashes/coordonnées) doit être **pré-déclarée** (intention fsync **AVANT** open) et journalisée comme une lecture test — sauf exception metadata explicitement qualifiée par le lead.
- Toute correction de ce document = nouveau document/erratum ; **jamais** d'overwrite.

## 6. Relecture

- **Re-revue tagi-5 (10:35:08) : les 5 corrections ✅** ; PASS D9 **après** l'ajustement résiduel (fenêtre ~10:26 déclarée + note rejeu seed) — intégré à la présente v2. Commit après **validation lead du schéma** ; aucune pré-certification.

---

*Rédigé par tagi-4 (OPS), 2026-09-23, sur instruction lead (msgId `m_mududxr6_a9399b32`).*
