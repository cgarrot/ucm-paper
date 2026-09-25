# INCIDENT 2026-09-23 — lecture du manifest scellé post-HOLD (audit trail append-only)

- **ID** : INCIDENT-2026-09-23-01 (manifest-read post-HOLD)
- **Statut** : HOLD V6/test2 **maintenu** — **aucun GO test2**, aucune génération/scellage/ouverture test2.
- **Nature** : divulgation d'**une lecture** de `artifacts/split-manifest-SIW-v1.json` (artifact V1 scellé) **après le HOLD**.
- **Append-only** : document **séparé** ; le §9 de l'incident 22/09 (committé `dbdbb61`) n'est **pas** modifié. Toute correction future = nouveau document/erratum, jamais d'overwrite.
- **Relais** : tagi-2 (déclarant) → lead 10:29:09 ; recoupement tagi-5 : **1 ouverture post-HOLD** confirmée.
- **Mécanisme lead (ruling tagi-5)** : événement **D9** — sidecar `artifacts/v1/incident-2026-09-23-manifest-read.json` (ce doc = audit trail associé) ; ancien sidecar D1–D8 et §9 incident 22/09 **non modifiés** (immutabilité historique). Ruling : metadata-only = **déviation HOLD stricte**, one-read du test **intact**.

## 1. Fait et statut d'attestation

| Élément | Valeur | Statut |
|---|---|---|
| Ouvertures | **1 FD déclaré** (open unique, lecture seule) | auto-déclaré (tagi-2) ; **existence d'une ouverture post-HOLD confirmée par tagi-5** |
| Périmètre | **metadata hashes de pools (pool200 DEV)** — pas d'épisodes test | auto-déclaré (non attesté ligne à ligne) |
| Couples/test/layout scellés ouverts | **0** (selon déclarant) | auto-déclaré |
| Horodatage effectif | **2026-09-23 10:21:58** (horaire effectif de l'accès — **distinct du mtime** 22/09 18:51:58) | relayé par le lead (ruling tagi-5) |
| Extraction de contenu | aucune (déclaré) | auto-déclaré |

**Attesté** : l'existence et la nature (metadata) de l'ouverture (recoupement tagi-5) ; les sha connus (ci-dessous, pré-HOLD) ; le mtime inchangé (stat, sans ouverture).
**Auto-déclaré (non attesté)** : FD exact, périmètre exact, absence d'extraction, timestamp précis — pas de journal par ouverture.

## 2. Provenance (sans réouverture)

- Fichier : `artifacts/split-manifest-SIW-v1.json` — **60 277 octets**, **mtime 2026-09-22 18:51:58** (vérifié par `stat`, **aucune ouverture**).
- sha256 fichier : `551371db9085e5af4e7f2a58e4ce8800de144b5c00bc3a931f4208f1f14732a2`
- scellé `manifest_sha256` : `0b28a93022c5d7ebd302d8f5f4ac6994da61ee1f8569f31ebadd40a4ff49c652`
- **Sources** : `scripts/compare_siw_seals.py` (committé `3b26c01`) + vérification M-V1a tagi-4 du 22/09 (pré-HOLD, déjà divulguée en D6 de l'incident 22/09). **Aucune réouverture** pour ce document.

## 3. Périmètre d'exposition

- Objet lu : **métadonnées de pools / coordonnées de scellé** (hashes) — pas le contenu des épisodes test, pas de couples, pas de layouts.
- Aucun résultat (issue de tâche, succès) n'est contenu dans ces métadonnées : **aucune sélection possible** sur cette base.
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

- **Relu par tagi-5 AVANT commit** (exigence lead 10:29:39).

---

*Rédigé par tagi-4 (OPS), 2026-09-23, sur instruction lead (msgId `m_mududxr6_a9399b32`).*
