# INCIDENT 2026-09-23 — lectures du manifest scellé post-HOLD (v3, audit trail append-only)

- **ID** : INCIDENT-2026-09-23-01 (manifest-read post-HOLD)
- **Statut** : HOLD V6/test2 **maintenu** — **aucun GO test2**, aucune génération/scellage/ouverture test2.
- **Nature** : **2 lectures déclarées** de `artifacts/split-manifest-SIW-v1.json` (artifact V1 scellé) après le HOLD — correction tagi-2 (10:40, après session records) : **2 opens, et non 1** ; l'`ops_note` v0.3 (`b9727cb`) avait fusionné abusivement les deux.
- **Append-only** : document **séparé** ; §9 de l'incident 22/09 (`dbdbb61`) **non modifié**. Toute correction future = **nouveau document/erratum**, jamais d'overwrite.
- **v3 supersede v1/v2** : `v1` (`...-v1.json/.md`, blob `5515097`) et `v2` (drafts non commités) contenaient `fd_declared:1` / fenêtre unique — **faux** ; ils sont conservés comme **copies historiques explicitement superseded** (aucune réécriture).
- **Relais** : tagi-2 (déclarant) → lead ; recoupement tagi-5 ; **correction 2 opens** (10:40).
- **Ruling** : metadata-only **2 FD** post-HOLD = **déviation HOLD stricte** ; one-read du test **intact**.

## 1. Faits déclarés (D9a / D9b)

| ID | Fenêtre (déclarée) | Méthode | Contenu | FD |
|---|---|---|---|---|
| **D9a** | **~10:21–10:23** (non attesté) | `Path(...).read_text()` + json parse | **420 hashes** de métadonnées (pool200 v0.2, mtime 10:21:58) — aucun contenu test | 1 |
| **D9b** | **~10:26** (non attesté) | `shasum -a 256` | **lecture intégrale des octets pour SHA256** — sans parsing ni extraction de champs (la lecture a bien lieu) | 1 |

- **Total : 2 FD auto-déclarés, NON attestés** (pas de journal par ouverture ; horodatages exacts inconnus).
- Couples/test/layouts scellés ouverts : **0** (selon déclarant).
- Aucune issue/outcome exposé ; **sélection structurelle** de layouts effectuée en D9a (exclusion iso-disjoint) — seule la sélection **outcome/perf** est impossible.
- **Attesté** : les sha connus (ci-dessous, pré-HOLD) ; le mtime inchangé (`stat`, sans ouverture). L'existence des ouvertures est **inférée** (recoupement tagi-5 : artefact pool200 + sources + mtime) — **non attestée par un log d'ouverture**.
- **Auto-déclaré (non attesté)** : les 2 FD, les fenêtres ~10:21–10:23 / ~10:26, le périmètre exact, les contenus.

## 2. Provenance (sans réouverture)

- Fichier : `artifacts/split-manifest-SIW-v1.json` — **60 277 octets**, **mtime 2026-09-22 18:51:58** (vérifié par `stat`, **aucune ouverture**).
- sha256 fichier : `551371db9085e5af4e7f2a58e4ce8800de144b5c00bc3a931f4208f1f14732a2`
- scellé `manifest_sha256` : `0b28a93022c5d7ebd302d8f5f4ac6994da61ee1f8569f31ebadd40a4ff49c652`
- **Sources** : `scripts/compare_siw_seals.py` (committé `3b26c01`) + vérification M-V1a tagi-4 du 22/09 (pré-HOLD, divulguée D6). **Aucune réouverture du scellé** pour ce document.

## 3. Périmètre d'exposition

- Objet lu : **métadonnées de pools / coordonnées de scellé** (hashes) — pas le contenu des épisodes test, pas de couples, pas de layouts.
- **Sélection structurelle de layouts : OUI** (D9a, exclusion iso-disjoint du pool DEV) ; **seule la sélection outcome/performance est impossible**.
- **Reproductibilité de l'artefact final** : chaîne **v0.3 (`b9727cb`) → v0.4 (`889f585`)** — v0.3 supersede `1a37eddc` avec exclusion dérivée du **REJEU SEED** (zéro lecture scellée), v0.3 **non effacée** ; la reproductibilité n'est pas entachée ; **cela n'annule pas l'ouverture déclarée** (D9a/D9b restent divulguées).

## 4. Politique metadata (tranchée par le lead)

- **Décision lead** : les métadonnées de scellé forment une **classe distincte** ; **HOLD strict** maintenu ; toute lecture metadata exige une **pré-intention** (fsync AVANT open) et une journalisation — **sauf dérogation explicite du lead**.
- Les accès D9a/D9b (post-HOLD, sans pré-intention) restent **non conformes au sens pré-enregistré** et sont divulgués comme tels.
- HOLD test2 intact ; aucun GO test2 ; aucune génération.

## 5. Correctif sans reread

- **Aucune réouverture** du scellé pour auditer ces faits (interdite) : recoupement par déclarations + historique + metadata (`stat`).
- **Règle de pré-intention metadata** : toute lecture future de métadonnées de scellé (pools/hashes/coordonnées) doit être **pré-déclarée** (intention fsync **AVANT** open) et journalisée comme une lecture test — sauf exception metadata explicitement qualifiée par le lead.
- Correctifs futurs : journal par ouverture **y compris metadata** ; garde instrumentée si la lecture metadata est jugée admissible.
- **Correction `ops_note` v0.4 — PRODUITE** (commit `889f585`) : l'`ops_note` v0.3 (`b9727cb`) fusionnait abusivement les 2 opens ; la v0.4 corrige, **sans reread** (v0.3 non effacée).
- Toute correction de ce document = **nouveau document/erratum** ; jamais d'overwrite.

## 6. Relecture

- **v3 soumis à l'audit tagi-5** ; **commit atomique** des nouveaux fichiers (v3 + erratum + copies historiques) après PASS + validation lead. Aucune pré-certification.

---

*Rédigé par tagi-4 (OPS), 2026-09-23, sur instruction lead 10:40:59 (msgId `m_mudusivx_b99a618b`).*
