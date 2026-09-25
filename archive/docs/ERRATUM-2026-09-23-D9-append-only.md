# ERRATUM 2026-09-23 — D9 : fail de processus append-only + chaîne des révisions v1/v2

- **Statut** : erratum **séparé**, daté. **Schéma à valider par tagi-5/lead AVANT commit** (exigence lead 10:35:41). Aucune lecture scellée.
- **Objet** : documenter le **fail de processus** de la divulgation D9 (lecture manifest post-HOLD) et **figer la chaîne des révisions** (v1 restaurée byte-identique, v2 corrigée), sans réécrire les fichiers publiés.

## 1. Chronologie exacte (Europe/Paris)

| Heure | Événement | Source |
|---|---|---|
| 10:29:09 | Relais lead de la divulgation D9 (tagi-2) | msg `m_mududaok_7684f20b` |
| 10:29:39 | Recoupement tagi-5 (existence **inférée**, pas de log FD) | msg `m_mududxr6_a9399b32` |
| 10:32:22 | « Audit vert » tagi-5 sur D9 — **erroné, rétracté depuis** | msg `m_muduhfwv_5a67e093` |
| **10:32:36** | **Commit `5515097`** — D9 v1 (inexactitudes : effective_time, extraction, attestation, sélection, §6) | git |
| **10:32:39** | **HOLD lead** — 5 bloqueurs factuels | msg `m_muduhsgw_ea70e5f7` |
| 10:33:36 | **Rétractation tagi-5** — PASS refusé, 5 points | msg `m_muduj0gx_d5878a66` |
| **10:33:45** | **Commit `897fe09`** — corrections **en place** des mêmes 2 fichiers (−11 lignes) | git |
| 10:34:04 | tagi-5 recommande un **erratum v2** (nouveaux noms) | msg `m_mudujm2k_d03758c0` |
| 10:35:41 | Lead : **STOP réécriture**, v1+v2 nouveaux noms + erratum séparé, schéma à valider | msg `m_mudulpmc_e0dadd18` |
| **10:40** | **tagi-2 CORRIGE : 2 opens** (D9a ~10:21–10:23 extraction 420 hashes ; D9b ~10:26 SHA-only) — l'`ops_note` v0.3 fusionnait abusivement les deux | lead `m_mudusivx_b99a618b` |
| **10:40:59** | **Ruling changé** : STOP commit chaîne gelée (`fd_declared:1` faux) ; créer **v3** D9a/D9b | msg `m_mudusivx_b99a618b` |

## 2. Fail de processus (explicite)

1. `5515097` a **publié D9 v1 avec des inexactitudes** (le « green » initial de tagi-5, 10:32:22, était **erroné** — rétracté 10:33:36).
2. `897fe09` (10:33:45) a **modifié EN PLACE** les mêmes 2 fichiers alors que le document proclame `append_only:true` et « toute correction future = nouveau document/erratum, jamais d'overwrite » → **violation d'append-only**. Les corrections étaient factuellement justifiées (5 bloqueurs du lead) mais la forme est contraire à la règle d'immutabilité que le document lui-même invoque.
3. `897fe09` a de plus été committé **après le HOLD (10:32:39) et après la rétractation (10:33:36)** — séquence non conforme.
4. **`897fe09` n'est pas effacé de l'histoire** ; le présent erratum le cite. Les fichiers publiés `incident-2026-09-23-manifest-read.json/.md` restent dans leur état `897fe09` — **aucune nouvelle réécriture** (STOP lead).
5. **`fd_declared:1` (v1/v2) était FAUX** : tagi-2 a corrigé à **2 opens post-HOLD** (D9a : extraction 420 hashes ~10:21–10:23 ; D9b : SHA-only ~10:26) — l'`ops_note` v0.3 (`b9727cb`) fusionnait abusivement les deux. La chaîne **v1/v2 est superseded par la v3**.

## 3. Chaîne des révisions (figée)

| Révision | Fichiers | Contenu | Statut |
|---|---|---|---|
| **v1** | `artifacts/v1/incident-2026-09-23-manifest-read-v1.json` + `docs/INCIDENT-2026-09-23-manifest-read-v1.md` | Restauré byte-identique depuis le blob `5515097` — contient `fd_declared:1` (**faux**) | **copie historique superseded** — publiée au commit de cet erratum |
| **v2** | `artifacts/v1/incident-2026-09-23-manifest-read-v2.json` + `docs/INCIDENT-2026-09-23-manifest-read-v2.md` | 5 points corrigés mais encore `fd_declared:1` / fenêtre unique | **copie historique superseded** — publiée au commit de cet erratum |
| **v3** | `artifacts/v1/incident-2026-09-23-manifest-read-v3.json` + `docs/INCIDENT-2026-09-23-manifest-read-v3.md` | **D9a/D9b** ; **2 FD auto-déclarés non attestés** ; fenêtres distinctes **~10:21–10:23** / **~10:26** ; schéma `ucm-incident-disclosure/0.2` ; supersede v1/v2 | **active — audit tagi-5 en cours** |
| publiés | `incident-2026-09-23-manifest-read.json/.md` | état `897fe09` | **conservés tels quels, plus de réécriture** |

## 4. Hashes

| Fichier | sha256 |
|---|---|
| v1 JSON (== blob `5515097`) | `9637c782803beb27167a88941c1d6197bc328e3362f9d787bb6a45ad2fdf0ce9` |
| v1 MD | `d11e234787f0d03173149dcb354470f5e8e85091eaa46224811676f4b1d4d5ff` |
| v2 JSON | `c79f3cbe6fbdb72b84e6677591aafdae823da91f9c57ee662af2cdc17c1391e9` |
| v2 MD | `da43d3ca66e3779479475a856501a13e3646af59bc88a3579dafba5a6a8d21fa` |
| publiés (état `897fe09`) JSON | `b07429c434b349bd866f1ca8e01e8c66818771d65381e216cc05fb4d1a4ff352` |
| publiés (état `897fe09`) MD | `619b1bea512ad4f365e8d6a56c212bb9296c39f7dec44d173588394ef76b07b5` |
| **v3 JSON** | `e0debcddd80180fcd07e9b9efdf65f85c4d8da5d5627094414e9ee065fce3d8e` |
| **v3 MD** | `d60e4089f5002c8ea8558a34d40feabfb71c12a33affd282543934ab4fb6cc69` |

**SHA de cet erratum** : publié **hors du fichier** (message/registre de commit) — jamais d'auto-référence (circularité).

Commits cités : `5515097` (2026-09-23 10:32:36) · `897fe09` (2026-09-23 10:33:45) — historique **intact**.

## 5. Règle rétablie

- **Append-only strict** pour toute la chaîne D9 : toute correction = **nouveau fichier + erratum daté**, jamais d'overwrite ; prochaine correction éventuelle = **v4 + erratum**.
- Le présent erratum (schéma : v1/v2 nouveaux noms + ce document) doit être **validé par tagi-5/lead avant commit**.

## 6. Suites

- **Re-revue tagi-5 (10:35:08)** : les 5 corrections de `897fe09` sont **appliquées ✅** ; ajustement résiduel (~10:26 + rejeu seed) intégré à la v2.
- **Correction majeure tagi-2 (10:40)** : **2 opens post-HOLD** (et non 1) — `fd_declared:1` de v1/v2 **faux** ; l'`ops_note` v0.3 fusionnait abusivement. Ruling lead : STOP commit chaîne gelée → **v3** (D9a/D9b, 2 FD auto-déclarés non attestés, fenêtres distinctes).
- **v3 créée** (schéma `ucm-incident-disclosure/0.2` — nit tagi-5 appliqué) ; audit tagi-5 puis **commit atomique** des nouveaux fichiers (v3 + cet erratum + copies historiques v1/v2).
- **Corrective `ops_note` v0.4 — PRODUITE** (commit `889f585`, lead 10:42) : la fusion abusive des 2 opens par l'`ops_note` v0.3 est corrigée en **v0.4** ; v0.3 **non effacée** ; aucun reread.

---

*Rédigé par tagi-4 (OPS), 2026-09-23, sur instruction lead 10:35:41 (msgId `m_mudulpmc_e0dadd18`).*
