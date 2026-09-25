# RÉCONCILIATION CHECKPOINTS — table de traçabilité (25/09 12:30)

**Statut : document de réconciliation exigé par la 6e revue.** Sources vérifiées depuis les shapes des tenseurs (pas les noms de fichiers) et le code de filtrage réellement exécuté.

## Table par checkpoint canon V1-bis

| Seed | Répertoire | Largeur (shape `node_enc.layers.0.bias`) | SHA-256 (12 hex) | Statut dans les runs |
|---|---|---|---|---|
| s0 | 20260922-155520-gate2c-B144-s0 | **144** ✅ | `7c2525a30773` | **UTILISÉ** S2b (canon_checkpoints_from_freeze) |
| s1 | 20260922-155614-gate2c-B144-s1 | **144** ✅ | `1a1611a63692` | **UTILISÉ** S2b |
| s2 | 20260922-155707-gate2c-B144-s2 | **144** ✅ | `2285cf782ebf` | **UTILISÉ** S2b |
| s3 | 20260922-155757-gate2c-B144-s3 | **144** ✅ | `2932e75a6b6a` | **UTILISÉ** S2b |
| s4 | 20260922-155847-gate2c-B144-s4 | **144** ✅ | `e0f895252969` | **UTILISÉ** S2b |
| s5 | 20260923-145341-gate2c-B144-s5 | **192** ⚠️ | `2d29b0f5abc6` | **EXCLU** S2b (filtre dimension) |
| s6 | 20260923-145444-gate2c-B144-s6 | **192** ⚠️ | `b05cd245db76` | **EXCLU** S2b |
| s7 | 20260923-145551-gate2c-B144-s7 | **192** ⚠️ | `35c5df2832d9` | **EXCLU** S2b |
| s8 | 20260923-145653-gate2c-B144-s8 | **192** ⚠️ | `6be93a6d07e3` | **EXCLU** S2b |
| s9 | 20260923-145755-gate2c-B144-s9 | **192** ⚠️ | `20264969db85` | **EXCLU** S2b |

## Découverte et résolution

**Les canon s5-s9 sont de largeur 192** (découverte tagi-3 à 14:2x du 24/09 — génération différente), **pas 144** comme leur nom `B144` le suggère. Le **filtre par dimension réelle** (`canon_checkpoints_from_freeze`, probe `node_enc.layers.0.bias.shape[-1] == 144`) les **exclut mécaniquement** : seuls s0-s4 (véritables d=144) sont utilisés dans le run S2b — **vérifié par le lead depuis le code et les shapes au 25/09 12:30**.

**Conséquences** :
1. Le run S2b n'a **jamais chargé** de checkpoint d=192 — les gardes `CANON NOT LOADED` (shapes mismatch) l'auraient refusé de toute façon (fail-closed, découverte documentée dans `d5c88e8`).
2. Le freeze v10 liste les 10 checkpoints dans `arms.pretrained_TGK.checkpoints` mais le **runtime filtre** à 5 — la description du freeze devrait le documenter plus explicitement (note : le filtre est dans le code committé, pas dans le manifeste).
3. **Impact sur V1-bis** : le run V1-bis (`freeze-v1bis-official-v10.json`) utilisait `canon_checkpoints_from_freeze()` qui filtre également — **les canon s5-s9 n'ont jamais été utilisés dans AUCUN run** (V1-bis comme S2b). L'incident est **fermé sans ré-exécution** : les 192 étrangers étaient présents dans le freeze mais jamais chargés.

## Liaison cell_id → source

Toutes les cellules S2b run3 portent `canon_sha256` dans leur artefact — les 5 valeurs distinctes correspondent aux sha s0-s4 ci-dessus, **une par seed** (mapping positionnel `_canon_for(seed)`).
