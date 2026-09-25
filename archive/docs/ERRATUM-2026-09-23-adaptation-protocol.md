# ERRATUM 2026-09-23 — Budget adaptation étape 4 : réconciliation des deux protocoles de mesure

- **Statut** : erratum **séparé**, daté. Aucune réécriture : les mesures et rapports antérieurs restent inchangés (append-only). Le présent document **acte la réconciliation** (ordre lead 17:57:52).
- **Objet** : le premier budget (`reports/adaptation-official-dev-profile.json` @ `f80629e`) avait été mesuré sous **protocole A cumulatif** (un même modèle enchaînait k=64→128→256) ; une **re-mesure sous protocole B indépendant** (modèle FRAIS par k) a été ordonnée puis exécutée. Le budget consolidé est établi ci-dessous avec **statistique primaire pré-déclarée**.

## 1. Chronologie

| Heure | Événement |
|---|---|
| ~16:02 | Mesure protocole A (f80629e) : 1 cellule k=64/128/256, updates=2000, batch 64, CPU — **11,52 h** extrapolées ; audit tagi-5 PASS borné |
| ~16:25-17:04 | Réps 2-3 protocole A (background) : **12,98 h** / **10,53 h** |
| 17:57:52 | **Ordre lead** : re-mesure 1 cellule **protocole indépendant** (modèle frais par k), même fixture, 2-3 réps |
| ~18:00-19:15 | Mesure protocole B (2 réps) : k=64 263,4 s · k=128 342,9 s · k=256 368,4 s (mean) → **max 11,5 h** / mean 10,83 / min 10,16 |
| 19:19:20 | GO étape 4 délivré (exécution parallèle 4 workers, fenêtre à confirmer) |

## 2. Réconciliation A vs B

- **Protocole A rep1 (11,52 h) ≈ protocole B max (11,50 h)** — écart **~0,2 %** : le caractère cumulatif du protocole A n'a **pas** distordu matériellement le coût par-k sur cette fixture (l'état accumulé change peu le coût marginal des updates suivants à budget fixe).
- L'étalement observé (**10,16 → 12,98 h**) est **piloté par la charge machine** (réps exécutées à load ~6-9), pas par le protocole : à noter, la rep2 (A) — la plus lente — a couru sous la charge la plus lourde.
- **Conséquence budget** : statistique **primaire = max = 12,98 h** (adaptation) ; **min = 10,16 h** = plancher/sensibilité (jamais la base du budget) ; la répétition calme au T0 **quantifiera l'écart de charge sans re-choisir le budget**.

## 3. Budget consolidé (pré-déclaré)

| Composante | Valeur | Base |
|---|---|---|
| Adaptation — primaire (**max**) | **12,98 h** | max des 4 extrapolations (A rep1-3 + B max) |
| Adaptation — mean | 12,41 h | mean des extrapolations |
| Adaptation — min (plancher) | 10,16 h | protocole B rep la moins chargée |
| Eval mesurée (72k ép.) | 1,5 h (mean) / **2,32 h** (p95) | `reports/eval-official-dev-profile.json` @ `e571ee4` |
| **Total primaire (max + eval p95)** | **≈ 15,3 h** | borne haute prudente |
| **Total typique (mean + eval mean)** | **≈ 13,4 h** | scénario médian |

## 4. Calibration M-V1b (documentée)

- Source : `artifacts/v1/transfer-report.json` (`finetune.wall_s`), **75 cellules** (15 à k=0 = 0 s ; 60 non-nulles).
- Protocole exact : réf `ucm/v1/FREEZE.md` ; `FinetuneConfig` par défaut **updates=2000, batch=64**, lr 3e-4, wd 1e-4, clip 1.0 ; **CPU** (`runner.py:27`) ; couples `siw-couples-k{0,100,500,2000,10000}.jsonl` (DEV), seeds 0-4.
- Wall : k=500 **mean 376,9 s** (min 334,6 / max 456,2) ; global non-nul mean 367,9 s (309,3-563,1).
- **Cross-check plateau** : k=500 (376,9 s) ≈ k=128 (342,9) / k=256 (368,4) mesurés indépendamment → concordance ~0,4 % au plateau (~377 s/cellule à updates=2000).
- ⚠️ **Limite** : charge machine **non journalisée** à l'époque (aucun load logger) — comparaison indicative.

## 5. Preuves brutes (hashes)

| Fichier | sha256 |
|---|---|
| rapport consolidé | `d27765832ff80c44d98118bbdad687cdbdef70788660c5b26b6ec72937f4f7cd` |
| protocole B rapport | `39750edbc8d1ff15b2a1c00b73064d3035158f47c7a0b36cd1de7cd83ff7b504` |
| protocole B log | `8dab0c073f0edfff1d819b0cb0c22105e87a53511cc4ab17e4ebd97622d6a577` |
| protocole B script | `dd9ba651672d7fad5b00cbeae2900e9b1c502c9c367b19278a546d12cf05f3e7` |
| protocole A rep2 rapport/log | `a75f1ea1f0c2c4d0a9dcf91149c031cba185e53c340c15b5cf149493aaed1f9e` / `280e1541eba69eb8bfe37e40f7686e66826efd43553b6373ef90e5470b5f95df` |
| protocole A rep3 rapport/log | `68ff4030df9e335e43de7e42c79bbc2fbe5cf4da39090adf98a624bcae16a1c8` / `352e9acac35d1db089f66b8eb3bb4d7696c9a12ba49e0c981fadd9139dbe63b2` |
| loader de charge (fenêtre) | `50ac6753da797d725efaaa96f25d118223a07d4bda3cadb0f85531ee0201955f` |

**SHA de cet erratum** : publié **hors fichier** (message/registre) — jamais d'auto-référence.

---

*Rédigé par tagi-4 (OPS), 2026-09-23, sur ordre lead 17:57:52 (msg id `m_mueaectg_30769bbb`).*
