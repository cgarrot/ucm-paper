# AUDIT — Distribution V1-bis DEV (support v3 vs distribution visitée)

- **Demandé par** : lead (2026-09-23 10:40), après acceptation du NO-GO distribution.
- **Auteur** : tagi-5 (auditeur QA/red-team indépendant).
- **Portée** : **commits propres** `2c98614` (v2) et `6f961a3` (v3), dans des **worktrees détachés** ; **fixtures DEV générées en mémoire uniquement — aucun fichier scellé lu**.
- **Objet** : confirmer/refuter le contre-examen (B1 crash adapter, B2 hors closure, B3 filtre rich vs masse visitée) et **définir les dénominateurs** sans confusion.

## 1. Fixture et définitions (explicites)

- **Layouts** : `generate_siw_layout(rng, rng.randint(2,4), with_dialog=rng.random()<0.5, profile="small")`, `rng=random.Random(20261010)`, **6 layouts**.
- **Support (v3)** : `generate_multistep_adaptation_couples([lay], seed=42+i, couples_per_layout=16)` — couples `schema ucm-siw-adaptation-couples/0.2`, tous dans la **closure forward**.
- **Définitions** :
  - **D_train** = l'ensemble des couples du support ; **ratio 0/0 train** = part des couples avec `filled=0` ET `chosen=0`.
  - **D_vis** = les **décisions visitées** de référence : chemins **optimaux oracle** depuis les états initiaux admissibles (`views × dialog admissible`, `filled/chosen/submitted` vides), **uniformes sur les inits**, par prédicat (VIEW/SET/CHOOSE/SUBMITTED). L'action suivie = `oracle.optimal_actions(st)[0]`.
  - **Part 0/0 décisions** = part des décisions de D_vis sur un état avec `filled=0` ET `chosen=0`.
  - **Couverture de MASSE par ÉTAT (point)** = part des décisions de D_vis dont l'**état exact** ∈ support.
  - **Couverture de MASSE par SIGNATURE** = part des décisions de D_vis dont la **signature** `(1[filed>0], 1[chosen>0], dialog_open, 1[submitted>0])` ∈ signatures du support.
  - **ESS** = taille d'échantillon effective (à calculer sur **fichier de production par k**) ; valeur DEV indicative seulement.

## 2. Résultats

**B1 — crash de matérialisation v2 (`2c98614`)** : sur l'échantillon (10 layouts, 170 couples), la matérialisation par `reset` env échoue **27/170 = 15.9 %**, **tous** sur des layouts **sans dialog** = **dialog fantôme** (`dialog_open=True` impossible). Le 25.8 % du contre-examen = même classe, taux dépendant de l'échantillon. **v3 = 0 crash** (la closure/init support exclut `dialog=True` sans dialog).

**B2 — hors closure v2 (`2c98614`)** : **80/170 = 47.1 %** hors closure forward (mon échantillon) ; le contre-examen rapporte 54.3 % (autre sample) — même classe. **v3 = 0/170** (`rejected_not_in_closure: 0`, invariant `couples_subset_closure`).

**B3 — filtre rich vs distribution visitée (v3 `6f961a3`)** :

| métrique | valeur |
|---|---|
| D_train | 102 couples |
| **ratio 0/0 TRAIN** | **5.9 %** (6 couples) — smoke 510 couples : 4.3 % |
| D_vis | 538 décisions |
| **part 0/0 DÉCISIONS** | **55.4 %** |
| **couverture de masse par ÉTAT (point)** | **12.5–14.5 %** (selon le run) |
| **couverture de masse par SIGNATURE** | **68.6–70.6 %** |
| **ESS DEV indicatif** | **≈ 1.3** (à recalculer sur production par k) |

Par prédicat (mass point) : VIEW 14.5 %, CHOOSE 14.9 %, SET 14.0 %, SUBMITTED 9.8 % (run 1).

**Ne pas confondre** : ratio 0/0 **train** (~5.9 %) ≠ part 0/0 **décisions** (55.4 %) ≠ couverture de **masse point** (~12.5–14.5 %) ≠ couverture de **signature** (~69–71 %) ≠ **ESS** (production par k).

## 3. Finding annexe — sensibilité de la référence (non-déterminisme)

Deux exécutions **identiques** (mêmes seeds/fixture/code, processus séparés) donnent des couvertures **différentes** (point 12.5 % vs 13.4 %, signature 70.6 % vs 68.6 %) alors que D_train/D_vis/0-0 sont **stables** → l'**identité** des états sélectionnés varie, pas les comptes. Cause probable : l'ordre de `oracle.optimal_actions(...)` (construction sur set/dict à clés chaînes → **dépendant de PYTHONHASHSEED**) → le chemin optimal choisi varie entre processus. **Conséquence pour le gate** : la mesure de couverture/ESS **doit être canonique** (trier les actions optimales et/ou fixer `PYTHONHASHSEED`) sinon elle n'est pas reproductible — à intégrer au protocole V1-bis.

## 4. Design statistique

- **Seeds = clusters indépendants** (l'unité de réplication) ; **les 4 prédicats = strates intra-seed** — ne pas traiter 40 observations indépendantes.
- Contrainte **1 form/layout** (profil small) → au plus **1 but SUBMITTED distinct par layout**.
- Gate à pré-enregistrer (à calculer sur **fichier de production par k**) : couverture de masse du support (prédicat × profondeur × features) + **ESS** + vraie masse initiale ; seuil à décider **avant** le test.

## 5. Verdict

**NO-GO DISTRIBUTION maintenu** : la v3 corrige B1/B2 (closure/dialog fantôme) mais **le support ne reproduit pas la distribution visitée** (0/0 = 55.4 % des décisions ; masse point couverte ~12.5–14.5 % ; train 0/0 5.9 %). Prérequis avant toute revendication V1-bis : définir la distribution cible, mesurer couverture/ESS sur **production par k** avec **ordre d'action canonique**, et pré-enregistrer les ratios (initial, STOP, profondeur). Aucun scellé ouvert ; HOLD V6/test2 intact.

## 6. Signature

tagi-5 — auditeur indépendant. Fixtures/seeds reproductibles (hors sensibilité §3 documentée) ; audit sur commits propres `2c98614`/`6f961a3` ; DEV-only.
