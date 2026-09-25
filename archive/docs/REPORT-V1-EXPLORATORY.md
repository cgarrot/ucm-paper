# Rapport V1 Exploratoire — Run M-V1b (75/75 cellules)

**Statut : STRICTEMENT EXPLORATOIRE (NO-GATE).** Aucun chiffre ne franchit un
gate. Le verdict one-read a été rendu (incident 21:27, ruling lead) : le run
est exploratoire par construction. La confirmation officielle (runner v6,
protocole gelé) tranchera seule.

**Sources (aucun scellé rouvert pour ce rapport) :**
- `artifacts/v1/transfer-report.json` — sha256 `587d13b1cb49ab6db4d5a0d0ec9827165b82279c1edacc44606ea255c852c553`
- `artifacts/v1/mv1b-run.log` — sha256 `f3beb3ad825cd2d69f9d9fb1e88c55984e8bd4c296d8f8bd1191654dadfbb02d`
- `artifacts/v1/interactions-log.json` — sha256 `397d62b8efc200c16b09559600c0506d0f1c5fda009af828211ff9b6b6e23f1c` (chaîne d'ÉVÉNEMENTS intacte)

## 1. Configuration

3 bras (scratch / pretrained-TGK / contrôle-non-ciblé) × 5 seeds × 5 budgets
(k=0/100/500/2000/10000) sur les couples scellés SIW (94/376/852/1191 uniques
réels) ; test = 600 épisodes scellés (150×4 prédicats, 167 layouts) ; init
fraîche partagée par seed (mx.random.seed(9000+seed)) ; fine-tune 2000 updates
batch 64 à budget fixe sans sélection cible ; CPU, FP32, séquentiel.

## 2. Résultats (IC AUTO-RAPPORTÉS, non recalculables indépendamment — cf. §3.3)

**Primaire k\*=500 :** pré-entraîné − scratch = **+12.57 pts**, IC95 apparié
hiérarchique auto-rapporté **[+2.56 ; +22.35] > 0** (formellement ≥ 5 pts,
EXPLORATOIRE ; IC non recalculable indépendamment faute de raw par épisode).

**Attribution (lecture §12.5) :**
- contrôle − scratch = **+13.37 pts** IC auto-rapporté [+3.77 ; +22.76]
- contrôle − pré-entraîné = **+0.81 pts** IC auto-rapporté [−13.32 ; +13.88]
  (IC traverse zéro — **contrôle comparable observé** ; mécanisme
  exposition/optimisation **plausible**, **aucune attribution décisionnelle
  établie**, ni absence d'effet décisionnel prouvée — pas un test
  d'équivalence).

**AULC log(1+k) :** scratch 0.199 / pretrained 0.253 / control 0.287.
**Seuil 80 % :** censuré « >10000 » pour les trois bras (aucun ne l'atteint).

**Succès par bras×budget (moyennes 5 seeds, 4 décimales, dérivé par script
de `transfer-report.json.cells`) :**

| bras | k=0 | k=100 | k=500 | k=2000 | k=10000 |
|---|---|---|---|---|---|
| scratch | 0.0000 | 0.2210 | 0.2887 | 0.2897 | 0.3517 |
| pretrained-TGK | 0.0000 | 0.2707 | 0.4143 | 0.3930 | 0.3447 |
| control-nontarget | 0.0000 | 0.3617 | 0.4223 | 0.3910 | 0.3750 |

## 3. Limites et erratum

1. **One-read NON tenu** (incident 21:27 + reconstruction) : le chemin de code
   du run a ouvert les fichiers scellés **11 fois** (8 FD couples = 4
   matérialisations + 4 content_hash ; 3 FD test = sniff + parse + hash). Le
   journal (6 entrées) est une **chaîne d'ÉVÉNEMENTS** (i0–i3 = démarrages,
   i4 = hashes post-consommation, i5 = déclaration post-open) — PAS un compte
   d'opens ni une chaîne d'intentions pré-open ; la déclaration « one read »
   (i5) était instrumentalement fausse. Borne hors-run de l'incident de
   tests : documentée séparément (incident 21:27, ruling lead `dbdbb61`, sidecar
   sha `704c0543…`).
2. **Attestation de provenance non fiable** : `materialized_records_hashes_per_budget`
   calculés sur `budgets.slice(k)` (ordre interne) ≠ slices réellement
   entraînées (`sealed_slices[k]`) — hash dupliqué k2000/k10000, divergence
   k100/k500. Le chemin de code d'entraînement est intact (les hashes fautifs
   n'ont nourri aucune cellule), mais c'est une conclusion de chemin de code,
   pas une preuve d'identité d'entraînement indépendante.
3. **Pas de raw par épisode ni checkpoints par cellule** dans ce run (le runner
   v1 ne les produisait pas) : le **point estimate** du primaire (+12.5667 pp)
   est recomputable depuis les 5 agrégats k=500 ; **l'IC apparié et les paires
   par épisode ne sont PAS recomputables** sans raw. Le runner v6 produit le
   raw et re-calcule tout depuis lui.
4. Succès par cellule positive-k (n=60) : **min 0.1417 / max 0.4983** ;
   non-monotonie en k observable — interprétation différée à la confirmation ;
   la couverture publiée (94/376/852/1191 couples uniques réels) éclaire le
   plateau.

## 4. Suite

Confirmation officielle = runner v6 (`f96c940` + `84f1874` + chemins absolus),
protocole à figer (freeze-manifest + data-manifest + S_gen/S_ep + test2 scellé
à générer) ; aucun run confirm sans gel ; aucun scellé rouvert entre-temps.
