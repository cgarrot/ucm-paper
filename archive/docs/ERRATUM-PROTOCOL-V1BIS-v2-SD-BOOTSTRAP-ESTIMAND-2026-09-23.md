# ERRATUM — protocole V1-bis v2 — SD de dimensionnement, bootstrap conditionnel, estimand d'évaluation

**Statut : erratum append-only AVANT tout run V1-bis ; supersede partiellement `303dbd3` (bootstrap).** Sources : revue externe 5 (`reports/web-watch-2026/FIFTH-REVIEW.md`, vérifications lead), diffs par seed M-V1b publiés. Aucune donnée V1-bis vue. Toute divergence future = nouveau document.

## 1. SD de dimensionnement corrigée

La sd exploratoire de référence = **12,61 pp** (sd n-1 des 5 diffs par seed M-V1b [14,33 ; 0,67 ; 25,17 ; 24,17 ; −1,50] ; moyenne 12,568) — la valeur 10,9 précédemment citée (doc puissance v02, propagation non recalculée) **n'est dérivable d'aucune convention** et est révoquée. Conséquences : à **10 seeds**, test apparié unilatéral 5 % (convention **t, df=9**), la puissance pour un effet de 10 pp est **≈ 75,0 %** (la normale donnerait 80,6 % unilat. — convention nommée : t) ; le **MDE à 80 % ≈ 10,8-11,2 pp** (t df=9 ; la règle 2,8×SE du protocole donne 11,16). **10 seeds sont conservés** ; l'effectif et ces chiffres corrigés sont la promesse de puissance finale préenregistrée.

## 2. Bootstrap : retour à UN niveau, claim CONDITIONNEL au benchmark (supersede `303dbd3` §bootstrap)

Les 4 prédicats à **poids égaux fixes** sont une **définition du benchmark**, pas un échantillon aléatoire de mélanges de tâches : rééchantillonner les prédicats répond à une question (généralisation du mélange) que le claim ne pose pas, et peut créer de l'incertitude artificielle (ex. +30/+30/−20/−20 par seed ⇒ moyenne +5 constante, le 2ᵉ niveau varie là où l'estimand est fixe). **Arbitrage final : bootstrap à UN niveau** — rééchantillonnage des **seeds comme blocs appariés** sur les agrégats par seed (moyenne égale des 4 strates, agrégation fixe) ; ≥10 000 rééchantillonnages, seed fixe 20260923 ; IC 95 % **unilatéral** = 5ᵉ percentile. **Le claim est explicitement CONDITIONNEL au mix de prédicats fixé** ; une généralisation à d'autres mélanges serait une expérience distincte. La partition du verdict (PASS/INDETERMINÉ/FAIL) est inchangée. Test synthétique exigé : cas +30/+30/−20/−20 (le 1-niveau doit refléter la stabilité de l'agrégat fixe) et comportement sous absence d'effet simulée.

## 3. Estimand d'évaluation clarifié (cellules ↔ épisodes)

Le seuil de masse ≥ 0,5 % définit le **sous-espace du claim de support d'ENTRAÎNEMENT** (gate §1 du protocole) ; il ne filtre **aucun épisode d'évaluation**. Le primaire = **succès d'épisode complet sur les 600 épisodes de test2, identiques pour tous les bras** (vérifié dans le code : évaluation intégrale, aucune admissibilité dépendante des trajectoires produites). La couverture du sous-espace par le tirage d'entraînement reste un **gate/diagnostic préalable**, pas un filtre d'évaluation. Le budget primaire/contrastes/pondérations sont ceux du protocole §3, reconstructibles depuis le freeze.

## 4. Champs de politique de checkpoint séparés

Deux champs distincts et compatibles : **source** = `best_validation` (checkpoints V0 canon/contrôles, inchangés) ; **adaptation cible** = `final_at_fixed_budget` (erratum `099af7f`). La contradiction apparente (revue 4, item 5) venait de la confusion des deux champs.

## 5. DAgger (annexe autonomie) — garde ajoutée

La règle de promotion (−30 % d'échecs-cycles OU +3 pts) doit **exclure** les gains obtenus en s'arrêtant plus tôt : promotion seulement si **aucune dégradation inacceptable du succès total ET des erreurs irréversibles** (seuils déclarés avec la règle).

Le présent erratum ne modifie ni l'estimand restreint du support d'entraînement, ni le k_plan, ni les seeds, ni la séquence vers test2. **HOLD test2 inchangé.**
