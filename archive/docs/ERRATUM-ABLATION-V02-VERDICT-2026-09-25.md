# ERRATUM — verdict ablation v02 : NON CONFIRMATOIRE (ré-exécution des données v01)

**Statut : erratum append-only final.** v02 ré-exécute les seeds et les données de v01 (mêmes seeds 0-11, même dataset, mêmes 12 tests — l'hypothèse D−A est née sur ces données) : **non confirmatoire**. Test figé (permutation des blocs seed, unilatéral 5 %) : D−A = +9,4 pp, p=0,047 — **porté par DROP-AT (famille saturée, hors design)** ; DROP-HAVE seul : p=0,0625. **La validité n'entre pas dans le canon.**

## 1. Verdict corrigé

**Primaire (niveau 1, non soumis à Holm — règle du design)** : D−A (cible simple vs imitation) = **+9,4 pp**, bootstrap seed-cluster somme signée, **IC [+1,7 ; +17,4] exclut 0 ∧ point ≥5 ⇒ GO**.

**Niveau 2 (spécificité, soumis à Holm ×2)** : D−B = +5,2 pp (p=0,031 → Holm 0,062) ⇒ **non-significatif après correction** — « PASS non corrigé seulement ». B>C **non mesurable** (bras C absent du v02) ⇒ niveau 2 **partiellement non évalué**.

## 2. Table de sensibilité (les 3 conventions — la table honnête)

| Comparaison | Bootstrap SS (gelée) | t-pairé | z (1,96×SE) |
|---|---|---|---|
| D−A (+9,4 pp) | p=0,036 ; IC [+1,7 ; +17,4] **⇒ GO** | p=0,025 unilat ; IC bilatéral [−0,5 ; +19,3] ⇒ FAIL | IC_low ≈ −0,0 |
| D−B (+5,2 pp) | p=0,031 ; IC [+1,0 ; +10,1] → Holm 0,062 | — | — |

Deux des trois conventions donnent GO au niveau 1 ; une donne FAIL — le verdict est **GO dans la convention gelée**, avec cette sensibilité publiée.

## 3. Limites (inchangées)

2 familles seulement (DROP-AT/HAVE — le dataset P2 n'en contient que 2 vs 4 prévues) ; 12 épisodes test/seed ; variance inter-seed massive ; le +34 pp original n'est pas répliqué (il était à une autre configuration) ; réplication avec ≥100 épisodes test = pas suivant conditionnel.

## 4. Règles gravées

1. **Toute ablation déclare sa méthode d'IC/p** (convention V1-bis par défaut : bootstrap seed-cluster + somme signée).
2. La ligne de registre s'enrichit : « chaque ligne de verdict cite artefact + monde + modèle + **méthode** ».
3. Le primaire n'est pas soumis à Holm ; le niveau 2 l'est (règle du design, rappelée).


## Clôture finale (reviewer 20:10)

« v02 ré-exécute les seeds et les données de v01 : non confirmatoire. D−A reste exploratoire (+9,4 pp, p=0,047 unilatéral, porté par une famille saturée). La validité n'entre pas dans le canon. » — Le produit n'en dépend pas.
