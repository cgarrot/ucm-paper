# RAPPORT FINAL DU CYCLE S2b→P2 + direction de fin de cycle (25 septembre 2026)

**Statut : rapport de clôture du goal `goal_d2ea9adcd654` (oracle PASS `244420d6…`) suivi de la direction de fin de cycle (revue tagi-review via tagi-ask) exécutée intégralement.** Toutes les sources sont des artefacts committés et pinnés ; les chiffres sont les moyennes 5 seeds (règle au registre).

## 1. Verdicts scientifiques du cycle

| Question | Verdict | Source |
|---|---|---|
| **S2b** : le calcul itératif (CIV T=32) bat-il le one-pass à budget égal sur les tâches profondes ? | **KILL** (−7,7 pp vs +15,8 préenregistré) — P1 one-pass ; asymétrie documentée (refine sur canon figé) | run3 20/20, `REPORT-S2B-FINAL` + erratum |
| **P2-1** : l'historique in-context améliore-t-il les décisions ? | **REQUALIFIÉ : non testé** — identifiabilité impossible par construction (contexte non pertinent) ; le FAIL mesuré = fragilité au contexte non pertinent ; **le modèle lit le contexte** (discriminant : sans 55,2 % vs avec 46,9 %) | erratum `a56cbac` + addendum `e2c1f99` |
| **§13.4 profondeur** : la calibration générale suffit-elle ? | **GO greedy** (+20,8 pp à p95 1,9 ms) — les données seules 46 % < 80 %, la recherche courte reste pertinente | Recherche A `9573acf` + erratum `d598ea7` |
| **Transfert compositionnel** : généralisation aux signatures tenues à l'écart ? | **GO intra-bande (97,0 %, +2,0 pp) / KILL strict (0,0 %)** — généralisation aux nouveaux layouts établie ; **transfert compositionnel strict NON établi** (la calibration ne suffit pas pour la composition) | Recherche B `f673050`+`5004dea` |

**Résultat gratuit** : la tête auxiliaire d'effet améliore le tronc de **+34 pp** (0,656 vs 0,312 natif sans tête) — Q4, à publier.

## 2. La table VISION (S5 tranche fine, chemin complet voix→Jev→UCM)

| Promesse VISION | Mesure | Statut |
|---|---|---|
| Compact (p95) | **1,36 ms** | ✅ (≤ 5 ms) |
| Bon marché | **1 027× moins cher** que Jev-par-pas (1 200 ms/décision) | ✅ (≥ 10×) |
| Sait dire « je ne sais pas » | **rappel ≥ 91 % / faux refus ≤ 7 %** (post-calibration v2 : paires discriminantes + refus non-initiaux) | ✅ (≥ 90 %) |
| Exécute sur non-vus | **97,0 %** (nouveaux layouts d\*2-12, +2,0 pp vs intra) / **0,0 %** (composition stricte d\*20-24) | ✅\* — **la composition stricte est le verrou suivant** |

\*La généralisation aux nouveaux layouts dans la bande d'entraînement est **établie** ; le transfert compositionnel strict **n'est pas établi** — les 60 tâches du KILL sont toutes AT_long_door (d\*20-24, traversées de porte longue distance jamais optimales dans le canon).

## 3. Ce que le cycle a bâti (réutilisable)

- **DSL générique vérifié** : interpréteur indépendant (zéro eval), BFS oracle, deux mondes natifs = deux instances, fermeture ≥10⁴ états all_equal, mutations 3/3 détectées.
- **Familles tenues à l'écart** : 63 TGK + 221 SIW signatures disjointes par construction (conservation 75/240 vérifiée), contrôle D1 planner kind-default.
- **Harnais P2** : 3 bras de contexte, tête d'effet conjointe, gate oracle-privilégié avec abort, runner O_EXCL avec RESUME — durci par 4 bugs traqués-gardés (STOP terminal, edge_type_emb, goal_refs, collate inversion).
- **Générateur d'impossibles constructifs** (24 épisodes, 2 classes) + **corpus de calibration du refus** (160 records, paires discriminantes).
- **Règles au registre** : §5.3 jamais baisser un seuil ; p_ref au budget FINAL ; contrôle négatif obligatoire pour tout gate ; secondaire exécuté ou déclaré en déviation ; chiffres = moyennes 5 seeds.

## 4. Leçons instrumentales (12+ bugs traqués-gardés sur les deux journées)

Chaque bug a produit une garde permanente avec contre-factuel **et** contrôle positif (l'exigence des deux directions est au registre depuis l'auditeur) : canon-non-chargé, optimiseur no-op, STOP terminal, embeddings hors-bornes, inversions de collate, launcher non-persistant, collision ckpt, boucle de réveil (4 pannes → boucle infinie loggée). La classe commune — **garde à cohérence interne sans contrôle du but** — est désormais fermée mécaniquement.

## 5. La direction d'après

- **La composition stricte est le verrou scientifique** (0 % sur signatures jamais-optimales) — pas la calibration (établie), pas la profondeur (greedy +20,8 pp suffit à ouvrir §13.4), pas l'in-context (gelé : effets latents par épisode + gate de valeur d'information ≥ 2×Δ_min).
- **Le produit S5 existe** : voix → Jev → but typé → UCM à 1,36 ms, 1 027× moins cher, refus calibré, exécution sur non-vus dans la bande. La démonstration de bout en bout (S5 complet avec la vraie voix et le vrai Jev) est l'étape produit suivante.
- **P1 one-pass confirmé** ; la profondeur attend §13.4 avec la baseline greedy ; la tête auxiliaire (+34 pp) mérite sa propre investigation.

**Portée : ce rapport clôt le cycle S2b→P2 et sa direction de fin. La composition stricte et la démo complète sont les prochains jalons — rien d'autre n'est autorisé par ce document.**
