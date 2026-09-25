# ERRATUM — surventes du rapport de cycle (revue de clôture, 25/09 11:29)

**Statut : erratum append-only majeur.** Le rapport `REPORT-CYCLE-S2B-P2-FINAL.md` (`a1f4d0e`) contient 4 surventes identifiées par la revue de clôture (recalcul depuis les artefacts). Les formulations corrigées ci-dessous prévalent pour toute diffusion externe.

## 1. « §13.4 GO » → NON TESTÉ

Le « greedy » était `use_beam=False` (la **politique réactive**), pas une recherche ; le beam était cassé (0,0) ; la baseline `canon_zs=0.252` est **codée en dur** (mesurée sur d'autres tâches). Les +20,8 pp **répliquent le fine-tune de l'annexe** (25→45 % sur d*13-24), pas une recherche courte. Le facteur « données profondes » n'a jamais été isolé. **Formulation correcte** : « le fine-tune sur d*2-12 généralise à d*13-24 (25→45 %) ; la recherche courte (beam/H1-H2) reste à exécuter ».

## 2. « Le modèle lit le contexte » → À RETIRER

L'artefact du discriminant montre : contexte **correct** (0,375-0,469) = contexte **mélangé** (0,469) sur les mêmes poids — le modèle réagit à la **PRÉSENCE** d'arêtes de contexte (−8,3 pp), pas à leur **CONTENU** (0,0 pp de différence correct-vs-mélangé). **Formulation correcte** : « le modèle est **distrayable par la présence** d'arêtes de contexte ; la lecture du contenu n'est pas établie ».

## 3. Table VISION : la ligne exec est FAUSSE

Le **97 %** vient de **TGK** (autre monde, autre modèle — Recherche B) ; le chemin S5 **SIW** fait 0,43/0,30/0,41 vs planner 1,0 — **GO-1 FAIL ×3**. **Vrai score : 3/4** (compact ✅, coût ✅, refus ✅, exec ❌). Le « 1 027× » est un ratio de **latence vs Jev stub assumé 1 200 ms** — à formuler comme tel.

## 4. « Strict 0 % » : le verrou n'est PAS localisé

Une **seule** signature (AT_long_door), d\*20-24 — **trois facteurs croisés** (profondeur hors bande + signature unique + traversée longue) non départagés. **Formulation correcte** : « échec complet sur UN point du cartésien (AT_long_door d*20-24, exploratoire) ; les facteurs responsables restent à départager (H1 : profondeur vs signature ; H2 : raccourci but→rôle) ».

## 5. Ce qui reste défendable (confirmé par la revue)

1. **TGK 97 % sur layouts neufs** à 1-2 ms (préenregistré, GO)
2. **Fine-tune 25→45 %** sur d*13-24 (calibration générale)
3. ⭐ **La perte auxiliaire d'effet DOUBLE DROP-HAVE : 31→66 %** (7/8 seeds, p≈0,008 — **la vraie découverte du cycle**)
4. L'instrument (DSL, familles, gardes, 6 règles)
5. Le refus appris (1 classe, 160 records de calibration)

## Règle gravée

**Chaque ligne de verdict cite artefact + monde + modèle**, vérifiés mécaniquement — la classe « garde cohérente sans contrôle du but » a récidivé sur les surventes (formulation externe sans vérification de la source exacte).
