# RAPPORT FINAL V1-bis — Run confirmatoire officiel (24 septembre 2026)

**Statut : `official_complete`.** Chaîne complète auditée pinnée : protocole `a1a8b9d` (+ errata `20df839`, `1e7a214`, `303dbd3` supersede), freeze **v10** (`7b1df1a`, canon `0851ee9c…`, lead-built), test2 scellé seed `20261003` (600 épisodes, SHA `248e044f…` publié AVANT le run), run unique **4 bras × 10 seeds × k[64,128,256]** = 120 cellules (3 h 37, 4 workers, ~14,5 h CPU-cumulées), **lecture unique** du scellé consommée dans l'éval Stage-B (§9.4, horizon 64, STOP natif), 72 000 raws persistés avant agrégation, verdicts préenregistrés recalculés indépendamment (tagi-5, PASS de clôture), publication bundle/pointer `stage-b-eval-20260924T114942`. Sources : `artifacts/v1bis-eval/` (raws 21,5 Mo, métriques, verdicts par k), `artifacts/v1bis-run/` (dump, 120 checkpoints, merge scellé). **Aucune déviation du protocole, aucune sélection post-hoc, aucune ré-essai.**

## 1. Verdicts préenregistrés

Partition figée (PASS = IC95 unilatéral > 0 **et** point ≥ 5 pp ; INDETERMINÉ = IC∋0 ∧ point ≥ 5 pp ; FAIL = tout le reste) ; bootstrap 1 niveau conditionnel au mix fixe (10 000, seed 20260923), SD exploratoire 12,61 pp (t df=9) :

| k (épisodes) | diff pré-entraîné−scratch | IC 95 % | p (permutation) | **Verdict** |
|---|---|---|---|---|
| 64 | **+0,8 pp** | [0,25 ; 1,97] | 0,0547 | **FAIL** (IC>0 mais point < 5 pp) |
| 128 | −0,1 pp | [−1,12 ; 2,4] | 0,873 | **FAIL** |
| 256 | −0,08 pp | [−0,78 ; 2,08] | 0,867 | **FAIL** |

**Conclusion (langage Pivot B préenregistré)** : la source TinyGraphKey B144, les contrôles, le support R* restreint (cellules ≥ 0,5 %), le budget N=256 épisodes et le test scellé précisément pré-enregistrés **n'établissent pas de gain de transfert ≥ 5 pp dans ce régime**. Cela ne réfute ni le transfert par poids en général, ni l'adaptation avec mémoire de conséquences, ni d'autres régimes de difficulté. INDETERMINÉ ≠ « aucun transfert » ; ici les verdicts sont FAIL, pas INDETERMINÉ — l'effet le plus favorable observé (+0,8 pp à k=64) est statistiquement distingué de zéro mais très loin du seuil de pertinence pratique conventionnel préenregistré.

## 2. Le fait central : effet de plafond

Succès moyens par bras (4 prédicats confondus, 6 000 épisodes/cellule, 10 seeds) :

| bras | k=64 | k=128 | k=256 |
|---|---|---|---|
| scratch | **97,1 %** | 98,2 % | 99,2 % |
| pré-entraîné TGK | 97,9 % | 98,1 % | 99,1 % |
| contrôle validité-informé | 96,2 % | 98,0 % | 99,0 % |
| contrôle null | 96,4 % | 98,2 % | 99,3 % |

**Le scratch atteint 97,1 % dès k=64 et 99,2 % à k=256** : la famille de tâches SIW-small, à ce niveau de difficulté (bandes d*[2,8], 161 layouts test iso-disjoints), est quasi saturée par l'adaptation supervisée seule. La marge maximale disponible pour TOUT effet de transfert est ≈ 1-3 pp — observée bien en deçà du seuil de 5 pp. **Observation brute, pas interprétation** (recalculée depuis les raws par l'auditeur). Implication de design : toute expérience future sur le transfert dans SIW exige soit des tâches plus difficiles (d* plus profonds, layouts plus grands, bruit/observation partielle), soit des budgets d'adaptation réduits (régime k≤16 où l'écart bras existe encore : à k=64 l'écart max entre bras est ~1,7 pp), soit les deux — condition sine qua non pour que la question du transfert soit ouverte.

## 3. Constats secondaires (descriptifs, exploratoires)

- **L'écart entre bras n'existe qu'à k=64** (max 1,7 pp entre null 96,4 % et pré-entraîné 97,9 %) et disparaît à k≥128 (tous ≤ 99,3 %) — cohérent avec un effet de source qui accélère la convergence précoce puis se fait rattraper.
- **VIEW est le prédicat le plus difficile** dans presque toutes les cellules (0,88-0,99 selon seed/bras à k=64 vs 1,0 pour SET/SUBMITTED dans la plupart) — la navigation reste la composante non triviale.
- **Les contrôles (validité-informé, null) se comportent comme le scratch** aux incertitudes près — aucun signal d'attribution exploitable dans ce régime plafonné.

## 4. Intégrité et reproductibilité

- One-read du scellé : consommé une fois, ancre par pointeur publié, jamais recalculée.
- Raws : 72 000 lignes, schéma 14 champs, conditions identiques entre bras (600 episode_ids identiques), persistés avant agrégation.
- Déterminisme : cellules indépendantes (init appariée par seed, modèle frais par k), merge scellé par contenu.
- Budget : 3 h 37 wall × 4 workers ≈ 14,5 h CPU-cumulées (+ éval 4 min).
- Historique complet des incidents : 3 crashs de lancement, tous fail-closed AVANT consommation du scellé (wrapper store, k-cumulatif parallèle, relecture store par chunk), chacun corrigé avec test de régression non-vacuous et freeze re-construit (v7→v8→v9→v10) ; une perte de résultats par launcher défaillant (run v9 complété non persisté — faute lead, corrigée par persistance dans le runner). Aucun de ces incidents n'a touché les données scellées.

## 5. Ce que ce run établit et n'établit pas

**Établi** : dans le régime SIW-small testé (support restreint, N=256, 2 000 updates, batch 64, 10 seeds), l'initialisation par poids TGK B144 n'apporte pas de gain de succès ≥ 5 pp sur politique fermée ; le régime lui-même est quasi saturé (plafond 97-99 %). L'instrument (chaîne complète, one-read, freeze, verdict préenregistré) est validé de bout en bout et réutilisable.

**N'établit pas** : le transfert dans les régimes non saturés ; l'adaptation in-context ; les mondes plus difficiles (P2) ; la valeur du pré-entraînement à très petit budget (k≤16, non testé ici).

## 6. Suite du programme (décisions déjà arbitrées, `fc026a3` + errata)

- **S1/S2b** : récupération exploratoire, DAgger diagnostic (règle de promotion préenregistrée avec garde anti-arrêt-précoce), factorielle données-vs-calcul.
- **P2** : in-context intégré (historique structurel court, tête de prédiction d'effet en perte auxiliaire, oracle à information égale, test d'identifiabilité correct>absent>mélangé préenregistré) — **sur des cibles plus difficiles que ce régime** (exigence de sensibilité issue du plafond documenté ici ; l'analyse indépendante de tagi-review sur ce point est attendue avant gel du design P2).
- **S3** : récurrence seulement si le contraste données-vs-calcul le justifie.

**Portée : ce rapport clôt le run V1-bis. Il n'autorise rien d'autre. Tout résultat cité doit renvoyer aux artefacts scellés ci-dessus.**
