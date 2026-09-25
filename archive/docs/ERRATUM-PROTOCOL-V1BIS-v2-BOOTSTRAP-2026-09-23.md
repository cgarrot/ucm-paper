# ERRATUM — protocole V1-bis v2 — clarification du bootstrap « hiérarchique »

**Statut : erratum append-only AVANT tout run V1-bis.** Le protocole `a1a8b9d` §3 exige un « IC 95 % bootstrap hiérarchique seed-cluster (≥10 000 rééchantillonnages) » sans définir le nombre de niveaux. L'audit du module d'analyse (tagi-5 18:00) a relevé l'ambiguïté : 1 niveau (seeds seuls, sur agrégats par seed) ou 2 niveaux (seeds puis strates de prédicats intra-seed). Aucune donnée V1-bis n'ayant été vue, l'interprétation est **clarifiée et verrouillée maintenant** :

- **Deux niveaux** : (i) rééchantillonner les **seeds** avec remise ; (ii) à l'intérieur de chaque seed tiré, rééchantillonner les **4 strates de prédicats** avec remise ; agrégat = moyenne pondérée égale des strates tirées, puis moyenne par seed ; IC 95 % **unilatéral** = 5ᵉ percentile de la distribution bootstrap des différences ; ≥10 000 rééchantillonnages, **seed fixe 20260923**.
- **Rationale** : conforme au design préenregistré « seeds = clusters, prédicats = strates intra-seed » ; le 1-niveau serait anti-conservateur (IC plus étroit) et risquerait un faux PASS.
- **Rappel de la partition du verdict (inchangée, conformité exigée)** : PASS = ci_low > 0 **et** point ≥ 5 pp ; INDETERMINÉ = IC contient 0 avec point ≥ 5 pp ; **FAIL = tout le reste** (y compris point < 5 pp même avec IC > 0, et ci_high < 0). INDETERMINÉ ≠ « aucun transfert ».

Le présent erratum ne modifie ni l'estimand, ni les seuils, ni le k_plan, ni la séquence. Toute divergence future = nouveau document. **HOLD test2 inchangé.**
