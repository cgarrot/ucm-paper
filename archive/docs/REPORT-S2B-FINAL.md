# RAPPORT FINAL S2b — factorielle données-vs-calcul (24 septembre 2026)

**Statut : verdict préenregistré appliqué — `docs/PROTOCOL-S2B-2026-09-24.md` (`83d5d588…`), run3 (ts 150529-164816), 20 cellules persistées, audit pinné PASS (tagi-5 17:20).** Sources : `artifacts/s2b-factorial/` (cellules ×20, ckpts ×20, summaries ×5, sondes). Historique : run1 invalidé (canon jamais chargé — bug corrigé `d5c88e8`), run2 invalidé (optimiseur subtree no-op — bug corrigé `b230307`), run3 = le verdict. Aucune donnée scellée SIW consommée.

## 1. Verdict : KILL de la récurrence CIV-simple pour P1

Prédicat gelé : point (moyenne des seeds) de CIV(T=32)−B144 ≥ +15,8 pp **ET** IC_low > 0, sur G4 d*∈[13,15], données oracle.

| Seed | B144-oracle | CIV(T=32)-oracle | Diff (pp) |
|---|---|---|---|
| 100 | 55,1 % | 51,3 % | −3,8 |
| 101 | 47,4 % | 47,4 % | 0,0 |
| 102 | 47,4 % | 42,3 % | −5,1 |
| 103 | 61,5 % | 28,2 % | −33,3 |
| 104 | 52,6 % | 56,4 % | +3,8 |
| **Moyenne** | 52,8 % | 45,1 % | **−7,7 pp** |

**FAIL net** (−7,7 ≪ +15,8). **Décision : P1 reste one-pass ; le CIV-simple (refine au-dessus d'un canon figé) est tué pour P1.** La profondeur relève de la recherche explicite §13.4 — pas d'une itération GRU légère au-dessus du one-pass.

## 2. Trois énoncés obligatoires (audit)

1. **Asymétrie documentée** : le CIV entraînait uniquement le refine sur un canon **figé**, le B144 adaptait tout — le test était **conservateur** pour la récurrence. Ce KILL tranche la question posée (refine sur canon figé) ; il **n'exclut pas** un design base-adaptée+refine, qui est une expérience différente à motiver séparément si la recherche profondeur la justifie.
2. **G1′ NON MESURÉ** : l'éval ne couvre que la strate G4 d*∈[13,15] (78 épisodes) — les bandes superficielles (d*≤8) ne sont pas dans le run. La condition « G1′ ≤ 2 pp » du protocole n'est **pas évaluable** ici ; le KILL tient sur le primaire seul. Tout futur usage de la récurrence devra mesurer la rétention.
3. **Sonde récepteur-champ dégénérée** : Spearman 0,125 sur la G4 seule — les buckets d* 1-3 sont vides (l'éval est une seule bande profonde) ; la sonde redeviendra informative sur une éval multi-bandes.

## 3. Constats secondaires (descriptifs)

- **Seed 103 : la loss converge** (2,52→0,053→0,004→0,014) — la chute de −33,3 pp est un **échec de généralisation** du refine sur l'éval profond, pas une instabilité d'optimisation (courbe publiée dans l'artefact).
- **Effet données (diagnostique, sans règle de promotion)** : oracle > recovery dans 4/5 seeds pour B144 (moyenne +6,4 pp ; +12,8/+10,3/+3,8/+11,5/−6,4 — seed 104 inversé) et 3/5 pour CIV (+6,4/+20,5/+6,4/−16,7/−7,7). La récupération f=0,3 qui REMPLACE de l'oracle **coûte** en moyenne sur cette strate — l'information oracle complète est la meilleure source à budget égal ici. Le seed 104 (inversion des deux bras) documente la variabilité inter-seeds ; l'analyse appariée avec IC (module verdict) peut être exécutée sur demande comme secondaire DESCRIPITF — le protocole n'en fait pas un endpoint.
- **Log-ratio (robuste au plafond)** : seed 103 rec-oracle échoue 71,8 % vs 38,6 % b144 — l'effet dramatique du refine sur ce seed est visible aussi en échecs relatifs.

## 4. Intégrité du run3

20/20 cellules, 20 ckpts seedés, 5 canons distincts vérifiés (s0-s4, sha par cellule), `refine_actually_trained` partout, courbes de loss publiées, budgets appariés (n=5492), RESUME actif. Les runs 1-2 invalides restent en place documentés (bugs : canon non chargé, optimiseur no-op — chacun corrigé avec garde + contre-factuel committés : `CANON NOT LOADED`, `REFINE NEVER TRAINED`).

## 5. Suite (annexe autonomie `fc026a3`)

- **P1 : one-pass maintenu.** La profondeur passe par la recherche explicite §13.4 si le programme la justifie — pas par le CIV-simple.
- **L'axe DONNÉES porte du signal** (oracle > recovery 4/5) : à qualifier en DEV sur la strate avant toute utilisation DAgger — la règle de promotion DAgger (§5 de l'annexe, avec garde anti-arrêt-précoce) reste préenregistrée pour son expérience propre.
- **P2 démarre** : familles tenues à l'écart (SIW-small résolu), DSL minimal (tests différentiels verts `650aa4a`), critères de sensibilité §5 du rapport V1-bis v2.

**Portée : ce rapport clôt S2b. P1 = one-pass. Rien d'autre n'est autorisé par ce document.**
