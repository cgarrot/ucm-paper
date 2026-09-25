# RAPPORT FINAL V1-bis v2 — correction de lecture centrale (24 septembre 2026)

**Statut : supersed `docs/REPORT-V1BIS-FINAL.md` (`3777f40`) après oracle FAIL (décision `0a5636be…`).** Corrections intégrées de l'analyse indépendante tagi-review (recalcul depuis les 72 000 raws, adoptée intégralement) : (1) **bug d'IC supérieur** dans `verdict_v1bis` (ci_high = MAX bootstrap au lieu du 95ᵉ percentile — corrigé au module, artefacts du run intacts append-only) ; (2) requalification du verdict en **non-démonstration en régime saturé** ; (3) k primaire N=256, p unilatéral, résultat positif STOP-artefact-éliminé, signal k=64 exploratoire, queue de difficulté, critères de sensibilité P2. Sources inchangées : `artifacts/v1bis-eval/` (scellés, pointer `stage-b-eval-20260924T114942`).

## 1. Verdict corrigé : NON-DÉMONSTRATION en régime saturé (k primaire N=256)

**Le seuil de 5 pp était inatteignable pour tout bras dans ce régime** : scratch = 97,10 / 98,18 / 99,18 % aux k=64/128/256 ⇒ marge restante 2,90 / 1,82 / 0,82 pp, toujours < 5 pp. **Aucun transfert, même parfait, n'aurait pu PASS.** Borne du gain absolu possible à N=256 : **≤ +0,73 pp** (95ᵉ percentile corrigé). Formulation exacte : « non-démonstration en régime saturé, pas une preuve d'absence ».

Verdicts (partition figée inchangée, IC corrigés au **95ᵉ percentile** — recalculés par le module corrigé depuis les verdict-raw scellés, vérifiés par le lead) :

| k | rôle | diff (pp) | IC 95 % corrigé | p (bilat./unilat.) | Verdict |
|---|---|---|---|---|---|
| **256** | **PRIMAIRE (protocole §3)** | **−0,08** | [−0,78 ; **0,73**] | 0,84 / 0,42 | **FAIL** |
| 128 | secondaire | −0,1 | [−1,12 ; **0,92**] | 0,86 / 0,43 | FAIL |
| 64 | secondaire | +0,8 | [0,25 ; **1,33**] | 0,055 / **≈0,028** | FAIL (point < 5 pp) |

p unilatéral k=64 ≈ 0,028 (recalcul tagi-review ; seeds à diff nulle exclus — 0 à k=64, 2 à k=256). **Puissance : leçon enregistrée** — la SD 12,61 pp héritée de M-V1b (base ~29 %) n'était pas transposable à une base 97 % ; toute puissance future est conditionnée au niveau de base mesuré en DEV.

## 2. LE résultat positif du run : l'artefact STOP est ÉLIMINÉ

**`goal_reached_without_stop` = 0 sur 72 000 épisodes** (vérifié par le lead depuis les raws ; contre 29-44 % des épisodes en V1/M-V1b). La correction R* (trajectoires physiquement atteignables, terminal STOP à d*=0) a supprimé la pathologie de terminaison diagnostiquée à V1 : **le diagnostic V1 est confirmé par la réparation**. C'est la démonstration de validité de construit la plus directe du projet.

## 3. Signal k=64 — strictement exploratoire, jamais « presque significatif »

+0,8 pp, p unilatéral ≈ 0,028 (Bonferroni ×3 → ≈ 0,085), soit **−28 % d'échecs relatifs** (126 vs 174 échecs/6 000) ; s'inverse à k=128 (×1,06) et k=256 (×1,10). Les contrôles sont SOUS scratch à k=64 (96,2/96,4 vs 97,1 %). Formulation liée : « signal exploratoire sous le seuil de pertinence, non répliqué aux budgets supérieurs ; hypothèse pour un régime non saturé ».

## 4. Ce que personne n'avait vu (descriptif, pour P2)

- **La difficulté réelle ≈ 6 % du test** : la queue de 38 épisodes (VIEW d*≥3, CHOOSE d*≥4, SET d*=4, SUBMITTED d*=7) concentre le succès 15-86 % ; à k=256, les 12 pires portent 68 % des échecs — effectif utile ~38 épisodes corrélés.
- **90 % des timeouts = boucles absorbantes** (≥60 invalides/64 pas) — pathologie V0, pas un défaut de transfert. Sonde S1/S2b : corréler échecs × distance/sauts (hypothèse champ récepteur 3 sauts).
- **Prior STOP du pré-entraîné survit atténué** : 17 vs 7 STOP prématurés à k=64, 17 vs 2 à k=128, 4 vs 3 à k=256.
- **SIW-small est désormais un monde RÉSOLU** (le planificateur oracle y fait aussi ~100 %) : toute question de transfert y est sans objet — P2 doit utiliser les familles tenues à l'écart.
- À vérifier : les pilotes DEV v07x montraient-ils déjà scratch ≈ 97 % à k=64 ? Si oui, le manque était un **gate de sensibilité** ; si non, DEV non représentatif de test2 — dans les deux cas, leçon pour P2.

## 5. Critères de sensibilité P2 à préenregistrer (chiffrés, avant tout scellé)

1. Pilote DEV : **p_ref ∈ [20 ; 80] %** à budget déclaré (pour SIW-small, imposerait N ≤ 16-32 — ou changer de famille).
2. **Δ_min ≤ (100 − p_ref)/3**.
3. **SD mesuré en DEV à cette base** ; exiger 2,8×SE ≤ Δ_min, sinon +seeds AVANT scellé.
4. Difficulté par design : ≥ 50 % épisodes d*≥4, ≥ 25 % d*≥6, signatures tenues à l'écart, niveau D1.
5. Secondaire robuste au plafond : **log du ratio d'échecs** avec seuil propre.
6. S2b : gate de sensibilité équivalent (bras de référence 20-80 % sur la strate DEV, sinon changer de strate AVANT scellé).

## 6. Intégrité (inchangé) et reproductibilité

Chaîne inchangée par rapport à v1 : one-read consommé une fois, 72 000 raws avant agrégation, verdicts recalculés indépendamment, publication scellée, incidents documentés. **Correction d'instrument enregistrée** : `verdict_v1bis.py` ci_high 95ᵉ percentile (bug trouvé par l'analyse indépendante post-run, fixé pour l'avenir ; les artefacts scellés du run restent intacts — les valeurs corrigées vivent dans ce rapport).

**Portée : ce rapport v2 est la lecture finale du run V1-bis. Le FAIL préenregistré devient une non-démonstration proprement qualifiée en régime saturé ; le résultat démontré est l'élimination de l'artefact STOP ; P2 exige des cibles non saturées selon les critères §5.**
