# RAPPORT FINAL V0 — Universal Control Model (TinyGraphKey)

**Date :** 22 septembre 2026 · **Statut :** V0 complète, chaîne de gates marquée et auditée
**Équipe :** tagi-1 (lead, WS-A env/oracle/génération), tagi-2 (WS-B données/splits/scellés), tagi-3 (WS-C modèles/éval/gates), tagi-4 (ops XMG, reproductibilité inter-machines), tagi-5 (audit indépendant)
**Matériel :** Apple M5 32 Go (principal) · XMG Debian/RTX 3070 (vérification secondaire, CPU only)
**Traçabilité :** chaque chiffre de ce rapport provient d'un artefact scellé référencé dans `artifacts/WS-C-artifact-index.json` (24 entrées + m4-inputs). Sources : commit sha entre parenthèses.

---

## 1. Réponse en une phrase

**Un contrôleur de 694 513 paramètres, élu par falsification sur une frontière capacité/coût mesurée, généralise à 97,5 % sur des mondes relationnels jamais vus, recombine à 99,4 % des compositions structurelles jamais entraînées, décide en ~0,5-1,2 ms sur CPU M5 — et sa seule limite identifiée est une profondeur de planification effective d'environ 12-15 actions, symptôme mécaniste connu, que seule la recherche (pas la taille) dépassera.**

## 2. Question et protocole

V0 demandait (spec §1.1, points 1-2) : (a) une petite politique peut-elle exécuter correctement des tâches en boucle fermée sur des instances nouvelles ; (b) à quel coût local. La méthode : gates préenregistrés, canon de données scellé, audit indépendant à chaque marquage, conséquences écrites avant les résultats.

- **Environnement :** TinyGraphKey (spec §4) — graphe 4-12 pièces, agent, clé, colis, porte verrouillable ; buts REACH/HAVE/AT ; actions MOVE/PICK/DROP/UNLOCK/STOP, invalides payantes ; horizon 64 ; oracle BFS exact.
- **Canon M0 (GATE-0, da9cf4a/9a19d8f4) :** 13 758 transitions / 2 599 épisodes ; pools train 200 / val 50 / test 127+163+81, extension G2 v2 +102 layouts (cfb48782, additive, v1 jamais muté) ; zéros initiaux train 10,0 % / test 0 strict ; prédicats 33×3 ; reproductibilité **byte-identique sur trois machines** (M5, XMG py3.13, replay API auditeur).
- **Budget consommé :** ≈ 4 h 50 d'accélérateur sur l'enveloppe 24 h ; entraînement+évaluation officiels : 86,3 min (durations.json, par seed) ; mémoire pic : entraînement RSS 749 Mo, inférence batch-1 RSS 67,6 Mo (mem-train/inference.json).

## 3. La chaîne de gates — chiffres officiels (tous reproduits à 1e-12 par l'auditeur)

| Gate | Verdict | Chiffre | Détail |
|---|---|---|---|
| **GATE-0** scellé | ✓ | sha `9a19d8f4` | 3 machines byte-identiques ; timings hors empreinte |
| **GATE-1** overfit ≥99 % | **A ÉCHOUE** — 0,9658 | A 366 337 p. | Trouvaille : limite 1-saut prouvée **au bit près** (logits 0,410357 == 0,410357 sur paire same-signature) |
| **GATE-5** architecture | **B144 élu** 4/4 | voir §4 | Critères figés avant runs ; 5 seeds ; audit exécute les mesures officielles |
| **GATE-2** généralisation | ✓ **0,9750** [0,9648 ; 0,9850] | 3 160 ép., 630 clusters, 5 seeds | REACH 0,988 / HAVE 0,997 / AT 0,940 ; regret 0,14 ; bras 4 greedy + 1 τ (choisis sur val uniquement) |
| **GATE-3** composition zéro-shot | ✓ **0,9935** [0,9860 ; 0,9988] | 494 ép. (2 skips documentés), 102 layouts, 2 470 lectures | Combinaison AT(clé, jonction) **jamais vue au train** ; regret 0,023 ; invalides 0,46 % |
| **M3** contrôles + descriptifs | ✓ | voir §6-7 | 5 contrôles, taxonomie, G3/G4, GATE-4 |

## 4. La courbe capacité/coût (GATE-5) — la contribution centrale

Strate d'échec préspécifiée (val, d\*≥3, layout-disjoint) ; coût = ratio intercalé soutenu officiel (protocole #12, auditeur, fenêtre calme) :

| Modèle | Params | Strate val | Coût officiel | Verdict |
|---|---:|---|---|---|
| A (Deep Sets 1-saut) | 366 337 | **0,257** (effondré) | référence 0,35-0,49 ms | capacité ÉCHEC |
| **B144 (GNN 3-hop, d=144)** | **694 513** | **0,890** (+63,1 pts, IC [56,8 ; 69,0]) | **1,921×** (IQR 0,31) | **ÉLU — seul point 4/4** |
| B160 | 856 161 | 0,912 (+65,3) | 2,035× (IQR 0,017) | coût ÉCHEC |
| B192 | 1 230 145 | 0,895 (+62,2) | 2,31-3,05× | coût ÉCHEC |

**Lecture :** +62-65 points de généralisation held-out s'acquièrent avec 3 hops de propagation ; dans cette famille et à ce protocole, la barre des 2× élimine tout ce qui dépasse ~700k paramètres. Sécurité d*≤2 : jamais dégradée (B144 0,997 vs A 0,841). Règle d'arrêt pré-engagée respectée — aucune itération au-delà de B144.

## 5. Ce que le modèle ne fait pas (contrôles M3, §9.1)

- **Sans but (tags et binding retirés) : 0,0** — il apprend le suivi de but, pas un réflexe
- **Candidats seuls : 0,0** — l'énumération ne résout pas (H1)
- **Sans relations : 0,0** — la structure est nécessaire (H3)
- **Contrefactuel : 0,960** — il suit le but affiché
- **Permutation G5 : 200/200** — invariance métamorphique au niveau épisode

## 6. Profil d'échec et baselines

- **83 échecs / 3 160 (GATE-2)**, taxonomie préspécifiée : 46 boucles (55 %) + 36 invalides répétées (43 %) + 1 autre. **Zéro** oubli-STOP, binding, mauvaise cible. 100 % = symptôme greedy+no-op identifié dès M1.
- **GATE-4 (non-vu) :** modèle **0,975** | heuristique R1-R6 **0,826** | random valide 0,606 | random syntaxique 0,046. Le modèle domine l'heuristique écrite de **15 points, précisément hors distribution**.

## 7. Trouvailles structurelles

1. **La profondeur qui généralise borne la planification.** G3 (9-12 pièces) : 0,945 global, dégradation monotone (1,0 à d\*≤3 → 0,25 à d\*12). G4 (d\*13-24) : 0,369, nul dès d\*≥16. Profondeur effective **~12-15 actions**, cohérente avec 3 tours de message passing + politique greedy sans recherche.
2. **Le déficit AT est un déficit d'horizon, pas de composition** (vérifié indépendamment) : AT stratifié = **1,000 à d\*3-5 (0 échec)**, 0,940 (6-8), 0,679 (9-11) ; tous les échecs AT à d\*≥6, L moyen 63 (timeouts) vs L\* 9,9. À horizon contenu, la composition réservée transfère à 0,994 (GATE-3).
3. **La limite 1-saut est informationnelle, pas d'apprentissage** : égalité bit-exacte des représentations sur paires same-signature (A), séparation nette (B144 : 16,13 vs −15,85).

## 8. Latence et mémoire (M5, CPU, protocoles §10.2 documentés)

- Inférence batch-1 : cellules calmes **0,52-0,66 ms** p95 ; intercalé soutenu **~1,04 ms** p95 médian (rampe thermique documentée sur les deux bras). Enveloppe §10.3 (≤20 ms) : **16-38× de marge**.
- Coût relatif B144/A : 1,92× soutenu, 1,41× en régime calme (les deux publiés, statistique officielle = soutenu).
- Mémoire pic : 749 Mo (train) / 67,6 Mo (inférence).

## 9. Limites V0 (déclarées)

Monde symbolique pleinement observable, une famille d'environnements, horizons courts, pas de mémoire/RL/langage (exclus par design). La généralisation mesurée est **au sein de la famille TinyGraphKey** (niveaux R1-R2 de la spec) ; G3/G4 sont descriptifs ; le pont logiciel (SIW) et le transfert inter-familles restent V1. Mineurs documentés : ordre de l'arbre de catégorisation, règle 1 à resserrer, seed du run descriptif, 2 épisodes dédup-mergés exclus (IDs publiés).

## 10. Leçons méthodologiques (registre PLAN §6, 14 entrées)

Les plus transférables : re-mesure par cellule après tout fix de distribution (#4) ; le harnais de verdict est du code à auditer avec fixtures bidirectionnelles (#7-8) ; décision figée = vérifiable mécaniquement, ambiguïté de mesure → barre la plus dure (#9) ; protocole à étages sur ensembles disjoints capacité/sélection/évaluation (#10) ; mesure contrôlée intercalée pour critères proches du seuil, symétrie binaire assumée (#12-13) ; la discipline est une propriété du processus, pas d'un rôle (l'exécutant a attrapé ce que l'auditeur avait manqué, et inversement, toute la journée).

**Sept interceptions avant contamination** : distribution 0 %-déséquilibrée, zéros en cellules test, sceau sur générateur obsolète, mapping harness T8, IC de verdict au signe inversé (×2), décisions figées non implémentées. Coût cumulé : ~1 h. Coût évité : chaque gate mesuré sur des données ou des verdicts faux.

## 11. Questions V1 ouvertes (justifiées par mesure)

1. **Dépasser la profondeur ~12-15** : recherche courte/world model (§13.4) — motivé par G4, pas par goût d'architecture.
2. **Symptôme greedy+no-op** : le bras τ le réduit déjà de moitié ; randomisation minimale à l'invalid = ablation déclarée.
3. **Pont logiciel SIW** (§12.6) : la question suivante de la roadmap — la recombinaison observée à GATE-3 suggère un terrain favorable, à tester.
4. **Langage naturel, mémoire (POMDP contrôlé)** : branches conditionnelles inchangées.

## 12. Répertoires

- Artefacts : `artifacts/` (index 24 entrées + m4-inputs ; chaque JSON auto-portant : protocole, seeds, per-seed, provenance).
- Scellés : canon `9a19d8f4` / manifest v1 `4d82ecaa` / v2 `cfb48782` / g2ext `f0563bfa` — jamais mutés, extensions additives uniquement.
- Registre : `docs/PLAN.md` §6 (14 leçons). Suite : **257 passed / 1 skip**, harnais d'audit intégrés en non-régression.

*V0 s'arrête ici par design : aucun ajout automatique de mémoire, RL ou world model. Les versions suivantes devront justifier expérimentalement chaque extension, exactement comme celle-ci a justifié sa profondeur.*
