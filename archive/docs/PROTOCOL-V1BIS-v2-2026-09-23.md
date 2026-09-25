# PROTOCOLE V1-bis — version complète candidate (v2)

**Statut : CANDIDAT À GEL, PAS ENCORE AUTORISANT.** Cette version complète le brouillon `534545f` (resté sans champ fixé) avec les entrées DEV auditées suivantes : S0 baselines (`b02ea36`+`dcd00a6`, PASS), entrées puissance v02 (`83eea8d`+`2755348`, PASS, contraste proxy qualifié), conceptions de couverture v02 (`2654720`, PASS, recommandation C+A **non adoptée à ce jour**). **Aucune génération, scellement, ouverture ou exécution test2 n'est déclenchée par ce document.** L'autorisation exige : (1) le vert scientifique de ce protocole (audit indépendant), (2) le vert instrumental final du chemin scellé, tous deux sur commit propre. Seeds `20261003`/`20261004` et scellés historiques : intouchables. Toute correction future = erratum nouveau chemin.

## 1. Question et estimand

Tester, sur le régime SIW défini ci-dessous, si des poids pré-entraînés TinyGraphKey B144 améliorent la politique fermée après adaptation supervisée sur des trajectoires physiquement atteignables post-action (R* schema 0.8). Comparaison primaire : **pré-entraîné − scratch**, même architecture et tranche cible par seed, même seed d'initialisation avant chargement des poids source. Les contrôles ne servent qu'à l'attribution déclarée, jamais à une causalité non prouvée.

**Estimand restreint (arbitrage lead : conception C + A, de `2654720`).** L'espace des cellules est `prédicat × σ-vecteur × κ-ensemble × d*` ; le claim porte **uniquement sur les cellules de masse de référence ≥ 0,5 %**. Les queues < 0,5 % sont **déclarées exclues** du claim (formulation assumée : « transfert dans le sous-espace fréquent »). **Qualification obligatoire** : ce seuil est **response-to-exploratory** — les trous de couverture fine ont été observés avant son choix (`3caad49` et errata) ; il a été choisi sur les réfs TRAIN 1-3 et validé sur les réfs HOLDOUT 4-5 jamais vues au choix (holdout : 35/35 et 36/36 couvertes, 0 % manquant, 4-7 cellules exclues stables). Il est **verrouillé par le présent document** : aucun ajustement sur données futures, test2 compris. La signature fine reste publiée à titre descriptif ; la conception B (coarse) est **abandonnée** comme redondante et moins transparente.

**Gate de couverture prospectif (pré-enregistré).** Pour chaque seed d'entraînement du run officiel, sur le fichier réellement généré : masse manquante du tirage sur le sous-espace retenu **= 0 cellule de masse ≥ 0,5 % non couverte**, vérifiée avant tout entraînement ; échec ⇒ régénération selon la règle §3-bis ci-dessous (jamais adapté après examen des résultats).

**§3-bis — règle de régénération (pré-enregistrée).** En cas d'échec du gate pour un seed : régénérer avec le **seed de remplacement pré-déclaré** suivant de la liste du freeze (liste exhaustive ordonnée, un remplacement par seed au maximum) ; si l'échec persiste après le remplacement, **arrêt NO-GO technique** documenté — jamais de troisième tirage, jamais d'ajustement du seuil, du sous-espace ou de la liste. Chaque tirage et son verdict de gate sont journalisés (SHA du fichier, masse manquante, cellules manquantes) avant l'entraînement.

## 2. Bras, données et budget

Quatre bras : **scratch**, **source TGK B144** (checkpoints canon committés, hashes dans le freeze), **contrôle source validité-informé** (random-valid), **contrôle source null/syntaxique**. Même architecture, updates, batch, optimiseur, règle de checkpoint et exposition cible pour tous ; l'information reçue par chaque contrôle est publiée. Trois niveaux de planificateurs (oracle/available/appris) restent des comparateurs **étiquetés**, pas des bras d'attribution.

**Données d'adaptation** : épisodes R* schema 0.8, départs `d*(s0)∈[2,8]`, terminal STOP à `d*=0`, multiplicité inter-épisodes conservée, ordre round-robin prédicat via les helpers partagés (`rstar_order.{round_robin_episodes, serialize_rr_records, write_rr_couples_file}`). **Budget N = 256 épisodes entiers** (conception A : N=256 couvre 37/39 sur la réf représentative ; le gate §1 confirme sur le tirage réel). Le plan d'exécution k est `[64, 128, 256]` épisodes (préfixes imbriqués, ordre exact lié au freeze par `k_plan_exec_order`). Nombres de records publiés par seed (attendu ~2-3× N selon d* moyen).

**Génération** : seeds de génération **nouveaux** (jamais utilisés dans les pilotes DEV 13/20261020-23 ni les réfs 20261013-17), déclarés dans le freeze avant génération. Réservoir de layouts : même profil `small`, pools iso-disjoints du test.

## 3. Endpoints, inférence et règles de décision

- **Primaire** : succès fermé (STOP vérifié sur état natif) à **N=256**, différence pré-entraîné−scratch en points de pourcentage, agrégée par seed (moyenne sur les 4 prédicats, poids égaux), **IC 95 % bootstrap hiérarchique seed-cluster** (≥10 000 rééchantillonnages) **et** test exact de permutation des signes par seed en sensibilité.
- **Secondaires séparés** (publiés, jamais substitués au primaire) : `goal_ever`, STOP correct/prématuré, timeout, invalidité (avec dénominateur), longueur, par prédicat et par k du plan.
- **Puissance (formelle, seed-cluster)** : entrée sd = **10,9 pp** (différences par seed M-V1b, exploratoire, déclarée comme telle — le proxy S0 6,9 pp est publié mais non transposé). Avec les formules 2,8×SE (80 %) / 3,24×SE (90 %) : MDE 10 pp ⇒ **10 seeds (80 %)** ; MDE 8 pp ⇒ 15 seeds. **Décision : 10 seeds d'entraînement** (unité indépendante), effectif figé ici ; **seuil de PASS = IC 95 % unilatéral > 0 ET point ≥ 5 pp** — le plancher de 5 pp est un **seuil de pertinence pratique conventionnel, pré-enregistré** ici, non dérivé du calcul de puissance (qui vise ~10 pp) ; indéterminé si IC contient 0 avec point ≥ 5 pp ; FAIL sinon. Multiplicité : primaire unique, secondaires non corrigés et étiquetés exploratoires.
- **Interdits** : analyses intermédiaires décisionnelles sur le test, choix post-hoc de sous-groupes, correction rétroactive de l'IC primaire. Raw par épisode publié intégralement avant toute agrégation.

## 4. Instrument (déjà vert en DEV — à confirmer sur chemin scellé)

Chaîne audtée pinnée : `read_once` (1 FD/scellé, SHA attendu, fail-closed) → `from_read_once` (canonicalité) → `preload_store` (1×) → `finetune_episode_budget_preloaded` (N épisodes entiers, préfixes SHA-256 64 hex == octets writer RR, entraînement sans slicing) → eval closed-loop §9.4 (raw par épisode complet, eval_seed séparé) → publication bundle/pointer v9 (probe, no-replace, recovery fail-closed, phase 3 kill process-crash PASS) → freeze-manifest 29 modules avec cross-check manifest↔args et `k_plan_exec_order`. **Le vert instrumental final portera sur le chemin scellé réel** (génération test2 incluse), sur commit propre, avec les trois refus adversariaux (non-hex, freeze mismatch avant open, registry mismatch après exactement 1 open).

## 5. Séquence vers le run (aucune horloge, que des conditions)

1. Vert scientifique de CE document (audit indépendant tagi-5, octets pinnés).
2. Génération des fichiers d'adaptation (seeds nouveaux) + vérification du gate de couverture §1 par tirage ; publication des SHA complets.
3. Freeze-manifest committé (code 29 modules + inputs + plan + seeds) ; vert instrumental final sur le chemin complet.
4. **Alors seulement** : génération/scellement test2 iso-disjoint (seeds 20261003/04 ou nouveaux déclarés), exécution unique 4 bras × 10 seeds × plan k, verdict préenregistré appliqué tel quel.
5. En cas de NO-GO ou d'indéterminé : publication honnête, report, jamais de ré-essai silencieux.

**Checklist de gel** : [x] estimand restreint + seuil qualifié ; [x] N=256 et plan k ; [x] 10 seeds + puissance formelle ; [x] seuil PASS/FAIL/indéterminé ; [ ] audit scientifique vert ; [ ] fichiers générés + gate couverture par tirage ; [ ] freeze committé ; [ ] vert instrumental final. Tant qu'une case est vide : **HOLD intégral**.
