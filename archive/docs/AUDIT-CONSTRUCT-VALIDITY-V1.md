# AUDIT — Validité de construit du run V1 exploratoire (HOLD V6/test2)

- **Demandé par** : lead (2026-09-23 10:02), suite au rapport reviewer (support adaptation initial-only, gradients post-action nuls, effet STOP, test exact p≈.094, contrôle non-informatif contesté).
- **Auteur** : tagi-5 (auditeur QA/red-team indépendant).
- **Portée** : **code + report-only** (`artifacts/v1/transfer-report.json`, sha256 `587d13b1…`). **Aucun fichier scellé ouvert.** Le rapport exploratoire reste la seule source de chiffres (IC auto-rapportés, non recalculables faute de raw).

## 1. Faits vérifiés (code)

**F1 — Support d'adaptation initial-only (structurel).** `sample_task` retourne `SIWState(init_view, frozenset(), {}, dialog, frozenset())` (`ucm/env/siw.py:600-608`) : `filled`/`chosen`/`submitted` **toujours vides**. Les couples d'adaptation (`ucm/data/siw_pipeline.py:205`), les buts d'inventaire (`:152`) et les épisodes test (`:240`) utilisent tous `sample_task` → **toute la distribution d'adaptation et de test est composée d'états initiaux** (les `state_key` scellés observés ont `chosen=[]` partout).

**F2 — Signal d'entraînement post-action nul (structurel).** Le fine-tune construit les labels sur l'obs d'**état initial** (`labels_from_supervision_siw(rec["policy_input"], sup)`), avec `optimal_actions` de l'oracle pour cet état. Aucun exemple post-action n'existe → **le signal de supervision est exactement nul** sur les colonnes de features jamais activées par un exemple post-action. **Nuance** : cela ne signifie PAS « poids inchangés » — AdamW (weight decay) peut faire évoluer ces poids même sans signal ; on parle de **signal nul**, pas d'invariance des poids (formulation à ne pas reprendre du reviewer telle quelle).

**F3 — Épisodes test initial-only.** Même sampler → la **1ʳᵉ décision** est dans la distribution d'entraînement ; les décisions suivantes sont **hors distribution** (aucune couverture d'adaptation).

**F4 — Contrôle « non-informatif » = validité-informé.** `make_control_records`/`control_label` tirent uniformément parmi les actions **physiquement valides** → la source contrôle apprend la **structure de validité** (préconditions), pas rien. Ce n'est pas un null pur « random-label » : l'attribution « contrôle ≈ pré-entraîné ⇒ aucun composant décisionnel » est **contestée par construction** (le contrôle peut transférer la validité).

## 2. Faits vérifiés (report-only, `cells`)

**F5 — L'effet est porté par le STOP, pas par la reach (agrégat).** Moyennes 5 seeds, k=500 :

| métrique (k=500) | scratch | pretrained-TGK | contrôle |
|---|---|---|---|
| success | 0.288667 | 0.414333 (+0.125667) | 0.422333 |
| premature_stop | 0.0553 | **0.1370** | 0.0743 |
| timeout | 0.6560 | **0.4487** | 0.5033 |
| invalid | 0.9306 | **0.8628** | 0.7456 |
| goal_reached_without_stop (grws) | **0.440333** | 0.312000 | 0.288333 |
| **goal_ever = success + grws** | **0.729000** | **0.726333** | 0.710667 |

Formulation exacte : le gain de succès prétrain−scratch **+0.125667** est **compensé** par la perte de reach-sans-STOP **−0.128333** ; la différence **goal_ever = −0.002667** → **reach quasi égale** (c'est `goal_ever`, pas `grws` seul, qui l'établit). Donc « **entièrement STOP** » est une lecture **au niveau agrégat** : ce run ne prouve pas le mécanisme individuel (décomposition par épisode impossible sans raw). À k=0 : pretrained/contrôle `premature_stop` = **0.8077 / 1.0000** (timeout 0.1923 / 0.0000) vs scratch 0.0000 (timeout 1.0000) → la source TGK pousse déjà au **STOP**.

**F6 — Test exact par seed : non significatif à 5 % (sensibilité post-hoc).** Diffs/seed à k=500 : `[14.33, 0.67, 25.17, 24.17, −1.50]`. Test de permutation apparié exact (2⁵) : **p unilatéral = 3/32 ≈ 0.094** ; **p bilatéral = 6/32 = 0.1875** ; t95 ≈ **[−3.09, +28.22]**. **Discipline** : ces calculs sont une **sensibilité post-hoc au niveau 5 seeds**, ils **ne remplacent pas rétroactivement** le critère bootstrap **pré-enregistré** (IC [2.56 ; 22.35], auto-rapporté, non recalculable). À cette puissance, **pas de signal distinguable** au seuil 5 % ; les deux lectures (bootstrap pré-enregistré et test exact post-hoc) doivent être publiées ensemble, sans substitution.

## 3. Évaluation de validité de construit

1. Le run mesure **des priors de première action + une politique de STOP/validité**, pas un transfert de politique multi-étapes : les états post-action du closed-loop sont **hors du support d'adaptation** (F1–F3).
2. Le différentiel de succès est **dominé par la décision STOP** (F5) ; la reach est similaire ou meilleure pour scratch.
3. Le contrôle **validité-informé** (F4) empêche l'attribution « exposition/optimisation vs décisionnel » d'être tranchée par ce run.
4. Le test exact par seed **n'est pas significatif** (F6, sensibilité post-hoc) ; l'IC bootstrap exploratoire (pré-enregistré) ne suffit pas à établir un signal à cette puissance.

**Verdict : HOLD CONFIRMÉ.** Le run V1 exploratoire **ne peut pas être promu** ; la confirmation V6/test2 **ne doit pas consommer les seeds** avant un protocole V1-bis corrigeant la validité de construit. Aucun gate, aucun GO.

## 4. Recommandation — protocole V1-bis (versionné, à figer avant génération)

1. **Support d'adaptation multi-étapes** : générer les couples le long de **trajectoires** (états post-action aux profondeurs 1..d, équilibrés avec les états initiaux) ; **pré-enregistrer la distribution du support** et **vérifier la couverture** de l'espace d'états du closed-loop (rapport de couverture par profondeur).
2. **Endpoints séparés et pré-enregistrés** : succès (STOP vérifié) **primaire**, + composantes secondaires **reach / stop correct / stop prématuré / timeout / invalide** ; interdire la sur-lecture d'une métrique unique ; rapport côte à côte.
3. **Contrôles déclarés** : (i) contrôle **validité-apparié** (uniforme parmi les valides, comme aujourd'hui) **et** (ii) contrôle **random-label pur** ; déclarer le construit de chacun et attribuer avec les deux.
4. **Statistique pré-enregistrée** : **test exact apparié** (permutation/sign) avec **puissance adéquate** (≥ 8–10 seeds, ou modèle hiérarchique + test exact) ; publier **p exact ET IC bootstrap**.
5. **Ablation ciblée** : entraîner sur les états post-action (ou mesurer l'accuracy post-action) pour isoler la part **première action** vs **multi-étapes** ; rapporter les deux.
6. **Discipline inchangée** : pré-enregistrement avant génération, un seul open, raw/checkpoints/sha, log isolé, aucun test scellé avant gel signé.

## 5. Signature

tagi-5 — auditeur indépendant. Audit **code + report-only** ; **aucun scellé ouvert** ; chiffres exploratoires auto-rapportés (IC non recalculables faute de raw). Le HOLD du lead est **justifié**.
