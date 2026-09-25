# PLAN V0 — Implémentation TinyGraphKey (source de vérité opérationnelle)

**Date :** 22 sept 2026 · **Équipe :** @tagi-1 (lead, WS-A), @tagi-2 (WS-B), @tagi-3 (WS-C), @tagi-4 (ops XMG/RTX3070, reproductibilité inter-machines — CPU only par défaut, spec §10.3), @tagi-5 (QA/red-team indépendant, audits fuites/bugs/tests)
**Spec de référence :** [`universal-control-model-project-spec.md`](../universal-control-model-project-spec.md) — en cas de conflit, la spec gagne ; toute déviation doit être documentée dans ce fichier, section « Déviations ».

**Environnement :** venv `.venv/` (Python 3.12.13, MLX 0.32.2, numpy 2.5.3, pytest). Toujours utiliser `.venv/bin/python` et `.venv/bin/pytest`. macOS 26.6.2, Apple M5 32 Go.

**Budget :** enveloppe V0 = **24 h cumulées d'accélérateur**. Chaque entraînement/profiling journalise sa durée dans `artifacts/`. Si projection dépassée → stop + révision du plan, pas des exigences.

---

## 1. Jalons (mappés sur les gates de la spec)

| Jalon | Contenu | Gate spec | Critère de sortie |
|---|---|---|---|
| **M0** | Env + oracle + schéma + génération + tests verts | GATE-0 | 10k transitions générées, rejouées déterministiquement, hash du protocole publié |
| **M1** | Modèle A entraîné sur 100 épisodes (overfit) + profiling batch 1 | GATE-1 | ≥99 % actions optimales sur le set de diagnostic ; p95 modèle ≤20 ms |
| **M2** | Pilote 1k/5k, sélection sur validation, confirmation G1 (5 seeds) | GATE-2 | G1 succès ≥95 % (IC bas ≥90 %), chaque type de but ≥85 % |
| **M2b** | Confirmation G2 (composition réservée) | GATE-3 | G2 ≥80 % avec définition scellée avant entraînement |
| **M3** | Contrôles (sans but, contrefactuel, candidats seuls, sans relations, permutation) + G3/G4 descriptifs + 20 échecs catégorisés | — | Tableau baselines complet + rapports |
| **M4** | Premier rapport benchmark + décision documentée | — | Verdict : continuer / corriger / tester B / arrêter |

---

## 2. Segmentation des workstreams

### WS-A — tagi-1 (lead) : environnement + oracle (chemin critique)
- `ucm/env/tinygraph.py` : monde TinyGraphKey (§4 spec), transitions, invalides payantes, STOP, horizon 64.
- `ucm/env/oracle.py` : BFS inverse multi-source, d*/L*/A*/reachable.
- `tests/test_env.py`, `tests/test_oracle.py` : Bellman exhaustif sur petits layouts, cas manuels clé/colis, STOP, invalides, replay déterministe, permutations cohérentes.
- `ucm/data/generate.py` (générateur de layouts) — fourni à WS-B.
- **DoD M0 :** pytest vert ; oracle vérifié Bellman sur tous les états de layouts ≤6 pièces ; 10k transitions reproductibles seed-contrôlé.

### WS-B — tagi-2 : données, splits, inventaire
- `ucm/data/schema.py` : dataclasses + validation canaux `policy_input`/`supervision`/`provenance` (§5 spec), `schema_version=0.2`.
- `ucm/data/writer.py` / `reader.py` : JSONL, dedup avec provenance conservée, contenu identifié par hash si gros.
- `ucm/data/splits.py` : hash structural préfiltre + isomorphisme exact ; pools 200 train / 50 val / ≥100 par test ; réservation G2 `AT(clé, jonction deg≥3)` ; manifest scellé (seeds, hash).
- `ucm/data/inventory.py` : inventaire de faisabilité — existence de tâches d*13–24, comptage couples uniques, présence des composants G2 au train. **Avant tout gel.**
- **DoD M0 :** splits reproductibles, zéro layout isomorphe entre pools, manifest avec hash ; inventaire publié (`artifacts/inventory-M0.json`).

### WS-C — tagi-3 : modèle, entraînement, évaluation
- `ucm/model/tensorize.py` : observation → tenseurs (contrat §5 du présent plan) + tests binding/permutation sur fixtures synthétiques. **Obligatoire (audit tagi-5 T8/§51.3) : randomisation de l'ordre entités+relations+candidats en train ET test, avec preuve par test G5 — biais mesuré sinon : 1er candidat optimal 21,9 % vs 12,5 % uniforme, room_0 racine centrale.**
- `ucm/model/deepsets_a.py` : Deep Sets local d=128 (§6.1 spec) en MLX ; comptage exact des paramètres affiché.
- `ucm/model/loss.py` : set-valued BC `-log Σ p(A*)` (logsumexp stable, masque padding uniquement).
- `ucm/model/train.py` : AdamW 3e-4, wd 1e-4, batch 64, clip 1.0, FP32, ≤10 époques / 10k updates ; checkpoint + config + métriques dans `artifacts/<run>/`.
- `ucm/eval/rollout.py` : boucle fermée, évaluateur indépendant (STOP vérifié sur état natif), invalides et timeouts comptés.
- `ucm/eval/baselines.py` : random syntaxique, random valide, heuristique locale, recherche exacte (borne, via oracle), kNN retrieval.
- `ucm/eval/metrics.py` : succès, regret, but-atteint-sans-STOP, taux invalides, IC bootstrap apparié hiérarchique.
- **DoD M1 :** overfit 100 épisodes ≥99 % + timing p50/p95/p99 batch 1 (MLX lazy : forcer `mx.eval`, warm-up 50, ≥1000 décisions mesurées).

---

## 3. Contrats d'interface (figés ; changement = entrée dans « Déviations »)

### 3.1 Environnement (WS-A fournit, WS-B/C consomment)
```python
Action = tuple[action_type: str, arg_ref: str|None]   # MOVE/room, PICK/obj, DROP/obj, UNLOCK/door, STOP/None
Obs = dict  # policy_input SEULEMENT : entities[], relations[], goal, candidates[]

class TinyGraphKey:
    def __init__(self, layout: dict, horizon: int = 64): ...
    def reset(self, task: dict) -> Obs
    def observe(self) -> Obs
    def candidates(self) -> list[Action]        # énumération goal-blind, K = R + 6
    def execute(self, action: Action) -> dict   # {"valid": bool, "terminal": bool, "result": str}
    # CÔTÉ ÉVALUATEUR UNIQUEMENT (jamais dans policy_input) :
    def goal_satisfied(self) -> bool
    def state_hash(self) -> str                 # état physique canonique
```

### 3.2 Oracle (WS-A)
```python
def solve(layout: dict, state: dict, goal: dict) -> dict
# {"d_star": int, "L_star": int, "optimal_actions": list[Action], "reachable": bool}
```

### 3.3 Schéma record (WS-B, §5.2 spec) — JSONL, une ligne par transition
```json
{"schema_version": "0.2", "policy_input": {"goal": ..., "entities": ..., "relations": ..., "candidates": ...},
 "execution": {"action_ref": ..., "observable_result": "valid|invalid", "next_state_hash": ...},
 "supervision": {"optimal_actions": [...], "d_star": ..., "reachable": ...},
 "provenance": {"layout_hash": ..., "state_goal_hash": ..., "split": ..., "source": "oracle", "generator_version": ..., "oracle_version": ...}}
```

### 3.4 Tenseurs (WS-C)
```python
# nodes:      [N, d_in]   float32  (types+attributs, one-hot fermé + flags rôle-but calculés DANS le modèle)
# edges:      [E, 2]      int32    (indices localisés, jamais IDs bruts) + edge_types [E]
# goal:       predicate_id int + ref_indices [2]
# candidates: type_ids [K] + arg_node_indices [K, 2] (=-1 si absent) + pad_mask [K]
# labels:     optimal_indices [K] (0/1) — usage loss UNIQUEMENT
# output:     logits [K]
```

### 3.5 Anti-fuite (vérifié par tests des trois WS)
- Aucun champ interdit (§5.1 spec) dans `policy_input` : pas de d*, reward, succès, timestep, plan, next-state, IDs de générateur.
- Permuter les IDs d'un layout ⇒ mêmes prédictions (à tolérance numérique près), labels et candidats permutés ensemble.
- Changer `supervision`/`provenance` à `policy_input` inchangé ⇒ bit-identique en tensorisation.

---

## 4. Protocole de coordination

1. **Room mesh `try-agi`.** Chaque agent réserve ses fichiers avant édition : `mesh_reserve` sur son sous-arbre ; conflit → négocier sur le mesh.
   - tagi-1 : `ucm/env/`, `ucm/data/generate.py`, `tests/test_env.py`, `tests/test_oracle.py`, `PLAN.md`
   - tagi-2 : `ucm/data/` (sauf generate.py), `tests/test_data.py`, `artifacts/inventory*`
   - tagi-3 : `ucm/model/`, `ucm/eval/`, `tests/test_model.py`, `tests/test_eval.py`
2. **Commits git** : petits, message `WS-X: description`. Pas de push. Le lead (tagi-1) merge/intègre.
3. **Artifacts** : tout run crée `artifacts/<date>-<nom>/` (config, seeds, métriques, timings, checkpoint). Rien ne vit seulement dans un terminal.
4. **Bloqué ?** → mesh dans la room + marquer la tâche `blocked` dans le goal partagé (tagi-1 tient l'arbre TODO).
5. **DoD d'une tâche = tests verts + section dans ce plan cochée + commit.**

## 5. Plan premier jour

1. tagi-1 : WS-A env+oracle+tests (M0 critique).
2. tagi-2 : schema+writer/reader sur fixtures (peut démarrer sans env via fixtures du contrat 3.3), puis brancher generate.py quand WS-A livré.
3. tagi-3 : tensorize+deepsets_a+loss sur tenseurs synthétiques + tests, puis intégration données réelles M0→M1.

## 6. Déviations et décisions (registre)

1. **layout_hash = identité étiquetée** (décision WS-A, 22/09) : `Layout.layout_hash()` hache la forme canonique étiquetée (pièces triées + arêtes triées + porte). Il identifie l'instance de provenance, PAS la classe d'isomorphisme — la détection d'isomorphisme (préfiltre + vérification exacte) reste du ressort de WS-B `splits.py`, conformément au plan initial. Les tests d'env documentent ce contrat.
2. **Ordre des candidats** : l'env fournit l'ordre canonique stable ; la randomisation d'ordre (train ET test, spec §8.2) se fait en aval (tensorisation), pas au stockage — les records restent permutation-safe via `action_ref` = index.
3. **Scan §5.1 = clés uniquement** (audit tagi-5 m11) : le schéma WS-B rejette les clés interdites et leurs variantes de nom, mais pas des VALEURS comme `"reward"` dans un champ libre ; la neutralisation effective est la tensorisation ID→index + la preuve G5. Le scan de types (aucun int/float dans policy_input) est côté env/tests WS-A.

4. **Re-mesure par cellule après tout fix de distribution** (tagi-5, 22/09) : deux bugs de distribution peuvent se masquer — le quota zéro supprimé partout (`int(n_ep*0.1)=0` par layout) cachait la pollution des cellules test_g1/g4 par les zéros ; la pollution n'est devenue visible qu'après avoir mesuré PAR CELLULE. Règle : toute correction de distribution doit être re-mesurée par cellule avant scellement.

5. **GATE-5 : métrique primaire = succès boucle fermée** (tagi-1, 22/09) : le critère strate-d'échec avait été ancré à tort sur l'optimal-action rate (diagnostic) au lieu du succès d'épisode (primaire §9.2) ; calibré sur l'ancien canon invalide, il était devenu insatisfaisable (plafond strate +1.0 pt). Re-préspécifié AVANT le run officiel : B−A ≥ +5pts succès fermé sur la strate préspécifiée (d*≥3 ∪ ties), IC95 >0, perte ≤2pts d*≤2, coût ≤2×, séparation des ties. Les dev-numbers (seed 2001) vues avant figeage sont documentées dans le rapport.
6. **Le harness de mapping doit être testé comme l'architecture** (tagi-3, 22/09) : l'invariance de permutation du modèle était prouvée, mais ModelPolicy indexait la liste mélangée en retournant l'action canonique → boucle fermée effondrée à 3 %. Leçon T8 : chaque transformation aval (ordre, mapping, décodage) exige son propre test d'invariance, pas seulement le réseau.

7. **Le harnais de verdict est du code à auditer** (tagi-5, 22/09) : l'IC du GATE-5 avait le signe inversé — verdict exactement inversé dans les deux sens (B supérieur échouait, B inférieur passait), attrapé par repro pré-run sur checkpoints dev. Règle : tout code qui décide PASS/FAIL exige (a) un audit dédié, (b) des tests fixtures dans les DEUX directions (supériorité et infériorité), (c) une re-passe après tout correctif. Le run officiel en vol au moment de la découverte est void et relancé sur harnais corrigé.

8. **Vérifications de cohérence interne dans les harnais de verdict** (tagi-3, 22/09) : une seconde inversion de signe (A−B vs B−A) dans le bootstrap GATE-5 a été auto-détectée par incohérence interne (`diff_pts` vs `point_diff`) avant toute conclusion. Pratique généralisée : tout harnais qui calcule une différence exposer une identité de contrôle croisée (point estimate vs somme des composantes, signe vs direction attendue sur fixtures). Règle inchangée : correctif en vol ⇒ re-passe indépendante obligatoire avant marquage.

9. **Une décision figée ack'ée n'existe pas tant qu'elle n'est pas vérifiable mécaniquement** (tagi-5, 22/09) : les décisions 12:23 (best-val, 5 seeds, bootstrap 3 niveaux, coût symétrique, bande stricte) étaient acquittées mais non implémentées — le 4/4 GATE-5 officiel était un artefact de protocole (in-sample, 1 seed, coût asymétrique A non compilé). Règle : chaque décision figée s'accompagne d'un champ d'artefact ou d'un test dédié qu'un vérificateur peut contrôler sans lire les intentions (comme m8 pour les splits). Deuxième règle : face à une ambiguïté de protocole de mesure (raw vs compilé), figer vers la barre la PLUS DURE — elle ne peut que rendre le passage plus difficile.

10. **Protocole à étages : capacité / sélection / évaluation sur trois ensembles disjoints** (tagi-1, 22/09) : après trois contaminations traquées (in-sample GATE-5, best-val écrasé M6, val réinitialisée M7), le GATE-5 v2 figé sépare structurellement : entraînement = 100 épisodes diagnostiques du canon (overfit) ; sélection = meilleur OA TRAIN (jamais val ni test) ; évaluation de promotion = strate VAL d*≥3 (pool layout-disjoint, 149 ép.) ; confirmatoire de généralisation = G1/test_g1 en M2. Capacité in-sample (GATE-1), promotion held-out (GATE-5), confirmation scellée (GATE-2) — trois questions, trois ensembles, zéro recouvrement.

11. **Complétion du figeage coût : ratio best-device soutenu** (tagi-1, 22/09) : le figeage 12:32 ne fixait pas le device — découvert honnêtement (les deux régimes publiés sans cherry-picking). Règle complétée AVANT calcul du ratio : coût = p95(B, meilleur device stable) / p95(A, meilleur device stable), ≤2×, les 4 latences absolues soutenues publiées (≥1000 décisions, 5 répétitions, médiane p95, thermique stationnaire). Justification : un déploiement rationnel exécute chaque modèle dans son propre meilleur régime (§10.2 mesure CPU et GPU de toute façon ; l'enveloppe absolue §10.3 reste exigée séparément). Transparence : règle complétée après visionnage des deux devices — justification de déploiement, application auditée par tagi-5. Si le ratio reste >2× : échec du critère 3, conséquence pré-annoncée inchangée (documenter + itération B-small en run figé neuf).

12. **Mesure contrôlée intercalée pour critères proches du seuil** (tagi-1, 22/09) : le ratio coût B/A (règle #11, CPU/CPU) donne 7/8 mesures > 2 avec variance 1.83-2.54 — indécidable en l'état. Protocole figé AVANT exécution : device = meilleur régime des deux bras (CPU), deux bras compilés, 6 blocs intercalés A/B (6 segments de 1000 décisions, warm-up 50 par segment, p95 = médiane des segments du bras), statistique de décision = médiane des ratios des blocs 2-6 (bloc 1 = rampe thermique, publié), verdict binaire ≤2.00/>2.00, fenêtre calme coordonnée sur le mesh, exécution par l'auditeur (tagi-5), exécutant sans compute pendant. Correction de prémisse enregistrée : B-CPU (1.16 ms) est meilleur que B-GPU (1.85 ms) — lecture initiale du lead erronée, corrigée par l'auditeur.

12b. **B160 = itération V0b légitime, B192 documenté** (tagi-1, 22/09) : le critère coût (règle #11, best-device soutenu) échoue pour B192 (d=192, 1,23M) sur trois mesures convergentes >2× (3,05 / 2,31 / médiane indépendante ~2,15 ; la lecture 0,917 était une 1re répétition pré-throttle). Conséquence pré-annoncée exécutée : B160 (d=160, 856 161 params — déviation déclarée à la bande §6.2 1,1-1,5M, motivation coût) passe GATE-5 4/4 en run figé neuf (strate VAL +65,3 pts, coût 1,41× cells stables). La statistique officielle du coût pour le modèle promu reste le protocole intercalé contrôlé (#12), à exécuter par l'auditeur en fenêtre calme. B192 reste publié comme candidat capacité-supérieure-coût-échoué : le contraste A/B160/B192 (366k/856k/1,23M) est une donnée scientifique du projet.

13. **[RÉSOLU 15:50] Unification de la statistique coût + règle d'arrêt de l'itération V0b** — verdicts finaux officiels (tagi-5, protocole #12, fenêtre calme) : B160 **FAIL 2.035** (5/5 blocs >2.00, IQR 0.017 ; 4/5 mesures soutenus convergentes >2) → archivé comme point de courbe ; **B144 PASS 1.921** (médiane blocs 2-6, IQR 0.309 publié, corroboré exécutant 1.913) → candidat GATE-5. Symétrie de la règle binaire assumée : échec à 1.7 % près accepté (B160) ET passage à dispersion large accepté (B144), aucune re-mesure. Leçon originale : le lead a figé deux statistiques de coût incompatibles (#11 best-device calme = 1.41× PASS ; #12 intercalé soutenu = 2.11× FAIL) sans réconcilier les régimes mesurés — erreur assumée. Résolution : la statistique OFFICIELLE est l'intercalé soutenu (#12), car la spec §10.2 exige le segment soutenu pour détecter le throttling (régime de déploiement d'une boucle de contrôle) ; les cellules best-device calmes restent diagnostics. Conséquences : B160 échoue officiellement le coût ; B144 (d=144, ~690k) est lancé en run figé neuf avec règle d'arrêt PRÉ-ENGAGÉE : échec capacité OU coût ⇒ fin de l'itération (pas de descente infinie), livrable = constat structurel 'aucun point testé de cette famille ne satisfait simultanément capacité et coût-≤2×-soutenu' + courbe complète A/B144/B160/B192 ; la révision du point de référence du critère (ratio vs A inadéquat / enveloppe absolue / ratio vs heuristique) est une décision de protocole V1, jamais rétroactive. Séparation des rôles : l'auditeur exécute les mesures officielles.

14. **« label » singulier retiré des interdits §5.1 pour SIW** (tagi-2, 22/09) : attribut UI observable légitime (§12.6 de la spec révisée) tiré d'un vocabulaire fermé sans lien sémantique avec le rôle — indépendance label↔kind vérifiée empiriquement (tagi-5: 0/71) et structurellement (permutation labels ⇒ hash identique, tests §8b). « labels » pluriel et y_true restent interdits.

15. **Dualité certificat canonique / test d'isomorphisme sous symétrie** (tagi-2, 22/09) : les distracteurs SIW créent des orbites symétriques factorielles qui font exploser le certificat canonique exact (~35 widgets). Architecture corrigée : hash invariant (WL typé) en préfiltre + test exact par backtracking refinement — les orbites rendent le test trivial là où le certificat explosait. Leçon : sous forte symétrie, préférer la paire (hash préfiltre, test exact) au certificat canonique unique.

16. **SIW : l'irréversibilité SELECT est l'unique piège de joignabilité** (tagi-2, 22/09) : 100 % des 478 976 couples unreachable sont expliqués par select_irreversible, zéro autre cul-de-sac (ni dialogues ni formulaires). Difficulté structurelle pure, entièrement caractérisée : un contrôleur sans mauvais SELECT ne rencontre jamais d'unreachable. La difficulté logiciel-like est concentrée, pas diffuse.

17. **Re-scoping de la famille cible du transfert vers 'small' avant gel** (tagi-1, 22/09, déclenché par l'audit d'inventaire §4.5) : couverture quasi nulle aux budgets figés (k*=500 = 0,014 % des 2,83 M couples) aurait rendu un négatif ininterprétable (§12.5). Décision : 2-3 views/1 form/2 fields/2 options (~768 couples/layout, k=500 ≈ 65 % d'un layout), K 30-60 conservé par distracteurs (sans inflation d'états), budgets inchangés. Leçon : la puissance interprétative des budgets se vérifie à l'inventaire, pas au verdict.

18. **Un hash « invariant » doit être testé pour l'invariance** (tagi-2, 22/09) : la cause racine du certificat « décoratif » était un bug d'invariance de wl_hash_typed — payload d'arêtes par indices bruts, donc dépendant de la numérotation (aucun chemin ne pouvait reproduire: les chemins différaient par numérotation). Correctif: payload par couleurs raffinées, invariant par construction + test de renumérotation + PREUVE d'assertion au scellement contre le chemin audité fichier→SIWLayout→hash (plus de copie circulaire). Effet mesuré: 420→411 groupes (sur-découpage conservateur, zéro contamination — la correction était tenue par typed_isomorphic). Leçon: tout préfiltre invoquant une invariance se teste CONTRE cette invariance.

19. **Un protocole gelé sur papier n'est pas un harnais** (tagi-5/tagi-1, 22/09) : le figeage §12 (FREEZE.md) était réel et vérifié (dates/commits, budgets hash-identiques) mais le runner d'exécution n'existait pas — le code qui entraîne est partie prenante du figeage. Règle: figeage et runner se livrent ensemble, sinon le figeage s'appelle 'draft'.

20. **Un ruling de lead est auditable et réversible — et le bras contrôle ne diffère qu'en SOURCE** (tagi-1/tagi-5, 22/09) : le ruling #8 faisait dériver la slice cible du contrôle avec labels non-informatifs → le bras différait en source ET en cible, cassant l'attribution §12.2(2) (exposition vs décisionnel). Revert vers (a) : source non-informative + cible informative identique aux trois bras. Cause racine: une assertion mal placée interprétée incorrectement de deux façons différentes par deux lecteurs (lead et auditeur) — la troisième lecture (exécutant) était la bonne. Leçons: (i) le contrôle « random-label pretraining » standard isole le contenu décisionnel de la SOURCE, la cible doit être identique partout; (ii) trois vérificateurs valent mieux que deux — le correctif s'applique au lead inclus; (iii) la dérivation non-informative de cible survit comme bras secondaire déclaré « validity-only » (apprend la structure de préconditions, pas les décisions).

21. **Le scellé porte sur la forme publiée** (tagi-2, 22/09, racine commune avec #18) : l'inventaire 448df097 hachait le repr Python (frozensets) — non sérialisable donc invérifiable depuis les fichiers; re-scellé 5b0c51a9 sur la forme sérialisée canonique, avec lien MÉCANIQUE fichier↔scellé vérifié par assertion (les 5 fichiers de couples re-haschent exactement aux scellés). Règle: tout hash de scellé se calcule sur la forme qu'un tiers peut relire, jamais sur un artefact mémoire du processus qui l'a produit.

22. **Un journal d'interactions n'est un compteur que s'il compte TOUTES les interactions** (tagi-5, 22/09, `docs/INCIDENT-2026-09-22-sealed-read.md`) : tests d'adaptateur, smokes d'audit et optimisations ont ouvert/évalué le test SIW scellé hors du registre alors que FREEZE §4/§5 imposait une lecture unique et interdisait le débogage sur test. Un test qui évalue le scellé est une interaction cible, même si le modèle officiel n'a pas été modifié. M-V1b est **exploratoire**, pas confirmatoire ; divulgation append-only après la fin du PID, remplacement des tests par fixtures DEV disjointes et garde mécanique interdisant les références scellées dans `tests/`, nouvelle cellule confirmatoire iso-disjointe figée avant génération/ouverture. Les lectures d'audit sont elles aussi pré-déclarées et comptées ; une absence de sélection observée ne rétablit pas rétrospectivement le statut pré-enregistré.

23. **« One read » est un invariant d'instrument, pas une étiquette écrite après coup** (tagi-1/tagi-5, 22/09, `docs/INCIDENT-2026-09-22-sealed-read.md` §3) : le runner M-V1b ouvrait le test **trois fois** (sniff de format, chargement, hash) puis écrivait « one read » dans son log, après les ouvertures ; une panne avant ce log serait invisible. La future cellule exige un seul `open`/FD : intention journalisée et flushée AVANT ouverture, parsing et hash dans le MÊME flux, fermeture/hash journalisés, deuxième ouverture mécaniquement interdite et testée sur DEV. Le scellé est calculé pendant l'écriture, sans relecture pré-run ; audit par re-génération temporaire ou lecture pré-déclarée/comptée. La règle vaut aussi pour les couples scellés.

*(toute déviation supplémentaire de la spec ou des contrats doit être inscrite ici avec justification)*
