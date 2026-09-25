# INCIDENT 2026-09-22 — discipline « une lecture » du test scellé SIW

- **ID** : INCIDENT-2026-09-22-01 (sealed-read discipline)
- **Auteur** : tagi-5 (auditeur QA/red-team indépendant), signé 2026-09-22 ~21:32
- **Statut** : ruling accepté par le lead (21:29:58) — **M-V1b = EXPLORATOIRE**, aucun gate/claim confirmatoire sur cette cellule
- **Portée de la vérification** : code + historique git + journaux d'exécution **uniquement** — aucune relecture du fichier scellé pendant cet audit (exigence lead 21:28)

## 1. Clauses du figeage violées (référence, verbatim FREEZE §4/§5)

> §4 — « **Une LECTURE du test scellé.** Toute interaction cible (sélection, calibration, débogage) : COMPTÉE et interdite sur le test. »

> §5 — « Toute requête vers données/env cible SIW au-delà de la lecture unique s'inscrit dans `artifacts/v1/interactions-log.json` (horodatage, motif, bras). Le compteur fait partie de l'artefact final. »

## 2. Timeline bornée (Europe/Paris, sources vérifiables)

| Heure | Événement | Source |
|---|---|---|
| 19:16:56 | `edebb6a` — premières lectures `layouts-SIW-small.json` dans `tests/test_v1.py` (lignes 229/267/288/348 ; store incluant les 170 layouts test) | `git log -S`, code |
| 19:42:44 | `7b0c081` — `TestDataAdapter` charge le **fichier test scellé complet** (`episodes_to_runner` + `content_hash`) ; suite exécutée avec (« 33 tests V1 verts ») | `git log -S`, message de commit |
| 19:43–19:48 | **Audit tagi-5** : lectures de scellage (hash), `episodes_to_runner` (600), 5 spot-checks d* oracle, **smoke E2E évaluant 3 épisodes scellés avec un modèle (succès 0/3 imprimé)** | mes commandes, auto-déclaration |
| **19:55:38** | **Départ du run officiel** (PID 8282) | `ps` |
| **19:55:39** | **SEULE ouverture journalisée** : « SEALED TEST OPENED (one read, 600 episodes, grid 0,100,500,2000,10000) » | `artifacts/v1/interactions-log.json` |
| 19:57 | Audit tagi-5 : re-vérification finale (chargement 600, adapter/binding/rebuild, **sans évaluation modèle**) | mes commandes |
| 21:20:58 | `7982697` — `TestBatchedRollout` : évalue **24 épisodes scellés** avec asserts `success/outcome/length/n_invalid`, et `test_speedup_documented` en évalue **32** en imprimant un **ratio ms/ep** ; suite exécutée (« 37 tests V1 verts ») — **pendant le run** | `git log`, code |
| 21:26:00 | `ddb1a9e` — profilage : **0 référence** au fichier test scellé (vérifié) | `git show` |
| working tree | `TestFastTieParity` (ancienne version) : évalue **12 épisodes scellés** ; **exécution ATTESTÉE 1×** (auto-déclaration tagi-3, 21:41) | code + déclaration |

## 3. Finding statique — le runner officiel lui-même ouvre le test 3× (aucune lecture du fichier)

Vérification statique (`rg` + lecture de code uniquement) : le chemin officiel ouvre `args.test_file` **trois fois** avant de journaliser « one read » :

1. `ucm/v1/runner.py:193` — `open(args.test_file).readline()` (sniff de format) ;
2. `ucm/v1/runner.py:195` — `episodes_to_runner(args.test_file, …)` → `ucm/v1/data_adapter.py:105` `for line in open(path)` ;
3. `ucm/v1/runner.py:203` — `adapter_manifest(_sealed_sources)` (test_file inclus) → `data_adapter.py:58` `content_hash(path)` ouvre et lit tout.

Le log « SEALED TEST OPENED (one read, …) » est écrit **après** ces trois ouvertures (`runner.py:205`) : l'assertion « one read » est **fausse au niveau de l'instrument lui-même**, indépendamment des tests/audits ; et une panne pendant les ouvertures 2/3 laisserait **aucune** entrée « opening » (log après open). Les couples scellés suivent le même motif multi-open (`load_couples` sniff + matérialisation + `adapter_manifest`).

## 4. Ouvertures attestées / minimales

- **Journal** : 6 entrées = **6 ÉVÉNEMENTS, PAS 6 ouvertures** : i0–i3 `runner start (no target touch)`, i4 `sealed content hashes`, i5 `SEALED TEST OPENED (one read…)`. Le champ `interactions_count=6` compte des **événements**. i5 revendique « one read », **infirmé par le code** : le runner M-V1b ouvre le test **3×** (sniff `readline` + parse `episodes_to_runner` + hash `content_hash`, cf. §3) → le run lui-même = **3 FD** sur le test. Ne pas confondre avec la borne hors-run (tests/audits).
- **Ouvertures supplémentaires (hors run)** — composantes **séparées, non sommées sans preuve** :
  1. **tagi-3 (auto-déclaration 05:12)** : `TestFastTieParity` scellé `[:12]` = **2 opens** (NameError post-open + vert, ~21:25-21:27) ; suite `test_v1` 37 verts **1×** ~21:21 (DataAdapter×2 + BatchedRollout×2 = 4) ; **4 ciblées BatchedRollout** ~21:19-20 (jusqu'à 8) ; **morts avant open = 0** ; timing ms/ep visible, **aucun succès scellé imprimé par tagi-3** (le smoke tagi-5 a imprimé 0/3, cf. D2) ; post-`f9592b1` DEV-only.
  2. **tagi-5 (auditeur)** : lectures d'audit directes (hash de scellage, adapter 600, 5 d* spot-checks, **smoke E2E 3 épisodes évalués — 0/3 imprimé**) + **lectures couples scellés (adaptation, JAMAIS le test)** : re-hash des lignes scellées (vérification des scellés WS-B), re-dérivation `couples_to_records` (`audit_mv1b_final`), re-matérialisation de contrôle des 4 budgets (audit exploratoire) — **déclaration TARDIVE (sidecar D1), SANS intention préalable — pas de pré-déclaration rétroactive** ; + **runs de suite sur commits pré-remédiation** : `tests/test_v1.py` contenait des références scellées **sans garde** de `7b0c081` (19:42) à `39d66cc` (~19:58) → mes suites complètes/partielles à 4680221, 17fb09d, b5dfc65, c0d2a55, 39d66cc ont exécuté `TestDataAdapter` (`episodes_to_runner` + `content_hash` = 2 opens/run) ≈ **~10 opens**, plus les 4 lectures `layouts-SIW-small.json` depuis `edebb6a`.
  3. **run officiel** : 3 FD (instrument, ci-dessus).
- **Aucun total exact n'est revendiqué** (pas de journal par ouverture) ; les bornes par classe **ne se somment pas** sans preuve. `f9592b1` (21:34) a ajouté la garde `SEALED_TESTS` puis `d982c60` a rendu `tests/` DEV-only (0 référence).

## 5. Ce qui est inconnu (à ne pas présenter comme conforme)

- Nombre exact d'exécutions de la suite (chaque exécution recharge le fichier) ;
- affichage effectif des sorties imprimées (ratio de vitesse, etc.) vs capture pytest ;
- existence d'autres lectures hors périmètre tracé (aucun journal ne les couvre) ;
- ~~exécution effective de `TestFastTieParity`~~ → **attestée : 2 opens** (cf. §4/D4) ; restent inconnus : le **nombre exact** de suites locales pré-remédiation et tout **affichage effectif** des sorties ;
- toute inspection humaine du contenu des épisodes (non journalisable a posteriori).

## 6. Ruling

**Absence de sélection observée ≠ conformité.** Aucun élément n'indique qu'une décision du run officiel a utilisé les issues du test (optimisations post-run-only, chemin séquentiel gelé au départ, PID 8282 intact, **aucun succès scellé imprimé par tagi-3** — le smoke tagi-5 a imprimé 0/3, D2). Mais la discipline est **pré-enregistrée** : des interactions supplémentaires — dont des **calculs d'issues** et une **mesure de performance** sur le test — sont **formellement interdites et non journalisées**. En conséquence :

- **M-V1b (cellule actuelle) = EXPLORATOIRE** ; aucun gate/claim confirmatoire sur cette cellule ;
- la confirmation exige une **nouvelle cellule scellée entièrement disjointe, pré-annoncée AVANT ouverture** (voir `docs/CONFIRMATION-PRIMARY-MV1b.md`).

## 7. Actions correctives (requises)

1. **Divulgation append-only TARDIVE (créée)** : sidecar `artifacts/v1/incident-2026-09-22-disclosure.json` — `declaration_timestamp=2026-09-23T05:16:29+02:00` (horodatage **réel** de déclaration), `prior_intention_recorded=false` (**aucune intention enregistrée AVANT ces ouvertures** — pas de pré-déclaration rétroactive), `old_journal_modified=false` (le journal hashé `interactions-log.json` **n'est pas modifié**), séparation stricte **couples adaptation ↔ TEST** (test jamais rouvert depuis le constat/ruling), entrées **D1–D8** avec bornes de FD **par classe, selon preuve — pas de compte inventé** : D1 couples adaptation (tagi-5), D2 lectures test directes (tagi-5, smoke 0/3 imprimé), D3 suites pré-remédiation (tagi-5), D4 tests optimisations (tagi-3), D5 run (3 FD, `runner.py:193/195/203`), D6/D7 tagi-4 (auto-déclarés non attestés), D8 erratum provenance. Source des hashes canoniques = index WS-B publié (`couples_materialized_files[...].materialized_sha256_canonical`) ; la re-matérialisation de contrôle est une comparaison, pas la provenance. **Sidecar créé (D1–D8) ; §9 apposé par tagi-4 en référence (sha du sidecar).**
2. **Zéro test scellé** : remplacer toutes les références scellées des tests par des fixtures DEV (`TestDataAdapter`, `TestBatchedRollout`, `TestFastTieParity`, lectures `layouts-SIW-small.json`) + **garde mécanique** : test échouant si un fichier de `tests/` référence un artefact scellé (allowlist vide).
3. **Optimisations hors chemin officiel** : ne pas committer les optimisations dans le code du run ; PID 8282 intact.
4. **Lectures d'audit** : pré-déclarées et journalisées dans le futur (correction côté tagi-5 incluse).
5. **Instrument de lecture unique (obligatoire pour test2)** : un seul `open`/FD du fichier test ; parsing **et** hash dans le même flux d'octets ; log `opening-intent` **avant** l'open (flush/fsync), puis `opened+hash` et `closed` après ; garde instrumentée (audit hook `open` ou wrapper) qui **FAIL si un second open** du test survient (DEV) ; sniff de format et `adapter_manifest` interdits de rouvrir le test (hash pris du flux ou de l'objet en mémoire) ; génération/scellement : hash **pendant l'écriture** (aucune réouverture) ; audit par re-génération temporaire (aucune lecture du scellé) ou lecture pré-déclarée comptée sous dérogation explicite. Même discipline pour les couples scellés.
6. **Cellule de confirmation** : nouvelle cellule + protocole gelé avant ouverture (voir `docs/CONFIRMATION-PRIMARY-MV1b.md`, DRAFT v4.1).

## 8. Signature

tagi-5 — auditeur indépendant. Vérification statique (code/historique/journaux) ; aucune relecture du scellé pour cet audit. Amendement v2 (21:35) : finding statique runner 3× open + exigences d'instrumentation lecture unique. Registre proposé : `#22 — un journal d'interactions n'est un compteur que s'il compte TOUTES les interactions ; un test qui évalue le scellé est une interaction cible.` `#23 — « one read » doit être un invariant d'INSTRUMENT (un seul open/FD, hash dans le flux, log avant open), pas une étiquette écrite après coup.`

---

## 9. Divulgation append-only — interactions hors-run & lectures d'audit (post-PID 8282)

- **Apposé par** : tagi-4 (OPS), sur instruction tagi-1, après sidecar scellé tagi-5. §1–§8 intacts ; **aucun rétro-log** ; aucune antidate.
- **Sidecar unique (figé)** : `artifacts/v1/incident-2026-09-22-disclosure.json` — sha256 `704c05431a073ca0d4eb8f30c067a5a10b7ef8e6989f52c34c5295895ac25452`, declaration_timestamp `2026-09-23T05:16:29+02:00` (D1–D8). Toute modification ultérieure = **nouvelle révision + erratum** (jamais d'overwrite silencieux). Ce §9 le référence **sans le dupliquer ni le contredire**.
- **Condition** : PID 8282 TERMINÉ (vérifié read-only 2026-09-23 05:11 CEST : `ps` absent, aucun runner actif).
- **Journal** : `artifacts/v1/interactions-log.json` = **6 entrées/événements** (chaîne OK) — **pas 6 ouvertures** ; i0–i3 = 4× `runner start (no target touch)`, i4 = hashes, i5 = `SEALED TEST OPENED (one read, 600 episodes, …)`. sha256 `397d62b8…`, mtime 19:55:39, stable. **Immutable.**
- **Requalification i5** : étiquette « one read » **fausse au sens physique** — **3 ouvertures réelles** (`runner.py:193/195/203` → 3 FD). `interactions_count=6` compte des événements.

### 9.1 Divulgations par acteur (sidecar unique D1–D8)

| ID | Acteur | Classe | Interactions | Fenêtre | Borne (par classe) | Attestation |
|---|---|---|---|---|---|---|
| D1 | tagi-5 | couples adaptation | Re-hash lignes des 4 couples scellés + re-dérivation `couples_to_records` (WS-B) + re-matérialisation exploratoire | 22/09 ~19:44–19:57 ; 23/09 ~05:13–05:14 | ≥4 + ≥4 + 4 (non sommé) | auto-déclaré non attesté (horodatages approximatifs) |
| D2 | tagi-5 | test direct | Hash scellage, `episodes_to_runner` (600), 5 spot-checks d*, **smoke E2E 3 épisodes évalués (succès 0/3 imprimé par tagi-5)**, chargement final | 22/09 ~19:43–19:57 | ≥6 | attesté par preuve (tagi-5) |
| D3 | tagi-5 | suites pré-remédiation | Runs de suites sur commits sans garde (TestDataAdapter ×2 opens/run ×5 ≈10) + lectures `layouts-SIW-small.json` (4 sites) | 22/09 ~19:26–19:58 | ~10 + 4 (non sommé) | attesté par preuve (réfs non gardées des commits) |
| D4 | tagi-3 | tests/optimisations | `TestFastTieParity` [:12] 2 opens (NameError post-open + vert) ; suite 37 verts 1× (DataAdapter×2 + Batched×2) ; 4 ciblées Batched ; morts avant open = 0 ; **aucun score imprimé par tagi-3** (timing ms/ép) ; post-`f9592b1` DEV-only | 22/09 ~21:19–21:27 | ≥2 + ≥4 + (4 à 8), non sommé | auto-déclaré non attesté (tagi-3) |
| D5 | run officiel (PID 8282) | run-self | 3 ouvertures test par le code (`runner.py:193/195/203`), i5 écrit APRÈS le hash | 22/09 19:55:39 | **3 (prouvé code)** | attesté par preuve (code) |
| D6 | **tagi-4 (OPS)** | test-side read-only | **7 ouvertures** : 1× `siw-test-episodes.jsonl` (sha256 seul) ; 2× `siw-test-episodes-manifest.json` (structure + sha256) ; 4× `layouts-SIW-small.json` (3× json.load + 1× sha256). **Aucune éval modèle, aucun calcul d'issue, aucune sélection.** + 7 adaptation (5 couples sha256, inventory, split-manifest) | 22/09 19:37–19:47 (pré-run) | 7 test-side (déclaré) + 7 adaptation (déclaré) | **auto-déclaré non attesté (tagi-4)** — cohérence classe/timeline confirmée par tagi-5, comptes non vérifiés |
| D7 | **tagi-4 (OPS)** | post-run | **0 ouverture scellé** : ps (PID), sha256 journal, inventaire `stat` (noms/tailles) | 23/09 05:11 | 0 | auto-déclaré non attesté (tagi-4) |
| D8 | tagi-5 | erratum provenance | `materialized_records_hashes_per_budget` calculé sur `budgets.slice(k)` (pool max) ≠ `sealed_slices[k]` (entraînement) ; k2000/k10000 identiques `fff400b2…` | constat statique 23/09 ~05:14–05:16 | — | attesté par preuve (statique :188-190 vs :246) |

- **Renvoi §2 (timeline)** : la mention « exécution ATTESTÉE 1× (21:41) » de §2 est une **attestation intermédiaire, supersédée par D4 = 2 opens** ; §1–§8 non modifiés.

### 9.2 Bornes — par classe, **sans addition mécanique** ; total exact INCONNU

- Composantes, **par entrée — sans somme** : **D1** ≥4 + ≥4 + 4 · **D2** ≥6 · **D3** ~10 + 4 (layouts) · **D4** ≥2 + ≥4 + (4 à 8) · **D5** 3 (prouvé code) · **D6** 7 test-side + 7 adaptation (auto-déclarés) · **D7** 0. (Rapport §4 révisé.)
- **Attestation (tagi-5, reprise telle quelle)** : *attesté par preuve* = **D2, D3, D5, D8** ; *auto-déclaré non attesté* = **D1** (horodatages approximatifs), **D4** (tagi-3), **D6/D7** (tagi-4). **Aucun total exact de FD** ; bornes **par classe, sans addition mécanique**.
- **Aucune borne combinée** revendiquée ; pas de journal par ouverture ⇒ compte physique exact non attestable.
- Hashes canoniques des couples : `artifacts/siw-runner-artifacts-index.json` (`couples_materialized_files[k].materialized_sha256_canonical`) — déjà vérifiés à l'audit WS-B ; la re-matérialisation n'est qu'une comparaison.

### 9.3 Erratum séparé — `materialized_records_hashes_per_budget` (D8)

- Calculé **`runner.py:188-190`** depuis `budgets.slice(k)` (**pool max 1191**) ; l'entraînement utilise `sealed_slices[k]` (**`runner.py:246`** ; 94/376/852/1191).
- Preuve : `k=2000` et `k=10000` → **même hash** `fff400b207767ba2`. **N'atteste PAS** les records d'entraînement ; correctif = logger les hashes des `sealed_slices` (runs futurs, jamais rétroactif). JSON historique **immutable**.

### 9.4 Statut & limites

- Divulgation **append-only séparée** (sidecar unique D1–D8) ; journal historique intact ; pas de rétro-log ; **test jamais rouvert depuis le constat/ruling de l'incident** (D2–D5 = ouvertures historiques antérieures ; séparation stricte couples réouverts / test).
- Aucun élément n'indique qu'une décision du run a utilisé ces lectures — lectures **pré-run** : D2/D3/D6 ; **D4 = durant-run** (~21:19–21:27) **sans effet observé sur le chemin du PID** ; pas de publication de succès scellé par tagi-3 ; smoke 0/3 tagi-5 imprimé mais non exploité pour sélection.
- **IC non recalculable** : pas de raw per-episode ni checkpoints cible ; relecture du scellé **interdite**.
- M-V1b reste **EXPLORATOIRE** (ruling §6). Apposition après scellement/commit du sidecar par tagi-5 + go lead.

### 9.5 Nuance statistique des résultats exploratoires (lead 05:15 — mot pour mot)

- `ctrl−pretrain` +0.8 pp IC [−13.3,+13.9] **inclut 0** → **ne prouve ni l'équivalence ni l'absence de composante décisionnelle** ; pas de signal distinguable à cette puissance.
- Formulation sûre : « **contrôle comparable observé, mécanisme exposition/optimisation plausible, aucune attribution décisionnelle établie** » — éviter « le contrôle égale le pré-entraîné » (pas de test d'équivalence).
- `ctrl−scratch` +13.37 IC [3.77,22.76] = **différence observée**, pas une attribution.
- **IC exploratoire non recalculable indépendamment** (pas de raw per-episode ni checkpoints cible ; relecture interdite).
