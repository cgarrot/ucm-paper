# PLAN DE CONFIRMATION PRIMAIRE M-V1b′ — nouvelle cellule test scellée disjointe

- **Statut** : **DRAFT v4 tagi-5** (corrections lead 21:32, 21:35, **21:38** intégrées) — soumis à adoption/amendement. **Aucun artefact généré ni ouvert** ; aucune lecture test2.
- **Contexte** : la cellule M-V1b (600 épisodes, manifest 0b28a930) est **exploratoire** (INCIDENT-2026-09-22-01). La confirmation porte sur une **nouvelle cellule test** ; l'**adaptation et les checkpoints source sont réutilisés** (un seul facteur change : la cellule test).

## 1. Portée : PRIMAIRE-SEUIL

- **Seul mesuré** : `k* = 500`, 3 bras (`scratch`, `pretrained-TGK`, `control-nontarget`) × 5 seeds (0..4), 600 épisodes test (150×4).
- **NON MESURÉS / ABSENTS** : grille complète `k`, AULC `log(1+k)`, seuil 80 %/censure, `control-validity-only`, courbes — attachés à la cellule exploratoire M-V1b, non produits ici.

## 2. Distribution RÉELLE et écart déclaré (correction lead 21:38)

- **Distribution retenue = code réel : `views ∈ [2, 5]`** (`scripts/repro_siw_seals.py::gen_layouts(seed, n, vmin=2, vmax=5)`), celle qui a produit les 420 layouts du manifest 0b28a930 et la cellule M-V1b.
- **Écart description/code déclaré comme fait** (registre) : la description du profil `small` (#17) annonce 2–3 views, le code utilisé est 2–5. Pour une confirmation **comparable**, test2 garde **2–5** ; l'écart est publié, **pas corrigé** en changeant la distribution. (Leçon : décrire par le code qui a produit, pas par l'intention.)

## 3. Générateur et sampler — version GELÉE exécutable (pas de placeholder)

- **Layouts** (stream unique, miroir du manifest) :
  - `rng = random.Random(S_gen)` créé **une fois** ; **exactement 600 candidats** générés : pour chaque candidat : `n_views = rng.randint(2, 5)` ; `with_dialog = rng.random() < 0.5` ; `generate_siw_layout(rng, n_views, with_dialog=…, k_bounds=(30, 60), profile="small")` (ordre RNG interne = code committé pinné).
  - **Sélection du pool test (déterministe, sans retirage)** : trier les 600 candidats par `layout_hash` ; retenir les **170 premiers** qui sont (a) `layout_hash` et groupe kind-aware/iso **disjoints des 450 layouts existants** (manifest 420 = 200/50/170 + inventaire dev 30) **et** (b) **mutuellement disjoints entre eux** (hash + iso). Si moins de 170 candidats qualifiés : **FAIL** (aucun tirage supplémentaire, aucune extension, aucun ajustement).
- **Sampler d'épisodes — OPTION B retenue : GELER le sampler actuel like-for-like** :
  - le code `build_siw_test_episodes` est gelé **tel quel** (boucle, ordre RNG, 3 rejets : prédicat ≠ cible → skip, unreachable → rejet, `d*` hors bande (2, 8) → rejet) — **aucun rejet de doublon ajouté** (le sampler actuel n'en a pas) ;
  - **ajouts passifs uniquement** (ne changent ni le flux RNG ni les épisodes produits) : compteur `attempts_total` **incrémenté AVANT chaque essai** + **`global_attempts_cap = 200_000` EXACT — même valeur dans le code et dans ce document** → `RuntimeError` visible si dépassé ; **preuve « cap non atteint » obligatoire** (manifest : `attempts_total < 200_000` + assertion dédiée), **cap DEV-testé** ; compteurs publiés `predicate_skips`, `rejected_unreachable`, `rejected_band`, `attempts_total`, **`duplicate_couples`** (nombre d'épisodes produits dont la clé `(layout_hash, init canonique, goal canonique)` est déjà présente — **descriptif, aucun filtrage/rejet**) ;
  - **bug de paramètre documenté** : `max_attempts_per_ep=400` est **déclaré mais NON utilisé** (vérifié : seule occurrence = signature) — publié comme tel, non promis ;
  - **fréquence des doublons publiée en descriptif** (pas de filtrage) — propriété du sampler gelé.
- **Alternative explicite (OPTION A, non retenue pour la confirmation)** : implémenter un sampler v2 (cap 400 réellement appliqué + rejets de doublons + compteurs) **avant gel** avec tests dédiés, et **annoncer le changement de sampler vs M-V1b** (consommation RNG différente). Non retenue : la confirmation doit être like-for-like ; si le lead la préfère, elle doit être implémentée/testée et déclarée avant le freeze, pas après.

## 4. Seeds, disjonction, adaptation

- **Seeds** : utilisés = 20260922 (M0/audit), 20260923 (g2ext), 20260924 (inventaire dev), 20260926 (manifest), 20260927 (épisodes M-V1b), **20260928 (DEV fixtures fb30fe7)**. Retenus : **`S_gen = 20261003`**, **`S_ep = 20261004`** (absents du dépôt).
- **Disjonction (double niveau, avant run)** : (a) `layout_hash` vs tous les pools/dev ; (b) **groupes kind-aware/iso** (`certificates_kind_aware` + prefilter WL typé + `typed_isomorphic`). Tests mécaniques committés.
- **Adaptation réutilisée** : slice **k=500 scellée** (376 couples) + checkpoints canon `gate2c-B144-s{seed}` / contrôle `v1-control-s{seed}` existants — **aucun nouveau pool d'adaptation** (un seul facteur change).

## 5. Snapshot propre (correction lead 21:38)

- Exécution depuis un **worktree de code isolé** : `git worktree add --detach <dir> <commit>` (ou `git archive`) au commit du protocole ; `git status --porcelain` **du worktree** vide ; hash de contenu des fichiers de génération/analyse (script de stats + génération) ; le `git status` global (artefacts non suivis) n'est **pas** le critère. **NB (audit WS-B)** : les checkpoints canon/contrôle sont **non suivis** → un clean-worktree ne les contient pas ; la reproduction dépend donc du **manifeste de sources §7ter-ter-9** (chemins read-only + sha 64 vérifiés avant l'ouverture test2), pas du seul commit.
- Enregistrer : `git rev-parse HEAD` du commit, chemin du worktree, hashes de contenu, config de run, et le chemin du **log isolé**.

## 6. Pré-enregistrement AVANT ouverture (ordre imposé)

1. **Committer** : ce plan + paramètres/algorithmes exacts (§3) + snapshot (§5) + log isolé + règle de verdict.
2. **Générer** la cellule (seeds/blocs gelés).
3. **Sceller PENDANT l'écriture** (hash incrémental, aucune réouverture) ; **disjonction (a)+(b) vérifiée EN MÉMOIRE** sur les objets générés ; la **vérification octets/hash se fait dans l'UNIQUE open du runner** (au run, §7). **Aucune re-génération pré-run** : reconstruire la cellule avec les mêmes seeds avant le run reviendrait à « ouvrir » la cellule test (interaction non journalisée) — une **re-génération indépendante n'a lieu qu'APRÈS le run, journalisée comme audit** (ou est explicitement hors périmètre).
4. **Run** : 3 bras × 5 seeds × k*=500, adaptateur/updates identiques (2000 updates, batch 64), init fraîche partagée par seed, checkpoint final à budget fixe sans sélection ; **un seul open** du test (§7).

## 7. Lecture unique INSTRUMENTÉE + journal isolé

- **Fichier test AUTO-CONTENU (recommandation lead)** : chaque ligne épisode embarque `layout_spec` (format runner direct, `layout_spec` inclus) → le run n'a besoin que d'**UN SEUL FD** pour la cellule test ; le layout store n'est utilisé qu'à la **génération** (pré-freeze) et n'est **pas** ouvert par le run. Si un protocole multi-fichiers était préféré, il devrait être **déclaré et compté** (et « one-read » renommé en conséquence) — non retenu par défaut.
- **Un seul `open`/FD** du fichier test : octets lus une fois → `sha256` sur ces octets → parsing depuis les **mêmes** octets → épisodes en mémoire ; **vérification octets/hash = ce même open** ; plus aucun accès fichier ensuite.
- **Interdits** : sniff `open().readline()`, `adapter_manifest`/`content_hash` rouvrant le test, re-hash pré-run par ouverture supplémentaire.
- **Log AVANT open** : `opening-intent` (chemin, hash attendu, horodatage) écrit avant l'`open` (flush/fsync), puis `opened+hash`, `closed`.
- **Garde instrumentée** : hook `open`/wrapper → **FAIL si > 1 open** du test (DEV) ; garde statique reader unique.
- **Scellement** : hash **pendant l'écriture** ; même discipline pour les couples scellés.
- **Log isolé** : `artifacts/v1-confirm/interactions-log.json` (append-only), séparé de M-V1b ; aucune lecture non journalisée ; lectures d'audit pré-déclarées (dérogation explicite) ; **l'audit indépendant par re-génération est post-run et journalisé**.
- **Garde no-sealed étendue** : test échouant si `tests/`/scripts référencent la cellule (allowlist vide ; durcie : fragments de noms, pas seulement `artifacts/<fichier>`).

## 7bis. Artefacts du runner de confirmation (sortie brute isolée)

Le runner actuel n'écrit que des `cells` résumés + `primary`/CI et **jamais** `all_results` bruts ni les checkpoints finaux (vérifié) → re-calcul indépendant impossible. Le runner de confirmation DOIT :

- écrire dans **`artifacts/v1-confirm/`** (isolement ; **pas** `OUT=artifacts/v1`) ;
- sérialiser **`EpisodeResult` brut par arm × seed × épisode** (+ `sha256` du fichier brut) — source du re-calcul indépendant ;
- sauvegarder le **checkpoint final par cellule** (2000 updates, budget fixe) + son `sha256` ;
- calculer `cells`/`primary`/IC depuis le fichier brut, et publier les compteurs/couverture.
- **Tests DEV obligatoires avant test2** : garde 1-open, sérialisation brute, checkpoints+sha ; **code générateur/runner/garde committé et hashé avant test2**.

## 7ter. Revue statique du runner_confirm WIP (7feda28) — NO-GO exécution

Vérifié par lecture de code (file:line), indépendamment de tagi-1 ; corrections obligatoires + tests DEV d'acceptation :

| # | Défaut constaté (file:line) | Correction attendue | Test DEV d'acceptation |
|---|---|---|---|
| 1 | `OUT = "artifacts/v1"` (`runner_confirm.py:35`) — sortie non isolée | `OUT = "artifacts/v1-confirm"`, log isolé `artifacts/v1-confirm/interactions-log.json` | assert OUT/log isolés ; aucun écrit dans `artifacts/v1` |
| 2 | `InteractionsLog()` défaut (`:42`) = log de M-V1b/PID 8282 ; guard `--test2-file` (`:58-61`) **après** makedirs + log + lecture adaptation + écritures tmp | log isolé explicite ; **failfast en TOUT PREMIER** (avant tout side effect : makedirs/log/lecture/écriture) | exécuter sans `--test2-file` → exit immédiat, **zéro side effect** (spy sur `open`/écritures, répertoire/log inchangés) |
| 3 | Patterns ckpt doublés (`:82-83` : glob retourne déjà `.../checkpoint.npz` + `"/checkpoint.npz"`) | construire le chemin sans double suffixe | DEV : les deux checkpoints source se chargent (existence + `build_arm` OK) |
| 4 | `cfg.seed` **jamais** mis à `seed` (le runner original le fait) | `cfg.seed = seed` par cellule | DEV : deux seeds → `cfg.seed` distincts et reproductibles ; résultats par seed cohérents |
| 5 | `episodes_to_runner(tmp2, None)` (`:69`) crash si `--test2-layout-store` absent ; **copie test2 sur disque** sous `OUT` hors garde (`:66-68`) | **aucun tmp test2** : fichier test2 **auto-contenu** (`layout_spec` bundlé) → `episodes_to_runner` lit les lignes déjà lues (ou adapter direct) ; `test2_layout_store` non requis au run | DEV : run sans layout-store test → OK ; aucun fichier temporaire contenant le test sur disque à aucun moment |
| 6 | Pas de `raw` par épisode, pas de checkpoints finaux ; `per_seed` placeholder vide (`:107`, `:116`) | `EpisodeResult` brut par arm×seed×ép + `sha256` ; checkpoint final + sha par cellule ; `cells/primary/IC` calculés depuis le brut | DEV : re-calcul du primaire depuis le brut == rapport ; `per_seed` non vide ; ckpts+sha présents |
| 7 | `SealedOpenRegistry` **par instance** (doc dit process-wide) ; hash **tronqué 16** et **non comparé** au hash complet attendu → corruption non bloquée (`sealed_reader.py`) | registre **class-level/process-wide** (ou singleton) ; hash **64 hex complets** ; **comparaison au hash scellé attendu** → abort si différent | DEV : 2 instances → 2ᵉ open RuntimeError ; hash altéré → abort avec message ; hash complet comparé |

Tant que ces 7 points + les tests DEV ne sont pas committés, **le runner_confirm reste NON EXÉCUTABLE** (même sans `--test2-file`), et le DRAFT v4.2/4.3 reste **non adopté**.

### 7ter-bis. Revue statique runner_confirm v2 (ddb2602) — NO-GO malgré corrections

Vérifié file:line (aucun test2 ouvert). Défauts restants + tests DEV d'acceptation :

| # | Défaut (file:line) | Correction attendue | Test DEV |
|---|---|---|---|
| 1 | `recorded_hash` lit `record["content_hash"]` inexistant (le record stocke `sha256_full`) → **KeyError après run** (`sealed_reader.py:66` ; appelé `runner_confirm.py:163-164`) | retourner `sha256_full` | après run DEV : `recorded_hash` renvoie le hash **64 hex** complet, sans réouverture |
| 2 | `raw_fh` encore **ouvert** quand `_per_seed_from_raw`/`_primary_from_raw` lisent le fichier (`:108` / `:166-167` / close `:174`) | `flush`+`fsync` (ou close) **avant** lecture | le fichier raw est complet (dernière ligne + `\n`) au moment de l'analyse |
| 3 | `"per_seed"` **dupliqué** : ligne 166 écrasée par le placeholder vide ligne 172 | supprimer le placeholder | `per_seed` non vide, cohérent avec le raw |
| 4 | `tmp2` copie test créée (`:85-98`) alors que les lignes sont déjà en mémoire | supprimer la copie | aucun `confirm-test2.tmp.jsonl` créé (spy/filesystem) |
| 5 | `--couples-hash`/`--test2-hash` **optionnels (None)** → comparaison 64 hex jamais forcée | **exiger** les hashes attendus (ou refuser de tourner sans) | hash faux → abort **avant tout side effect** |
| 6 | sha checkpoint **tronqué** `[:16]` (`:141`) ; **raw non hashé** ; **primary/CI publiés non recalculés depuis le raw** (`:161-167` : le `primary` vient de `results`) | sha **64 hex**, sha du raw, **primary/CI = recalculés depuis le raw** (ou égalité stricte exigée) | re-calcul indépendant depuis le raw == rapport publié |
| 7 | raw `seed = r.seed` (=100+seed) sans `train_seed` explicite (`:127`) | champs `train_seed` (=seed), `eval_seed` (=r.seed), `k` | champs présents et cohérents |
| 8 | `os.makedirs(OUT, exist_ok=True)` (`:61`) → écrasement/rerun ; singleton **par processus** réinitialisé entre processus | sortie **exclusive** (refuser si `confirm-report.json`/dir non vide, ou `--out` versionné) + **registre on-disk** (log/manifest) pour le cross-process | rerun → abort ; second processus → refuse |
| 9 | `_failfast` calcule un hash de protocole **post-hoc** sans comparaison au hash gelé (`:41-56`) | exiger `--protocol-hash` (committé **avant** génération) et abort si différent | hash de protocole faux → abort |

### 7ter-ter. Critère de sortie DEV E2E (gate d'adoption, lead 21:50)

Un commit **propre** (jamais le working tree en mouvement) doit passer cette séquence **sur fichiers DEV uniquement**, sans aucun test2 :

1. **writer DEV → fichier scellé** (hash incrémental à l'écriture == vérification post).
2. **one-open reader** : `expected_hash` **64 hex OBLIGATOIRE** (hash tronqué/non fourni → refus) ; 2ᵉ open (même process ou second process) → abort ; **zéro tmp test** créé. **Garde on-disk atomique/lock** : **l'intention écrite compte comme tentative consommée** — un crash avant la lecture **ne permet pas** un replay silencieux (le registre on-disk + lock le refusent) ; **test DEV à deux processus** : le second est refusé **sans ouvrir**.
3. **1 cellule** finetune/eval (1 bras × 1 seed × k*=500, 2000 updates) avec `cfg.seed` par cellule.
4. **raw flush+fsync** puis **sha256 raw (64)** ; **checkpoint sha (64)** par cellule.
5. **primary/CI reconstitués depuis le raw** == rapport publié.
6. **failfast zéro side-effect** sur : args manquants (`--test2-file`, `--couples-hash`/`--test2-hash`, `--protocol-hash`), **OUT existant** (rerun → abort), hash de protocole ≠ hash gelé — vérifié par spy (aucun makedirs/log/écriture avant abort).
7. Sortie/log **isolés** (`artifacts/v1-confirm/`), registre **on-disk atomique/lock** pour le cross-process (intention = tentative consommée, cf. §2).
8. **Aucun chiffre test2 émis** (rapport, print, artefact) tant que **raw sha + checkpoint sha + CI reconstitué depuis le raw ne sont pas scellés et vérifiables** — sinon le runner **abort sans publier de chiffres**.
9. **Sources checkpoints (lead 22:05/22:06)** : chemins **read-only explicites** (`--canon-ckpt-path`/`--control-ckpt-path` ou patterns stricts) + **sha256 complets (64) vérifiés AVANT l'unique ouverture test2** ; **manifeste des sources** (chemin + sha256 + updates **prouvés par les artefacts** — `final.json.updates_run`/logs du run source, **jamais inférés** de la config ou du nom) publié/committé. La dépendance de reproduction révélée par le worktree propre (checkpoints **non suivis**) est ainsi fermée : un checkout propre + le manifeste de sources suffisent à reproduire. **NB (audit 87a4fa1 + 0004730 + 3276fbc)** : `build_freeze_manifest` **ne réécrit plus** le manifeste source (read-only ✅, relative → failfast avant write ✅, tests sha input inchangé) et `_verify_source_manifest` rejette aussi les relatifs ✅. **MAIS** le builder ne valide que l'**absolutude** : il n'appelle **pas** `_verify_source_manifest` → un manifeste source invalide (role|seed manquant/dupliqué, checkpoint absent, sha faux, updates non prouvés) peut être **figé**, et la **génération test2 peut consommer S_gen/S_ep AVANT** que `_failfast` ne le découvre au run. **Exiger** : appeler `_verify_source_manifest` **dans le builder, AVANT d'écrire le freeze et AVANT toute génération** (échec = aucun fichier de freeze écrit) ; **test DEV manifeste source invalide → abort + zéro freeze** ; docstring à replacer en tête de fonction. **+ Création EXCLUSIVE du freeze** : `open(freeze_path,'w')` (3276fbc:479) **écrase silencieusement** un freeze existant → utiliser `open(...,'x')`/`O_EXCL` (abort si préexistant, avant tout write) ; **test DEV freeze préexistant → fail + sha du freeze existant inchangé**. Statut : **defense-in-depth** — la garantie indispensable reste le **commit/publication du freeze AVANT génération** + le binding `--protocol-hash` = sha canonique du freeze committé vérifié live par `_failfast`.

Aucune adoption du plan ni génération test2 avant que ce critère ne soit vert sur un commit propre ; le freeze test2 reste **signé** et postérieur au code versionné.

### 7ter-quater. Revue statique runner_confirm v3 (6c99264) — 1 MAJEUR + 3 gaps

Vérifié file:line (aucun test2 ouvert). Le NO-GO est maintenu :

| # | Défaut (file:line) | Correction attendue | Test DEV |
|---|---|---|---|
| M1 | `EXPECTED_LAYOUTS = 150` + `per_layout_counts != 4` (`runner_confirm.py:57,104-109`) : contrainte **FAUSSE** — 150 = cible **par prédicat**, sampler réel = tirages **avec remise** sur pool 170 (M-V1b : ~167 layouts, comptes/layout variables) → le test2 réel **crashe APRÈS l'unique open** | valider **600 lignes = 150 × 4 PRÉDICATS** (`task.goal.predicate`), layouts ⊆ pool **sans contrainte de compte/layout** ; `layout_spec ↔ layout_hash` **recalculé** ; disjonction adaptation ; hash attendu. **Un fichier à 150 layouts×4 épisodes est LICITE** s'il satisfait 600=150/prédicat, spec↔hash cohérents et pool disjoint — **ne pas le refuser pour son seul compte de layouts** | **Positif DEV = vrai sampler** (comptes/layout variables) ; **négatif DEV = 150 layouts×4 MAIS prédicats déséquilibrés** (ou autre violation contractuelle : spec↔hash incohérent, chevauchement pool) → refus attendu pour le **bon** motif |
| G2 | Checkpoints source glob/sha **APRÈS** l'open test2 (`_failfast` ne valide que `{seed}` ; `_ck_path`+`build_arm`+`_sha256_file` dans la boucle cellules, après `read_once`) → absent/mismatch **consume la cellule**. **Précision lead 01:31** : `final.json.updates_run` prouve la **durée du run source**, PAS l'update du `checkpoint.npz` chargé (`train.py:306-310` = best validation ; ex. statiques : canon s0 `updates_run=860` mais `best.update=800` ; contrôle s0 `850` mais `800`, `checkpoint_policy=best_validation`) | manifeste source vérifié **AVANT l'open** : `path`, `sha256(64)`, `checkpoint_policy`, **`source_run_updates_run`** (860/850) ET **`loaded_checkpoint_update`** (`best.update` si best_validation, sinon `updates_run` = 800), `final_json_sha256` ; **ne jamais** écrire « ckpt à 850 » | DEV : absent/hash faux/update incohérent → abort **avant** l'open (spy : test2 jamais ouvert) ; le champ `loaded_checkpoint_update` est vérifié contre `final.json.best.update` ET la sélection effective |
| G3 | raw `goal_type` **vide** : le parse n'ajoute pas `goal_type`, `evaluate` lit `ep.get('goal_type')` absent (writer fournit `task.goal`) | mapper `goal_type = task.goal.predicate` (parse ou raw) | DEV : `goal_type` non vide et == prédicat du task, par épisode |
| G4 | hash de protocole **incomplet** (vérifié 19c894d) : stage-A lie `_code_sha` (runner_confirm seulement) + `source_manifest_sha` + **chemins** couples/store ; **pas** le sha de contenu couples/store, pas les hashes générateur/sampler/stats, pas le snapshot commit, pas `S_gen`/`S_ep`/distribution/600→170, pas méthode+chemin du manifest test2 → **deux pré-enregistrements différents peuvent avoir le même protocol_hash** | **freeze-manifest committé+haché** liant TOUTES les variables pré-génération : hashes de code (runner_confirm, générateur/sampler, stats, tensorizer si utilisé) + snapshot commit (`HEAD` + arbre propre), `S_gen=20261003`/`S_ep=20261004`, distribution (views 2–5, p(dialog)=0.5, profile small, goals/layout, règle pool 600→170 + exclusion 450 + FAIL), **sha de contenu** couples + store, manifeste source (champs G2), **méthode+chemin** du manifest test2 (pas son hash), counts 600/150×4 ; `_failfast` **vérifie le freeze-manifest avant l'open** ; stage-B test2 sha séparé, publié après génération | DEV : modifier un code hash / un seed / un paramètre de distribution / le sha couples → **mismatch et abort** ; deux pré-enregistrements différents ⇒ hashes différents |

## 8. Critère de verdict (figé)

- Primaire : `k*=500`, `pré-entraîné − scratch ≥ 5 pts`, borne basse **IC95 apparié hiérarchique > 0** (seeds → layouts → tâches), 5 seeds. Sinon échec du gate primaire, publié tel quel.
- Couverture en couples uniques (adaptation k=500) publiée à côté ; aucune autre analyse.

## 9. Checklist de revue indépendante (tagi-5, après le run)

1. Protocole/snapshot/hash antérieurs aux scellés ; **disjonction en mémoire + hash à l'écriture** vérifiés ; scellés reproduits (**re-génération post-run journalisée**, pas de lecture pré-run) ; disjonction (a)+(b) **des 170 entre eux et vs 450 existants** ; **distribution 2–5 respectée** ; sampler = option B gelée (pas de rejet doublon ajouté) ; **`attempts_total < 200_000` prouvé** ; compteurs et fréquence de doublons publiés ; **artefacts bruts + checkpoints par cellule présents et hashés** (re-calcul indépendant possible) ; sortie isolée `artifacts/v1-confirm/`.
2. **Instrument** : un seul open du test (garde), log avant open, closed+hash après, aucun re-hash pré-run ; log isolé complet.
3. Re-calcul indépendant des stats du primaire (point + IC apparié) ; cohérence 5 seeds.
4. Checkpoint final à budget fixe, aucune sélection ; pas de measure-shopping.
5. Verdict consigné (pré-enregistré).

### 7ter-quinquies. Revue statique runner_confirm v5 (88ff02b) — 1 BLOQUANT + 2 MAJEURS

Vérifié file:line (aucun test2 ouvert ; patch en cours dans le working tree — revue à refaire sur commit propre).

| # | Défaut (file:line) | Correction attendue | Test DEV |
|---|---|---|---|
| B1 (BLOQUANT #23) | `_freeze_manifest_sha` hache **physiquement le couples** (`fm["inputs"]` → `_sha256_file(spec["path"])`, `runner_confirm.py:146-150`) **avant** le log d'intention, puis `main` fait `reg.read_once(args.couples_file, …)` (`:266`) → **2 FD sur couples**, dont un **hors registre** | préflight : **ne PAS hacher physiquement couples** — comparer le SHA attendu du freeze-manifest/CLI seulement ; l'**unique** `read_once` (hash complet) après le log d'intention + fsync | spy `open` sur couples : **count == 1**, ordre intent → open ; aucun FD couples avant le log |
| B2 (MAJEUR G2) | la sélection des checkpoints reste **glob-driven** : `_ck_path(args.canon_ckpt_pattern, seed)` / control (`:289-290`) après l'open test2, **non liée** au manifeste vérifié → un fichier rogue matché plus tard peut être chargé malgré un manifeste propre | dériver **exactement les 10 chemins du manifeste** par (rôle, seed) — unicité rôle+seed exigée, **fail pré-open** si absent/doublon — et les passer **directement** à `build_arm` ; les patterns wildcard ne gouvernent **pas** la sélection confirmatoire | DEV : sélection effective == manifeste (chemin par chemin) ; rôle/seed manquant/dupliqué → abort **avant** l'open ; un fichier rogue matchant le glob n'est **pas** utilisé |
| B3 (MAJEUR stage-B) | le **DATA manifest** (test2 sha64) publié avant le run n'est **pas chargé/vérifié** : le freeze lie méthode+pattern (`:399`) mais `_failfast` ne compare pas `--test2-hash` au manifeste DATA | **précision lead 02:38** : **ne PAS lier le sha du DATA manifest au freeze stage-A** (inconnu à cette date) — charger/vérifier un **DATA manifest publié/committé ENTRE génération et l'unique run**, comparer **chemin test2 + sha64** à la CLI **AVANT** l'open, puis `read_once` compare le hash **physique** pendant l'unique ouverture | DEV : DATA manifest absent / chemin ou sha mismatch → abort **avant** l'open ; `read_once` vérifie le physique à l'ouverture |

### 7ter-sexies. Revue v6 (1972616) — B3 encore partiel

| # | Défaut (file:line) | Correction attendue | Test DEV |
|---|---|---|---|
| B3bis | `_failfast` lit le DATA manifest mais ne compare que `attested == args.test2_hash` (`runner_confirm.py:198-206`) : **pas** de comparaison `dm['path'] == args.test2_file` ; fallback `dm.get(dm_spec['method']) or dm.get('sha256_write_stream') or dm.get('sha256')` **tolère une méthode non figée** ; les tests créent un manifest sans `path` (faiblesse validée) | exiger **`dm['path'] == args.test2_file`** (normalisation stable selon convention figée) ET le champ **exact `sha256_write_stream`** (méthode figée, **aucun fallback**) ; test **chemin ou méthode mismatch → abort avant ouverture** | DEV : DATA manifest sans `path`/mauvais path → abort ; méthode ≠ `sha256_write_stream` → abort ; chemin+sé méthode corrects → passe |

### 7septies. Audit EXPLORATOIRE — finding de provenance (hashes erronés)

**Nuance statistique (lead 05:15) — formulation sûre obligatoire** : `ctrl−pretrain` +0.8 pp IC [−13.3,+13.9] **inclut 0** → cela **ne prouve NI l'équivalence NI l'absence de composante décisionnelle** ; il n'y a **pas de signal distinguable à cette puissance**. Formulation retenue : « **contrôle comparable observé, mécanisme exposition/optimisation plausible, aucune attribution décisionnelle établie** » — **jamais** « contrôle égale le pré-entraîné » (pas de test d'équivalence). `ctrl−scratch` +13.37 pp IC [3.77,22.76] hors zéro = différence observée, à ne pas transformer en attribution. **IC exploratoire non recalculable indépendamment** (pas de raw per-episode).

Cause statique exacte (confirmée file:line, aucun scellé test ouvert) : en mode `--couples-pattern`, `sealed_slices[k] = couples_to_records(...)` par budget (`runner.py:169-176`) ; `records = sealed_slices[max]` (1191) puis `budgets = UniqueCoupleBudgets(records)` (`:176-177`) ; les hashes publiés calculent `_recs_hash(budgets.slice(k))` (`:188-190`) alors que **l'entraînement** utilise `slice_records = sealed_slices[k]` (`:246`).

- **Conséquence** : `materialized_records_hashes_per_budget` = **hashes de provenance ERRONÉS** — ils ne tracent PAS les slices d'entraînement ; **doublon k=2000/k=10000** expliqué (`slice(2000)==slice(10000)` sur 1191 uniques).
- **Limite d'audit** : le code **prouve le chemin d'entraînement**, pas l'identité a posteriori du contenu sans raw/hash fiable. Ne pas nier l'effet en extrapolant ; les chiffres du run (cellules/AULC/primaire) restent auto-rapportés exploratoires.
- **Valeurs canoniques** citées (faf8810a/bb8c227c/eb4901ed/5bf5de9c) : source **déjà publiée** = index WS-B `artifacts/siw-runner-artifacts-index.json` → `couples_materialized_files[...].materialized_sha256_canonical` (vérifié à l'audit WS-B). Ma vérification récente a **re-matérialisé les 4 fichiers couples** (adaptation, PAS le test) pour comparer → lecture d'audit **rétro-divulguée** (intention+registre requis ; correction côté tagi-5).

### 7ter-septies. Revue v6 (f96c940) — B3bis path/méthode OK ; edge HEX à corriger

Vérifié statiquement (aucun test2 ouvert ; `tests/test_confirm_v3.py` non rejoué — run ciblé **timeout 1200 s**, tests E2E lourds : audit statique complet, exécution à refaire sur sélection légère).

| # | Constat (file:line) | Correction | Test DEV |
|---|---|---|---|
| B3bis | **corrigé** : `dm.get("path")` manquant → abort ; `_norm(dm['path']) == _norm(--test2-file)` ; méthode **stricte** `dm_spec['method']` (plus de fallback) ; `attested` comparé à la CLI | — | 3 tests négatifs (mauvais chemin, mauvaise méthode, path manquant) → abort pré-open |
| B1 | **pas de régression** : `_freeze_manifest_sha` **skippe `couples`** (`if key == "couples": continue`), sha comparé par **string** dans `_failfast` ; unique `read_once` après intent | — | spy couples == 1 FD + ordre intent |
| B2 | **pas de régression** : boucle cellules = `args._bound_ckpts[f"canon\|s{seed}"]`/`control\|s{seed}` (chemins liés role\|seed), **zéro glob** | — | rogue-glob jamais chargé |
| B3bis-edge (MAJEUR) | `len(attested) != 64` **sans regex hex** (`:217`) : `'Z'*64` accepté si CLI/DATA concordants → `_failfast` passe, `read_once` **ouvre test2** puis échoue le hash → **cellule consommée pour un arg malformé** (idem `couples_hash` CLI non-hex → ouvre couples) | valider `re.fullmatch(r"[0-9a-f]{64}", …)` du **hash DATA** ET des **hashes CLI test2/couples** (protocol = recalculé/comparé ✅) **AVANT toute ouverture scellée** | DEV : valeur 64-char non-hex → **abort, 0 FD** (spy) ; valeur 63/65 → abort |

**Writer réel** : `write_selfcontained_episodes` retourne `{path, episodes, sha256_write_stream, bytes}` ; `build_freeze_manifest` fixe `generation.test2_data_manifest = {method: "sha256_write_stream", manifest_path: pattern}` → clés cohérentes ✅.

---

*DRAFT v4.21 tagi-5 — §7ter-ter-9 NB : + création **exclusive** du freeze (`open(...,'x')`/O_EXCL, abort si préexistant avant write, test DEV freeze préexistant → sha inchangé) ; statut **defense-in-depth** (l'indispensable = commit/publication du freeze avant génération + binding `--protocol-hash`). Inclut v4.20 (builder doit valider `_verify_source_manifest` avant freeze/génération), v4.19, v4.18, v4.17, v4.16, §7septies. À adopter/amender ; aucune génération/ouverture test2.*
