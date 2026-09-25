# QUATRIÈME REVUE EXTERNE — audit ligne-de-code du chemin officiel V1-bis (reçue 23/09/2026 ~17:30)

*Modèle externe différent, sollicité par cgarrot. Audit lecture seule sans pytest, sans modification. Verbatim archivé (quelques artefacts de tokenisation du modèle d'origine conservés tels quels : « 공식 », « berhenti », « résamplages », « psychosis », « dolk », «小额 », etc.). Verdict global du lead après vérification directe : CONFIRMÉ sur les 3 bloquants majeurs (store 0/26, k-cells cumulatives, init non appariée) — message mesh m_muea5pho_81e397ff. Conséquences actées : deadline 20:00 ANNULÉE, GO glissé à demain matin, HOLD test2 intégral, test d'acceptation E2E couples⊆store obligatoire partout.*

---

## Verdict global

Ce projet est scientifiquement sérieux et il y a une vraie intuition produit derrière lui. Mais le chemin V1-bis « officiel » n'est pas encore assez fiable pour produire un résultat confirmatoire.

Mon avis franc :
- La preuve V0 est solide, mais uniquement dans la famille TinyGraphKey.
- Le concept de transfert vers une UI synthétique est pertinent, mais le bridge SIW actuel est encore très simplifié.
- Le run V1 n'est pas un résultat négatif ni positif : il n'a pas testé validablement ce qu'il prétendait tester.
- V1-bis est une bonne reconstruction méthodologique, mais son instrumentation n'est pas encore au niveau du protocole qu'elle annonce.
- Le plus grand risque n'est plus scientifique : c'est qu'un run officiel produise un chiffre apparemment sérieux avec un contrat d'entraînement ou d'évaluation non respecté.

Recommandation claire : maintenir le HOLD sur test2 et terminer le chemin DEV complet avant toute confirmation.

## 1. Ce que vous êtes réellement en train de construire

L'idée est la suivante :

Demande complexe → gros modèle (comprend et décompose) → objectif structuré → petit UCM (choisit et exécute les actions) → environnement → résultat vérifié.

UCM n'est pas un LLM et ne génère pas de texte. C'est une petite politique conditionnée par un but, qui : 1. reçoit un état structuré ; 2. reçoit les actions candidates ; 3. score chaque action complète ; 4. en sélectionne une ; 5. répète la boucle jusqu'à STOP, erreur ou timeout. Documenté dans docs/VISION.md:19-42.

En interne : petit GNN relationnel. Entraînement par imitation d'un oracle exact, évaluation en closed loop.

Thèse la plus précise : **Un petit réseau relationnel peut apprendre une politique de contrôle closed-loop à faible coût local sur des environnements synthétiques fully observable.**

Ce n'est pas encore : un planificateur explicite ; un agent avec mémoire ; un système qui sait demander de l'aide ; un système de récupération après erreur ; un contrôleur universel ; encore moins une forme d'AGI. Le modèle actuel est surtout une policy state+goal → action. Comportement séquentiel en boucle fermée, mais pas de mémorisation d'historique ni planification explicite.

## 2. État réel des phases

| Phase | État | Interprétation |
|---|---|---|
| TinyGraphKey V0 | Réalisé et convaincant | Généralisation dans une seule famille synthétique |
| SIW | Réalisé, très simplifié | Benchmark de structures UI, pas un proxy validé du logiciel réel |
| V1 M-V1b | Exécuté, invalide pour confirmer le transfert | Le transfert multi-étapes n'a pas été testé correctement |
| V1-bis R* | En préparation | Meilleure distribution d'adaptation, contrat officiel incomplet |
| Jev / Muse | Hors dépôt | Intégration non réalisée |
| DOM → UCM | Absent | Le pont produit principal n'existe pas |
| Mémoire / récupération / abstention | Conceptuels | Pas de produit fiable |

## 3. Ce que V0 prouve réellement

Meilleur accomplissement scientifique du dépôt : B144, ~695k params, ~97,5 % closed-loop, ~1 ms/décision, bonne recombinaison des objectifs (docs/REPORT-V0.md:10-31). Limites correctement déclarées (REPORT-V0.md:71-74).

Deux corrections documentaires :
1. La spec définit R1 = généralisation intra-famille, R2 = transfert inter-familles (spec:76-83) — V0 ne soutient que R1, pas « R1-R2 » comme écrit dans REPORT-V0.md:73.
2. Dérive documentaire : REPORT-V0.md:56 annonce 83 échecs ; l'artefact officiel = 79 ; l'erratum reconnaît la différence (replay M3) ; le regret est présenté à 0,14 alors que l'artefact donne ~0,0558. Réconcilier avec une source de vérité unique.

## 4. Le problème majeur de V1 : le test ne testait pas le transfert multi-étapes

Le +12,57 pts à k=500 n'est pas interprétable comme transfert : couples d'adaptation quasi exclusivement d'états initiaux ; décisions post-action non représentées ; le contrôle Null apprenait la validité des actions ; l'essentiel du gain porté par STOP ; test exact 5 seeds p≈0,094 non significatif ; raw par épisode absents. Références : docs/AUDIT-CONSTRUCT-VALIDITY-V1.md:7-15, :19-41 ; docs/REPORT-V1-EXPLORATORY.md:47-68.

Formulation correcte = celle du HOLD : « Le transfert d'une politique multi-étapes n'a pas été testé valablement dans M-V1b. » Vous avez invalidé votre propre run — vrai point fort.

## 5. Blockers actuels de V1-bis

### 5.1 Store de layouts incompatible (bloquant absolu)
Le freeze déclare artifacts/inventory-siw-dev-layouts.json (freeze-v4:390-392). Les 10 fichiers d'adaptation existent, SHA conformes, utilisent 26 layouts ; le store contient 30 AUTRES entrées ; les 26 sont absents. L'adapter fait store[c["layout_hash"]] (data_adapter.py:63-75) → preload_store() réussit, SHA store OK, puis KeyError au premier couple. Le manifest peut sembler valide tout en étant incompatible avec les données à entraîner.

Correction : store complet des 26 layouts, OU couples auto-suffaisants (layout inline), OU spec complète versionnée par couple. + test d'acceptation : set(layout_hash des couples) ⊆ set(clés du store), vérification SIWLayout.layout_hash() par entrée.

### 5.2 Cellules k=64/128/256 non indépendantes
Le protocole exige des cellules indépendantes de 2000 updates par bras×seed×k (PROTOCOL-V1BIS-v2:15-21, ERRATUM:3-8). Mais le runner officiel crée UN modèle puis fine-tune successivement k=64, puis k=128 sur le modèle déjà entraîné, puis k=256 (runner_official.py:139-150 ; idem runner_parallel.py:71-80). Cela mesure un entraînement CUMULATIF, confondant temps de calcul/optimisation/historique avec la taille du dataset.

Correction : par seed : base fraîche → copie par bras → copie par k → charger poids source sur la copie du bras → 2000 updates exactement → publier le hash d'initialisation de chaque cellule.

### 5.3 Initialisation appariée par seed non implémentée
Le runner construit le modèle PUIS appelle mx.random.seed(9000+seed) (runner_official.py:35-58, :139-144). Les embeddings SIW/têtes/nouveaux paramètres peuvent différer entre bras. L'infrastructure existe : build_arm(..., fresh_init=...) (transfer.py:163-209).

### 5.4 Le runner officiel ne fait pas l'évaluation
eval_refuse(None) retourne None (runner_official.py:84-91) ; le full path retourne {"status": "full_run"} sans évaluation closed-loop, lecture test2, raw, checkpoints, bootstrap, test exact, rapport, publication atomique. Un statut « full_run » peut être produit sans le test scientifique. Renommer : instrument_verified / training_complete / parallel_training_complete / official_complete (le dernier seul = expérience confirmatoire exécutée).

### 5.5 Le freeze ne couvre pas le vrai chemin officiel
BOUND_CODE (freeze_v1bis.py:18-53) ne contient PAS runner_official.py, runner_parallel.py, seal_test2.py, test2_generation.py, interactions_merge.py. Le freeze actuel : hash canonique conforme, 30 checkpoints conformes, zéro drift sur les 29 modules liés — mais les modules qui exécutent réellement le run et le scellement ne sont pas liés. Calculer le SHA juste avant chargement des checkpoints, pas seulement à la construction du manifest.

### 5.6 Le scelleur test2 non validé de bout en bout
Corrections locales réelles (import generate_test2_episodes ; retour dict ; constante de hash globale) mais chemin complet non prouvé : validate_m1() attend 600 épisodes, le smoke en fournit 2 factices puis pytest.skip() (test_seal_test2.py:84-124) — le test peut « ne pas échouer » sans que le full path soit publishable. CLI toujours dry_run=True (seal_test2.py:345-350) ; disjonction v1bis-gen hash-only pas exactement isomorphique (test2_generation.py:49-77) ; publication relit le fichier plusieurs fois (seal_test2.py:192-215) ; journal recovery avale une queue partielle au lieu de refuser l'état ambigu (:40-75) ; sortie par défaut sous /tmp (:37). Docstring obsolète (annonce stub alors qu'un chemin réel existe).

## 6. Contrat de checkpoint contradictoire
Erratum : « best_validation sur un split de validation de l'adaptation » (ERRATUM:5-6). Code : checkpoint_policy = "final_at_fixed_budget" (transfer.py:146-155), aucun split de val d'adaptation créé/évalué. Choisir explicitement ; recommandation : final à budget fixe (plus simple, aucune sélection, plus reproductible).

## 7. L'analyse primaire annoncée n'est pas implémentée
Le protocole demande : agrégation égale des 4 prédicats par seed ; bootstrap seed-cluster 10 000 rééchantillonnages ; IC unilatéral ; test exact des signes ; PASS/FAIL/indéterminé selon règles figées (PROTOCOL:23-28). Le module actuel fournit un bootstrap bilatéral générique (metrics.py:102-132). Il manque la fonction canonique : groupe les 600 épisodes par seed et prédicat → calcule l'estimand exact → IC unilatéral → test exact → verdict → rapport depuis le raw. Tant que le verdict n'est pas une fonction testable des résultats bruts, le protocole reste une intention.

## 8. Puissance statistique fragile
SD exploratoire 10,9 pts → 10 seeds ≈ 80 % de puissance pour un effet de 10 pts, mais le seuil PASS est 5 pts. Calibré pour un transfert fort, pas pour démontrer fièrement +5 pts. Ne pas interpréter « indéterminé / IC contient zéro » comme « aucun transfert ». Maintenir la nuance jusque dans le rapport final.

## 9. Distribution R* meilleure, claim à rester étroit
Audit DEV : ~5,9 % de couples 0/0 dans le training vs ~55,4 % de décisions visitées 0/0 ; couverture de masse par état ~12,5-14,5 % ; par signature ~68,6-70,6 % (AUDIT-DISTRIBUTION-V1BIS-DEV.md:26-45). R* corrige le défaut initial-only — progrès réel — mais ne couvre pas toute la distribution du closed loop. La restriction aux cellules de masse ≥0,5 % (PROTOCOL:5-13) est défendable ; le claim en cas de succès sera : « le préentraînement TGK améliore la politique SIW fermée après adaptation R* dans un sous-espace fréquent de cellules » — PAS « UCM se transfère universellement ». Restriction scientifique et marketing importante.

## 10. SIW trop simple pour la vision produit
1 interface, 2 champs, 2 options, peu de transitions, distracteurs sans effet, labels non interprétés (tensorize_siw.py:1-6 ignore volontairement les labels), pas de saisie textuelle arbitraire, pas d'API error, pas de permission, pas d'état partiellement observable. Correct pour tester le binding par référence ; ne teste pas la compréhension des labels, le mapping DOM, les textes arbitraires, les actions avec copie, les erreurs logicielles, le contenu dynamique. SIW = benchmark synthétique de structure UI, pas encore un proxy du web.

## 11. Vision produit pertinente mais non implémentée
VISION.md:27-35 évoque in-context, récupération, escalade, « je ne sais pas ». Or le modèle n'a ni mémoire, ni historique, ni action ASK_HELP, ni calibration d'incertitude, ni abstention explicite, ni rollback. STOP = « tâche terminée », pas « je ne sais pas ». À ajouter plus tard : score d'incertitude, politique d'abstention, escalade vers Jev, mémoire courte, détection de boucle, reprise. Pas le prochain chantier : finir V1-bis d'abord.

## 12. Gaps d'ingénierie et reproductibilité
- README 12 lignes, décrit encore V0 (README.md:1-12) ; spec dit encore « aucun modèle implémenté » (spec:5-8, :43-45) ; plusieurs sources de vérité se contredisent.
- Manifests avec chemins absolus /Users/cgarrot/... ; checkpoints ignorés par Git → un clone propre ne peut pas reproduire les runs ; artifacts/ ~740 Mo ; aucun pyproject.toml/requirements/lockfile/CI ; reproductibilité dépendante d'un venv local macOS/MLX.
- Qualité des tests : plusieurs vérifient surtout la présence de chaînes/nombres/import — ne prouvent pas l'indépendance des cellules k, l'init commune, la détection d'un checkpoint modifié avant chargement, le full seal path sans skip, l'inclusion exacte support↔store, la publication de raw/verdicts depuis une exécution réelle.
- Manque un STATUS.md unique : V0 terminé (claim R1) ; V1 exploratoire invalide ; V1-bis protocole candidat ; test2 absent ; runner officiel non exécutable ; store incompatible ; freeze actif aucun ; seeds réservés intouchables.

## 13. Ce que j'aime réellement
- Discipline scientifique exceptionnelle (invalidation de ses propres résultats dès que le construit ne tient pas — rare).
- Séparation des canaux saine (policy_input/supervision/provenance).
- Oracle indépendant de l'évaluateur (pas d'accès à d*, plan, succès, next-state).
- Publication atomique bien pensée (bundle, pointer, no-replace, fsync, recovery).
- Limites documentées (pas de prétention AGI/universelle/web-ready).
- Taille comme contrainte plutôt que dogme.

## 14. Plan recommandé
Priorité immédiate : PAS de nouvelle architecture — terminer la chaîne existante.
1. Snapshot propre (committer les corrections locales, documenter la version auditée).
2. Réparer le contrat données (store complet 26 layouts + hashs + test d'acceptation store↔couples).
3. Réécrire le runner (base fraîche/seed → copies par bras et k → 2000 updates → SHA des poids avant chargement → statuts non ambigus).
4. Implémenter le vrai Stage-B (lecture unique test2, 600 raw, checkpoints finaux, bundle atomique, stats depuis raw, test exact, verdict canonique).
5. Fixture E2E DEV complet (génération synthétique → store → couples → 4 bras × 2 seeds × 3 k = 24 cellules indépendantes → raw → checkpoints → stats → rapport → publication → relecture du bundle), incapable de skipper une erreur du chemin officiel.
6. Seulement ensuite : test2 (audit indépendant, commit propre, seed 20261003, scellement, UN run, publication, aucune reprise silencieuse).

## 15. Opinion finale
- Scientifique : très prometteur (politique structurée compacte ; question bonne ; discipline de preuve réelle).
- « Universal Control Model » : nom trop grand pour l'état actuel — c'est aujourd'hui « un contrôleur relationnel compact sur des mondes structurés, avec une vision d'exécution d'interfaces ». Contribution valable.
- Produit : pertinent mais préliminaire ; le prochain saut réel = adaptateur DOM/accessibility tree + calibration/abstention + escalade Jev + test sandbox réel + comparaison « LLM à chaque décision » vs « LLM planificateur + UCM ».

| Axe | Avis |
|---|---|
| Discipline scientifique | Très forte |
| V0 | Réussi, claim étroit |
| V1 | Non interprétable comme preuve de transfert |
| V1-bis | Bonne idée, instrumentation pas encore fiable |
| Produit | Encore très préliminaire |
| Repro locale | Moyenne |
| Repro depuis clone propre | Insuffisante |
| GO test2 aujourd'hui | NON |

« Le projet ne doit pas être abandonné. Il doit être recentré. Le prochain jalon : un chemin DEV complet, fail-closed, reproductible et statistiquement correct, prouvant que le protocole est réellement celui que le code exécute. Une fois cela fait, le prochain test2 aura enfin une vraie valeur. »

*Audit réalisé sans pytest et sans modification de fichiers. Le worktree n'est pas propre et évolue rapidement ; conclusions à revalider sur le commit exact choisi avant toute action officielle.*
