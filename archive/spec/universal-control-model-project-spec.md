# Universal Control Model — spécification de recherche révisée

## Petit contrôleur structuré, conditionné par but, sans génération autoregressive de texte

**Version :** 0.2 — 22 septembre 2026.  
**Statut :** protocole proposé ; aucun modèle, dataset ni benchmark implémenté par cette révision.  
**Matériel principal :** Apple M5, 32 Go de mémoire unifiée.  
**Runtime initial proposé :** MLX ; CPU et GPU à profiler.  
**Matériel secondaire optionnel :** RTX 3070 laptop 8 Go, seulement si un besoin concret justifie un second backend.  
**Nom court expérimental :** UCM ; « Universal » reste une ambition, **pas un résultat**.

Documents associés : [revue critique en 16 sections](reports/ucm-review/revue-critique-fr.md), [bibliographie annotée](reports/ucm-review/sources.md), [version originale conservée](reports/ucm-review/universal-control-model-project-spec.original.md).

---

## 1. Statut, objectif et discipline de preuve

### 1.1 Objectif scientifique

Déterminer si une petite politique sur observations structurées et actions typées peut :

1. exécuter correctement des tâches en boucle fermée sur des instances nouvelles ;
2. rester peu coûteuse **en incluant toute l'interface** ;
3. après préentraînement, s'adapter à de nouvelles dynamiques avec un avantage mesurable sur un scratch correctement entraîné.

Le modèle vise la décision, pas la génération de prose, la connaissance encyclopédique ou la programmation arbitraire. Un LLM peut fournir des données hors ligne, traduire exceptionnellement une demande ou servir de secours déclaré. Il ne doit pas être nécessaire à chaque décision du chemin rapide.

**V0 ne teste que le point 1 et le coût local du point 2. Le point 3 nécessite une expérience distincte.**

### 1.2 Distinctions obligatoires

- **Établi :** les politiques conditionnées par objectif, l'imitation, les ensembles/graphes, les pointeurs, la distillation et la planification latente ont des antécédents importants.
- **Hypothèse UCM :** un transfert utile peut subsister à très petite taille et sous contrôle strict des interfaces, données et calculs.
- **Extrapolation :** un résultat structuré pourrait aider une classe d'interfaces logicielles.
- **Spéculation :** un même cœur servirait de “System 1 universel” pour Web, jeux, bureautique et robotique.

Aucun seuil de cette spec n'est une performance déjà obtenue. Les seuils sont des décisions de recherche proposées ; toute modification après consultation de résultats doit créer une nouvelle version du protocole et une nouvelle confirmation scellée.

### 1.3 Définition de non-autoregressif

Une décision V0 évalue en parallèle les candidats complets, puis sélectionne un élément. Elle ne déroule pas une chaîne de tokens de sortie. La boucle environnementale reste séquentielle. Un modèle à diffusion ou une recherche à plusieurs appels peut être non autoregressif au sens textuel tout en ayant un coût itératif ; déclarer le nombre d'appels.

### 1.4 Limite de déploiement

Aucun entraînement, accès réel à des comptes, automatisation destructive, API payante ou déploiement n'est autorisé par le seul fait que cette spec existe. Le travail demandé ici est documentaire ; l'implémentation et les expériences feront l'objet d'une étape ultérieure.

---

## 2. Hypothèses et niveaux de revendication

### 2.1 Formulation

```text
observation observable o_t --A_e--> x_t
objectif structuré g
candidats publics C_t
mémoire m_t, uniquement dans une extension motivée

πθ(x_t, g, C_t, m_t) --> scores des actions complètes
```

L'adaptateur `A_e` ne reçoit pas le plan expert ni les labels. La génération de candidats ne reçoit pas le but. Le cœur n'utilise pas d'identifiant arbitraire de domaine comme raccourci. Les attributs réellement observables et nécessaires à la dynamique restent autorisés et documentés.

### 2.2 Registre d'hypothèses

| ID | Hypothèse | Falsification prioritaire |
|---|---|---|
| H0 | Le schéma conserve l'information nécessaire | Entrées identiques exigeant des actions différentes sans mémoire disponible |
| H1 | Le problème n'est pas résolu par les candidats | Candidats seuls/sans but presque aussi performants |
| H2 | Une petite politique apprend le contrôle de base | Échec de surapprentissage ou réussite fermée insuffisante après correction mécanique |
| H3 | La structure relationnelle aide | Pas de gain après retrait des relations ou contre une heuristique simple |
| H4 | Le préentraînement aide l'adaptation | Échec du critère primaire de transfert défini en §12 |
| H5 | La diversité source explique ce gain | A+B ne dépasse pas les sources seules à budget égal |
| H6 | Une extension apporte une valeur marginale | Mémoire/recherche/langage/hiérarchie sans gain sur le défaut ciblé |

### 2.3 Niveaux autorisés de conclusion

- **R0 :** pipeline correct, sans résultat de généralisation.
- **R1 :** généralisation à de nouvelles instances d'une même famille.
- **R2 :** transfert vers les dynamiques/familles explicitement testées.
- **R3 :** réplication externe ou sur plusieurs familles indépendamment construites.

Même R3 n'autorise pas une universalité hors support. De nouvelles sémantiques d'action non observables ne sont pas identifiables sans conventions, démonstrations, descriptions ou exploration.

---

## 3. Périmètre et non-objectifs V0

### 3.1 Inclus

- Un environnement déterministe pleinement observable, TinyGraphKey.
- Trois types d'objectifs structurés.
- Actions complètes à arguments finis, STOP compris.
- Oracle exact hors ligne et évaluateur indépendant.
- Un seul réseau minimal A entraîné par imitation.
- Baselines classiques, tests de contamination/invariance et évaluations fermées.
- Profiling réel de la chaîne de décision.

### 3.2 Exclus

Pas de langage naturel, pixels, audio, navigateur réel, ALFWorld, robotique, mémoire, RL, valeur/progression auxiliaire, world model, options, latent skills, diffusion ou LLM runtime.

Pas de schéma prétendant couvrir immédiatement tous les outils. Pas d'objectif arbitraire de 10–20M paramètres. Pas de trois architectures implémentées simultanément. Pas de framework multi-domaines avant la première expérience interprétable.

### 3.3 Sens du premier résultat

Réussir V0 établit qu'une interface explicite permet l'apprentissage d'un contrôle fermé sur de nouvelles instances. Cela n'établit ni transfert inter-familles ni supériorité sur un agent LLM, encore moins intelligence générale.

---

## 4. TinyGraphKey : environnement normatif

### 4.1 Objets et état

Un layout contient :

- `R` pièces, graphe non orienté connecté, sans arêtes multiples ; `R=4..8` au train ;
- exactement une arête portant une porte ;
- un agent localisé dans une pièce ;
- deux objets transportables distincts : une clé et un colis ;
- une porte dont l'attribut dynamique est `locked ∈ {true,false}` ;
- une relation statique liant la clé à cette porte.

Chaque objet est soit dans une pièce, soit porté. Un seul objet au maximum peut être porté. Il n'existe pas de capacité de pièce, d'orientation, de distance métrique cachée, de hasard ou d'autre état dynamique.

L'observation contient la topologie complète, la position de l'agent et des objets, la possession, les extrémités de la porte, son état et la relation clé-porte. L'état est intentionnellement symbolique et privilégié par rapport à un agent visuel : **cette simplification fait partie du périmètre déclaré**.

Les IDs sont des références locales de liaison, pas des catégories apprises. Les pièces ne sont pas numérotées dans un ordre de chemin vers le but.

### 4.2 Buts

| Prédicat | Arguments | Condition de satisfaction |
|---|---|---|
| `REACH` | pièce `r` | agent dans `r` |
| `HAVE` | objet `o` | agent porte `o` |
| `AT` | objet `o`, pièce `r` | `o` posé dans `r`, pas simplement porté par un agent dans `r` |

Le but fait directement référence aux entités : aucun parsing linguistique n'est évalué. Les trois types sont équilibrés au niveau des épisodes d'entraînement, sous les restrictions de split. La tâche n'impose aucune préférence esthétique entre plans de même coût.

### 4.3 Actions et transitions

Tous les candidats ci-dessous sont toujours énumérés pour les entités du layout, même lorsque leurs préconditions sont fausses.

| Action | Précondition pour effet valide | Effet |
|---|---|---|
| `MOVE(r)` | `r` adjacent à la pièce courante ; si l'arête porte la porte, celle-ci est déverrouillée | Déplace l'agent vers `r` ; l'objet porté reste porté |
| `PICK(o)` | `o` posé dans la pièce courante et main vide | `o` devient porté |
| `DROP(o)` | `o` actuellement porté | Pose `o` dans la pièce courante |
| `UNLOCK(d)` | agent à l'une des deux extrémités de `d`, porte verrouillée, clé correspondante portée | Porte déverrouillée définitivement ; clé reste portée |
| `STOP` | Aucune condition pour être sélectionnée | Termine immédiatement ; succès si et seulement si le but est satisfait |

Une action physique invalide ne modifie pas l'état, retourne un résultat `invalid` et **consomme un pas**. `MOVE` vers la pièce courante est invalide ; `UNLOCK` d'une porte déjà déverrouillée est invalide. La clé n'est ni consommée ni automatiquement déposée. Pas d'action CLOSE, création, téléportation ou macro.

Avec deux objets et une porte, le nombre de candidats est `K = R + 6`. L'ordre de candidats est randomisé, avec liaison correcte des labels et références. Aucun masque de validité physique n'est fourni dans le bras principal : seul le padding est masqué.

### 4.4 Coût et terminaison

Chaque décision, STOP incluse, coûte un pas. Horizon runtime fixe `H=64` décisions, non fourni comme compteur à la politique.

- STOP avec but vrai : terminal, succès.
- STOP avec but faux : terminal, échec `premature_stop`.
- Après 64 décisions sans STOP : terminal, échec `timeout`, même si le but physique vient d'être atteint.
- Toute action invalide compte dans la longueur et dans le taux d'invalides.

Reporter séparément « but atteint physiquement » et « tâche terminée correctement ». Aucun vérificateur ne corrige silencieusement un STOP prématuré pour sauver le score.

### 4.5 Génération et reachability

Générer des graphes connectés avec variété de branchements et cycles ; imposer une fraction documentée de portes sur ponts pour qu'une partie des tâches nécessite réellement la clé. Ne pas conserver seulement des épisodes où la porte est sans importance.

Les états initiaux respectent capacité et types. L'oracle vérifie la solvabilité. Les rejets de génération et leur motif sont comptés ; ils ne sont pas supprimés des rapports de coût. Les tâches insolubles sont hors population principale V0 ; une future étude d'abandon devra les inclure explicitement.

Train principal : distance optimale physique `d*=2..12`. Cas déjà satisfaits `d*=0` séparés : environ 10 % des épisodes train peuvent les couvrir ; ils ne gonflent pas les scores G1/G2 non triviaux. Les états de fin de démonstration incluent toujours une décision STOP supervisée.

Les cellules de test à horizon long demandent `d*=13..24` lorsqu'elles existent sous la sémantique choisie. Avant gel du protocole, un inventaire oracle de développement doit démontrer l'existence d'assez de tâches dans cette bande. Si elle est trop rare ou impossible, publier l'inventaire et revoir le générateur **avant** de créer/ouvrir le test ; ne pas inventer un benchmark long horizon.

### 4.6 Oracle exact et multiples optimums

À layout fixé, borne brute des états physiques : `2R(R+1)^2` ; exclure notamment les deux objets simultanément portés. À `R=12`, cette borne vaut 4 056. L'espace état-but est plus grand et le nombre de layouts n'est pas inclus dans cette borne.

Construire les transitions physiques déterministes puis un BFS inverse multi-source depuis tous les états satisfaisant le but. Les actions invalides sont des self-loops de coût positif ; STOP est traité séparément.

```text
d*(s,g) = nombre minimal d'actions physiques avant satisfaction
L*(s,g) = d*(s,g) + 1                       # inclut STOP
A*(s,g) = {STOP}                           si d*=0
A*(s,g) = {a : d*(T(s,a),g)=d*(s,g)-1}     sinon, si d* fini
```

Un état sans chemin vers le but est marqué `unreachable` ; ne jamais lui fabriquer une action experte. Dans ce MDP, toute action de A* fait strictement décroître d* ; plusieurs optimums ne permettent pas une boucle optimale.

Tests oracle : petits layouts vérifiés indépendamment par recherche avant/arrière, cas manuels avec récupération de clé/dépôt, Bellman sur toutes les transitions de ces layouts, STOP et invalides. Le modèle exact disponible à l'oracle est une connaissance privilégiée de dynamique ; la baseline de recherche l'utilisant doit être décrite comme telle.

---

## 5. Contrat d'information et schéma de données

### 5.1 Trois canaux séparés

| Canal | Contenu | Peut entrer dans le modèle V0 ? |
|---|---|---|
| `policy_input` | But, entités/attributs observables, relations, candidats syntaxiques | Oui, via allowlist stricte |
| `supervision` | A*, d*, succès, état suivant, labels de qualité | Non |
| `provenance` | Layout canonique, seed, version, split, source, numéro de pas | Non |

L'évaluateur dispose de la vérité de l'environnement. L'adaptateur ne dispose que du canal observable. Les logs d'exécution peuvent contenir action précédente et résultat, mais ils ne sont pas des entrées de la V0 sans mémoire.

**Champs interdits en entrée :** reward, progress, distance au but, succès/terminal oracle, plan, prochaine observation, expert, masque optimal, timestep, compteur de budget, numéro d'épisode, split et identifiant de générateur/domaine.

Le schéma doit distinguer absent/inconnu/false dans les extensions ; en V0 aucune information physique listée n'est inconnue.

### 5.2 Exemple de structure de record

Exemple documentaire, pas code exécutable ni spécification d'une librairie :

```json
{
  "schema_version": "0.2",
  "policy_input": {
    "goal": {"predicate": "AT", "object_ref": "obj_b", "room_ref": "room_q"},
    "entities": [],
    "relations": [],
    "candidates": []
  },
  "execution": {
    "behavior_action_ref": "candidate_ref",
    "observable_result": "valid_or_invalid",
    "next_raw_observation_ref": "content_hash"
  },
  "supervision": {
    "optimal_action_refs": [],
    "distance_physical": 0,
    "reachable": true
  },
  "provenance": {
    "layout_hash": "canonical_layout_hash",
    "state_goal_hash": "canonical_state_goal_hash",
    "split": "train",
    "source": "oracle_or_perturbation_or_dagger",
    "generator_version": "version",
    "oracle_version": "version"
  }
}
```

Les valeurs de l'exemple sont schématiques ; un validateur impose la cohérence effective. Conserver raw et normalized observations, inline pour le jouet ou par référence content-addressed. Dédupliquer sans perdre la provenance des visites.

### 5.3 Règles d'adaptateur et de candidats

- `observe` et `normalize` ne reçoivent pas le but ni l'oracle.
- `enumerate_candidates` utilise seulement les entités typées et signatures publiques ; aucun ranking ou top-k orienté tâche en V0.
- Un changement de but à état identique ne change ni la normalisation de l'état ni la liste de candidats.
- Une permutation d'IDs met à jour simultanément buts, relations, candidats et labels.
- Les IDs servent à retrouver les états d'entités ; pas d'embedding appris sur leurs chaînes/numéros.
- Un changement de labels/provenance à `policy_input` inchangé doit produire exactement les mêmes entrées tensorisées et prédictions en mode déterministe.
- Tout dépassement de capacité de sérialisation/tenseur est une erreur visible, pas une troncature silencieuse éliminant une action correcte.

Capacités initiales suggérées pour le harness : `N≤64` entités et `K≤32` candidats ; les mondes V0 actuels utilisent beaucoup moins d'entités et `K≤18` jusqu'à 12 pièces. Déclarer padding dynamique/fixe dans les timings. Ces capacités ne constituent pas une preuve de contrôle à 64 entités.

---

## 6. Architecture A et comparateurs conditionnels

### 6.1 A — modèle minimal obligatoire

Pooling relationnel local à un saut (Deep Sets local), largeur `d=128`, estimation totale **0,3–0,7M paramètres**.

- Embeddings de types et attributs observables à vocabulaire fermé.
- Le modèle ajoute des tags de rôle issus des références du but : objet demandé, pièce demandée. Ces tags sont un calcul explicite sur des entrées autorisées, pas une annotation de progression. L'adaptateur reste indépendant du but.
- Encodeur partagé de nœuds à deux couches.
- Pour chaque entité, pooling invariant des relations incidentes avec type, sens et représentation de l'autre extrémité ; une mise à jour locale contextualise cette entité. L'agent et les objets sont ainsi localisables via les relations AT/HELD.
- Pooling global des entités contextualisées ; but construit depuis son prédicat et les représentations référencées.
- Score de chaque candidat avec un MLP partagé recevant contexte global, but, type d'action et représentations contextualisées de ses arguments ; STOP possède un type propre et des arguments nuls.
- Aucun position embedding sur le rang arbitraire des listes.

```text
nœuds + tags du but --> MLP --> pooling incident par entité --> update local
                                          |                         |
                              références restent liées         h_entités
                                                                    |
                                                      pool global + but
                                                                    |
candidat_i --> type + h_arguments -----------------------------------+--> score_i
```

Un simple pooling GLOBAL de triples dont les extrémités ne portent que leur type serait insuffisant : beaucoup de pièces et de graphes deviendraient indiscernables. Le pooling local et le binding par références sont donc normatifs. A est un petit encodeur relationnel à un saut, pas une baseline prétendument dépourvue de tout biais de graphe. Il ne garantit pas une recherche multi-sauts ; cette limite peut motiver B.

A est une **baseline falsificatrice** : son succès sur TinyGraphKey ne fonde aucune conclusion architecturale générale. Les interfaces logicielles (SIW, Web) imbriquent des dépendances plus profondes que ce monde à un objet dominant ; B y est le candidat naturel — à démontrer par les gates, jamais à supposer d'avance.

Une ablation « pooling global sans mise à jour locale par entité » est le diagnostic obligatoire avant d'interpréter tout échec d'A, et une condition d'entrée de GATE-5 pour tout passage à B/C.

### 6.2 B — GNN si l'échec d'A le motive

`d=192`, trois blocs distincts de messages relationnels et mises à jour, résidus/normes, pooling et scorer comparables à A. Tags de rôle du but ajoutés avant le message passing ; l'ablation sans but les retire aussi. Estimation **1,1–1,5M**.

Un bloc dominant avec message `3d→d→d` et update `2d→d→d` vaut environ `7d²` ; trois blocs donnent 774 144 poids avant entrées/têtes/biais. Trois passages limitent la propagation locale ; mesurer la dépendance au diamètre du graphe. Une variante à poids partagés/itérations variables serait une autre expérience.

### 6.3 C — Transformer si comparaison supplémentaire justifiée

Quatre blocs encoder, `d=256`, quatre têtes, FFN 1024 ; tokens d'entités et but, tags de rôle référencés avant attention, relations intégrées en biais/attributs, pas de positions de sérialisation arbitraires ; scorer partagé comparable. Les biais relationnels et tags sont permutés avec les entités. Estimation **3,5–4,5M**.

Backbone dominant : `4 × (4×256² + 2×256×1024) = 3 145 728` poids. Les paramètres totaux, embeddings et têtes seront comptés par le code futur.

```text
[but, entités] + relations --> encodeur ×4 --> états contextualisés
                                                     |
                                    pool + refs + candidats --> scores
```

### 6.4 Règles de comparaison

Comparer d'abord au même corpus et même protocole, puis à paramètres/calcul comparables si l'on revendique un effet d'architecture. Rapport obligatoire : nombre exact de paramètres totaux et entraînables, précision, longueur d'entrée, candidats, appels réseau et mémoire/latence.

L'agrandissement à six blocs `d=384` (~10,62M backbone) ou au-delà de 20M est interdit par défaut : autorisation seulement après bénéfice de capacité établi sur validation et coût mesuré.

---

## 7. Données, splits, couverture et curriculum

### 7.1 Ordre et plafonds

| Phase | Volume proposé | Rôle |
|---|---:|---|
| Diagnostic | 100 épisodes | Tests mécaniques et surapprentissage ; aucun claim |
| Pilote | ~1 000 puis ~5 000 épisodes | Courbe de données et choix sur développement |
| BC V0 | Au plus 10 000 épisodes et 100k transitions fournies, première limite atteinte | Expérience fermée |
| DAgger conditionnel | Jusqu'à 50k transitions supplémentaires | Seulement après diagnostic de distribution shift |

Conserver les épisodes complets : arrêter la collecte avant le dépassement d'un plafond, plutôt que couper une trajectoire en supprimant STOP. Le ratio nombre d'épisodes/transitions dépend des longueurs réellement obtenues.

### 7.2 Unités à publier

- Nombre de layouts distincts et groupes isomorphes.
- Épisodes générés, acceptés, rejetés et motifs.
- Transitions brutes et couples uniques `(layout canonique, état physique, but)`.
- Nombre de fois où ces couples sont exposés à l'optimiseur.
- Nombre d'actions optimales annotées, cardinalité de A*.
- Nombre d'états/états-buts calculés par l'oracle, coût CPU/mémoire et durée.
- Couverture par layout lorsque l'énumération la rend calculable.

Une augmentation par renommage/permutation ne compte pas comme nouveau couple physique. Un oracle ayant calculé toute une table ne peut pas être présenté comme ayant coûté seulement le nombre de labels ensuite sélectionnés.

### 7.3 Groupes train/validation/test

Proposition initiale : 200 layouts train, 50 validation ; au moins 100 layouts par cellule de test confirmatoire, cinq tâches par layout. Ajuster la génération sur développement avant gel si l'inventaire structural est insuffisant, jamais après les scores tests.

- Isomorphisme : préfiltre par hash structural, puis comparaison exacte dans les buckets susceptibles de collision.
- Le grouping porte sur topologie et position structurelle de la porte ; tous les états/buts associés restent dans le même pool.
- Les variantes par renommage, perturbation, réétiquetage et DAgger héritent du pool source.
- Les seeds et manifest de splits sont figés avant l'entraînement confirmatoire.
- Les tests sont scellés ; aucun tri des checkpoints/configurations sur leur résultat.

### 7.4 Cellules

- **G0 diagnostic :** nouveaux états/buts sur layouts train.
- **G1 principal :** nouveaux layouts non isomorphes, mêmes tailles et difficulté `d*=2..12`, à l'exclusion de la combinaison réservée G2.
- **G2 composition limitée :** réserver `AT(clé, pièce de degré ≥3)`. Le train et la validation de sélection excluent exactement cette combinaison, mais contiennent `AT(clé, pièce de degré ≤2)`, `AT(colis, pièce de degré ≥3)`, `HAVE(clé)` et `REACH(pièce de degré ≥3)`. Les tests G2 utilisent des layouts disjoints comportant des jonctions, buts solvables et `d*=2..12`. Le degré est dérivable des relations observables, pas une information oracle.
- **G3 taille/topologie :** 9–12 pièces ; topologies retenues décrites, comparaison séparée avec taille constante lorsque possible.
- **G4 horizon :** distance `d*=13..24` après vérification de faisabilité (§4.5), histogrammes publiés.
- **G5 invariance :** même problème sous 20 permutations cohérentes d'IDs et ordres ; test métamorphique, pas nouvelle famille.
- **G-STOP :** épisodes initialement satisfaits et non satisfaits, taux de STOP erroné ; rapport séparé du succès non trivial.

G2 mesure uniquement la recombinaison d'un prédicat, d'un rôle d'objet et d'une propriété topologique ; ce n'est pas un test de nouveaux programmes ou de logique temporelle. Atteindre le but demande éventuellement navigation, gestion de la clé et dépôt, mais aucune nouvelle grammaire d'action n'est revendiquée. Vérifier la présence de chaque composant au train et assez d'instances G2 avant gel ; manifest et comptages constituent un artifact obligatoire.

### 7.5 Perturbations et couverture

Une perturbation choisit une action non optimale depuis un état train, avance effectivement l'environnement et sollicite l'oracle sur l'état atteint. Conserver l'action erronée comme provenance comportementale, pas comme cible BC.

DAgger : rollouts de la politique sur layouts train, annotation des états visités, déduplication et agrégation. Deux rondes maximum dans l'enveloppe additionnelle. Comparer à un ajout égal de couples uniques échantillonnés uniformément parmi les états atteignables. Sur un sous-ensemble, comparer aussi l'entraînement exhaustif ; si cette solution est la plus simple, la retenir.

---

## 8. Apprentissage, curriculum et sélection

### 8.1 Objectif

```text
L = -log somme des probabilités des actions de A*
```

Implémentation future numériquement stable avec logsumexp. Seul le padding est masqué dans la distribution. Le masque A* est utilisé dans la loss, jamais pour supprimer les erreurs possibles au runtime.

Aucune loss de progression, valeur, reconstruction, terminaison séparée ou dynamique dans le bras principal. STOP est entraîné comme action complète. Une augmentation de loss est une ablation avec coût déclaré, pas un changement invisible de baseline.

### 8.2 Point de départ MLX

Configuration proposée, à valider :

- AdamW, learning rate `3e-4`, weight decay `1e-4` ;
- batch 64, clipping de norme 1 ;
- FP32 pour la première validation numérique ;
- BC limité à 10 époques ou 10k updates, première limite atteinte ;
- évaluation validation toutes les 500 updates et à la fin ;
- sélection au meilleur succès validation, puis longueur/coût en départage ;
- budget de trois learning rates au maximum par architecture sur développement, même droit de recherche pour chaque bras.

Le diagnostic de surapprentissage sur 100 épisodes peut utiliser jusqu'à 5k updates et n'est pas soumis au plafond de 10 époques. Une incapacité à le passer déclenche un audit mécanique avant tout scaling.

Trois seeds exploratoires, cinq seeds confirmatoires. Ne pas publier seulement le meilleur seed. Les permutations se font au train **et** au test, en conservant toutes les références.

### 8.3 Curriculum

Commencer avec instances courtes pour vérifier le pipeline. La population d'entraînement finale conserve un mélange stratifié de buts et bandes de d* ; publier proportions et poids. Le numéro de phase, d* et la position dans le curriculum ne sont pas des features d'entrée.

Ne pas mélanger données expertes, random et contre-factuelles dans une BC aveugle : seuls les labels d'action optimum contrôlés sont des cibles. Des transitions sans label peuvent être archivées, mais aucun monde latent n'est entraîné par défaut.

### 8.4 Diagnostic avant ajout

Ordre des questions : erreur d'oracle ? fuite ? information manquante ? candidats incomplets ? défaut de binding ? supervision insuffisante ? distribution des états ? biais relationnel ? capacité ?

Le passage à un plus gros réseau n'est autorisé qu'après exclusion documentée des problèmes précédents. L'augmentation des données et l'augmentation des updates sont deux interventions différentes ; ne pas les confondre dans les courbes.

---

## 9. Évaluation, baselines et statistiques

### 9.1 Baselines V0

1. Random sur tous les candidats syntaxiques.
2. Random sur actions physiquement valides : bras informé par préconditions, clairement étiqueté.
3. Heuristique locale explicite, sans distances ni plan oracle ; documenter toutes ses règles.
4. Recherche exacte sur modèle connu : borne de supervision et concurrent système, avec connaissance privilégiée des transitions déclarée.
5. Retrieval/kNN sur cas train si son coût d'implémentation est faible ; pas de voisins test.
6. Modèle A ; B et C uniquement si décision de recherche motivée.

Contrôles indispensables : sans but (tags de rôle et tout binding du but également retirés), but contrefactuel valide, candidats seuls, sans relations, permutation cohérente, et bras secondaire avec masque de validité observable. Le masque de validité ne doit jamais être comparé comme s'il n'apportait aucune information supplémentaire.

### 9.2 Mesures

**Primaire V0 :** succès d'épisode complet avec STOP et vérification indépendante.

**Secondaires :** but physiquement atteint, prématurité de STOP, timeouts, invalides, longueur, coût total et difficulté initiale.

- `L* = d* + 1`, STOP inclus.
- Regret `L-L*` rapporté sur succès **avec** taux d'échec associé.
- Score de coût tronqué : `L` sur succès, `H+1` sur échec ; formule explicitement donnée, jamais présenté comme regret exact.
- Optimal-action rate et probabilité sur A* stratifiés par `|A*|` ; diagnostics, pas substituts du succès fermé.
- Succès par but, bande de d*, taille, topologie et besoin réel de déverrouillage.
- Rappel des candidats : présence d'au moins une action optimale ; 100 % attendu dans ce monde à énumération complète.
- Timings p50/p95/p99 batch 1 et coût par épisode réussi, pas seulement débit batch.

### 9.3 Appariement et intervalles

Même liste d'épisodes et mêmes perturbations contrôlées entre bras. Cinq seeds confirmatoires ; au moins 100 layouts avec cinq tâches par cellule. Bootstrap apparié hiérarchique à 95 % sur graines, layouts puis tâches ; publier aussi valeurs par seed et nombre effectif de clusters.

Les 500 épisodes d'une cellule ne sont pas 500 layouts indépendants. Les tests d'architecture répétés sont exploratoires tant qu'une comparaison primaire n'a pas été fixée sur développement. Pas d'extrapolation d'un intervalle de confiance à une garantie de sûreté opérationnelle.

### 9.4 STOP, sécurité et erreurs

L'évaluateur mesure les postconditions sur l'état natif, pas la prédiction du réseau. Une sortie `SUCCESS` du réseau ne vaut jamais attestation. En V0, STOP prématuré termine en échec ; aucun fallback ne le récupère silencieusement.

Les erreurs de génération/adapter sont des catégories séparées et visibles. Si une entrée dépasse le périmètre déclaré, la qualifier d'unsupported ; rapporter son nombre et son traitement, ne pas l'enlever du dénominateur sans déclaration préalable.

---

## 10. Budget M5/MLX et profiling

### 10.1 Mémoire

Ordres de grandeur, **pas mesures** : poids 16 bits ≈2 octets/paramètre ; entraînement Adam avec gradients/moments/master weights selon configuration ≈16–20 octets/paramètre hors activations/buffers.

| Taille | Poids seuls 16 bits | Enveloppe paramètres d'entraînement |
|---:|---:|---:|
| 0,5M | 1 Mo | 8–10 Mo |
| 1,3M | 2,6 Mo | 21–26 Mo |
| 4M | 8 Mo | 64–80 Mo |
| 12M | 24 Mo | 192–240 Mo |

Les 32 Go sont partagés avec OS et processus. L'attention, les activations, les copies et le dataset peuvent dominer les poids. Pas de chargement intégral du corpus nécessaire ; commencer avec une lecture simple streamée, JSONL pour pilote puis format colonnaire seulement si le profiling le justifie.

### 10.2 Mesure correcte

Consulter les règles MLX de [mémoire unifiée](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html) et d'[évaluation paresseuse](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html).

Le harness doit :

- forcer l'évaluation et attendre la fin des calculs chronométrés ;
- séparer compilation, 50 itérations de warm-up et au moins 1 000 décisions d'inférence mesurées ;
- comparer CPU/GPU pour les petits inputs, même précision ;
- mesurer les tailles réelles puis le profil limite déclaré `N=64,K=32`, padding documenté ;
- mesurer normalisation, candidats, tensorisation, modèle, sélection et environnement séparément ;
- mesurer entraînement sur au moins 200 updates après warm-up et refaire un segment soutenu pour détecter le throttling ;
- enregistrer modèle de machine, OS, MLX, précision, batch, taille de séquence, alimentation, mémoire et swap ;
- déclarer si énergie mesurée avec un compteur système adéquat sur une longue série, sinon `non_mesuree`.

Le timing de construction d'un graphe paresseux n'est pas une latence d'inférence. L'énergie du noyau seule n'est pas l'énergie d'une décision système.

### 10.3 Budget expérimental

Enveloppe initiale proposée : **24 heures cumulées d'accélérateur pour V0**, incluant explorations/confirmations ; journaliser séparément génération CPU et travail humain. Arrêter et revoir le plan si la projection dépasse cette enveloppe. Un budget atteint sans puissance statistique suffisante produit un résultat inconclusif.

Règle d'estimation : `T ≈ updates × temps_step_mesuré + génération + évaluation + I/O`. Publier paramètres, expositions et estimation de calcul, pas seulement heures-machine.

Objectifs opérationnels initiaux : p95 modèle batch 1 ≤20 ms, p95 décision avec adaptateur ≤50 ms sur profil déclaré. Ce sont des critères proposés ; ils ne doivent pas être présentés comme acquis sur M5.

La RTX 3070 ne fait pas partie du protocole principal. Une migration exige son propre profiling et des résultats séparés ; pas d'attribution à l'architecture d'un gain provenant du backend ou de la précision.

---

## 11. Gates et conditions d'arrêt

| Gate | Critère proposé | Conséquence |
|---|---|---|
| GATE-0 intégrité | Tests oracle/replay, séparation labels, candidats, permutations et STOP passent ; hash du protocole confirmatoire et de la grille d'analyse figé avant toute ouverture de test/cible ; aucun échec inexpliqué | Sinon aucun entraînement sérieux ni score scientifique |
| GATE-1 apprentissage | ≥99 % d'actions optimales sur le diagnostic ; comportements de fin cohérents | Sinon audit mécanique avant scaling |
| GATE-2 instances | G1 succès moyen ≥95 %, borne IC basse ≥90 %, chaque type de but ≥85 % | Sinon diagnostic/correction ou résultat négatif limité |
| GATE-3 composition | G2 succès ≥80 % ; définition de composition scellée | Sinon pas de revendication de composition |
| GATE-4 utilité/coût | Latence mesurée dans enveloppe, baselines fortes et coût complet rapportés | Si heuristique/recherche suffit mieux, la retenir |
| GATE-5 complexité | Ablation pooling global/local réalisée ; B/C améliore la cellule d'échec préspécifiée ≥5 points, IC gain >0, perte G1 ≤2 points, coût ≤2× A | Sinon garder A ; pas de stacking de composants |

Pour G5 métamorphique, comparer les distributions après permutation inverse : tolérance proposée `1e-5` en FP32, `1e-3` en précision réduite, ajustable uniquement avant test confirmatoire à partir de répétitions numériques. Les ties d'argmax sont distingués d'une violation d'équivariance des scores.

G3/G4 sont des stress tests descriptifs, pas une obligation d'inventer une capacité absente. Ne pas masquer un échec OOD derrière une moyenne ID élevée.

**Stop immédiat d'une revendication**, pas forcément du projet entier : fuite détectée, split contaminé, oracle erroné, protocole changé après test, fallback non compté ou comparaison inéquitable.

**Stop de complexification :** baseline simple suffisant, gains négligeables, préentraînement non amortissable, parsing/candidats dominant le coût, ou coût de recherche supérieur à l'information attendue. Publier la raison et les données de diagnostic.

---

## 12. Protocole de transfert — après V0 seulement

### 12.1 Question et population

Mesurer si un préentraînement améliore le **succès fermé** après une quantité fixée de supervision cible sur de nouvelles dynamiques. Pas simplement la loss d'imitation, pas seulement l'apprentissage de nouveaux noms.

Exemples de sources candidates : TinyGraphKey ; OfficeWorld avec copie sans déplacement, création, permissions et approbations. Exemples de cibles : UI entièrement observable à sélection/commit ; workflow à ressources et dépendances. Ces exemples ne sont pas encore des benchmarks validés.

Une cible de type UI doit être la **cible pont désignée** : le Monde d'Interaction Synthétique défini en §12.6. Ce choix empêche une dérive où le projet deviendrait de la recherche pure de planning sur graphes et repousserait indéfiniment les problèmes structurels du logiciel (labels arbitraires, pointeurs d'entités, arguments, many-candidates).

Avant confirmation : spécifier les tables de transition, non-isomorphisme des dynamiques, observabilité, buts, coût des adaptateurs et oracles. Si deux familles sont essentiellement un renommage, appeler le test transfert de représentation/schéma, pas transfert de dynamique.

Les cibles nouvelles doivent permettre des budgets de supervision partiels : publier le nombre d'états-buts possibles/atteignables, et la couverture aux budgets choisis. Si l'espace entier tient déjà dans le plus petit budget, revoir le test avant gel.

Nommer la revendication pour ce qu'elle est : à couverture partielle par layout, le régime testé est une **généralisation inter-layout à couverture partielle**, pas un simple « le transfert aide ». Un lecteur doit savoir que les +5 points du gate primaire démontrent exactement cela.

### 12.2 Sources et bras

Fixer une seule architecture sur développement, pas choisir la meilleure architecture après inspection des cibles.

- Scratch.
- A seul.
- B seul.
- **A+B joint**, batch équilibré, total source et updates/expositions identiques aux bras mono-source.
- Préentraînement de contrôle : source ou tâche auxiliaire ne ciblant pas les mêmes dépendances, précisément documentée ; sert à tester “n'importe quel préentraînement aide”, sans garantir une absence absolue de structure partagée.

Exemple de budget source : 100k couples uniques totaux, donc 50k A + 50k B contre 100k d'une seule source. Pas d'ensemble de politiques, de moyenne de checkpoints ni de fine-tune séquentiel implicite dans le bras “A+B”.

Reporter la couverture des sources dans cette même unité de couples uniques : des volumes égaux en transitions peuvent cacher des couvertures très différentes entre bras et réintroduire un déséquilibre invisible.

### 12.3 Cibles et budgets

Choisir deux cibles confirmatoires C1/C2 et une famille distincte de développement pour les choix d'hyperparamètres. Sceller les pools de layouts d'adaptation et de test.

Budgets secondaires : `k = 0, 100, 500, 2000, 10000` **couples uniques état-but** dans des layouts d'adaptation seulement. Les sous-ensembles sont imbriqués et identiques entre bras. Les layouts tests restent disjoints. Les alias/permutations ne créent pas de nouveaux exemples.

Aucun oracle cible n'est interrogé durant les rollouts tests de la politique. L'oracle évalue/annote hors chemin de décision. Toute interaction target pour sélectionner un hyperparamètre, calibrer ou déboguer est comptée et interdite sur le test scellé.

### 12.4 Critère primaire qui peut échouer

**Point primaire :** `k*=500`, cinq seeds par bras et par cible, comparaison A+B contre scratch.

Même budget cible d'updates, même précision, même optimiseur sélectionné sur développement, mêmes nouveaux embeddings/têtes initialisés, même adaptateur. Proposition de départ : 2 000 updates cible au batch 64 ; vérifier convergence/coût sur la famille de développement, puis figer avant toute ouverture cible. Checkpoint final à budget fixe, sans sélection sur performance cible.

**Passage : sur chacune de C1 et C2**, gain moyen de succès fermé ≥5 points de pourcentage et borne basse de l'IC apparié 95 % >0. Sinon, pas de revendication de transfert répliqué. Réussir une seule cible autorise uniquement une observation limitée à cette cible.

Le protocole est d'abord piloté sur développement. Avant toute ouverture des cibles, publier le hash du protocole confirmatoire et de la grille d'analyse (comparaisons, unités, statistiques, règles de censure). La confirmation complète est conditionnelle à une projection de coût compatible avec une enveloppe additionnelle proposée de 48 heures d'accélérateur ; si elle ne tient pas, réduire la question avant de voir les cibles, pas les exigences après les résultats.

### 12.5 Analyses secondaires et attribution

- Courbes de succès fermé et coût par `k` ; AULC normalisée sur `log(1+k)` avec la même grille pour tous.
- **Couverture atteinte en couples uniques, par bras et par budget, publiée à côté du succès fermé** — surtout à `k*=500`. Un négatif à `k*=500` avec couverture proche de zéro est ininterprétable : le rapporter comme tel, sans le présenter comme une réfutation du transfert.
- Nombre d'exemples pour 80 % : censuré `>10000` si seuil non atteint. Une réduction par deux n'est revendiquée que si des croisements observés et un protocole d'interpolation fixé le permettent ; ce n'est pas une condition optionnelle du gate primaire.
- Contrôle scratch renforcé avec mêmes labels mais updates/expositions comparables au calcul source + cible ; FLOPs estimés et temps observé. Ne pas faire du wall-clock thermiquement variable l'unique variable de matching.
- Si le scratch convergé rejoint le pretrained, conclure éventuellement à une accélération d'optimisation, pas automatiquement à une réduction durable des besoins de supervision.
- A+B contre meilleur mono-source : nécessaire pour attribuer un gain à la diversité, en tenant compte de la sélection du meilleur dans l'analyse.
- Reset tête, reset encodeur, gel versus fine-tune complet : localiser le transfert plutôt que le qualifier globalement d'intelligence.
- Compter coût source, oracle, adaptateurs et parsing ; estimer amortissement sur plusieurs nouvelles tâches.

Avec deux cibles seulement, le résultat reste « observé sur ces deux familles ». Une revendication large exige d'autres familles indépendantes et, idéalement, un autre constructeur d'adaptateur.

### 12.6 Cible pont : Monde d'Interaction Synthétique (SIW)

**Position.** Étape suivant le transfert §12 et précédant tout Web réel : elle teste si le principe survit quand l'environnement commence à ressembler au logiciel. Réussir TinyGraphKey puis le transfert entre graphes ne dit presque rien sur DOM, labels textuels, arguments d'action et foule de candidats ; SIW est conçu pour exposer précisément ces différences, sans coût réel.

**Structure minimale.**

- Entités typées : boutons, champs, listes déroulantes, menus, formulaires, boîtes de dialogue, éléments inactifs et distracteurs.
- Labels textuels arbitraires : vocabulaire généré fermé avec alias, sans encodeur de texte préentraîné requis ; le binding se fait par référence d'entité, pas par mémoire de label.
- Buts structurés : remplir, sélectionner, naviguer, soumettre, réordonner ; satisfaction vérifiée par l'état interne.
- Actions complètes à arguments : `CLICK(cible)`, `SELECT(cible, option)`, `TYPE(cible, pointeur/copie depuis le but)`, `NAVIGATE(vue)`, `SUBMIT(formulaire)` ; ordre et dépendances de validation rendent l'ordre d'action parfois obligatoire.
- État entièrement observable et déterministe ; oracle par machine à états explicite ; layouts scellés avec les mêmes règles de split/isomorphisme que §7.3 (isomorphisme sur le graphe de widgets et le machine à états, pas sur les labels).
- Nombre de candidats volontairement élevé (30–60) pour mesurer le coût de many-candidates par décision.

**Ce que SIW mesure en propre.**

1. La **taxe d'adaptation de l'interface** sur un cas logiciel : le même contrôleur sous adaptateurs progressivement moins filtrants (pré-brut typé, sans pré-filtrage, avec distracteurs).
2. Le binding d'entités sous labels arbitraires : permutation G5 doit y être répétée.
3. Les arguments d'action (pointer/copy) sans génération de texte.
4. Le rappel du générateur de candidats quand K explose.
5. La survie du gain de préentraînement (si V1 a réussi) sur une famille structurellement plus proche du logiciel.

**Gates SIW.** Mêmes règles que §12.4 (succès fermé, couverture par bras, seeds, IC appariés). Comparaisons obligatoires : scratch SIW, préentraînement source graphique, et — si V1 est positif — transfert A+B→SIW. Stop si le préentraînement graphique n'aide plus SIW : c'est un résultat majeur sur la frontière du transfert, pas un échec à cacher ; le publier comme tel et revoir l'ambition « contrôle transférable » avant d'approcher le Web.

**Explicitement hors SIW.** Pixels, sites réels, partial observability, erreurs de réseau, contenu dynamique — tout cela reste du Web (§13.6).

---

## 13. Extensions conditionnelles

### 13.1 Mémoire

Déclencheur : paires d'histoires ayant même observation courante mais décisions optimales différentes. Ajouter une variante POMDP contrôlée, pas cacher arbitrairement l'état sans moyen de le récupérer.

Comparer : politique réactive ; court historique explicite ; GRU ; éventuellement mémoire par entité. Une GRU d'entrée/état 256 ajouterait environ 0,39M paramètres. Mesurer réussite par délai depuis l'information pertinente, boucles et capacité à choisir une action d'inspection. Aucune présence de timestep oracle.

### 13.2 Langage

Déclencheur : contrôle structuré fiable et besoin utilisateur défini. Comparer but structuré parfait, parseur déterministe sur langage contraint, petit encodeur préentraîné puis éventuellement encodeur entraîné de zéro.

Séparer erreurs d'interprétation et de contrôle. Compter toutes les tailles et latences d'encodeurs. Les paraphrases de professeur restent liées au même groupe de provenance et ne traversent pas les splits. Un span/copy head convient seulement lorsque l'argument requis existe dans le texte fourni.

### 13.3 Arguments et action spaces plus grands

Au-delà de petits K, comparer candidats complets versus scores de tuples `type,target,args` avec contraintes de compatibilité. Des têtes indépendantes peuvent produire une combinaison invalide. Mesurer rappel du générateur avant succès du scorer.

Pour Web : une opération TYPE pouvant exiger un texte nouveau n'est pas couverte par une simple sélection d'entité. Déclarer quand génération, copie, requête utilisateur ou appel exceptionnel à un LLM est nécessaire. Compter ces sorties hors chemin rapide.

### 13.4 Recherche et world model

Déclencheur : erreurs identifiées de lookahead que davantage de données réactives ne résout pas économiquement.

Ordre :

1. Recherche courte avec transitions exactes, sur petit ensemble diagnostique.
2. Comparateur policy à calcul accru ou petite valeur supervisée.
3. Modèle appris seulement si le modèle vrai apporte un bénéfice utile.
4. Apprendre effets/récompenses/valeurs nécessaires au choix d'action, pas forcément toute l'observation.
5. Évaluer ranking d'actions, succès des plans et erreur multi-step ; une faible MSE latente ne suffit pas.

Fixer budget d'expansions/appels, horizon et largeur sur validation ; compter tous les candidats évalués. Comparer same-compute et same-latency séparément. Un world model qui améliore une métrique auxiliaire mais détériore le succès est rejeté.

### 13.5 Options, chunks et RL

Options : initiation, effets attendus, terminaison et interruption explicitement définis. Comparer à politique plate à mêmes données/budget et à macros humaines dont le coût est déclaré. Pas de nom abstrait magique remplaçant plusieurs actions d'environnement.

RL : uniquement si un bénéfice attendu n'est pas accessible par imitation/oracle, avec coût d'interaction, reward hacking et vérification. Ne pas importer tout un pipeline RL dans V0.

### 13.6 Web et perception

Prérequis : SIW (§12.6) doit avoir montré la survie du principe (ou documenté l'écart) avant tout Web réel. Sinon, le risque est de découvrir très tard que le Web exige une architecture fondamentalement différente, après avoir optimisé un monde de graphes purs.

Utiliser BrowserGym et un benchmark/version précis avant de reconstruire un harness. Commencer sandbox, tâches réversibles, états/snapshots reproductibles. DOM/accessibility/screenshot sont des canaux distincts avec coûts et limites différents.

Les données WebLINX sont utiles à l'imitation ; action offline correcte et tâche online réussie sont des endpoints différents. Une comparaison de remplacement d'un agent LLM nécessite mêmes outils, mêmes informations, mêmes tâches, budget complet et réussite vérifiée.

---

## 14. Sûreté hors environnement jouet

La confiance ne remplace jamais autorisations, isolation et postconditions.

- Lecture seule : permissions minimales et journalisation.
- Mutations réversibles : limites de périmètre, état avant/après, rollback testé.
- Irréversible, sensible ou financière : confirmation explicite par défaut.
- Vérification indépendante du modèle lorsque possible ; erreurs du vérificateur mesurées.
- Politique d'abstention/fallback évaluée par risque-couverture, taux d'appel, coût et résultat final.
- Tous les appels LLM et interventions humaines comptent dans le score système.
- Aucun retry infini, aucune extension silencieuse des permissions ou du budget.

Calibration : NLL/Brier, ECE en diagnostic, erreurs par buckets de candidats/domaines et courbes de risque-couverture ; température sur validation. Pas de promesse de calibration OOD. L'abstention ne fait pas disparaître les tâches difficiles du dénominateur.

---

## 15. Artifacts, ordre de travail et définition de terminé

### 15.1 Structure minimale future

Ne créer que les composants nécessaires :

```text
configs/                 # environnement, données, modèle, entraînement
ucm/env/                 # TinyGraphKey + oracle, clairement séparés
ucm/data/                # validation, splits, writer/loader
ucm/model/               # A d'abord
ucm/eval/                # rollouts, baselines, métriques, profiling
tests/                   # oracle, replay, fuite, équivariance, STOP
artifacts/<run>/         # config, manifests, seeds, scores, timings, checkpoint
```

Cette arborescence est une proposition, pas un travail déjà réalisé. Aucun fichier de code n'est requis dans la présente révision documentaire.

### 15.2 Artifact obligatoire par run

Version du code ou hash d'archive si pas de dépôt Git ; versions runtime/OS ; configuration ; seed ; hash du protocole confirmatoire et de la grille d'analyse figés avant ouverture des tests/cibles ; dataset/schema/générateur/oracle ; manifest et hash des splits ; nombres uniques/bruts/expositions ; logs par épisode ; checkpoint ; règle de sélection ; temps de génération/entraînement/évaluation ; mémoire, timings et état de mesure de l'énergie ; modifications du protocole ; échecs connus.

Les résultats ne doivent pas exister uniquement dans un terminal. Les rapports distinguent observé, calculé, estimé et non mesuré.

### 15.3 Ordre d'implémentation ultérieure

1. Tables de transitions, frontières d'information, cas manuels.
2. Environnement/oracle, tests de replay/Bellman/STOP/invalides.
3. Audit candidat/goal-blind, permutations et contamination.
4. Inventaire de faisabilité des splits/compositions/horizons ; gel du manifest.
5. Baselines classiques et profiling du harness sans réseau.
6. 100 épisodes, modèle A, surapprentissage et vérification des entrées.
7. Pilote 1k/5k ; courbe de données et coût.
8. Confirmation G1/G2 et description G3/G4/G5 ; vingt échecs inspectés selon catégories.
9. Décision documentée : arrêter, rester sur baseline, corriger données/interface, tester B ou préparer transfert.
10. Après un V0 interprétable et un transfert §12 évalué : pont logiciel SIW (§12.6) avant toute approche Web.
11. Aucun ajout automatique de mémoire, langage, monde latent ou RL.

### 15.4 Définition de terminé pour V0 future

V0 est documentée quand l'intégrité, les mesures, les contrôles et le verdict sont reproductibles, **même si le verdict est négatif**. V0 est un succès de capacité seulement si ses gates correspondants passent. Aucun résultat de transfert n'est sous-entendu par la fin de V0.

---

## 16. Traçabilité, références et limites

### 16.1 Préservé de la spec originale

Objectifs structurés, imitation d'abord, données natives conservées, schémas versionnés, monde jouet à oracle exact, baselines/ablations, environnement de transfert retenu, world model différé, LLM hors chemin rapide et sûreté des actions réelles.

### 16.2 Changé substantiellement

- V0 monofamille séparée de la revendication de transfert.
- A sous le million de paramètres avant B/C et 10–20M.
- Progress/termination heads supprimées ; STOP action et évaluateur indépendants.
- Contrat d'information et candidats goal-blind normatifs.
- Invalides payantes, oracle set-valued, coût optimal défini avec STOP.
- Compte des exemples uniques, couverture et coût réel de l'oracle.
- Comparaison primaire de transfert fixée, critères censurés/non atteints explicites.
- Paramètres, calcul, temps, mémoire et coût d'adaptateur séparés.
- Roadmap comme arbre de décisions, pas accumulation obligatoire de modules.
- Jalon pont SIW introduit entre transfert et Web pour empêcher une dérive vers du planning pur sur graphes.

### 16.3 Références et statut de nouveauté

Voir la [bibliographie annotée](reports/ucm-review/sources.md), notamment UVFA, DAgger, ASNets, Set Transformer, Gato/JAT, ACT, Diffusion Policy, Octo, TD-MPC2, DreamerV3, MuZero, LAPA, OGBench, WebLINX/BrowserGym et calibration. Les références complémentaires non récupérées intégralement sont signalées comme telles.

Aucune nouveauté forte n'est démontrée par cette spec. La contribution envisagée est expérimentale et méthodologique : isoler un transfert de contrôle compact sous audit d'interface. La recherche web a rencontré ses quotas horaires ; une affirmation de priorité nécessiterait une revue ciblée supplémentaire.

### 16.4 Limites de cette livraison

Pas de code, pas de modèles entraînés, pas de mesures M5/RTX, pas de puissance statistique empiriquement estimée, pas de générateurs C1/C2 validés. Les constantes, tailles et seuils sont des propositions directement testables ; les gates d'intégrité et le pilote sont précisément destinés à vérifier leur faisabilité avant une confirmation.

**Principe final :** préférer une expérience courte qui réfute une hypothèse précise à une grande architecture dont le succès ne permet pas de savoir ce qui a été appris.
