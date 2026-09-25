# Revue de recherche critique — Universal Control Model

**Date :** 22 septembre 2026. **Objet :** revue et réécriture de spécification, pas implémentation.

**Documents :** [spécification originale conservée](universal-control-model-project-spec.original.md), [spécification révisée complète](../../universal-control-model-project-spec.md), [sources et limites de vérification](sources.md).

Aucun résultat expérimental UCM, entraînement ou benchmark M5 n'est présenté ici. Les chiffres d'acceptation sont des **seuils proposés à préenregistrer**, les tailles de réseaux des **estimations analytiques**, les résultats bibliographiques des **résultats rapportés par leurs auteurs**. La recherche est substantielle mais non exhaustive ; les quotas web ont interrompu son extension.

Légende : **[É]** résultat ou principe établi dans son périmètre ; **[H]** hypothèse à tester ; **[X]** extrapolation plausible mais non démontrée ; **[S]** spéculation. Un résultat publié n'est pas une garantie transférable au projet. Les renvois §§ numérotés (ex. §§16/29) suivent la numérotation de la [spécification originale archivée](universal-control-model-project-spec.original.md).

## 1. Verdict exécutif

1. **[É] Le projet ne propose pas une nouvelle catégorie fondamentale de modèles.** Une politique conditionnée par objectif, compacte, non textuelle et sans génération token par token est une idée ancienne. UVFA, imitation, distillation, politiques relationnelles et politiques robotiques couvrent déjà une grande partie de ce territoire.
2. **[H] La bonne question est celle du transfert utile à budget contrôlé**, pas « peut-on choisir une action sans LLM ? ». Cette dernière question est déjà résolue dans d'innombrables systèmes.
3. **Le meilleur résultat possible n'est pas “un petit Gato”.** C'est une preuve étroite et reproductible qu'une représentation/initialisation de contrôle réduit les besoins d'adaptation sur des dynamiques non vues, sans déplacer la résolution dans l'adaptateur.
4. **Le mot “universel” est actuellement une ambition de nommage.** L'absence de `if domain == ...` dans le cœur ne démontre ni abstraction partagée ni transfert causal.
5. **La spec est moins naïve que son introduction.** Elle prévoit déjà un petit environnement à oracle exact (§§55–58), des objectifs structurés, l'imitation, des ablations et un world model différé (§36). Il faut préserver ces dispositions, pas prétendre les inventer.
6. **Sa contradiction principale est le périmètre de V0.** §§16/29 exigent plusieurs familles et du transfert ; §§55/66 décrivent d'abord un seul toy world. Une V0 de validation mécanique et une expérience de transfert doivent avoir des critères distincts.
7. **[H] L'adaptateur peut contenir la majorité de “l'intelligence”.** Un état parfaitement symbolisé, un but déjà ancré et une liste d'actions filtrée par pertinence peuvent transformer le contrôle en reconnaissance triviale.
8. **Laya est une inspiration d'interface, pas une preuve de faisabilité de ce projet.** Les checkpoints annoncés sont de 322–421M paramètres ; les décisions typées ne prouvent pas le contrôle fiable à long horizon. Ses latences T4 ne sont pas des mesures M5.
9. **Commencer à 10–20M serait prématuré.** Un baseline de 0,3–0,7M, puis éventuellement un GNN d'environ 1,1–1,5M, peuvent invalider des hypothèses essentielles. Le Transformer de 3,5–4,5M est un comparateur, pas une obligation.
10. **[É] Une interface d'actions finie simplifie fortement le problème.** Elle ne résout pas les arguments libres, les nouvelles opérations, l'observabilité partielle ni les opérations irréversibles. Ces limites doivent figurer dans le titre des résultats.
11. **Supprimer progress/value/world model de la V0.** STOP reste une action apprise, évaluée par un vérificateur indépendant. Une tête de progression n'est pas nécessaire pour démontrer le contrôle de base.
12. **Le planificateur symbolique est un concurrent sérieux, pas seulement un professeur.** Si les préconditions, effets et buts sont déjà symboliques, une recherche classique peut être plus fiable et suffisamment rapide. Un résultat honnête peut être : ne pas déployer le réseau.
13. **Le danger statistique majeur est un transfert mal contrôlé.** Plus de préentraînement, une tête réinitialisée différemment, des cibles choisies après inspection ou un adaptateur privilégié peuvent fabriquer un gain.
14. **Compter les exemples uniques et leur couverture.** Dans un petit univers à oracle exact, “moins de labels” signifie d'abord moins de supervision fournie à l'apprenant, pas nécessairement moins de travail d'annotation réellement effectué.
15. **[É] Non-autoregressif ne signifie ni une seule passe, ni absence de dépendances temporelles.** Diffusion, recherche et contrôle récurrent ont des coûts itératifs. Compter les évaluations réseau et toute la boucle.
16. **Le matériel M5 32 Go suffit pour étudier de petits modèles structurés en mémoire, sous réserves de profiling.** Il ne permet pas de promettre un débit, une latence ou un coût d'entraînement à partir du seul nombre de paramètres.
17. **[X] Un contrôleur spécialisé, transférable dans une classe d'interfaces, paraît crédible. [S] Un “System 1 universel” entre jeux, bureautique, Web et robotique reste une extrapolation considérable.**
18. **Recommandation : poursuivre comme programme de falsification, pas comme construction d'une grande architecture.** Une V0 petite, un test de transfert primaire préenregistré, puis arrêt, publication d'un résultat négatif ou extension conditionnelle.

## 2. Reformulation scientifique

### 2.1 Problème réellement étudié

Pour un environnement `e`, un état latent `s_t` produit une observation `o_t`. L'adaptateur observable `A_e` construit un ensemble structuré `x_t`, un but `g` et des candidats `C_t`. Le contrôleur calcule :

```text
x_t = A_e(o_t)                     # aucune vérité cachée injectée
m_t = U(m_{t-1}, x_t, a_{t-1}, résultat_observable)   # seulement si nécessaire
p_t = πθ(. | x_t, g, C_t, m_t)
a_t ∈ C_t
```

Les tâches V0 sont des MDP déterministes pleinement observés. Le problème général est un POMDP ; appeler une observation normalisée `state` ne la rend pas markovienne. Une mémoire peut compenser un historique manquant, pas une information jamais disponible.

**Hypothèse centrale proposée :** sur une classe déclarée d'environnements discrets à observations structurées et actions typées, un préentraînement multi-source améliore le succès en boucle fermée après adaptation à une nouvelle dynamique, à supervision cible, budget d'optimisation et interface contrôlés.

Le qualificatif « petit » concerne séparément : cœur, encodeurs, mémoire, adaptateurs appris, générateur de candidats, vérificateur et éventuel LLM. Un cœur de 5M derrière un encodeur de 500M n'est pas un système de 5M.

### 2.2 Hypothèses falsifiables, dans l'ordre de coût

| Hypothèse | Test le moins cher | Ce qui l'invalide dans ce périmètre |
|---|---|---|
| H0 : l'interface conserve l'information nécessaire | Chercher deux états normalisés identiques dont les ensembles d'actions optimales sont disjoints | Collision irréductible sans mémoire ou champ supplémentaire |
| H1 : les candidats ne donnent pas la solution | Retirer but/état ; conserver seulement candidats, ordre et types ; tester random/heuristique | Succès élevé expliqué par filtrage, ordre ou cardinalité |
| H2 : la supervision de contrôle est apprenable | Surapprendre 100 épisodes ; vérifier STOP et replay | Échec sur données d'entraînement : problème mécanique avant problème de généralisation |
| H3 : une petite politique est utile sur instances nouvelles | Comparer au meilleur baseline simple sur graphes scellés | Ne gagne pas, ou perd son avantage en coût total |
| H4 : les relations partagées aident vraiment | Ablations relations, permutation, contre-factuels de but | Gain dû aux IDs, au gabarit ou à un tri orienté but |
| H5 : le préentraînement améliore l'adaptation | Comparaison primaire à 500 exemples cibles uniques, mêmes adaptateurs, cinq graines | Gain nul, incertain ou inférieur au seuil d'intérêt préenregistré |
| H6 : la diversité des sources explique le gain | A+B contre A seul, B seul et source de contrôle, budgets égaux | Une seule source ou n'importe quel préentraînement explique le résultat |
| H7 : mémoire/recherche apporte un bénéfice net | Évaluer uniquement après diagnostic de l'échec correspondant | Pas de gain sur succès/risque à budget d'inférence contrôlé |

### 2.3 Limite d'identifiabilité

Deux environnements peuvent exposer exactement les mêmes observations, buts et noms d'actions, mais inverser les effets de deux actions. Avant observation informative ou démonstration, un contrôleur ne peut pas savoir lequel il affronte. Une promesse de zéro-shot doit donc déclarer ce qui ancre la sémantique : conventions communes, schéma d'effets, texte interprétable, exemples ou exploration.

Changer tous les identifiants de pièces est un test d'invariance. Changer la signification de `MOVE` sans fournir de contexte est un autre problème. Confondre les deux produirait des échecs injustement imputés au réseau ou des revendications trop fortes.

## 3. État de l'art : ce qui existe, ce qui manque

Les liens Sxx pointent vers une bibliographie annotée, avec statut de vérification.

### 3.1 Contrôle conditionné par but, imitation et politiques relationnelles

- **UVFA** [S01](sources.md#s01) : état + but, généralisation de valeurs. Le conditionnement par objectif n'est pas une nouveauté.
- **DAgger** [S02](sources.md#s02) : apprend sur les états visités par la politique ; c'est une réponse directe au décalage de distribution de l'imitation. Plus approprié que l'ajout réflexe d'un world model.
- **ASNets** [S03](sources.md#s03) : partage relationnel exploitant les schémas de planification, entraînement sur petites instances et généralisation à d'autres instances du même domaine. **Omission importante de la spec originale.** Une V0 sur entités, relations, préconditions et actions ressemble davantage à cette famille qu'à une révolution post-LLM.
- **Deep Sets, Set Transformer, graph networks, pointeurs** [S04](sources.md#s04), [S27–S29](sources.md#s27) : briques adaptées aux ensembles variables et à la structure relationnelle. Elles fournissent des biais inductifs, pas automatiquement des algorithmes de planification généralisables.
- **Policy Distillation** [S30](sources.md#s30) : compresser un professeur et plusieurs politiques est établi. La contribution éventuelle serait le protocole et le transfert obtenu, non l'idée « LLM enseigne, petit modèle exécute ».

### 3.2 Politiques généralistes et représentation des trajectoires

- **Gato** [S05](sources.md#s05) : un ensemble de poids partagé sur 604 tâches ; précédent fort de diversité, mais autoregressif et avec des données/prétraitements spécifiques.
- **JAT** [S07](sources.md#s07) : autre précédent ouvert multi-domaines. L'écosystème ne se limite pas aux agents conversationnels.
- **Decision Transformer** [S06](sources.md#s06) : transforme un problème de contrôle offline en modélisation causale de trajectoires conditionnée par retour. À distinguer d'une politique sur candidats et d'un planificateur.
- **AnyMDP/in-context RL** [S18](sources.md#s18) : diversité de dynamiques et adaptation en contexte. Question utile : transférer une politique fixe, une représentation, ou une procédure d'adaptation ? La spec mélange parfois ces trois objets.

### 3.3 Robotique : non-autoregressif et “petit” ont déjà plusieurs sens

- **ACT** [S11](sources.md#s11) : prédit des chunks d'actions avec un CVAE/Transformer ; environ 80M paramètres décrits, entraînement par tâche. Un contre-exemple à l'idée qu'une sortie non textuelle serait nouvelle, pas une preuve d'universalité.
- **Diffusion Policy** [S12](sources.md#s12) : distribution multimodale de séquences d'actions et horizon glissant. Pas de génération textuelle autoregressive, mais un débruitage itératif. Pour choisir parmi quinze candidats discrets, c'est a priori de la complexité inutile.
- **Octo** [S08](sources.md#s08) : politique robotique généraliste, adaptation et modularité, préentraînée sur 800k trajectoires. Très pertinent pour la méthode expérimentale ; les conventions d'action et le corpus ne sont pas gratuits.
- **RT-2** [S09](sources.md#s09) : transfert de connaissances vision-langage. On ne peut pas demander la même connaissance du monde à un petit contrôleur aléatoirement initialisé.
- **SmolVLA** [S10](sources.md#s10) : documentation consultée à 450M. « Small » dans ce contexte reste un ordre de grandeur très différent.

**Conséquence :** le projet doit annoncer « politique structurée compacte » et non utiliser les performances VLA pour justifier ses ambitions sémantiques.

### 3.4 Modèles du monde et planification

- **MuZero** [S15](sources.md#s15) : la représentation peut être suffisante pour récompense/valeur/politique sans reconstruction complète des observations.
- **DreamerV3** [S14](sources.md#s14) : robustesse d'une configuration sur plus de 150 tâches. **Une même configuration ne signifie pas un même checkpoint.**
- **TD-MPC2** [S13](sources.md#s13) : planification latente, configurations partagées sur 104 tâches ; expérience distincte d'un modèle 317M partagé sur 80 tâches. Les tailles 1M–317M étudiées montrent aussi qu'il faut relier capacité, données et domaine.
- **V-JEPA 2** [S16](sources.md#s16) : intéressant pour les représentations prédictives vidéo ; aucune raison d'importer son coût et ses hypothèses dans une V0 symbolique.
- **Universal Planning Networks** [S31](sources.md#s31) : voisin à approfondir avant une revendication de nouveauté sur un espace latent orienté planification.

La question pertinente n'est pas « avons-nous un world model ? », mais « sa recherche améliore-t-elle les décisions par rapport à une politique plus simple au même coût ? ».

### 3.5 Actions latentes, hiérarchie et tâches structurées

**LAPA** [S17](sources.md#s17) montre l'intérêt de variables d'action latentes apprises depuis la vidéo puis ancrées par des actions annotées. Un codebook n'aligne pas magiquement les effets entre interfaces. Les options et politiques hiérarchiques sont antérieures au projet ; elles exigent initiation, terminaison, exécution et interruption, pas seulement un nom `NAVIGATE`.

**Reward machines** [S32](sources.md#s32) constituent un voisin pour objectifs temporels et progression explicite. **HER** [S33](sources.md#s33) rappelle que le réétiquetage de buts existe déjà. Aucun des deux ne légitime l'injection d'un progrès oracle à l'inférence.

### 3.6 Logiciels/Web et décisions typées

**WebLINX** [S20](sources.md#s20) distingue utilement la tâche d'imitation offline de la réussite interactive ; ses difficultés sur sites non vus sont directement pertinentes. **BrowserGym** [S21](sources.md#s21) et **WebArena** [S22](sources.md#s22) permettent de réutiliser une infrastructure au lieu de bâtir un écosystème entier.

**Laya** [S24](sources.md#s24) matérialise une interface intéressante : encodage bidirectionnel et décisions typées. Mais son dépôt n'est pas une preuve comparative d'apprentissage de politiques à long horizon. Les 322–421M paramètres et les timings T4 annoncés doivent être présentés comme tels. Ni « aucun token généré », ni « probabilités », ni « décision typée » n'impliquent calibration, récupération ou succès d'une tâche.

### 3.7 Ce qu'il reste raisonnable d'étudier

Un **benchmark contrôlé d'interfaces et de transfert**, avec adaptateurs audités, modèles très petits et coût système mesuré. La bibliographie ne permet pas d'affirmer que cette combinaison est inédite ; elle permet de formuler une contribution expérimentale précise. OGBench [S19](sources.md#s19) fournit un modèle utile de benchmark goal-conditioned, mais ses actions et observations ne sont pas identiques au périmètre logiciel proposé.

## 4. Carte de nouveauté

| Élément proposé | Antécédents | Statut raisonnable | Preuve nécessaire pour une contribution |
|---|---|---|---|
| Politique état/but → action | Goal-conditioned RL, UVFA, imitation | Établi | Pas une revendication de nouveauté |
| Décision non autoregressive | Politiques discriminatives, ACT, diffusion | Établi ; mécanismes différents | Gain à information et coût comparables |
| Score de candidats variables | Ranking, politiques sur actions structurées, ASNets | Établi | Meilleure généralisation d'interface démontrée |
| Pointeur vers entités | Pointer Networks, architectures d'ensembles | Établi | Contrôle des arguments et des dépendances |
| Adaptateurs vers un schéma | Interfaces de simulation, graphes et politiques modulaires | Ingénierie fréquente | Mesurer information et coût des adaptateurs |
| Plusieurs domaines, mêmes poids | Gato, JAT, Octo | Établi | Pas seulement coexistence de spécialistes implicites |
| Professeur coûteux, élève rapide | Imitation et distillation | Établi | Meilleur compromis fiabilité/coût total |
| Progression et STOP | Prédiction auxiliaire, valeurs, automates de tâches | Établi | Valeur marginale prouvée et absence de fuite |
| Mémoire et options | Contrôle récurrent, POMDP, RL hiérarchique | Établi | Amélioration causale sur aliasing/horizon |
| Modèle latent + recherche | MuZero, Dreamer, TD-MPC, UPN | Établi | Gain de choix d'action, pas juste faible MSE |
| Petit cœur partagé entre interfaces réellement différentes | Recouvrement important avec politiques généralistes et relationnelles | Hypothèse expérimentale intéressante | Transfert scellé, supervision et calcul contrôlés |
| Audit “intelligence du modèle vs de l'interface” | Lié aux contrôles de raccourcis et à la validité des benchmarks | Potentiel apport méthodologique | Protocole réutilisable révélant des conclusions fausses |
| Résultat négatif sur frontières du transfert | Pas une architecture nouvelle | Contribution possible | Contrôles assez forts pour rendre le négatif interprétable |

**Additionner dix composants connus ne suffit pas à fabriquer une nouveauté scientifique.** Une mesure causale bien contrôlée peut en revanche être une vraie contribution sans bloc architectural inédit.

## 5. Critique de l'architecture et de la spec actuelle

### 5.1 Douze corrections précises

| Sections originales | Problème | Correction |
|---|---|---|
| §3, « universal representation before universal model » | Universalité supposée de l'interface ; perte d'information et coût de symbolisation non bornés | Schéma minimal falsifiable, audit de suffisance et coût de chaque champ |
| §§4/14, `evaluate_progress`, `signals` | Pas de frontière normative assez stricte entre données observables, labels et évaluateur | Trois canaux séparés ; allowlist d'entrée testée |
| §§7/10 | Génération et filtrage des candidats insuffisamment spécifiés | Enumération syntaxique goal-blind ; invalides conservées en V0 ; audit de couverture |
| §8 | “10–20M” non relié à la configuration 4–6×256 | Compter tous les blocs ; ce backbone représente environ 3,1–4,7M hors entrées/têtes |
| §§9/20/59 | Multiplication des têtes et poids initiaux égaux sans échelle comparable | Action/STOP uniquement ; chaque auxiliaire doit gagner une ablation |
| §12 | `progress ∈ [0,1]` sans sens invariant entre tâches | Retirer de V0 ; séparer probabilité de succès, distance, coût et satisfaction |
| §§16/29 vs §§55/66 | Deux définitions incompatibles de V0 | V0 mécanique monofamille ; transfert à un jalon distinct |
| §18 | Split par configuration insuffisant si graphes/gabarits isomorphes ou annotations liées | Groupe layout, isomorphisme, but et provenance ; vérifier contamination |
| §§22/23/58 | Bonne intention de transfert mais budgets source, optimisation et têtes non contrôlés | Protocole source/cible explicite, scratch fort, point primaire fixé |
| §37 | Largeur/profondeur de beam ne définissent pas le coût ni l'utilité | Compter expansions et appels réseau ; comparer au modèle vrai et à compute accru sans recherche |
| §§38/39 | Confiance susceptible d'être confondue avec sécurité | Risque-couverture et vérification indépendante ; OOD et autorisations hors softmax |
| §60 | Arrêt sur prédiction `SUCCESS` sans assertion de postcondition dans la boucle | STOP proposé par politique, succès attribué exclusivement par évaluateur |

### 5.2 Choix des représentations

| Représentation | Avantage | Coût ou risque | Décision |
|---|---|---|---|
| Vecteur aplati à slots fixes | Simple, rapide, bon contrôle | Ordre et taille fixes, généralisation fragile | Baseline seulement |
| Ensemble d'entités + but typé | Petit, invariance accessible | Pooling seul perd les relations fines | Premier modèle économique |
| Graphe d'entités/relations | Bon biais pour dépendances et taille variable | Portée bornée du message passing, schéma déjà très informatif | Comparateur prioritaire |
| Transformer d'entités + biais relationnels | Interactions globales, flexible | Coût quadratique, raccourcis de position, ambiguïtés de binding | Comparateur conditionnel |
| Texte/JSON brut | Intégration aisée, sémantique préentraînée possible | Tokenisation, vocabulaire, longueur et parsing ; compare aussi un encodeur | Pas V0 principale |
| Pixels/DOM brut | Plus proche du réel | Perception souvent dominante ; grande hausse de données et compute | Nouvelle étude après le contrôle structuré |
| État latent opaque | Compact et potentiellement suffisant | Collapse, perte de causalité, debugging difficile | Aucune priorité sans bénéfice mesuré |

Conserver les observations natives est une bonne disposition existante. Mais conserver le raw offline ne restaure pas une information absente de l'entrée runtime. Le schéma doit représenter explicitement **inconnu**, **absent** et **non applicable**, sans les confondre avec `false`.

### 5.3 Interfaces d'action

**Candidats complets.** Choix recommandé lorsque `K` est petit et les arguments finis. Un candidat correspond à une action exécutable complète. Coût du scoring approximativement linéaire en `K` après encodage, mais coût de génération parfois combinatoire. Mesurer le rappel : la bonne action peut manquer avant même le réseau. Ne pas construire les candidats à partir d'un plan expert.

**Type + pointeurs + arguments.** Utile quand l'énumération `types × targets × args` explose. Des sorties indépendantes peuvent combiner un type et une cible incompatibles. Utiliser des scores conjoints ou un décodage contraint de tuples. Un petit routage conditionnel de têtes n'est pas une génération autoregressive de langage, mais son coût séquentiel doit être compté. Les arguments copiés évitent la génération seulement s'ils existent effectivement dans l'entrée.

**Actions latentes.** Elles compactent une distribution d'effets ; sans ancrage, le code 17 dans un environnement n'a aucune raison de signifier le même effet ailleurs. Tester identifiabilité, réalisabilité et transfert des effets avant de parler de langage universel.

**Options/chunks.** Réduisent la fréquence de décision au prix d'une moindre réactivité. Pour des mutations logicielles, un chunk aveugle de dix actions peut être beaucoup plus risqué que dix décisions courtes vérifiées. Comparer chunks fixes, options apprises et macros écrites à la main en indiquant où se trouve le travail humain.

### 5.4 Politique contre planification

Une politique peut **amortir** le résultat de la planification sur une distribution. Cela ne signifie pas qu'elle sait exécuter un nouvel algorithme de recherche. Tester les tailles/horizons au-delà de l'entraînement est essentiel ; un Transformer profond fixe n'offre pas automatiquement un nombre de raisonnements croissant avec la difficulté.

Pour V0, le modèle exact du simulateur existe. Utiliser : oracle exact comme borne de supervision ; recherche symbolique comme concurrent système avec privilège de connaissance des transitions explicitement signalé ; recherche courte avec modèle exact comme diagnostic ultérieur. Si même le modèle vrai avec petit horizon n'aide pas, apprendre ce modèle ne semble pas prioritaire.

### 5.5 Mémoire, progression, valeurs

- Une GRU est inutile pour prouver une capacité requérant seulement l'état pleinement observé. Elle peut même cacher une représentation insuffisante ou mémoriser le temps écoulé.
- En POMDP, construire des **paires d'histoires à observation courante identique mais décision optimale différente** ; c'est un besoin de mémoire testable.
- Le « progrès » n'est pas toujours monotone : récupérer une clé peut éloigner spatialement du colis. Une distance oracle normalisée ne doit ni devenir entrée ni fournir un shaping non déclaré.
- La probabilité de succès dépend de la politique, de l'horizon, du risque et du budget restant ; elle n'est pas une quantité universelle indépendante de ces conditions.
- Séparer terminé, objectif satisfait, timeout, abandon et environnement invalide. Les quatre classes §13 ne constituent pas une ontologie suffisante sans définitions.

## 6. Architecture recommandée et budget matériel

### 6.1 Pipeline V0

```text
Simulateur observable ----> adaptateur sans but ----> entités / relations
But structuré observable --------------------------> références du but
Signatures publiques ----> candidats syntaxiques ---> tuples d'action
                                                        |
                                                        v
                  encodeur partagé + score conjoint candidat
                                                        |
                                                 logits, une passe
                                                        |
                         argmax, dont STOP possible ----+
                                                        |
                                                  environnement
                                                        |
                                      évaluateur indépendant du succès

État complet / BFS ----> labels hors ligne seulement ----> loss d'entraînement
            INTERDITS EN ENTRÉE : distance, succès oracle, futur, plan, timestep
```

### 6.2 Trois candidats concrets — pas trois implémentations simultanées

Estimations hors grand encodeur de texte/image, embeddings d'entités par **type/attribut**, jamais par identité persistante arbitraire.

| Candidat | Configuration | Estimation paramètres | Pourquoi le tester / limite |
|---|---|---:|---|
| A — Deep Sets local | `d=128`, MLP nœud, pooling relationnel incident **par entité** (type, sens, autre extrémité), update local, tags de rôle du but, MLP score conjoint | **0,3–0,7M** | Suffisant pour invalider l'utilité d'une architecture plus grosse ; le pooling local à un saut est normatif, un pooling global seul rendrait des pièces indiscernables |
| B — GNN relationnel | `d=192`, 3 blocs distincts message/update, pooling + refs du but, score candidat | **1,1–1,5M** | Dépendances relationnelles ; trois pas de message passing ne résolvent pas automatiquement des chemins arbitrairement longs |
| C — Transformer d'entités | 4 blocs, `d=256`, 4 têtes, FFN 1024, relations en biais/attributs, aucun rang arbitraire, MLP candidat | **3,5–4,5M** | Interactions globales ; coûts et raccourcis à surveiller |

```text
A : node MLP + tags but --> pooling incident par entité --> update local --> h_i
                                             |                          |
                                    refs du but restent liées         pool global + concat(g, pool, h_args, type) --> score_i

B : entités <-- messages typés, ×3 --> entités contextualisées
                   |                           |
                   +--> pooling + goal --------+--> score_i

C : [goal refs, entités] --> encoder sans ordre arbitraire, ×4
                                 |             |
                            pool global     pointeurs
                                 +------ score_i ------+
```

Calcul transparent : un bloc Transformer standard compte environ `4d² + 2d f` poids dominants. À `d=256, f=1024`, quatre blocs donnent `3 145 728` poids avant biais, normes, embeddings et têtes. Une variante six blocs `d=384, f=1536` a déjà environ `10,62M` dans le backbone ; elle n'est justifiée que par une courbe capacité/données positive.

Pour B, un message `(3d → d → d)` et un update `(2d → d → d)` représentent environ `7d²` par bloc, soit `0,774M` sur trois blocs à 192, avant entrées/têtes. Les comptes exacts seront produits par le futur code, pas inventés ici.

**Choix :** A est la V0 minimale. Si A passe, ne pas construire B et C par réflexe. Si A plafonne sur une dépendance relationnelle diagnostiquée, tester B ; C sert ensuite de contrôle de capacité/biais. Une GRU 256 ajouterait environ 0,39M pour entrée et état de largeur 256, mais seulement dans une étude de mémoire.

### 6.3 M5 32 Go : faisabilité versus promesse

Pour Adam en FP32, poids, gradients et deux moments donnent environ **16 octets/paramètre**, hors activations et buffers. Certaines configurations mixtes/master copies conduisent à une enveloppe voisine de **16–20 octets/paramètre** ; inspecter la version effectivement utilisée.

| Paramètres | Poids 16 bits seuls | État entraînement estimé, hors activations |
|---:|---:|---:|
| 0,5M | 1 Mo | 8–10 Mo |
| 1,3M | 2,6 Mo | 21–26 Mo |
| 4M | 8 Mo | 64–80 Mo |
| 12M | 24 Mo | 192–240 Mo |
| 20M | 40 Mo | 320–400 Mo |

Ces nombres ne prédisent pas le pic mémoire du processus. Pour une attention matérialisée en 16 bits, `B × L × h × N² × 2` octets valent environ 8 Mio à `B=64,L=4,h=4,N=64`, mais 512 Mio à `N=512`, avant gradients et autres activations. Les kernels fusionnés changent ce profil. En V0 structurée, la taille des séquences et les copies/données sont plus susceptibles de coûter cher que les poids seuls.

**Protocole obligatoire :** runtime MLX unique pour A/B/C ; première validation FP32 ; précision réduite seulement après test de stabilité ; profiler CPU et GPU sur batch 1 ; distinguer compilation, warm-up et régime établi. MLX est paresseux [S26](sources.md#s26) : forcer le calcul et attendre sa fin. Mesurer adapter, encodage, scoring, choix et exécution séparément. Même état, même précision, même harness.

Mesurer p50/p95/p99 batch 1, débit d'entraînement, RSS/mémoire GPU disponible, swap, régime thermique soutenu et, si accessible, énergie système corrigée de l'idle sur une longue série. Si l'énergie n'est pas mesurable proprement, écrire **non mesurée**, pas “négligeable”.

**Enveloppe de décision, non estimation de vitesse :** profiling/pilote avant tout gros run ; maximum initial de 24 heures cumulées d'accélérateur pour V0 ; révision explicite si dépassement. Un run de `U` updates coûte approximativement `U × t_step + génération + évaluations`. À titre purement arithmétique, 5 000 updates à 0,05/0,2/1 seconde donnent 4,2/16,7/83,3 minutes hors surcoûts. Aucun de ces débits n'a été mesuré sur M5.

La RTX 3070 laptop 8 Go est un recours, pas une raison de maintenir deux stacks dès le premier jour. Comparer une exécution PyTorch seulement si MLX bloque réellement ou si une reproduction extérieure l'exige ; documenter TGP, versions et précision. Un numéro de GPU laptop ne détermine pas son débit soutenu.

## 7. Stratégie de données

### 7.1 Le meilleur premier générateur est plus petit que “TinyWorld”

**TinyGraphKey** : graphe non orienté connecté de 4–8 pièces ; agent, clé, colis, une porte verrouillable sur une arête ; capacité de transport un ; tous les états physiques pertinents observables. Buts : `REACH(room)`, `HAVE(object)`, `AT(object, room)`. Une porte déverrouillée le reste. Aucun texte libre, vision, permutation cachée de sémantique ou action macro.

Candidats : `MOVE` vers chaque pièce, `PICK` et `DROP` pour chaque objet, `UNLOCK` pour la porte, `STOP`. Ils ne dépendent ni du but ni d'une distance oracle. Action invalide : no-op physique **mais consomme un pas** et retourne un résultat observable d'échec. La politique doit apprendre les préconditions ; un bras secondaire peut recevoir un masque de validité, explicitement privilégié par rapport à cette interface.

À layout fixé avec `R` pièces, une borne brute des états physiques est `2R(R+1)²` avant exclusion des deux objets simultanément portés : à 12 pièces, 4 056. C'est assez petit pour un oracle exact. Le nombre de couples **état-but**, puis de layouts, est plus grand : ne pas confondre l'énumérabilité d'un layout avec la couverture de toute une famille.

### 7.2 Supervision et provenance

Pour chaque layout et but, un BFS inverse calcule la distance restante `d*`, puis toutes les actions réduisant cette distance d'un pas. À but satisfait, seul STOP est optimal. L'oracle sert à la supervision et au scoring d'évaluation ; il ne contribue pas à l'adaptateur runtime.

Stocker séparément : observation native, observation normalisée, candidats ; action réellement exécutée et résultat ; labels optimaux, distance et succès dans un canal inaccessible au modèle ; layout/générateur/graines dans un canal de provenance.

Après une perturbation, **ne pas imiter l'action erronée**. Annoter l'état atteint avec les actions de récupération de l'oracle. Un état irrécupérable est marqué tel quel, pas doté d'un “expert” fictif.

### 7.3 Échelle et comptabilité

- Diagnostic : 100 épisodes pour surapprentissage/replay, pas comme test scientifique.
- Pilote : environ 1 000 épisodes, puis 5 000 ; plafond 10 000 seulement si la courbe de données le justifie.
- Plafond V0 : 100k transitions supervisées fournies à l'apprenant ; DAgger optionnel +50k maximum ; arrêter aux frontières d'épisodes.
- Reporter transitions brutes, **couples uniques `(layout canonique, état, but)`**, nombres d'actions annotées, états effectivement calculés par l'oracle, temps oracle et expositions d'optimisation.
- Répéter cent fois un même label ne produit pas cent labels nouveaux. Les permutations d'un même exemple sont des augmentations, pas de nouvelles situations physiques.

L'oracle calcule déjà une grande partie des distances en une seule recherche. Dans ce jouet, une courbe en exemples uniques mesure l'efficacité **statistique de supervision**, pas un coût humain d'annotation. Sur un sous-ensemble de layouts, le baseline exhaustif quantifie la couverture ; si l'exhaustif est meilleur et moins coûteux, ne pas ritualiser DAgger.

### 7.4 Splits anti-contamination

Définir avant génération : environ 200 layouts train, 50 validation, puis pools disjoints d'au moins 100 layouts par test confirmatoire. Utiliser un hash structural comme préfiltre et un contrôle d'isomorphisme pour exclure les doublons entre pools. Regrouper tous les états, buts, perturbations et réétiquetages d'un layout dans le même pool.

Tests séparés : instances ID, associations but/objet/destination absentes du train, topologies retenues, tailles 9–12, distances optimales plus longues, permutations cohérentes. Cinq tâches par layout donnent 500 épisodes par cellule ; ce sont **100 clusters**, pas 500 mondes indépendants.

Ne pas appeler “composition” un simple changement de couleur. Les primitives doivent avoir été vues séparément ; l'assemblage testé doit réellement être absent. Aucun choix de générateur, de seuil ou de checkpoint après lecture du test scellé.

### 7.5 Sources ultérieures

MiniGrid/BabyAI : réutilisation utile après audit de l'observation réellement offerte. ALFWorld : plus coûteux et sémantiquement riche, pas un simple ajout de noms d'entités. OfficeWorld : attention à ne pas rebaptiser pièces en dossiers et `PICK` en `OPEN`. Une véritable deuxième dynamique peut imposer création, duplication, envoi sans déplacement, permissions et dépendances d'approbation. Synthetic UI peut introduire sélection, formulaires, commits atomiques et navigation d'état, d'abord pleinement observable pour ne pas confondre transfert et mémoire.

Les trajectoires LLM ne viennent qu'après les oracles quand ces derniers sont disponibles. Documenter taux d'échec, coût, vérification, filtrage et couverture ; une paraphrase ne crée pas une nouvelle compétence de contrôle.

## 8. Stratégie d'entraînement

### 8.1 V0 : une seule loss principale

```text
A*(s,g) = ensemble des actions optimales, STOP seul si but satisfait
L_action = -log Σ[a ∈ A*(s,g)] πθ(a | observation, but, candidats)
```

La loss ne pénalise pas arbitrairement une autre première action optimale. Enregistrer la cardinalité `|A*|` pour interpréter les diagnostics. Dans ce MDP exact à coût strictement positif, **toute action de A* diminue d*** : choisir successivement de vraies actions optimales ne peut pas créer une oscillation. Les boucles observées proviendraient d'erreurs de politique/oracle ou de dynamiques différentes, pas de l'existence de plusieurs optimums.

Utiliser softmax/logsumexp stable et masque de padding uniquement. Ne jamais transformer le masque oracle en masque d'entrée ou de logits. Aucune perte progress/value/next-state en V0 ; elles seraient des traitements expérimentaux distincts.

### 8.2 Ordre d'apprentissage

1. Tests de replay, invariance, candidats et oracle avant tout entraînement.
2. Surapprentissage du minuscule ensemble de diagnostic, avec STOP présent.
3. BC sur mélange contrôlé des trois types de buts et des bandes de difficulté ; un curriculum ne doit pas supprimer définitivement les cas courts.
4. Évaluation fermée sur validation ; classifier les échecs : représentation, action invalide, mauvaise cible, oubli de STOP, détour, boucle.
5. Ajouter données de perturbation ou DAgger seulement si le défaut observé est une sortie du support d'entraînement.
6. Comparer à un nombre égal de nouveaux états uniformes/oracle ; un gain de DAgger doit provenir de la distribution des états, pas de plus de labels.
7. Un second encodeur seulement si les données/interfaces n'expliquent plus le plafond.

Configuration initiale proposée : AdamW, `lr=3e-4`, `weight_decay=1e-4`, batch 64, clipping norme 1, FP32, maximum 10 époques ou 10k updates pour le premier BC, première limite atteinte. Sauvegarde et évaluation validation toutes les 500 updates et à la fin. Le test de surapprentissage a son propre budget, jusqu'à 5k updates. Ce sont des points de départ, pas des optimums connus.

Autoriser au plus trois learning rates sur le développement, même budget de recherche par architecture ; sélectionner sur succès validation, avec coût/longueur en départage. Trois graines exploratoires ; cinq graines pour les revendications finales. Aucun choix sur le meilleur seed.

### 8.3 Quand ne pas ajouter une technique

Pas de préentraînement auto-supervisé si BC suffit ; pas de RL avant d'avoir un échec que de bonnes démonstrations ne résolvent pas ; pas de mémoire dans un monde pleinement observable pour masquer un mauvais encodeur ; pas d'options si les horizons sont déjà courts ; pas de diffusion pour quinze actions discrètes ; pas de monde latent seulement parce que les transitions ont été stockées.

Les représentations de langage préentraînées sont un bras ultérieur : comparer coût total, connaissances importées et objectifs structurés parfaits. Le parsing du but est une tâche séparée avec son taux d'erreur, pas un oracle gratuit.

## 9. Évaluation, baselines et critères de décision

### 9.1 Matrice de généralisation

| Cellule | Ce qui change | Ce qui reste constant | Ce que le résultat autorise |
|---|---|---|---|
| G0 | Nouveaux états/buts sur layouts connus | Dynamique et topologie | Interpolation, diagnostic seulement |
| G1 | Layouts non isomorphes non vus | Taille, primitives, difficulté | Généralisation d'instances |
| G2 | Combinaisons but/objets/destinations retenues | Primitives individuellement connues | Composition définie, pas “raisonnement général” |
| G3 | Taille 9–12 et topologies retenues | Sémantique des actions | Extrapolation structurelle |
| G4 | `d*` 13–24 contre 2–12 en train | Même sémantique | Généralisation d'horizon ; publier les histogrammes |
| G5 | IDs et ordres permutés simultanément partout | Même problème physique | Invariance/équivariance, pas nouvelle tâche |
| G6 | Observation partielle contrôlée | Dynamique connue | Besoin de mémoire et de collecte d'information |
| G7 | Effets/préconditions nouveaux, interface documentée | Budget d'adaptation fixé | Transfert vers une dynamique nouvelle |
| G8 | Nouvelle famille et nouvel adaptateur | Information offerte et coût déclarés | Transfert inter-familles limité au protocole |

Les cellules G3/G4 sont rapportées séparément ; les croiser est un stress test supplémentaire, pas une manière de confondre taille et horizon.

### 9.2 Baselines indispensables, par étape

**V0 économique :** random syntaxique, random valide explicitement privilégié, heuristique locale sans accès aux distances, recherche exacte sur modèle connu, retrieval/kNN de cas train, Deep Sets A. Une heuristique recevant la distance exacte serait déjà une recherche déguisée : l'indiquer.

**Ablations d'interface :** sans but, but contrefactuel valide, sans relations, candidats seuls, ordre/IDs randomisés, candidats syntaxiques contre masque de validité observable. Le bras sans but est un **contrôle négatif**, pas une preuve suffisante de qualité s'il perd.

**Architecture, conditionnellement :** B puis C ; à données communes d'abord, puis capacité comparable si l'on veut isoler le biais architectural. Éviter de conclure “les graphes gagnent” en comparant simplement des tailles et budgets différents.

**Transfert :** scratch ; préentraînement A seul ; B seul ; joint A+B ; source auxiliaire de contrôle ; éventuellement encodeur préentraîné gelé versus fine-tune complet. Pas besoin d'un LLM dans V0 pour prouver que des réseaux peuvent choisir des actions.

**Web ou démonstration de remplacement d'un LLM :** petit modèle textuel/LLM local et professeur fort sur les mêmes tâches, mêmes outils, même visibilité et budget ; compter parsing, prompts, tokens, tentatives, exceptions et réussite finale. Tant que cette étude n'a pas eu lieu, retirer l'affirmation “remplace les agents LLM”.

### 9.3 Mesures primaires et diagnostics

**Primaires :** succès d'épisode vérifié, coût/pas de bout en bout, risque d'arrêt prématuré. STOP ne rapporte succès que si la postcondition indépendante est vraie ; but atteint sans STOP avant timeout compte comme échec de la tâche complète, avec métrique secondaire “but physiquement atteint”.

**Longueur/efficacité :** `L*` inclut STOP ; `L` inclut invalides. Reporter regret `L−L*` sur succès, séparément du taux d'échec pour éviter le biais du survivant ; ajouter coût tronqué `H+1` aux échecs comme score secondaire explicite, non comme distance optimale prétendue. Échecs, timeouts et invalides restent visibles.

**Diagnostics :** optimal-action rate, masse sur A*, taux d'actions invalides, longueur et répétitions, couverture des candidats, variance inter-graines, résultats par type de but/horizon/taille. Une accuracy d'action élevée ne suffit pas : même le modèle simplifié d'erreurs indépendantes donne `0,99^50 ≈ 0,61` ; ce calcul est illustratif, pas une prédiction d'un système avec récupération et erreurs corrélées.

**Confiance, plus tard :** NLL/Brier, courbes de calibration, risque-couverture, sélectivité et erreurs catastrophiques, par nombre de candidats et OOD. `max softmax` n'est pas directement une probabilité de succès futur. Température sur validation [S23](sources.md#s23), jamais sur test ; pas de garantie OOD annoncée.

### 9.4 Statistiques et gates V0

Même pool d'épisodes pour tous les bras. Cinq seeds d'entraînement confirmatoires, au moins 100 layouts × cinq tâches par cellule. Intervalles à 95 % par bootstrap apparié hiérarchique : resampler graines et layouts, puis tâches dans layout ; publier également chaque graine. Cinq graines restent une petite réplication, pas une certification.

Gates proposés :

- **Intégrité** : replay exact, pas de fuite, optimal coverage 100 % sur l'univers V0 vérifié, STOP et invalides conformes ; tolérance de permutation explicitée par précision.
- **Apprentissage** : ≥99 % d'actions optimales sur le diagnostic de surapprentissage ; aucun appel oracle en runtime modèle.
- **G1** : succès moyen ≥95 %, borne basse IC ≥90 %, aucune catégorie de but <85 % ; **G2** ≥80 % comme gate de composition. Seuils de faisabilité, pas vérités universelles.
- **Utilité d'un réseau plus complexe** : gain de succès ≥5 points sur la cellule d'échec préspécifiée, IC du gain excluant zéro, sans dégradation G1 >2 points ; coût ≤2× A. Sinon garder A.
- **Objectif de latence V0** : p95 modèle batch 1 ≤20 ms et p95 décision avec adaptateur ≤50 ms sur le profil déclaré, sinon reprofiler/simplifier. Seuil opérationnel proposé, aucun timing acquis.

Si un baseline simple atteint déjà les seuils, le résultat n'autorise pas une nouvelle architecture : il justifie au contraire de s'arrêter à cette solution. Si une limite compute empêche une comparaison suffisante, verdict **inconclusif**, non réfutation générale.

### 9.5 Transfert : protocole qui peut effectivement échouer

**Ce qui est mesuré.** Succès **en boucle fermée sur des layouts cibles non vus**, après apprentissage sur `k` exemples cibles uniques. L'axe horizontal mesure la supervision donnée au modèle, pas la difficulté d'obtenir un label oracle. Rapporter couverture physique et état-but par layout, **et la couverture atteinte par bras à chaque budget, notamment `k*=500`** : un négatif à couverture quasi nulle est ininterprétable et doit être rapporté comme tel, pas présenté comme réfutation du transfert.

À couverture partielle, le régime réellement testé est une **généralisation inter-layout à couverture partielle** — c'est le nom correct de la revendication, plus fort et plus falsifiable qu'un vague « le transfert aide ».

**Sources.** `A+B` signifie entraînement joint, minibatches équilibrés, même total source que A seul ou B seul, et mêmes updates/expositions. Pas de fusion de poids, ensemble ou A→B séquentiel caché. Par exemple 100k exemples uniques totaux : 50k A + 50k B, contre 100k A ou 100k B. La source de contrôle étudie si un préentraînement non orienté vers les mêmes dépendances suffit ; déclarer son contenu et sa limite, sans présumer qu'il sera parfaitement “sans rapport”. **Reporter la couverture des sources dans la même unité de couples uniques** : des volumes égaux en transitions peuvent cacher des couvertures très différentes.

**Cibles.** Deux familles `C1/C2` préspécifiées, dynamiques effectivement différentes et générateurs scellés. Les essais d'hyperparamètres utilisent une famille de développement distincte, jamais les succès cibles. À deux familles, conclure au mieux « transfert répliqué sur ces deux familles », pas fréquence de réussite sur toutes les interfaces.

**Comparaison primaire préenregistrée :** à `k*=500` couples état-but uniques, A+B contre scratch, cinq seeds par bras et par cible, mêmes tuples d'adaptation imbriqués, même initialisation des nouveaux embeddings/têtes, mêmes updates cible et même sélection de checkpoint. Un checkpoint final à budget d'updates fixé évite la recherche opportuniste sur la cible.

**Gate primaire :** sur **chacune** de C1 et C2, différence moyenne de succès ≥5 points et borne basse IC apparié à 95 % >0. Si l'une échoue, pas de revendication répliquée. Le protocole de confirmation sera figé après pilote sur développement mais avant ouverture des cibles. Ce critère échoue explicitement en cas d'incertitude excessive.

**Courbes secondaires :** `k ∈ {0,100,500,2000,10000}`, AULC du **succès fermé** en fonction de `log(1+k)` sur grille commune. Nombre de labels pour 80 % : valeur censurée `>10000` si jamais atteint, jamais “ignore cette condition si non croisée”. Une division par deux de ce nombre serait une revendication secondaire séparée, nécessitant des croisements observés ; elle n'est pas une échappatoire au gate primaire.

**Contrôle de calcul.** La comparaison principale fixe les updates cibles. Un scratch renforcé reçoit, avec les mêmes labels, un budget supplémentaire d'updates/expositions comparable au calcul source + adaptation ; rapporter FLOPs estimés et temps réel. Le temps M5 est observé, pas l'unique variable de matching : température, backend et throttling le rendent fragile. Si scratch convergé rejoint A+B, parler d'avantage d'optimisation, pas automatiquement d'efficacité en données persistante.

**Attribution.** A+B doit dépasser le meilleur préentraînement mono-source si l'on veut attribuer le gain à la diversité. Sinon : transfert positif, contribution multi-source non démontrée. Réinitialiser séparément encodeur et tête, puis comparer gel/fine-tune, permet de localiser le transfert. Le coût source et le coût humain des adaptateurs ne disparaissent pas du bilan ; calculer aussi le nombre de déploiements nécessaire pour amortir le préentraînement.

## 10. V0 minimale directement implémentable

**Une phrase :** apprendre à choisir une action complète, STOP compris, dans un petit monde relationnel entièrement observable, puis mesurer les échecs sur layouts disjoints.

**Inclus :** TinyGraphKey déterministe ; trois prédicats de but ; interface versionnée ; candidats syntaxiques ; oracle exact offline ; un encodeur A ; BC set-valued ; harness fermé ; baselines random/heuristique/recherche ; permutation, contamination, coût et succès.

**Exclus :** NL, Web, ALFWorld, pixels, audio, mémoire, embeddings de domaine, valeur/progression, world model, RL, options, latent skills, LLM runtime, diffusion, dizaines de millions de paramètres et framework multi-adaptateurs anticipé.

Ordre de travail ultérieur :

1. Décrire transitions/préconditions/postconditions et petit univers énumérable.
2. Tester oracle, candidats, STOP, invalides, séparation inputs/labels et replay.
3. Sceller layouts et règles de génération ; produire manifest/hash et rapport de couverture.
4. Implémenter baseline de recherche et heuristique **avant** réseau.
5. Générer le diagnostic de 100 épisodes ; implémenter uniquement A.
6. Surapprendre puis profiler, pour distinguer bugs et capacité insuffisante.
7. Pilote 1k/5k épisodes ; choisir la configuration sur validation dans un budget borné.
8. Confirmer G1/G2, puis mesurer G3/G4 sans masquer les échecs.
9. Inspecter vingt échecs tirés de catégories définies, pas seulement vingt exemples spectaculaires.
10. Décision : garder le baseline, corriger interface/données, tester B, ou arrêter. **Ne pas lancer automatiquement V1.**

Livrables attendus lors de l'implémentation future : spécification de transitions, tests, manifest de splits, fichier de configuration, données/version, seeds, checkpoint, scores par épisode, profiling et rapport incluant résultats négatifs. La présente mission n'en simule aucun.

## 11. Roadmap conditionnelle

| Étape | Question autorisée | Déclencheur | Comparaison obligatoire | Stop si… |
|---|---|---|---|---|
| V0a | L'interface permet-elle un contrôle non trivial sans fuite ? | Aucun | Random, heuristique, oracle, A | Bugs, trivialité, manque d'information |
| V0b | Des relations/capacité supplémentaires aident-elles ? | Plafond diagnostiqué d'A | B puis C, mêmes données | Gain absent ou coût disproportionné |
| V1 | Existe-t-il un transfert d'initialisation utile ? | V0 fiable et sources vraiment différentes | Scratch fort, A, B, A+B, contrôle ; point primaire fixe | Gain absent, coût non amortissable ou reskin |
| V1-S (pont SIW) | Le principe survit-il aux interfaces logiciel-like ? | Transfert évalué sur C1/C2 | Scratch SIW, préentraînement graphique, taxe d'adaptateur mesurée ; labels arbitraires, arguments, many-candidates | Gain absent ou reskin : publier la frontière et revoir l'ambition avant le Web |
| V2-M | Une mémoire est-elle nécessaire ? | Aliasing démontré dans des histoires | Sans mémoire, historique explicite, GRU | Pas de gain spécifique aux cas ambigus |
| V2-L | Quel est le coût réel du langage ? | Besoin produit et objectifs structurés robustes | Parseur contrôlé, petit encodeur, oracle de parsing | Gain de contrôle masqué par erreurs sémantiques |
| V3 | Une recherche courte vaut-elle son coût ? | Ambiguïtés de décision identifiées | Modèle vrai, modèle appris, policy avec compute égal | Le modèle vrai n'aide pas ou l'appris exploite ses erreurs |
| V4 | Options/chunks améliorent-ils l'horizon ? | Tâches longues et frontières stables | Politique plate avec mêmes données et budget | Moins de réaction, échec à l'interruption |
| V5 | Le transfert survit-il à un vrai adaptateur Web ? | Gain de transfert confirmé | BrowserGym, scratch web, modèle textuel ; mêmes infos | Coût de parsing/candidats domine ou réussite online insuffisante |

L'ordre mémoire/langage est un embranchement, pas une obligation chronologique. World model et RL ne sont pas des récompenses automatiques pour avoir terminé la phase précédente.

## 12. Red team : dix modes d'échec scientifiques

| # | Échec | Symptôme trompeur | Expérience adversariale bon marché | Réponse |
|---:|---|---|---|---|
| 1 | Adaptateur-oracle | Petit modèle “très intelligent” | Candidats seuls, sans but ; audit des dépendances du code | Retirer filtrage orienté tâche, réattribuer le résultat |
| 2 | Fuite de labels/progression | STOP parfait, actions prévisibles | Allowlist ; modifier labels sans modifier sortie ; exclure timestep/distance/futur | Régénérer les entrées, invalider scores contaminés |
| 3 | Contamination isomorphe | OOD étonnamment facile | Renommages, hash + isomorphisme, groupe de provenance | Refaire split avant nouveaux résultats |
| 4 | Mémorisation d'IDs/ordre | Score fragile à un shuffle | Permuter observation, but, relations et candidats ensemble | Architecture équivariante, pas simple augmentation cosmétique |
| 5 | Confusion interpolation/composition | Nouvelles couleurs qualifiées de nouvelles tâches | Holdout d'assemblages d'opérateurs/prédicats | Restreindre le claim |
| 6 | Aliasing/non-observabilité | Plafond attribué au manque de paramètres | Deux histoires, même entrée courante, actions optimales disjointes | Mémoire ou nouvelle observation ; ne pas scaler aveuglément |
| 7 | Imitation sans contrôle | Bonne accuracy, faible succès | Rollouts, invalides, détours, données uniformes vs DAgger | Réparer couverture et feedback |
| 8 | Faux transfert | Courbe pretrained au-dessus de scratch | Budgets source/cible égaux, scratch convergé, heads/adapter identiques | Distinguer warm start, connaissances et gain de données |
| 9 | World model exploité | Bonne reconstruction, mauvais plans | Rank des actions, recherche avec modèle vrai, erreurs multi-step | Réduire horizon, apprendre tâches pertinentes ou abandonner |
| 10 | Résultat sélectionné et coût incomplet | Meilleur seed, ms du noyau, fallback caché | Protocole scellé, tous seeds, coût système et appel LLM compté | Retirer la revendication jusqu'à mesure contrôlée |

Un onzième risque transversal mérite mention : l'architecture peut être correcte et le projet inutile si le planificateur symbolique disponible est déjà plus simple, rapide et fiable. Ce n'est pas un échec d'implémentation ; c'est une réponse scientifique valide.

## 13. Opportunités de recherche moins évidentes

Ces pistes sont des hypothèses, pas des inventions revendiquées.

| Opportunité | Hypothèse testable | Expérience discriminante | Pourquoi cela peut compter / risque |
|---|---|---|---|
| **Taxe d'adaptation de l'interface** | Une grande part du gain vient de l'information offerte plutôt que du cœur | Même contrôleur sous niveaux d'adaptateur : brut typé, préconditions, préconditions+effets, filtrage | Benchmark de validité utile même si UCM échoue ; risque de comparer des tâches non équivalentes sans le dire |
| **Plancher d'information pour le transfert** | Certaines interfaces rendent le zéro-shot non identifiable | Paires de dynamiques aliasées, puis ajout progressif de descriptions/démos/exploration | Délimite précisément les promesses possibles ; petit résultat négatif plus solide qu'une grande démo |
| **Contrôle comme compilation amortie** | Le réseau compresse des résultats de recherche sans généraliser l'algorithme | Entraîner sur petits graphes, tester diamètre/branches croissants ; comparaison recherche exacte | Identifier le domaine où la compilation vaut son coût ; ne pas confondre imitation d'un solveur et solveur appris |
| **Calcul adaptatif avec poids partagés** | Répéter des mises à jour relationnelles aide plus que grossir le modèle | Même GNN, 1/3/6/12 itérations, coût mesuré, tâches à profondeur contrôlée | Explore la profondeur de calcul plutôt que les paramètres ; risque de stabilité hors profondeur train |
| **Localisation du transfert** | Le gain est dans l'encodage d'état, pas dans la politique | Reset tête/encodeur, gel, probes d'effets, permutation de sémantique documentée | Une conclusion plus précise qu'“intelligence de contrôle” ; probes seules insuffisantes sans succès fermé |
| **Valeur de l'information avant valeur d'action** | Un petit agent peut apprendre quand inspecter plutôt que quand agir | POMDP minuscule : indice caché, action INSPECT coûteuse, distracteurs ; comparaison mémoire seule | Teste collecte d'information et pas simple replay ; un oracle de visibilité rendrait le résultat trivial |
| **Vérificateur indépendant avant confiance sophistiquée** | Un contrôle de postconditions apporte plus de fiabilité par ms qu'un plus gros policy | Même politique avec/sans vérificateur, erreurs d'observation injectées, coût et faux refus | Sépare capacité et sûreté ; le vérificateur peut lui-même être privilégié et doit être audité |

La piste la plus rentable pourrait être **une méthode pour démontrer où réside la compétence**, plutôt qu'une nouvelle architecture. Elle transforme un échec de généralité en connaissance réutilisable.

## 14. Verdict de nouveauté et ambition défendable

**Aujourd'hui :** aucune nouveauté architecturale forte n'est établie ; la recette est surtout une combinaison de techniques connues. L'état de l'art consulté ne justifie pas “premier contrôleur universel non autoregressif”, “remplacement général des LLM” ou “System 1 général”.

**Contribution minimale crédible :** une comparaison reproductible de petits contrôleurs structurés et de baselines classiques, avec audit d'interface, coûts complets et tests OOD correctement scellés.

**Contribution plus forte :** démontrer, sur plusieurs dynamiques/familles non vues, un gain de succès fermé sous adaptation limitée, qui survit aux contrôles de calcul, données, head/encoder et coût d'adaptateur. Cela reste une généralité **dans un support déclaré**.

**Contribution potentiellement originale :** isoler une propriété de représentation, de calcul adaptatif ou de supervision qui explique causalement ce transfert et résiste à un contrôle contrefactuel. La revendication exacte devra faire l'objet d'une nouvelle passe bibliographique ciblée, incluant ASNets, généralisation relationnelle, politiques multi-tâches et universal planning.

**Résultat négatif publiable :** montrer que les gains prétendument universels disparaissent lorsqu'on compte l'intelligence des adaptateurs, l'exposition aux données et les choix de cibles. Ce n'est pas une promesse de publication : c'est un résultat plus informatif qu'un benchmark favorable non contrôlé.

## 15. Priorités P0–P3

### P0 — Avant le premier entraînement

- Unifier le sens de V0 ; écrire les hypothèses et non-objectifs.
- Contrat observable/labels/évaluateur ; candidats goal-blind ; invalides avec coût ; STOP vérifié.
- Oracle exact testé, split structurel/provenance, unités d'exemples uniques.
- Baselines random, heuristique et recherche ; métrique primaire fermée.
- Profiling réel sur M5, budget maximal, manifest reproductible.

### P1 — Première preuve expérimentale

- Un seul petit encodeur A ; contrôle négatif sans but et équivariance.
- Courbe de données, BC versus couverture/oracle ; DAgger seulement motivé.
- Cinq seeds confirmatoires, IC appariés et rapport des échecs.
- Décider si un deuxième encodeur est nécessaire, non prévu par défaut.

### P2 — Seulement après V0 interprétable

- Source additionnelle aux dynamiques distinctes ; cibles scellées ; protocole de transfert.
- Mémoire **ou** langage selon besoin démontré.
- Calibration sélective et vérification des postconditions pour un contexte réaliste.

### P3 — Spéculatif/conditionnel

- Modèle latent, recherche, options et RL.
- Web/vision, nouvelles sémantiques d'action, arguments générés.
- Montée vers 10–20M puis plus ; aucune raison a priori d'aller à 50M.
- Voix et produit généraliste : interfaces périphériques, pas preuve de contrôle.

## 16. Spécification révisée

La **réécriture complète et autonome** remplace le fichier [`universal-control-model-project-spec.md`](../../universal-control-model-project-spec.md). Elle n'est pas un simple addendum aux contradictions originales. L'original est conservé à l'identique dans ce dossier.

Structure normative de la nouvelle spec :

1. Statut, limites et définition du résultat recherché.
2. Hypothèses et niveaux de revendication.
3. Périmètre et non-objectifs V0.
4. TinyGraphKey : états, transitions, actions, buts, terminaison, oracle.
5. Contrat d'information et schéma de données.
6. Architecture A et comparateurs conditionnels B/C.
7. Génération, couverture, splits et curriculum.
8. Apprentissage et sélection de modèles.
9. Évaluation, baselines et règles statistiques.
10. Budget M5/MLX, profiling et limites de ressources.
11. Critères de passage/arrêt et diagnostic.
12. Protocole de transfert conditionnel.
13. Branches mémoire/langage/recherche/pont SIW/Web.
14. Sûreté hors toy world.
15. Artifacts, ordre d'implémentation future et définition de terminé.
16. Traçabilité des changements et limites de la revue.

**Ce qui a été supprimé de V0 :** têtes de progression/terminaison séparées, plusieurs familles obligatoires, mémoire, langage, planification latente, skills et infrastructure anticipée. **Ce qui a été renforcé :** contrôle de l'interface, pooling local par entité et tags de rôle du but normatifs, gestion des optimums multiples, succès vérifié, unités de supervision avec couverture par bras, baselines classiques, comparaison de transfert falsifiable, hash du protocole avant ouverture des cibles et comptabilité du calcul.

**Conclusion :** construire d'abord une expérience qui peut honnêtement répondre « non ». Si elle répond « oui », on saura beaucoup mieux à quoi elle a répondu.
