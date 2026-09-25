# Sources et discipline de preuve — revue UCM

Date de consultation : 22 septembre 2026. Revue ciblée, **non exhaustive**. Les recherches ont privilégié articles, pages de conférences, documentations officielles et dépôts des auteurs. Les quotas horaires de recherche, puis de récupération web, ont été atteints. Aucun benchmark, entraînement ou test matériel n'a été réalisé pour cette revue.

« Consulté » signifie que la source primaire, sa page de publication ou son résumé a été récupéré, **pas** que chaque expérience a été reproduite ni chaque annexe intégralement examinée. Les extractions longues d'ACT et de Diffusion Policy étaient tronquées ; les passages utilisés ci-dessous étaient présents. Les chiffres restent ceux des auteurs. Les pages de dépôts sont modifiables : épingler leurs commits avant une reproduction.

## Sources consultées

<a id="s01"></a>
### S01 — Universal Value Function Approximators, Schaul et al., ICML 2015
https://proceedings.mlr.press/v37/schaul15.html

Précédent pour la généralisation conditionnée par état et objectif. Il s'agit d'approximation de fonctions de valeur ; cela ne démontre pas un contrôleur logiciel universel ni un alignement automatique de sémantiques d'action.

<a id="s02"></a>
### S02 — A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning, Ross et al., AISTATS 2011
https://proceedings.mlr.press/v15/ross11a.html

DAgger traite le décalage de distribution induit par les actions de l'apprenant via collecte et annotation sur ses états visités. Référence centrale pour comparer BC, perturbations et corrections en boucle fermée. Les hypothèses théoriques ne sont pas une garantie de performance pour cette V0.

<a id="s03"></a>
### S03 — Action Schema Networks: Generalised Policies with Deep Learning, Toyer et al., AAAI 2018
https://ojs.aaai.org/index.php/AAAI/article/view/12089

ASNets exploite la structure relationnelle des problèmes de planification et un partage de poids pour généraliser entre instances d'un domaine. C'est un voisin beaucoup plus direct de la V0 structurée que les seuls agents LLM. Généraliser en taille dans un domaine n'est pas transférer entre familles arbitraires.

<a id="s04"></a>
### S04 — Set Transformer, Lee et al., ICML 2019
https://proceedings.mlr.press/v97/lee19d.html

Architecture pour ensembles, interactions et invariance par permutation ; mécanisme à points inducteurs pour réduire le coût lié à la taille des ensembles. Ne pas attribuer automatiquement l'invariance à un Transformer comportant des positions séquentielles arbitraires.

<a id="s05"></a>
### S05 — A Generalist Agent / Gato, Reed et al., 2022
https://arxiv.org/abs/2205.06175v1

Un ensemble de poids partagé sur 604 tâches de différentes modalités. Précédent majeur de politique généraliste ; modèle autoregressif. Ce nombre ne quantifie pas une capacité à apprendre une nouvelle interface sans données.

<a id="s06"></a>
### S06 — Decision Transformer, Chen et al., 2021
https://arxiv.org/abs/2106.01345v1

Apprentissage de contrôle par modélisation causale de séquences conditionnée par retour. Un retour désiré n'est pas, à lui seul, une représentation suffisante d'un objectif logiciel structuré.

<a id="s07"></a>
### S07 — Jack of All Trades, Master of Some / JAT, 2024
https://arxiv.org/pdf/2402.09844

Précédent ouvert d'agent Transformer multi-tâches et multi-domaines. Utile pour réfléchir au format des données et aux évaluations ; aucun chiffre précis de performance JAT n'est nécessaire à la recommandation.

<a id="s08"></a>
### S08 — Octo: An Open-Source Generalist Robot Policy, 2024
https://arxiv.org/abs/2405.12213
https://github.com/octo-models/octo

Politique robotique généraliste à diffusion, préentraînée sur 800 000 trajectoires ; modularité des observations/actions et adaptation. Ne pas confondre non-autoregressif, une seule passe réseau et coût faible du système entier. L'ampleur du corpus compte dans la comparaison.

<a id="s09"></a>
### S09 — RT-2, présentation officielle, 2023
https://www.deepmind.com/blog/rt-2-new-model-translates-vision-and-language-into-action

Transfert de connaissances vision-langage vers des actions robotiques. Source de contexte et non preuve qu'un petit modèle structuré entraîné de zéro possédera cette connaissance sémantique.

<a id="s10"></a>
### S10 — SmolVLA, documentation officielle LeRobot
https://huggingface.co/docs/lerobot/main/en/smolvla

La documentation consultée décrit un modèle préentraîné de 450M paramètres. « Petit VLA » n'est donc pas synonyme de contrôleur de 1–20M paramètres. Compter également perception, encodage du langage et coût du préentraînement.

<a id="s11"></a>
### S11 — Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware / ACT, Zhao et al., 2023
https://arxiv.org/abs/2304.13705

Prédiction de chunks d'actions avec un Transformer et un CVAE ; temporal ensembling ; apprentissage par imitation. L'article décrit environ 80M paramètres et un entraînement distinct par tâche. Précédent contre la nouveauté d'une sortie d'actions non textuelle/parallèle, pas démonstration de transfert logiciel universel. Les résultats réels varient fortement selon la tâche ; ne pas résumer tout le papier par « 80–90 % de succès ».

<a id="s12"></a>
### S12 — Diffusion Policy: Visuomotor Policy Learning via Action Diffusion, Chi et al., 2023, version étendue consultée
https://arxiv.org/abs/2303.04137

Distribution conditionnelle de séquences d'actions, débruitage itératif et commande à horizon glissant. Une alternative à la génération token par token, **mais plusieurs itérations d'inférence**. Les coûts d'une version ou d'un GPU ne prédisent pas ceux d'un M5.

<a id="s13"></a>
### S13 — TD-MPC2: Scalable, Robust World Models for Continuous Control, Hansen et al., 2023/2024
https://arxiv.org/abs/2310.16828
http://www.tdmpc2.com/

Planification dans un espace latent. Distinguer les évaluations sur 104 tâches avec une configuration commune et l'expérience multi-tâches à un seul modèle de 317M sur 80 tâches. Échelle de modèles étudiée : 1M–317M. Ne pas convertir « même algorithme » en « mêmes poids ».

<a id="s14"></a>
### S14 — DreamerV3, page officielle du projet
https://danijar.com/project/dreamerv3/

Le projet rapporte plus de 150 tâches avec une même configuration. Ce n'est pas l'affirmation qu'un unique checkpoint contrôle toutes ces tâches. La récupération HTML de la version arXiv initiale a échoué ; la revue s'appuie ici sur la page officielle, pas sur une lecture intégrale de cette version du papier.

<a id="s15"></a>
### S15 — Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model / MuZero, Schrittwieser et al., 2019/2020
https://arxiv.org/abs/1911.08265v1

Modèle appris au service de récompenses, valeurs et politiques utilisées dans la recherche ; une reconstruction complète de l'observation n'est pas une condition nécessaire. Cela ne supprime ni le coût des données ni les erreurs de modèle hors distribution.

<a id="s16"></a>
### S16 — V-JEPA 2, 2025
https://arxiv.org/abs/2506.09985

Prédiction de représentations vidéo et extensions orientées action/planification. Précédent pertinent pour modèles latents, mais les résultats à grande échelle ne démontrent pas une recette locale bon marché pour cette V0.

<a id="s17"></a>
### S17 — Latent Action Pretraining from Videos / LAPA, 2024
https://latentactionpretraining.github.io/
https://arxiv.org/html/2410.11758v1

Actions latentes quantifiées à partir de transitions vidéo, préentraînement puis adaptation avec actions annotées. Une variable latente discrète n'est pas une sémantique universelle d'action identifiable sans ancrage.

<a id="s18"></a>
### S18 — AnyMDP / méta-entraînement pour in-context RL, version 4 consultée
https://arxiv.org/pdf/2502.02869v4.pdf

Environnements MDP procéduralement diversifiés pour étudier l'adaptation en contexte. Sert de contrepoint à une simple randomisation cosmétique. La revue n'utilise aucun chiffre précis de ce travail.

<a id="s19"></a>
### S19 — OGBench, page officielle
https://seohong.me/projects/ogbench/

Benchmark d'offline goal-conditioned RL : huit types d'environnements et 85 datasets annoncés sur la page consultée. Source d'idées et de baselines sur stitching, stochasticité et horizons longs ; pas un remplacement exact d'un benchmark de contrôle d'interfaces typées.

<a id="s20"></a>
### S20 — WebLINX, 2024
https://arxiv.org/abs/2402.05930v2

Environ 100K interactions, 2 300 démonstrations et plus de 150 sites. Les résultats sur des sites non vus restent difficiles ; une bonne prédiction offline de prochaine action n'établit pas le succès d'une tâche en ligne.

<a id="s21"></a>
### S21 — BrowserGym, dépôt officiel
https://github.com/ServiceNow/BrowserGym

Infrastructure existante pour plusieurs benchmarks web, notamment MiniWoB, WebArena et WorkArena. Éviter de reconstruire un framework de navigation complet avant d'avoir prouvé une contribution de contrôle. BrowserGym est une infrastructure, pas un corpus homogène ni un résultat de performance.

<a id="s22"></a>
### S22 — WebArena, site du projet
https://webarena.dev/
https://webarena.dev/og/

Le domaine principal consulté pointe aujourd'hui vers une suite élargie ; distinguer cette suite du benchmark original. Pour une reproduction, figer la version exacte des environnements, tâches et évaluateurs.

<a id="s23"></a>
### S23 — On Calibration of Modern Neural Networks, Guo et al., ICML 2017
https://proceedings.mlr.press/v70/guo17a.html

Mauvaise calibration possible des réseaux modernes ; temperature scaling comme baseline. Aucun certificat de fiabilité OOD ni de sûreté d'une mutation logicielle ne découle d'un score softmax calibré sur validation.

<a id="s24"></a>
### S24 — Laya, dépôt des auteurs
https://github.com/NandhaKishorM/laya

Le README consulté liste des checkpoints ModernBERT-large à 421M et mmBERT-base à 322M paramètres, avec décisions typées et sorties non autoregressives. Les latences T4 annoncées, notamment 33 ms pour une question et 7,2 ms/question en batch, sont des **chiffres des auteurs**, pas des mesures indépendantes ni des latences M5. Le dépôt n'établit pas une fiabilité de contrôle interactif à long horizon. Inspiration de sortie typée, non validation scientifique du projet UCM.

<a id="s25"></a>
### S25 — MLX, mémoire unifiée, documentation officielle
https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html

Mémoire partagée CPU/GPU. Les 32 Go sont aussi utilisés par le système et les autres processus ; ce n'est pas une enveloppe de 32 Go exclusivement allouable au modèle.

<a id="s26"></a>
### S26 — MLX, évaluation paresseuse, documentation officielle
https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html

Les expressions ne sont pas nécessairement calculées lors de leur construction. Les benchmarks doivent forcer l'évaluation effective et attendre son achèvement. Versionner les primitives exactes utilisées ; ne pas chronométrer seulement l'enqueue ou la construction du graphe.

## Références fondatrices complémentaires — non récupérées intégralement durant cette revue

Ces références cadrent des concepts établis ; elles ne fondent ici **aucun chiffre comparatif**. Les vérifier avant d'affirmer une nouveauté de publication.

<a id="s27"></a>
### S27 — Deep Sets, Zaheer et al., 2017
https://arxiv.org/abs/1703.06114

Précédent pour encodeurs d'ensembles invariants par permutation.

<a id="s28"></a>
### S28 — Relational inductive biases, deep learning, and graph networks, Battaglia et al., 2018
https://arxiv.org/abs/1806.01261

Cadre des graph networks et du biais relationnel ; pertinent pour comparer un GNN au Transformer générique.

<a id="s29"></a>
### S29 — Pointer Networks, Vinyals et al., 2015
https://arxiv.org/abs/1506.03134

Précédent des sorties pointant vers des éléments de l'entrée. La version originale est séquentielle ; utiliser une tête pointeur dans UCM ne rend pas toute l'architecture originale de Pointer Networks non autoregressive.

<a id="s30"></a>
### S30 — Policy Distillation, Rusu et al., 2015/2016
https://arxiv.org/abs/1511.06295

Précédent de compression/distillation de politiques, notamment multi-tâches.

<a id="s31"></a>
### S31 — Universal Planning Networks, Srinivas et al., 2018
https://arxiv.org/abs/1804.00645

Voisin pertinent pour objectifs, représentations et planification différentiable. Tentative de récupération bloquée par quota : référence à approfondir, pas source d'une revendication expérimentale détaillée.

<a id="s32"></a>
### S32 — Using Reward Machines for High-Level Task Specification and Decomposition in Reinforcement Learning, Icarte et al., ICML 2018
https://proceedings.mlr.press/v80/icarte18a.html

Précédent pour structurer explicitement tâches et progression. Tentative de récupération bloquée par quota ; utile pour la prochaine passe bibliographique.

<a id="s33"></a>
### S33 — Hindsight Experience Replay, Andrychowicz et al., 2017
https://arxiv.org/abs/1707.01495

Précédent de réétiquetage d'objectifs. Les restrictions de faisabilité et de causalité restent à contrôler pour des tâches logicielles.

## Ce que cette bibliographie ne permet pas d'affirmer

- Que personne n'a déjà combiné adaptateurs, petits contrôleurs et imitation.
- Que les modèles listés ont reçu les mêmes données ou les mêmes informations d'entrée.
- Qu'un score robotique, une action web offline et une tâche web exécutée sont des métriques interchangeables.
- Que des paramètres peu nombreux impliquent une faible latence de bout en bout.
- Que le projet UCM a déjà un résultat expérimental positif.
