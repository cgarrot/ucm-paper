# UCM — Judge & Constraint Layer

## 0. Statut et intention

**Date de création :** 23 septembre 2026  
**Statut :** document de référence et feuille de route conceptuelle — **non normatif**  
**Objet :** expliquer LeJudge/Jev comme éventuelle couche de jugement, de contraintes et de vérification autour d'UCM  
**Ne fait pas :** modifier le modèle UCM, autoriser un run, modifier le protocole V1-bis ou rendre une intégration LeJudge obligatoire.

Ce document est destiné à servir de **future documentation de conception et d'intégration**. Il peut évoluer lorsque V1-bis sera stabilisé et lorsqu'un bridge expérimental aura été validé sur des fixtures de développement.

Documents de référence du projet :

- [`docs/VISION.md`](VISION.md) — motivation produit et séparation Jev/UCM ;
- [`universal-control-model-project-spec.md`](../universal-control-model-project-spec.md) — spec scientifique ;
- [`docs/PROTOCOL-V1BIS-v2-2026-09-23.md`](PROTOCOL-V1BIS-v2-2026-09-23.md) — protocole de transfert ;
- [`docs/DECISION-V1-CONSTRUCT-HOLD.md`](DECISION-V1-CONSTRUCT-HOLD.md) — HOLD actuel ;
- [LeJudge — repository](https://github.com/AbdelStark/lejudge-jev-jepa) ;
- [LeJudge — paper PDF](https://github.com/AbdelStark/lejudge-jev-jepa/blob/main/paper/LeJudge-natural-language-constraints-for-latent-world-model-planning.pdf).

---

## 1. Résumé exécutif

LeJudge est un système qui ajoute des **contraintes en langage naturel à un planner latent** :

```text
world model
    → imagine des séquences d'actions
    → probes décrivent les futurs en mots
    → Jev juge les contraintes sur ces descriptions
    → le code convertit les probabilités en coût
    → le planner sélectionne un plan
```

UCM n'est pas un planner latent. UCM est une **petite politique conditionnée par un but**, qui choisit une action structurée à partir de l'état observable :

```text
état structuré + objectif + candidats → scores d'actions → action
```

La relation recommandée est donc :

```text
Jev / gros modèle : comprendre, décomposer, occasionnellement juger
Planner/world model : anticiper et comparer des plans
UCM : exécuter rapidement les actions locales
Constraint judge : vérifier les règles et les postconditions
```

**Décision de conception :** LeJudge/Jev ne doit pas être intégré au cœur de `ucm/model` à ce stade. Il doit rester une couche externe, optionnelle et indépendamment testable. Une intégration future doit passer par une interface de contraintes explicite, sans ajouter Jev aux entrées du policy UCM.

---

## 2. Qu'est-ce que LeJudge ?

### 2.1 Le problème traité

Un world model comme LeWorldModel optimise principalement une distance entre un état final imaginaire et un état objectif :

```text
coût = || état_final_imaginaire - objectif ||
```

Cette distance ne peut pas exprimer directement :

- « ne passe pas par là » ;
- « ne touche pas cet élément avant l'étape 3 » ;
- « garde l'objet droit » ;
- « sois doux » ;
- « préfère l'action réversible ».

LeJudge ajoute un second terme de coût :

```text
coût_total = coût_objectif + λ × pénalité_contraintes
```

### 2.2 Le pipeline LeJudge

#### Imagine

LeWM/CEM génère plusieurs séquences d'actions candidates dans un espace latent.

Dans le protocole PushT publié :

- 300 candidats ;
- 30 itérations CEM ;
- plusieurs étapes d'horizon ;
- une image objectif encodée en latent.

#### Describe

Des probes — principalement des modèles linéaires dans le MVP — transforment les latents en symboles et en mots d'un vocabulaire fermé.

Par exemple :

```text
block = centre-left
block_edge = none
block_angle = upright
agent = bottom-centre
contact = false
block_speed = slow
```

Jev ne reçoit pas les coordonnées brutes. Il reçoit une description symbolique et contrôlée.

#### Judge

Jev répond à des questions typées, par exemple :

```text
At step t of candidate k, does the block or agent violate constraint c?
```

Les réponses sont des `Noul`/probabilités ou des scores. Jev ne fait pas l'arithmétique finale et ne génère pas de texte de contrôle.

#### Decide

Le code Python agrège les réponses :

| Famille | Agrégation indicative |
|---|---|
| `never` | `max_t p(t)` |
| `always` | `1 - min_t p(t)` |
| `soft` | pénalité espérée à partir d'un score |
| `temporal_before` | comparaison de premiers événements |

Le coût résultant est ensuite minimisé par CEM.

### 2.3 Ce que LeJudge démontre

Dans le README et le paper publiés, LeJudge rapporte notamment :

- une reproduction du planner **LeWM non contraint** sur PushT autour de 86 % avec trois frames de contexte et 90 % avec une frame — ce n'est pas un résultat de LeJudge sous contraintes ;
- une exactitude Jev d'environ 86–88 % sur des faits symboliques dans leur étude ;
- un AUROC d'environ 0,94 sur le judge-only study ;
- des performances plus faibles sur certaines paraphrases et surtout sur les near-miss negatives, avec environ 38–53 % de faux positifs selon le setting ;
- environ 5,5 appels Jev et 31k tokens d'entrée par épisode dans le protocole publié **avec memoïsation et une version Jev épinglée** ; ce n'est pas un coût universel ;
- des résultats nuls dans une première étude, principalement parce que les contraintes étaient souvent incompatibles avec l'objectif ou la trajectoire disponible ;
- un drift du world model dominant dans certaines analyses de l'erreur restante, mais pas nécessairement dans toute expérience.

Ces chiffres ne doivent pas être transposés tels quels à SIW, TinyGraphKey ou au web. Ils servent à comprendre le type de système et ses limites. Le protocole public utilise notamment une version Jev épinglée à `jev-1.13.0`, un cache et une population d'évaluation spécifique ; ce ne sont pas des propriétés universelles de Jev ou de LeJudge.

### 2.4 Distinction importante : LeJudge publié vs proposition UCM

LeJudge publié est un **coût de planification** pour un planner latent. Dans son implémentation-native, il ne fournit pas directement :

- un statut `ALLOW/BLOCK/ASK` ;
- un vérificateur de postconditions UI ;
- un filtre d'actions UCM ;
- le schéma `UCMFacts` ;
- une autorisation d'action irréversible.

Les sections qui parlent de `BLOCK`, `ASK`, `ESCALATE`, de permissions et de postconditions décrivent une **couche UCM future**, à construire comme adaptateur. Elles ne doivent pas être interprétées comme une fonctionnalité déjà présente dans LeJudge.

---

## 3. Relation avec UCM

### 3.1 Rôles différents

| Composant | Rôle dans la vision UCM | Rôle possible avec LeJudge |
|---|---|---|
| Muse | Transcription de la voix | Peut fournir une demande textuelle avant planification |
| Jev | Comprendre et décomposer | Peut juger des contraintes typées sur des faits symboliques |
| UCM | Exécuteur local d'actions | Reste le fast path ; ne devient pas le juge global |
| LeWM / planner | Anticipation de futurs | Peut proposer des plans ou des trajectoires candidates |
| LeJudge | Coût de contraintes | Peut filtrer ou réordonner des plans |
| Oracle UCM | Vérification déterministe | Reste la référence de sécurité et de mesure |
| Vérificateur postcondition | Contrôle après action | Peut confirmer qu'une action a produit l'effet attendu |

### 3.2 Ce que LeJudge ne fait pas

LeJudge ne remplace pas UCM pour les raisons suivantes :

1. LeJudge est construit pour PushT et un world model continu, pas pour les actions typées de SIW ;
2. LeJudge classe ou réordonne des séquences imaginées, alors qu'UCM scorerait des actions locales ;
3. UCM n'a pas de modèle de transition latent nécessaire à CEM ;
4. ajouter Jev dans `policy_input` modifierait le modèle, ses entrées et le protocole scientifique ;
5. un juge approximatif ne doit pas devenir l'unique autorité pour une action irréversible.

### 3.3 Le bon découpage

```text
┌──────────────────────────────────────────────┐
│ Jev / modèle de raisonnement                 │
│ comprendre, décomposer, poser des questions   │
└──────────────────────┬───────────────────────┘
                       │ objectif + contraintes
                       ▼
┌──────────────────────────────────────────────┐
│ Planner optionnel                            │
│ LeWM, CEM, modèle symbolique ou oracle      │
└──────────────────────┬───────────────────────┘
                       │ plan / hypothèses
                       ▼
┌──────────────────────────────────────────────┐
│ UCM                                          │
│ policy state + goal → action                 │
└──────────────────────┬───────────────────────┘
                       │ action + résultat
                       ▼
┌──────────────────────────────────────────────┐
│ Judge / verifier                             │
│ contraintes, postconditions, risque          │
└──────────────────────┬───────────────────────┘
                       │
       exécuter / retry / stop / escalade
```

---

## 4. Catalogue des utilisations possibles

Cette section recense les utilisations envisageables. Toutes ne sont pas recommandées au même niveau.

### 4.1 Utilisations de planification

#### U1. Gate de contraintes avant génération du plan

Le judge reçoit l'objectif, les contraintes et l'état initial. Il signale :

- une contrainte impossible ;
- une contradiction entre objectif et règle ;
- une contrainte non exprimable dans le vocabulaire ;
- un besoin de clarification.

**Exemple :** ne pas lancer un plan « soumettre le formulaire » si les champs obligatoires ne sont pas visibles ou remplis.

**Avantage :** évite de lancer un planner sur une tâche impossible.  
**Risque :** faux négatif si le judge accepte une contradiction.  
**Exigence :** pré-vérification déterministe avant le judge.

#### U2. Réordonnancement de plans candidats

Le planner produit plusieurs plans. Le judge réordonne les plans selon :

- violations de contraintes ;
- préférences ;
- risque ;
- coût d'exécution.

C'est l'usage le plus proche de LeJudge.

**Exemple :** deux plans atteignent le même goal, mais l'un clique sur `DELETE` et l'autre sur `ARCHIVE`.

#### U3. Ajout d'une contrainte au coût du planner

Le judge transforme une phrase en pénalité. Le planner minimise alors :

```text
coût_objectif + λ1 × risque + λ2 × violations + λ3 × coût_action
```

Cette intégration doit rester dans le planner, pas dans les poids d'UCM.

#### U4. Planification avec plusieurs horizons

Le judge peut être appelé :

- une fois au début de la tâche ;
- à chaque replan ;
- uniquement quand l'état change significativement ;
- uniquement pour les zones à risque.

Cette stratégie est préférable à un appel Jev à chaque décision UCM.

#### U5. Vérifier la faisabilité avant l'exécution

Le judge peut refuser un plan même si le planner est confiant. Il faut distinguer :

```text
plan valide selon l'oracle
plan impossible selon Jev
plan incertain selon Jev
```

#### U6. Sélection d'un plan de récupération

Après un échec, plusieurs plans de correction sont proposés :

- recharger l'écran ;
- revenir en arrière ;
- demander une confirmation ;
- changer de vue ;
- escalader.

Le judge peut sélectionner le plan de récupération le moins risqué.

---

### 4.2 Utilisations autour de l'execution UCM

#### U7. Pre-action gate

Avant d'exécuter une action, le judge vérifie la contrainte la plus importante.

```text
état observable + action candidate + contraintes
    → ALLOW / BLOCK / ASK / UNKNOWN
```

C'est l'intégration la plus simple à tester sur SIW.

#### U8. Post-action verifier

Après l'action, le système vérifie la postcondition :

```text
action SUBMIT
    → le formulaire a-t-il réellement changé ?
    → une erreur est-elle apparue ?
    → le goal est-il atteint ?
```

Le juge ne doit pas considérer « l'action a été envoyée » comme preuve de réussite.

#### U9. Filtre d'actions risquées

Le juge peut interdire ou confirmer :

- suppression ;
- paiement ;
- envoi ;
- permission ;
- fermeture de session ;
- changement de configuration ;
- action irréversible.

Dans UCM, les filtres déterministes doivent précéder le juge pour les actions critiques.

#### U10. Confirmation humaine

Le judge peut transformer une incertitude en demande de confirmation :

```text
« Cette action supprimera définitivement 3 éléments. Confirmer ? »
```

C'est une forme d'escalade, pas une preuve de sécurité.

#### U11. Détection de boucle

Un juge ou un moniteur dédié peut détecter :

- répétition de la même action ;
- oscillation entre deux vues ;
- timeout local ;
- absence de progrès observable.

Le terme `progress` doit être calculé à partir de faits autorisés, jamais d'un signal oracle interdit dans `policy_input`.

#### U12. Replan après violation

Si une violation est détectée après une action :

```text
pause
 → état réel
 → diagnostic
 → nouveau plan
 → UCM reprend
```

Cette boucle est importante pour une future version avec récupération.

---

### 4.3 Utilisations de sécurité et de confiance

#### U13. Abstention explicite

Le système doit distinguer :

```text
PASS
FAIL
UNKNOWN
NOT_APPLICABLE
```

`UNKNOWN` ne doit pas être converti en `PASS`.

#### U14. Escalade vers Jev

Lorsque UCM est incertain :

```text
UCM : « je ne sais pas »
    ↓
Jev : demande de clarification ou nouveau plan
    ↓
UCM : reprend l'exécution
```

#### U15. Escalade vers un humain

Pour les actions :

- irréversibles ;
- financières ;
- sensibles ;
- ambigües ;
- hors permissions.

#### U16. Analyse risque–couverture

Mesurer le compromis :

```text
couverture = fraction des tâches traitées automatiquement
risque = violations sur les tâches traitées
```

Un système peut être très prudent mais peu utile, ou très utile mais dangereux. Il faut publier ces deux axes.

#### U17. Judge de contrainte comme garde-fou secondaire

Le juge peut bloquer une action seulement si :

- un check déterministe a déjà échoué ;
- ou si le risque dépasse un seuil pré-enregistré.

Il ne faut pas laisser un juge probabiliste unique architecturer toutes les actions.

---

### 4.4 Utilisations dans l'apprentissage

#### U18. Filtrage de données d'adaptation

Un juge peut filtrer les couples :

```text
couple candidat
    → contrainte satisfaite ?
    → conserver / corriger / supprimer
```

Cela peut réduire les labels invalides ou les trajectoires qui violent une règle métier.

**Attention :** le judge ne doit pas devenir une source de vérité cachée. L'oracle et les règles déterministes doivent rester auditables.

#### U19. DAgger / aggregation de trajectoires

Pendant une rollout :

- UCM visite un état ;
- le judge signale une violation ;
- un expert ou oracle produit une correction ;
- le couple est ajouté aux données d'adaptation.

Le judge peut donc servir de déclencheur de collecte, pas seulement de sanction.

#### U20. Génération de curriculum

Classer les états par :

- fréquence ;
- violation ;
- risque ;
- ambiguïté ;
- profondeur ;
- coût d'adaptation.

Le judge peut aider à sélectionner les exemples qui couvrent des situations/frontières plutôt que seulement des cas fréquents.

#### U21. Enseignant hors ligne

Jev peut être utilisé hors ligne pour :

- annoter des préférences ;
- classer des actions ;
- produire des scores de risque ;
- filtrer des plans d'experts.

C'est acceptable uniquement si :

- les réponses sont cachées et versionnées ;
- l'annotation est séparée du chemin déterministe ;
- l'effet du judge hors ligne est mesuré par rapport à un oracle ou un checker.

#### U22. Distillation dans UCM

Les scores du juge peuvent être transformés en :

- labels binaires ;
- pénalités ;
- features de risque ;
- targets d'abstention.

Mais cette distillation est une nouvelle expérience. Elle ne doit pas être confondue avec le transfert TGK → SIW.

#### U23. Contrefactuels

Pour une même observation, comparer :

```text
action A sous contrainte c1
action B sous contrainte c1
```

Le judge peut produire un signal de préférence ou de risque, mais la validité doit être vérifiée avec un oracle sur les états dangereux.

#### U24. Validation de couverture

Évaluer si les données d'adaptation couvrent les contraintes et les états nécessaires :

- référents présents ;
- predicates visibles ;
- relations nécessaires ;
- préconditions ;
- postconditions.

Cela rejoint les audits de distribution UCM.

---

### 4.5 Utilisations dans l'évaluation et la recherche

#### U25. Judge-only benchmark

Mesurer le juge seul sur des faits étiquetés :

- exactitude ;
- AUROC ;
- ECE ;
- paraphrase ;
- near-miss ;
- calibration par contrainte.

#### U26. Oracle-on-facts

Utiliser un checker parfait sur les mêmes entrées que le judge pour obtenir une borne supérieure. Cela permet de distinguer :

```text
erreur du juge
erreur du coût
erreur du planner
erreur du world model
erreur de l'environnement
```

#### U27. Attribution d'erreur

Construire une taxonomie :

| Erreur | Exemple |
|---|---|
| `bad_facts` | le probe a décrit un mauvais état |
| `bad_referent` | la contrainte vise le mauvais objet |
| `judge_false_positive` | violation signalée à tort |
| `judge_false_negative` | violation manquée |
| `bad_aggregation` | max/score mal calculé |
| `bad_cost` | contrainte sans effet sur le plan |
| `planner_drift` | le futur imaginé n'est pas le futur réel |
| `execution_failure` | le plan est correct mais l'exécution échoue |
| `unknown_handled_as_pass` | une incertitude traitée comme conformité |

#### U28. Test metamorphique

Vérifier que le résultat est stable sous :

- paraphrase de contrainte ;
- permutation des IDs ;
- réordonnancement des candidats ;
- changement de granularité du vocabulaire ;
- ajout d'une contrainte sans effet ;
- duplication d'une contrainte équivalente.

#### U29. Étude de latence

Mesurer séparément :

- Parsing des faits ;
- appel Jev ;
- cache hit/miss ;
- agrégation ;
- décision UCM ;
- vérification postcondition ;
- escalation.

#### U30. Étude de coût

Comparer :

```text
UCM seul
UCM + oracle local
UCM + checker déterministe
UCM + Jev
UCM + Jev + fallback
```

sur le coût total par tâche réussie, pas seulement par appel.

#### U31. World-model drift

Si un planner latent est utilisé, mesurer :

- violation imaginée ;
- violation réellement exécutée ;
- drift à chaque horizon ;
- taux de plans invalides après exécution.

C'est l'erreur dominante observée dans LeJudge.

#### U32. Drift de contraintes

Mesurer la dégradation lorsque :

- le vocabulaire change ;
- les entities changent ;
- les contraintes deviennent plus complexes ;
- le modèle Jev change de version ;
- l'interface introduit de nouveaux labels.

---

### 4.6 Utilisations produit

#### U33. Préférences utilisateur

Transformer des préférences en contraintes de coût :

```text
« privilégie l'action réversible »
« évite les dialogues si une alternative existe »
« utilise la vue la plus courte »
```

Ces préférences ne doivent pas être mélangées aux préférences de sécurité.

#### U34. Accessibilité

Exemples :

- ne pas cliquer sur un élément caché ;
- conserver le focus ;
- respecter l'ordre de lecture ;
- éviter une action qui rend le statut inaccessible.

Le juge peut aider à détecter ces risks, mais un audit structurel doit rester déterministe.

#### U35. Permissions

Le juge peut proposer :

```text
ALLOW si permission explicite
BLOCK si permission absente
ASK si permission ambiguë
```

La permission ne doit pas être déduite d'un simple label.

#### U36. Replan ciblé après erreur

Quand le résultat est mauvais :

```text
Jev diagnostique
UCM inspecte
judge vérifie les options de récupération
UCM ré-exécute
```

C'est la voie naturelle vers la « récupération » mentionnée dans `VISION.md`.

#### U37. Multi-objectif

Le judge peut aider à arbitrer :

- goal principal ;
- sécurité ;
- coût ;
- preferences ;
- latence ;
- impact utilisateur.

Les poids et hiérarchies doivent être explicites.

#### U38. Planification multi-agents

À terme, plusieurs rôles pourraient utiliser le même contrat :

- planner ;
- executor UCM ;
- judge ;
- human reviewer.

Le contrat doit permettre de tracer qui a proposé, exécuté et validé chaque décision.

---

## 5. Modes de judge possibles

### 5.1 Judge déterministe

Exemples :

- prédicats structurés ;
- expressions régulières sûres ;
- tables de règles ;
- state machine ;
- postcondition checker ;
- oracle.

**Usage recommandé :** sécurité, permissions, actions irréversibles, tests.

**Avantages :** reproductible, rapide, auditable.  
**Limites :** peu expressif, maintenance manuelle.

### 5.2 Judge keyword / grammar

Un parser simple transforme une contrainte en predicats.

**Usage recommandé :** baseline, fallback local, contraintes fermées.

**Avantage :** très rapide et déterministe.  
**Limite :** fragile aux paraphrases et aux références implicites.

### 5.3 Probe + vocabulary + Jev

C'est le motif LeJudge :

```text
représentation → symboles → mots → Jev
```

**Usage recommandé :** préférences, contraintes souples, vérification ambiguë.

**Limites :** erreurs de probe, near-miss, coût, version de Jev.

### 5.4 Judge appris

Un petit modèle peut apprendre à classer des facts ou des actions.

**Usage recommandé :** classification de risque, Fallback local, distillation.

**Limite :** nécessite labels, calibration et surveillance de drift.

### 5.5 LLM généraliste

Un LLM peut être utile pour :

- clarification ;
- génération de contraintes candidates ;
- diagnostic ;
- interface de développement.

**Usage recommandé :** hors fast path ou adjudication.  
**Non-recommandé :** autorisation unique d'une action critique sans checks déterministes.

### 5.6 Humain

Le judge humain est la référence pour :

- ambiguïté ;
- permissions ;
- risques irréversibles ;
- conflits de valeurs.

Le système doit apprendre à demander de l'aide, pas à cacher l'incertitude.

---

## 6. Contrat conceptuel commun

LeJudge, les checkers déterministes et les futures implémentations UCM devraient exposer une interface commune, sans dépendre de LeWorldModel.

**Les dataclasses ci-dessous sont un contrat conceptuel pour une future couche UCM. Ce ne sont pas les dataclasses natives de LeJudge.** Le mapping depuis le contrat natif — probabilités, confiance, tokens, cache et `failed` — devra être défini et testé dans un adaptateur séparé.

### 6.1 Contraintes

```python
@dataclass(frozen=True)
class Constraint:
    id: str
    text: str
    family: str                 # never | always | soft | temporal_before | preference
    weight: float = 1.0
    risk_level: str = "normal"  # low | normal | high | critical
    source: str = "user"        # user | policy | system | derived
    referents: tuple[str, ...] = ()
```

### 6.2 Faits

```python
@dataclass(frozen=True)
class UCMFacts:
    episode_id: str
    step: int
    view: str
    entities: tuple[str, ...]
    candidate_types: tuple[str, ...]
    target_enabled: bool | None
    required_fields_filled: bool | None
    dialog_open: bool | None
    submitted: bool | None
    observable_error: str | None
    risk_tags: tuple[str, ...] = ()
```

Ces champs ne doivent pas inclure :

- `d_star` ;
- plan oracle ;
- succès futur ;
- reward ;
- gradient ;
- réponse humaine non autorisée ;
- secrets ou credentials.

### 6.3 Question/judgment

```python
@dataclass(frozen=True)
class JudgeResult:
    constraint_id: str
    status: str                 # PASS | FAIL | UNKNOWN | NOT_APPLICABLE
    score: float | None
    confidence: float | None
    reason_codes: tuple[str, ...]
    source: str                 # oracle | keyword | jev | learned | human
    model_version: str | None
    cache_hit: bool
```

### 6.4 Décision finale

```python
@dataclass(frozen=True)
class ActionDecision:
    action: str
    disposition: str            # ALLOW | BLOCK | ASK | RETRY | ESCALATE
    reason_codes: tuple[str, ...]
    judge_results: tuple[JudgeResult, ...]
    deterministic_checks: tuple[str, ...]
```

Le `disposition` doit être calculé par une politique versionnée. Le judge ne doit pas décider implicitement de l'action finale.

---

## 7. Agrégation et sémantique de l'incertitude

### 7.1 Règle importante

`UNKNOWN` n'est pas `PASS`.

Pour une contrainte critique :

```text
UNKNOWN → BLOCK ou ASK
```

Pour une contrainte soft :

```text
UNKNOWN → coût pessimiste ou ESCALATE
```

Ne pas utiliser la convention « erreur du judge = pénalité nulle » dans un système de sécurité.

**Important :** cette règle est une exigence de l'adaptateur UCM futur, pas une propriété automatique de LeJudge publié. Le chemin natif LeJudge possède notamment un état `failed` et des chemins de confidence gate qui peuvent traiter un candidat comme held ou sans pénalité. Un adaptateur UCM doit convertir ces cas en `UNKNOWN` et appliquer sa propre politique fail-closed.

### 7.2 Agrégation proposée

| Situation | Décision possible |
|---|---|
| toutes les contraintes passent | `ALLOW` si les checks déterministes passent aussi |
| une contrainte critique échoue | `BLOCK` |
| une contrainte critique est inconnue | `ASK` ou `ESCALATE` |
| une contrainte soft échoue | pénalité ou autre candidat |
| une contrainte soft est inconnue | coût pessimiste ou plan alternatif |
| aucune action sûre | `ESCALATE` |

### 7.3 Gate de confiance

Un juge probabiliste peut être utilisé avec :

- seuil de confiance ;
- marge de doute ;
- pénalité pessimiste ;
- classe `UNKNOWN` ;
- demande de vérification déterministe.

Une simple porte « incertain = zéro pénalité » doit être interdite dans le mode safety-critical.

---

## 8. Options d'intégration

### Option 0 — dépôt ou expérience séparés

```text
UCM inchangé
LeJudge séparé
adaptateur futur dans un autre repo ou worktree
```

**Avantage :** zéro risque pour V1-bis.  
**Usage :** recommandé immédiatement.

### Option 1 — sidecar optionnel

Le dépôt principal expose un protocole, mais aucun modèle n'importe LeJudge :

```text
ucm/constraints/
    schema.py
    protocol.py

adapters/
    lejudge/
        adapter.py
```

**Avantage :** intégration progressive.  
**Exigence :** dépendance optionnelle et tests sans réseau.

### Option 2 — couche de planification externe

```text
UCM core
Jev décomposition
Planner externe
LeWM/LeJudge verifier
```

**Usage :** système complet, pas modèle UCM.

### Option 3 — intégration directe dans UCM

```text
policy_input + texte naturel + Jev + world model
```

**Statut :** non recommandée.

### Option 4 — distillation

Les sorties du judge deviennent des labels ou un signal d'entraînement pour UCM.

**Usage :** phase ultérieure, avec protocole et ablation dédiés.

---

## 9. Expérience DEV recommandée

Cette expérience ne doit utiliser que des fixtures SIW de développement et ne doit toucher aucun test2. Elle est une **recherche séparée** : elle ne modifie pas les quatre bras, le protocole, les seeds ou les gates de V1-bis. Le HOLD test2 et les conditions d'ouverture de `docs/PROTOCOL-V1BIS-v2-2026-09-23.md` restent inchangés.

### 9.1 Conditions

```text
A. UCM seul
B. UCM + oracle-in-loop (baseline expérimentale uniquement)
C. UCM + checker déterministe simple
D. UCM + Jev sur facts symboliques
E. UCM + Jev + fallback déterministe
F. UCM + plan/controller externe, si disponible
```

La condition B est une borne supérieure expérimentale, pas une modalité de déploiement et pas un bras V1-bis. Dans les conditions officielles, l'oracle reste hors ligne et sert uniquement à l'évaluation.

### 9.2 Contraintes SIW initiales

```text
Ne jamais soumettre avant que les champs obligatoires soient remplis.
Ne jamais cliquer sur un élément désactivé.
Ne jamais effectuer une action irréversible sans confirmation explicite.
Toujours conserver la vue du formulaire avant soumission.
Préférer l'action réversible à l'action destructive.
```

Chaque contrainte doit avoir :

- une version ;
- un oracle ;
- un checker déterministe ;
- des paraphrases ;
- des near-miss negatives ;
- un niveau de risque.

### 9.3 Mesures

- succès closed-loop ;
- `success_and_no_violation` ;
- violation rate ;
- faux positifs ;
- faux négatifs ;
- near-miss FPR ;
- escalade rate ;
- refus corrects ;
- refus à tort ;
- latence p50/p95 ;
- coût par tâche réussie ;
- nombre d'appels judge ;
- taux de cache hit ;
- drift après plusieurs actions.

### 9.4 Règles de sécurité de l'expérience

- aucun appel Jev sur un fichier test2 ;
- aucun résultat test2 utilisé pour choisir les paramètres ;
- faits et texte utilisateur dans des champs séparés ;
- `UNKNOWN` journalisé ;
- l'oracle est utilisé seulement pour l'évaluation dans les conditions ordinaires ; la condition B, explicitement baseline expérimentale, est la seule exception autorisée et ne doit jamais entrer dans V1-bis ou un déploiement ;
- toutes les décisions journalisées ;
- les paramètres du judge fixés avant l'évaluation.

---

## 10. Roadmap d'intégration

### Phase 0 — Terminer V1-bis

Priorité absolue :

- corriger le store de layouts ;
- garantir l'indépendance des cellules `k` ;
- réinitialiser correctement les modèles ;
- vérifier les checkpoints ;
- implémenter l'évaluation Stage-B ;
- publier raw, checkpoints et statistiques ;
- garder le HOLD test2.

### Phase 1 — Contrat de contraintes

Créer un protocole stable :

- schéma `Constraint` ;
- schéma `UCMFacts` ;
- schéma `JudgeResult` ;
- politique `ALLOW/BLOCK/ASK/ESCALATE` ;
- tests de sécurité ;
- baseline oracle et checker déterministe.

Aucun appel Jev externe n'est nécessaire à cette phase.

### Phase 2 — SIW DEV

Comparer, dans une expérience séparée :

- UCM seul ;
- UCM + oracle-in-loop, uniquement comme baseline supérieure ;
- UCM + checker déterministe ;
- UCM + judge abstrait mocké.

Cette phase ne crée aucun bras, seed, test2 ou gate V1-bis.

### Phase 3 — Jev réel

- réponses cacheées ;
- version épinglée ;
- judge-only study ;
- audit near-miss ;
- mesure coût/latence ;
- aucune influence sur V1-bis.

### Phase 4 — Planner externe

Tester un planner symbolique ou latent :

- sans modifier UCM ;
- avec réordonnancement de plans ;
- avec vérification post-exécution ;
- avec budget et horizon déclarés.

### Phase 5 — Web sandbox

Uniquement après validation :

- DOM/accessibility tree → facts ;
- Jev pour décomposition ;
- UCM pour exécution ;
- judge pour permissions et risques ;
- sandbox réversible ;
- journal complet ;
- comparaison avec un LLM à chaque décision, uniquement comme comparateur hors ligne ou shadow, jamais comme composant du runtime UCM.

---

## 11. Critères de promotion

Une intégration ne doit pas être ajoutée au runtime principal tant que les conditions suivantes ne sont pas remplies :

1. le contrat fonctionne avec un checker déterministe ;
2. le mock judge et l'oracle sont testés ;
3. `UNKNOWN` ne peut pas devenir `PASS` pour une action critique ;
4. le judge ne reçoit pas de secrets ou de supervision interdite ;
5. les near-miss sont mesurés ;
6. la latence et le coût sont publiés ;
7. dans une future étude de constraints, l'effet du judge est supérieur à UCM seul sur un endpoint pré-enregistré et distinct du protocole V1-bis ;
8. la dégradation de `success` est mesurée ;
9. le cache et la version du juge sont reproductibles ;
10. le mode offline est disponible ;
11. le système peut fonctionner sans réseau pour les contrôles déterministes ;
12. l'intégration ne modifie pas le protocole V1-bis.

---

## 12. Erreurs à éviter

### Ne pas faire

- mettre le texte naturel directement dans `policy_input` UCM ;
- appeler Jev avant chaque action par défaut ;
- traiter une erreur réseau comme une contrainte satisfaite ;
- remplacer l'oracle par un juge probabiliste ;
- laisser une paraphrase near-miss déclencher une action irréversible ;
- mesurer uniquement le taux de succès ;
- cacher le coût Jev dans la latence « UCM » ;
- modifier le modèle UCM avant d'avoir un problème de contrôle précis ;
- fusionner les résultats LeJudge et UCM dans un même endpoint primaire ;
- utiliser les données test2 pour développer les règles.

### Préférer

- des faits séparés du texte propriétaire ;
- des sorties typées ;
- des statuts explicites ;
- un oracle et un checker déterministe ;
- une escalade explicite ;
- un cache versionné ;
- des rapports raw ;
- des erreurs attribuées à une couche ;
- des décisions fail-closed pour le risque critique.

---

## 13. Questions ouvertes

- Quelle est la bonne granularité des faits pour SIW ?
- Faut-il autoriser des nombres dans un checker déterministe tout en les refusant à Jev ?
- Quelle famille de contraintes est pertinente pour le web ?
- Quand une préférence doit-elle devenir une contrainte dure ?
- Comment mesurer la calibration de Jev sur des faits UCM ?
- Le judge doit-il voir les labels UI ou seulement des références typées ?
- Quelle stratégie de cache est valide pour des contraintes utilisateur variables ?
- Comment gérer les contraintes composées ?
- Comment Versionner les relations entre contraintes et permissions ?
- Quand le planner externe devient-il nécessaire plutôt qu'optionnel ?
- Comment mesurer le coût total d'une escalade Jev ?
- Quel fallback déterministe est acceptable pour chaque niveau de risque ?

---

## 14. Décision de cadrage

Pour le projet actuel :

```text
UCM core
    = inchangé

V1-bis
    = uniquement transfert et instrumentation UCM

LeJudge/Jev
    = futur constraint/judge layer

LeWM/CEM
    = futur planner optionnel

Web/DOM
    = futur adaptateur
```

La phrase de positionnement recommandée est :

> **UCM est un exécuteur compact et rapide pour des actions structurées. Un juge externe peut ajouter des contraintes, de la vérification et de l'escalade sans être appelé à chaque décision. LeJudge fournit une implémentation de référence et une méthodologie, pas un remplacement d'UCM.**

---

## 15. Références

- [LeJudge — repository GitHub](https://github.com/AbdelStark/lejudge-jev-jepa)
- [LeJudge — paper PDF](https://github.com/AbdelStark/lejudge-jev-jepa/blob/main/paper/LeJudge-natural-language-constraints-for-latent-world-model-planning.pdf)
- [LeJudge — RFC-0004 : JevCost et intégration planner](https://github.com/AbdelStark/lejudge-jev-jepa/blob/main/rfcs/RFC-0004-jevcost-and-planner-integration.md)
- [LeJudge — RFC-0005 : language goals](https://github.com/AbdelStark/lejudge-jev-jepa/blob/main/rfcs/RFC-0005-language-goals-retrieval.md)
- [`docs/VISION.md`](VISION.md)
- [`docs/DECISION-V1-CONSTRUCT-HOLD.md`](DECISION-V1-CONSTRUCT-HOLD.md)
- [`docs/PROTOCOL-V1BIS-v2-2026-09-23.md`](PROTOCOL-V1BIS-v2-2026-09-23.md)

*Fin du document. Toute évolution majeure doit être versionnée dans un nouveau document ou une décision append-only, sans modifier silencieusement le protocole UCM en cours.*
