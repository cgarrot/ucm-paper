# TROISIÈME REVUE EXTERNE — synthèse des résultats + stratégie (reçue 23/09/2026, ~12:15)

*Revue documentaire des rapports V0/V1, de la veille web (REPORT.md) et de la revue tagi-review (croisement veille × revue). Pas une reproduction du dépôt. Source : modèle externe sollicité par cgarrot. Verbatim, archivé pour l'équipe.*

---

**Oui, vous avez une base crédible pour construire un modèle de contrôle rapide, utile, et complémentaire des autres modèles. Mais je modifierais la trajectoire actuelle : la priorité ne devrait plus être seulement de faire réussir le transfert TinyGraphKey → SIW, ni d'ajouter davantage de calcul au GNN.**

La cible recommandée :

> **Un modèle compact qui apprend à exécuter des objectifs dans différents environnements, utilise les conséquences de ses actions pour s'adapter, et ne sollicite un gros modèle que lorsque cela apporte réellement quelque chose.**

Cela conserve l'ambition. Cela évite aussi de confondre trois choses : **avoir entraîné un nouveau modèle**, **avoir inventé une nouvelle architecture**, et **avoir démontré une capacité réellement nouvelle**.

## 1. Ce que vous avez vraiment construit

### La V0 est une base expérimentale intéressante

B144 : GNN 694 513 paramètres, 97,50 % sur la cellule principale, 99,35 % sur la composition réservée. Latence CPU M5 : 0,52–0,66 ms p95 (calme), 1,04 ms médiane p95 (soutenue intercalée).

| Résultat documenté | Ce qu'il permet de conclure |
| --- | --- |
| Réussite sur de nouveaux layouts | Généralise à de nouvelles instances de TinyGraphKey. |
| Réussite de la composition réservée | Réussit **cette recombinaison précise**, pas encore une diversité ouverte de nouvelles tâches. |
| Supériorité sur l'heuristique locale | Valeur par rapport à cette alternative simple (97,5 % vs 82,6 %). |
| Décisions ~1 ms | Calcul local peu coûteux dans le régime mesuré. |

La démarche de diagnostic (échec d'A → cause de représentation, pas « plus gros ») est particulièrement bonne. Réserves : environnement symbolique/déterministe/pleinement observable ; audits internes, pas réplication externe.

### La V1 ne démontre pas encore le transfert recherché

| À k=500 nominal | Arrêt correct | goal_ever |
| --- | ---: | ---: |
| Scratch | 28,87 % | ≈ 72,9 % |
| Pré-entraînement TGK | 41,43 % | ≈ 72,6 % |
| Contrôle | 42,23 % | ≈ 71,1 % |

**Le pré-entraînement améliore la terminaison, pas l'atteinte du but.** Vraie composante de contrôle, mais pas la preuve d'un transfert de stratégie multi-étapes. Support initial-only + contrôle apprenant la validité → le HOLD était juste. Conclusion correcte : **ni confirmé, ni réfuté**.

Test de permutation recalculé : **p = 0,09375 unilatéral, 0,1875 bilatéral** — confirme le diagnostic post-hoc, ne remplace pas l'analyse préenregistrée.

Budgets nominaux ≠ réels : k=500 → 376 couples, k=10000 → 1191. Les courbes futures doivent montrer les quantités réelles.

**Bilan : bon contrôleur compact dans une famille ; PAS encore le contrôleur partagé/adaptable visé.**

## 2. Sur la nouveauté : les deux revues sont encore trop optimistes

- Ståhlberg-Bonet-Geffner : bloc relationnel à poids partagés itéré, supervision V* → **ICAPS 2022** (pas KR).
- Manquent deux familles :
  - **A Generalist Neural Algorithmic Learner** (arXiv 2209.11142) : processeur GNN partagé sur tâches algorithmiques.
  - **Learning Domain-Independent Heuristics for Grounded and Lifted Planning** (AAAI 2024, arXiv 2312.11143) : heuristiques GNN indépendantes du domaine → le terrain « représentation de planification partageable » n'est PAS vide.
- **Jev** : « pas de multi-étapes » trop catégorique — Wikiracing (décisions successives vers une page cible), Doom sur état structuré.
- **cua-s1-forms** : distinction juste (scoreur d'options, ordre codé en dur) — apprendre l'ordre/dépendances est une différence pertinente.

**Conclusion : la différenciation doit être démontrée sur une propriété concrète** (adaptation à nouvelles règles avec peu d'interactions ; réduction des appels au gros modèle ; meilleure fiabilité à latence totale comparable), pas sur l'assemblage « petit + graphe + récursif + multi-mondes ».

## 3. Ce que je garderais et corrigerais dans la dernière revue (tagi-review)

**A. La profondeur est une hypothèse, pas une explication.** Nombre d'actions ≠ distance graphe ≠ itérations internes nécessaires. L'expérience manquante : **mêmes paramètres, meilleures données de trajectoire/récupération d'abord** — si ça corrige l'essentiel, on aura évité de complexifier le modèle pour une mauvaise raison.

**B. C1 (marge Q)** : le problème DFKI (régression sur seules actions du professeur) n'est pas le nôtre (classification set-valued + Q* exact pour tous). D'accord pour retirer des indispensables ; **mais annoncer un effet nécessairement nul est excessif** — expérience isolée éventuelle.

**C. C2 : récupérer, pas détecter.** En déterministe-observable, un cycle se détecte par hash — moniteur appris inutile ; transformer timeout en STOP prématuré ne résout rien. Pièges : interdire une action parce que son état futur est connu = modèle de transition privilégié ; interdire de revisiter une pièce supprime des solutions exigeant d'y revenir avec la clé. **La mémoire doit porter sur les informations pertinentes, pas seulement la position.**

**D. C3 : trois expériences distinctes** : (1) changer le but de manière cohérente → le comportement suit le nouveau but, on évalue le nouveau but ; (2) renommage sans changement sémantique → invariance ; (3) détruire l'information de but → dépendance. « Battre le hasard après permutation = fuite » est trop fort (préconditions, buts qui se recouvrent, structure du monde).

**E. C5 : DSL oui, réécriture massive non.** Quelques primitives + tests différentiels contre les env existants. **TGK et SIW réécrits ne redeviennent pas « jamais vus »** — tests de régression ; réserver d'autres familles pour la confirmation.

**F. Comparateur planificateur indispensable mais équitable** : distinguer l'oracle à vraies transitions / le planificateur avec informations réellement disponibles / un planificateur avec modèle de transitions appris des mêmes observations qu'UCM. Battre un planificateur aux règles par défaut erronées n'établit pas une supériorité générale.

## 4. Le modèle que je construirais

```
Demande humaine → Modèle de compréhension (objectif, contraintes, infos utiles)
→ État observable + actions possibles + mémoire d'interaction
→ UCM (agir / poursuivre le calcul / demander de l'aide)
→ Exécuteur autorisé → observation du résultat → mémoire → décision suivante
```

- **Le gros modèle comprend ; UCM apprend à exécuter.** Si le gros modèle fournit toute la séquence ou si l'adaptateur code chaque dépendance, le contrôle a quitté UCM — rendre visible la part réellement apprise.
- **Vocabulaire de représentation partagé** : descriptions composables (entités, relations, attributs, signatures d'action, arguments, indices d'effets). Distinguer fourni / inféré / observé / inconnu. Le bras « vocabulaire aligné » diagnostique, mais un alignement écrit main = information de l'ingénieur, pas preuve d'adaptation autonome.
- **Cœur compact à calcul ajustable** : B144 référence + bloc partagé récurrent. D'abord la courbe réussite/latence à budgets fixes, l'arrêt adaptatif ensuite. Prudence PTRM : gains sans réentraînement sur modèle récursif DÉJÀ entraîné avec tête de correction — pas la preuve que bruit + B144 donnera pareil ; K trajectoires multiplient une partie du coût.
- **Mémoire d'interaction ≠ calcul interne** : « cette commande ouvre un dialogue », « cette action n'a eu aucun effet ». Ne pas l'ajouter pour un score artificiel — créer une expérience où elle est NÉCESSAIRE et mesurer sa valeur.
- **Décisions vérifiables + recours explicite** : agir / déclarer terminé / demander de l'aide. L'abstention n'est pas un STOP réussi. Une distance prédite faible n'est pas une probabilité calibrée ; une action qui ne diminue pas d* peut être un détour récupérable. Usages de la tête auxiliaire évalués séparément.

## 5. L'expérience qui manque le plus : apprendre de nouvelles règles sans réentraîner

Deux interfaces identiques, effets de boutons inversés. Sans indice : non identifiable (limite de la spec elle-même). Avec courte démonstration / description partielle / exploration sans danger :

> **Après quelques observations des conséquences de ses actions, le même modèle peut-il réussir de nouveaux objectifs dans ce monde, sans modification des poids ?**

Bras : historique correct / sans historique / historique mélangé ; planificateur avec les mêmes informations. Mesurer : interactions nécessaires, réussite, erreurs irréversibles, coût de calcul. Pas une invention sans précédent — mais bien plus directement alignée avec l'ambition que les transferts entre vocabulaires réinitialisés.

## 6. Ordre recommandé

1. **Clore V1-bis proprement** (campagne ciblée dimensionnée par pilote ; budgets réels ; séparation progression/terminaison ; raw conservés). **Un échec limiterait une hypothèse précise dans ce régime — PAS le transfert par poids en général** (la conclusion Pivot B de la revue précédente est trop générale).
2. **Séparer le problème de données du problème de calcul** :

| Variante | Question isolée |
| --- | --- |
| B144 actuel | Référence |
| B144 + données couvrant erreurs et récupération | Le problème vient-il des états rencontrés en exécution ? |
| Bloc partagé récurrent, même supervision | Davantage de calcul aide-t-il ? |
| Récurrent + supervision intermédiaire | Gain supplémentaire de cette supervision ? |

   Tête Q en ablation distincte. DAgger = piste établie, alternative à tester.
3. **Mondes composés + adaptation par contexte** : DSL minimal ; mesurer séparément nouvelles instances / nouvelles compositions / règles à identifier. Multi-source vs mono-source à budgets comparables.
4. **Petite démo logicielle de bout en bout MAINTENANT** (formulaire conditionnel, dialogue, export+vérification), exploratoire déclarée. Comparaison décisive : **même tâche/informations/autorisations : gros-modèle-à-chaque-décision vs gros-modèle+UCM**. Résultat intéressant : la tâche reste fiable avec moins d'appels/temps/coût.

## 7. Corrections méthodologiques sans ralentir

- **Lecture unique ≠ absence de contamination** : protéger les décisions adaptatives, l'accès aux résultats, le gel des analyses ; engagements actuels respectés ou amendés PROSPECTIVEMENT. Ne pas minimiser les incidents (de vraies évaluations sur scellé ont eu lieu).
- **La couverture exhaustive n'est pas une condition générale d'apprentissage** : un défaut de support invalide certains estimateurs (pondération d'importance), pas automatiquement la généralisation. Éviter états impossibles + absence de catégories essentielles (états post-action) ; certaines règles de blocage mélangent les notions.
- **10 seeds ne garantissent pas la puissance** ; dépend de variabilité et gain minimal. Les 4 prédicats = strates intra-seed, PAS réplications indépendantes.
- **Incohérence chiffrée à réconcilier** : 83 échecs / 3 160 épisodes = 97,37 %, pas 97,50 % — relier les deux chiffres dans le rapport.

## Verdict final

Poursuivre ; ne pas repartir de zéro ; ne pas réduire à un classificateur d'actions. Déplacer le centre de gravité :

- **Aujourd'hui** : petit réseau apprend une politique efficace dans un monde structuré.
- **Prochaine capacité** : un même contrôleur exploite des représentations partagées, **apprend les conséquences de ses actions**, adapte son effort de calcul.
- **Validation pratique** : avec un modèle de compréhension, exécute réellement des tâches avec meilleur compromis fiabilité/coût/latence.

2-3 M de paramètres acceptables ; compacité = contrainte mesurée, pas définition de la réussite. **Le potentiel : une couche d'exécution partageable entre environnements, qui utilise les gros modèles sans en dépendre à chaque action.**

### Références citées
[1] arXiv 2109.10129 (Ståhlberg-Bonet-Geffner, ICAPS 2022) · [2] arXiv 2209.11142 (Generalist Neural Algorithmic Learner) · [3] arXiv 2312.11143 (AAAI 2024, heuristiques indépendantes du domaine) · [4] cua-ai/cua-s1-forms (HF) · [5] TypeSafe/Jev blog · [6] arXiv 2603.17544 (DFKI Q+marge) · [7] arXiv 2605.19943 (PTRM) · [8] Ross et al. 2011 (DAgger)
