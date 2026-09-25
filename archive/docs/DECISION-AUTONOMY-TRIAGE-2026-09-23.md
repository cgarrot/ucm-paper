# Annexe de décision — triage de l'axe autonomie (23 septembre 2026)

**Statut : décision de priorisation POST-V1-bis, PAS une autorisation.** Source : triage documentaire `reports/web-watch-2026/TRIAGE-2026-09-23.md` (tagi-review via tagi-ask), arbitré par le lead. Nulle modification du protocole V1-bis (`a1a8b9d`), nul test2, nul scellé, nul run. Tout ce qui suit s'applique aux étapes S2b/P2 et suivantes, **sous pré-enregistrement** : rien d'adopté sans règle déclarée avant données. Append-only ; correction = erratum nouveau chemin.

## Adopté

1. **Base conservée** : générateur+oracle+gates V1-bis restent le socle ; récupération ≤15 % comme cible indicative déclarée ; curriculum P2 en dose-réponse (jamais adaptatif sans règle préenregistrée).
2. **DAgger = diagnostic en S2b**, avec règle de promotion PRÉENREGISTRÉE avant toute collecte : passage en standard pour P1/P2 (2 tours) si **+3 pts de succès OU −30 % d'échecs-cycles vs R\*** sur DEV — règle de **décision de promotion, non-confirmatoire**, les deux métriques publiées ensemble. Conditions de validité : collecte **une fois** depuis B144, données **gelées par hash**, **volume égal** à l'arm comparée (sinon la factorielle données-vs-calcul est rompue), impasses comptées (états sans action menant au but).
3. **In-context AVANCÉ à l'intérieur de P2** (ni avant — échec prévisible sans lecture d'historique ; ni après) : bras D0-seul vs D0+D1 mixte, **fraction D1 déclarée** (ordre de grandeur ~30 %), **D1 non lu si D0 échoue**, interférence tolérée si mixte ≥ D0−2 pts (seuil de tolérance déclaré, pas un gate de succès). **Historique encodé structurellement, pas brut** : arêtes « effet observé » attachées aux nœuds candidat/genre, 1-2 observations par genre — l'apprentissage in-context à contexte long (~1K-10K pas, cf. veille) est hors de portée et hors design ; cet encodage court et typé est précisément notre différence testable.
4. **Tête de prédit d'effet** : perte auxiliaire dans P2 — entrée (s,a), sortie changement d'état typé, supervisée par le simulateur, **entraînement conjoint** (pas de pré-entraînement séparé). Pièce critique pour D1-D3, in-context, P3, Web sans oracle.
5. **Hors file** (jusqu'à nouvelle preuve préenregistrée) : RL (imitation+DAgger dominent avec oracle exact), SSL générique (labels exacts gratuits), rollouts model-based (pivot UNIQUEMENT si S3 KILL, adossé à la tête d'effet), mémoire persistante §13.1 (remplacée par l'historique en entrée, auditable correct/absent/mélangé).

## Gate nouveau — « l'oracle privilégié »

Dès que les effets sont à découvrir (in-context, D1-D3), l'oracle exact sait ce que le modèle ne peut pas savoir : l'imitation enseignerait l'impossible (imitation gap) et le succès normalisé à l'oracle attribuerait à la politique des échecs d'**identifiabilité**. **Parade préenregistrée pour chaque étage d'autonomie** : (a) recalculer l'oracle **à information égale** par BFS sur l'état de croyance, dont l'**énumération des hypothèses d'effets est préenregistrée** (règle de génération déclarée avant tout entraînement) ; (b) vérifier l'identifiabilité **avant** tout entraînement, avec **ordre prédit à pré-enregistrer** : historique **correct > absent > mélangé** — un faux contexte nuit davantage qu'une absence (test de NOCIVITÉ, pas seulement d'usage) ; **condition de PASS : correct − absent ≥ +10 pts ET mélangé ≤ absent** ; (c) publier l'écart oracle-exact vs oracle-à-information-égale. Même famille que la validité de construit — intégré au registre.

## Positionnement

Collision 2026 **partielle** (OmniRL/AnyMDP, L2World, PSALM les plus proches — sans structure relationnelle typée). Revendiquer la **propriété mesurée** — contrôleur ≤5 M non-autorégressif sur graphe typé, effets inférés d'un historique court, fraction d'épisodes mélangés contrôlée, oracle exact — et non la nouveauté par combinaison de termes (cf. décision du 23 septembre, §Terminologie).

## Séquence convergée (conditions, pas horloge)

S1 (récupération exploratoire) → S2 (V1-bis) → S2b (DAgger gelé, règle ci-dessus) → S3 (récurrence, si le contraste données-vs-calcul le justifie) → P2 avec in-context intégré (mixed-episodes + tête d'effet + oracle à information égale). Chaque étape conserve ses deux verts séparés et ses errata append-only.
