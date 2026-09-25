# CINQUIÈME REVUE EXTERNE — état ~18:15 du 23/09/2026 (reçue ~18:40)

*Modèle externe différent, sollicité par cgarrot. Revue documentaire + vérifications externes ciblées ; dépôt non exécuté. Verbatim archivé. Vérification orchestrateur : SD 10.9 CONFIRMÉ introuvable depuis les 5 diffs publiés (12.609 n-1 / 11.278 n) ; freeze v4 fichiers 10/10 correct (rejeté 20261031 absent, remplacement 20261040 présent) — ce point de la revue était déjà propre. Relayé au lead (m_muebd1lb).*

---

**Oui, cette mise à jour va dans une meilleure direction. Plusieurs problèmes importants ont été corrigés, et le projet est mieux aligné avec l'objectif : un petit modèle qui exécute, s'adapte et évite de solliciter un gros modèle à chaque décision.**

Mais il faut distinguer deux avancées : **vous avez nettement amélioré la préparation de l'expérience. Vous n'avez pas encore démontré une nouvelle capacité du modèle.** Le test confirmatoire reste en HOLD. Mon avis général : **continuer, sans changer à nouveau toute la direction, mais régler quelques points scientifiques avant de lancer la confirmation** — notamment une incohérence numérique dans le dimensionnement statistique et une réserve sérieuse sur le bootstrap proposé.

## 1. Où vous en êtes réellement

Ce qui a progressé : épisodes complets d'adaptation avec états intermédiaires et STOP terminal (corrige la cause principale de l'ininterprétabilité de V1) ; quatrième bras de contrôle ; budgets en épisodes ; dix seeds. La quatrième revue a identifié trois problèmes — store, héritage k, init — corrigés avec tests dont un vrai test de non-héritage par corruption. L'écart 79/83 échecs en V0 est expliqué (97,50 % officiel = 79/3160 ; les 83 = autre replay ; RNG et harnais diffèrent — attribution au hasard non prouvée, nuance correcte).

Ce qui n'est pas encore démontré : aucun nouveau résultat confirmatoire ; récupération, récurrence, adaptation par historique, démo logicielle restent à réaliser ; couche Judge conceptuelle non normative.

Formule exacte : V0 acquise dans son périmètre ; V1 exploratoire ; V1-bis beaucoup mieux préparée mais pas confirmée ; capacités futures mieux définies.

## 2. Décisions particulièrement bonnes
Vision produit plus claire (compréhension/exécution/escalade explicites ; compacité = contrainte mesurée, 2-3M acceptables) — conserver. « Données avant calcul » = bon arbitrage. Résultats négatifs mieux encadrés (échec V1-bis limiterait une combinaison précise, pas le transfert en général). Ne pas rouvrir la stratégie.

## 3. Trois points scientifiques avant le GO

### A. Le « sous-espace fréquent » doit correspondre à un score précis
Seuil masse ≥0,5 % choisi après exploration puis verrouillé : défendable, à formuler honnêtement. Mais : comment passe-t-on d'un sous-espace de DÉCISIONS (cellules prédicat×attrs×action×profondeur) au score de RÉUSSITE D'ÉPISODES COMPLETS (600 épisodes) ? Il ne faut surtout pas que l'inclusion dépende de la trajectoire produite par chaque modèle (population évaluée variable par bras, pouvant retirer précisément ses échecs). Options : (simple) réussite épisodes complets = primaire, couverture = diagnostic ; (restreint) admissibilité des épisodes définie à l'avance sur référence indépendante des modèles. Expliciter budget primaire, contraste primaire, pondérations. Couvrir ≠ savoir agir : N=512 = expérience secondaire préannoncée, pas prolongation post-test2.

### B. Le bootstrap « seeds puis prédicats » pas justifié tel que formulé (réserve principale)
Le score agrège 4 prédicats DÉTERMINÉS à poids égaux — pas 4 catégories tirées au hasard. Rééchantillonner les observations dans des catégories fixées ≠ rééchantillonner les catégories. Exemple : +30/+30/−20/−20 par seed → moyenne +5 constante ; le 2e niveau crée une incertitude sur le mélange de tâches que le benchmark n'a pas. Choisir explicitement : (i) conclusion CONDITIONNELLE au benchmark fixe → vecteur des 4 scores par seed, rééchantillonner les seeds en blocs appariés ; (ii) conclusion sur nouveaux layouts/épisodes → inclure cette variabilité en respectant les regroupements. « 1 niveau anti-conservateur, 2 niveaux corrigent » non valide comme règle générale : il faut rééchantillonner les BONNES unités. Demande : test synthétique + comportement sous absence d'effet simulée.

### C. L'écart-type de dimensionnement introuvable
10,9 pts annoncé ; les 5 différences publiées (14,33 ; 0,67 ; 25,17 ; 24,17 ; −1,50) donnent moyenne 12,568, SD(n−1) 12,609, SD(n) 11,278 — 10,9 introuvable avec aucune convention. Expliciter la provenance. Indicatif : t apparié unilatéral 5 %, 10 paires, effet 10 pts, SD 12,61 → ~75 % de puissance. Réconcilier avant de figer la promesse. FAIL = « critère de promotion non atteint », pas « absence de transfert ».

## 4. Chantier technique : correctifs bons, validation globale manquante
HOLD maintenu. Livraison attendue : UNE exécution DEV complète sur un snapshot précis (données → store → inits → entraînements indépendants → évaluations → raw → stats → publication → relecture+recalcul), pas une collection de validations partielles. Fixture 24 cellules (4 bras × 2 seeds × 3 k) = bon format, petit budget OK si VRAI chemin. Trois détails : complétude par CELLULES exactes (120 IDs, sans doublon/manquant), pas seulement workers ; freeze distinguant fichiers acceptés (10) des historiques (11, tirage rejeté) — [vérifié orchestrateur : déjà propre en v4] ; politiques de checkpoint SOURCE (best_validation) et CIBLE (final_at_fixed_budget) en champs SÉPARÉS. Process : intégrateur unique + snapshot candidat + état de référence unique. [Vérifié orchestrateur : SD confirmé introuvable.]

## 5. P2 intéressant mais doit rester identifiable
Tête d'effet : (observation, action, historique pertinent) → DISTRIBUTION des effets possibles en régime inconnu ; arêtes « effet observé » = observations réelles, jamais règles causales injectées. Test historique mélangé : cohérent-autre-règle = bon contrefactuel (la décision doit suivre l'information) vs manifestement incohérent = robustesse (un bon modèle peut l'ignorer sans être insuffisant). Seuil +10 pts à piloter DEV (si sans-historique à 95 % → impossible). Oracle à information égale : « BFS sur croyance » sous-spécifié (optimiste/robuste/probabiliste ?) — commencer par historique identifiant la règle + nouveaux objectifs sans update ; exploration active = expérience séparée. S2b : factorielle 2×2 lisible (B144×récurrent)×(oracle×récupération), supervision intermédiaire ensuite seulement. DAgger : « −30 % cycles » peut vouloir dire arrêt plus précoce — exiger absence de dégradation inacceptable du succès et des erreurs irréversibles.

## 6. Couche Judge : bon découpage, pas une priorité d'intégration
Séparations validées (cœur/juge/planificateur/politique d'autorisation ; UNKNOWN distinct de PASS ; juge probabiliste jamais seul autorisateur d'irréversible). LeJudge README : module de COÛT pour un planificateur (PushT), faux positifs 38-53 % sur conformes proches d'une violation — pas transposable en couche de permissions. Trois corrections : précondition fausse MAINTENANT ≠ but impossible (« SUBMIT non autorisé maintenant » vs « aucun plan autorisé n'atteint la soumission » — la confusion bloquerait les tâches multi-étapes) ; contrat de jugement lié à action+arguments+version d'état+observation fraîche+provenance (pas de PASS caché réutilisé ; « A avant B » nécessite mémoire d'événements) ; juge comparé au checker DÉTERMINISTE, pas seulement à UCM seul (avantage sur le meilleur comparateur local, à coût déclaré, avec refus injustifiés mesurés). Décision : réserve de conception ; ne pas intégrer LeWM/CEM/Jev au chantier V1-bis.

## 7. Coûts : ordre de grandeur crédible, incertitude à présenter correctement
120 entraînements, 72 000 épisodes d'évaluation — cohérent. Extrapolations ~10,53 h et ~12,98 h vs annoncé 11,5 h : plausible. Mais p95×72 000 ≠ p95 de la durée totale (présenter comme extrapolation à coût élevé par épisode) ; profil sur modèle frais et machine partagée ; distinguer CPU / cumulé workers / écoulé ; « heures accélérateur » imprécis pour des mesures CPU.

## 8. Quoi faire maintenant
1. Rendre V1-bis interprétable et exécutable (SD, score+bootstrap exacts, store+fichiers acceptés dans le freeze final, fixture complète — audits scientifique/technique en parallèle mais validation finale sur le MÊME candidat).
2. Exécuter la campagne et la clore honnêtement (positif = preuve limitée mais utile ; indéterminé ≠ échec général ; « demain matin » = objectif conditionnel, pas autorisation acquise).
3. Concentrer le travail modèle sur UNE capacité nouvelle mesurable — meilleur candidat : même contrôleur, nouvelles règles, quelques observations, aucun réentraînement, réussite de nouveaux objectifs (+ petite démo logicielle vérifiant la réduction d'appels au gros modèle).

### Verdict
Direction meilleure ; le projet sait remettre en cause ses propres résultats — précieux. Inquiétude principale : accumuler protocoles/couches/audits sans livrer une expérience complète puis une capacité supplémentaire. **Pas besoin d'un nouveau grand pivot : fermer proprement V1-bis, puis prouver une chose nouvelle que B144 ne sait pas encore faire.**
