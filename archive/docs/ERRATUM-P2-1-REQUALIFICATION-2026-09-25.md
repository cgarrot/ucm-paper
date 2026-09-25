# ERRATUM — verdict P2-1 requalifié : « in-context NON TESTÉ » (revue de fin de cycle, 25/09)

**Statut : erratum append-only majeur.** Le verdict P2-1 « FAIL in-context » publié à la clôture du goal S2b→P2 est **requalifié** après la revue indépendante de fin de cycle : l'expérience n'a **pas testé l'in-context** — elle a mesuré la **fragilité du modèle à un contexte non pertinent**.

## 1. Pourquoi l'in-context était impossible par construction

`context_arms.py` : la sémantique TGK est **constante** entre épisodes, les cibles du contexte sont **canoniques** (agent/key/parcel/door0 — toujours identiques), et la source de l'historique est prise dans le plan optimal d'**un autre épisode** — le contexte ne portait **aucune information inférable** sur la tâche courante. Aucun modèle, aucun encodage, ne pouvait extraire de signal : l'identifiabilité était **impossible par construction**.

Le gate « oracle à information égale » (93,3 %) était **vacuement satisfait** : il mesurait « résoluble sans UNLOCK observé », pas « le contexte porte de l'information » — même classe de vacuité que le canon-non-chargé (cohérence interne du garde sans contrôle du but).

## 2. Ce que le run a réellement mesuré

- **DROP-HAVE −18,8 pp (8/8 seeds, p=0,031)** : le modèle est **distrayable par des arêtes non pertinentes** — un résultat réel et utile (robustesse au bruit de contexte), mais ce n'est pas un test d'in-context. « Le modèle lit le contexte » reste non établi (3 entraînements comparés, pas les mêmes poids — discriminant gratuit : réévaluer ckpts-run4 sans contexte).
- **DROP-AT +0,0 pp** : fenêtre **inatteignable par construction** (épisode 0050 échoue 24/24 → plafond effectif 91,7 %, marge 1,0 pp vs Δ_min 10,2). **§5.1 violée** : p_ref mesuré au budget pilote (300 updates), budget 4000 appliqué APRÈS gel.
- **Aucun transfert testé** : le modèle s'est entraîné SUR les familles dites « tenues à l'écart » — la question compositionnelle (entraîner signatures utilisées, 0-shot sur tenues-à-l'écart) reste **entière**.

## 3. Ce qui en découle

- L'ouvert « autre encodage » du rapport de clôture est **faux** : le problème n'est pas l'encodage mais l'absence de variable latente par épisode — tout encodage échouerait dans ce monde.
- **In-context gelé** jusqu'à : (i) effets latents par épisode dans le générateur ; (ii) gate de valeur d'information préenregistré (oracle avec contexte − oracle sans contexte ≥ 2×Δ_min).
- La direction suivante (revue de fin de cycle) : **S5 tranche fine** (voie ingénierie, VISION), **Recherche A** profondeur (données fraîches d*13-24 × beam), **Recherche B** transfert compositionnel (le vrai test).

## 4. Règles gravées au registre

1. **p_ref mesuré au budget FINAL** du run (jamais au budget pilote).
2. **Tout gate a un contrôle négatif** qui doit le faire échouer (sinon le gate est vacuement satisfait).
3. **Un secondaire préenregistré est exécuté ou déclaré en déviation** — jamais silencieusement omis.


## 5. Addendum (25/09 06:08) — le modèle LIT le contexte (discriminant de l'auditeur)

Le discriminant gratuit (96 évals, zéro entraînement, mêmes poids ckpts-run4, 3 conditions : sans contexte / avec correct / avec mélangé — artefact `artifacts/p2-run/discriminant/p2-discriminant-20260925T060749.json` sha `a4368beb…`, exécuté par l'auditeur) **établit** le chaînon manquant du §2 : **le modèle lit le contexte et il le distrait** — DROP-HAVE poids D1-mixed : sans contexte **55,2 %** vs avec **46,9 %** (5/8 seeds appariés, les deux conditions) ; les poids D1-shuffled l'ignorent (27,1/28,1/27,1 — l'asymétrie rend le discriminant concluant). La fragilité est un **résultat positif publiable** (l'encodage rend les arêtes saillantes, leur non-pertinence les rend nuisibles). **Caveat inchangé** : le contexte reste non pertinent par construction — cela n'établit pas la valeur de l'in-context (gelé avec ses 2 conditions). Le discriminant entre au registre comme outil standard.
