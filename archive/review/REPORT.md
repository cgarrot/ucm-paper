# Veille web 2025–2026 — UCM (Universal Control Model)

**Agent :** web-watch (subagent délégué) · **Date de veille :** 24 sept. 2026 (au plus tôt ; derniers incidents repo datés du 23/09).
**Outils :** Ollama Cloud `web_search` + `web_fetch` (12 recherches, ~10 récupérations avant quotas horaires atteints).
**Périmètre :** papiers 2026 prioritaires (derniers mois), puis fin 2025, sur les 6 thèmes demandés ; recoupement P1/P2/P3 ; opportunités falsifiables branchables sur le harnais UCM (oracle exact, ~19 h accélérateur restant, discipline scellée).

**Convention de statut** (règle maison §sources) :
- **[V]** = vérifié (page intégrale ou large extrait lu via web_fetch) ;
- **[A]** = abstract/résumé seulement (extrait de résultat de recherche ou abstract arXiv) ;
- **[S]** = supposé, non re-vérifié (titre + snippet).

Contexte projet utilisé pour le recoupement (lecture autorisée) : `universal-control-model-project-spec.md` (§12, §12.6, §13), `docs/PLAN.md` (§6 registre des leçons), `docs/REPORT-V0.md`, `docs/REPORT-V1-EXPLORATORY.md`, `reports/ucm-review/sources.md` (bibliographie déjà citée). Aucun fichier scellé ouvert ; aucune écriture hors de `reports/web-watch-2026/`.

---

## A. Cartographie par thème

### A1. Politiques généralistes compactes (sub-1M à ~10M params)

**Le fait marquant du trimestre : la catégorie « System One Models » existe désormais commercialement et en open source.**

1. **TypeSafe AI, « Introducing System One Models & Jev », 15 sept. 2026** — https://typesafe.ai/blog/introducing-system-one-models-and-jev **[V]** (page intégrale). Startup (ex-OpenAI) qui lance une classe de modèles « frontier » **non autoregressifs** : état non structuré en entrée, **décisions typées probabilistes en sortie** (jamais de texte), échantillonnage **parallèle** (toutes les sorties en une passe), calibration via « RLCD » (Reinforcement Learning for Calibrated Decisions), 70–500 ms/appel, ~2 ordres de grandeur moins cher que les LLM sur leurs « workflow evals », cardinalité jusqu'à 255 options (au-delà : scoring en 2 étapes). Démos notables : contrôle temps réel de Doom (~10 requêtes/s, ~7 $/h) et navigation Wikiracing (choix parmi des centaines de liens). Les nuances de l'annonce sont honnêtes (biais d'évaluateurs internes, vitesse mesurée depuis leurs laptops).

2. **cua-ai/cua-s1-forms (trycua), 18–19 sept. 2026** — https://huggingface.co/cua-ai/cua-s1-forms **[V]** (model card intégrale) ; annonce Show HN du 19/09 : https://news.ycombinator.com/item?id=49767564 **[A]**. **706 048 paramètres** (2,8 Mo), Transformer encoder byte-level 2 couches width 128 : scoreur d'options **one-pass** pour le remplissage de formulaires GUI — même contrat entrée/sortie que Jev. Points de détail précieux pour UCM :
   - entraînement 100 % synthétique (10 000 épisodes, catalogue de 55 concepts, confusers forcés type `email` vs `street`) ;
   - **splits disjoints par signature de champs** (le set de champs d'un form test n'est jamais vu au train) ;
   - **contrôle shuffled-context : 37 %** vs 99,95 % top-1 sur test synthétique form-disjoint (~15k décisions), 100 % sur une petite éval réelle (196 décisions) ;
   - comparaison publiée contre le Jev hébergé (99,7 % vs 83,6 %) ;
   - l'ordre d'exécution (fills → checks → click) est décidé par du code aval, pas par le modèle — le modèle ne fait que scorer.

**Lecture pour UCM :** la thèse architecturale d'UCM (couche de décision typée, non autoregressive, one-pass, ~0,7 M params) n'est plus une excentricité — c'est une catégorie produit ET un artefact open source répliqué. Personne dans cette vague ne fait : états relationnels/graphes, conditionnement par but multi-étapes avec oracle exact, transfert entre familles de mondes. Les pratiques d'évaluation convergent avec les nôtres (splits par signature, contrôle shuffled) : validation externe de notre discipline.

3. **SOD: Step-wise On-policy Distillation for Small Language Model Agents** — arXiv:2605.07725 (mai 2026) **[S]** (Tencent/ZJU ; snippet). Distillation on-policy pas-à-pas pour petits agents LM.
4. **EAGLE: Embodiment-Aware Generalist Specialist Distillation** (humanoid WBC) — arXiv:2602.02960 (fév. 2026) **[A]** (abstract lu). Boucle itérative généraliste↔spécialistes : fork de spécialistes depuis le généraliste, raffinement par robot, distillation DAgger retour dans le généraliste, jusqu'à convergence. Précédent pour « itérer la distillation plutôt qu'augmenter le modèle », dans un autre domaine (5 robots, 4 réels).

### A2. Méta-apprentissage sur mondes procéduraux/générés

1. **AnyMDP, « Towards Large-Scale In-Context RL by Meta-Training in Randomized Worlds »** — arXiv:2502.02869 (v4, 2025), **accepté NeurIPS 2025** : https://proceedings.neurips.cc/paper_files/paper/2025/hash/fa9e9b2a5176c6f72e78269087b9fe60-Abstract-Conference.html **[A]**. Déjà cité en S18 de notre bibliographie ; la nouveauté est le statut NeurIPS 2025. Monde = MDP randomisés, pas de structure relationnelle ni de « genres » partagés.

2. **RACES, « Verifiable Environments Are LEGO Bricks: Recursive Composition for Reasoning Generalization »** — arXiv:2606.12373, soumis 10 juin 2026 **[V]** (abstract + §1–3 lues). Cadre : chaque environnement vérifiable a une **signature typée (domaine, codomaine)** ; deux environnements sont composables si codomaine(e1)=domaine(e2) ; opérateurs **SEQUENTIAL / PARALLEL / SORT / SELECT** ; 300 environnements de base → dizaines de milliers de composites. Résultats (RLVR sur LLM 14B) : +3,1 pts moyen (48,2→51,3) sur 6 benchmarks jamais vus ; **50 environnements de base ≈ 300 environnements individuels**. C'est du raisonnement LLM, pas du contrôle — mais c'est la démonstration 2026 la plus claire que la **composition typée de mondes vérifiables** multiplie la diversité d'entraînement à coût quasi constant.

3. **Agent-World (RUC + ByteDance Seed)** — arXiv:2604.18292 (avr. 2026) **[A]** (abstract court). « Scaling Real-World Environment Synthesis for Evolving General Agent Intelligence » : synthèse à grande échelle d'environnements réels pour agents généraux.
4. **AgentMercury** — arXiv:2608.20634, soumis 21 août 2026 **[S]** (titre+snippet). Les agents synthétisent eux-mêmes des environnements vérifiables pour scénarios métier, à grande échelle.
5. **C-World: A Computer Use Agent Environment Creator** — ACL 2026 https://aclanthology.org/2026.acl-long.2001/ **[S]**. Génération à grande échelle d'environnements pour computer-use.
6. **World State Generator** — arXiv:2609.24744 (sept. 2026) **[S]**. Génération d'états-monde pour agents.
7. **Unsupervised Learning of Efficient Exploration: Pre-training Adaptive Policies via Self-Imposed Goals** — ICLR 2026 https://openreview.net/pdf?id=UmxTIxHWkl **[A]** (abstract). Pré-entraînement de politiques adaptatives par buts auto-imposés.

**Lecture pour UCM :** la vague 2026 est « générer des environnements à la chaîne pour des agents LLM » (texte, RLVR) ; AnyMDP reste la référence contrôle mais randomise des MDP non relationnels. **Le créneau « mondes de contrôle procéduraux à genres d'interaction typés pour contrôleur relationnel compact » reste vide** (voir §B/P2).

### A3. Transfert en imitation learning et faux positifs documentés

1. **Per-Domain Generalizing Policies: On Learning Efficient and Robust Q-Value Functions** (DFKI/Sarre) — arXiv:2603.17544 (mars 2026) **[V]** (abstract + §1–3 + tableaux). Le piège documenté qui nous concerne : **la régression supervisée vanille de Q\* (depuis des plans optimaux d'un planificateur exact) apprend bien les valeurs des actions du prof mais ne SÉPARE PAS les actions du prof des autres** (différence moyenne Q(prof)–Q(non-prof) de 0,71–0,96 seulement) → politique quasi aléatoire à la généralisation, malgré un fit quasi parfait. **Fix publié** : régularisateur de marge Ω imposant Q(a_prof) < Q(a_non-prof), borne B=h\*(s)+1 → dépasse systématiquement les politiques V(s) sur 10 domaines PDDL, 3 architectures GNN, et devient **compétitif avec LAMA-first**. Bonus coût : politique Q = une seule passe sur l'état courant vs V = une passe par successeur → **3–18× plus rapide** (3,3 h vs 0,7 h cumulés R-GNN en moyenne). Code + données sur Zenodo.
2. **Monitoring Web Agents Without Internal Signals** — arXiv:2609.02057, soumis 2 sept. 2026 (UMN/Purdue/PSU) **[V]** (abstract + §1–3.2). Documente le faux positif de supervision qui nous a coûté V1 : **propager l'étiquette terminale d'un échec à tous les préfixes de la trajectoire est une supervision temporellement inexacte** (les préfixes valides d'un run échoué sont marqués à tort comme échec) — miroir exact de notre défaut de support V1 (supervision sur états initiaux seulement). Leur fix : étiqueter le **premier pas critique non corrigé** (key-step boundary) et garder « on track » les préfixes précédents.
3. **Causal IL Under Measurement Error and Distribution Shift** — arXiv:2601.22206, soumis 29 janv. 2026 **[S]**. IL offline avec état bruité + shift train/déploiement : conditions d'identifiabilité.
4. Contexte classique (déjà connu du projet, non comptés comme sources 2026) : DAgger (S02), Causal Confusion in IL (de Haan 2019), Data Quality in IL (Belkhale 2023).
5. **RoboTransfer** (CVPR 2026) **[S]** — transfert par diffusion vidéo pour manipulation : autre sens du mot « transfert », hors périmètre contrôle compact.

### A4. Contrôleurs récursifs/itératifs (anytime compute, deep supervision, value/Q\* distillation)

1. **Per-Domain Generalizing Policies (cf. A3.1)** **[V]** — l'ingrédient « distillation Q\* depuis planificateur exact dans un GNN, une passe » est publié, avec son piège et son fix. PAS d'itération, PAS d'anytime, PAS de supervision profonde : GNN à profondeur fixe, un domaine par modèle.
2. **Learning to Search and Searching to Learn for Generalization in Planning** (RWTH/Geffner) — arXiv:2605.25720 (mai 2026) **[V]** (abstract + §1–2). Boucle auto-améliorante : heuristique Q en GNN relationnel guide un **WA\*** (best-first, pas la recherche temps-réel du RL), les données de recherche ré-entraînent le Q. Généralisation en taille forte : **heuristiques entraînées sur Blocksworld <30 blocs résolvent des instances à 488 blocs SANS recherche** ; Sokoban, PushWorld, The Witness, IPC'23. Message structurel : quand le modèle est connu, best-first > temps-réel pour explorer — pertinent pour notre oracle exact.
3. **Agentic Test-Time Scaling for WebAgents** (Berkeley, Mahoney/Keutzer/Gholami) — arXiv:2602.12276 (fév. 2026) **[A]** (première page PDF via recherche). Le test-time scaling sur tâches agentiques multi-étapes se comporte différemment du single-step ; petit modèle + budget de calcul adaptatif ≈ gros modèle (détails non récupérés — quota).
4. **BrowseConf: Confidence-Guided Test-Time Scaling for Web Agents** (Tongyi/Alibaba) — ACL 2026 Findings https://aclanthology.org/2026.findings-acl.21.pdf **[S]**. TTS guidé par confiance pour agents web.
5. **Generalized Agent Iteration** — arXiv:2609.13406, soumis 11 sept. 2026 **[V]** (abstract intégral). Cadre formel unifiant policy iteration classique et « recursive self-improvement » : deux cadrans (le mécanisme d'amélioration est-il dans l'agent ; la norme est-elle externe). Papier de cadre, pas d'expériences contrôle compact — utile comme vocabulaire de positionnement, pas comme collision.
6. **Neural Value Iteration** — PMLR v337 (2026) https://proceedings.mlr.press/v337/you26a.html **[S]**. Itération de valeur NEURALE pour POMDP (α-vecteurs) : résonne avec P1 (« le réseau FAIT l'itération de valeur »), non vérifié en détail.
7. **Efficient Lookahead Encoding and Abstracted Width** — arXiv:2605.18674 (mai 2026) **[S]**. Encodage de lookahead pour politiques générales.
8. **Trust Region Policy Distillation / Extreme Region Policy Distillation** — arXiv:2607.04751, 2605.25582 **[S]**. Distillation on-policy stable (contexte LLM).

### A5. Petits agents UI/web (GUI/VLM)

1. **cua-s1-forms + Jev** (cf. A1.1–A1.2) **[V]** — les deux références du thème.
2. **Monitoring Web Agents Without Internal Signals** (cf. A3.2) **[V]** — monitoring black-box (31 features « macro » dont **répétitions d'actions et boucles** ; features « micro » de cohérence par requêtes répétées) : compétitif avec les signaux internes (logits) pour prédire l'échec depuis un préfixe, transfère à des catégories de sites jamais vues ; permet l'intervention précoce sous budget de fausses coupes fixé.
3. **Learning Simple Test-Time Environments for LLM Web Agents** — arXiv:2608.29305 (août 2026) **[S]**. Environnements simples appris pour agents web (les complexes font s'effondrer la perf).
4. **ShowUI** (2024, déjà connu) et les VLM edge : **Supertron3-0.8B** (HF, compact VLM for GUI agents) https://huggingface.co/Surpem/Supertron3-0.8B **[S]** — la vague « GUI agent <1B à l'edge » est réelle mais reste autoregressive/VLM.
5. BrowserGym/WebArena/WebLINX : inchangés par rapport à notre bibliographie (S20–S22).

### A6. Évaluation préenregistrée / audits de validité en RL/IL

1. **Preregistration for Experiments with AI Agents** (Vaccaro, MIT IDSS) — arXiv:2606.11217 (juin 2026) **[V]** (abstract + §1–2). Étend le préenregistrement aux expériences sur agents IA ; catalogue des **degrés de liberté du chercheur** spécifiques (prompt × modèle × température × seed × parsing) ; « le coût marginal bas + la flexibilité de spécification = specification search invisible » ; template de préenregistrement proposé. **UCM est en avance sur cette littérature** (gates, canons scellés, registre des leçons §6, dev-numbers documentés) — notre discipline est publiable en soi.
2. **Automated Benchmark Auditing (ABA)** — arXiv:2605.26079 (mai 2026, Duke/Together/Stanford) **[V]** (abstract + §1–3). Audit agentique de **168 benchmarks** : >25,7 % des tâches auditées ont un problème critique (instruction ambiguë, environnement en conflit, ground-truth faux) ; **filtrer ces tâches décale les classements et ajoute +9,9 % (SWE-bench Verified) / +9,6 % (Terminal-Bench 2)**. Preuve chiffrée 2026 que la validité des instruments est un problème de première grandeur.
3. **Construct Validity Failures in Agentic AI Benchmarks: An Empirical Audit** — atelier KDD 2026 https://kdd-eval-workshop.github.io/agenticai-evaluation-kdd2026/assets/papers/48_Construct_Validity%20(1).pdf **[S]**. Audit de validité de construct (psychométrie) appliqué aux benchmarks agentiques.
4. **The Double Measurement Confound in Agent Benchmarks** — arXiv:2609.09218, soumis 6 sept. 2026 **[S]**. De-scaffolding vs ground-truth scoring, fiabilité au-delà de la moyenne.
5. **Are LLM Benchmarks Already Contaminated?** (GEM @ ACL 2026) **[S]** ; **ReplicatorBench** arXiv:2602.11354 **[S]** ; **AgentSuite** (ICML 2026 poster) **[S]**.
6. **When Agents Do Not Stop: Infinite Agentic Loops** — arXiv:2607.01641, 2 juil. 2026 (HUST) **[V]** (abstract + §1–3). Les **boucles infinies agents (IAL)** sont une classe de défaillance reconnue : IAL-Scan (analyse statique, ALDG) sur 6 549 repos → 68 échecs confirmés dans 47 projets, précision 91,9 %. Les bornes de framework (max_turns, recursion_limit) sont « sémantiquement fragiles ». Contexte LLM, mais consacre « la non-terminaison » comme failure class de premier plan — exactement nos cycles absorbants V0.

---

## B. Recoupement P1 / P2 / P3 — collisions

### P1 — contrôleur itératif à champ de valeur (bloc relationnel partagé itéré T fois, deep supervision, distillation Q\* depuis oracle exact, « la limite d\*~12–15 est une limite de CALCUL pas de taille »)

**Verdict : collision PARTIELLE, deux fois, en 2026 — aucun des deux ne couvre le paquet complet.**

- **Qui s'en est approché :**
  - DFKI/Sarre (arXiv:2603.17544 **[V]**) : distillation Q\* depuis planificateur exact dans un GNN relationnel, une passe, 10 domaines PDDL, compétitif LAMA-first. **Mais** : profondeur fixe (pas d'itération/anytime), pas de deep supervision le long des itérations, un domaine par modèle, pas de familles de mondes procéduraux.
  - RWTH/Geffner (arXiv:2605.25720 **[V]**) : GNN relationnel Q + boucle auto-améliorante avec best-first search, généralisation en taille spectaculaire (30→488 blocs zero-shot). **Mais** : l'itération est dans la BOUCLE APPRENTISSAGE (search↔learn), pas dans le réseau au moment de la décision ; pas d'anytime compute par décision.
- **Pièges documentés à importer tels quels :** (1) la régression Q\* vanille ne sépare pas optimal/non-optimal → perte de marge obligatoire (A3.1) ; (2) quand le modèle est exact, best-first bat le temps-réel pour générer des données d'entraînement (A4.2) ; (3) les politiques Q one-pass sont 3–18× moins chères que V+successeurs — argument de déploiement pour notre tête de candidats.
- **Ce qui reste à nous :** bloc relationnel **partagé itéré T fois au moment de la décision** (anytime : T adaptatif au doute), deep supervision par itération, oracle exact sur **familles de mondes procéduraux** (pas PDDL figé par domaine), contrôleurs <1,2 M params. Aucun papier trouvé là-dessus dans les requêtes effectuées.

### P2 — milliers de mondes procéduraux à « genres d'interaction » partagés (zéro embedding frais par monde)

**Verdict : PAS de collision directe trouvée. Les ingrédients convergent depuis trois directions distinctes, sans personne sur l'intersection.**

- AnyMDP (NeurIPS 2025 **[A]**) : randomisation de MDP pour l'ICRL — pas de structure relationnelle ni de genres typés partagés.
- RACES (juin 2026 **[V]**) : composition typée d'environnements vérifiables — exactement notre idée de « genres à interfaces typées », mais pour le RAISONNEMENT LLM (RLVR), pas pour des politiques de contrôle, et sans contrôleur compact.
- Agent-World / AgentMercury / C-World / World State Generator (2026 **[A/S]**) : synthèse massive d'environnements — pour agents LLM textuels.
- XLand (2021–2025, non re-vérifié ici) : mondes 3D massifs, budget de calcul hors de notre échelle.
- **L'intersection « genres d'interaction typés × contrôle relationnel compact × zéro embedding frais × transfert compositionnel mesuré » est vide** au vu des requêtes effectuées. Résultat encouragement de RACES à importer : **50 briques composées ≈ 300 briques individuelles** — la composition est le levier de diversité le moins cher connu en 2026.

### P3 — boucle prédire-vérifier runtime (récupération d'erreur ; 100 % des échecs V0 = cycles absorbants)

**Verdict : le problème est désormais consacré chez les agents LLM ; personne ne l'attaque pour un contrôleur compact sur mondes synthétiques.**

- Monitoring Web Agents (sept. 2026 **[V]**) : prédiction de risque depuis préfixe observable (répétitions/boucles comptables !), étiquetage key-step, intervention précoce sous budget de fausses coupes. Transposition directe et peu coûteuse à notre harnais (nous avons logs d'épisodes + échecs catégorisés).
- IAL-Scan (juil. 2026 **[V]**) : les boucles infinies agents = classe de défaillance nommée, analysée statiquement ; les bornes de terminaison existantes sont fragiles.
- BrowseConf / Agentic TTS WebAgents **[A/S]** : calcul adaptatif guidé par la confiance — le « combien réfléchir » à la demande, côté LLM.

### Collisions écosystème (ni P1/P2/P3, mais structurantes)

**Jev + cua-s1-forms (sept. 2026 [V]) occupent la niche « couche de décision typée one-pass pour UI ».** Ce n'est pas notre revendication (eux : un domaine, entrée texte byte-level, pas de buts multi-étapes structurés, pas d'oracle exact, pas de transfert entre familles), mais : (a) la comparaison publique existe désormais (leur 706k vs notre 695k — frappant) ; (b) leurs contrôles (splits par signature, shuffled-context) valident notre discipline ; (c) tout rapport UCM doit désormais se positionner explicitement par rapport à cette vague.

---

## C. Opportunités non exploitées — 5 expériences falsifiables branchées sur le harnais

> Toutes réutilisent l'existant : oracle TinyGraphKey/SIW, logs d'épisodes V0/V1, splits scellés, cellule DEV disjointe, protocole à étages PLAN #10. Aucune n'ouvre un scellé hors protocole déclaré. Coûts en heures d'accélérateur sur l'enveloppe ~19 h.

### C1. Tête Q\* à marge de séparation (adoption directe du fix DFKI → P1)
- **Origine :** arXiv:2603.17544 [V] — « vanilla SL of Q\* ne sépare pas les actions ; une marge le corrige ».
- **H :** à corpus oracle identique (labels `optimal_actions` + `d_star` déjà dans les records), une perte Q\* + marge (imposer Q(s,a\*) < Q(s,a) − m pour tout non-optimal, m calibré sur DEV) améliore le succès fermé vs notre set-valued BC actuel, surtout sur la strate à ties (multiples optimums = exactement là où la séparation compte).
- **T :** même architecture B144/160, 2 bras de perte × 5 seeds, entraînement identique (updates, batch, FP32), sélection sur TRAIN (règle #10), évaluation sur strate VAL d\*≥3 ∪ ties, confirmation G1 scellé existant si la strate VAL passe le barre GATE-5.
- **S :** Δsuccès fermé ≥ +5 pts (IC95 apparié > 0) sur la strate d\*≥3 ∪ ties, sans perte > 2 pts sur d\*≤2. Échec déclaré si l'IC traverse 0 → le set-valued BC reste le choix, résultat négatif publié dans le registre.
- **Coût :** ~2–3 h accél. (2 bras × 5 seeds × 10k updates, modèles <1 M params, CPU/FP32 possible).

### C2. Moniteur de préfixe « signaux observables » pour cycles absorbants (→ P3)
- **Origine :** arXiv:2609.02057 [V] — features macro comptables (répétitions, boucles) suffisent à prédire l'échec depuis un préfixe, sans signaux internes ; arXiv:2607.01641 [V] — la non-terminaison est une failure class nommée.
- **H :** à partir de features purement observables déjà loggées (répétition d'action, retour à un `state_hash` déjà visité, stagnation du score candidat max, entropie des scores), un prédicteur linéaire/petit MLP détecte l'entrée en cycle absorbant avant l'épuisement de l'horizon.
- **T :** (1) OFFLINE : entraîner sur les logs DEV existants (échecs catégorisés V0 + interactions-log V1), étiqueter au « premier pas critique non corrigé » (key-step, cf. 2609.02057) ; évaluer AUROC par préfixe sur DEV disjoint. (2) BOUCLE FERMÉE (DEV uniquement) : intervention = STOP précoce + mesure des décisions évitées.
- **S :** AUROC ≥ 0,85 au niveau préfixe ; en boucle fermée : rappel cycles ≥ 0,8 à taux de fausses coupes ≤ 0,1 sur épisodes qui auraient réussi ; aucune baisse de succès (le STOP précoce n'est jamais compté succès). Si AUROC < 0,7 : les cycles absorbants V0 ne sont pas détectables par signaux observables → info négative utile pour P3.
- **Coût :** ~0,5–1 h accél. (entraînement trivial), le reste en CPU sur logs existants. **Meilleur ratio coût/valeur du lot.**

### C3. Contrôles shuffled-goal / shuffled-context généralisés (→ validité de V1-bis et de tout run SIW)
- **Origine :** cua-s1-forms [V] — contrôle shuffled-context à 37 % (vs 99,95 %) comme preuve que le modèle lit l'élément et non les stats de candidats.
- **H :** le modèle V1-bis lit le but et l'état ; si on permute les BUTS entre épisodes test (ou les labels des widgets SIW, vocabulaire fermé §12.6), le succès doit s'effondrer vers ≤ 2× le hasard.
- **T :** évaluation pure sur cellule DEV (aucun scellé) : (a) buts permutés intra-pool ; (b) labels permutés (déjà couvert par G5 permutation, à répliquer côté SIW) ; (c) candidats seuls (déjà en M3).
- **S :** chute à ≤ 2× random valide le construct ; un succès qui reste élevé sous but permuté = fuite détectée → blocage du run confirmatoire avant ouverture des scellés.
- **Coût :** < 1 h, évaluation uniquement. **À faire avant TOUTE confirmation V1-bis.**

### C4. Deep supervision itérative T-fois sur bloc relationnel partagé (cœur de P1)
- **Origine :** notre piste P1, maintenant armée par deux résultats 2026 : la marge de séparation (C1) et « best-first + données de search » (2605.25720 [V]) — et par la non-collision démontrée (§B/P1).
- **H :** à nombre de paramètres égal (~695k), un bloc relationnel partagé itéré T=4 avec supervision Q à CHAQUE itération (deep supervision) et données échantillonnées sur états intermédiaires (pas seulement initiaux — leçon V1) lève la barre d\*~12–15 : la limite devient une limite de CALCUL (choisir T), pas de capacité.
- **T :** 3 bras (T=1 / T=4 supervisé / T=4 non-supervisé-entre-itérations = contrôle compute) × 3 seeds pilote sur la strate d\*13–24 déjà identifiée à l'inventaire M0 ; si le pilote passe, confirmation 5 seeds + cellule scellée à figer avant génération (leçons #19/#22 : runner + freeze livrés ensemble).
- **S :** ≥ +15 pts succès sur d\*13–24 pour T=4 supervisé vs T=1 (IC95 > 0), sans perte > 2 pts sur d\*≤12 ; le bras contrôle compute ne doit PAS gagner (sinon c'est un effet de capacité, pas de calcul).
- **Coût :** pilote 4–6 h accél., confirmation conditionnelle +4 h. **C'est le pari majeur ; ne le lancer qu'après C1+C2 (C1 fournit la bonne perte pour C4).**

### C5. Pilote mini-RACES : genres d'interaction typés composés (cœur de P2)
- **Origine :** RACES [V] — composition typée = diversité combinatorielle quasi gratuite (50 briques ≈ 300) ; non-collision §B/P2.
- **H :** avec G=4 genres (navigation, verrou-clé, port/contenant, validation/choix) à interfaces typées, l'entraînement du MÊME contrôleur (zéro embedding frais) sur des mondes COMPOSÉS 2-genres généralise zero-shot aux compositions non vues et au 5e genre exclu, là où un entraînement mono-genre à volume égal ne généralise pas.
- **T :** générateur par genres typés branché sur `ucm/data/generate.py` (isomorphisme/splits inchangés §7.3) ; bras composé vs mono-genre à volume identique de couples uniques (leçon §12.5 : couverture publiée par bras) ; 5 seeds ; test = compositions jamais vues + genre exclu.
- **S :** ≥ +10 pts zero-shot sur compositions non vues (IC95 > 0) pour composé vs mono-genre. Un croisement nul ou négatif réfute la version faible de P2 (« la diversité de genres aide le transfert compositionnel ») à petite échelle.
- **Coût :** dév générateur 6–10 h de code (pas d'accél.), puis 3–4 h accél. pour le pilote. **À ne lancer que si C4 est reporté ou déjà en vol.**

**Total si tout est fait dans l'ordre : ~10–14 h accél. sur 19 h disponibles, avec C1+C2+C3 ≈ 4–5 h seulement.**

---

## D. Trous non couverts, risques et limites de la présente veille

1. **Quotas atteints** (12 `web_search`, ~10 `web_fetch` sur la fenêtre horaire). Requêtes NON exécutées faute de quota, à rejouer : « Genie 3 / world models génératifs 2026 », « task saturation / generalist RL small models 2026 », « XLand 3 » (frappe directe n'a rien donné de neuf — [S]), « adversarial sufficiency of demonstrations », « BabyAI/MiniHack 2026 », veille blogs (Henderson, spans de threads X/Reddit pour cua-s1/Jev).
2. **Détails non vérifiés** : Agentic Test-Time Scaling for WebAgents (seule la 1re page du PDF a été vue [A]) ; Agent-World, AgentMercury, C-World, World State Generator, BrowseConf, SOD, Supertron3-0.8B ne sont connus que par abstract/snippet [A/S]. Ne pas bâtir de décision sur ces seuls extraits.
3. **Dates :** les mois arXiv sont déduits de l'ID (YYMM) et confirmés sur la page pour 2606.12373 (10/06/2026), 2609.13406 (11/09/2026), 2608.20634 (21/08/2026), 2609.02057 (02/09/2026), 2609.09218 (06/09/2026), 2607.01641 (02/07/2026), 2601.22206 (29/01/2026). La date du jour est déduite (≥ 23/09/2026).
4. **Aucune affirmation de priorité absolue** : « P2 sans collision » vaut pour les requêtes exécutées ci-dessus, pas pour l'espace bibliographique entier ; le projet avait déjà noté (§16.3 spec) qu'une revendication de nouveauté exigerait une passe ciblée supplémentaire — ce rapport en est une, pas LA dernière.
5. **Rien trouvé sur :** « préenregistrement appliqué au RL/IL contrôle compact hors LLM » (les papiers 2026 trouvés traitent des agents LLM ou des benchmarks) — requêtes : « preregistration reinforcement learning evaluation 2026 », « validity audit RL imitation learning control 2026 ». Le monopole de fait reste aux pratiques maison UCM ; c'est aussi un angle de publication.

---

## Synthèse — ce qu'on devrait faire lundi matin

Classement coût/valeur (budget ~19 h accél.) :

1. **C3 — contrôles shuffled (≤ 1 h, éval pure).** Garde de validité AVANT toute confirmation V1-bis. Un refus de s'effondrer sous but permuté = fuite à corriger avant d'ouvrir le moindre scellé. Rien ne justifie de l'attendre.
2. **C2 — moniteur de préfixe (≤ 1 h accél.).** Branché sur des logs qui existent déjà ; transforme « 100 % des échecs V0 sont des cycles absorbants » en un résultat positif (détection/intervention) ou en une réfutation propre de P3. Adoption directe d'un papier de septembre 2026.
3. **C1 — perte Q\* + marge (2–3 h).** Adoption d'un piège documenté + fix publié (DFKI 2026). Améliore le socle et fournit la perte dont C4 a besoin. Falsifiable sur la strate VAL existante.
4. **C4 — deep supervision itérative T-fois (4–6 h pilote).** LE pari P1, maintenant dérisqué côté « quelqu'un l'a-t-il fait ? » (non, cf. §B) et côté « quel piège éviter » (C1). Ne lancer qu'après C1 ; confirmation conditionnelle seulement.
5. **C5 — pilote genres typés composés (3–4 h accél. + dev).** P2 à échelle pilote, validation du levier le moins cher de diversité (RACES). En visière, pas en vol, tant que C4 n'est pas décidé.

**Trois phrases à retenir :** (1) la niche « décision typée one-pass compacte pour UI » est devenue une catégorie commerciale et un artefact open source de 706k params avec nos propres pratiques d'audit — UCM doit s'y positionner explicitement, et peut s'en servir comme benchmark externe de comparaison ; (2) P1 a deux collisions partielles en 2026 (Q\*-GNN one-pass avec marge ; search↔learn auto-améliorant) qui nous livrent le piège et le fix sans menacer le cœur anytime/deep-supervision du paquet ; (3) P2 et P3 restent des terrains vides côté contrôle compact, avec des leviers 2026 prêts à l'emploi (composition typée ; monitoring de préfixe observable).
