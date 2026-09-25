# SPÉC NORMATIVE SIW — Monde d'Interaction Synthétique (V1, §12.6)

**Statut :** contrat figé pour M-V1a. Toute déviation → registre PLAN §6.
**Interface :** contrat V0 §3.1 inchangé (observe/candidates/execute/goal_satisfied + canaux policy_input/supervision/provenance, schéma 0.2 étendu).

## 1. Objectif de conception

Exposer les problèmes structurels du logiciel (bindings sous labels arbitraires, arguments, many-candidates, ordre de validation, navigation) **sans** encodeur de texte, pixels, ni contenu réel. SIW est le pont désigné du protocole de transfert §12 : la question est de savoir si le principe TinyGraphKey survit à une interface logiciel-like.

## 2. Entités et observations

### 2.1 Types d'entités (registre fermé)

| Type | Attributs observables | Notes |
|---|---|---|
| `view` | — | Écran ; la navigation forme un graphe connecté (2-5 views) |
| `button` | `onclick_kind` ∈ {`submit`,`confirm`,`dismiss`,`none`} | `none` = distracteur cliquable sans effet |
| `field` | `filled` ∈ {true,false} | Champ texte ; valeur = binaire (rempli/vide) |
| `select` | `chosen_option_ref` (nullable) | Liste ; une option choisie parmi les siennes |
| `option` | — | Appartient à exactement un `select` |
| `form` | `status` ∈ {`draft`,`complete`,`submitted`} — **DÉRIVÉ** (M4) : complete ⟺ tous ses fields remplis ET son select choisi ; jamais une dimension d'état indépendante | Agrège des fields/selects + un bouton submit |
| `dialog` | `open` ∈ {true,false} | Boutons confirm/dismiss |

**Interdits en policy_input** : liste V0 §5.1 inchangée + `d_star`, rôle cible du but au-delà des références, plan, prochaine observation. Le rôle `onclick_kind` EST observable (le modèle voit ce qu'un bouton fait typiquement — comme il voit la topologie des pièces en V0) ; la difficulté vient de l'ordre, des arguments et de la navigation, pas de boutons mystères.

### 2.2 Relations (fermé)

`on_view(widget, view)` ; `nav_edge(view, view)` ; `part_of(field|select, form)` ; `option_of(option, select)` ; `in_dialog(button, dialog)` ; `submits(button, form)` ; `current_view(agent, view)` ; `filled(field, agent)` ; `chosen(option, select)`.

**Représentation unique (M2, réconciliée)** : les attributs portent l'état (`field.filled`, `select.chosen_option_ref`), les relations ci-dessus le mirent à des fins structurelles (certificat WS-B). Aucun marqueur polymorphe.

### 2.3 Labels

Chaque widget porteur de texte reçoit un **label généré** d'un vocabulaire fermé (~60 mots) avec **alias** (paires synonymes), tiré aléatoirement par layout — **le label n'a aucun lien sémantique avec le rôle** (un bouton `submit` peut porter le label "annuler"). Le binding est **par référence d'entité uniquement**. Le label est dans policy_input comme attribut textuel mais **aucun embedding préentraîné** : il sert uniquement à garantir l'absence de raccourci par mémorisation de label. **L'isomorphisme WS-B ignore les labels** (§4).

## 3. Actions et candidats

### 3.1 Actions (fermé)

| Action | Précondition de validité | Effet |
|---|---|---|
| `NAVIGATE(view)` | view adjacent (nav_edge) à la courante | change current_view |
| `CLICK(button)` | bouton visible (on_view courant) | selon `onclick_kind` (§3.2) |
| `TYPE(field)` | field visible ET non rempli | filled := true (remplissage binaire — §5 révisé) |
| `SELECT(option)` | option visible, select non choisi | chosen := option |
| `STOP` | — | terminal, évaluateur §6 |

**Candidats = énumération syntaxique goal-blind ET state-blind COMPLÈTE** (m6) — ne dépendent que du layout, comme V0 ; `NAVIGATE(view courante)` = **invalide** (parallèle V0) : `NAVIGATE` × toutes les views, `CLICK` × tous les boutons, `TYPE` × tous les fields, `SELECT` × toutes les options, `STOP`. **K = V + B + F + O + 1**, cible 30-60 par layout (§4). Ordre canonique stable, randomisation en aval (identique V0).

### 3.2 Effets de CLICK selon kind

**Règle de visibilité (M3, normative)** : `visible(w) = on_view(current_view) ET (non in_dialog OU dialog.open)` — les widgets hors dialog restent visibles dialog ouvert (pas de modale bloquante V1, déclaré).

- `submit` : si form `complete` → form `submitted` ET tout dialog ouvert se ferme (choix déclaré) ; sinon **valide mais sans effet** (paid no-op, §3.3)
- `confirm` / `dismiss` : **≡ en V1** (déclaré, m5) — ferment le dialog s'il est ouvert ; sinon no-op valide
- `none` (distracteur) : **toujours valide, jamais d'effet** — la classe « action légale inutile » que V0 n'avait pas

### 3.3 Invalides vs no-ops valides (distinction normative)

- **Invalide** (no-op + coût + flag `invalid`) : NAVIGATE non-adjacente, CLICK bouton non visible, TYPE field non-visible/rempli/hors-but, SELECT non-visible/déjà choisi.
- **No-op valide** (coût, flag `valid`, zéro effet) : distracteurs, submit prématuré, confirm/dismiss sans dialog.
Le modèle doit apprendre les DEUX classes. Taux publiés séparément (m7) : `observable_result` reste binaire (schéma inchangé) ; le taux de no-op valide est dérivé par l'évaluateur = P(valid ∧ state_hash inchangé).

### 3.4 Dialogues

Un dialog peut s'ouvrir en état initial (générateur) — pas de déclenchement dynamique en V1 (déclaré : simplification). Confirmer/dismiss le ferme.

## 4. Génération de layouts

- Graphe de views connecté (2-5 views, arbre + extra arêtes), nav_edges bidirectionnels.
- 1-2 forms par layout : chacun avec 2-4 fields + 1 select (2-4 options) + 1 bouton submit, tous sur une même view.
- Boutons libres : **≥3 distracteurs (kind none), non bornés** — le remplissage jusqu'à K cible (ci-dessous) les fait monter à 17-49 (G1 résolu : la plage 3-8 initiale était incompatible avec K 30-60) ; ± 1 confirm/dismiss si dialog initial.
- 0-1 dialog initial ouvert (30 % des layouts).
- Contrainte K : **30 ≤ K ≤ 60**, garantie par **remplissage de distracteurs** jusqu'à la cible tirée dans les bornes (M1 : les plages de base ne donnent que 11-34 — le générateur comble) ; rejets comptés si la structure de base seule déborde.
- Seed déterministe ; labels tirés du vocabulaire+alias ; **deux layouts isomorphes (widgets+FSM) peuvent avoir des labels totalement différents** (c'est le point §2.3).

## 5. Buts (prédicats fermés)

| Prédicat | Références | Satisfaction |
|---|---|---|
| `VIEW` | view | current_view == view |
| `SET` | field | field.filled == true |
| `CHOOSE` | option | son select.chosen == option |
| `SUBMITTED` | form | form.status == submitted |

**Argument TYPE (règle V1 révisée après smoke test)** : TYPE est **unaire** (remplissage binaire filled=true), validité goal-blind (visible + vide) — la règle initiale « uniquement le field du but » rendait les buts SUBMITTED insolvables (aucune action ne pouvait remplir les champs d'un formulaire visé). Le mécanisme de copie span/copy §13.3 de la spec générale est trivialement satisfait par le binding par référence ; la difficulté de liaison au but est portée par CHOOSE (bonne option parmi 2-4) et SUBMITTED (ordre de complétion). La génération de texte reste hors périmètre V1. **Leçon registre : toute contrainte de validité 'goal-conditionnée' doit être vérifiée contre la solvabilité de TOUS les prédicats de but avant figeage.**

## 6. Terminaison, coûts, évaluateur

Horizon 64 décisions. Chaque décision coûte 1 (STOP inclus — leçon V0 M1). STOP : terminal, succès ssi prédicat du but vrai sur l'état natif (évaluateur indépendant, jamais dans le chemin de décision). Timeout = échec même si but physiquement atteint. `L* = d* + 1`.

## 7. Oracle (FSM exact)

Espace d'états **énumérable** (M4 : status exclu car dérivé) : views × (filled ⊆ fields) × (chosen : select→option) × (dialog open) × (submitted ⊆ forms). Backward BFS multi-source depuis tous les états satisfaisant le but — structure identique à l'oracle V0 (successor explicite, invalides = self-loops exclus, no-ops valides = self-loops inclus mais jamais optimaux). `A*`, `d*`, `L*`, `reachable` — mêmes définitions. **Tests requis** : Bellman exhaustif sur petits layouts, différentiel env↔oracle (leçon T2), contrexemples ciblés (ordre de validation, distracteurs, dialogues, navigation multi-sauts).

## 8. Anti-fuite (héritage V0, adaptations)

Canaux §5.1 inchangés. Nouveaux points de vigilance : (a) `filled`/`chosen`/`status` sont des attributs d'État observables (légitimes) — mais leur valeur ne doit JAMAIS encoder la distance au but ; (b) les labels ne portent aucune information de rôle (test : permutation des labels entre widgets isofonctionnels ⇒ mêmes labels-candidats, mêmes optimums) ; (c) K ne doit pas corréler avec le but.

### 8b. Générateur (m8/m9)

Labels tirés uniformément **indépendamment du rôle** (indépendance par construction ; test statistique dans la suite). Échantillonnage des buts : cycle équilibré des 4 prédicats + relax compté après 1/3 des tentatives (leçon V0 M3), quota d*=0 pool-level train-only (leçon V0 M2), rejets par motif, bandes fixées par l'inventaire WS-B avant gel.

## 9. Livrables WS-A

`ucm/env/siw.py` (env), `ucm/env/siw_oracle.py` (oracle FSM), `tests/test_siw.py` + `tests/test_siw_oracle.py`, générateur `ucm/data/generate_siw.py` (seedé, rejets comptés, modes §7.4 équivalents si pertinents). Interface strictement §3.1 V0 pour que le harnais WS-B/WS-C se branche sans adaptation.
