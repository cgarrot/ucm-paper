# FREEZE V1a — Harnais de transfert §12 (figé AVANT toute inspection des cibles SIW)

**Figé par :** tagi-3 (WS-C), sur directive tagi-1 17:44 (m_mucuheyw) · **avant réception des fixtures SIW.**
**Spéc de référence :** §12 (transfert), §12.6 (SIW). Toute déviation → registre PLAN + ici.

## 1. Architecture (figée — aucun choix après inspection cible)

**Cœur : GNNB d=144 (B144, 694 513 params au canon), INVARIANTE.**
Adaptation SIW = vocabulaire d'entités/relation/actions fermé propre à SIW,
embeddings de types NOUVEAUX initialisés frais dans TOUS les bras (§12.4),
même forme d'encodeur/scorer :

    node encoder: MLP[D_in_SIW + n_tags → d → d] (nouveaux poids, d=144)
    3 blocs message-passing distincts (3d→d→d / 2d→d→d), résidus + LayerNorm
    pooling/scorer: identiques à B144 (but par prédicat+refs, tags de rôle)

**Échanges gelés du canon (bras pré-entraîné) :** les 3 blocs message-passing
+ tête de scorer (poids transférés). **Frais dans tous les bras :** embeddings
de types SIW (entités/relations/actions/prédicats) + encodeur de nœuds (car
D_in change). Le contrôle §12.5 (reset tête seule / reset encodeur / gel vs
fine-tune) est une analyse SECONDAIRE pré-déclarée, pas le bras primaire.

## 2. Bras (3, gelés)

1. **scratch** — B144-arch initialisée de zéro, entraînée au budget k uniquement.
2. **pretrained-TGK** — cœur transféré des checkpoints canon 9a19d8f4
   (gate2c-B144-s{seed}, 5 seeds), fine-tune complet au budget k.
3. **control-nontarget** (design VALIDÉ lead 17:48, trois conditions) —
   même cœur transféré, source = canon TGK à supervision NON-INFORMATIVE:
   (1) RÈGLE EXACTE (seed-contrôlée, ucm/v1/control_source.py): pour chaque
   record canon, le label devient UN candidat tiré UNIFORMÉMENT parmi les
   actions PHYSIQUEMENT VALIDES de l'état (préconditions observables,
   valid_actions — STOP inclus si but satisfait); rng seedé par hash du
   couple → bit-reproductible, indépendant de l'ordre d'appel;
   (2) PUBLICATION: le contrôle partage la distribution d'entrée EXACTE
   (géométrie TGK, budget source identique) — lectures informatives:
   contrôle ≈ pré-entraîné ⇒ gain = exposition/optimisation (§12.5
   'accélération d'optimisation'); contrôle ≈ scratch ⇒ gain = décisionnel;
   les deux vont au rapport;
   (3) REGISTRE: le contrôle partage la géométrie PAR DESIGN (spec §12.2
   l'autorise, 'sans garantir une absence absolue de structure partagée') —
   l'interprétation en tiendra compte; une source structurellement
   différente reste une extension possible en run figé neuf si le contrôle
   s'avère trop fort/faible (décision sur résultats).

## 3. Budgets & couples uniques

- Unité = **couple (layout canonique, état physique, but)** UNIQUE (hash
  contenu). k ∈ {0, 100, 500, 2000, 10000}.
- **Sous-ensembles IMBRIQUÉS, IDENTIQUES entre bras** : ordre de sélection =
  hash du couple (seed 0), découpé en tranches cumulatives ; le même couple
  index i est dans tout budget k ≥ rang(i). Aucun alias/permutation ne crée
  de nouvel exemple (§12.3).
- Layouts d'adaptation DISJOINTS des layouts test (scellés par WS-B SIW).
- Fine-tune : 2 000 updates cible, batch 64, AdamW 3e-4/wd 1e-4/clip 1.0,
  FP32, checkpoint FINAL à budget fixe, SÉLECTION INTERDITE sur perf cible.

## 4. Critère primaire (figé, conforme directive)

- **k\* = 500** · succès boucle fermée (STOP vérifié par évaluateur indépendant,
  horizon SIW) · pré-entraîné(TGK) − scratch ≥ 5 points · IC95 apparié
  hiérarchique (seeds → layouts → tâches) > 0 · 5 seeds/bras.
- Secondaires : AULC succès-fermé vs log(1+k) sur grille commune ; seuil 80 %
  censuré « >10000 » si non atteint ; **couverture atteinte en couples uniques
  par bras/budget publiée à côté du succès** (un négatif à k*=500 avec
  couverture ≈ 0 est ININTERPRÉTABLE — rapporté comme tel).
- Une LECTURE du test scellé. Toute interaction cible (sélection, calibration,
  débogage) : COMPTÉE et interdite sur le test.

## 5. Interactions cibles — registre

Toute requête vers données/env cible SIW au-delà de la lecture unique
s'inscrit dans `artifacts/v1/interactions-log.json` (horodatage, motif, bras).
Le compteur fait partie de l'artefact final.

### 5ter. Bras SECONDAIRE déclaré 'control-validity-only' (rulings 19:21→19:31)

REVERT du ruling 19:21 (arbitrage tagi-5, ruling final lead 19:31): le bras
contrôle de RÉFÉRENCE reçoit la slice cible INFORMATIVE identique aux trois
bras — il diffère en SOURCE uniquement (checkpoint TGK non-informatif,
validé au chargement par load_control_records). C'est la lecture originale
§12.2: 'contrôle ≈ pré-entraîné ⇒ exposition/optimisation; contrôle ≈ scratch
⇒ décisionnel' exige la même cible entre bras.

derive_control_records_siw (mêmes couples/ordre/budgets, labels uniformes
parmi les SIW-VALIDES) est conservé comme bras SECONDAIRE DÉCLARÉ
'control-validity-only' (flag --secondary-validity-only, run optionnel,
JAMAIS la comparaison de référence; rapporté séparément).

## 5bis. Variante déclarée du canal supervision (ruling lead 19:15)

Pour source=control uniquement: supervision = {optimal_actions, reachable=True,
**d_star=None**, control_note} — inverse exact de la règle V0 (d_star entier),
jamais mixte. Validée AU CHARGEMENT par `load_control_records` (règle locale) ;
la validation globale V0 n'est PAS relâchée. Les checkpoints contrôle existants
sont insensibles à la variante (d_star n'entre jamais dans la loss — le
fine-tune ne lit que optimal_actions).

## 6. Règle d'arrêt pré-annoncée (lead 17:44)

Si le pré-entraînement graphique n'aide pas SIW : résultat MAJEUR publié
comme tel ; ambition « contrôle transférable » revue avant tout Web réel.
La revendication sera nommée **« généralisation inter-layout à couverture
partielle »**.

## 7. Enveloppe

24 h accélérateur V1 (journalisées par run dans artifacts/v1/).
