# ERRATUM — Audit distribution R* : estimands ESS (Simpson vs importance) et κ (label vs prédicat)

- **Nature** : erratum **séparé** du rapport `docs/AUDIT-DISTRIBUTION-RSTAR-SIGNATURE.md` (commit `0ff898b`) — le rapport historique **n'est pas modifié** (toute correction = nouveau document, jamais d'overwrite).
- **Déclencheur** : audit lead 2026-09-23 11:10 — deux **risques d'estimand** pour le futur gate.
- **Portée** : DEV-only, aucun scellé rouvert ; aucun GO.

## 1. `ESS = 1/Σp²` = indice de diversité de **Simpson** — PAS l'ESS d'importance-reweighting

- Dans le rapport `0ff898b`, la quantité notée « ESS » (`1/Σ_s P(s)²`, états ou cellules) est le **nombre effectif de Simpson** (indice de diversité). **Renommage proposé pour tout usage futur : `N_eff_Simpson`.**
- **Ne pas comparer** les valeurs 27.3 (cellules) ou 48.2 (états) à un critère `ESS ≥ 0.8·k` : ce critère vise l'**ESS d'importance-reweighting**, défini par :
  - `w_i = p_ref(cell_i) / q_train(cell_i)` (poids d'importance de la référence sur la distribution d'entraînement) ;
  - `ESS_importance = (Σ_i w_i)² / (Σ_i w_i²)`, sommé sur les cellules **supportées** (`q_train(cell_i) > 0`) ;
  - **support FAIL** : toute cellule avec `p_ref(cell_i) > 0` et `q_train(cell_i) = 0` → à **compter et publier séparément** (avec la masse de référence non couverte), pas à ignorer.
  - Le `k` du critère éventuel = la **taille d'échantillon par tranche de production** (fichier/budget), pas le nombre de cellules.
- **Exemple DEV descriptif** (cellules σ×pred×d\*, R* 40 ép. seed 42 vs réf MC 1200 seed 999) : 37 cellules train / 39 réf ; **2 cellules en support FAIL** (`((0,0,1,0),SUBMITTED,6)`, `((1,0,0,0),SUBMITTED,2)`), **masse réf ≈ 1.3 %** ; `ESS_importance` (cellules supportées) = **26.7** ; `N_eff_Simpson` (train) = **27.3**. Les deux métriques sont proches ici mais **ne sont pas interchangeables**.

## 2. `κ` : type d'action du **label** (reviewer) vs prédicat du **but** (rapport)

- Le rapport `0ff898b` définit la cellule `σ×κ×d*` avec **κ = prédicat du but** (VIEW/SET/CHOOSE/SUBMITTED). Le reviewer visait **κ = type d'action du LABEL** (classe d'action supervisée : NAVIGATE/CLICK/TYPE/SELECT/STOP — selon le vocabulaire exact à figer).
- Conséquences à documenter pour tout protocole/pilote futur :
  - la TV `σ×pred×d*` **peut manquer** un décalage de distribution **d'actions** (labels) entre entraînement et référence ;
  - la TV au niveau **état** utilise une clé **qui n'inclut pas le but** (les états sont agrégés tous buts confondus) → un décalage par but peut être masqué.
- **À définir dans le nouveau protocole/pilote** : `κ_action` (type d'action du label) **et** une métrique de **contraste but/label** (p. ex. TV des cellules `σ×κ_action×d*` **et** TV par prédicat de but / par classe d'action), publiées côte à côte ; préciser le vocabulaire d'actions du label à figer.

## 3. Statut

- Le rapport `0ff898b` reste **descriptif** et valable pour ses chiffres (TV multi-seeds, préfixes, incertitudes) ; **aucun seuil n'était promis**.
- Le **pilote de puissance** devra fixer : `κ_action` vs prédicat, la métrique de contraste but/label, l'`ESS_importance` (formule ci-dessus) et la taille `k` par tranche, avant tout gate.
- Blocages séparés inchangés : test **full-main**, **E2E writer→reader→adapter**, `O_EXCL` (crash-atomicité). **HOLD test2** ; aucun scellé.

## 4. Signature

tagi-5 — auditeur indépendant. Erratum append-only ; aucun scellé rouvert ; aucune promesse de seuil.
