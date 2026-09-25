# ERRATUM — 2026-09-23 — Écart « 83 vs 79 » des failures GATE-2 vs M3

**Statut** : erratum **append-only**, **non scellé**. Ce document ne modifie **aucun** artefact, rapport, log ou SHA existant.
**Auteur** : tagi-5 (audit QA/red-team). **Déclencheur** : constat du lead (83 vs 79) — audit arithmétique / pointage demandé.

## 1. Objets cités (lecture seule — SHA-256 externes)

| Objet | SHA-256 |
|---|---|
| `artifacts/gate2-confirmation/gate2-confirmation.json` | `e85f2775eed9be3d2a804daa642ed74dd9ceeff8e2adc68b0f115f068c8bcedc` |
| `artifacts/m3/m3-report.json` | `bdbca4adced0f6040317af92a818165968dbf4ab41f3ee389e2f334ccda75b30` |
| `artifacts/m3-failures-run.log` | `a045950725ace0a286b6f02147b242888217de06ac3e2ec744083ce189c01084` |

## 2. Faits établis

- **GATE-2 official** (`gate2-confirmation.json` → `official_test_g1`) : `n = 3160`, `success_rate = 0.975`
  ⇒ échecs = 3160 × (1 − 0.975) = **79**.
  RNG déclaré (`rng_discipline.official_g1_read`) : `mx.random.seed(20000 + seed)` par seed.
- **M3** (`m3/m3-report.json` → `failure_categorization`) : `total_failures = 83`
  (boucle 46 + action_invalide_repetee 36 + autre 1 = 83), `total_episodes = 3160`.
  RNG déclaré dans `ucm/eval/m3_failures.py` : `mx.random.seed(40_000 + s)`.
- **Taux explicites** : GATE-2 officiel **3081/3160 = 97,50 %** (échecs **79/3160 = 2,50 %**) ;
  M3 **3077/3160 ≈ 97,3734 %** (échecs **83/3160 ≈ 2,6266 %**). Le taux M3 de 97,37 % est
  donc **3077/3160**, **PAS** `83/3160` (83 est le nombre d'échecs M3, pas un taux).
- **Périmètre identique** : mêmes checkpoints GATE-2 (tous seeds), mêmes bras choisis
  (`chosen_arm_per_seed`) et **même ensemble d'épisodes de test**
  (`load_canon_episodes(CANON, split="test_g1")`). Deux différences **documentées** :
  (i) le **flux RNG d'évaluation** (`20000+s` vs `40_000+s`) ;
  (ii) le **harness de rollout** diffère — GATE-2 : `rollout(...)` → `run_episode(...)`
  (`ucm/eval/gate2_confirmation.py:47`, `ucm/eval/rollout.py:54`) ; M3 : `trace_episode(...)`
  (`ucm/eval/m3_failures.py:92`). L'attribution du delta (4 échecs) au seul RNG est
  **plausible au niveau agrégé**, mais **n'est pas une preuve contrefactuelle causale**
  sans équivalence complète des deux harness.

## 3. Pointage par seed (échecs)

| seed | bras | GATE-2 official | M3 re-run |
|---|---|---|---|
| 0 | greedy | 16 | 16 |
| 1 | greedy | 21 | 21 |
| 2 | **τ1.0** | **7** | **11** |
| 3 | greedy | 20 | 20 |
| 4 | greedy | 15 | 15 |
| **total** | | **79** | **83** |

Source par-seed official : `per_seed_test_g1[s].summary` (n = 632/seed, échecs = 632 × (1 − succès)).
Source par-seed M3 : `artifacts/m3-failures-run.log` (« seed s: failures … »).

## 4. Conclusion

- L'écart **83 − 79 = 4** est **localisé à la seed 2 (bras τ1.0)** : divergence
  stochastique de re-run. Les deux évaluations utilisent des **flux RNG différents**
  (20000+s vs 40000+s) et deux **harness distincts** (`run_episode` vs `trace_episode`) ⇒
  **les épisodes de test sont les MÊMES**, seules les **trajectoires / actions / outcomes
  stochastiques** peuvent différer ⇒ comptage différent. Aucune erreur arithmétique :
  16+21+7+20+15 = **79** ; 16+21+11+20+15 = **83**. L'attribution du delta au seul RNG reste
  **plausible (agrégée), pas causale**.
- **83/3160 n'est PAS le taux du GATE-2 official.** Le taux officiel d'échecs est
  **79/3160 = 2,50 %** (succès **3081/3160 = 97,50 %**) ; le re-run M3 donne
  **83/3160 ≈ 2,6266 %** (succès **3077/3160 ≈ 97,3734 %**).
- La catégorisation M3 (boucle 46 / action invalide répétée 36 / autre 1) reste valable **pour le
  re-run 40000+s**, pas pour le comptage officiel.

## 5. Recommandations

1. Citer le taux officiel depuis `artifacts/gate2-confirmation/gate2-confirmation.json` (**79/3160**).
2. Étiqueter tout usage de `m3/m3-report.json` comme « re-run M3, RNG `40000+s`, divergence
   stochastique seed 2 (τ1.0 : 11 vs 7) » — ne jamais le présenter comme le taux GATE-2.
3. Discipline append-only : aucune réécriture des artefacts/rapports/logs ; toute correction
   ultérieure = **nouveau document daté**.
4. Toute citation de l'écart doit nommer **les deux différences** (RNG + harness) et éviter
   l'attribution causale du delta (4 échecs) au seul RNG.
