# PROTOCOLE S2b — factorielle données-vs-calcul (pré-enregistrement, 24 septembre 2026)

**Statut : PRÉENREGISTRÉ avant exécution.** Complète la décision troisième revue (S2b avant P1) et le triage autonomie (`fc026a3`). Dépendances auditées : gate de sensibilité v04 (`a6fd6c1`, strate admissible, Δ_min 15,8 pp gelé), harnais factorielle + CIV normatif (`a2a1af8`+`0112b8a`+`8702fe0`+`e4a8e06`, gardes figées, chemin TGK fermé), calibration (`472348c`, p95 PASS 14,08 ms CPU / 5,99 GPU ; coût rec 11,9× p50 ; RSS 9,7 Go). Append-only ; toute divergence = erratum nouveau chemin.

## 1. Question et estimand

La question P1 : sur les tâches profondes TGK où le one-pass plafonne, **le calcul itératif (CIV T=32) bat-il le one-pass (B144) à budget de données égal** — et la source (oracle vs récupération à fraction contrôlée) interagit-elle avec le modèle ? Factorielle 2×2 : (B144 | CIV-T32) × (oracle | recovery@f).

**Endpoint primaire** : succès fermé (§9.4, STOP natif, horizon 64) sur la **strate G4 d\*∈[13,15]** (n=78 épisodes step-0 — strate gelée au gate v04, Δ_min = 15,8 pp). **Secondaires descriptifs** : bandes d*∈[16,24] (n=25) et [13,24] agrégée (n=103) ; **G1′ rétention** : perte sur bandes superficielles (d*≤8, test split) ≤ 2 pp ; sondes (champ récepteur par sauts, Spearman échec×d*, log-ratio d'échecs robuste au plafond).

## 2. Cellules, données, budgets

4 cellules × **5 seeds** (0-4, diagnostique) : {B144-oracle, B144-recovery, CIV-oracle, CIV-recovery}. Données : TGK train (5 492 records) — **même volume pour toutes les cellules** ; recovery@**f=0,3** REMPLACE (déviation d'une action non-optimale + re-solve, unreachable→garde oracle comptée — `recovery_mix` audité). Entraînement : **updates=2000, batch=64, CPU** ; B144 = canon gelé + adapté (subtree pour CIV : refine seul, base jamais passée à l'optimiseur, sha vérifié avant/après dans l'artefact) ; T=32 effectif prouvé mécaniquement (compteur). Éval : `eval_seed=50000+seed`, épisodes identiques toutes cellules.

## 3. Verdict préenregistré (PASS/KILL de la récurrence pour P1)

- **PASS récurrence (prédicat unique)** : point (moyenne des seeds) de CIV(T=32)−B144 ≥ **+15,8 pp** **ET** IC_low (5e percentile, bootstrap seed-cluster 10 000 seed 20260923) > 0, sur la strate primaire — **ET** G1′ perte ≤ 2 pp. p95 ≤ 20 ms : **déjà PASS en calibration** (reporté, non re-mesuré sauf si le code change).
- **KILL** : différence < seuil OU G1′ > 2 pp ⇒ la récurrence n'est pas retenue pour P1 ; publication honnête, P1 reste one-pass.
- La source (recovery) est **diagnostique** : rapportée par cellule, aucune règle de promotion dans CE run (DAgger = expérience séparée avec sa règle préenregistrée).
- Si PASS ⇒ la supervision intermédiaire (P1 récurrent) s'ouvre AVEC les mêmes gardes.
- **Asymétrie documentée** : le bras B144 adapte TOUT le canon ; le CIV n'adapte que le refine (base gelée) ⇒ le test est **conservateur pour la récurrence** — un KILL n'exclut pas qu'une base adaptée + refine ferait mieux ; il tranche la question posée (refine au-dessus d'un canon figé).
- **Seuil conservateur** : le MDE 15,8 pp fut calculé à n=78 épisodes ; l'exécution mesure 5 seeds × 78 = 390 décisions par cellule ⇒ la puissance réelle est ≥ au calcul du gate.

## 4. Dimensionnement et fenêtre (formule committée, leçon annonce/bits)

`fenêtre ≈ Σ_cellules(updates × per-update_bras)/workers_bras + eval` avec `workers_rec ≤ RAM_libre/9,7 Go` (**borne RSS**), base **p50** (11,9×) : 10 cellules rec × 2000 × 2,183 s / 2 workers ≈ **6,1 h** ; 10 cellules B144 × 2000 × 0,183 s / 4 workers ≈ 0,26 h ; eval ≈ 0,1 h ⇒ **~6,5 h une nuit calme** (check de charge immédiat avant T0, leçon OPS `211481c`). Disque ~80 Mo checkpoints. GPU : option documentée (p95 5,99 ms) mais le protocole fige **CPU** (cohérence avec V1-bis).

## 5. Intégrité

Persistance O_EXCL par cellule + summary (déjà en place) ; budgets appariés assertés ; gardes (gel base, T effectif) dans chaque artefact ; aucune donnée scellée SIW consommée (le canon TGK est une source PUBLIÉE jamais mutée — discipline never-mutate, pas one-read) ; rapport final préenregistré = ce document + raws. **Deux verts avant tout usage confirmatoire ultérieur ; ce run est diagnostique — ses résultats orientent P1, ils ne scellent rien.**
