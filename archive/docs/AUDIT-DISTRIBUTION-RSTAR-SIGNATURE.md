# AUDIT — Distribution R* rev0.7 : signature σ×κ×d*, TV/ESS et incertitude

- **Demandé par** : lead (2026-09-23 11:07), après l'audit DEV initial.
- **Auteur** : tagi-5 (auditeur indépendant).
- **Portée** : **DEV uniquement, aucun scellé**. Code : R* rev0.7 `5190434` (`ucm/data/siw_pipeline.py::generate_adaptation_episodes_rstar`), runner fail-closed `ecd8d10` (audit instrument séparé, non validé). Ce rapport est **descriptif** : il ne fixe **aucun seuil** et ne promet **aucun** critère « TV ≤ seuil » avant un pilote de puissance.

## 1. Fixture et reproductibilité

- **Layouts** : `generate_siw_layout(rng, rng.randint(2,4), with_dialog=rng.random()<0.5, profile="small")`, `rng=random.Random(20261010)`, 6 layouts — hashes : `39798f4b91ca70bf`, `6032a0cb179caf42`, `8cf65d497737c4bc`, `0598d283f7da3655`, `ab738a6735e7dced`, `cfda87693bc245f3`.
- **R\*** : `generate_adaptation_episodes_rstar(layouts, seed∈{42,43,44}, n_episodes=40, d0_band=(2,8))` (quota exact 10/prédicat).
- **Références Monte-Carlo indépendantes** : même sampler, `seed∈{999,1000,1001}`, `n_episodes=1200` (4617/4590/4612 couples, 130 états chacune).
- Commande type : génération puis agrégation des distributions (états et cellules) ; aucun hash de la cellule scellée n'est lu.

## 2. Définitions exactes

- **État** : clé canonique `(layout_hash, view, filled_trié, chosen_trié, dialog_open, submitted_trié)`.
- **Cellule σ×κ×d\*** : **σ** = signature de features `(1[filled>0], 1[chosen>0], dialog_open, 1[submitted>0])` ; **κ** = **prédicat du but** (VIEW/SET/CHOOSE/SUBMITTED) ; **d\*** = distance oracle de l'état (0..6). Cellule = (σ, κ, d\*).
- **TV (total variation)** : `TV(P,Q) = ½ · Σ_{s ∈ supp(P)∪supp(Q)} |P(s) − Q(s)|` — distributions **normalisées** (Σ=1) sur le **support union** ; au niveau **état** ou **cellule** selon le cas (précisé à chaque ligne).
- **ESS** : `ESS(P) = (Σ_s P(s))² / Σ_s P(s)² = 1 / Σ_s P(s)²` (nombre effectif d'états/cellules ; Simpson inverse).
- **Dénominateurs** : R\* = **couples pondérés par la multiplicité** (154/154/158 pour 40 ép. seeds 42/43/44) ; référence = couples MC (4617/4590/4612). Les cellules agrègent les états par signature×prédicat×d\*.

## 3. Résultats (état)

| R\* seed | couples | états | ESS | TV vs ref 999 | TV vs ref 1000 | TV vs ref 1001 |
|---|---|---|---|---|---|---|
| 42 | 154 | 71 | 48.2 | 0.318 | 0.330 | 0.307 |
| 43 | 154 | 68 | 43.3 | 0.351 | 0.335 | 0.357 |
| 44 | 158 | 78 | 51.4 | 0.293 | 0.278 | 0.302 |

**Préfixes imbriqués (seed 42, moyenne [min ; max] sur les 3 références)** : 12 ép. TV **0.472 [0.469 ; 0.475]** ESS 31.2 · 24 ép. **0.415 [0.407 ; 0.420]** ESS 37.1 · 36 ép. **0.360 [0.349 ; 0.369]** ESS 40.0 · 40 ép. **0.318 [0.306 ; 0.330]** ESS 48.2.

## 4. Résultats (cellules σ×κ×d\*)

- R\* (seed 42, 40 ép.) : **37 cellules**, ESS **27.3** ; référence : 39 cellules ; **TV cellule = 0.120**.
- Top cellules (part) : R\* `((1,1,0,1), SUBMITTED, 0)` 6.5 % (ref 6.5 %) ; `((0,0,0,0), VIEW, {0,1,2})` **5.8 % chacune** vs ref **4.2 %** → **sur-représentation VIEW d\*0/1/2** d'environ +1.6 pp par cellule ; la ref a `((0,0,1,0), CHOOSE, 2)` 3.4 % absente du top-5 R\*.

## 5. Incertitude et limites (aucune exclusion de biais)

1. **Bruit de la référence MC** (1200 ép.) : petite sur le TV état (±0.01 entre les 3 références) mais **non nulle** — la « vérité » n'est pas connue analytiquement.
2. **Bruit inter-seed R\*** : **dominant** (TV état 0.291–0.348 sur 3 seeds à 40 ép.) → à cette taille, le TV est un estimateur bruité ; la décroissance 0.472→0.318 (préfixes imbriqués) est **compatible** avec la convergence mais **ne prouve pas l'absence de biais** (pas de bande de confiance multi-seed par préfixe ; un seul fixture de 6 layouts, sans réplication par layout).
3. **Cellulaire** : TV 0.120 (cellules plus grossières → plus proches) mais la sur-représentation VIEW×{0,1,2} est un candidat de biais à confirmer/infirmer par le pilote (échantillon 40 ép. vs 1200).
4. **Aucun seuil** n'est fixé ici : le TV/ESS cible (par fichier/budget k) doit être **pré-enregistré après un pilote de puissance** (répliquer plusieurs seeds R\* × plusieurs références × fixtures), pas promis a priori.

## 6. Blocages séparés (rappel)

- Test **audit-hook full-main** (`runner.main` : test/couples/pattern + intention fsync) **absent** → instrument non validé.
- **E2E writer→reader→adapter** par tranche non prouvé.
- Freeze : `O_EXCL` documenté, crash-atomicité non revendiquée.
- **HOLD test2** ; aucun scellé ouvert.

## 7. Signature

tagi-5 — auditeur indépendant. Chiffres reproductibles par les seeds/hashes ci-dessus ; audit DEV-only, aucune lecture scellée ; aucune promesse de seuil.
