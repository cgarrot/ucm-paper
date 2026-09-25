# AUDIT — Support V1-bis DEV : closure forward et couples impossibles

- **Demandé par** : lead (2026-09-23 10:28), suite au finding `m_mudu8hd2_85870190`.
- **Auteur** : tagi-5 (auditeur QA/red-team indépendant).
- **Portée** : **v2 committé `2c98614`** (audit dans un **worktree détaché propre**), fixtures DEV générées en mémoire — **aucun fichier scellé lu** (ni manifest, ni inventaire, ni couples, ni épisodes test). Aucun accès à `split-manifest-SIW-v1.json` pendant cet audit.
- **Objet** : vérifier que le générateur v2 sélectionne des couples via le **produit cartésien** (`enumerate_states`) filtrés par `SIWOracle.reachable` = **goal-reachable** (BFS inverse depuis les états satisfaisant le but) et **non** par la forward-reachability depuis le support initial — d'où des états **physiquement impossibles**.

## 1. Fixture et méthode

- **Layouts** : `generate_siw_layout(rng, rng.randint(2,4), with_dialog=rng.random()<0.5, profile="small")`, `rng = random.Random(20261010)`, 10 layouts (hashes complets ci-dessous).
- **Couples** : `generate_multistep_adaptation_couples([lay], seed=42+i, couples_per_layout=16)` → 17 couples/layout, **170 couples** (dénominateur).
- **Closure forward** : BFS depuis les états initiaux **admissibles** (`views × dialog_open ∈ {False, True si dialog présent}`, `filled/chosen/submitted` vides), transitions par `_successor(lay, st, candidate)` sur **tous** les candidats de l'oracle ; ensemble des clés d'état atteintes.
- **Critère** : un couple est **impossible** si sa `state_key` n'appartient pas à la closure forward.

## 2. Résultat (v2 committé `2c98614`)

| layout_hash | closure | couples | hors closure |
|---|---|---|---|
| `39798f4b91ca70bf` | 78 | 17 | **8** |
| `6032a0cb179caf42` | 56 | 17 | **13** |
| `8cf65d497737c4bc` | 104 | 17 | **8** |
| `0598d283f7da3655` | 104 | 17 | **6** |
| `ab738a6735e7dced` | 78 | 17 | **6** |
| `cfda87693bc245f3` | 104 | 17 | **7** |
| `ccb13755008c6d23` | 28 | 17 | **9** |
| `75337ced3527b54c` | 78 | 17 | **8** |
| `22f2c30f53be72f8` | 56 | 17 | **11** |
| `ab7551fb33e6c87c` | 78 | 17 | **4** |

**Total : 80/170 couples HORS closure = 47.1 %** — par prédicat : **VIEW 25, CHOOSE 24, SET 17, SUBMITTED 14** (tous touchés) ; par layout 4–13/17 (24–76 %).

**Exemples d'états impossibles** (Réels, v2) :
1. `39798f4b91ca70bf` · VIEW · `d*=0` · `["vw2", ["fd0_0","fd0_1"], [["sl0","op0_1"]], true, ["fm0"]]` · source `d0_stop` ;
2. `39798f4b91ca70bf` · SET · `d*=0` · `["vw0", ["fd0_0","fd0_1"], [], false, ["fm0"]]` · source `d0_stop` — formulaire soumis dans une vue où la soumission n'est pas atteignable/configurable ;
3. `39798f4b91ca70bf` · SUBMITTED · `d*=0` · `["vw2", ["fd0_0"], [], true, ["fm0"]]` · source `d0_stop` — **formulaire soumis avec UN SEUL field rempli** (formulaire à 2 fields) = état **physiquement impossible**.

**Cause** : `enumerate_states` = produit cartésien syntaxique (`siw_oracle.py:19`) ; `SIWOracle.reachable` = `state.key() in self._dist` où `_dist` est la BFS **inverse** depuis les états satisfaisant le but (`siw_oracle.py:232`) → goal-reachable ≠ forward-reachable. Le `submitted` histogramme (56) ne prouve donc **pas** un support physique réel.

## 3. Statut du correctif (v3, working tree — NON commité à la date de cet audit)

- Le working tree contient une v3 (`forward_reachable_closure`, `assert couples ⊆ closure`, `rejected_not_in_closure`, `closure_sizes`) ; sur la **même fixture**, la v3 produit **0/170 hors closure**.
- **Cet audit ne vaut PAS validation de la v3** : elle doit être **committée**, puis re-auditée sur **commit propre** (closure forward, support initial/STOP non-vacuë, writer E2E), conformément à la règle « pas de revue sur working tree en mouvement ». Aucun scellé ouvert pour cet audit.

## 4. Provenance (séparée, à ne pas confondre avec mes lectures)

- **Mes lectures** : DEV générées en mémoire uniquement ; **aucun FD scellé** de ma part.
- **Déclaration tagi-2 (pool200, `seal 1a37eddc`)** : 1 FD post-HOLD sur `artifacts/split-manifest-SIW-v1.json` (métadonnées : hashes de pools) + 1 inventaire DEV non scellé, zéro couples/test/layouts → **ruling tagi-5 : metadata-only, nouvelle ouverture de scellé métadonnées à enregistrer séparément (D9)**, one-read du test **intact**, aucun outcome.

## 5. Verdict

- **Faille v2 CONFIRMÉE** : 47.1 % de couples hors closure forward sur l'échantillon (tous prédicats) — **NO-GO DONNÉES maintenu pour la v2**.
- **Correctif v3 requis et à re-auditer sur commit propre** (closure forward + assert `couples ⊆ closure` + stats de couverture **sur la closure** + ratio initial pré-enregistré).
- **Aucun scellé ouvert** par cet audit ; provenance metadata séparée et déclarée.

## 6. Signature

tagi-5 — auditeur indépendant. Preuve DEV reproductible (seeds et hashes ci-dessus) ; audit sur v2 committé dans un worktree détaché propre (`2c98614`), v3 en attente de commit.
