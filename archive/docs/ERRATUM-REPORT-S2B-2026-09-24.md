# ERRATUM — rapport S2b final (`af752d2`) — 4 corrections (revue indépendante, 24/09 17:52)

**Statut : erratum append-only.** Le rapport `docs/REPORT-S2B-FINAL.md` (blob `56d77a7e…`, PASS pinné) reste intact ; les corrections suivantes s'y superposent (revue tagi-review, vérifiées lead). Le verdict **KILL du CIV-simple reste inchangé** — les corrections portent sur la portée, les endpoints manquants, la force des claims secondaires et la puissance.

## 1. Portée du KILL reformulée

`loss_first` CIV = 2,5-5,7 vs B144 = 0,006-0,08 : le GRU×32+LayerNorm **non-entraîné, sans init chemin-identité**, détruit la représentation du canon avant la tête figée. Le KILL porte donc sur **« refine sans chemin identité au-dessus d'une tête figée »** — il **ne se généralise pas** à « aucune itération légère » (un refine avec init identité, ou une tête adaptée conjointement, reste à tester si la recherche profondeur le justifie).

## 2. G1′ préenregistré mais NON EXÉCUTÉ (+ secondaires manquants)

Le protocole préenregistrait G1′ (rétention d*≤8 ≤ 2 pp) et les bandes secondaires [16,24]/[13,24] — l'éval n'a couvert que [13,15]. **Ce n'est pas « non évaluable », c'est non exécuté** : les checkpoints étant persistés (`ckpts-run3`), l'évaluation complémentaire **sera exécutée depuis les checkpoints** (pas de ré-entraînement) et publiée en annexe ; le KILL tient sur le primaire seul.

## 3. Effet données : NON CONCLUANT (sur-interprétation corrigée)

B144 : +6,4 pp, p unilatéral 0,094 (n=5) ; CIV : +1,8 pp, p=0,41. À n=5, **aucune conclusion de supériorité de l'oracle** — la formulation correcte est **NON CONCLUANT** (effet possible, non établi ; l'inversion seed 104 documentée). Toute utilisation DAgger exigera l'expérience dédiée avec sa règle préenregistrée.

## 4. Puissance : MDE réel ≈ 18,5 pp (erratum du gate)

Le gate (v04) supposait l'effet seed ≈ 0 (SD inter-seeds V0 G5 = 0,008) ; or **B144 varie de 5,9 pp entre seeds** sur cette strate — les décisions intra-seed ne sont pas indépendantes et le MDE effectif ≈ **18,5 pp**, supérieur au 15,8 gelé. Conséquence : le FAIL était **encore plus net** que le prédicat ne l'exigeait (−7,7 vs un seuil réel ~18,5) ; le KILL n'en est pas affecté — mais la méthode de gate est corrigée pour l'avenir : **la SD inter-seeds doit être mesurée au niveau de base réel avant tout gel de seuil**.

## 5. Intégrité (red flag 5, résolu)

Les artefacts run3 (20 cellules, 20 ckpts, 5 summaries) étaient **non committés** au moment de la revue — committés depuis (`7dcc545`, shas des summaries : `a6950341…`/`e810091a…`/`7f871d56…`/`0928ce37…`/`3bc62c6a…`). Le rapport final référence désormais ces commits.

## Règle §5.3 (figée au registre, suite au red flag P2-1)

**Un seuil gelé ne se baisse jamais pour satisfaire une MDE mesurée** : si la puissance est insuffisante, on augmente n ou le nombre de seeds AVANT le gel ; l'alignement Δ_min↔MDE du gate v04 (15,8) est rétrospectivement qualifié de **précédent dangereux** — la défense (le triage avait préenregistré +15-20 pts avant toute mesure) l'excute rétrospectivement mais ne la reproduira plus. **La règle entre au registre des leçons §6.**
