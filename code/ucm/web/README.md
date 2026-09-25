# ucm/web — Compilateur web v2 (page HTML → policy_input UCM)

Étend la sonde v1 (`compiler_probe.py`, commit 90a303a — lecture seule,
conservé intact) en traitant les 3 blocages documentés et en couvrant les
LIENS. Sortie: **policy_input STRICTEMENT typé JSON** (schéma SIW
`ucm/env/siw.py::observe`, adapté au web — le modèle consommateur n'est PAS
la responsabilité de ce module).

## Modules

| Fichier | Rôle |
|---|---|
| `vocabulary.py` | Vocabulaire web/2.0 **fermé** + `validate_policy_input` (validation stricte, lève `PolicyInputError`) |
| `compiler_v2.py` | `compile_page(html, url)` → `(policy_input, stats)`; `compile_pages()` (4 pages sonde); `build_report()`; `write_artifacts()`; `python -m ucm.web.compiler_v2` régénère tout |
| `make_snapshots.py` | Snapshots statiques déterministes des 4 pages (hors-ligne, calibrés sur les compteurs v1 — voir ci-dessous) |
| `compiler_probe.py` | SONDE v1 (90a303a) — non modifié, réutilisé en lecture seule pour rejouer les définitions v1 (`recompute_v1_on_pages`) |

## Blocages v1 → traitement v2

1. **Vocabulaire fermé TGK** (`'view'` rejeté) → vocabulaire web/2.0 propre
   (`view, form, field, button, select, option, link` + actions
   `TYPE, CLICK, SELECT, NAVIGATE, STOP`), validation systématique de toute
   sortie. La **largeur de collate (D_IN 21 vs 7)** reste hors périmètre:
   v2 produit du JSON typé, la tensorisation appartient au consommateur.
2. **Ids DOM dupliqués** (`fname` ×2) → dédup **déterministe**: 1ʳᵉ occurrence
   garde l'id nu, suivantes suffixées `_2`, `_3`… (`fname → fname_2`);
   renommages tracés dans `stats["dedup_renames"]`.
3. **Représentation des liens** (476/512 actionnables non couverts) → chaque
   `<a href>` = entité `link` avec `href_raw / href_resolved / href_kind /
   text` + relation `on_view` (+ `part_of` si imbriqué dans un form) +
   candidat **NAVIGATE**. Résolution via `urljoin`:
   `relative | site_root_relative | absolute | scheme_relative | anchor |
   same_page | other_scheme | unresolvable` (sans URL de page).

## Définitions d'actionnable — EXACTEMENT v1 (90a303a)

Dénominateur (identique sonde v1, pour une comparaison valide avec la
baseline 5.86 %): inputs `type∈{text,email,password,tel,number,search,url,
date}` + `textarea` + `button` (tag ou `input type∈{submit,button,reset}`) +
`select` + `<a>` avec href **présent et non vide** (sémantique véridique v1
`a.get("href")` — les `<a>` sans href ou `href=""` ne comptent pas). Les
`<option>` reçoivent des candidats
SELECT (couverts, non actionnables — comme v1). Un `select` est couvert dès
qu'une de ses options l'est.

## Décisions (v2)

- **Éléments non actionnables au sens v1** (inputs `radio/checkbox/hidden/
  file/sans type`, `<a>` sans href, `<label>`): **ni entité ni candidat**,
  comptés dans `stats["non_actionable_skipped"]`. Choix de stricte
  comparabilité v1/v2 (extension future possible: SETCHECK/UPLOAD).
- **Goal placeholder** `{"predicate": "VIEW", "args": {"view": "page"}}`
  (hérité v1; le but réel est fixé par l'orchestrateur en aval).
- **STOP** unique, en dernier candidat (arg `None`), comme v1.
- **Ordre déterministe**: entités/candidats en ordre document → sortie
  reproductible octet-à-octet (SHA-256 dans le rapport).
- **Label**: `attrs["label"]` peuplé uniquement via `<label for="id">`
  (les labels enveloppants sans `for` ne sont pas associés).

## Pages de la sonde (snapshots hors-ligne)

La sonde v1 n'a **pas persisté l'HTML** des 4 pages et le contrat interdit
le réseau: `artifacts/web-probe-v2/pages/` contient des reconstructions
**déterministes** calibrées pour reproduire exactement les compteurs v1
persistés (512 actionnables / 30 couverts → 5.86 %). Le transfert est
**vérifié** en rejouant le module v1 (lecture seule) sur les snapshots
(`recompute_v1_on_pages`, assert dans `make_snapshots.ensure_snapshots`,
test dédié). La page w3schools-forms contient les ids dupliqués `fname` ×2,
`lname` ×3 et des href de tous genres.

## Résultats (artifacts/web-probe-v2/completeness-report.json)

Complétude par page v2 = 100 % ≥ 90 % (GO) sur les 4 pages, contre 5.86 %
global en v1 (mêmes définitions, mêmes dénominateurs, vérifiés).

## Couvert / NON couvert

Couvert (statique): formulaires (champs texte-likes, textareas, boutons,
selects+options, labels for), liens href (relatifs/absolus/racine-site/
protocol-relatifs/ancres/vides/mailto/javascript:), imbrication form/élément,
ids dupliqués, sortie JSON strictement typée et validée.

NON couvert (documenté, hors périmètre v2):
- **JS dynamique**: contenu injecté par scripts, SPAs (rendu serveur statique
  uniquement — le DOM parsé est celui du HTML source).
- **iframes / framesets / shadow DOM**: non traversés.
- **États**: disabled/hidden/readonly ignorés (tout élément actionnable au
  sens v1 est candidat, même désactivé).
- **Événements**: onclick & co non représentés (seul `submits` est déduit).
- **ARIA/role, tabindex, tables de layout, datalists**: non exploités.
- **Tensorisation** (vocabulaire TGK, D_IN): responsabilité consommateur.

## Tests

`.venv/bin/pytest tests/test_web_compiler.py` — ids dupliqués (déterminisme
inclus), liens sans href, imbrication form/link, résolution relative/absolue/
ancre/protocol-relative, validation vocabulaire (violations rejetées), 4
pages compilent sans erreur, parité v1 sur snapshots (512/30/5.86 %),
complétude ≥ 90 % par page, rapport régénéré hermétiquement.
