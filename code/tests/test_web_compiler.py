"""Tests du compilateur web v2 (ucm/web/compiler_v2.py).

Couvre: ids dupliqués (dédup déterministe fname->fname_2), liens sans href,
imbrication form/link, résolution relative/absolue/ancre/protocol-relative,
validation du vocabulaire fermé web/2.0, les 4 pages de la sonde compilent
sans erreur, parité des définitions v1 sur les snapshots (baseline 5.86%),
complétude >= 90% par page, rapport régénéré hermétiquement.

Hors-ligne: aucune dépendance nouvelle, aucun réseau, aucun chargement du
modèle UCM (sortie JSON structurée uniquement).
"""
import json
import os
import shutil

import pytest

from ucm.web.compiler_v2 import (BASELINE_V1, DEFAULT_PAGES_DIR, build_report,
                                 compile_page, compile_pages,
                                 recompute_v1_on_pages, write_artifacts)
from ucm.web.make_snapshots import ensure_snapshots, generate_pages
from ucm.web.vocabulary import (PolicyInputError, ensure_valid,
                                validate_policy_input)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "https://www.w3schools.com/html/html_forms.asp"


# --------------------------------------------------------------------------- #
# Déduplication déterministe des ids (bloqueur n°3)
# --------------------------------------------------------------------------- #

def test_duplicate_ids_get_deterministic_suffix():
    html = ('<form id="f"><input type="text" id="fname">'
            '<input type="text" id="fname"></form>')
    pi, stats = compile_page(html, url=BASE)
    ids = [e["id"] for e in pi["entities"] if e["type"] == "field"]
    assert ids == ["fname", "fname_2"]
    assert stats["dedup_renames"] == [{"from": "fname", "to": "fname_2"}]
    ensure_valid(pi)


def test_triple_collision_and_suffix_collision():
    # trois id="x" -> x, x_2, x_3 ; puis un id nu "x_2" -> doit être déplacé
    html = ('<input type="text" id="x"><input type="text" id="x">'
            '<input type="text" id="x"><input type="text" id="x_2">')
    pi, stats = compile_page(html, url=BASE)
    ids = [e["id"] for e in pi["entities"] if e["type"] == "field"]
    assert ids == ["x", "x_2", "x_3", "x_2_2"]
    assert {"from": "x", "to": "x_2"} in stats["dedup_renames"]
    assert {"from": "x_2", "to": "x_2_2"} in stats["dedup_renames"]
    ensure_valid(pi)


def test_compilation_is_deterministic():
    html = ('<form id="f"><input type="text" id="fname" name="fname">'
            '<a href="/x">x</a><input type="text" id="fname"></form>')
    pi1, s1 = compile_page(html, url=BASE)
    pi2, s2 = compile_page(html, url=BASE)
    assert json.dumps(pi1, sort_keys=True) == json.dumps(pi2, sort_keys=True)
    assert s1["dedup_renames"] == s2["dedup_renames"]


# --------------------------------------------------------------------------- #
# Liens (bloqueur n°2: représentation) — sans href, résolution, imbrication
# --------------------------------------------------------------------------- #

def test_links_without_href_are_not_actionable():
    html = ('<a name="top">ancre</a><a class="disabled">off</a>'
            '<a href="/real">real</a>')
    pi, stats = compile_page(html, url=BASE)
    links = [e for e in pi["entities"] if e["type"] == "link"]
    assert len(links) == 1 and links[0]["id"] != ""
    assert stats["actionable"] == 1
    assert stats["non_actionable_skipped"] == 2
    navs = [c for c in pi["candidates"] if c["action"] == "NAVIGATE"]
    assert len(navs) == 1 and navs[0]["arg"] == links[0]["id"]
    ensure_valid(pi)


def test_href_resolution_kinds():
    html = ('<a href="html_intro.asp">1</a>'                    # relative
            '<a href="/css/default.asp">2</a>'                  # racine site
            '<a href="https://developer.mozilla.org/x">3</a>'   # absolute
            '<a href="//www.w3schools.com/about">4</a>'         # protocol-rel
            '<a href="#top">5</a>'                              # ancre
            '<a href="mailto:a@b.co">7</a>'                     # other_scheme
            '<a href="javascript:void(0)">8</a>')               # other_scheme
    pi, stats = compile_page(html, url=BASE)
    links = {e["attrs"]["href_raw"]: e["attrs"] for e in pi["entities"]
             if e["type"] == "link"}
    assert links["html_intro.asp"]["href_kind"] == "relative"
    assert links["html_intro.asp"]["href_resolved"] == \
        "https://www.w3schools.com/html/html_intro.asp"
    assert links["/css/default.asp"]["href_kind"] == "site_root_relative"
    assert links["/css/default.asp"]["href_resolved"] == \
        "https://www.w3schools.com/css/default.asp"
    assert links["https://developer.mozilla.org/x"]["href_kind"] == "absolute"
    assert links["//www.w3schools.com/about"]["href_kind"] == "scheme_relative"
    assert links["//www.w3schools.com/about"]["href_resolved"] == \
        "https://www.w3schools.com/about"
    assert links["#top"]["href_kind"] == "anchor"
    assert links["#top"]["href_resolved"] == BASE + "#top"
    assert links["mailto:a@b.co"]["href_kind"] == "other_scheme"
    assert links["mailto:a@b.co"]["href_resolved"] == "mailto:a@b.co"
    assert links["javascript:void(0)"]["href_kind"] == "other_scheme"
    assert stats["actionable"] == 7 and stats["covered"] == 7
    ensure_valid(pi)


def test_resolve_href_empty_is_same_page_but_not_actionable():
    # resolve_href (fonction pure) sait résoudre href="" -> même page...
    from ucm.web.compiler_v2 import resolve_href
    assert resolve_href("", BASE) == (BASE, "same_page")
    # ...mais la sémantique v1 (a.get("href") véridique) rend href=""
    # NON actionnable: parité stricte du dénominateur v1/v2.
    pi, stats = compile_page('<a href="">empty</a><a href="/x">x</a>',
                             url=BASE)
    links = [e for e in pi["entities"] if e["type"] == "link"]
    assert len(links) == 1
    assert stats["actionable"] == 1 and stats["covered"] == 1
    assert stats["non_actionable_skipped"] == 1
    v1 = __import__("ucm.web.compiler_probe", fromlist=["compile_page"])
    _pi1, a1, c1 = v1.compile_page('<a href="">empty</a><a href="/x">x</a>')
    assert (a1, c1) == (1, 0)  # v1 ignore aussi le href vide


def test_unresolvable_hrefs_without_page_url():
    pi, stats = compile_page('<a href="next.asp">n</a><a href="#x">a</a>')
    for e in pi["entities"]:
        if e["type"] == "link":
            assert e["attrs"]["href_resolved"] is None
            assert e["attrs"]["href_kind"] == "unresolvable"
    assert stats["covered"] == 2  # actionnable+couvert malgré l'absence d'URL


def test_link_nested_in_form():
    html = ('<form id="f1" action="/go">'
            '<a href="next.html">Next</a>'
            '<input type="text" id="q">'
            '<button type="submit">Go</button>'
            '</form>')
    pi, stats = compile_page(html, url=BASE)
    rels = {(r["subj"], r["pred"], r["obj"]) for r in pi["relations"]}
    ents = {e["id"]: e for e in pi["entities"]}
    link_id = next(i for i, e in ents.items() if e["type"] == "link")
    assert (link_id, "on_view", "page") in rels
    assert (link_id, "part_of", "f1") in rels
    assert ("q", "part_of", "f1") in rels
    # bouton submit d'un form -> relation submits (alignée siw)
    btn = next(i for i, e in ents.items()
               if e["type"] == "button" and e["attrs"]["input_type"] == "submit")
    assert (btn, "submits", "f1") in rels
    acts = [(c["action"], c["arg"]) for c in pi["candidates"]]
    assert ("NAVIGATE", link_id) in acts
    assert ("TYPE", "q") in acts
    assert ("CLICK", btn) in acts
    assert stats["actionable"] == 3 and stats["covered"] == 3
    ensure_valid(pi)


# --------------------------------------------------------------------------- #
# Définitions d'actionnable EXACTEMENT v1
# --------------------------------------------------------------------------- #

def test_non_actionable_inputs_are_skipped():
    html = ('<input type="radio" name="s" value="1">'
            '<input type="checkbox" name="c">'
            '<input type="hidden" name="csrf">'
            '<input type="file" name="up">'
            '<input name="legacy">')
    pi, stats = compile_page(html, url=BASE)
    assert stats["actionable"] == 0 and stats["covered"] == 0
    assert stats["non_actionable_skipped"] == 5
    assert [c for c in pi["candidates"] if c["action"] != "STOP"] == []


def test_select_and_options_v1_accounting():
    html = ('<select id="s1"><option id="a" value="1">A</option>'
            '<option id="b" value="2">B</option></select>')
    pi, stats = compile_page(html, url=BASE)
    ents = {e["id"]: e for e in pi["entities"]}
    assert ents["s1"]["type"] == "select"
    rels = {(r["subj"], r["pred"], r["obj"]) for r in pi["relations"]}
    for oid in ("a", "b"):
        assert ents[oid]["type"] == "option"
        assert (oid, "option_of", "s1") in rels
    # comptabilité v1: select actionnable (1), options couvertes non
    # actionnables (2); v2 couvre en plus le select via ses options (+1).
    assert stats["actionable"] == 1
    assert stats["categories"]["options"] == 2
    assert stats["covered"] == 3
    sel = [c for c in pi["candidates"] if c["action"] == "SELECT"]
    assert sorted(c["arg"] for c in sel) == ["a", "b"]
    ensure_valid(pi)


def test_goal_placeholder_and_stop_terminal():
    pi, _ = compile_page('<a href="/x">x</a>', url=BASE)
    assert pi["goal"] == {"predicate": "VIEW", "args": {"view": "page"}}
    assert pi["candidates"][-1] == {"action": "STOP", "arg": None}
    assert sum(1 for c in pi["candidates"] if c["action"] == "STOP") == 1
    assert pi["schema_version"] == "web/2.0"


def test_strict_json_typing_roundtrip():
    pi, _ = compile_page(
        '<form id="f"><input type="text" id="q"><a href="/x">x</a>'
        '<select id="s"><option id="o1">1</option></select></form>', url=BASE)
    assert json.loads(json.dumps(pi)) == pi  # JSON-natif
    for e in pi["entities"]:
        assert isinstance(e["id"], str) and e["id"]
        for v in e["attrs"].values():
            assert isinstance(v, (str, bool, int)) or v is None
            assert not isinstance(v, float)
    ensure_valid(pi)


# --------------------------------------------------------------------------- #
# Vocabulaire fermé web/2.0 (bloqueur n°1) — violations rejetées
# --------------------------------------------------------------------------- #

def _minimal_pi():
    pi, _ = compile_page('<a href="/x">x</a>', url=BASE)
    return json.loads(json.dumps(pi))


def test_validator_rejects_unknown_entity_type():
    pi = _minimal_pi()
    pi["entities"].append({"id": "weird", "type": "banana", "attrs": {}})
    errs = validate_policy_input(pi)
    assert any("banana" in e for e in errs)
    with pytest.raises(PolicyInputError):
        ensure_valid(pi)


def test_validator_rejects_duplicate_entity_ids():
    pi = _minimal_pi()
    pi["entities"].append(dict(pi["entities"][0]))
    assert any("dupliqu" in e for e in validate_policy_input(pi))


def test_validator_rejects_unknown_action_and_bad_arg_type():
    pi = _minimal_pi()
    pi["candidates"][0]["action"] = "PRESS"                     # hors vocab
    assert any("PRESS" in e for e in validate_policy_input(pi))
    pi = _minimal_pi()                                          # arg mal typé
    pi["candidates"][0] = {"action": "TYPE", "arg": pi["entities"][0]["id"]}
    errs = validate_policy_input(pi)
    assert any("attendu" in e for e in errs)
    pi = _minimal_pi()                                          # arg inconnu
    pi["candidates"][0] = {"action": "NAVIGATE", "arg": "ghost"}
    assert any("ghost" in e for e in validate_policy_input(pi))


def test_validator_rejects_dangling_relation_and_missing_stop():
    pi = _minimal_pi()
    pi["relations"].append({"subj": "ghost", "pred": "on_view", "obj": "page"})
    assert any("ghost" in e for e in validate_policy_input(pi))
    pi = _minimal_pi()
    pi["candidates"] = pi["candidates"][:-1]  # STOP retiré
    assert any("STOP" in e for e in validate_policy_input(pi))


# --------------------------------------------------------------------------- #
# Snapshots + 4 pages de la sonde + baseline v1 (comparabilité)
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def pages_dir(tmp_path_factory):
    return ensure_snapshots(str(tmp_path_factory.mktemp("pages")))


def test_snapshots_reproduce_v1_baseline_exactly(pages_dir):
    """Le module v1 (lecture seule) rejoué sur les snapshots doit reproduire
    EXACTEMENT les compteurs persistés de la baseline 5.86%."""
    with open(BASELINE_V1, encoding="utf-8") as fh:
        baseline = json.load(fh)
    recomputed = recompute_v1_on_pages(pages_dir)
    for name, page in baseline["pages"].items():
        assert recomputed["pages"][name]["actionable"] == page["actionable"], name
        assert recomputed["pages"][name]["covered"] == page["covered"], name
    assert recomputed["totals"] == baseline["totals"] == {
        "actionable": 512, "covered": 30}
    assert recomputed["overall_completeness"] == \
        baseline["overall_completeness"] == 0.0586


def test_four_probe_pages_compile_without_error(pages_dir):
    results = compile_pages(pages_dir)
    assert list(results) == ["httpbin-post", "example", "httpbin",
                             "w3schools-forms"]
    for name, r in results.items():
        ensure_valid(r["policy_input"])          # aucune erreur, sortie valide
        assert r["stats"]["completeness"] >= 0.90, (name, r["stats"])
    cat = {n: r["stats"]["categories"] for n, r in results.items()}
    assert cat["example"]["links"] == 1
    assert cat["httpbin"]["links"] == 5
    assert cat["httpbin-post"]["links"] == 0
    assert cat["w3schools-forms"]["links"] == 476          # le trou v1
    assert cat["httpbin-post"]["fields"] == 1 and cat["httpbin-post"]["buttons"] == 1
    w3 = results["w3schools-forms"]["stats"]
    assert w3["actionable"] == 504 and w3["covered"] == 504  # 100% vs 5.56% v1
    assert w3["categories"]["fields"] == 16
    assert w3["categories"]["buttons"] == 12


def test_w3schools_exercises_documented_blockers(pages_dir):
    results = compile_pages(pages_dir)
    w3 = results["w3schools-forms"]
    renames = {r["from"]: r["to"] for r in w3["stats"]["dedup_renames"]}
    assert renames.get("fname") == "fname_2"      # bloqueur n°3 documenté
    assert renames.get("lname") in ("lname_2", "lname_3")
    ids = [e["id"] for e in w3["policy_input"]["entities"]]
    assert len(ids) == len(set(ids))              # dédup complète
    kinds = set(w3["stats"]["href_kinds"])
    assert {"relative", "site_root_relative", "absolute", "scheme_relative",
            "anchor", "other_scheme"} <= kinds


def test_report_build_and_write_hermetic(pages_dir, tmp_path):
    results = compile_pages(pages_dir)
    report = build_report(results, pages_dir=pages_dir)
    v2 = report["v2"]
    assert v2["go_completeness_ge_90_per_page"] is True
    assert v2["go_completeness_ge_90_overall"] is True
    assert v2["overall_completeness"] == 1.0
    assert report["baseline_v1"]["overall_completeness"] == 0.0586
    assert report["v1_recomputed_on_snapshots"]["overall_completeness"] == 0.0586
    for name, p in v2["per_page"].items():
        assert p["completeness"] >= 0.90
        assert p["v1_recomputed_actionable"] == p["actionable"]  # même dénominateur
        assert p["schema_valid"] is True
    assert set(report["blockers_treated"]) == {
        "1_vocabulaire_ferme", "2_largeur_collate", "3_ids_dupliques",
        "liens_representation"}
    written = write_artifacts(results, report, out_dir=str(tmp_path))
    assert os.path.exists(written["report"])
    for name in results:
        with open(written[name], encoding="utf-8") as fh:
            ensure_valid(json.load(fh))


def test_committed_artifacts_are_reproducible():
    """Les artefacts committés (s'ils existent) sont identiques à la
    régénération déterministe (aucune dépendance au live)."""
    if not os.path.isdir(DEFAULT_PAGES_DIR):
        pytest.skip("artifacts/web-probe-v2/pages/ non encore généré")
    pages = generate_pages()
    for name, html in pages.items():
        with open(os.path.join(DEFAULT_PAGES_DIR, f"{name}.html"),
                  encoding="utf-8") as fh:
            assert fh.read() == html, f"{name}: snapshot divergent"
