"""Compilateur web v2: page HTML -> policy_input STRUCTURÉ (schéma SIW étendu).

Étend la sonde v1 (commit 90a303a, ucm/web/compiler_probe.py, LECTURE SEULE)
en traitant les 3 blocages documentés:

  1. VOCABULAIRE (bloqueur 'view' hors vocab TGK): vocabulaire web/2.0 fermé
     propre (ucm/web/vocabulary.py) incluant `view` ET `link` + action
     NAVIGATE; toute sortie validée par validate_policy_input (JSON strict).
     La tensorisation (largeur de collate D_IN 21 vs 7) reste la
     responsabilité du consommateur: v2 produit du JSON typé, pas des tensors.
  2. IDS DOM DUPLIQUÉS ('fname' ×2): déduplication déterministe par suffixe —
     la première occurrence garde l'id nu, les suivantes prennent _2, _3, ...
     (fname -> fname_2). Renvois tracés dans stats["dedup_renames"].
  3. REPRÉSENTATION DES LIENS (476/512 actionnables non couverts en v1):
     chaque <a href> devient une entité `link` {href_raw, href_resolved,
     href_kind, text} + relation on_view (+part_of si imbriqué dans un form)
     + candidat NAVIGATE. Résolution relative/absolue/ancre/protocol-relative
     via urllib.parse.urljoin.

DÉFINITIONS D'ACTIONNABILITÉ: EXACTEMENT celles de la sonde v1 (90a303a),
pour que la comparaison avec la baseline 5.86% soit valide. Dénominateur =
  * <input type∈{text,email,password,tel,number,search,url,date}>
  * <textarea>
  * <button> ou <input type∈{submit,button,reset}>
  * <select>
  * <a> avec href PRÉSENT ET NON VIDE (sémantique véridique v1:
    `a.get("href")` — href="" ou absent ⇒ non actionnable)
Les <option> reçoivent des candidats SELECT (comptés couverts, NON
actionnables — identique v1). Inputs radio/checkbox/hidden/file/sans-type:
ni entité ni candidat (non actionnables au sens v1 — décision documentée
dans ucm/web/README.md).

Usage:
    from ucm.web.compiler_v2 import compile_page, compile_pages
    pi, stats = compile_page(html, url="https://example.com/")
    results   = compile_pages()          # 4 pages de la sonde (snapshots)
    report    = build_report(results)    # rapport complétude vs baseline v1
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections import OrderedDict
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

from ucm.web.vocabulary import (ACTIONS, ENTITY_TYPES, GOAL_PREDICATES,
                                PREDICATES, SCHEMA_VERSION, PolicyInputError,
                                ensure_valid, validate_policy_input)

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_PAGES_DIR = os.path.join(_REPO, "artifacts", "web-probe-v2", "pages")
DEFAULT_OUT_DIR = os.path.join(_REPO, "artifacts", "web-probe-v2")
BASELINE_V1 = os.path.join(_REPO, "artifacts", "web-probe",
                           "compiler-probe-20260925T152416.json")

# Définitions v1 (90a303a) — NE PAS MODIFIER (comparabilité baseline).
V1_TEXTUAL_INPUT_TYPES = frozenset({
    "text", "email", "password", "tel", "number", "search", "url", "date"})
V1_BUTTON_INPUT_TYPES = frozenset({"submit", "button", "reset"})
_WANTED_TAGS = frozenset({"input", "button", "select", "textarea", "option",
                          "a", "form", "label"})

_HREF_OTHER_SCHEMES = ("javascript:", "mailto:", "tel:", "data:")


# --------------------------------------------------------------------------- #
# Parsing DOM (stdlib html.parser — aucune dépendance nouvelle)
# --------------------------------------------------------------------------- #

class _Dom:
    """DOM plat minimal: éléments d'intérêt en ordre document + labels."""

    def __init__(self) -> None:
        self.elements: list[dict] = []   # ordre document
        self.labels: dict[str, str] = {}  # id DOM brut -> texte du <label for>

    def add(self, tag: str, attrs: dict, form: str | None,
            select: str | None) -> None:
        self.elements.append({"tag": tag, "attrs": attrs, "form": form,
                              "select": select, "n": len(self.elements)})


class _PageParser(HTMLParser):
    """Collecte les éléments d'intérêt + imbrication form/select + textes."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.dom = _Dom()
        self._form_stack: list[str] = []      # ids DOM bruts de forms ouvertes
        self._select_stack: list[str] = []
        self._label_for: str | None = None
        self._label_buf: list[str] = []
        self._text_tag: str | None = None     # 'a' | 'button' | None
        self._text_buf: list[str] = []
        self._text_target: dict | None = None

    # -- helpers ----------------------------------------------------------- #
    @staticmethod
    def _attrs(raw) -> dict:
        out: dict[str, str] = {}
        for k, v in raw:
            k = k.lower()
            if k not in out:                 # première occurrence gagne
                out[k] = v if v is not None else ""
        return out

    def handle_starttag(self, tag, raw_attrs):
        if tag not in _WANTED_TAGS:
            return
        attrs = self._attrs(raw_attrs)
        form = self._form_stack[-1] if self._form_stack else None
        select = self._select_stack[-1] if self._select_stack else None
        if tag == "form":
            fid = attrs.get("id") or f"form{len(self.dom.elements) + 1}"
            self._form_stack.append(fid)
            self.dom.add(tag, attrs, None, None)
            # form id brut conservé dans attrs pour le mapping part_of
            self.dom.elements[-1]["attrs"]["__dom_id"] = fid
            return
        if tag == "select":
            sid = attrs.get("id") or attrs.get("name") or f"select{len(self.dom.elements) + 1}"
            self._select_stack.append(sid)
            self.dom.add(tag, attrs, form, None)
            self.dom.elements[-1]["attrs"]["__dom_id"] = sid
            return
        if tag == "label":
            self._label_for = attrs.get("for") or None
            self._label_buf = []
            return
        self.dom.add(tag, attrs, form, select)
        if tag in ("a", "button"):
            self._text_tag, self._text_buf = tag, []
            self._text_target = self.dom.elements[-1]

    def handle_startendtag(self, tag, raw_attrs):
        # <input .../> auto-fermant: même traitement que starttag (jamais
        # refermé par handle_endtag).
        if tag == "input":
            self.handle_starttag(tag, raw_attrs)
        elif tag in _WANTED_TAGS and tag not in ("a", "button", "textarea",
                                                 "select", "form", "label"):
            self.handle_starttag(tag, raw_attrs)

    def handle_endtag(self, tag):
        if tag == "form" and self._form_stack:
            self._form_stack.pop()
        elif tag == "select" and self._select_stack:
            self._select_stack.pop()
        elif tag == "label":
            if self._label_for is not None:
                txt = re.sub(r"\s+", " ", "".join(self._label_buf)).strip()
                if txt:
                    self.dom.labels[self._label_for] = txt[:120]
            self._label_for, self._label_buf = None, []
        elif tag in ("a", "button") and self._text_tag == tag:
            txt = re.sub(r"\s+", " ", "".join(self._text_buf)).strip()
            if self._text_target is not None:
                self._text_target["text"] = txt[:80]
            self._text_tag, self._text_buf, self._text_target = None, [], None

    def handle_data(self, data):
        if self._label_for is not None:
            self._label_buf.append(data)
        elif self._text_tag is not None:
            self._text_buf.append(data)


def parse_dom(html: str) -> _Dom:
    p = _PageParser()
    p.feed(html)
    p.close()
    return p.dom


# --------------------------------------------------------------------------- #
# Déduplication déterministe des ids (bloqueur n°3)
# --------------------------------------------------------------------------- #

class _IdAllocator:
    """fname, fname, fname -> fname, fname_2, fname_3 (déterministe)."""

    def __init__(self) -> None:
        self._used: set[str] = set()
        self.renames: list[dict] = []

    def alloc(self, raw: str) -> str:
        final = raw
        k = 2
        while final in self._used:
            final = f"{raw}_{k}"
            k += 1
        self._used.add(final)
        if final != raw:
            self.renames.append({"from": raw, "to": final})
        return final


# --------------------------------------------------------------------------- #
# Résolution des href (bloqueur n°2 — représentation des liens)
# --------------------------------------------------------------------------- #

def resolve_href(href: str, base_url: str | None) -> tuple[str | None, str]:
    """(href_resolved, href_kind). kind ∈ absolute|relative|site_root_relative|
    scheme_relative|anchor|same_page|other_scheme|unresolvable."""
    raw = href.strip()
    low = raw.lower()
    if low.startswith(_HREF_OTHER_SCHEMES):
        return raw, "other_scheme"
    if raw.startswith("//"):
        if base_url:
            scheme = urlsplit(base_url).scheme or "https"
            return f"{scheme}:{raw}", "scheme_relative"
        return None, "unresolvable"
    if raw.startswith("#"):
        if base_url:
            return urljoin(base_url, raw), "anchor"
        return None, "unresolvable"
    if raw == "":
        if base_url:
            return urljoin(base_url, ""), "same_page"
        return None, "unresolvable"
    if urlsplit(raw).scheme:
        return raw, "absolute"
    if base_url:
        resolved = urljoin(base_url, raw)
        kind = "site_root_relative" if raw.startswith("/") else "relative"
        return resolved, kind
    return None, "unresolvable"


# --------------------------------------------------------------------------- #
# Compilation page -> policy_input
# --------------------------------------------------------------------------- #

def compile_page(html: str, url: str | None = None) -> tuple[dict, dict]:
    """Compile une page HTML en policy_input web/2.0 STRICTEMENT typé.

    Retourne (policy_input, stats). stats suit les définitions v1:
    actionable/covered + catégories + renommages de dédup + kinds de href.
    Lève PolicyInputError si la sortie violerait le schéma (ne doit jamais
    arriver: auto-validation systématique).
    """
    dom = parse_dom(html)
    ids = _IdAllocator()

    entities: list[dict] = []
    relations: list[dict] = []
    candidates: list[dict] = []
    stats = {
        "url": url,
        "actionable": 0, "covered": 0,
        "categories": {"fields": 0, "buttons": 0, "selects": 0,
                       "links": 0, "options": 0, "forms": 0},
        "non_actionable_skipped": 0,
        "dedup_renames": [],
        "href_kinds": {},
        "n_entities": 0, "n_relations": 0, "n_candidates_actionable": 0,
    }

    def emit(ent_id_raw: str, etype: str, attrs: dict,
             on_view: bool = True, part_of: str | None = None,
             view: str | None = "page") -> str:
        final = ids.alloc(ent_id_raw)
        ent = {"id": final, "type": etype, "attrs": attrs}
        if view is not None:
            ent["view"] = view
        entities.append(ent)
        if on_view:
            relations.append({"subj": final, "pred": "on_view", "obj": "page"})
        if part_of is not None:
            relations.append({"subj": final, "pred": "part_of", "obj": part_of})
        return final

    # Page = vue racine (entité 'view' — bloqueur n°1: légal en web/2.0).
    ids.alloc("page")
    entities.append({"id": "page", "type": "view", "attrs": {
        "url": url} if url else {}})

    form_ids: dict[str, str] = {}      # id DOM brut form -> id final
    select_ids: dict[str, str] = {}    # id DOM brut select -> id final
    pending_selects: dict[str, dict] = {}  # id final select -> stats bucket

    for el in dom.elements:
        tag, a = el["tag"], el["attrs"]
        raw_id = a.get("id") or a.get("name") or f"{tag}{el['n']}"
        itype = (a.get("type") or "").lower()
        dom_id = a.pop("__dom_id", None)
        text = el.get("text", "")

        if tag == "form":
            stats["categories"]["forms"] += 1
            fid = emit(dom_id or raw_id, "form", {
                "action": a.get("action"), "method": (a.get("method") or "get").lower(),
            }, on_view=True)
            form_ids[dom_id or raw_id] = fid
            continue

        if tag == "label":
            continue  # texte déjà capturé dans dom.labels

        part_of = form_ids.get(el["form"]) if el["form"] else None

        if tag == "input" and itype in V1_TEXTUAL_INPUT_TYPES:
            attrs = {"input_type": itype, "value": a.get("value")}
            _copy_if(attrs, a, "dom_id", "id")
            _copy_if(attrs, a, "dom_name", "name")
            _label_attr(attrs, dom.labels, a)
            fid = emit(raw_id, "field", attrs, part_of=part_of)
            candidates.append({"action": "TYPE", "arg": fid})
            stats["actionable"] += 1
            stats["covered"] += 1
            stats["categories"]["fields"] += 1

        elif tag == "textarea":
            attrs = {"input_type": "textarea", "value": None}
            _copy_if(attrs, a, "dom_id", "id")
            _copy_if(attrs, a, "dom_name", "name")
            _label_attr(attrs, dom.labels, a)
            fid = emit(raw_id, "field", attrs, part_of=part_of)
            candidates.append({"action": "TYPE", "arg": fid})
            stats["actionable"] += 1
            stats["covered"] += 1
            stats["categories"]["fields"] += 1

        elif tag == "button" or (tag == "input" and itype in V1_BUTTON_INPUT_TYPES):
            # input_type: le type RÉEL du bouton (submit/button/reset);
            # <button> sans type = submit par défaut (spec HTML) mais on
            # conserve la valeur DOM "" telle quelle dans dom_type.
            attrs = {"input_type": itype if tag == "input" else (itype or "button"),
                     "text": text or a.get("value")}
            _copy_if(attrs, a, "dom_id", "id")
            _copy_if(attrs, a, "dom_name", "name")
            bid = emit(raw_id, "button", attrs, part_of=part_of)
            is_submit = (tag == "button" and itype in ("submit", "")) or \
                (tag == "input" and itype == "submit")
            if is_submit and part_of:
                relations.append({"subj": bid, "pred": "submits", "obj": part_of})
            candidates.append({"action": "CLICK", "arg": bid})
            stats["actionable"] += 1
            stats["covered"] += 1
            stats["categories"]["buttons"] += 1

        elif tag == "select":
            attrs = {"multiple": "multiple" in a or a.get("multiple") == "multiple"}
            _copy_if(attrs, a, "dom_id", "id")
            _copy_if(attrs, a, "dom_name", "name")
            sid = emit(raw_id, "select", attrs, part_of=part_of)
            select_ids[dom_id or raw_id] = sid
            pending_selects[sid] = {"options": 0}
            stats["actionable"] += 1            # actionnable v1...
            stats["categories"]["selects"] += 1  # ...couvert ssi ≥1 option

        elif tag == "option":
            attrs = {"value": a.get("value"), "text": text}
            sel_final = select_ids.get(el["select"]) if el["select"] else None
            oid = emit(raw_id, "option", attrs, on_view=False, view=None)
            relations.append({"subj": oid, "pred": "option_of",
                              "obj": sel_final or "page"})
            candidates.append({"action": "SELECT", "arg": oid})
            stats["covered"] += 1               # couvert, NON actionnable (v1)
            stats["categories"]["options"] += 1
            if sel_final:
                pending_selects[sel_final]["options"] += 1

        elif tag == "a":
            href = a.get("href")
            # Sémantique v1 EXACTE (sonde 90a303a: `a.get("href")` véridique):
            # un href absent OU VIDE n'est pas actionnable — ni entité, ni
            # candidat, compté comme ignoré. (Parité stricte du dénominateur.)
            if not href:
                stats["non_actionable_skipped"] += 1
                continue
            resolved, kind = resolve_href(href, url)
            attrs = {"href_raw": href, "href_resolved": resolved,
                     "href_kind": kind, "text": text}
            _copy_if(attrs, a, "dom_id", "id")
            lid = emit(raw_id, "link", attrs, part_of=part_of)
            candidates.append({"action": "NAVIGATE", "arg": lid})
            stats["actionable"] += 1
            stats["covered"] += 1               # ← le trou v1 est comblé
            stats["categories"]["links"] += 1
            stats["href_kinds"][kind] = stats["href_kinds"].get(kind, 0) + 1

        else:
            # input radio/checkbox/hidden/file/sans-type, etc.: non
            # actionnables au sens v1 -> ni entité ni candidat.
            stats["non_actionable_skipped"] += 1

    # Un select est couvert dès qu'au moins une de ses options l'est.
    for sid, b in pending_selects.items():
        if b["options"] > 0:
            stats["covered"] += 1

    candidates.append({"action": "STOP", "arg": None})
    stats["dedup_renames"] = ids.renames
    stats["n_entities"] = len(entities)
    stats["n_relations"] = len(relations)
    stats["n_candidates_actionable"] = len(candidates) - 1
    stats["completeness"] = round(stats["covered"] / max(stats["actionable"], 1), 4)

    pi = {
        "schema_version": SCHEMA_VERSION,
        "entities": entities,
        "relations": relations,
        "goal": {"predicate": "VIEW", "args": {"view": "page"}},  # placeholder
        "candidates": candidates,
    }
    ensure_valid(pi)  # jamais de sortie non typée
    return pi, stats


def _copy_if(dst: dict, attrs: dict, dst_key: str, src_key: str) -> None:
    if attrs.get(src_key) is not None:
        dst[dst_key] = attrs[src_key]


def _label_attr(dst: dict, labels: dict, a: dict) -> None:
    for_id = a.get("id")
    if for_id and for_id in labels:
        dst["label"] = labels[for_id]


# --------------------------------------------------------------------------- #
# Compilation des 4 pages de la sonde + rapport de complétude
# --------------------------------------------------------------------------- #

def compile_pages(pages_dir: str | None = None) -> "OrderedDict[str, dict]":
    """Compile les pages listées par le manifeste pages.json de la sonde v2.

    Retourne OrderedDict{name: {"url", "file", "policy_input", "stats"}}.
    """
    pages_dir = pages_dir or DEFAULT_PAGES_DIR
    with open(os.path.join(pages_dir, "pages.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    out: OrderedDict[str, dict] = OrderedDict()
    for page in manifest["pages"]:
        with open(os.path.join(pages_dir, page["file"]), encoding="utf-8") as fh:
            html = fh.read()
        pi, stats = compile_page(html, url=page["url"])
        out[page["name"]] = {"url": page["url"], "file": page["file"],
                             "policy_input": pi, "stats": stats}
    return out


def recompute_v1_on_pages(pages_dir: str | None = None) -> dict:
    """Rejoue les définitions v1 (module sonde, LECTURE SEULE) sur les
    snapshots: prouve que le dénominateur/numérateur v1 se reproduit
    exactement sur les pages v2 (transfert de la baseline 5.86%)."""
    from ucm.web.compiler_probe import compile_page as v1_compile_page  # noqa
    pages_dir = pages_dir or DEFAULT_PAGES_DIR
    with open(os.path.join(pages_dir, "pages.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)
    out = {"pages": {}, "totals": {"actionable": 0, "covered": 0}}
    for page in manifest["pages"]:
        with open(os.path.join(pages_dir, page["file"]), encoding="utf-8") as fh:
            html = fh.read()
        _pi, a, c = v1_compile_page(html)
        out["pages"][page["name"]] = {
            "actionable": a, "covered": c,
            "completeness": round(c / max(a, 1), 4)}
        out["totals"]["actionable"] += a
        out["totals"]["covered"] += c
    out["overall_completeness"] = round(
        out["totals"]["covered"] / max(out["totals"]["actionable"], 1), 4)
    return out


def build_report(results: "OrderedDict[str, dict]",
                 pages_dir: str | None = None,
                 baseline_path: str | None = None) -> dict:
    """Rapport de complétude v2 vs baseline v1 (mêmes définitions)."""
    baseline_path = baseline_path or BASELINE_V1
    with open(baseline_path, encoding="utf-8") as fh:
        baseline = json.load(fh)
    v1_on_snaps = recompute_v1_on_pages(pages_dir)

    per_page, tot_a, tot_c = {}, 0, 0
    for name, r in results.items():
        s = r["stats"]
        per_page[name] = {
            "url": r["url"],
            "actionable": s["actionable"], "covered": s["covered"],
            "completeness": s["completeness"],
            "go_ge_90": s["completeness"] >= 0.90,
            "categories": s["categories"],
            "href_kinds": s["href_kinds"],
            "dedup_renames": s["dedup_renames"],
            "n_entities": s["n_entities"],
            "n_relations": s["n_relations"],
            "n_candidates": s["n_candidates_actionable"],
            "schema_valid": not validate_policy_input(r["policy_input"]),
            "v1_baseline_completeness": baseline["pages"][name]["completeness"],
            "v1_recomputed_actionable": v1_on_snaps["pages"][name]["actionable"],
            "v1_recomputed_covered": v1_on_snaps["pages"][name]["covered"],
        }
        tot_a += s["actionable"]
        tot_c += s["covered"]
    overall = round(tot_c / max(tot_a, 1), 4)
    return {
        "ts": time.strftime("%Y%m%dT%H%M%S"),
        "compiler": "ucm/web/compiler_v2.py (web/2.0)",
        "source": ("snapshots statiques hors-ligne artifacts/web-probe-v2/pages/"
                   " — reconstructions DÉTERMINISTES calibrées sur les compteurs"
                   " v1 (l'HTML original n'a pas été persisté par la sonde v1;"
                   " pas de réseau per contrat). Étiquette: pas via page-agent live."),
        "actionability_definition": (
            "EXACTEMENT v1 (commit 90a303a): dénominateur = inputs "
            "type∈{text,email,password,tel,number,search,url,date} + textareas "
            "+ <button>/input submit|button|reset + selects + <a href>; "
            "options = couverts non actionnables"),
        "baseline_v1": {
            "artifact": "artifacts/web-probe/compiler-probe-20260925T152416.json",
            "overall_completeness": baseline["overall_completeness"],
            "totals": baseline["totals"],
            "pages": {k: {"actionable": v["actionable"], "covered": v["covered"],
                          "completeness": v["completeness"]}
                      for k, v in baseline["pages"].items()},
        },
        "v1_recomputed_on_snapshots": v1_on_snaps,
        "v2": {"per_page": per_page,
               "totals": {"actionable": tot_a, "covered": tot_c},
               "overall_completeness": overall,
               "go_completeness_ge_90_per_page": all(
                   p["go_ge_90"] for p in per_page.values()),
               "go_completeness_ge_90_overall": overall >= 0.90},
        "blockers_treated": {
            "1_vocabulaire_ferme": (
                "vocabulaire web/2.0 fermé propre (ucm/web/vocabulary.py): "
                "types view/form/field/button/select/option/link + actions "
                "TYPE/CLICK/SELECT/NAVIGATE/STOP; validation systématique "
                "validate_policy_input (JSON strict). 'view' et 'link' ne "
                "passent plus par le vocab TGK fermé."),
            "2_largeur_collate": (
                "HORS périmètre compilateur: v2 sort du JSON strictement typé; "
                "la tensorisation (D_IN 21 vs 7) est la responsabilité du "
                "consommateur — documenté dans ucm/web/README.md."),
            "3_ids_dupliques": (
                "dédup déterministe par suffixe _N (première occurrence garde "
                "l'id nu): " + "; ".join(
                    f"{name}: " + (", ".join(
                        f"{r['from']}->{r['to']}" for r in p["dedup_renames"])
                        or "aucun renommage")
                    for name, p in per_page.items())),
            "liens_representation": (
                "chaque <a href> = entité link {href_raw, href_resolved, "
                "href_kind, text} + on_view (+part_of si imbriqué) + candidat "
                "NAVIGATE; résolution relative/absolue/racine-site/protocol-"
                "relative/ancre/same_page/other_scheme via urljoin."),
        },
        "policy_inputs_dir": "policy-inputs/",
    }


def write_artifacts(results: "OrderedDict[str, dict]", report: dict,
                    out_dir: str | None = None) -> dict:
    """Écrit completeness-report.json + policy-inputs/{name}.json."""
    out_dir = out_dir or DEFAULT_OUT_DIR
    pi_dir = os.path.join(out_dir, "policy-inputs")
    os.makedirs(pi_dir, exist_ok=True)
    written = {}
    for name, r in results.items():
        path = os.path.join(pi_dir, f"{name}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(r["policy_input"], fh, indent=1, sort_keys=False,
                      ensure_ascii=False)
        written[name] = path
    rpath = os.path.join(out_dir, "completeness-report.json")
    with open(rpath, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, sort_keys=True, ensure_ascii=False)
    written["report"] = rpath
    return written


def main() -> dict:
    from ucm.web.make_snapshots import ensure_snapshots
    pages_dir = ensure_snapshots()          # idempotent, déterministe
    results = compile_pages(pages_dir)
    report = build_report(results, pages_dir=pages_dir)
    report["snapshots_sha256"] = {
        p["file"]: hashlib.sha256(
            open(os.path.join(pages_dir, p["file"]), "rb").read()
        ).hexdigest() for p in (
            json.load(open(os.path.join(pages_dir, "pages.json"))))["pages"]}
    written = write_artifacts(results, report)
    report["written"] = {k: os.path.relpath(v, _REPO) for k, v in written.items()}
    # ré-écrit le rapport final (avec written) — même chemin, écrase
    with open(written["report"], "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, sort_keys=True, ensure_ascii=False)
    print(json.dumps({"overall_v2": report["v2"]["overall_completeness"],
                      "v1_recomputed": report["v1_recomputed_on_snapshots"]["overall_completeness"],
                      "go_per_page": report["v2"]["go_completeness_ge_90_per_page"],
                      "written": report["written"]}, indent=1))
    return report


if __name__ == "__main__":
    main()
