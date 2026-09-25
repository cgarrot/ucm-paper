"""Snapshots statiques des 4 pages de la sonde (v2) — HORS-LIGNE, déterministes.

CONTEXTE: la sonde v1 (90a303a) a fetché les 4 pages en live mais n'a PAS
persisté l'HTML; le contrat v2 interdit le réseau. Ces snapshots sont donc des
RECONSTRUCTIONS déterministes, calibrées pour reproduire EXACTEMENT les
compteurs v1 persistés (artifacts/web-probe/compiler-probe-20260925T152416.json):

    page             actionable(v1)  covered(v1)
    example                1              0
    httpbin                5              0
    httpbin-post           2              2
    w3schools-forms      504             28
    TOTAL                512             30   -> 5.86% baseline

Le transfert de baseline est VÉRIFIÉ en rejouant le module v1 (lecture seule)
sur ces snapshots (compiler_v2.recompute_v1_on_pages) — la définition
d'actionnable et les dénominateurs sont donc identiques v1/v2, et la
comparaison de complétude est valide. La page w3schools-forms contient les
ids dupliqués 'fname' ×2 et 'lname' ×3 (bloqueur n°3) et des href de tous
genres (relatifs, racine-site, absolus, protocol-relatifs, ancres, vides,
javascript:, mailto:).

Étiquette: pages statiques reconstruites, pas un fetch live (pas de réseau).
"""
from __future__ import annotations

import json
import os

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_OUT = os.path.join(_REPO, "artifacts", "web-probe-v2", "pages")

# Calibration v1 (compiler-probe-20260925T152416.json) — invariants durs.
CALIBRATION = {
    "example":        {"actionable": 1,   "covered": 0},
    "httpbin":        {"actionable": 5,   "covered": 0},
    "httpbin-post":   {"actionable": 2,   "covered": 2},
    "w3schools-forms": {"actionable": 504, "covered": 28},
}

# --------------------------------------------------------------------------- #

def _example() -> str:
    return """<!doctype html>
<html>
<head>
    <title>Example Domain</title>
    <meta charset="utf-8" />
    <meta http-equiv="Content-type" content="text/html; charset=utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
</head>
<body>
<div>
    <h1>Example Domain</h1>
    <p>This domain is for use in illustrative examples in documents. You may use this
    domain in literature without prior coordination or asking for permission.</p>
    <p><a href="https://www.iana.org/domains/example">More information...</a></p>
</div>
</body>
</html>
"""


def _httpbin() -> str:
    return """<!doctype html>
<html>
<head>
    <title>httpbin.org</title>
    <meta charset="utf-8">
</head>
<body>
    <h1>httpbin.org</h1>
    <p>A simple HTTP Request &amp; Response Service.</p>
    <ul>
        <li><a href="/forms/post">HTML form (pizza order)</a></li>
        <li><a href="/get">GET request</a></li>
        <li><a href="/post">POST request</a></li>
        <li><a href="https://github.com/psf/httpbin">Run locally: github.com/psf/httpbin</a></li>
        <li><a href="#endpoints">Jump to endpoints</a></li>
    </ul>
    <h2 id="endpoints">Endpoints</h2>
    <p>See the <a name="kenneth-reitz">author</a> page for details.</p>
</body>
</html>
"""


def _httpbin_post() -> str:
    # Calibré v1: 2 actionnables (custname text + submit) — les radios sont
    # non-actionnables au sens v1. Pas de liens sur cette page.
    return """<!doctype html>
<html>
<head>
    <title>httpbin.org - form post</title>
    <meta charset="utf-8">
</head>
<body>
    <h1>Pizza order form</h1>
    <p>This is an example pizza order form:</p>
    <form method="post" action="/post">
        <p><label for="custname">Customer name:</label>
           <input type="text" id="custname" name="custname"></p>
        <fieldset>
            <legend>Pizza size</legend>
            <p><label><input type="radio" name="size" value="small" checked> Small</label></p>
            <p><label><input type="radio" name="size" value="medium"> Medium</label></p>
            <p><label><input type="radio" name="size" value="large"> Large</label></p>
        </fieldset>
        <p><input type="submit" id="submit-order" value="Submit order"></p>
    </form>
</body>
</html>
"""


# --------------------------- w3schools-forms -------------------------------- #

_TOPNAV = [
    ("HTML", "/html/default.asp"), ("CSS", "/css/default.asp"),
    ("JavaScript", "/js/default.asp"), ("SQL", "/sql/default.asp"),
    ("Python", "/python/default.asp"), ("Java", "/java/default.asp"),
    ("PHP", "/php/default.asp"), ("Bootstrap", "/bootstrap/default.asp"),
    ("jQuery", "/jquery/default.asp"), ("C", "/c/index.php"),
    ("C++", "/cpp/default.asp"), ("C#", "/cs/index.php"),
    ("R", "/r/default.asp"), ("Kotlin", "/kotlin/index.php"),
    ("Node.js", "/nodejs/default.asp"), ("React", "/react/default.asp"),
    ("XML", "/xml/default.asp"), ("Angular", "/angular/default.php"),
    ("MySQL", "/mysql/default.asp"), ("Excel", "/excel/index.php"),
]

_CHAPTERS = [
    "html_intro", "html_editors", "html_basic", "html_elements",
    "html_attributes", "html_headings", "html_paragraphs", "html_styles",
    "html_formatting", "html_quotation", "html_comments", "html_colors",
    "html_colors_rgb", "html_colors_hex", "html_css", "html_links",
    "html_links_colors", "html_bookmarks", "html_images", "html_images_imagemap",
    "html_images_picture", "html_favicon", "html_tables", "html_table_borders",
    "html_table_sizes", "html_table_headers", "html_table_padding_spacing",
    "html_table_colspan_rowspan", "html_table_styling", "html_table_colgroup",
    "html_lists", "html_lists_unordered", "html_lists_ordered", "html_lists_other",
    "html_blocks", "html_classes", "html_id", "html_iframe", "html_javascript",
    "html_filepaths", "html_head", "html_layout", "html_responsive",
    "html_computercode", "html_symbols", "html_emoji", "html_charset",
    "html_urlencode", "html_xhtml",
    "html_forms", "html_forms_attributes", "html_form_elements",
    "html_form_input_types", "html_form_attributes",
    "html_form_attributes_form", "html_form_attributes_formaction",
    "html_form_attributes_enctype", "html_form_attributes_target_autofocus",
    "html_form_attributes_novalidate", "html_form_input_button",
    "html_form_input_checkbox", "html_form_input_color",
    "html_form_input_date", "html_form_input_datetime_local",
    "html_form_input_email", "html_form_input_file", "html_form_input_hidden",
    "html_form_input_image", "html_form_input_month",
    "html_form_input_number", "html_form_input_password",
    "html_form_input_radio", "html_form_input_range",
    "html_form_input_search", "html_form_input_submit",
    "html_form_input_tel", "html_form_input_text", "html_form_input_time",
    "html_form_input_url", "html_form_input_week",
    "html_form_elements_select", "html_form_elements_datalist",
    "html_form_elements_optgroup", "html_form_elements_option",
    "html_form_elements_textarea", "html_form_elements_button",
    "html_forms_validation",
]

_TAGS = [
    "tag_abbr", "tag_acronym", "tag_address", "tag_area", "tag_article",
    "tag_aside", "tag_audio", "tag_b", "tag_base", "tag_bdi", "tag_bdo",
    "tag_blockquote", "tag_body", "tag_br", "tag_button", "tag_canvas",
    "tag_caption", "tag_cite", "tag_code", "tag_col", "tag_colgroup",
    "tag_data", "tag_datalist", "tag_dd", "tag_del", "tag_details",
    "tag_dfn", "tag_dialog", "tag_div", "tag_dl", "tag_dt", "tag_em",
    "tag_embed", "tag_fieldset", "tag_figcaption", "tag_figure",
    "tag_font", "tag_footer", "tag_form", "tag_frame", "tag_frameset",
    "tag_h1", "tag_h2", "tag_h3", "tag_h4", "tag_h5", "tag_h6", "tag_head",
    "tag_header", "tag_hr", "tag_html", "tag_i", "tag_iframe", "tag_img",
    "tag_input", "tag_ins", "tag_kbd", "tag_label", "tag_legend", "tag_li",
    "tag_link", "tag_main", "tag_map", "tag_mark", "tag_meta", "tag_meter",
    "tag_nav", "tag_noframes", "tag_noscript", "tag_object", "tag_ol",
    "tag_optgroup", "tag_option", "tag_output", "tag_p", "tag_param",
    "tag_picture", "tag_pre", "tag_progress", "tag_q", "tag_rp", "tag_rt",
    "tag_ruby", "tag_s", "tag_samp", "tag_script", "tag_search", "tag_section",
    "tag_select", "tag_small", "tag_source", "tag_span", "tag_strike",
    "tag_strong", "tag_style", "tag_sub", "tag_summary", "tag_sup",
    "tag_svg", "tag_table", "tag_tbody", "tag_td", "tag_template",
    "tag_textarea", "tag_tfoot", "tag_th", "tag_thead", "tag_time",
    "tag_title", "tag_tr", "tag_track", "tag_tt", "tag_u", "tag_ul",
    "tag_var", "tag_video", "tag_wbr",
]

_BREADCRUMB = [
    ("Home", "/index.php"),
    ("HTML", "/html/default.asp"),
    ("HTML Forms", "html_forms.asp"),
]

_CONTENT_LINKS = [
    # (label, href) — tous genres de href (blocage n°2: représentation liens)
    ("Try it Yourself »", "#tryit1"),
    ("Try it Yourself »", "#tryit2"),
    ("Try it Yourself »", "#tryit3"),
    ("Try it Yourself »", "#tryit4"),
    ("Try it Yourself »", "#tryit5"),
    ("Try it Yourself »", "#tryit6"),
    ("Try it Yourself »", "#tryit7"),
    ("Try it Yourself »", "#tryit8"),
    ("the form element", "https://www.w3schools.com/tags/tag_form.asp"),
    ("the input element", "https://www.w3schools.com/tags/tag_input.asp"),
    ("the select element", "https://www.w3schools.com/tags/tag_select.asp"),
    ("the textarea element", "https://www.w3schools.com/tags/tag_textarea.asp"),
    ("the button element", "https://www.w3schools.com/tags/tag_button.asp"),
    ("form attributes", "html_forms_attributes.asp"),
    ("input types", "html_form_input_types.asp"),
    ("CSS tutorial", "/css/default.asp"),
    ("HTML exercises", "/html/exercise.asp"),
    ("form validation", "html_forms_validation.asp"),
    ("MDN: form docs", "https://developer.mozilla.org/en-US/docs/Web/HTML/Element/form"),
    ("MDN: input docs", "https://developer.mozilla.org/en-US/docs/Web/HTML/Element/input"),
    ("About W3Schools", "//www.w3schools.com/about/default.asp"),
    ("Contact us (mailto)", "mailto:help@w3schools.com"),
    ("Search the site", "search.asp"),
    ("Back to top", "#top"),
    ("Back to top", "#top"),
    ("Back to top", "#top"),
    ("Try the editor", "tryit/default.asp"),
    ("HTML quiz", "quiztest/quiztest.asp?qtest=HTML"),
    ("Donate (no-op)", "javascript:void(0)"),
    ("HTML forms intro", "html_forms.asp"),
]

_FOOTER_FIXED = [
    ("About", "/about/default.asp"),
    ("About W3Schools", "//www.w3schools.com/about/default.asp"),
    ("Privacy Policy", "/about/about_privacy.asp"),
    ("Terms of Use", "/about/about_terms.asp"),
    ("Cookies", "/about/about_cookies.asp"),
    ("Careers", "/about/about_careers.asp"),
    ("Contact", "/about/about_contact.asp"),
    ("Forum", "/forum/default.asp"),
    ("Newsletter", "/newsletter/default.asp"),
    ("Report Error", "/about/about_error.asp"),
    ("Advertise", "/about/about_advertise.asp"),
    ("Tutorials", "/tutorial/default.asp"),
    ("HTML Tutorial", "/html/default.asp"),
    ("CSS Tutorial", "/css/default.asp"),
    ("JavaScript Tutorial", "/js/default.asp"),
    ("Python Tutorial", "/python/default.asp"),
    ("SQL Tutorial", "/sql/default.asp"),
    ("Java Tutorial", "/java/default.asp"),
    ("References", "/references/index.php"),
    ("HTML Reference", "/tags/default.asp"),
    ("CSS Reference", "/cssref/index.php"),
    ("JS Reference", "/jsref/default.asp"),
    ("Python Reference", "/python/python_reference.asp"),
    ("SQL Reference", "/sql/sql_ref_keywords.asp"),
    ("Examples", "/html/html_examples.asp"),
    ("HTML Examples", "/html/html_examples.asp"),
    ("CSS Examples", "/css/css_examples.asp"),
    ("JS Examples", "/js/js_examples.asp"),
    ("Python Examples", "/python/python_examples.asp"),
    ("Exercises", "/exercises/index.php"),
    ("HTML Exercises", "/html/exercise.asp"),
    ("CSS Exercises", "/css/exercise.php"),
    ("JS Exercises", "/js/js_exercises.asp"),
    ("Quizzes", "/quiztest/default.asp"),
    ("Certificates", "/cert/default.asp"),
    ("Courses", "/courses/default.asp"),
    ("Bootcamp", "/bootcamp/index.php"),
    ("Spaces", "/spaces/index.php"),
    ("Website", "/website/default.php"),
    ("Plus", "/plus/default.asp"),
    ("Pro", "/pro/index.php"),
    ("Student Discount", "/about/about_w3schools_student.asp"),
    ("Teacher Discount", "/about/about_w3schools_teacher.asp"),
]

_N_SIDENAV = 380
_N_LINKS_TOTAL = 476


def _sidenav() -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for ch in _CHAPTERS:
        label = ch.replace("html_", "").replace("_", " ").capitalize()
        entries.append((label, f"{ch}.asp"))
    for tg in _TAGS:
        # échappé: sinon le texte du lien « <select> tag » serait parsé comme
        # une VRAIE balise par v1 (regex) ET par html.parser (fausse page)
        label = f"&lt;{tg.split('_')[1]}&gt; tag"
        entries.append((label, f"{tg}.asp"))
    for css in ("css_intro", "css_syntax", "css_selectors", "css_colors",
                "css_backgrounds", "css_borders", "css_margin", "css_padding",
                "css_boxmodel", "css_font", "css_text", "css_links",
                "css_lists", "css_tables", "css_display", "css_positioning",
                "css_float", "css_inline-block", "css_align", "css_combinators"):
        label = "CSS " + css.replace("css_", "").replace("_", " ").capitalize()
        entries.append((label, f"/css/{css}.asp"))
    i = 1
    while len(entries) < _N_SIDENAV:
        entries.append((f"HTML exercise {i}", f"exercise_html{i}.asp"))
        i += 1
    assert len(entries) == _N_SIDENAV, len(entries)
    return entries


def _footer() -> list[tuple[str, str]]:
    entries = list(_FOOTER_FIXED)
    i = 1
    target = _N_LINKS_TOTAL - (len(_TOPNAV) + _N_SIDENAV + len(_BREADCRUMB)
                               + len(_CONTENT_LINKS))
    while len(entries) < target:
        entries.append((f"Video tutorial {i}", f"/videos/index.php?v={i}"))
        i += 1
    assert len(entries) == target, (len(entries), target)
    return entries


def _w3schools_forms() -> str:
    """Page calibrée: 476 liens href + 28 actionnables formulaire.

    Non-link actionnables (28, définitions v1):
      * 16 champs texte-likes (14 <input> + 2 <textarea>) — dont les ids
        dupliqués 'fname' ×2 et 'lname' ×3 (bloqueur n°3)
      * 12 boutons (4 <input type=submit>, 8 <button>)
    Non actionnables (ignorés v1): radios, checkboxes, hidden, file,
    input sans type, <a> sans href.
    """
    L: list[str] = []
    add = L.append
    add("<!DOCTYPE html>")
    add('<html lang="en-US">')
    add('<head><title>HTML Forms</title><meta charset="utf-8"></head>')
    add('<body id="top">')
    # -- topnav (20 liens) ---------------------------------------------------
    add('<div id="topnav" class="w3-bar">')
    add('  <button id="menu-btn" type="button">☰ Menu</button>')
    for label, href in _TOPNAV:
        add(f'  <a href="{href}" class="w3-bar-item">{label}</a>')
    add('  <button id="search-btn" type="button">Search</button>')
    add('  <button id="login-btn" type="submit">Log in</button>')
    add("</div>")
    # -- sidenav (380 liens) -------------------------------------------------
    add('<div id="sidenav" class="w3-sidebar">')
    for label, href in _sidenav():
        add(f'  <a href="{href}">{label}</a>')
    add("</div>")
    # -- fil d'ariane (3 liens) ----------------------------------------------
    add('<div id="breadcrumbs">')
    for label, href in _BREADCRUMB:
        add(f'  <a href="{href}">{label}</a> »')
    add("</div>")
    add("<h1>HTML Forms</h1>")
    add('<p>An HTML form is used to collect user input.</p>')
    # -- form 1: exemple classique (3 actionnables) --------------------------
    add('<h2>The &lt;form&gt; Element</h2>')
    add('<form id="form1" action="/action_page.php" method="get">')
    add('  <label for="firstname">First name:</label>')
    add('  <input type="text" id="firstname" name="firstname">')
    add('  <label for="lname">Last name:</label>')
    add('  <input type="text" id="lname" name="lname">')
    add('  <input type="submit" name="submit1" value="Submit">')
    add("</form>")
    # -- form 2: démo champs texte — 1re occurrence de fname (4 actionnables) -
    add("<h2>Text Fields</h2>")
    add('<form id="form2" action="/action_page2.php">')
    add('  <label for="fname">First name:</label>')
    add('  <input type="text" id="fname" name="fname">')
    add('  <label for="lname">Last name:</label>')
    add('  <input type="text" id="lname" name="lname">')
    add('  <label>City: <input type="text" name="city"></label>')
    add('  <button type="submit">Submit</button>')
    add("</form>")
    # -- form 3: répétition de la démo (ids DUPPLIQUÉS) + 3 champs (6 actionnables)
    add("<h2>The Input Element (démo répétée)</h2>")
    add('<form id="form3" action="/action_page3.php">')
    add('  <label for="fname">First name:</label>')
    add('  <input type="text" id="fname" name="fname">')
    add('  <label for="lname">Last name:</label>')
    add('  <input type="text" id="lname" name="lname">')
    add('  <label for="email">E-mail:</label>')
    add('  <input type="email" id="email" name="email">')
    add('  <label for="bday">Birthday:</label>')
    add('  <input type="date" id="bday" name="bday">')
    add('  <label>Homepage: <input type="url" name="homepage"></label>')
    add('  <input type="submit" id="submit3" value="Submit">')
    add("</form>")
    # -- form 4: textareas (3 actionnables) ----------------------------------
    add("<h2>Textarea</h2>")
    add('<form id="form4" action="/action_page4.php">')
    add('  <label for="msg">Message:</label>')
    add('  <textarea id="msg" name="message" rows="4" cols="30">Default text</textarea>')
    add('  <label>Reviews: <textarea name="reviews" rows="3"></textarea></label>')
    add('  <button type="submit">Send</button>')
    add("</form>")
    # -- form 5: radios (0 actionnable v1 — 1 bouton) -------------------------
    add("<h2>Radio Buttons</h2>")
    add('<form id="form5" action="/action_page5.php">')
    add('  <input type="radio" id="html" name="fav_language" value="HTML">')
    add('  <label for="html">HTML</label>')
    add('  <input type="radio" id="css" name="fav_language" value="CSS">')
    add('  <label for="css">CSS</label>')
    add('  <button type="button" id="vote-btn">Vote</button>')
    add("</form>")
    # -- form 6: checkboxes + reset (1 actionnable) ---------------------------
    add("<h2>Checkboxes</h2>")
    add('<form id="form6" action="/action_page6.php">')
    add('  <input type="checkbox" id="vehicle1" name="vehicle1" value="Bike">')
    add('  <label for="vehicle1"> I have a bike</label>')
    add('  <input type="checkbox" id="vehicle2" name="vehicle2" value="Car">')
    add('  <label for="vehicle2"> I have a car</label>')
    add('  <button type="reset" id="reset-btn">Reset</button>')
    add("</form>")
    # -- champs hors formulaire (4 actionnables) ------------------------------
    add("<h2>More Input Types</h2>")
    add('  <label for="q">Search:</label>')
    add('  <input type="search" id="q" name="q">')
    add('  <label for="phone">Phone:</label>')
    add('  <input type="tel" id="phone" name="phone">')
    add('  <label for="pw">Password:</label>')
    add('  <input type="password" id="pw" name="pw">')
    add('  <label>Quantity: <input type="number" name="quantity"></label>')
    # -- non-actionnables variés (ignorés, définitions v1) --------------------
    add('  <input type="hidden" name="csrf" value="abc123">')
    add('  <input type="file" name="upload">')
    add('  <input name="legacy">')
    # -- liens de contenu ------------------------------------------------------
    for label, href in _CONTENT_LINKS:
        add(f'<p><a href="{href}">{label}</a></p>')
    add('<p><a name="anchor-no-href">Ancre nommée sans href</a></p>')
    add('<p><a class="disabled">Lien désactivé sans href</a></p>')
    add('  <button id="theme-btn" type="button">Dark mode</button>')
    # -- footer (liens complémentaires) ----------------------------------------
    add('<div id="footer">')
    for label, href in _footer():
        add(f'  <a href="{href}">{label}</a>')
    add('  <button id="run-btn" type="button">Try it editor</button>')
    add('  <input type="submit" id="newsletter-sub" value="Subscribe">')
    add("</div>")
    add("</body>")
    add("</html>")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- #

PAGES = [
    {"name": "httpbin-post", "url": "https://httpbin.org/forms/post",
     "file": "httpbin-post.html", "html": _httpbin_post},
    {"name": "example", "url": "https://example.com/",
     "file": "example.html", "html": _example},
    {"name": "httpbin", "url": "https://httpbin.org/",
     "file": "httpbin.html", "html": _httpbin},
    {"name": "w3schools-forms", "url": "https://www.w3schools.com/html/html_forms.asp",
     "file": "w3schools-forms.html", "html": _w3schools_forms},
]


def generate_pages() -> dict[str, str]:
    """name -> html (déterministe: aucun timestamp, aucun aléa)."""
    return {p["name"]: p["html"]() for p in PAGES}


def ensure_snapshots(out_dir: str | None = None, force: bool = False) -> str:
    """Écrit les snapshots + manifeste pages.json (idempotent) et vérifie la
    calibration v1 sur les pages générées. Retourne out_dir."""
    from ucm.web.compiler_probe import compile_page as v1_compile_page

    out_dir = out_dir or DEFAULT_OUT
    os.makedirs(out_dir, exist_ok=True)
    pages = generate_pages()

    # Vérification de calibration AVANT écriture: les définitions v1 doivent
    # reproduire exactement les compteurs persistés de la baseline 5.86%.
    for p in PAGES:
        _pi, actionable, covered = v1_compile_page(pages[p["name"]])
        want = CALIBRATION[p["name"]]
        assert (actionable, covered) == (want["actionable"], want["covered"]), (
            p["name"], actionable, covered, want)

    manifest = {
        "source": ("reconstructions statiques déterministes calibrées sur les "
                   "compteurs v1 (90a303a); étiquette: hors-ligne, pas un "
                   "fetch live"),
        "pages": [{"name": p["name"], "url": p["url"], "file": p["file"]}
                  for p in PAGES],
    }
    mpath = os.path.join(out_dir, "pages.json")
    if force or not os.path.exists(mpath):
        with open(mpath, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=1, ensure_ascii=False)
    for p in PAGES:
        path = os.path.join(out_dir, p["file"])
        if force or not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(pages[p["name"]])
    return out_dir


def __main() -> None:
    out = ensure_snapshots(force=True)
    print(json.dumps({"pages_dir": out,
                      "pages": [p["name"] for p in PAGES]}, indent=1))


if __name__ == "__main__":
    __main()
