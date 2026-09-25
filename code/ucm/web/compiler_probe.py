"""SONDE COMPILATEUR page->description: DOM reels -> policy_input."""
from __future__ import annotations
import json, os, re, time, urllib.request

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_OUT = os.path.join(_REPO, "artifacts/web-probe")

FORMS = [
    ("httpbin-post", "https://httpbin.org/forms/post"),
    ("example", "https://example.com"),
    ("httpbin", "https://httpbin.org"),
    ("w3schools-forms", "https://www.w3schools.com/html/html_forms.asp"),
]


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=20).read().decode("utf-8", "replace")


def parse_dom(html):
    tags = []
    for m in re.finditer(r"<(input|button|select|textarea|option|a|form|label)\b([^>]*)>",
                         html, re.I):
        tag, attrs_s = m.group(1).lower(), m.group(2)
        attrs = dict(re.findall(r'([\w-]+)\s*=\s*"([^"]*)"', attrs_s))
        tags.append({"tag": tag, "attrs": attrs, "pos": m.start()})
    return tags


def compile_page(html):
    tags = parse_dom(html)
    entities, relations, candidates = [], [], []
    actionable = covered = 0
    eid = form_n = 0
    entities.append({"id": "page", "type": "view", "attrs": {}})
    for t in tags:
        a = t["attrs"]
        tid = a.get("id") or a.get("name") or f"{t['tag']}{eid}"
        eid += 1
        kind, itype = t["tag"], (a.get("type") or "").lower()
        if kind == "form":
            form_n += 1
            entities.append({"id": f"form{form_n}", "type": "form", "attrs": {}})
            relations.append({"subj": f"form{form_n}", "pred": "on_view", "obj": "page"})
        elif kind == "label":
            continue
        elif kind in ("input", "textarea") and (kind == "textarea" or itype in
                ("text", "email", "password", "tel", "number", "search", "url", "date")):
            entities.append({"id": tid, "type": "field", "attrs": {}})
            relations.append({"subj": tid, "pred": "on_view", "obj": "page"})
            candidates.append({"action": "TYPE", "arg": tid})
            actionable += 1; covered += 1
        elif kind == "button" or (kind == "input" and itype in ("submit", "button", "reset")):
            entities.append({"id": tid, "type": "button", "attrs": {}})
            relations.append({"subj": tid, "pred": "on_view", "obj": "page"})
            candidates.append({"action": "CLICK", "arg": tid})
            actionable += 1; covered += 1
        elif kind == "select":
            entities.append({"id": tid, "type": "select", "attrs": {}})
            relations.append({"subj": tid, "pred": "on_view", "obj": "page"})
            actionable += 1
        elif kind == "option":
            entities.append({"id": f"opt-{tid}", "type": "option", "attrs": {}})
            relations.append({"subj": f"opt-{tid}", "pred": "option_of", "obj": "page"})
            candidates.append({"action": "SELECT", "arg": f"opt-{tid}"})
            covered += 1
        elif kind == "a" and a.get("href"):
            actionable += 1     # liens: NON couverts par UCM v1
    candidates.append({"action": "STOP", "arg": None})
    return ({"schema_version": "p2/0.1", "entities": entities,
             "relations": relations,
             "goal": {"predicate": "VIEW", "args": {"view": "page"}},
             "candidates": candidates}, actionable, covered)


def run_probe(out_dir=_OUT):
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    results = {}
    tot_a = tot_c = 0
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    from ucm.v1.tensorize_siw import tensorize_siw_obs as tensorize_obs_p2
    from ucm.model.tensorize import collate as _c
    def _collate_p2(x): return _c(x)
    for name, url in FORMS:
        try:
            html = fetch(url)
        except Exception as e:
            results[name] = {"error": str(e)[:100]}
            continue
        pi, a, c = compile_page(html)
        exec_ok = None
        try:
            ex = tensorize_obs_p2(pi); ex["labels"] = None
            _collate_p2([ex])
            exec_ok = True
        except Exception as e:
            exec_ok = f"FAIL: {str(e)[:80]}"
        results[name] = {"actionable": a, "covered": c,
                         "completeness": round(c / max(a, 1), 4),
                         "exec_tensorizes": exec_ok,
                         "n_candidates": len(pi["candidates"]) - 1}
        tot_a += a; tot_c += c
    overall = tot_c / max(tot_a, 1)
    art = {"ts": ts,
           "source": "fetch statique de pages publiques reelles (compilateur "
                     "DOM->policy_input; etiquette: pas via page-agent live)",
           "overall_completeness": round(overall, 4),
           "go_completeness_ge_90": overall >= 0.90,
           "totals": {"actionable": tot_a, "covered": tot_c},
           "pages": results}
    apath = os.path.join(out_dir, f"compiler-probe-{ts}.json")
    _fd = os.open(apath, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(_fd, "w") as fh:
        json.dump(art, fh, indent=1, sort_keys=True)
    art["persisted"] = apath
    return art


if __name__ == "__main__":
    print(json.dumps(run_probe(), indent=1, default=str))
