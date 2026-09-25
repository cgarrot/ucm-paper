"""ASSEMBLAGE DÉMO BOUT EN BOUT (lead 20:07) — VRAI NAVIGATEUR.

Chaîne: WebBridge snapshot (DOM live) → compilateur v2 → UCM (s5-full)
→ décision → pont candidat→action WebBridge (click/fill natifs).

Mappings documentés (shims honnêtes):
  - entité 'link' → type 'button' pour le modèle SIW (vocab fermé; le
    modèle CLIQUE les deux — le mapping est tracé dans l'artefact);
  - SELECT web: exécuté par click sur l'<option> (sélecteur d'option);
  - NAVIGATE: tool navigate (cible = href résolu par le compilateur).
Couche voix: transcripts étiquetés (Muse externe — adaptateur testé côté
agent_viso_v2, branchement réel = pièce d'intégration suivante).
Jev: stub étiqueté (latence mesurée).
"""
from __future__ import annotations

import json
import os
import time
import urllib.request

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DAEMON = "http://127.0.0.1:10087"


def tool(name, args, timeout=30):
    req = urllib.request.Request(
        f"{DAEMON}/api/tool",
        data=json.dumps({"name": name, "args": args}).encode(),
        headers={"content-type": "application/json"})
    out = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    if isinstance(out, dict) and out.get("error"):
        raise RuntimeError(f"WebBridge {name}: {out['error']}")
    return out.get("data", out)


def snapshot_html():
    d = tool("snapshot", {})
    return d["html"], d["tabId"]


def compile_live(url=None):
    """DOM live → policy_input v2 (+ stats)."""
    import ucm.web.compiler_v2 as c2
    html, tab_id = snapshot_html()
    pi, stats = c2.compile_page(html, url=url)
    return pi, stats, html, tab_id


def _shim_for_model(policy_input):
    """Adapte le policy_input web au vocab SIW du modèle (link→button),
    copie annotée — l'original reste la référence."""
    import copy
    pi = copy.deepcopy(policy_input)
    for e in pi["entities"]:
        if e["type"] == "link":
            e["type"] = "button"
            e.setdefault("attrs", {})["web_kind"] = "link"
    return pi


class UCMBrowserPolicy:
    """UCM (s5-full) sur policy_input web compilé."""

    def __init__(self, model):
        self.model = model

    def decide(self, policy_input):
        import numpy as np
        from ucm.v1.tensorize_siw import tensorize_siw_obs, collate_siw
        pi = _shim_for_model(policy_input)
        ex = tensorize_siw_obs(pi)
        ex["labels"] = None
        b = collate_siw([ex])
        import mlx.core as mx
        logits = self.model(b)
        out = np.asarray(logits[0].tolist())
        return policy_input["candidates"][int(np.argmax(out))]


def load_s5_model():
    import glob
    import mlx.core as mx
    import mlx.nn as nn
    from ucm.model.siw_model import make_siw_model
    cks = sorted(glob.glob(os.path.join(
        _REPO, "artifacts/s5-demo", "s5-full-ckpt-*.npz")))
    if not cks:
        raise RuntimeError("ckpt s5-full requis")
    m = make_siw_model(d=144)
    w = dict(mx.load(cks[-1]))
    own = dict(nn.utils.tree_flatten(m.parameters()))
    assert set(own) == set(w)
    m.update(nn.utils.tree_unflatten(sorted(w.items())))
    mx.eval(m.parameters())
    return m


def execute_candidate(cand, policy_input, tab_id):
    """Pont candidat UCM → outils natifs WebBridge. Retourne le résultat."""
    kind, arg = cand["action"], cand["arg"]
    if kind == "STOP":
        return {"executed": "stop"}
    # retrouver l'entité cible (id dans le policy_input compilé)
    ent = next((e for e in policy_input["entities"] if e["id"] == arg), None)
    if ent is None:
        return {"executed": None, "reason": "entité inconnue"}
    if kind == "CLICK" or (kind == "TYPE" and ent["type"] == "button"):
        sel = _selector_for(ent)
        r = tool("click", {"selector": sel})
        return {"executed": "click", "selector": sel, "result": r}
    if kind == "TYPE":
        sel = _selector_for(ent)
        r = tool("fill", {"selector": sel, "value": "demo"})
        return {"executed": "fill", "selector": sel, "result": r}
    if kind == "SELECT":
        sel = _selector_for(ent)
        r = tool("click", {"selector": sel})
        return {"executed": "select-click", "selector": sel, "result": r}
    if kind == "NAVIGATE":
        href = ent.get("attrs", {}).get("href_resolved") or ent.get("attrs", {}).get("href")
        if href:
            r = tool("navigate", {"url": href})
            return {"executed": "navigate", "url": href, "result": r}
        return {"executed": None, "reason": "pas de href"}
    return {"executed": None, "reason": f"kind {kind}"}


def _selector_for(ent):
    eid = ent["id"]
    base = eid.rstrip("0123456789_") or eid
    if base == eid:
        return f"#{eid}" if not eid[0].isdigit() else f"[id='{eid}']"
    return f"[id='{eid}'], #{base}"


def run_smoke(out_dir=None):
    """Smoke bout-en-bout: page réelle → compile → UCM décide → exécution."""
    out_dir = out_dir or os.path.join(_REPO, "artifacts/web-probe/e2e")
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    model = load_s5_model()
    pi, stats, html, tab_id = compile_live()
    pol = UCMBrowserPolicy(model)
    t0 = time.perf_counter()
    cand = pol.decide(pi)
    dt = (time.perf_counter() - t0) * 1000
    ex = execute_candidate(cand, pi, tab_id)
    art = {"ts": ts, "tab_id": tab_id,
           "page_actionables": stats["actionable"],
           "n_candidates": len(pi["candidates"]),
           "decision": cand, "decision_ms": round(dt, 3),
           "execution": ex, "shims": ["link→button (vocab SIW)"]}
    ap = os.path.join(out_dir, f"e2e-smoke-{ts}.json")
    _fd = os.open(ap, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(_fd, "w") as fh:
        json.dump(art, fh, indent=1, sort_keys=True, default=str)
    art["persisted"] = ap
    return art


if __name__ == "__main__":
    print(json.dumps(run_smoke(), indent=1, default=str))
