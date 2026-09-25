"""S5 TRANCHE FINE (lead 05:59-1) — le chemin VISION de bout en bout:

  VOIX (transcript) → Jev (COMPREND: but typé) → UCM (EXÉCUTE) sur SIW non vus.

Jev est EXTERNE (TypeSafe, hors du labo): dans cette tranche fine son rôle
(COMPREND → but typé) est joué par un COMPILATEUR DÉTERMINISTE de transcripts
vers buts typés SIW — documenté comme stub honnête: la tranche mesure
l'EXÉCUTION UCM (le petit modèle) contre ses baselines, pas la compréhension.

Baselines et GO chiffrés (gelés par le lead):
  - planner BFS   = SIWOracle (DSL-vérifié, jalon 2)
  - Jev-par-pas   = coût modélisé: le gros modèle appelé à CHAQUE décision
                    (coût par appel documenté dans le veilleur; on mesure le
                    ratio coût total UCM vs Jev-par-pas sur les mêmes épisodes)
  - GO: exec ≥ planner − 3pp ∧ p95 ≤ 5 ms ∧ ≥10× moins cher que Jev-par-pas
         ∧ rappel escalade ≥ 90% sur buts impossibles
Escalade: STOP-prématuré = « je ne sais pas / je suis bloqué » → escalade au
gros modèle (VISION §2: UCM sait dire qu'il ne sait pas).
"""
from __future__ import annotations

import json
import os
import random
import time

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_OUT = os.path.join(_REPO, "artifacts/s5-demo")

# Transcript → but typé (rôle Jev, stub déterministe documenté)
TRANSCRIPT_TO_GOAL = [
    ("va sur la vue deux", {"predicate": "VIEW", "args": {"view": "vw2"}}),
    ("remplis le champ cx", {"predicate": "SET", "args": {"field": "fx0"}}),
    ("choisis lo dans le menu ms", {"predicate": "CHOOSE",
                                    "args": {"select": "sx0", "option": "ox0"}}),
    ("envoie le formulaire fm", {"predicate": "SUBMITTED", "args": {"form": "fm0"}}),
]


_SPOKEN_NUM = {"zéro": "0", "zero": "0", "un": "1", "une": "1", "deux": "2",
               "trois": "3", "quatre": "4", "cinq": "5", "six": "6",
               "sept": "7", "huit": "8", "neuf": "9"}


def normalize_spoken_numbers(text: str) -> str:
    """« ox zéro » → « ox0 »: le STT épelle les IDs à voix haute (lead 15:32)
    — normalise nombre-parlé→chiffre avant le matching."""
    out = []
    for tok in text.lower().replace("'", " ").split():
        out.append(_SPOKEN_NUM.get(tok, tok))
    # recolle les paires id+chiffre collées: ['ox', '0'] → 'ox0'
    res, i = [], 0
    while i < len(out):
        if (i + 1 < len(out) and out[i].isalpha() and out[i + 1].isdigit()
                and len(out[i + 1]) == 1):
            res.append(out[i] + out[i + 1])
            i += 2
        else:
            res.append(out[i])
            i += 1
    return " ".join(res)


def jev_stub(transcript: str) -> dict:
    """Rôle Jev (COMPREND): transcript → but typé. Déterministe, documenté
    comme stub — la tranche mesure l'EXÉCUTION, pas la compréhension.
    Normalise les nombres parlés (fix lead 15:32) et résout les LIBELLÉS
    lisibles (l'utilisateur dit le label, pas l'ID technique)."""
    t = normalize_spoken_numbers(transcript)
    VERB_TO_GOAL = {"va": "VIEW", "remplis": "SET", "choisis": "CHOOSE",
                    "envoie": "SUBMITTED"}
    for verb, pred in VERB_TO_GOAL.items():
        if verb in t:
            tmpl = next(g for p, g in TRANSCRIPT_TO_GOAL
                        if g["predicate"] == pred)
            g = json.loads(json.dumps(tmpl))
            # référence parlée: le dernier token (id normalisé OU libellé) —
            # la résolution label→id se fait côté appelant (layout)
            g["_spoken_ref"] = t.split()[-1] if len(t.split()) > 1 else None
            return {"goal": g, "source": "jev_stub", "transcript": transcript}
    raise ValueError(f"transcript non compris: {transcript!r}")


def _unseen_layouts(n, seed=777):
    """Layouts SIW NON VUS (génération fraîche, hors DEV30/inventaire)."""
    from ucm.env.siw import SIWLayout, generate_siw_layout
    rng = random.Random(seed)
    seen = set(json.load(open(os.path.join(
        _REPO, "artifacts/inventory-siw-dev-layouts.json"))).keys())
    out = []
    while len(out) < n:
        lay = generate_siw_layout(rng, n_views=3, with_dialog=True)
        h = lay.layout_hash()
        if h not in seen:
            out.append(lay)
            seen.add(h)
    return out


def _train_executor(seed=3):
    """UCM exécutant SIW: B144-class SIW entraîné sur couples DEV (jamais
    sur les layouts non vus)."""
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    from ucm.model.siw_model import make_siw_model
    from ucm.v1.runner import finetune
    from ucm.v1.transfer import CoverageTracker, FinetuneConfig
    from ucm.env.siw import SIWLayout
    from ucm.data.siw_pipeline import generate_adaptation_episodes_rstar
    from ucm.v1.data_adapter import _couples_core, _spec_of
    inv = json.load(open(os.path.join(
        _REPO, "artifacts/inventory-siw-dev-layouts.json")))
    lays = [SIWLayout(_spec_of(e)) for e in list(inv.values())[:10]]
    couples, _ = generate_adaptation_episodes_rstar(lays, seed=seed, n_episodes=256,
                                                    d0_band=(1, 10))
    lines = [json.dumps(c, sort_keys=True, default=str) for c in couples]
    store = {c["layout_hash"]: inv[c["layout_hash"]] for c in couples
             if c["layout_hash"] in inv}
    records = _couples_core(store, [l for l in lines
                                    if json.loads(l)["layout_hash"] in store])
    mx.random.seed(seed)
    m = make_siw_model(d=144)
    finetune(m, records, FinetuneConfig(updates=4000, seed=seed),
             CoverageTracker(), "s5-exec", len(records), trainable="all")
    return m


def run_s5(n_layouts=6, n_tasks_per_layout=4, seed=3, out_dir=_OUT):
    import mlx.core as mx
    import numpy as np
    mx.set_default_device(mx.cpu)
    from ucm.env.siw import SIW, SIWLayout, sample_task
    from ucm.env.siw_oracle import SIWOracle
    from ucm.eval.rollout import run_episode
    from ucm.v1.policy_siw import SIWModelPolicy
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")

    model = _train_executor(seed)
    lays = _unseen_layouts(n_layouts, seed=991)
    rng = random.Random(seed + 1)

    # tâches possibles + IMPOSSIBLES (init↔but injoignable — vérifié oracle)
    tasks, impossible = [], []
    for lay in lays:
        for _ in range(2 * n_tasks_per_layout):
            st, goal = sample_task(rng, lay)
            o = SIWOracle(lay, goal)
            t = {"init": {"view": st.view, "filled": sorted(st.filled),
                          "chosen": dict(st.chosen), "dialog_open": st.dialog_open,
                          "submitted": sorted(st.submitted)},
                 "goal": goal}
            if o.reachable(st) and 1 <= o.d_star(st) <= 8:
                tasks.append((lay, t, o.d_star(st)))
            elif not o.reachable(st) and len(impossible) < 12:
                impossible.append((lay, t))
    # compléter les impossibles: buts valides-référencés mais injoignables
    # (ex: SET un champ de dialog alors que le dialog ne peut PAS s'ouvrir —
    # vérifié oracle)
    tries = 0
    for lay in lays:
        while len(impossible) < 12 and tries < 5000:   # BORNÉ (bug: boucle
            tries += 1                                  # infinie si aucun
            st, goal = sample_task(rng, lay)            # injoignable trouvé)
            o = SIWOracle(lay, goal)
            t = {"init": {"view": st.view, "filled": sorted(st.filled),
                          "chosen": dict(st.chosen), "dialog_open": st.dialog_open,
                          "submitted": sorted(st.submitted)},
                 "goal": goal}
            if not o.reachable(st):
                impossible.append((lay, t))
                break
    # IMPOSSIBLES — scénario produit « UI cassée » (lead GO 4): l'utilisateur
    # demande un champ dans un dialog dont on a RETIRÉ le bouton d'ouverture
    # (variant synthétique du layout, documentée) — l'oracle confirme
    # l'injoignabilité; UCM doit ESCALADER (STOP prématuré), pas brûler 64 pas.
    if not impossible:
        import copy
        for lay in lays:
            if len(impossible) >= 12:
                break
            dlgs = {w["id"] for w in lay.widgets.values() if w["type"] == "dialog"}
            flds = [w for w in lay.widgets.values()
                    if w["type"] == "field" and w.get("in_dialog") in dlgs]
            openers = [w for w in lay.widgets.values()
                       if w["type"] == "button" and w.get("kind") in ("confirm", "dismiss")]
            if not (flds and openers):
                continue
            spec = {"views": list(lay.views),
                    "nav_edges": [list(e) for e in lay.nav_edges],
                    "widgets": [dict(w) for w in lay.widgets.values()],
                    "labels": dict(lay.labels)}
            # retire TOUT bouton confirm/dismiss → le dialog ne peut plus fermer/
            # rouvrir après fermeture; init dialog_open=False → champ injoignable
            spec["widgets"] = [w for w in spec["widgets"]
                               if not (w["type"] == "button"
                                       and w.get("kind") in ("confirm", "dismiss"))]
            from ucm.env.siw import SIWLayout as _SL
            try:
                lay2 = _SL(spec)
            except Exception:
                continue
            st, _ = sample_task(rng, lay)
            goal = {"predicate": "SET", "args": {"field": flds[0]["id"]}}
            o2 = SIWOracle(lay2, goal)
            t = {"init": {"view": st.view, "filled": sorted(st.filled),
                          "chosen": dict(st.chosen), "dialog_open": False,
                          "submitted": sorted(st.submitted)},
                 "goal": goal}
            if not o2.reachable(st):
                impossible.append((lay2, t))
    print(f"[s5] tâches possibles: {len(tasks)} | impossibles: {len(impossible)}",
          flush=True)

    # GO 1: exec vs planner BFS (SIWOracle suit le plan — succès par définition
    # si le plan existe; le planner EST la borne haute: succès=100% par
    # construction sur tâches possibles — le GO mesure exec ≥ planner−3pp)
    exec_succ = 0
    latencies = []
    n_dec = 0
    for lay, task, d0 in tasks:
        env = SIW(lay)
        env.reset(task)
        pol = SIWModelPolicy(model, name="s5-ucm", seed=500)
        t0 = time.perf_counter()
        r = run_episode(env, pol, f"s5-{d0}", 500, "s5-ucm", d_star=d0)
        dt = time.perf_counter() - t0
        exec_succ += r.success
        n_dec += r.length
        latencies.append(dt / max(r.length, 1))
    p95 = sorted(latencies)[int(0.95 * len(latencies)) - 1] * 1000 if latencies else None
    planner_rate = 1.0   # BFS sur tâches possibles = borne haute

    # GO 3: coût vs Jev-par-pas (modèle: 1 appel Jev ≈ 1200 ms/décision —
    # System One Models, veille 2026; documenté, paramétrable)
    JEV_MS_PER_DECISION = 1200.0
    ucm_ms_total = sum(latencies) * 1000 * (n_dec / max(len(latencies), 1))
    jev_ms_total = JEV_MS_PER_DECISION * n_dec
    cost_ratio = jev_ms_total / max(ucm_ms_total, 1e-9)

    # GO 4: rappel d'escalade sur buts impossibles (STOP prématuré = escalade)
    escalations = 0
    for lay, task in impossible:
        env = SIW(lay)
        env.reset(task)
        pol = SIWModelPolicy(model, name="s5-ucm", seed=500)
        r = run_episode(env, pol, "s5-imp", 500, "s5-ucm")
        if r.outcome == "premature_stop":
            escalations += 1
    esc_recall = escalations / max(len(impossible), 1)

    exec_rate = exec_succ / max(len(tasks), 1)
    art = {"ts": ts, "n_tasks": len(tasks), "n_impossible": len(impossible),
           "n_decisions": n_dec,
           "exec_success": round(exec_rate, 4),
           "planner_bfs_success": planner_rate,
           "go1_exec_ge_planner_minus3pp": exec_rate >= planner_rate - 0.03,
           "p95_ms_per_decision": round(p95, 3) if p95 else None,
           "go2_p95_le_5ms": (p95 is not None and p95 <= 5.0),
           "jev_ms_per_decision_model": JEV_MS_PER_DECISION,
           "cost_ratio_vs_jev_per_step": round(cost_ratio, 1),
           "go3_10x_cheaper": cost_ratio >= 10.0,
           "escalation_recall": round(esc_recall, 4),
           "go4_escalation_ge_90": esc_recall >= 0.90,
           "notes": "Jev=stub déterministe documenté (la tranche mesure "
                    "l'EXÉCUTION); planner BFS = borne haute par construction "
                    "sur possibles; escalade = STOP prématuré"}
    art["GO"] = all([art["go1_exec_ge_planner_minus3pp"], art["go2_p95_le_5ms"],
                     art["go3_10x_cheaper"], art["go4_escalation_ge_90"]])
    apath = os.path.join(out_dir, f"s5-{ts}.json")
    _fd = os.open(apath, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(_fd, "w") as fh:
        json.dump(art, fh, indent=1, sort_keys=True)
    art["persisted"] = apath
    return art


if __name__ == "__main__":
    print(json.dumps(run_s5(), indent=1, default=str))
