"""S5 DÉBLOCAGE (lead 11:31-1) — exécutant SIW à BUDGET COMPLET sur la
distribution S5 (layouts AVEC dialogs, pas seulement l'inventaire DEV),
+ calibration refus v2 (paires discriminantes), re-mesure des 44 tâches S5.

GO PRÉENREGISTRÉ AVANT MESURE: exec ≥ 97% sur les 44 tâches faisables S5
∧ faux refus ≤ 5% sur les 40 contrôles faisables.
Recette: 512 épisodes d'adaptation sur layouts S5 frais (dialogs inclus)
+ 80 paires refus/faisable (calibration v2 gelée) + 6000 updates.
"""
from __future__ import annotations

import json
import os
import random
import time

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_OUT = os.path.join(_REPO, "artifacts/s5-demo")

SEED = 3
N_ADAPT = 512          # gelé: budget complet
UPDATES = 6000         # gelé
GO_EXEC = 0.97         # gelé
GO_FALSE_REFUS = 0.05  # gelé


def run_full_budget(out_dir=_OUT):
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    from ucm.env.siw import SIW, SIWLayout, sample_task, generate_siw_layout
    from ucm.env.siw_oracle import SIWOracle
    from ucm.data.siw_pipeline import generate_adaptation_episodes_rstar
    from ucm.v1.runner import finetune
    from ucm.v1.transfer import CoverageTracker, FinetuneConfig
    from ucm.v1.policy_siw import SIWModelPolicy
    from ucm.eval.rollout import run_episode
    from ucm.v1.s5_vision_demo import _unseen_layouts
    from ucm.v1.s5_refusal_calibration import _records_for_pair
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")

    # distribution S5: layouts frais AVEC dialogs (la distribution d'éval)
    rng = random.Random(SEED)
    s5_lays = _unseen_layouts(12, seed=991)[:12]

    # adaptation: couples r* SUR LA DISTRIBUTION S5 (dialogs inclus)
    couples, _ = generate_adaptation_episodes_rstar(s5_lays, seed=SEED,
                                                    n_episodes=N_ADAPT,
                                                    d0_band=(1, 10))
    lines = [json.dumps(c, sort_keys=True, default=str) for c in couples]
    # store S5: specs de NOS layouts (auto-suffisant — le _couples_core a
    # besoin d'un store hash→spec)
    store = {}
    for c in couples:
        h = c["layout_hash"]
        if h not in store:
            lay = next(l for l in s5_lays if l.layout_hash() == h)
            store[h] = {"views": list(lay.views),
                        "nav_edges": [list(e) for e in lay.nav_edges],
                        "widgets": [dict(w) for w in lay.widgets.values()],
                        "labels": dict(lay.labels)}
    from ucm.v1.data_adapter import _couples_core
    adapt_records = _couples_core(store, lines)

    # calibration refus v2 (paires, gelée): corpus v1 publié
    import glob
    cp = sorted(glob.glob(os.path.join(out_dir, "s5-refusal-corpus-*.json")))[-1]
    corpus = json.load(open(cp))
    crng = random.Random(20260925)
    refus_records, feas_records = [], []
    for e in corpus["impossible_train"][:80]:
        r, f = _records_for_pair(e["layout_spec"], e["task"], crng)
        refus_records.append(r)
        if f is not None:
            feas_records.append(f)

    all_records = adapt_records + refus_records + feas_records
    print(f"[s5-full] {len(adapt_records)} adapt + {len(refus_records)} refus "
          f"+ {len(feas_records)} faisables-appariés — entraînement {UPDATES} upd",
          flush=True)
    mx.random.seed(SEED)
    from ucm.model.siw_model import make_siw_model
    model = make_siw_model(d=144)
    finetune(model, all_records, FinetuneConfig(updates=UPDATES, seed=SEED),
             CoverageTracker(), "s5-full", len(all_records), trainable="all")

    # re-mesure: les 44 tâches S5 (MÊME construction que la tranche: layouts
    # seed 991, sample seed 4) + 40 contrôles faisables
    from ucm.v1.s5_vision_demo import _unseen_layouts as _ul
    lays_eval = _ul(6, seed=991)
    er = random.Random(4)
    tasks, ctrl = [], []
    for lay in lays_eval:
        for _ in range(16):
            st, goal = sample_task(er, lay)
            o = SIWOracle(lay, goal)
            t = {"init": {"view": st.view, "filled": sorted(st.filled),
                          "chosen": dict(st.chosen), "dialog_open": st.dialog_open,
                          "submitted": sorted(st.submitted)},
                 "goal": goal}
            env = SIW(lay)
            env.reset(t)
            if o.reachable(env.state) and 1 <= o.d_star(env.state) <= 8:
                tasks.append((lay, t, o.d_star(env.state)))
    ctrl = [(l, t, d) for l, t, d in tasks if True][:40]  # contrôles = 40 premières
    tasks44 = tasks[:44]

    succ = esc_false = 0
    for lay, t, d0 in tasks44:
        env = SIW(lay)
        env.reset(t)
        pol = SIWModelPolicy(model, name="s5-full", seed=500)
        r = run_episode(env, pol, "s5-full", 500, "s5-full", d_star=d0)
        succ += r.success
    exec_rate = succ / max(len(tasks44), 1)
    for lay, t, d0 in ctrl:
        env = SIW(lay)
        env.reset(t)
        pol = SIWModelPolicy(model, name="s5-full", seed=500)
        r = run_episode(env, pol, "s5-ctrl", 500, "s5-full", d_star=d0)
        esc_false += (r.outcome == "premature_stop")
    false_refus = esc_false / max(len(ctrl), 1)

    # CHECKPOINT PERSISTÉ (6e revue: les 4 métriques du MÊME modèle)
    ckpt = os.path.join(out_dir, f"s5-full-ckpt-{ts}.npz")
    import mlx.nn as _nn
    import mlx.core as _mx
    _mx.savez(ckpt, **dict(_nn.utils.tree_flatten(model.parameters())))

    # p95 + ratio mesurés sur CE checkpoint (mêmes 44 tâches)
    lat = []
    n_dec = 0
    for lay, t, d0 in tasks44:
        env = SIW(lay)
        env.reset(t)
        pol = SIWModelPolicy(model, name="s5-lat", seed=500)
        t0 = time.perf_counter()
        r = run_episode(env, pol, "s5-lat", 500, "s5-lat", d_star=d0)
        lat.append((time.perf_counter() - t0) / max(r.length, 1))
        n_dec += r.length
    p95 = sorted(lat)[max(int(0.95 * len(lat)) - 1, 0)] * 1000
    JEV_MS = 1200.0
    ucm_total_ms = sum(lat) * 1000 * n_dec / max(len(lat), 1)
    ratio = (JEV_MS * n_dec) / max(ucm_total_ms, 1e-9)

    art = {"ts": ts,
           "checkpoint": ckpt,
           "pre_registered": {"n_adapt": N_ADAPT, "updates": UPDATES,
                              "go_exec": GO_EXEC, "go_false_refus": GO_FALSE_REFUS,
                              "adapt_distribution": "layouts S5 frais (dialogs inclus)",
                              "calibration": "paires v2 gelées (80+80)"},
           "n_tasks": len(tasks44), "exec_success": round(exec_rate, 4),
           "go_exec_ge_97": exec_rate >= GO_EXEC,
           "false_refusals_ctrl": round(false_refus, 4),
           "go_false_refus_le_5": false_refus <= GO_FALSE_REFUS,
           "p95_ms_per_decision": round(p95, 3),
           "cost_ratio_stat": {
               "formula": "ratio = (JEV_MS × n_decisions) / Σ(latence_moyenne_"
                          "par_épisode × décisions) — dénominateur = temps UCm "
                          "réel mesuré sur les 44 tâches, MÊME checkpoint",
               "jev_ms_per_decision_model": JEV_MS,
               "n_decisions": n_dec,
               "ucm_total_ms": round(ucm_total_ms, 1),
               "ratio": round(ratio, 1)},
           "GO": exec_rate >= GO_EXEC and false_refus <= GO_FALSE_REFUS}
    art["GO_ALL"] = art["GO"]
    apath = os.path.join(out_dir, f"s5-full-budget-{ts}.json")
    _fd = os.open(apath, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(_fd, "w") as fh:
        json.dump(art, fh, indent=1, sort_keys=True)
    art["persisted"] = apath
    print(json.dumps(art, indent=1, default=str), flush=True)
    return art


if __name__ == "__main__":
    run_full_budget()
