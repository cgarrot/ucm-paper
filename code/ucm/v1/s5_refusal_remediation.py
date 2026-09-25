"""S5 GO-4 REMÉDIATION (lead 08:02): superviser le refus.

{STOP} inconditionnel sur états vérifiés-unreachable (contrainte DSL du
lead: dialog fermé init + widget cible in_dialog — impossible PAR
CONSTRUCTION, cf 06ca583). MIX PRÉENREGISTRÉ: la fraction et les splits
sont gelés AVANT l'entraînement (ci-dessous, publiés dans l'artefact):
  - corpus: 120 impossibles (contrainte DSL, layouts SIW non vus,种子 fixe)
    + 120 FAISABLES APPARIÉS (mêmes layouts, buts atteignables oracle);
  - splits DISJOINTS: impossibles train 80 / test 40 (jamais les mêmes);
    faisables contrôle 40 (jamais entraînés avec STOP-sans-but);
  - mix: chaque épisode refus-train contribue SES records d'état initial
    supervisés {STOP} — fraction documentée = n_records_refus /
    (n_records_refus + n_records_DEV) mesurée et publiée;
  - ré-mesure: GO-4 sur 40 impossibles TEST + FAUSSES ESCALADES sur les 40
    faisables contrôle (exigence tagi-5: le refus ne doit pas casser
    l'exécution normale).
"""
from __future__ import annotations

import json
import os
import random
import time

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_OUT = os.path.join(_REPO, "artifacts/s5-demo")

GEN_SEED = 20260925          # gelé
N_IMPOSSIBLE = 120           # gelé: 80 train + 40 test (disjoints)
N_FEASIBLE_CTRL = 40         # gelé: contrôle fausses escalades
DEV_EPISODES = 256           # recette exécutant S5 inchangée
UPDATES = 4000


def _gen_impossible(lays, rng, n):
    """Contrainte DSL (06ca583, appliquée au NIVEAU LAYOUT — les générateurs
    frais ne placent que confirm/dismiss dans les dialogs): on RELOCALISE un
    field ou un submit DANS le dialog (in_dialog=dlg, view retiré) — but sur
    ce widget + init dialog fermé = unreachable PAR CONSTRUCTION, VÉRIFIÉ
    oracle (la vérification oracle reste le juge, indépendamment du
    constructeur)."""
    from ucm.env.siw import SIW, SIWLayout, sample_task
    from ucm.env.siw_oracle import SIWOracle
    out = []
    tries = 0
    while len(out) < n and tries < 50_000:
        tries += 1
        lay = lays[rng.randrange(len(lays))]
        spec = _spec_of_layout(lay)
        dlg_ids = [w["id"] for w in spec["widgets"] if w["type"] == "dialog"]
        if not dlg_ids:
            continue
        dlg = rng.choice(dlg_ids)
        fields = [w for w in spec["widgets"] if w["type"] == "field"]
        submits = [w for w in spec["widgets"]
                   if w["type"] == "button" and w.get("kind") == "submit"]
        rng.shuffle(fields)
        rng.shuffle(submits)
        variant = None
        if fields and rng.random() < 0.5:
            w = dict(fields[0])
            w["in_dialog"] = dlg
            w.pop("view", None)
            goal = {"predicate": "SET", "args": {"field": w["id"]}}
            gt = "SET_dialog_field"
            variant = w
        elif submits:
            w = dict(submits[0])
            w["in_dialog"] = dlg
            w.pop("view", None)
            goal = {"predicate": "SUBMITTED", "args": {"form": w["submit_for"]}}
            gt = "SUBMITTED_dialog_submit"
            variant = w
        if variant is None:
            continue
        spec["widgets"] = [x for x in spec["widgets"] if x["id"] != variant["id"]]
        spec["widgets"].append(variant)
        try:
            lay2 = SIWLayout(dict(spec, widgets=list(spec["widgets"])))
        except Exception:
            continue
        st, _ = sample_task(rng, lay2)
        o = SIWOracle(lay2, goal)
        t = {"init": {"view": st.view, "filled": sorted(st.filled),
                      "chosen": dict(st.chosen), "dialog_open": False,
                      "submitted": sorted(st.submitted)},
             "goal": goal}
        env = SIW(lay2)
        env.reset(t)
        if not o.reachable(env.state):
            out.append({"episode_id": f"refusal-{GEN_SEED}-{len(out):04d}",
                        "goal_type": gt, "layout_hash": lay2.layout_hash(),
                        "layout_spec": _spec_of_layout(lay2), "task": t,
                        "reachable": False})
    return out


def _spec_of_layout(lay):
    return {"views": list(lay.views), "nav_edges": [list(e) for e in lay.nav_edges],
            "widgets": [dict(w) for w in lay.widgets.values()],
            "labels": dict(lay.labels)}


def _gen_feasible(lays, rng, n):
    from ucm.env.siw import SIW, sample_task
    from ucm.env.siw_oracle import SIWOracle
    out = []
    tries = 0
    while len(out) < n and tries < 50_000:
        tries += 1
        lay = lays[rng.randrange(len(lays))]
        st, goal = sample_task(rng, lay)
        o = SIWOracle(lay, goal)
        t = {"init": {"view": st.view, "filled": sorted(st.filled),
                      "chosen": dict(st.chosen), "dialog_open": st.dialog_open,
                      "submitted": sorted(st.submitted)},
             "goal": goal}
        env = SIW(lay)
        env.reset(t)
        if o.reachable(env.state) and 1 <= o.d_star(env.state) <= 8:
            out.append({"episode_id": f"feasible-{GEN_SEED}-{len(out):04d}",
                        "layout_hash": lay.layout_hash(),
                        "layout_spec": _spec_of_layout(lay), "task": t,
                        "d_star": o.d_star(env.state), "reachable": True})
    return out


def run_remediation(out_dir=_OUT):
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    from ucm.env.siw import SIW, SIWLayout
    from ucm.eval.rollout import run_episode
    from ucm.v1.policy_siw import SIWModelPolicy
    from ucm.v1.s5_vision_demo import _train_executor, _unseen_layouts
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")

    rng = random.Random(GEN_SEED)
    lays = _unseen_layouts(8, seed=GEN_SEED + 1)
    impossible = _gen_impossible(lays, rng, N_IMPOSSIBLE)
    feasible = _gen_feasible(lays, rng, N_FEASIBLE_CTRL)
    assert len(impossible) == N_IMPOSSIBLE and len(feasible) == N_FEASIBLE_CTRL
    imp_train, imp_test = impossible[:80], impossible[80:]   # DISJOINTS

    # ---- MIX PRÉENREGISTRÉ (gelé avant entraînement) ----
    # exécutant DEV (recette S5) + records refus des 80 impossibles train:
    # à l'état initial de chaque épisode impossible, superviser {STOP}.
    # Fraction publiée = records_refus / total records.
    from ucm.v1.data_adapter import _couples_core, _spec_of
    from ucm.data.siw_pipeline import generate_adaptation_episodes_rstar
    from ucm.v1.runner import finetune
    from ucm.v1.transfer import CoverageTracker, FinetuneConfig
    from ucm.model.siw_model import make_siw_model

    inv = json.load(open(os.path.join(
        _REPO, "artifacts/inventory-siw-dev-layouts.json")))
    dl = [SIWLayout(_spec_of(e)) for e in list(inv.values())[:10]]
    couples, _ = generate_adaptation_episodes_rstar(dl, seed=3, n_episodes=DEV_EPISODES,
                                                    d0_band=(1, 10))
    store = {c["layout_hash"]: inv[c["layout_hash"]] for c in couples
             if c["layout_hash"] in inv}
    dev_records = _couples_core(store, [json.dumps(c, sort_keys=True, default=str)
                                        for c in couples
                                        if c["layout_hash"] in store])

    refusal_records = []
    for e in imp_train:
        lay = SIWLayout(dict(e["layout_spec"],
                             widgets=list(e["layout_spec"]["widgets"])))
        env = SIW(lay)
        obs = env.reset(e["task"])
        # supervision: {STOP} UNIQUE (indice du candidat STOP)
        stop_idx = next(i for i, c in enumerate(obs["candidates"])
                        if c["action"] == "STOP")
        refusal_records.append({
            "policy_input": obs,
            "supervision": {"optimal_actions": [stop_idx], "d_star": -1,
                            "reachable": False},
            "provenance": {"episode_id": e["episode_id"], "split": "refusal-train",
                           "source": "supervised-refusal (lead 08:02)"}})

    fraction = len(refusal_records) / (len(refusal_records) + len(dev_records))

    # entraînement: DEV + refus (le finetune SIW consomme les records bruts)
    all_records = dev_records + refusal_records
    mx.random.seed(3)
    model = make_siw_model(d=144)
    finetune(model, all_records, FinetuneConfig(updates=UPDATES, seed=3),
             CoverageTracker(), "s5-remed", len(all_records), trainable="all")

    # ---- (3) ré-mesure sur splits DISJOINTS ----
    def _eval(eps, want_escalation):
        esc = 0
        for e in eps:
            lay = SIWLayout(dict(e["layout_spec"],
                                 widgets=list(e["layout_spec"]["widgets"])))
            env = SIW(lay)
            env.reset(e["task"])
            pol = SIWModelPolicy(model, name="s5-remed", seed=500)
            r = run_episode(env, pol, e["episode_id"], 500, "s5-remed")
            if r.outcome == "premature_stop":
                esc += 1
        return esc

    esc_imp = _eval(imp_test, True)
    esc_fea = _eval(feasible, False)
    recall = esc_imp / len(imp_test)
    false_esc = esc_fea / len(feasible)

    art = {"ts": ts,
           "pre_registered": {"gen_seed": GEN_SEED,
                              "n_impossible": N_IMPOSSIBLE,
                              "split": "80 train / 40 test DISJOINTS",
                              "n_feasible_ctrl": N_FEASIBLE_CTRL,
                              "dev_episodes": DEV_EPISODES, "updates": UPDATES,
                              "refusal_records": len(refusal_records),
                              "dev_records": len(dev_records),
                              "refusal_fraction": round(fraction, 4)},
           "escalation_recall_test40": round(recall, 4),
           "go4_ge_90": recall >= 0.90,
           "false_escalations_ctrl40": round(false_esc, 4),
           "artefact_splits": {"imp_train_ids": [e["episode_id"] for e in imp_train],
                               "imp_test_ids": [e["episode_id"] for e in imp_test],
                               "feasible_ctrl_ids": [e["episode_id"] for e in feasible]}}
    apath = os.path.join(out_dir, f"s5-escalade-remediee-{ts}.json")
    _fd = os.open(apath, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(_fd, "w") as fh:
        json.dump(art, fh, indent=1, sort_keys=True, default=str)
    # corpus complet persisté (rejouabilité)
    cpath = os.path.join(out_dir, f"s5-refusal-corpus-{ts}.json")
    _fd = os.open(cpath, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(_fd, "w") as fh:
        json.dump({"impossible_train": imp_train, "impossible_test": imp_test,
                   "feasible_ctrl": feasible}, fh, sort_keys=True, default=str)
    art["persisted"] = apath
    art["corpus"] = cpath
    return art


if __name__ == "__main__":
    print(json.dumps(run_remediation(), indent=1, default=str)[:1600])
