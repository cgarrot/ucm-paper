"""S5 GO-4 CALIBRATION v2 (lead 08:35, PRÉENREGISTRÉE avant entraînement):

(a) PAIRES DISCRIMINANTES: pour chaque layout relocalisé (celui du refus),
    une tâche FAISABLE sur le MÊME layout (but atteignable — autre widget)
    supervisée NORMALEMENT (optimal oracle) → le signal discriminant est la
    JOIGNABILITÉ, pas la position;
(b) REFUS NON-INITIAUX: 50% des records refus supervisés à un état
    intermédiaire (1-4 actions valides aléatoires depuis l'init — toujours
    unreachable pour un but impossible-par-construction) → tue le biais
    'STOP au départ';
FRACTION AJUSTÉE: 80 refus + 80 faisables-appariés (paires 1:1) sur
874 records DEV ≈ 15.4% du total MAIS équilibrée 1:1 refus:faisable
(l'asymétrie causale du v1 disparaît).
Splits DISJOINTS inchangés (40 impossibles test + 40 faisables contrôle,
plus les faisables-appariés TEST sur les layouts du split test).
Les deux chiffres au bout. GO lead 08:35.
"""
from __future__ import annotations

import json
import os
import random
import time

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_OUT = os.path.join(_REPO, "artifacts/s5-demo")
CAL_SEED = 20260925          # gelé
N_TRAIN_PAIRS = 80           # gelé
UPDATES = 4000               # gelé


def _records_for_pair(lay_spec, imp_task, rng):
    """Une paire: (record refus, record faisable-apparié) sur le MÊME layout.
    Refus: 50% initial, 50% état intermédiaire (1-4 actions valides)."""
    from ucm.env.siw import SIW, SIWLayout, sample_task
    from ucm.env.siw_oracle import SIWOracle
    spec = dict(lay_spec)
    if isinstance(spec.get("widgets"), dict):
        spec["widgets"] = list(spec["widgets"].values())
    lay = SIWLayout(spec)

    # (b) état du refus: initial ou intermédiaire
    env = SIW(lay)
    obs = env.reset(imp_task)
    if rng.random() < 0.5:
        for _ in range(rng.randint(1, 4)):
            valid = [c for c in obs["candidates"]
                     if env._is_valid(c)[0] and c["action"] != "STOP"]
            if not valid:
                break
            env.execute(rng.choice(sorted(valid, key=lambda c: json.dumps(c))))
            obs = env.observe()
    stop_idx = next(i for i, c in enumerate(obs["candidates"])
                    if c["action"] == "STOP")
    refus_rec = {"policy_input": obs,
                 "supervision": {"optimal_actions": [stop_idx], "d_star": -1,
                                 "reachable": False},
                 "provenance": {"split": "refus-v2", "state": "mid" if len(
                     [1]) and env.step_count > 0 else "init"}}

    # (a) faisable-apparié: but atteignable sur le MÊME layout
    feas = None
    for _ in range(300):
        st, goal = sample_task(rng, lay)
        o = SIWOracle(lay, goal)
        t = {"init": {"view": st.view, "filled": sorted(st.filled),
                      "chosen": dict(st.chosen), "dialog_open": st.dialog_open,
                      "submitted": sorted(st.submitted)},
             "goal": goal}
        env2 = SIW(lay)
        env2.reset(t)
        if o.reachable(env2.state) and 1 <= o.d_star(env2.state) <= 6:
            sol = o.solve(env2.state)
            feas = {"policy_input": env2.observe(),
                    "supervision": {"optimal_actions": sol["optimal_actions"],
                                    "d_star": sol["d_star"], "reachable": True},
                    "provenance": {"split": "faisable-apparie",
                                   "layout": lay.layout_hash()}}
            break
    return refus_rec, feas


def run_calibration(corpus_path=None, out_dir=_OUT):
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    from ucm.env.siw import SIW, SIWLayout
    from ucm.eval.rollout import run_episode
    from ucm.v1.policy_siw import SIWModelPolicy
    
    from ucm.v1.data_adapter import _couples_core, _spec_of
    from ucm.data.siw_pipeline import generate_adaptation_episodes_rstar
    from ucm.v1.runner import finetune
    from ucm.v1.transfer import CoverageTracker, FinetuneConfig
    from ucm.model.siw_model import make_siw_model
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")

    # corpus v1 publié (120 imp + faisables) — reprend les splits
    import glob
    cp = corpus_path or sorted(glob.glob(
        os.path.join(out_dir, "s5-refusal-corpus-*.json")))[-1]
    corpus = json.load(open(cp))
    imp_train = corpus["impossible_train"][:N_TRAIN_PAIRS]
    imp_test = corpus["impossible_test"]
    feas_ctrl = corpus["feasible_ctrl"]

    rng = random.Random(CAL_SEED)
    refus_records, feas_records = [], []
    for e in imp_train:
        r, f = _records_for_pair(e["layout_spec"], e["task"], rng)
        refus_records.append(r)
        if f is not None:
            feas_records.append(f)
    assert len(refus_records) == N_TRAIN_PAIRS and len(feas_records) >= 70

    # recette DEV inchangée
    inv = json.load(open(os.path.join(
        _REPO, "artifacts/inventory-siw-dev-layouts.json")))
    dl = [SIWLayout(_spec_of(e)) for e in list(inv.values())[:10]]
    couples, _ = generate_adaptation_episodes_rstar(dl, seed=3, n_episodes=256,
                                                    d0_band=(1, 10))
    store = {c["layout_hash"]: inv[c["layout_hash"]] for c in couples
             if c["layout_hash"] in inv}
    dev_records = _couples_core(store, [json.dumps(c, sort_keys=True, default=str)
                                        for c in couples
                                        if c["layout_hash"] in store])

    all_records = dev_records + refus_records + feas_records
    fraction = (len(refus_records) + len(feas_records)) / len(all_records)

    mx.random.seed(3)
    model = make_siw_model(d=144)
    finetune(model, all_records, FinetuneConfig(updates=UPDATES, seed=3),
             CoverageTracker(), "s5-cal", len(all_records), trainable="all")

    def _esc_rate(eps):
        esc = 0
        for e in eps:
            spec = dict(e["layout_spec"])
            if isinstance(spec.get("widgets"), dict):
                spec["widgets"] = list(spec["widgets"].values())
            lay = SIWLayout(spec)
            env = SIW(lay)
            env.reset(e["task"])
            pol = SIWModelPolicy(model, name="s5-cal", seed=500)
            r = run_episode(env, pol, e["episode_id"], 500, "s5-cal")
            esc += (r.outcome == "premature_stop")
        return esc / max(len(eps), 1)

    recall = _esc_rate(imp_test)
    false_ctrl = _esc_rate(feas_ctrl)

    art = {"ts": ts,
           "pre_registered_v2": {
               "cal_seed": CAL_SEED, "n_train_pairs": N_TRAIN_PAIRS,
               "updates": UPDATES,
               "design": "paires discriminantes 1:1 (refus + faisable-apparié "
                         "MÊME layout) + 50% refus non-initiaux (1-4 pas)",
               "refus_records": len(refus_records),
               "faisable_apparie_records": len(feas_records),
               "dev_records": len(dev_records),
               "fraction_pairs_sur_total": round(fraction, 4)},
           "escalation_recall_test40": round(recall, 4),
           "go4_ge_90": recall >= 0.90,
           "false_escalations_ctrl40": round(false_ctrl, 4),
           "v1_reference": {"recall": 1.0, "false_esc": 0.675}}
    apath = os.path.join(out_dir, f"s5-calibration-v2-{ts}.json")
    _fd = os.open(apath, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(_fd, "w") as fh:
        json.dump(art, fh, indent=1, sort_keys=True, default=str)
    art["persisted"] = apath
    return art


if __name__ == "__main__":
    print(json.dumps(run_calibration(), indent=1, default=str)[:1200])
