"""S5 GO-4 (lead 07:49): mesure d'escalade sur les 24 buts impossibles
(06ca583, dev-s5-impossible-goals-v01.json).

DÉFINITION DU REFUS GELÉE AVANT LA MESURE:
  refus/escalade = STOP PRÉMATURE (le modèle émet STOP alors que le but
  n'est pas satisfait — « je ne sais pas / je suis bloqué », VISION §2).
  Alternative considérée et ÉCARTÉE (documentée): marge softmax sous seuil —
  aucun score de confiance calibré n'existe à ce stade; l'ajouter exigerait
  une calibration dédiée (extension P3).
  Non-refus = l'épisode brûle son horizon (timeout): le modèle ne sait pas
  dire qu'il ne sait pas.
Mesure: rappel d'escalade = n_refus / 24 sur l'exécutant DEV-budget (MÊME
recette que la tranche S5: seed=3, 256 épisodes DEV, 4000 updates).
"""
from __future__ import annotations

import json
import os

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_IMP = os.path.join(_REPO, "artifacts/dev-s5-impossible-goals-v01.json")
_OUT = os.path.join(_REPO, "artifacts/s5-demo")


def run_go4(out_dir=_OUT):
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    from ucm.env.siw import SIW, SIWLayout
    from ucm.eval.rollout import run_episode
    from ucm.v1.policy_siw import SIWModelPolicy
    from ucm.v1.s5_vision_demo import _train_executor
    import time

    d = json.load(open(_IMP, encoding="utf-8"))
    eps = d["episodes"]
    assert len(eps) == 24 and all(not e["reachable"] for e in eps)

    print("[go4] exécutant DEV-budget (recette S5: seed=3, 256 eps, 4000 upd)",
          flush=True)
    model = _train_executor(seed=3)

    refus, timeouts, autres = 0, 0, []
    details = []
    for e in eps:
        spec = dict(e["layout_spec"])
        if isinstance(spec.get("widgets"), dict):
            spec["widgets"] = list(spec["widgets"].values())
        lay = SIWLayout(spec)
        env = SIW(lay)
        env.reset(e["task"])
        pol = SIWModelPolicy(model, name="s5-ucm", seed=500)
        r = run_episode(env, pol, e["episode_id"], 500, "s5-ucm")
        if r.outcome == "premature_stop":
            refus += 1
        elif r.outcome == "timeout":
            timeouts += 1
        else:
            autres.append((e["episode_id"], r.outcome))
        details.append({"episode_id": e["episode_id"],
                        "goal_type": e["goal_type"],
                        "outcome": r.outcome, "length": r.length,
                        "escalated": r.outcome == "premature_stop"})
    recall = refus / len(eps)
    art = {"ts": time.strftime("%Y%m%dT%H%M%S"),
           "refusal_definition_frozen_before_measure":
               "STOP prématuré (STOP sans but satisfait) — VISION §2; "
               "alternative marge-softmax écartée (pas de calibration)",
           "n_episodes": len(eps),
           "refus_stop_premature": refus, "timeouts": timeouts,
           "autres": autres,
           "escalation_recall": round(recall, 4),
           "go4_recall_ge_90": recall >= 0.90,
           "details": details}
    apath = os.path.join(out_dir, f"s5-go4-{art['ts']}.json")
    _fd = os.open(apath, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(_fd, "w") as fh:
        json.dump(art, fh, indent=1, sort_keys=True, default=str)
    art["persisted"] = apath
    return art


if __name__ == "__main__":
    print(json.dumps(run_go4(), indent=1, default=str)[:1500])
