"""P2-1 — Records d'entraînement par épisode (trajectoire optimale DSL) avec
contexte par bras + labels d'effet."""
from __future__ import annotations

from ucm.dsl.core import DSLInterpreter
from ucm.dsl.tgk_program import TGK_DSL_PROGRAM
from ucm.dsl.bfs import DSLBFS
from ucm.p2 import context_arms as ca
from ucm.p2.model_p2 import effect_label


def episode_records(ep: dict, ctx_ents=None, ctx_rels=None, max_steps=14):
    """Trajectoire optimale via l'INVARIANT CENTRALISÉ (lead 12:15): le
    record terminal {STOP} est garanti par ucm.dsl.records_util — ce
    générateur ne le construit plus lui-même (le bug 20:23 est né ici)."""
    from ucm.dsl.core import DSLInterpreter
    from ucm.dsl.tgk_program import TGK_DSL_PROGRAM
    from ucm.dsl.records_util import walk_optimal_plan, assert_terminal_stop_invariant
    from ucm.p2 import context_arms as ca
    from ucm.p2.model_p2 import effect_label

    interp = DSLInterpreter(TGK_DSL_PROGRAM, ep["layout_tables"])
    lay = _native_layout(ep["layout_tables"])
    out = []
    task = ep["task"]

    def emit(st, meta):
        obs = _obs_at(lay, st, task["goal"])
        if ctx_ents or ctx_rels:
            obs = ca.augment_obs(obs, ctx_ents or [], ctx_rels or [])
        cands = obs["candidates"]
        if meta["terminal"]:
            eff = [0] * len(cands)
        else:
            eff = []
            for c in cands:
                interp.state = st
                if c["action"] == "STOP":
                    eff.append(0)
                    continue
                spec = interp._action_spec(c["action"])
                valid = interp._guard_ok(spec, c)
                eff.append(effect_label(c["action"], c["arg"], valid))
        out.append({"policy_input": obs,
                    "supervision": {"optimal_actions": meta["optimal_actions"],
                                    "d_star": meta["d_star"], "reachable": True},
                    "effect_labels": eff,
                    "episode_id": ep["episode_id"]})

    all_meta = walk_optimal_plan(interp, task, max_steps=max_steps,
                                  emit_record=emit)
    assert_terminal_stop_invariant(all_meta)
    return out


def _native_layout(tables):
    from ucm.env.tinygraph import Layout
    adj = [tuple(x) for x in tables["adj"]]
    door = tables["door_room_list"]
    door_idx = next(i for i, e in enumerate(adj)
                    if tuple(sorted(e)) == tuple(sorted(door)))
    return Layout(tables["rooms"], adj, door_idx)


def _obs_at(lay, dsl_state, goal):
    """policy_input TGK native à l'état DSL (reset-only: reconstruit le
    task init et reset l'env natif)."""
    from ucm.env.tinygraph import TinyGraphKey
    init = {"agent": dsl_state.fields["agent"],
            "carried": dsl_state.fields["carried"],
            "key": dsl_state.fields["key_room"],
            "parcel": dsl_state.fields["parcel_room"],
            "door_locked": dsl_state.fields["door_locked"]}
    env = TinyGraphKey(lay)
    return env.reset({"init": init, "goal": goal})
