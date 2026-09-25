"""Recherche A: profondeur 2×2 DEV — données fraîches d*13-24 + beam search w≤8.

Évalue les ckpts B144-oracle existants sur de NOUVELLES tâches d*13-24 TGK
générées par DSL, avec et sans beam search w≤8 sur transitions DSL.
GO si ≥+15pp à p95≤20ms; KILL si <+5pp; données-seules ≥80% ⇒ §13.4 inutile.
"""
from __future__ import annotations
import argparse, hashlib, json, random, subprocess, sys, time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ucm.env.tinygraph import Layout, PhysicalState, TinyGraphKey, task_goal, goal_satisfied
from ucm.env.oracle import solve as tgk_solve, enumerate_states, candidate_actions
from ucm.model.gnn_b import GNNB
from ucm.model.tensorize import tensorize_obs
from ucm.data.writer import seal_manifest

N_SEEDS = 5
BEAM_WIDTH = 8
HORIZON = 64


def load_b144(seed_idx: int) -> GNNB:
    path = f"artifacts/s2b-factorial/ckpts-run3/b144-oracle-s10{seed_idx}.npz"
    state = np.load(path)
    model = GNNB(d=144, n_blocks=3)
    model.update(nn.utils.tree_unflatten([(k, mx.array(v)) for k, v in state.items()]))
    mx.eval(model.parameters())
    return model


def policy_action(model, env) -> tuple:
    """Greedy: utilise ModelPolicy (interface correcte pour GNNB/DeepSetsA)."""
    from ucm.eval.rollout import ModelPolicy
    pol = _POLICY_CACHE.get("pol")
    if pol is None:
        pol = ModelPolicy(model, name="probe", seed=0, randomize_order=False)
        _POLICY_CACHE["pol"] = pol
    obs = env.observe()
    cands = obs["candidates"]
    action = pol(obs)  # retourne {"action","arg"} ou tuple
    if isinstance(action, dict):
        return (action["action"], action["arg"])
    return action


_POLICY_CACHE = {}


def beam_search_action(model, env, layout, width: int = BEAM_WIDTH) -> tuple:
    """Beam search w≤8 sur transitions DSL: simule les w meilleurs candidats,
    évalue les états résultants, choisit l'action du meilleur beam."""
    obs = env.observe()
    cands = obs["candidates"]
    state = env.state
    try:
        scores = model(tensorize_obs(obs))
        top_k = mx.argsort(scores)[-width:].tolist() if len(scores) > width else list(range(len(scores)))
    except Exception:
        top_k = list(range(min(width, len(cands))))

    best_action, best_score = None, float("-inf")
    from ucm.env.oracle import _successor
    for idx in top_k:
        c = cands[idx]
        action = {"action": c["action"], "arg": c["arg"]}
        if action["action"] == "STOP":
            # STOP: score = succès si but satisfait
            if goal_satisfied(state, env.goal):
                score = 100.0
            else:
                score = -100.0
        else:
            nxt = _successor(layout, state, action)
            if nxt is None:
                continue
            # évalue l'état suivant avec le modèle
            env2 = TinyGraphKey(layout)
            env2.goal = env.goal
            env2.state = nxt
            try:
                obs2 = env2.observe()
                s2 = model(tensorize_obs(obs2))
                score = float(mx.max(s2).item())
            except Exception:
                score = 0.0
        if score > best_score:
            best_score = score
            best_action = (action["action"], action["arg"])

    return best_action or ("STOP", None)


def generate_deep_tasks(rng: random.Random, n_tasks: int = 60):
    """Génère des tâches TGK d*13-24 par layouts DSL 9-12 pièces."""
    tasks = []
    while len(tasks) < n_tasks:
        n_rooms = rng.randint(9, 12)
        rooms = [f"r{i}" for i in range(n_rooms)]
        rng.shuffle(rooms)
        edges = [(rooms[i], rooms[rng.randrange(i)]) for i in range(1, n_rooms)]
        # + quelques extra edges
        for _ in range(rng.randint(0, 2)):
            a, b = rng.sample(rooms, 2)
            if (a, b) not in edges and (b, a) not in edges:
                edges.append((a, b))
        door_edge = rng.randrange(len(edges))
        layout = Layout(rooms, edges, door_edge)
        # init
        agent = rng.choice(rooms)
        key_room = rng.choice(rooms)
        parcel_room = rng.choice(rooms)
        goal_pred = rng.choice(["REACH", "HAVE", "AT"])
        if goal_pred == "REACH":
            goal = task_goal("REACH", room=rng.choice(rooms))
        elif goal_pred == "HAVE":
            goal = task_goal("HAVE", object=rng.choice(["key", "parcel"]))
        else:
            goal = task_goal("AT", object=rng.choice(["key", "parcel"]), room=rng.choice(rooms))
        state = PhysicalState(agent, None, key_room, parcel_room, True)
        sol = tgk_solve(layout, state, goal)
        if sol["reachable"] and 13 <= sol["d_star"] <= 24:
            tasks.append((layout, state, goal, sol["d_star"]))
    return tasks


def run_episode(model, layout, state, goal, d_star, use_beam, beam_layout=None):
    env = TinyGraphKey(layout, horizon=HORIZON)
    env.goal = goal
    env.state = state.copy()
    env.step_count = 0
    env.terminal = False
    t0 = time.perf_counter()
    steps = 0
    while not env.terminal and steps < HORIZON:
        if use_beam:
            a = beam_search_action(model, env, beam_layout or layout)
            action = {"action": a[0], "arg": a[1]}
        else:
            a = policy_action(model, env)
            action = {"action": a[0], "arg": a[1]}
        if action["action"] == "STOP":
            success = goal_satisfied(env.state, env.goal)
            env.terminal = True
            latency = (time.perf_counter() - t0) * 1000 / max(1, steps + 1)
            return {"success": success, "steps": steps, "latency_ms": latency}
        res = env.execute(action)
        steps += 1
        if env.terminal:
            latency = (time.perf_counter() - t0) * 1000 / max(1, steps)
            return {"success": goal_satisfied(env.state, env.goal), "steps": steps,
                    "latency_ms": latency, "timeout": True}
    return {"success": False, "steps": steps, "latency_ms": (time.perf_counter()-t0)*1000/max(1,steps), "timeout": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/dev-research-a-beam-v01.json")
    ap.add_argument("--producer-commit", default=None)
    ap.add_argument("--n-tasks", type=int, default=60)
    args = ap.parse_args()
    producer = args.producer_commit or subprocess.run(["git","rev-parse","HEAD"],
        capture_output=True,text=True,check=True).stdout.strip()

    rng = random.Random(20261001)
    tasks = generate_deep_tasks(rng, args.n_tasks)
    print(f"tâches d*13-24 générées: {len(tasks)}")

    results = {"greedy": {"per_seed": [], "latencies": []},
               "beam_w8": {"per_seed": [], "latencies": []}}

    for seed_idx in range(N_SEEDS):
        model = load_b144(seed_idx)
        # Greedy
        successes = []
        latencies = []
        for layout, state, goal, d_star in tasks:
            r = run_episode(model, layout, state, goal, d_star, use_beam=False)
            successes.append(r["success"])
            latencies.append(r["latency_ms"])
        rate = sum(successes) / len(successes)
        results["greedy"]["per_seed"].append(rate)
        results["greedy"]["latencies"].extend(latencies)
        print(f"  s10{seed_idx} greedy: {rate:.3f}")

        # Beam w=8 (échantillon réduit pour le temps)
        beam_tasks = tasks[:20]  # 20 tâches pour le beam (coûteux)
        beam_succ = []
        beam_lat = []
        for layout, state, goal, d_star in beam_tasks:
            r = run_episode(model, layout, state, goal, d_star, use_beam=True)
            beam_succ.append(r["success"])
            beam_lat.append(r["latency_ms"])
        beam_rate = sum(beam_succ) / len(beam_succ)
        results["beam_w8"]["per_seed"].append(beam_rate)
        results["beam_w8"]["latencies"].extend(beam_lat)
        print(f"  s10{seed_idx} beam_w8: {beam_rate:.3f} (sur {len(beam_tasks)} épisodes)")

    # Agrégats
    greedy_mean = sum(results["greedy"]["per_seed"]) / N_SEEDS
    beam_mean = sum(results["beam_w8"]["per_seed"]) / N_SEEDS
    greedy_lat = sorted(results["greedy"]["latencies"])
    beam_lat = sorted(results["beam_w8"]["latencies"])
    p95_greedy = greedy_lat[int(0.95 * len(greedy_lat))] if greedy_lat else 0
    p95_beam = beam_lat[int(0.95 * len(beam_lat))] if beam_lat else 0

    canon_zs = 0.252  # annexe G4 13-24 canon zéro-shot
    gain_greedy = (greedy_mean - canon_zs) * 100
    gain_beam = (beam_mean - canon_zs) * 100

    print(f"\ngreedy: mean={greedy_mean:.3f} gain={gain_greedy:.1f}pp p95={p95_greedy:.1f}ms")
    print(f"beam_w8: mean={beam_mean:.3f} gain={gain_beam:.1f}pp p95={p95_beam:.1f}ms")

    # Verdict
    if gain_beam >= 15 and p95_beam <= 20:
        verdict = "GO"
    elif greedy_mean >= 0.80:
        verdict = "§13.4 INUTILE (données seules ≥80%)"
    elif gain_beam < 5:
        verdict = "KILL"
    else:
        verdict = "NI GO NI KILL"

    art = seal_manifest({
        "schema": "ucm-research-a-beam/0.1",
        "what": "Recherche A: profondeur 2×2 DEV — beam search w≤8 sur d*13-24 frais",
        "producer": {"code_commit": producer, "script": "scripts/dev_research_a_beam.py"},
        "tasks": {"n": len(tasks), "d_star_range": "13-24", "generator": "DSL 9-12 rooms, seed 20261001"},
        "greedy": {"per_seed": [round(r,3) for r in results["greedy"]["per_seed"]],
                    "mean": round(greedy_mean, 3), "p95_ms": round(p95_greedy, 1)},
        "beam_w8": {"per_seed": [round(r,3) for r in results["beam_w8"]["per_seed"]],
                     "mean": round(beam_mean, 3), "p95_ms": round(p95_beam, 1),
                     "n_eval_per_seed": 20},
        "canon_zero_shot": {"g4_13_24": canon_zs, "source": "annexe s2b"},
        "gain": {"greedy_pp": round(gain_greedy, 1), "beam_pp": round(gain_beam, 1)},
        "verdict": verdict,
        "criteria": {"GO": "≥+15pp à p95≤20ms", "KILL": "<+5pp", "inutile": "données seules ≥80%"},
        "hold": "DEV eval seule, zéro entraînement, zéro scellé",
    })
    Path(args.out).write_text(json.dumps(art, sort_keys=True, indent=2))
    print(f"→ {args.out} (scellé {art['manifest_sha256'][:12]})")
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
