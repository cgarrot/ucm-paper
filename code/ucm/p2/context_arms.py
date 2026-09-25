"""P2-1 (B) — Bras de contexte: historique structuré court « effet observé ».

Encodage §annexe (arbitrage fc026a3): arêtes « observed_effect » attachées
aux nœuds de GENRE (1-2 observations par genre) — PAS un historique brut.
Une observation = (genre K, effet TYPÉ, cible):
  MOVE  → agent_at room  (le nœud genre pointe la pièce d'arrivée)
  PICK  → carried obj    (le nœud genre pointe l'objet ramassé)
  DROP  → obj_at obj     (le nœud genre pointe l'objet posé)
  UNLOCK→ door           (le nœud genre pointe la porte)
Nouveaux types d'entités ctx_<kind> + relation observed_effect — vocabs P2
(tensorizer P2 dédié, D_IN étendu; le modèle P2 est B144-class FRAIS).
D1-mixed: floor(0.30×n) épisodes portent l'historique (RNG seed gelé).
D1-shuffled: même volume, cibles PERMUTÉES entre genres (nocivité).
"""
from __future__ import annotations

import random

CTX_KINDS = ("MOVE", "PICK", "DROP", "UNLOCK")
CTX_ENTITY_TYPES = [f"ctx_{k.lower()}" for k in CTX_KINDS]
CTX_REL = "observed_effect"

CTX_SEED = 20260924  # gelé (déclaré)
D1_FRACTION = 0.30


def extract_effects(interp, start_state, task, max_steps=14):
    """Effets typés observés le long du plan optimal DSL (reset-only)."""
    from ucm.dsl.bfs import DSLBFS
    interp.reset(task)
    st = interp.state
    effects = []
    for _ in range(max_steps):
        interp.state = st
        if interp.goal_satisfied():
            break
        sol = DSLBFS(interp).solve(st, max_states=20_000)
        if not sol["reachable"] or not sol["optimal_actions"]:
            break
        cand = interp.candidates()[sorted(sol["optimal_actions"])[0]]
        if cand["action"] == "STOP":
            break
        valid, nxt = interp.successor(cand)
        if not valid or nxt is None:
            break
        k = cand["action"]
        if k == "MOVE":
            effects.append((k, "agent_at", cand["arg"]))
        elif k == "PICK":
            effects.append((k, "carried", cand["arg"]))
        elif k == "DROP":
            effects.append((k, "obj_at", cand["arg"]))
        elif k == "UNLOCK":
            effects.append((k, "door_unlocked", cand["arg"]))
        st = nxt
    return effects


def _canonical_target(kind, arg):
    """Cible CANONIQUE (présente dans TOUTE observation TGK): l'observation
    informe le GENRE→TYPE d'effet, pas une pièce d'un layout étranger.
    MOVE → agent (l'agent se déplace), PICK/DROP → l'objet, UNLOCK → door0."""
    if kind == "MOVE":
        return "agent"
    if kind in ("PICK", "DROP"):
        return arg            # key | parcel
    if kind == "UNLOCK":
        return "door0"
    raise ValueError(kind)


def build_context_entities(effects, per_kind=(1, 2), rng=None):
    """1-2 observations par genre → entités ctx + arêtes observed_effect.
    Cibles canoniques (indépendantes du layout source)."""
    rng = rng or random.Random(0)
    by_kind = {}
    for eff in effects:
        by_kind.setdefault(eff[0], []).append(eff)
    ents, rels = [], []
    for kind in CTX_KINDS:
        obs_list = by_kind.get(kind, [])
        if not obs_list:
            continue
        k = rng.randint(per_kind[0], min(per_kind[1], len(obs_list)))
        chosen = rng.sample(obs_list, k)
        for j, (_, typ, target) in enumerate(chosen):
            eid = f"ctx-{kind.lower()}-{j}"
            ents.append({"id": eid, "type": f"ctx_{kind.lower()}", "attrs": {}})
            rels.append({"subj": eid, "pred": CTX_REL,
                         "obj": _canonical_target(kind, target)})
    return ents, rels


def build_context_entities_shuffled(effects, per_kind=(1, 2), rng=None):
    """D1-shuffled: MÊME volume, cibles PERMUTÉES entre genres."""
    ents, rels = build_context_entities(effects, per_kind, rng)
    if len(rels) >= 2:
        targets = [r["obj"] for r in rels]
        rng.shuffle(targets)
        rels = [dict(r, obj=targets[i]) for i, r in enumerate(rels)]
    return ents, rels


def augment_obs(obs: dict, ctx_ents: list, ctx_rels: list) -> dict:
    """Concatène le contexte à une observation (copie, jamais mutation).
    SELF-LOOP INIT (arbitrage lead 20:36): chaque nœud ctx reçoit une
    self-arête — sans elle, il est CIBLE du message-passing sans message
    entrant (agrégat vide) et le tronc P2GNNB diverge (NaN, diagnostic
    20:36: stable sans ctx, divergent avec). La self-loop lui donne une
    représentation propre; l'arête sémantique ctx→cible est conservée."""
    self_loops = [{"subj": e["id"], "pred": CTX_REL, "obj": e["id"]}
                  for e in ctx_ents]
    return {"schema_version": "p2/0.1",
            "entities": list(obs["entities"]) + list(ctx_ents),
            "relations": list(obs["relations"]) + list(ctx_rels) + self_loops,
            "goal": dict(obs["goal"]),
            "candidates": list(obs["candidates"])}


def d1_episode_ids(episode_ids, fraction=D1_FRACTION, seed=CTX_SEED):
    """floor(fraction×n) épisodes porteurs d'historique (déterministe)."""
    import math
    n = math.floor(fraction * len(episode_ids))
    rng = random.Random(seed)
    return set(rng.sample(sorted(episode_ids), n)) if n else set()
