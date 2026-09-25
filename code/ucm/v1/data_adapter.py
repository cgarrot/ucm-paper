"""V1b data adapter — sealed SIW files → runner-consumable records/episodes.

    couples   {goal, layout_hash, state_key} → {policy_input, supervision,
              provenance}: layout resolved from the sealed layout store
              (inventory for adaptation pools), env/obs reconstructed from
              state_key (SIWState fields), supervision via SIWOracle,
              provenance (layout_hash, state_goal_hash).
    episodes  {episode_ref, layout_hash, init, goal, d_star, reachable} →
              {episode_id, layout_spec, task, d_star, layout_hash}.

Also publishes the CONTENT hash (sha256 of canonical line list) of each
consumed sealed file into the run artifacts (audit 19:39 item 3).
"""

from __future__ import annotations

import hashlib
import json

from ucm.env.siw import SIW, SIWLayout, SIWState, goal_satisfied
from ucm.env.siw_oracle import SIWOracle


def _load_layout_store(path: str) -> dict[str, dict]:
    store = json.load(open(path))
    if isinstance(store, list):  # tolerate list form
        return {l["layout_hash"] if "layout_hash" in l else h: l
                for h, l in ((None, x) for x in store)} if False else \
               {json.dumps(l, sort_keys=True, default=str)[:16]: l for l in store}
    return store


def _spec_of(entry: dict) -> dict:
    spec = dict(entry)
    if isinstance(spec.get("widgets"), dict):
        spec["widgets"] = list(spec["widgets"].values())
    return spec


def _state_of(state_key) -> SIWState:
    view, filled, chosen, dialog, submitted = state_key
    return SIWState(view=view, filled=frozenset(filled),
                    chosen=dict(chosen), dialog_open=bool(dialog),
                    submitted=frozenset(submitted))


def _state_goal_hash(layout_hash: str, state: SIWState, goal: dict) -> str:
    payload = json.dumps([layout_hash, list(state.key().__reduce__()[1][0]) if False else
                          [state.view, sorted(state.filled), sorted(state.chosen.items()),
                           state.dialog_open, sorted(state.submitted)], goal],
                         sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def content_hash(path: str) -> str:
    """sha256 over the canonical serialization of the line list (bytes-level
    content identity, audit item 3)."""
    lines = [l.rstrip("\n") for l in open(path) if l.strip()]
    canon = "\n".join(lines) + ("\n" if lines else "")
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


def _couples_core(store: dict, lines: list[str],
                  cache: dict | None = None) -> list[dict]:
    """SINGLE implementation: takes store as a dict (no file I/O)."""
    cache = cache if cache is not None else {}
    out = []
    for line in lines:
        if not line.strip():
            continue
        c = json.loads(line)
        lay_entry = store[c["layout_hash"]]
        spec = _spec_of(lay_entry)
        lay = SIWLayout(spec)
        state = _state_of(c["state_key"])
        goal = {"predicate": c["goal"]["predicate"], "args": c["goal"]["args"]}
        env = SIW(lay)
        obs = env.reset({"init": {"view": state.view,
                                  "filled": sorted(state.filled),
                                  "chosen": state.chosen,
                                  "dialog_open": state.dialog_open,
                                  "submitted": sorted(state.submitted)},
                          "goal": goal})
        oracle = cache.get((c["layout_hash"], goal["predicate"], json.dumps(goal["args"], sort_keys=True)))
        if oracle is None:
            oracle = SIWOracle(lay, goal)
            cache[(c["layout_hash"], goal["predicate"], json.dumps(goal["args"], sort_keys=True))] = oracle
        sol = oracle.solve(state)
        if not sol["reachable"]:
            raise ValueError(f"sealed couple unreachable: {c}")
        out.append({
            "policy_input": obs,
            "supervision": {"optimal_actions": sol["optimal_actions"],
                            "d_star": sol["d_star"], "reachable": True},
            "provenance": {"layout_hash": c["layout_hash"],
                            "state_goal_hash": _state_goal_hash(c["layout_hash"], state, goal),
                            "split": "siw-adaptation"},
        })
    return out


def episodes_lines_to_runner(lines: list[str], layout_store_path: str) -> list[dict]:
    """In-memory episodes adaptation (no reopen — single-FD discipline)."""
    store = json.load(open(layout_store_path))
    out = []
    for line in lines:
        if not line.strip():
            continue
        e = json.loads(line)
        lay_entry = store[e["layout_hash"]]
        out.append({
            "episode_id": e.get("episode_id", e.get("episode_ref", e.get("episode_ref"))),
            "layout_spec": lay_entry,
            "task": e.get("task", {"init": e.get("init"), "goal": e.get("goal")}),
            "d_star": e.get("d_star"),
            "layout_hash": e["layout_hash"],
        })
    return out


def episodes_to_runner(path: str, layout_store_path: str) -> list[dict]:
    store = json.load(open(layout_store_path))
    out = []
    for line in open(path):
        if not line.strip():
            continue
        e = json.loads(line)
        lay_entry = store[e["layout_hash"]]
        out.append({
            "episode_id": e["episode_ref"],
            "layout_spec": lay_entry,           # raw store entry (id→widget map OK)
            "task": {"init": e["init"], "goal": e["goal"]},
            "d_star": e["d_star"],
            "layout_hash": e["layout_hash"],
        })
    return out


def adapter_manifest(paths: list[str]) -> dict:
    return {"content_hashes": {p: content_hash(p) for p in paths},
            "note": "sha256 over canonical line list; publish+re-seal with the "
                    "run artifacts (audit 19:39 item 3)"}


# ---------------------------------------------------------------------------
# REGISTRE DES DÉCISIONS DE MAPPING (lead 19:39: aucune décision libre non
# enregistrée). Chaque entrée: décision, justification, alternative écartée.
# ---------------------------------------------------------------------------
MAPPING_REGISTRY = {
    "layout_store_couples": ("inventory-siw-dev-layouts.json (30/30 hashes couverts)",
                             "pointeur tagi-5 19:39; les pools d'adaptation vivent dans l'inventaire",
                             "layouts-SIW-small (0/30 couverts — écarté)"),
    "layout_store_episodes": ("layouts-SIW-small.json (167/167 hashes couverts)",
                              "pointeur tagi-5 19:39; pools test dans le manifeste scellé",
                              "inventory (0/167 couverts — écarté)"),
    "state_key_to_state": ("[view, filled, chosen, dialog, submitted] → SIWState direct "
                           "(listes→frozensets, dict(chosen), bool)",
                           "champs SIWState.key() dans l'ordre — aucune interprétation",
                           "aucune (mapping bijectif)"),
    "state_goal_hash": ("sha256([layout_hash, [view, sorted(filled), sorted(chosen.items), "
                        "dialog, sorted(submitted)], goal])",
                        "identité de contenu canonique, déterministe",
                        "hash fourni par le scellé si WS-B le publie (remplaçable par config)"),
    "candidate_binding": ("indices oracle == ordre env.candidates() — VÉRIFIÉ 50/50 "
                          "identiques sur les couples scellés (test de non-régression)",
                          "SIWOracle reconstruit le layout en ordre trié stable, "
                          "identique à l'ordre d'énumération de l'env",
                          "aucune (si divergence: lever, jamais remapper)"),
    "supervision": ("SIWOracle.solve(state) embarqué tel quel (optimal_actions, d_star)",
                    "oracle exact §12.6; unreachable scellé → ValueError visible",
                    "aucune"),
}


def couples_from_store_dict(lines: list[str], store: dict,
                             oracle_cache: dict | None = None) -> list[dict]:
    """Dict-based wrapper: delegates to _couples_core (no file open)."""
    return _couples_core(store, lines, oracle_cache)


def _couples_impl(lines: list[str], layout_store_path: str,
                  oracle_cache: dict | None = None) -> list[dict]:
    """Path-based wrapper: loads store from file, delegates to _couples_core."""
    store = json.load(open(layout_store_path))
    return _couples_core(store, lines, oracle_cache)


def couples_lines_to_records(lines: list[str], layout_store_path: str,
                              oracle_cache: dict | None = None) -> list[dict]:
    """In-memory variant (no temp file on disk — lead 21:44 (4))."""
    return _couples_impl(lines, layout_store_path, oracle_cache)


def couples_to_records(path: str, layout_store_path: str,
                       oracle_cache: dict | None = None) -> list[dict]:
    lines = [l.rstrip("\n") for l in open(path)]
    return _couples_impl(lines, layout_store_path, oracle_cache)


