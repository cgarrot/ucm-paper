"""Tests generate.py — déterminisme, conformité record, rejets comptés."""

import json
import subprocess
import sys

from ucm.data.generate import generate, generate_layout
from ucm.env.tinygraph import Layout

import random

FORBIDDEN = {"reward", "progress", "d_star", "L_star", "success", "terminal",
             "optimal_actions", "plan", "next_state", "timestep", "step",
             "episode", "split", "source"}


def test_generate_deterministic_same_seed():
    l1, s1 = generate(episodes=40, n_layouts=8, seed=7)
    l2, s2 = generate(episodes=40, n_layouts=8, seed=7)
    assert l1 == l2, "même seed doit produire des lignes identiques"
    assert s1["episodes"] == s2["episodes"]


def test_records_match_contract():
    lines, stats = generate(episodes=60, n_layouts=10, seed=11)
    assert stats["episodes"] > 0 and stats["transitions"] == len(lines)
    for ln in lines:
        r = json.loads(ln)
        assert set(r) == {"schema_version", "policy_input", "execution",
                          "supervision", "provenance"}
        assert r["schema_version"] == "0.2"
        pi = r["policy_input"]
        assert set(pi) == {"goal", "entities", "relations", "candidates"}
        # aucune clé interdite dans policy_input (§5.1)
        def walk(n):
            if isinstance(n, dict):
                for k, v in n.items():
                    assert k not in FORBIDDEN, f"forbidden {k}"
                    walk(v)
            elif isinstance(n, list):
                for v in n:
                    walk(v)
        walk(pi)
        cands = pi["candidates"]
        K = len(cands)
        n_rooms = len([e for e in pi["entities"] if e["type"] == "room"])
        assert K == n_rooms + 6
        assert cands[-1] == {"action": "STOP", "arg": None}
        ai = r["execution"]["action_ref"]
        assert isinstance(ai, int) and 0 <= ai < K
        assert ai in r["supervision"]["optimal_actions"]
        prov = r["provenance"]
        assert prov["split"] == "unassigned" and prov["source"] == "oracle"
        assert len(prov["layout_hash"]) == 16
        assert len(prov["state_goal_hash"]) == 16


def test_band_and_zero_fraction():
    lines, stats = generate(episodes=200, n_layouts=20, seed=3)
    hist = stats["by_d_star_hist"]
    assert sum(v for k, v in hist.items() if k != "0") > 0
    zeros = hist.get("0", 0)
    assert zeros <= 200 // 10 + 2, "fraction d*=0 plafonnée ~10 %"
    in_band = [int(k) for k in hist if k != "0"]
    assert all(2 <= k <= 12 for k in in_band), "bande d* respectée"


def test_door_meta_truthful_and_fraction_influences():
    rng = random.Random(5)
    lays = [generate_layout(rng, rng.randint(4, 8), 1.0) for _ in range(40)]
    # meta toujours véridique (jamais de False silencieux, audit tagi-5 m3)
    for l in lays:
        assert l.meta["door_on_bridge"] == l.is_bridge(l.door_edge)
    with_bridges = [l for l in lays if any(l.is_bridge(i)
                                           for i in range(len(l.edges)))]
    assert with_bridges, "la plupart des layouts ont au moins un pont"
    assert all(l.meta["door_on_bridge"] for l in with_bridges)
    rng = random.Random(6)
    lays0 = [generate_layout(rng, rng.randint(4, 8), 0.0) for _ in range(40)]
    for l in lays0:
        assert l.meta["door_on_bridge"] == l.is_bridge(l.door_edge)
    # fraction 0 : parmi les layouts ayant une arête de cycle, porte hors pont
    with_cycles = [l for l in lays0
                   if any(not l.is_bridge(i) for i in range(len(l.edges)))]
    assert with_cycles
    assert not any(l.meta["door_on_bridge"] for l in with_cycles)


def test_rejections_counted():
    _, stats = generate(episodes=50, n_layouts=10, seed=13)
    rj = stats["rejections"]
    assert set(rj) == {"unreachable", "band", "zero_fraction_cap",
                       "g2_reservation", "non_g2"}


def test_g2_reservation_modes():
    # exclude : aucune tâche réservée dans le train ; require : que ça
    _, ex = generate(episodes=300, n_layouts=40, seed=21, g2_mode="exclude")
    assert ex["rejections"]["g2_reservation"] > 0
    _, rq = generate(episodes=60, n_layouts=40, seed=21, g2_mode="require")
    assert rq["rejections"]["non_g2"] > 0
    assert rq["episodes"] > 0, "le mode require doit trouver des tâches G2"


def test_provenance_episode_and_step():
    lines, _ = generate(episodes=20, n_layouts=6, seed=9,
                        episode_prefix="pilot-")
    import json as _json
    recs = [_json.loads(l) for l in lines]
    refs = {r["provenance"]["episode_ref"] for r in recs}
    assert all(r.startswith("pilot-") for r in refs)
    # au moins un épisode multi-étapes : steps consécutifs 0,1,2...
    per_ep = {}
    for r in recs:
        per_ep.setdefault(r["provenance"]["episode_ref"], []).append(
            r["provenance"]["step"])
    assert any(st == list(range(len(st))) for st in per_ep.values())


def test_cli_smoke(tmp_path):
    out = tmp_path / "ep.jsonl"
    res = subprocess.run(
        [sys.executable, "-m", "ucm.data.generate", "--episodes", "30",
         "--layouts", "6", "--seed", "1", "--out", str(out)],
        capture_output=True, text=True, cwd=".", check=True)
    assert out.exists()
    lines = out.read_text().strip().splitlines()
    assert len(lines) >= 30
    summary = json.loads(res.stdout)
    assert summary["lines"] == len(lines)


def test_predicates_balanced_and_zero_floor():
    """Audit tagi-5 M2/M3 : équilibre §4.2 (~1/3 par prédicat) et fraction
    d*=0 §4.5 (~10 %) — non-régression distributionnelle."""
    for seed in (8, 21):
        _, st = generate(episodes=600, n_layouts=200, seed=seed)
        n = st["episodes"]
        for pred in ("REACH", "HAVE", "AT"):
            frac = st["by_goal"][pred] / n
            assert 0.28 <= frac <= 0.39, f"{pred}={frac:.2f}"
        zeros = st["by_d_star_hist"].get("0", 0) / n
        assert 0.07 <= zeros <= 0.13, f"zeros={zeros:.2f}"
        # bande 2-12 uniquement hors zéros
        assert all(2 <= int(k) <= 12 for k in st["by_d_star_hist"] if k != "0")


def test_require_mode_no_zeros_no_fallbacks():
    """Re-vér tagi-5 §2: cellule test G2 = bande stricte, zéros interdits ;
    relax désactivé (fallbacks=0, tirages non morts)."""
    _, st = generate(episodes=200, n_layouts=60, seed=11, g2_mode="require")
    assert st["episodes"] > 0
    assert "0" not in st["by_d_star_hist"], "zéro initial interdit en require"
    assert st.get("predicate_fallbacks", 0) == 0
    assert set(st["by_goal"]) == {"AT"}
    assert all(2 <= int(k) <= 12 for k in st["by_d_star_hist"])
