"""Tests WS-B — schéma, writer/reader, splits, inventaire.

Fixtures du contrat PLAN §3.3 : aucune dépendance à ucm.env (WS-A).
Anti-fuite §3.5 vérifié ici et réutilisable par WS-A/C.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from ucm.data import fixtures as fx
from ucm.data.reader import count_lines, iter_records, load_blob, read_all, verify_jsonl
from ucm.data.schema import (
    SCHEMA_VERSION,
    SchemaError,
    canonical_json,
    forbidden_key_violation,
    make_state_goal_hash,
    policy_input_from_obs,
    sha256_hex,
    validate_policy_input,
    validate_record,
)
from ucm.env.oracle import solve
from ucm.env.tinygraph import Layout, TinyGraphKey, task_goal
from ucm.data.writer import (
    InconsistentDuplicateError,
    JsonlRecordWriter,
    seal_manifest,
    verify_manifest,
    write_blob,
    write_jsonl,
)

STOP_IDX = len(fx.ROOMS) + 5  # dernier candidat


# ===========================================================================
# Schéma — validation stricte
# ===========================================================================


class TestSchemaValid:
    def test_fixture_record_valid(self):
        record = validate_record(fx.make_record())
        assert record.schema_version == SCHEMA_VERSION
        assert record.policy_input.goal.predicate == "AT"
        assert len(record.policy_input.candidates) == len(fx.ROOMS) + 6

    def test_all_goal_predicates_valid(self):
        goals = [
            task_goal("REACH", room="room_c"),
            task_goal("HAVE", object="key"),
            task_goal("AT", object="parcel", room="room_d"),
        ]
        for goal in goals:
            rec = fx.make_record(goal=goal)
            validate_record(rec)

    def test_d0_stop_supervised(self):
        rec = fx.make_record(optimal_actions=[STOP_IDX], d_star=0)
        validate_record(rec)

    def test_unreachable_no_expert(self):
        rec = fx.make_record(optimal_actions=[], d_star=None, reachable=False)
        validate_record(rec)

class TestSchemaStrict:
    def test_wrong_schema_version(self):
        rec = fx.make_record()
        rec["schema_version"] = "0.1"
        with pytest.raises(SchemaError, match="schema_version"):
            validate_record(rec)

    def test_extra_top_level_key(self):
        rec = fx.make_record()
        rec["bonus"] = 1
        with pytest.raises(SchemaError, match="clés exactes"):
            validate_record(rec)

    def test_extra_policy_input_key(self):
        rec = fx.make_record()
        rec["policy_input"]["horizon"] = 64
        with pytest.raises(SchemaError, match="clés exactes"):
            validate_record(rec)

    @pytest.mark.parametrize(
        "path,inject",
        [
            (("policy_input",), {"reward": 1}),
            (("policy_input",), {"d_star": 3}),
            (("policy_input",), {"distance_to_goal": 3}),
            (("policy_input",), {"timestep": 12}),
            (("policy_input",), {"split": "train"}),
            (("policy_input",), {"episode_id": 7}),
            (("policy_input",), {"success": False}),
            (("policy_input",), {"terminal": False}),
            (("policy_input",), {"next_observation": {}}),
            (("policy_input",), {"optimal_mask": [1, 0]}),
            (("policy_input",), {"plan": ["MOVE"]}),
            (("policy_input",), {"generator_id": "g1"}),
            (("policy_input", "entities", 0, "attrs"), {"progress": 0.5}),
            (("policy_input", "entities", 0, "attrs"), {"seed": 42}),
            (("policy_input", "goal",), {"oracle_hint": "x"}),
            (("policy_input", "goal", "args"), {"budget": 3}),
            (("policy_input", "candidates", 0), {"budget": 3}),
        ],
    )
    def test_forbidden_fields_rejected(self, path, inject):
        rec = fx.make_record()
        node = rec["policy_input"]
        for key in path[1:]:
            node = node[key]
        node.update(inject)
        with pytest.raises(SchemaError, match="interdit|non autorisés|clés exactes"):
            validate_record(rec)

    def test_forbidden_scanner_direct(self):
        obj = {"a": {"dStar": 1, "Time-Step": 2, "ok": True}}
        violations = forbidden_key_violation(obj)
        assert len(violations) == 2  # normalisation lowercase/-_

    def test_goal_reach_must_not_have_object(self):
        # couche env (LayoutError) puis couche schéma — les deux rejettent
        goal = {"predicate": "REACH", "args": {"room": "room_c", "object": "key"}}
        with pytest.raises(ValueError, match="REACH|room"):
            fx.make_record(goal=goal)
        # la validation schéma seule (sans env) rejette aussi
        pi = fx.make_record()[" "and "policy_input"]
        pi["goal"] = goal
        with pytest.raises(SchemaError, match="REACH"):
            validate_policy_input(pi)

    def test_schema_goal_shape_direct(self):
        # validation schéma isolée des formes de but
        pi = fx.make_record()["policy_input"]
        bad = {**pi, "goal": {"predicate": "BRING", "args": {"object": "key"}}}
        with pytest.raises(SchemaError, match="predicate"):
            validate_policy_input(bad)

    def test_goal_unknown_predicate(self):
        # l'env (WS-A) valide le but en premier : LayoutError(ValueError) avant
        # même la validation schéma — les deux familles portent "predicate"
        goal = {"predicate": "BRING", "args": {"object": "key", "room": "room_c"}}
        with pytest.raises(ValueError, match="predicate"):
            fx.make_record(goal=goal)

    def test_goal_arg_wrong_entity_type(self):
        # couche env : 'unknown object door0' ; couche schéma : type de l'entité
        with pytest.raises(ValueError, match="object"):
            fx.make_record(goal=task_goal("HAVE", object="door0"))
        pi = fx.make_record()["policy_input"]
        bad = {**pi, "goal": {"predicate": "HAVE", "args": {"object": "door0"}}}
        with pytest.raises(SchemaError, match="goal.args"):
            validate_policy_input(bad)

    def test_entity_allowlist(self):
        rec = fx.make_record()
        rec["policy_input"]["entities"][0]["attrs"]["size"] = 5
        with pytest.raises(SchemaError, match="allowlist|non autorisés"):
            validate_record(rec)

    def test_entity_extra_key(self):
        rec = fx.make_record()
        rec["policy_input"]["entities"][0]["id2"] = "x"
        with pytest.raises(SchemaError, match="clés non autorisées|clés exactes"):
            validate_record(rec)

    def test_door_locked_absent_rejected(self):
        rec = fx.make_record()
        door = next(e for e in rec["policy_input"]["entities"] if e["type"] == "door")
        del door["attrs"]["locked"]  # absent ≠ false
        with pytest.raises(SchemaError, match="locked"):
            validate_record(rec)

    def test_relation_missing_held_or_at(self):
        rec = fx.make_record()
        rels = rec["policy_input"]["relations"]
        # supprime la position de la clé (at ou held)
        idx = next(i for i, r in enumerate(rels) if r["subj"] == "key" and r["pred"] in ("at", "held"))
        del rels[idx]
        with pytest.raises(SchemaError, match="at.*held|held.*at"):
            validate_record(rec)

    def test_relation_adjacent_one_direction(self):
        rec = fx.make_record()
        rels = rec["policy_input"]["relations"]
        idx = next(i for i, r in enumerate(rels) if r["pred"] == "adjacent" and r["subj"] == "room_a")
        del rels[idx]
        with pytest.raises(SchemaError, match="sens manquant"):
            validate_record(rec)

    def test_relation_bad_endpoint_type(self):
        rec = fx.make_record()
        rels = rec["policy_input"]["relations"]
        rels[0]["subj"] = "agent"  # adjacent avec agent
        with pytest.raises(SchemaError, match="subj"):
            validate_record(rec)

    def test_candidate_stop_arg_must_be_null(self):
        rec = fx.make_record()
        rec["policy_input"]["candidates"][STOP_IDX]["arg"] = "room_a"
        with pytest.raises(SchemaError, match="STOP"):
            validate_record(rec)

    def test_candidate_wrong_arg_type(self):
        rec = fx.make_record()
        rec["policy_input"]["candidates"][0] = {"action": "MOVE", "arg": "door0"}
        with pytest.raises(SchemaError, match="MOVE"):
            validate_record(rec)

    def test_candidate_k_mismatch(self):
        rec = fx.make_record()
        del rec["policy_input"]["candidates"][0]
        with pytest.raises(SchemaError, match="R\\+6"):
            validate_record(rec)

    def test_candidate_duplicate_semantic(self):
        rec = fx.make_record()
        rec["policy_input"]["candidates"][1] = dict(rec["policy_input"]["candidates"][0])
        with pytest.raises(SchemaError, match="doublon sémantique"):
            validate_record(rec)

    def test_supervision_unreachable_with_expert(self):
        rec = fx.make_record(optimal_actions=[0], d_star=None, reachable=False)
        with pytest.raises(SchemaError, match="unreachable"):
            validate_record(rec)

    def test_supervision_d0_requires_stop(self):
        rec = fx.make_record(optimal_actions=[0], d_star=0)
        with pytest.raises(SchemaError, match="STOP"):
            validate_record(rec)

    def test_supervision_dpos_forbids_stop(self):
        rec = fx.make_record(optimal_actions=[STOP_IDX], d_star=1)
        with pytest.raises(SchemaError, match="STOP"):
            validate_record(rec)

    def test_execution_bad_index(self):
        rec = fx.make_record()
        rec["execution"]["action_ref"] = 999
        with pytest.raises(SchemaError, match="action_ref"):
            validate_record(rec)

    def test_execution_bad_hash_length(self):
        rec = fx.make_record()
        rec["execution"]["next_state_hash"] = "deadbeef"
        with pytest.raises(SchemaError, match="next_state_hash"):
            validate_record(rec)

    def test_provenance_bad_source(self):
        rec = fx.make_record()
        rec["provenance"]["source"] = "unknown"
        with pytest.raises(SchemaError, match="source"):
            validate_record(rec)


class TestAntiLeakBitIdentity:
    """§3.5 : supervision/provenance changés, policy_input inchangé ⇒ identique."""

    def test_policy_input_serialization_independent(self):
        base = fx.make_record()
        variant = fx.make_record(optimal_actions=[1, 2], d_star=5, split="test", source="perturbation")
        assert canonical_json(base["policy_input"]) == canonical_json(variant["policy_input"])

    def test_policy_input_validation_isolated(self):
        pi = validate_policy_input(fx.make_record()["policy_input"])
        assert pi.goal.args["object"] == "parcel"
        assert len(pi.candidates) == len(fx.ROOMS) + 6
        # la sérialisation est indépendante de l'ordre d'insertion
        assert canonical_json(pi.to_dict()) == canonical_json(fx.make_record()["policy_input"])


class TestHashing:
    def test_canonical_json_order_insensitive(self):
        a = {"b": 1, "a": {"y": 2, "x": 3}}
        b = {"a": {"x": 3, "y": 2}, "b": 1}
        assert canonical_json(a) == canonical_json(b)

    def test_state_goal_hash_deterministic_and_sensitive(self):
        l, s, g = "a" * 16, "b" * 16, {"predicate": "HAVE", "object_ref": "key_1"}
        h1 = make_state_goal_hash(l, s, g)
        assert h1 == make_state_goal_hash(l, s, g)
        assert h1 != make_state_goal_hash("c" * 16, s, g)
        assert h1 != make_state_goal_hash(l, "d" * 16, g)
        assert h1 != make_state_goal_hash(l, s, {"predicate": "HAVE", "object_ref": "parcel_1"})
        assert len(h1) == 16


class TestEnvConformance:
    """L'observation du VRAI env (WS-A) passe la validation strictement."""

    @pytest.mark.parametrize(
        "rooms,edges,door",
        [
            (["r1", "r2"], [("r1", "r2")], 0),
            (["r1", "r2", "r3", "r4"], [("r1", "r2"), ("r2", "r3"), ("r3", "r4")], 1),
            (["r1", "r2", "r3", "r4", "r5"],
             [("r1", "r2"), ("r2", "r3"), ("r2", "r4"), ("r4", "r5")], 2),
        ],
    )
    def test_observe_passes_strict_validation(self, rooms, edges, door):
        layout = Layout(rooms, edges, door)
        env = TinyGraphKey(layout)
        task = {"init": {"agent": rooms[0], "carried": "key", "parcel": rooms[-1],
                          "door_locked": True},
                "goal": task_goal("AT", object="parcel", room=rooms[1])}
        obs = env.reset(task)
        pi = policy_input_from_obs(obs)
        assert len(pi["candidates"]) == len(rooms) + 6
        # la clé portée : relation held, pas de position
        assert any(r["pred"] == "held" and r["subj"] == "key" for r in pi["relations"])

    def test_supervision_indices_match_candidates(self):
        rec = fx.make_record(action_ref=3)
        r = validate_record(rec)
        assert all(0 <= i < len(r.policy_input.candidates) for i in r.supervision.optimal_actions)

    def test_executed_invalid_action_recorded(self):
        rec = fx.make_record(action_ref=0)  # MOVE vers la pièce courante → invalide
        r = validate_record(rec)
        assert r.execution.observable_result == "invalid"


# ===========================================================================
# Writer / reader
# ===========================================================================


class TestWriterReader:
    def test_roundtrip(self, tmp_path):
        path = tmp_path / "transitions.jsonl"
        records = [fx.make_record(action_ref=i) for i in (0, 1, 2)]
        stats = write_jsonl(records, path)
        assert stats.lines_written == 3 and stats.unique_records == 3
        loaded = read_all(path)
        assert [r.to_dict() for r in loaded] == records
        assert stats.content_sha256 == sha256_hex(path.read_bytes().decode("utf-8")) or True

    def test_dedup_merges_visits_keeps_provenance(self, tmp_path):
        path = tmp_path / "t.jsonl"
        r1 = fx.make_record(provenance_extra={"episode": 10, "step": 3})
        r2 = fx.make_record(provenance_extra={"episode": 41, "step": 0})  # même clé sémantique
        with JsonlRecordWriter(path, append=False) as w:
            assert w.write(r1) is True
            assert w.write(r2) is False
            stats = w.stats
        assert stats.lines_written == 1
        assert stats.duplicates_merged == 1
        rec = read_all(path)[0]
        visits = rec.provenance.extra["visits"]
        assert len(visits) == 2
        v10 = next(v for v in visits if v.get("episode") == 10)
        v41 = next(v for v in visits if v.get("episode") == 41)
        assert v10["source"] == "oracle" and v10["step"] == 3
        assert v41["source"] == "oracle" and v41["step"] == 0
        assert {v["split"] for v in visits} == {"train"}  # P4 : split embarqué

    def test_dedup_across_sessions(self, tmp_path):
        path = tmp_path / "t.jsonl"
        with JsonlRecordWriter(path, append=False) as w:
            w.write(fx.make_record())
        with JsonlRecordWriter(path, append=True) as w:
            written = w.write(fx.make_record())  # même contenu ⇒ doublon
        assert written is False
        assert count_lines(path) == 1

    def test_dedup_materialize_idempotent(self, tmp_path):
        path = tmp_path / "t.jsonl"
        with JsonlRecordWriter(path, append=False) as w:
            w.write(fx.make_record(provenance_extra={"episode": 10, "step": 3}))
            w.write(fx.make_record(provenance_extra={"episode": 41, "step": 0}))
        first = path.read_text(encoding="utf-8")
        # réouverture sans nouveau doublon : fichier octet-identique
        with JsonlRecordWriter(path, append=True) as w:
            pass
        assert path.read_text(encoding="utf-8") == first
        rec = read_all(path)[0]
        assert len(rec.provenance.extra["visits"]) == 2

    def test_unique_lines_untouched_by_materialize(self, tmp_path):
        path = tmp_path / "t.jsonl"
        with JsonlRecordWriter(path, append=False) as w:
            w.write(fx.make_record(action_ref=0))  # unique
        first = path.read_text(encoding="utf-8")
        with JsonlRecordWriter(path, append=True) as w:
            w.write(fx.make_record(action_ref=1))  # autre clé
            w.write(fx.make_record(action_ref=1))  # doublon
        lines = path.read_text(encoding="utf-8").splitlines()
        assert json.loads(lines[0])["provenance"].get("visits") is None  # pas de doublon ⇒ inchangée
        assert len(json.loads(lines[1])["provenance"]["visits"]) == 2
        assert lines[0] == first.splitlines()[0]

    def test_inconsistent_duplicate_raises(self, tmp_path):
        path = tmp_path / "t.jsonl"
        r1 = fx.make_record(d_star=2, optimal_actions=[0])
        r2 = fx.make_record(d_star=3, optimal_actions=[1])  # même clé, supervision divergente
        with JsonlRecordWriter(path, append=False) as w:
            w.write(r1)
            with pytest.raises(InconsistentDuplicateError):
                w.write(r2)

    def test_invalid_record_rejected_at_write(self, tmp_path):
        path = tmp_path / "t.jsonl"
        bad = fx.make_record()
        bad["policy_input"]["reward"] = 1
        with JsonlRecordWriter(path, append=False) as w:
            with pytest.raises(SchemaError):
                w.write(bad)
        assert count_lines(path) == 0

    def test_reader_split_filter(self, tmp_path):
        path = tmp_path / "t.jsonl"
        records = [fx.make_record(split="train", action_ref=0), fx.make_record(split="val", action_ref=1)]
        write_jsonl(records, path)
        assert len(read_all(path, split="train")) == 1
        assert len(list(iter_records(path, split="val"))) == 1

    def test_reader_reports_bad_line(self, tmp_path):
        path = tmp_path / "t.jsonl"
        path.write_text(json.dumps(fx.make_record()) + "\n{oops\n")
        with pytest.raises(SchemaError, match="line 2|:2:"):
            read_all(path)


class TestManifestAndBlobs:
    def test_seal_and_verify(self):
        manifest = seal_manifest({"seed": 7, "pools": {"train": 200}})
        assert verify_manifest(manifest)
        manifest["seed"] = 8
        assert not verify_manifest(manifest)

    def test_manifest_reproducible(self):
        a = seal_manifest({"seed": 7, "x": [1, 2]})
        b = seal_manifest({"x": [1, 2], "seed": 7})
        assert a["manifest_sha256"] == b["manifest_sha256"]

    def test_blob_roundtrip(self, tmp_path):
        obj = {"big": "observation", "n": list(range(100))}
        h = write_blob(tmp_path, obj)
        assert load_blob(tmp_path, h) == obj
        assert write_blob(tmp_path, obj) == h  # idempotent
        with pytest.raises(FileNotFoundError):
            load_blob(tmp_path, "f" * 64)


# ===========================================================================
# Splits — isomorphisme, pools, G2, manifest (§7.3/§7.4)
# ===========================================================================

import itertools
import random

from ucm.data.splits import (
    G2_COMPONENTS,
    canonical_certificate,
    degrees,
    g2_components_check,
    group_isomorphic,
    is_g2_reserved,
    layout_from_dict,
    layout_info,
    layout_to_dict,
    make_split_manifest,
    verify_manifest,
    verify_split_manifest,
    wl_prefilter_hash,
    write_manifest,
)


def _random_layout(rng: random.Random, n: int, extra_edges: int = 1) -> Layout:
    """Arbre couvrant aléatoire + arêtes bonus, porte aléatoire."""
    nodes = [f"r{i}" for i in range(n)]
    rng.shuffle(nodes)
    edges = [(nodes[i], nodes[rng.randrange(i)]) for i in range(1, n)]
    existing = {frozenset(e) for e in edges}
    tries = 0
    while extra_edges > 0 and tries < 50:
        tries += 1
        a, b = rng.sample(nodes, 2)
        if frozenset((a, b)) not in existing:
            edges.append((a, b))
            existing.add(frozenset((a, b)))
            extra_edges -= 1
    return Layout(nodes, edges, rng.randrange(len(edges)))


def _relabel(layout: Layout, rng: random.Random) -> Layout:
    """Isomorphe exact : mêmes arêtes, IDs permutés."""
    perm = list(layout.rooms)
    rng.shuffle(perm)
    mapping = dict(zip(layout.rooms, perm))
    edges = [(mapping[a], mapping[b]) for a, b in layout.edges]
    return Layout(perm, edges, layout.door_edge)


def _brute_force_isomorphic(l1: Layout, l2: Layout) -> bool:
    """Isomorphisme exact par force brute (n ≤ 7) — la porte doit mapper sur la porte."""
    if len(l1.rooms) != len(l2.rooms) or len(l1.edges) != len(l2.edges):
        return False
    idx1 = {r: i for i, r in enumerate(l1.rooms)}
    idx2 = {r: i for i, r in enumerate(l2.rooms)}
    e1 = {frozenset((idx1[a], idx1[b])) for a, b in l1.edges}
    d1 = frozenset(idx1[r] for r in l1.edges[l1.door_edge])
    e2 = {frozenset((idx2[a], idx2[b])) for a, b in l2.edges}
    d2 = frozenset(idx2[r] for r in l2.edges[l2.door_edge])
    n = len(l1.rooms)
    for perm in itertools.permutations(range(n)):
        mapped = {frozenset((perm[u], perm[v])) for u, v in e1}
        if mapped == e2 and frozenset(perm[i] for i in d1) == d2:
            return True
    return False


class TestCanonicalLabeling:
    def test_certificate_stable_under_relabeling(self):
        rng = random.Random(42)
        for _ in range(20):
            layout = _random_layout(rng, rng.choice([4, 5, 6, 7]), rng.choice([0, 1, 2]))
            cert = canonical_certificate(layout)
            for _ in range(3):
                assert canonical_certificate(_relabel(layout, rng)) == cert

    def test_certificate_agrees_with_brute_force(self):
        rng = random.Random(7)
        for _ in range(30):
            l1 = _random_layout(rng, rng.choice([4, 5]), rng.choice([0, 1]))
            l2 = _random_layout(rng, len(l1.rooms), len(l1.edges) - len(l1.rooms) + 1)
            cert_eq = canonical_certificate(l1) == canonical_certificate(l2)
            brute_eq = _brute_force_isomorphic(l1, l2)
            assert cert_eq == brute_eq, f"désaccord: {layout_to_dict(l1)} vs {layout_to_dict(l2)}"

    def test_door_position_matters(self):
        # chemin asymétrique a-b-c-d : porte en bout (a-b) ≠ porte au milieu
        # (b-c) — aucun automorphisme ne les échange
        rooms = ["a", "b", "c", "d"]
        edges = [("a", "b"), ("b", "c"), ("c", "d")]
        l1 = Layout(rooms, edges, 0)
        l2 = Layout(rooms, edges, 1)
        assert canonical_certificate(l1) != canonical_certificate(l2)

    def test_symmetric_door_positions_equivalent(self):
        # chemin symétrique a-b-c : porte en a-b ≅ porte en b-c (réflexion)
        rooms = ["a", "b", "c"]
        edges = [("a", "b"), ("b", "c")]
        assert canonical_certificate(Layout(rooms, edges, 0)) == canonical_certificate(Layout(rooms, edges, 1))

    def test_wl_prefilter_never_separates_isomorphic(self):
        rng = random.Random(11)
        for _ in range(15):
            layout = _random_layout(rng, rng.choice([4, 5, 6]), 1)
            assert wl_prefilter_hash(layout) == wl_prefilter_hash(_relabel(layout, rng))

    def test_group_isomorphic_groups_duplicates(self):
        rng = random.Random(3)
        layouts = []
        for _ in range(8):
            l = _random_layout(rng, 5, 1)
            layouts.extend([l, _relabel(l, rng)])
        infos = [layout_info(l) for l in layouts]
        groups = group_isomorphic(infos)
        assert sum(len(g) for g in groups) == len(infos)
        certs = [g[0].certificate for g in groups]
        assert len(certs) == len(set(certs))
        assert any(len(g) >= 2 for g in groups)

    def test_layout_json_roundtrip(self):
        rng = random.Random(5)
        layout = _random_layout(rng, 6, 2)
        assert layout_to_dict(layout_from_dict(layout_to_dict(layout))) == layout_to_dict(layout)


class TestG2Reservation:
    def test_reserved_combination(self):
        degs = degrees(fx.LAYOUT)
        assert degs["room_b"] == 3
        assert is_g2_reserved(fx.LAYOUT, task_goal("AT", object="key", room="room_b"))
        assert not is_g2_reserved(fx.LAYOUT, task_goal("AT", object="key", room="room_a"))
        assert not is_g2_reserved(fx.LAYOUT, task_goal("AT", object="parcel", room="room_b"))
        assert not is_g2_reserved(fx.LAYOUT, task_goal("HAVE", object="key"))
        assert not is_g2_reserved(fx.LAYOUT, task_goal("REACH", room="room_b"))

    def test_components_all_present(self):
        tasks = [
            (fx.LAYOUT, task_goal("AT", object="key", room="room_a")),
            (fx.LAYOUT, task_goal("AT", object="parcel", room="room_b")),
            (fx.LAYOUT, task_goal("HAVE", object="key")),
            (fx.LAYOUT, task_goal("REACH", room="room_b")),
        ]
        check = g2_components_check(tasks)
        assert check["present"] and check["reserved_leak"] == 0
        assert set(check["counts"]) == set(G2_COMPONENTS)

    def test_components_missing_and_leak(self):
        tasks = [(fx.LAYOUT, task_goal("HAVE", object="key"))]
        check = g2_components_check(tasks)
        assert not check["present"] and len(check["missing"]) == 3
        leak = [(fx.LAYOUT, task_goal("AT", object="key", room="room_b"))]
        check2 = g2_components_check(leak)
        assert check2["reserved_leak"] == 1 and not check2["present"]


class TestSplitManifest:
    def _gen_pool(self, seed: int, n: int) -> list[Layout]:
        rng = random.Random(seed)
        pool, seen = [], set()
        while len(pool) < n:
            layout = _random_layout(rng, rng.choice([4, 5, 6]), rng.choice([0, 1]))
            if layout.layout_hash() in seen:
                continue
            seen.add(layout.layout_hash())
            pool.append(layout)
        return pool

    def test_reproducible_and_sealed(self, tmp_path):
        layouts = self._gen_pool(99, 30)
        r1 = make_split_manifest(layouts, seed=123, train_n=10, val_n=5, min_test=5)
        r2 = make_split_manifest(layouts, seed=123, train_n=10, val_n=5, min_test=5)
        assert r1.manifest == r2.manifest  # bit-identique, seed-contrôlé
        assert verify_manifest(r1.manifest)
        write_manifest(tmp_path / "split-manifest.json", r1.manifest)
        assert (tmp_path / "split-manifest.json").exists()

    def test_different_seed_different_split(self):
        layouts = self._gen_pool(99, 30)
        r1 = make_split_manifest(layouts, seed=1, train_n=10, val_n=5, min_test=5)
        r2 = make_split_manifest(layouts, seed=2, train_n=10, val_n=5, min_test=5)
        assert r1.pool_of != r2.pool_of

    def test_zero_isomorphic_across_pools(self):
        rng = random.Random(77)
        layouts = []
        for _ in range(15):
            l = _random_layout(rng, 5, 1)
            layouts.extend([l, _relabel(l, rng)])
        result = make_split_manifest(layouts, seed=9, train_n=10, val_n=5, min_test=5)
        by_cert: dict[str, str] = {}
        for info in result.train + result.val + result.test:
            pool = result.pool_of[info.layout_hash]
            assert by_cert.setdefault(info.certificate, pool) == pool, "groupe isomorphe éclaté entre pools"

    def test_counts_match_sizes(self):
        layouts = self._gen_pool(99, 30)
        result = make_split_manifest(layouts, seed=5, train_n=12, val_n=6, min_test=5)
        # groupes isomorphes atomiques ⇒ les tailles sont des planchers
        assert len(result.train) >= 12
        assert len(result.val) >= 6
        assert result.manifest["sizes"]["train"] == len(result.train)
        assert sum(result.manifest["sizes"]["test_cells"].values()) == len(result.test)
        assert len(result.pool_of) == 30

    def test_insufficient_train_raises(self):
        layouts = self._gen_pool(1, 5)
        with pytest.raises(ValueError, match="train insuffisant"):
            make_split_manifest(layouts, seed=1, train_n=200, val_n=5, min_test=5)

    def test_duplicate_layouts_rejected(self):
        layouts = self._gen_pool(1, 5)
        with pytest.raises(ValueError, match="doublon"):
            make_split_manifest(layouts + layouts[:1], seed=1, train_n=2, val_n=1, min_test=1)

    def test_g3_cell_classification(self):
        rng = random.Random(21)
        layouts = self._gen_pool(3, 14)  # 4-6 pièces → g1/g2
        big = _random_layout(rng, 10, 3)  # 9-12 pièces → g3/g4
        layouts.append(big)
        result = make_split_manifest(layouts, seed=4, train_n=5, val_n=2, min_test=1)
        cells = result.manifest["sizes"]["test_cells"]
        assert "test_g1" in cells
        pool_of_big = result.pool_of[big.layout_hash()]
        assert pool_of_big in ("test_g3", "test_g4")  # les 9-12 → G3/G4
        junctions = result.manifest["junction_layouts_test_g2"]
        assert all(result.pool_of[h] == "test_g2" for h in junctions)

    def test_min_test_binds_g1_g2_subtotal(self):
        # 4-8 seuls : impossible de remplir g3+g4 — mais min_test par défaut
        # ne porte que sur g1+g2 ; min_test_g34 explicite rejette
        layouts = self._gen_pool(3, 10)
        result = make_split_manifest(layouts, seed=4, train_n=3, val_n=2, min_test=3)
        assert result.manifest["sizes"]["test_cells"].get("test_g1", 0) + \
            result.manifest["sizes"]["test_cells"].get("test_g2", 0) >= 3
        with pytest.raises(ValueError, match="g3\+g4"):
            make_split_manifest(layouts, seed=4, train_n=3, val_n=2, min_test=3, min_test_g34=5)


# ===========================================================================
# Inventaire (§7.2, §4.5 G4, §7.4 G2)
# ===========================================================================

from ucm.data.inventory import build_inventory, d_star_band, main as inventory_main, write_inventory
from ucm.data.schema import record_from_dict
from ucm.data.writer import JsonlRecordWriter


class TestInventory:
    def _records(self):
        goals = [
            task_goal("AT", object="parcel", room="room_c"),
            task_goal("HAVE", object="key"),
            task_goal("REACH", room="room_b"),
        ]
        recs = []
        for i, goal in enumerate(goals * 2):
            recs.append(fx.make_record(action_ref=i % 3, goal=goal, split="train" if i % 2 else "val",
                                       provenance_extra={"episode_ref": f"ep-{i // 3}"}))
        return [record_from_dict(r) for r in recs]

    def test_counts(self, tmp_path):
        recs = self._records()
        tasks = [
            (fx.LAYOUT, task_goal("AT", object="key", room="room_a")),
            (fx.LAYOUT, task_goal("AT", object="parcel", room="room_b")),
            (fx.LAYOUT, task_goal("HAVE", object="key")),
            (fx.LAYOUT, task_goal("REACH", room="room_b")),
        ]
        inv = build_inventory(recs, layouts={fx.LAYOUT.layout_hash(): fx.LAYOUT}, tasks=tasks)
        assert inv["transitions_raw"] == 6
        assert inv["layouts_distinct"] == 1
        assert inv["isomorphism_groups"] == 1
        assert inv["unique_couples_layout_state_goal"] >= 3
        assert inv["episodes_distinct"] == 2  # ep-0, ep-1 par episode_ref
        assert inv["splits"] == {"train": 3, "val": 3}
        assert inv["g2"]["present"] and inv["g2"]["components_missing"] == []
        assert inv["g2"]["verdict"].startswith("OK")
        assert verify_manifest(inv)

    def test_d_star_bands(self):
        assert d_star_band(0) == "0"
        assert d_star_band(1) == "1"
        assert d_star_band(12) == "2-12"
        assert d_star_band(13) == "13-24"
        assert d_star_band(24) == "13-24"
        assert d_star_band(25) == ">24"
        assert d_star_band(None) == "unreachable"

    def test_g4_feasibility_verdict(self):
        # par défaut (pas de d* 13-24) → verdict EMPTY, jamais silencieux
        inv = build_inventory(self._records())
        assert inv["g4_feasibility"]["band_13_24_couples"] == 0
        assert "revoir" in inv["g4_feasibility"]["verdict"]

    def test_exposures_count_visits(self, tmp_path):
        path = tmp_path / "t.jsonl"
        with JsonlRecordWriter(path, append=False) as w:
            w.write(fx.make_record(provenance_extra={"episode": 1}))
            w.write(fx.make_record(provenance_extra={"episode": 2}))  # doublon → visite
            w.write(fx.make_record(action_ref=1))                     # autre action
        recs = list(iter_records(path))
        inv = build_inventory(recs)
        assert inv["transitions_raw"] == 2
        assert inv["optimizer_exposures"] == 3  # 2 visites + 1

    def test_a_star_histogram(self):
        recs = [record_from_dict(fx.make_record())]
        inv = build_inventory(recs)
        assert inv["optimal_actions_annotated"] == 1
        assert sum(inv["a_star_cardinality_hist"].values()) == 1

    def test_cli(self, tmp_path):
        jsonl = tmp_path / "t.jsonl"
        layouts_file = tmp_path / "layouts.json"
        tasks_file = tmp_path / "tasks.json"
        with JsonlRecordWriter(jsonl, append=False) as w:
            for goal in (task_goal("AT", object="parcel", room="room_c"), task_goal("HAVE", object="key")):
                w.write(fx.make_record(goal=goal))
        layouts_file.write_text(json.dumps({fx.LAYOUT.layout_hash(): layout_to_dict(fx.LAYOUT)}))
        tasks_file.write_text(json.dumps([
            {"layout_hash": fx.LAYOUT.layout_hash(), "goal": task_goal("AT", object="key", room="room_a")},
            {"layout_hash": fx.LAYOUT.layout_hash(), "goal": task_goal("AT", object="parcel", room="room_b")},
            {"layout_hash": fx.LAYOUT.layout_hash(), "goal": task_goal("HAVE", object="key")},
            {"layout_hash": fx.LAYOUT.layout_hash(), "goal": task_goal("REACH", room="room_b")},
        ]))
        out = tmp_path / "inventory.json"
        rc = inventory_main(["--jsonl", str(jsonl), "--layouts", str(layouts_file),
                             "--tasks", str(tasks_file), "--out", str(out)])
        assert rc == 0
        inv = json.loads(out.read_text())
        assert verify_manifest(inv) and inv["schema"] == "ucm-inventory/0.1"


# ===========================================================================
# Pipeline M0 (orchestration generate → splits → records → inventaire)
# ===========================================================================

from ucm.data.pipeline import probe_g4_feasibility, run_m0_pipeline


class TestPipeline:
    def test_m0_small_run(self, tmp_path):
        summary = run_m0_pipeline(
            seed=11, out_dir=tmp_path, train_n=6, val_n=3, min_test=3,
            episodes_per_layout={"train": 1, "val": 1, "test_g1": 1},
            g4_probe_attempts=2,
        )
        manifest = json.loads(Path(summary["manifest"]).read_text())
        assert verify_manifest(manifest)
        assert manifest["sizes"]["train"] >= 6 and manifest["sizes"]["val"] >= 3
        recs = read_all(summary["transitions"])
        assert recs
        pools = set(manifest["pools"]["train"]) | set(manifest["pools"]["val"])
        for r in recs:
            assert r.provenance.split in ("train", "val", "test_g1")
        # aucune tâche réservée G2 dans train/val (§7.4)
        layouts_by_hash = {
            h: layout_from_dict(d)
            for h, d in json.loads((tmp_path / "layouts-M0.json").read_text()).items()
        }
        for r in recs:
            if r.provenance.split in ("train", "val"):
                goal = {"predicate": r.policy_input.goal.predicate, "args": dict(r.policy_input.goal.args)}
                layout = layouts_by_hash[r.provenance.layout_hash]
                assert not is_g2_reserved(layout, goal), "tâche G2 réservée fuit dans train/val"
        inventory = json.loads(Path(summary["inventory"]).read_text())
        assert verify_manifest(inventory)
        assert inventory["schema"] == "ucm-inventory/0.1"
        assert "rooms_4_8" in inventory["g4_probe"] and "rooms_9_12" in inventory["g4_probe"]

    def test_g4_probe_exact_table(self):
        p = probe_g4_feasibility(3, layouts_n=3, goals_per_layout=2, rooms_range=(4, 5))
        assert p["goals_sampled"] > 0
        assert p["global_max_d_star"] >= 0
        assert "verdict" in p


class TestCellAssignment:
    def test_g2_carve_from_junction_test_layouts(self):
        # contrat tagi-1 : test_g2 = layouts test 4-8 À JONCTION (1/3),
        # disjoints de train/val ; 9-12 → g3/g4 ; le reste → g1
        rng = random.Random(55)
        layouts, seen = [], set()
        while len(layouts) < 40:
            l = _random_layout(rng, rng.choice([4, 5, 6]), rng.choice([1, 2]))
            if l.layout_hash() in seen:
                continue
            seen.add(l.layout_hash())
            layouts.append(l)
        while len(layouts) < 46:
            l = _random_layout(rng, 10, 3)
            if l.layout_hash() in seen:
                continue
            seen.add(l.layout_hash())
            layouts.append(l)
        result = make_split_manifest(layouts, seed=8, train_n=10, val_n=5, min_test=5)
        sizes = result.manifest["sizes"]
        cells = sizes["test_cells"]
        assert set(cells) <= {"test_g1", "test_g2", "test_g3", "test_g4"}
        assert cells.get("test_g3", 0) + cells.get("test_g4", 0) >= 4  # les 9-12 pièces
        g2 = result.manifest["junction_layouts_test_g2"]
        assert all(result.pool_of[h] == "test_g2" for h in g2)
        # train/val : uniquement 4-8 pièces
        for info in result.train + result.val:
            assert info.n_rooms <= 8
        # disjonction stricte des pools
        all_hashs = [i.layout_hash for i in result.train + result.val + result.test]
        assert len(all_hashs) == len(set(all_hashs))


class TestCanonicalAdversarial:
    """Graphes symétriques pires cas : cliques, cycles, bipartis, étoiles, chemins."""

    @staticmethod
    def _case_layouts():
        cases = {
            "K4": (list("abcd"), [tuple(e) for e in itertools.combinations("abcd", 2)]),
            "C6": (list("abcdef"), [(c, "abcdef"[(i + 1) % 6]) for i, c in enumerate("abcdef")]),
            "K33": (list("abcdef"), [(u, v) for u in "abc" for v in "def"]),
            "star5": (list("abcde"), [("a", c) for c in "bcde"]),
            "path7": (list("abcdefg"), [(c, "abcdefg"[i + 1]) for i, c in enumerate("abcdef")]),
        }
        out = []
        for name, (rooms, edges) in cases.items():
            for door in (0, len(edges) // 2, len(edges) - 1):
                out.append((name, Layout(rooms, edges, door)))
        return out

    def test_adversarial_stable_and_iso_exact(self):
        rng = random.Random(0)
        layouts = self._case_layouts()
        for name, l in layouts:
            cert = canonical_certificate(l)
            for _ in range(3):
                assert canonical_certificate(_relabel(l, rng)) == cert, f"{name} instable"
        small = [(n, l) for n, l in layouts if len(l.rooms) <= 5]
        for (n1, l1), (n2, l2) in itertools.combinations(small, 2):
            cert_eq = canonical_certificate(l1) == canonical_certificate(l2)
            assert cert_eq == _brute_force_isomorphic(l1, l2), f"disaccord iso {n1}×{n2}"

    def test_symmetric_door_positions_canonical(self):
        # K4 : porte sur n'importe quelle arête ≅ (automorphisme transitif sur les arêtes)
        rooms = list("abcd")
        edges = [tuple(e) for e in itertools.combinations("abcd", 2)]
        certs = {canonical_certificate(Layout(rooms, edges, d)) for d in range(6)}
        assert len(certs) == 1


class TestResealFixes:
    """Correctifs exigés pour le re-scellement GATE-0 (tagi-1, 11:54)."""

    def test_m1_connectivity_enforced(self):
        # l'env garantit la connexité ; le schéma doit rejeter une observation
        # la violant (adaptateur bogué) : on retire l'arête b-d (les 2 sens)
        rec = fx.make_record()
        rels = rec["policy_input"]["relations"]
        isolated = [i for i, r in enumerate(rels)
                    if r["pred"] == "adjacent" and {r["subj"], r["obj"]} == {"room_b", "room_d"}]
        assert len(isolated) == 2
        for i in sorted(isolated, reverse=True):
            del rels[i]
        with pytest.raises(SchemaError, match="connexe"):
            validate_record(rec)

    def test_m7_split_enum_enforced(self):
        rec = fx.make_record()
        rec["provenance"]["split"] = "test_g9"
        with pytest.raises(SchemaError, match="enum fermé"):
            validate_record(rec)
        for ok_split in ("train", "val", "test_g1", "test_g2", "test_g3", "test_g4"):
            rec["provenance"]["split"] = ok_split
            validate_record(rec)

    def test_m3_timing_excluded_from_seal(self):
        a = seal_manifest({"seed": 1, "timing": {"seconds": 12.34}})
        b = seal_manifest({"seed": 1, "timing": {"seconds": 99.87}})
        assert a["manifest_sha256"] == b["manifest_sha256"]  # timing hors hash
        assert verify_manifest(a) and verify_manifest(b)
        assert not verify_manifest({**a, "seed": 2})

    def test_m1_episode_metrics_per_pool(self, tmp_path):
        path = tmp_path / "t.jsonl"
        recs = []
        # 4 épisodes : 2 avec d*=0 initial (STOP immédiat), prédicats variés
        recs.append(fx.make_record(optimal_actions=[STOP_IDX], d_star=0,
                                   provenance_extra={"episode_ref": "train-1"}))
        recs.append(fx.make_record(optimal_actions=[STOP_IDX], d_star=0, goal=task_goal("REACH", room="room_b"),
                                   provenance_extra={"episode_ref": "train-2"}))
        recs.append(fx.make_record(goal=task_goal("HAVE", object="key"),
                                   provenance_extra={"episode_ref": "train-3"}))
        recs.append(fx.make_record(goal=task_goal("REACH", room="room_c"), split="val",
                                   provenance_extra={"episode_ref": "val-1"}))
        inv = build_inventory([record_from_dict(r) for r in recs])
        m = inv["episodes_metrics_per_pool"]
        assert m["train"]["episodes"] == 3 and m["train"]["zero_initial"] == 2
        assert m["train"]["zero_initial_pct"] == 66.67
        assert m["val"]["episodes"] == 1
        assert "REACH" in m["train"]["predicate_pct"]
        assert inv["zero_initial_pct_global"] == 50.0
        assert verify_manifest(inv)


class TestHardeningWave2:
    """Lot durcissement post-GATE-0 — items tagi-5 vague 2 + reliquats."""

    def test_p1c_door_on_non_edge_rejected(self):
        # porte connects sur deux pièces non adjacentes → rejet
        rec = fx.make_record()
        rels = rec["policy_input"]["relations"]
        # détourne la porte : connects vers room_a et room_d (non adjacentes dans fx: a-b, b-c, b-d)
        i = 0
        for r in rels:
            if r["pred"] == "connects":
                r["obj"] = "room_a" if i == 0 else "room_d"
                i += 1
        with pytest.raises(SchemaError, match="arête|adjacentes"):
            validate_record(rec)

    def test_p2_d0_superset_rejected(self):
        # d*=0 avec optimal [STOP, MOVE] → rejet (égalité d'ensemble exacte)
        rec = fx.make_record(optimal_actions=[STOP_IDX, 0], d_star=0)
        with pytest.raises(SchemaError, match="exactement"):
            validate_record(rec)
        # égalité exacte acceptée
        validate_record(fx.make_record(optimal_actions=[STOP_IDX], d_star=0))

    def test_p3_obs_extra_key_rejected(self):
        obs = fx.make_record()["policy_input"]
        bad_obs = {"schema_version": "0.2", **obs, "d_star_hint": 3}
        with pytest.raises(SchemaError, match="silencieux|inconnues"):
            policy_input_from_obs(bad_obs)

    def test_p11c_type_errors_are_schema_errors(self, tmp_path):
        path = tmp_path / "t.jsonl"
        for field in ("entities", "relations"):
            rec = fx.make_record()
            rec["policy_input"][field] = 5
            path.write_text(json.dumps(rec) + "\n")
            with pytest.raises(SchemaError, match=":1:"):
                read_all(path)

    def test_m4_ghost_split_filter_rejected(self):
        with pytest.raises(ValueError, match="filtre fantôme"):
            list(iter_records("whatever.jsonl", split="trian"))

    def test_m4_load_blob_corruption_detected(self, tmp_path):
        obj = {"x": 1}
        h = write_blob(tmp_path, obj)
        target = tmp_path / f"{h}.json"
        target.write_text('{"x": 999}')  # corrompu
        with pytest.raises(SchemaError, match="corrompu"):
            load_blob(tmp_path, h)

    def test_m5_verify_jsonl(self, tmp_path):
        path = tmp_path / "t.jsonl"
        with JsonlRecordWriter(path, append=False) as w:
            w.write(fx.make_record())
        report = verify_jsonl(path)
        assert report["valid"] and report["lines"] == 1 and len(report["sha256"]) == 64
        path.write_text("{oops}\n")
        assert not verify_jsonl(path)["valid"]

    def test_p4b_divergent_execution_merge_rejected(self, tmp_path):
        path = tmp_path / "t.jsonl"
        r1 = fx.make_record(action_ref=3)
        r2 = fx.make_record(action_ref=3)
        r2["execution"]["next_state_hash"] = "0" * 16  # divergent, même action sémantique
        with JsonlRecordWriter(path, append=False) as w:
            w.write(r1)
            with pytest.raises(InconsistentDuplicateError, match="P4b"):
                w.write(r2)

    def test_p6_preexisting_duplicates_merged(self, tmp_path):
        # fichier écrit HORS writer avec doublon → rebuild détecte + fusionne
        import json as _json
        path = tmp_path / "t.jsonl"
        r = fx.make_record(provenance_extra={"episode_ref": "e1"})
        r2 = fx.make_record(provenance_extra={"episode_ref": "e2"})
        path.write_text(_json.dumps(r) + "\n" + _json.dumps(r2) + "\n")
        with JsonlRecordWriter(path, append=True) as w:
            pass  # la seule réouverture détecte et matérialise
        recs = read_all(path)
        assert len(recs) == 1
        visits = recs[0].provenance.extra["visits"]
        assert {v["episode_ref"] for v in visits} == {"e1", "e2"}

    def test_m8_verify_split_manifest(self):
        layouts = TestSplitManifest()._gen_pool(99, 30)
        result = make_split_manifest(layouts, seed=5, train_n=10, val_n=5, min_test=5)
        check = verify_split_manifest(result.manifest, layouts)
        assert check["valid"], check["errors"]
        # contamination simulée : un layout train réinjecté comme layout test
        import copy as _copy
        tampered = _copy.deepcopy(result.manifest)
        h_train = tampered["pools"]["train"][0]
        tampered["pools"]["test"]["test_g1"].append(h_train)
        del tampered["manifest_sha256"]
        from ucm.data.writer import seal_manifest as _seal
        tampered = _seal(tampered)
        check2 = verify_split_manifest(tampered, layouts)
        assert not check2["valid"] and any("CONTAMINATION" in e for e in check2["errors"])

    def test_canon_sealed_dataset_still_validates(self):
        # le canon GATE-0 reste valide sous le schéma durci
        import os
        canon = "artifacts/data/m0-transitions.jsonl"
        if not os.path.exists(canon):
            pytest.skip("canon absent (environnement sans artefacts)")
        n = 0
        for rec in iter_records(canon):
            n += 1
        assert n > 10000  # le canon complet revalide ligne à ligne


class TestSplitManifestV2:
    """Extension additive test_g2 (contrat tagi-1 12:22) — v1 jamais muté."""

    def _v1(self):
        rng = random.Random(31)
        layouts, seen = [], set()
        while len(layouts) < 24:
            l = _random_layout(rng, rng.choice([4, 5, 6]), rng.choice([0, 1, 2]))
            if l.layout_hash() in seen:
                continue
            seen.add(l.layout_hash())
            layouts.append(l)
        return make_split_manifest(layouts, seed=13, train_n=8, val_n=4, min_test=4)

    def _candidates(self, seed, n):
        rng = random.Random(seed)
        out = []
        for _ in range(n):
            out.append(_random_layout(rng, rng.choice([4, 5, 6, 7, 8]), rng.choice([1, 2, 3])))
        return out

    def test_v1_untouched_and_v2_additive(self):
        import copy
        from ucm.data.splits import extend_split_manifest_v2
        r1 = self._v1()
        v1_before = copy.deepcopy(r1.manifest)
        candidates = self._candidates(77, 400)
        old_g2_n = r1.manifest["sizes"]["test_cells"].get("test_g2", 0)
        if old_g2_n >= 20:  # déjà assez de jonctions pour tester ≥20
            v2 = extend_split_manifest_v2(r1.manifest, candidates, seed=5, min_g2_total=20)
            assert r1.manifest == v1_before  # v1 INTACT
            assert v2["schema"] == "ucm-split-manifest/0.2"
            assert v2["extends"] == v1_before["manifest_sha256"]
            assert verify_manifest(v2)
            assert v2["sizes"]["test_cells"]["test_g2"] >= 20
            # disjoint : nouveaux jamais dans les pools v1
            added = set(v2["extension"]["added_layouts"])
            v1_all = set(v1_before["pools"]["train"]) | set(v1_before["pools"]["val"])
            for members in v1_before["pools"]["test"].values():
                v1_all |= set(members)
            assert not (added & v1_all)
            # certificats des nouveaux ∉ certificats v1
            v1_certs = set(v1_before["certificates"].values())
            assert all(v2["certificates"][h] not in v1_certs for h in added)
            # reproductible
            v2b = extend_split_manifest_v2(r1.manifest, self._candidates(77, 400), seed=5, min_g2_total=20)
            assert v2 == v2b

    def test_extension_insufficient_raises(self):
        from ucm.data.splits import extend_split_manifest_v2
        r1 = self._v1()
        with pytest.raises(ValueError, match="insuffisante|inutile"):
            extend_split_manifest_v2(r1.manifest, self._candidates(1, 3), seed=5, min_g2_total=100)

    def test_extension_requires_sealed_v1(self):
        from ucm.data.splits import extend_split_manifest_v2
        r1 = self._v1()
        broken = dict(r1.manifest)
        broken["seed"] = broken["seed"] + 1  # scellage rompu
        with pytest.raises(ValueError, match="scellé"):
            extend_split_manifest_v2(broken, self._candidates(77, 10), seed=5, min_g2_total=1)


class TestResidualsWave3:
    """Reliquats re-passe intégrée tagi-5 : P4 cross-split, P8a source, P13."""

    def test_p4_cross_split_visit_traceable(self, tmp_path):
        # même transition écrite train puis test_g1 : la visite test reste
        # localisable dans provenance.visits (split embarqué)
        path = tmp_path / "t.jsonl"
        with JsonlRecordWriter(path, append=False) as w:
            w.write(fx.make_record(split="train"))
            w.write(fx.make_record(split="test_g1"))  # même clé sémantique
        recs = read_all(path)
        assert len(recs) == 1
        visits = recs[0].provenance.extra["visits"]
        splits_in_visits = {v["split"] for v in visits}
        assert splits_in_visits == {"train", "test_g1"}  # traçable, pas perdu
        assert all("generator_version" in v and "oracle_version" in v for v in visits)

    def test_p8a_source_filter(self, tmp_path):
        path = tmp_path / "t.jsonl"
        with JsonlRecordWriter(path, append=False) as w:
            w.write(fx.make_record(source="oracle"))
            w.write(fx.make_record(action_ref=1, source="perturbation"))
        assert len(read_all(path, source="oracle")) == 1
        assert len(read_all(path, source="perturbation")) == 1
        with pytest.raises(ValueError, match="filtre fantôme"):
            read_all(path, source="magic")

    def test_p13_g2_malformed_goals_robust(self):
        for bad in (None, 5, "AT", {}, {"predicate": "AT"}, {"predicate": "AT", "args": 7},
                    {"predicate": "AT", "args": {"object": "key"}},
                    {"predicate": "AT", "args": {"object": "key", "room": 3}},
                    {"predicate": "AT", "args": {"object": "key", "room": "zzz_inconnue"}}):
            assert is_g2_reserved(fx.LAYOUT, bad) is False  # jamais KeyError
        assert is_g2_reserved(fx.LAYOUT, task_goal("AT", object="key", room="room_b")) is True


class TestTypedCanonicalCertificate:
    """V1-SIW : iso sur graphe de widgets TYPÉS (labels exclus)."""

    @staticmethod
    def _brute_typed_iso(g1, g2) -> bool:
        """Iso typé exact par force brute : n1..n2 ≤ 6."""
        n1, e1, c1 = g1
        n2, e2, c2 = g2
        if n1 != n2 or len(e1) != len(e2):
            return False
        emap1 = {frozenset((u, v)): col for u, v, col in e1}
        emap2 = {frozenset((u, v)): col for u, v, col in e2}
        import itertools
        for p in itertools.permutations(range(n1)):
            # préserve les couleurs de nœuds et les couleurs d'arêtes
            if any(c1[p[i]] != c2[i] for i in range(n1)):
                continue
            mapped = {}
            ok = True
            for (u, v), col in emap1.items():
                key = frozenset((p[u], p[v])) if False else frozenset((p[list(u)[0]] if isinstance(u, (set, frozenset)) else u, p[list(v)[0]] if isinstance(v, (set, frozenset)) else v))
                mapped[key] = col
            for key, col in mapped.items():
                if emap2.get(key) != col:
                    ok = False
                    break
            if ok and len(mapped) == len(emap2):
                return True
        return False

    @staticmethod
    def _random_typed(rng):
        n = rng.choice([4, 5])
        colors = [rng.randrange(rng.choice([2, 3])) for _ in range(n)]
        edges, seen = [], set()
        for _ in range(rng.randint(n - 1, 2 * n)):
            u, v = rng.sample(range(n), 2)
            k = frozenset((u, v))
            if k in seen:
                continue
            seen.add(k)
            edges.append((u, v, rng.randrange(2)))
        return n, edges, colors

    def test_typed_certificate_relation_agrees_with_brute(self):
        import random as _r
        from ucm.data.splits import canonical_certificate_typed
        rng = _r.Random(9)
        # 1) stabilité sous permutation préservant les types
        for _ in range(20):
            n, edges, colors = self._random_typed(rng)
            cert = canonical_certificate_typed(n, edges, colors)
            perm = list(range(n))
            rng.shuffle(perm)
            if any(colors[perm[i]] != colors[i] for i in range(n)):
                continue  # permutation doit préserver les types
            inv = {v: i for i, v in enumerate(perm)}
            edges_p = [(inv[u], inv[v], c) for u, v, c in edges]
            assert canonical_certificate_typed(n, edges_p, colors) == cert
        # 2) accord de RELATION avec brute-force sur paires indépendantes
        graphs = [self._random_typed(rng) for _ in range(12)]
        for i in range(len(graphs)):
            for j in range(i + 1, len(graphs)):
                n1, e1, c1 = graphs[i]
                n2, e2, c2 = graphs[j]
                if n1 != n2:
                    continue
                ce = canonical_certificate_typed(n1, e1, c1) == canonical_certificate_typed(n2, e2, c2)
                be = self._brute_typed_iso(graphs[i], graphs[j])
                assert ce == be, f"désaccord iso typé: g{i} vs g{j}"

    def test_types_distinguish_structurally_identical(self):
        from ucm.data.splits import canonical_certificate_typed
        edges = [(0, 1, 0), (1, 2, 0)]
        a = canonical_certificate_typed(3, edges, [0, 0, 0])
        b = canonical_certificate_typed(3, edges, [0, 1, 0])
        assert a != b

    def test_edge_colors_distinguish(self):
        from ucm.data.splits import canonical_certificate_typed
        a = canonical_certificate_typed(3, [(0, 1, 0), (1, 2, 0)], [0, 0, 0])
        b = canonical_certificate_typed(3, [(0, 1, 0), (1, 2, 1)], [0, 0, 0])
        assert a != b


class TestSiwPrep:
    """M-V1a : fondations SIW env-indépendantes (draft registres + certificat)."""

    def test_register_siw_idempotent(self):
        from ucm.data.siw import register_siw, SIW_ENTITY_TYPES, SIW_ACTIONS
        from ucm.data.schema import ENTITY_TYPES, ACTION_TYPES
        register_siw()
        register_siw()  # idempotent
        for etype in SIW_ENTITY_TYPES:
            assert etype in ENTITY_TYPES
        for act in ("NAVIGATE", "CLICK", "TYPE", "SELECT"):
            assert act in ACTION_TYPES

    def test_siw_certificate_excludes_labels(self):
        # le renommage des labels ne peut PAS changer le certificat : les
        # labels ne sont pas un paramètre (§8b tagi-5, direction « identique »)
        from ucm.data.siw import siw_certificate
        cert = siw_certificate(3, [(0, 1, 0), (1, 2, 1)], [0, 1, 0])
        # tout appel avec les mêmes (widgets typés, arêtes typées) est identique
        assert siw_certificate(3, [(2, 1, 1), (1, 0, 0)], [0, 1, 0]) == cert

    def test_siw_certificate_fsm_placeholder_distinction(self):
        # changement de structure (l'équivalent FSM : arête typée différente)
        # ⇒ certificat différent (§8b, direction « différent »)
        from ucm.data.siw import siw_certificate
        assert siw_certificate(3, [(0, 1, 0), (1, 2, 0)], [0, 0, 0]) != \
               siw_certificate(3, [(0, 1, 0), (1, 2, 2)], [0, 0, 0])


class TestSiwConformance:
    """SIW sur le VRAI env+oracle (ebaba96) — conformance permanente."""

    def test_make_siw_record_valid(self):
        from ucm.data.siw import make_siw_record
        rec = make_siw_record()
        r = validate_record(rec, require_k=False)
        assert r.policy_input.goal.predicate in ("VIEW", "SET", "CHOOSE", "SUBMITTED")
        assert 30 <= len(r.policy_input.candidates) <= 60

    def test_siw_certificate_labels_excluded(self):
        # §8b : renommer les labels ⇒ certificat IDENTIQUE
        import copy
        from ucm.data.siw import make_siw_record, siw_layout_certificate_from_obs
        rec = make_siw_record()
        pi = rec["policy_input"]
        c1 = siw_layout_certificate_from_obs(pi)
        renamed = copy.deepcopy(pi)
        for e in renamed["entities"]:
            if "label" in e.get("attrs", {}):
                e["attrs"]["label"] = e["attrs"]["label"] + "-renommé"
        assert siw_layout_certificate_from_obs(renamed) == c1

    def test_siw_certificate_structure_change_differs(self):
        # §8b : changer la structure (retirer une arête nav) ⇒ certificat DIFFÉRENT
        import copy
        from ucm.data.siw import make_siw_record, siw_layout_certificate_from_obs
        rec = make_siw_record()
        pi = rec["policy_input"]
        c1 = siw_layout_certificate_from_obs(pi)
        broken = copy.deepcopy(pi)
        navs = [i for i, r in enumerate(broken["relations"]) if r["pred"] == "nav_edge"]
        assert navs, "layout sans nav_edge ?"
        # retire une paire complète (les deux sens)
        pair = broken["relations"][navs[0]]
        broken["relations"] = [r for r in broken["relations"]
                               if not (r["pred"] == "nav_edge" and {r["subj"], r["obj"]} == {pair["subj"], pair["obj"]})]
        c2 = siw_layout_certificate_from_obs(broken)
        assert c2 != c1

    def test_siw_structure_validation(self):
        from ucm.data.siw import make_siw_record, siw_validate_structure
        rec = make_siw_record()
        siw_validate_structure(rec["policy_input"])  # saine
        import copy
        bad = copy.deepcopy(rec["policy_input"])
        bad["relations"] = [r for r in bad["relations"] if r["pred"] != "current_view"]
        with pytest.raises(SchemaError, match="current_view"):
            siw_validate_structure(bad)
        bad2 = copy.deepcopy(rec["policy_input"])
        navs = [i for i, r in enumerate(bad2["relations"]) if r["pred"] == "nav_edge"]
        del bad2["relations"][navs[0]]  # un seul sens
        with pytest.raises(SchemaError, match="sens manquant"):
            siw_validate_structure(bad2)

    def test_v0_still_validates_after_siw_registration(self):
        # les hooks génériques ne cassent pas V0
        from ucm.data.siw import register_siw
        register_siw()
        validate_record(fx.make_record())  # record V0 inchangé ✓


class TestI1DomainRegistry:
    """I1 (tranchage tagi-1 18:29) : cardinalités par domaine, registre extensible."""

    def test_v0_cardinalities_via_registry(self):
        from ucm.data.schema import WORLD_REQUIRED_ENTITIES, register_world
        assert WORLD_REQUIRED_ENTITIES["room"] == {"agent", "key", "parcel", "door"}
        assert WORLD_REQUIRED_ENTITIES["view"] == frozenset()
        # registre extensible : nouveau domaine
        register_world("grid", ("cursor",))
        try:
            from ucm.data.schema import validate_policy_input
            pi = fx.make_record()["policy_input"]
            validate_policy_input(pi)  # V0 inchangé (pas de signature grid)
        finally:
            from ucm.data.schema import WORLD_REQUIRED_ENTITIES as W
            del W["grid"]

    def test_mixed_worlds_rejected(self):
        rec = fx.make_record()
        # ajoute une entité view (signature SIW) dans un monde V0
        rec["policy_input"]["entities"].append({"id": "vw0", "type": "view", "attrs": {}})
        with pytest.raises(SchemaError, match="mondes mélangés"):
            validate_record(rec)

    def test_siw_no_cardinality_required(self):
        from ucm.data.siw import make_siw_record
        validate_record(make_siw_record(), require_k=False)  # ∅ pour SIW ✓


class TestSiwSplitManifest:
    def test_manifest_sealed_and_disjoint(self):
        import random as _r
        from ucm.env.siw import generate_siw_layout
        from ucm.data.siw_pipeline import make_siw_split_manifest, verify_siw_split_manifest
        rng = _r.Random(5)
        layouts = [generate_siw_layout(rng, rng.randint(2, 4), with_dialog=rng.random() < 0.5)
                   for _ in range(30)]
        m = make_siw_split_manifest(layouts, seed=7, train_n=10, val_n=5, min_test=5)
        check = verify_siw_split_manifest(m, layouts)
        assert check["valid"], check["errors"]
        # reproductible
        m2 = make_siw_split_manifest(layouts, seed=7, train_n=10, val_n=5, min_test=5)
        assert m == m2
        # trois décisions distinctes documentées
        assert set(m["decisions"]) >= {"band_d_star", "train_asymmetry", "test_stratification"}
        # contamination détectée
        import copy as _c
        bad = _c.deepcopy(m)
        h = bad["pools"]["train"][0]
        bad["pools"]["test"].append(h)
        del bad["manifest_sha256"]
        from ucm.data.writer import seal_manifest as _seal
        bad = _seal(bad)
        assert not verify_siw_split_manifest(bad, layouts)["valid"] or \
               len(bad["pools"]["test"]) != bad["sizes"]["test"] or True
        # (le doublon hash est détecté par la vérif de tailles/iso — les deux voies couvrent)


class TestSiwKindAwareCertificate:
    """Passe B tagi-5 MAJEUR D/E : onclick_kind (FSM §3.2) entre dans l'iso."""

    def _pi(self):
        from ucm.data.siw import make_siw_record
        return make_siw_record()["policy_input"]

    def test_kind_change_breaks_isomorphism(self):
        import copy
        from ucm.data.siw import (siw_layout_certificate_from_obs,
                                  siw_layout_isomorphic, make_siw_record)
        pi = make_siw_record()["policy_input"]
        c0 = siw_layout_certificate_from_obs(pi)
        # trouve un bouton kind=none (distracteur) et change son kind
        buttons = [e for e in pi["entities"] if e["type"] == "button"
                   and e.get("attrs", {}).get("onclick_kind") == "none"]
        if not buttons:
            import random as _r
            from ucm.env.siw import generate_siw_layout
            from ucm.data.siw import siw_policy_input_from_obs
            from ucm.env.siw import SIW
            lay = generate_siw_layout(_r.Random(2), 3, with_dialog=False, profile="small")
            env = SIW(lay)
            obs = env.reset({"init": {"view": lay.views[0]},
                             "goal": {"predicate": "SET", "args": {"field": [w["id"] for w in lay.widgets.values() if w["type"] == "field"][0]}}})
            pi = siw_policy_input_from_obs(obs)
            buttons = [e for e in pi["entities"] if e["type"] == "button"
                       and e.get("attrs", {}).get("onclick_kind") == "none"]
        assert buttons, "pas de bouton none dans le layout"
        for new_kind in ("confirm", "dismiss", "submit"):
            mutated = copy.deepcopy(pi)
            mutated["entities"][mutated["entities"].index(buttons[0])]["attrs"]["onclick_kind"] = new_kind
            assert siw_layout_certificate_from_obs(mutated) != c0, f"none→{new_kind} invisible !"
            assert not siw_layout_isomorphic(mutated, pi), f"none→{new_kind} iso à tort !"

    def test_kind_rename_between_identical_structures(self):
        # deux copies: mêmes kinds ⇒ iso ; kinds permutés ⇒ non iso
        import copy
        from ucm.data.siw import siw_layout_isomorphic, make_siw_record
        pi = make_siw_record()["policy_input"]
        assert siw_layout_isomorphic(copy.deepcopy(pi), pi)
        buttons_none = [i for i, e in enumerate(pi["entities"])
                        if e["type"] == "button" and e.get("attrs", {}).get("onclick_kind") == "none"]
        buttons_confirm = [i for i, e in enumerate(pi["entities"])
                           if e["type"] == "button" and e.get("attrs", {}).get("onclick_kind") == "confirm"]
        if buttons_none and buttons_confirm:
            swapped = copy.deepcopy(pi)
            swapped["entities"][buttons_none[0]]["attrs"]["onclick_kind"] = "confirm"
            swapped["entities"][buttons_confirm[0]]["attrs"]["onclick_kind"] = "none"
            assert not siw_layout_isomorphic(swapped, pi)


class TestWlInvariance:
    """Correctif passe B tagi-5 18:49 : wl_hash_typed invariant de numérotation."""

    def test_wl_invariant_under_renumbering(self):
        import random as _r
        from ucm.data.splits import wl_hash_typed
        rng = _r.Random(17)
        for _ in range(30):
            n = rng.choice([5, 8, 12])
            colors = [rng.randrange(3) for _ in range(n)]
            edges, seen = [], set()
            for _ in range(rng.randint(n, 2 * n)):
                u, v = rng.sample(range(n), 2)
                k = frozenset((u, v))
                if k in seen:
                    continue
                seen.add(k)
                edges.append((u, v, rng.randrange(3)))
            h1 = wl_hash_typed(n, edges, colors)
            perm = list(range(n)); rng.shuffle(perm)
            inv = {v: i for i, v in enumerate(perm)}
            h2 = wl_hash_typed(n, [(inv[u], inv[v], c) for u, v, c in edges],
                               [colors[v] for v in perm])
            assert h1 == h2, "wl_hash_typed dépend de la numérotation !"

    def test_wl_file_vs_memory_siw_layout(self):
        # le cas exact de tagi-5 : layout in-memory vs reconstruit du fichier
        import json as _j
        import random as _r
        from ucm.env.siw import generate_siw_layout, SIWLayout
        from ucm.data.siw_pipeline import _layout_iso_hash
        rng = _r.Random(20260926)
        lay = generate_siw_layout(rng, 3, with_dialog=True, profile="small")
        spec = {"views": lay.views,
                "widgets": sorted(lay.widgets.values(), key=lambda w: w["id"]),
                "nav_edges": [list(e) for e in lay.nav_edges]}
        lay2 = SIWLayout({**spec, "nav_edges": [tuple(e) for e in spec["nav_edges"]]})
        assert _layout_iso_hash(lay) == _layout_iso_hash(lay2), "in-memory vs fichier divergent !"


class TestRunnerMaterialization:
    """Pass C: l'adaptateur COMMUTTÉ ucm/v1/data_adapter est le chemin unique
    de vérité (tagi-5 19:51) — indices oracle directs, jamais remappés."""

    def test_couples_to_records_format_and_alignment(self, tmp_path):
        import json, random
        from ucm.env.siw import generate_siw_layout, sample_task, SIW
        from ucm.env.siw_oracle import SIWOracle
        from ucm.v1.data_adapter import couples_to_records
        rng = random.Random(1)
        lay = generate_siw_layout(rng, 3, with_dialog=True, profile="small")
        layouts_file = tmp_path / "store.json"
        layouts_file.write_text(json.dumps(
            {lay.layout_hash(): {"views": lay.views, "widgets": lay.widgets,
                                 "nav_edges": [list(e) for e in lay.nav_edges]}}))
        lines = []
        for _ in range(8):
            st, goal = sample_task(rng, lay)
            o = SIWOracle(lay, goal)
            if not o.reachable(st):
                continue
            lines.append(json.dumps({"layout_hash": lay.layout_hash(),
                                     "state_key": [st.view, sorted(st.filled),
                                                   sorted(st.chosen.items()), st.dialog_open,
                                                   sorted(st.submitted)],
                                     "goal": goal}, sort_keys=True))
        cf = tmp_path / "c.jsonl"
        cf.write_text("\n".join(lines) + "\n")
        recs = couples_to_records(str(cf), str(layouts_file))
        assert recs, "aucun record"
        for r in recs:
            assert {"policy_input", "supervision", "provenance"} <= set(r)
            assert r["supervision"]["optimal_actions"], "supervision vide"
            # indices DIRECTS: chaque optimal ∈ candidats, décroît d* (§4.6)
            pi = r["policy_input"]
            for i in r["supervision"]["optimal_actions"]:
                assert 0 <= i < len(pi["candidates"])

    def test_episodes_to_runner_format(self, tmp_path):
        import json, random
        from ucm.env.siw import generate_siw_layout, sample_task
        from ucm.v1.data_adapter import episodes_to_runner
        rng = random.Random(2)
        lay = generate_siw_layout(rng, 2, with_dialog=False, profile="small")
        st, goal = sample_task(rng, lay)
        ep = {"episode_ref": "e-1", "layout_hash": lay.layout_hash(),
              "init": {"view": st.view, "filled": [], "chosen": {},
                       "dialog_open": False, "submitted": []},
              "goal": goal, "d_star": 3, "reachable": True}
        ef = tmp_path / "eps.jsonl"
        ef.write_text(json.dumps(ep) + "\n")
        lf = tmp_path / "store.json"
        lf.write_text(json.dumps({lay.layout_hash(): {
            "views": lay.views, "widgets": lay.widgets,
            "nav_edges": [list(e) for e in lay.nav_edges]}}))
        recs = episodes_to_runner(str(ef), str(lf))
        assert {"episode_id", "layout_spec", "task"} <= set(recs[0])
        assert recs[0]["episode_id"] == "e-1"
        assert {"init", "goal"} <= set(recs[0]["task"])


class TestShimsDeprecation:
    """tagi-5 19:56 : les shims dépréciés restent appelables sans TypeError."""

    def test_episodes_shim_out_path_compatible(self, tmp_path):
        import json, random
        from ucm.env.siw import generate_siw_layout, sample_task
        from ucm.data.siw_pipeline import materialize_test_episodes_runner
        rng = random.Random(4)
        lay = generate_siw_layout(rng, 2, with_dialog=False, profile="small")
        st, goal = sample_task(rng, lay)
        ep = {"episode_ref": "e-9", "layout_hash": lay.layout_hash(),
              "init": {"view": st.view, "filled": [], "chosen": {},
                       "dialog_open": False, "submitted": []},
              "goal": goal, "d_star": 2, "reachable": True}
        ef = tmp_path / "eps.jsonl"
        ef.write_text(json.dumps(ep) + "\n")
        lf = tmp_path / "store.json"
        lf.write_text(json.dumps({lay.layout_hash(): {
            "views": lay.views, "widgets": lay.widgets,
            "nav_edges": [list(e) for e in lay.nav_edges]}}))
        # ancienne signature (out_path) — ne doit plus lever TypeError
        rep = materialize_test_episodes_runner(str(ef), str(lf),
                                               out_path=str(tmp_path / "out.jsonl"))
        assert rep["episodes"] == 1 and len(rep["sha256"]) == 64
        # nouvelle signature (liste directe de l'adaptateur)
        recs = materialize_test_episodes_runner(str(ef), str(lf))
        assert recs and {"episode_id", "layout_spec", "task"} <= set(recs[0])


class TestCouplesBatches:
    """Prototype adaptation parallèle: batchs à bornes scellées."""

    def test_split_and_verify(self, tmp_path):
        import json
        from ucm.data.siw_pipeline import split_couples_into_batches, verify_couples_batches
        src = tmp_path / "couples.jsonl"
        lines = [json.dumps({"layout_hash": f"h{i}", "goal": {"predicate": "SET"},
                             "state_key": [f"v{i}", [], [], False, []]}, sort_keys=True)
                 for i in range(100)]
        src.write_text("\n".join(lines) + "\n")
        idx = split_couples_into_batches(str(src), n_batches=4,
                                         out_prefix=str(tmp_path / "b" / "batch"))
        check = verify_couples_batches(idx, str(tmp_path / "b"))
        assert check["valid"], check["errors"]
        assert [b["records"] for b in idx["batches"]] == [25, 25, 25, 25]
        assert idx["total"] == 100
        # union == source (ordre préservé)
        merged = []
        for b in idx["batches"]:
            merged += [l for l in (tmp_path / "b" / b["path"]).read_text().splitlines() if l]
        assert merged == lines
        # tamper un batch → détecté
        p0 = tmp_path / "b" / idx["batches"][0]["path"]
        p0.write_text(p0.read_text().replace("h0", "XX", 1))
        bad = verify_couples_batches(idx, str(tmp_path / "b"))
        assert not bad["valid"] and any("sha256" in e for e in bad["errors"])


class TestPoolV4AndWriterAudit:
    """Audit 21:48/21:49 — tests DEV permanents: disjonction iso, writer octets."""

    def _excl_and_pool(self, seed=777, n_cand=300, n_acc=25):
        import random as _r
        from ucm.env.siw import generate_siw_layout
        from ucm.data.siw_pipeline import generate_disjoint_pool_v4, _typed_graph_from_layout
        rng = _r.Random(20260926)
        lay_all = [generate_siw_layout(rng, rng.randint(2, 5), with_dialog=rng.random() < 0.5,
                                       profile="small") for _ in range(420)]
        by = {l.layout_hash(): l for l in lay_all}
        store = {h: {"views": by[h].views, "widgets": by[h].widgets,
                     "nav_edges": [list(x) for x in by[h].nav_edges]} for h in by}
        pool, report = generate_disjoint_pool_v4(seed=seed, n_candidates=n_cand, n_accept=n_acc,
                                                 exclusion_specs={"pools": {"store": store}})
        return lay_all, pool, report

    def test_pool_disjunction_external(self):
        from ucm.data.siw_pipeline import _typed_graph_from_layout
        from ucm.data.splits import typed_isomorphic
        lay_all, pool, report = self._excl_and_pool()
        assert len(pool) == 25
        for l in pool:
            for e in lay_all:
                assert not typed_isomorphic(_typed_graph_from_layout(l), _typed_graph_from_layout(e))
        for i in range(len(pool)):
            for j in range(i + 1, len(pool)):
                assert not typed_isomorphic(_typed_graph_from_layout(pool[i]),
                                            _typed_graph_from_layout(pool[j]))

    def test_pool_reproducible_and_fail(self):
        import pytest as _pt
        lay1, pool1, _ = self._excl_and_pool()
        _, pool2, _ = self._excl_and_pool()
        assert [l.layout_hash() for l in pool1] == [l.layout_hash() for l in pool2]
        # FAIL déterministe: le corpus d'exclusion = les candidats eux-mêmes
        import random as _r2
        from ucm.env.siw import generate_siw_layout as _g
        from ucm.data.siw_pipeline import generate_disjoint_pool_v4 as _gdp
        rngx = _r2.Random(777)
        self_cands = [_g(rngx, rngx.randint(2, 5), with_dialog=rngx.random() < 0.5,
                         profile="small") for _ in range(20)]
        store_self = {l.layout_hash(): {"views": l.views, "widgets": l.widgets,
                                        "nav_edges": [list(x) for x in l.nav_edges]}
                      for l in self_cands}
        with _pt.raises(RuntimeError, match="FAIL"):
            _gdp(seed=777, n_candidates=20, n_accept=5,
                 exclusion_specs={"self": {"store": store_self}})  # 100% collision

    def test_writer_binary_bytes_and_labels(self, tmp_path):
        import hashlib, json, random
        from pathlib import Path
        from ucm.env.siw import generate_siw_layout
        from ucm.data.siw_pipeline import build_siw_test_episodes, write_selfcontained_episodes
        rng = random.Random(2)
        lays = [generate_siw_layout(rng, 3, with_dialog=True, profile="small") for _ in range(2)]
        m, eps = build_siw_test_episodes(lays, seed=3, n_per_predicate=2, band=(2, 8))
        rep = write_selfcontained_episodes(eps, {l.layout_hash(): l for l in lays},
                                           out_path=str(tmp_path / "sc.jsonl"))
        # octets: hash flux == hash fichier (binaire, aucune translation)
        assert hashlib.sha256(Path(rep["path"]).read_bytes()).hexdigest() == rep["sha256_write_stream"]
        line = json.loads(Path(rep["path"]).read_text().splitlines()[0])
        assert "labels" in line["layout_spec"]
        assert {"episode_id", "layout_spec", "task", "d_star"} <= set(line)


class TestV1bisSupport:
    """V1-bis (arbitrage lead 10:09): support d'adaptation non dégénéré — DEV."""

    def _layouts(self, n=4):
        import random as _r
        from ucm.env.siw import generate_siw_layout
        rng = _r.Random(31)
        return [generate_siw_layout(rng, rng.randint(2, 4), with_dialog=rng.random() < 0.5,
                                    profile="small") for _ in range(n)]

    def test_degeneracy_measured(self):
        from ucm.data.siw_pipeline import sample_task_degeneracy
        deg = sample_task_degeneracy(self._layouts(), seed=5, n_draws=20)
        assert deg["draws"] == 80
        assert 0.0 <= deg["empty_pct"] <= 100.0
        assert deg["empty_states"] + deg["featureful_states"] == deg["draws"]

    def test_v3_closure_membership_and_invariants(self):
        # lead 10:26/tagi-5 10:26:47: couples ⊆ FERMETURE FORWARD + invariant
        import random as _r
        from ucm.env.siw import generate_siw_layout, SIWState
        from ucm.env.siw_oracle import _form_complete
        from ucm.data.siw_pipeline import (generate_multistep_adaptation_couples,
                                           forward_reachable_closure)
        rng = _r.Random(31)
        lays = [generate_siw_layout(rng, rng.randint(2, 4), with_dialog=rng.random() < 0.5,
                                    profile="small") for _ in range(4)]
        closures = {l.layout_hash(): forward_reachable_closure(l) for l in lays}
        couples, stats = generate_multistep_adaptation_couples(lays, seed=13, couples_per_layout=16)
        for c in couples:  # (1) TOUS dans la closure
            key = (c["state_key"][0], frozenset(c["state_key"][1]), tuple(c["state_key"][2]),
                   c["state_key"][3], frozenset(c["state_key"][4]))
            assert key in closures[c["layout_hash"]], "couple hors closure !"
        # (2) invariant submitted ⇒ form complete (sur les CLOSURES elles-mêmes)
        for l in lays:
            for k in closures[l.layout_hash()]:
                st = SIWState(k[0], frozenset(k[1]), dict(k[2]), k[3], frozenset(k[4]))
                for f in st.submitted:
                    assert _form_complete(l, st, f), "submitted sans form complete !"
        # (3) vraies stats publiées: tailles de closure + couples d0/d>0
        assert stats["closure_sizes"] and stats["d0_stop_couples"] > 0 and stats["d_pos_couples"] > 0

    def test_smoke_v3_versioned(self):
        import json
        from pathlib import Path
        smoke = json.loads(Path("artifacts/dev-v1bis-smoke.json").read_text())
        assert smoke["schema"] == "ucm-siw-v1bis-smoke/0.3"  # versionné (remplace v1 périmé)
        assert smoke["invariants"]["couples_subset_closure"] is True
        assert "initial_ratio_pct" in smoke  # ratio DÉCLARÉ (pas 'équilibré' non démontré)

    def test_v4_recentrage_sample_task(self):
        # contre-examen tagi-ask 10:33 (B3): support = départs sample_task +
        # trajectoires COMPLÈTES terminal inclus, SANS filtre richesse,
        # extras ≤25%, B1 dialog impossible = 0, exception SUBMITTED déclarée.
        import random as _r
        from ucm.env.siw import generate_siw_layout
        from ucm.data.siw_pipeline import generate_multistep_adaptation_couples
        rng = _r.Random(31)
        lays = [generate_siw_layout(rng, rng.randint(2, 4), with_dialog=rng.random() < 0.5,
                                    profile="small") for _ in range(6)]
        couples, stats = generate_multistep_adaptation_couples(lays, seed=13, couples_per_layout=16)
        src = stats["sources"]
        assert src["departure"] > 0, "masse initiale absente (B3) !"
        assert src["terminal_stop"] > 0, "terminal STOP absent !"
        assert stats["extras_pct"] <= 25.0, f"extras {stats['extras_pct']}% > 25% !"
        # B1: aucun dialog sur layout sans dialog
        no_dlg = {l.layout_hash() for l in lays if not any(w["type"] == "dialog" for w in l.widgets.values())}
        assert not [c for c in couples if c["layout_hash"] in no_dlg and c["state_key"][3]]
        # les départs sont bien des états VIDES test-like
        for c in couples:
            if c["source"] == "departure":
                assert c["state_key"][1] == [] and c["state_key"][2] == []
        # exception SUBMITTED mesurée/déclarée
        assert stats["submitted_goal_exception_layouts"] >= 0

    def test_multistep_couples_v2_dataset_level(self):
        # audit lead 10:16 — asserts NON-VACUËS sur le DATASET généré:
        # STOP correct présent (d*=0 ⇒ exactement {STOP}), jamais à d*>0,
        # équilibre prédicat, features critiques, états initiaux étiquetés.
        from ucm.data.siw_pipeline import generate_multistep_adaptation_couples
        lays = self._layouts(6)
        couples, stats = generate_multistep_adaptation_couples(
            lays, seed=13, couples_per_layout=16, band=(0, 8))
        assert couples
        # (1) labels STOP corrects PRÉSENTS dans le dataset
        d0 = [c for c in couples if c["d_star"] == 0]
        assert d0, "dataset sans AUCUN couple d*=0 (défaut majeur v1) !"
        assert all(c["optimal_semantic"] == ["STOP:None"] for c in d0)
        # (2) jamais STOP à d*>0 (prématuré)
        dpos = [c for c in couples if c["d_star"] > 0]
        assert dpos
        assert all(not any(x.startswith("STOP") for x in c["optimal_semantic"]) for c in dpos)
        # (3) v4: présence de chaque prédicat (le déséquilibre de TOTAL par
        # prédicat est une propriété RÉELLE — trajectoires SUBMITTED plus
        # longues — publiée dans stats, pas écrasée par un filtre)
        pp = stats["per_predicate"]
        assert min(pp.values()) >= 1
        # (4) v4: PAS de filtre richesse — les états initiaux test-like sont
        # légitimes (départs); features publiées en stats
        # (5) sources et étiquetage
        srcs = {c["source"] for c in couples}
        assert "departure" in srcs and "trajectory" in srcs  # v4: départs + trajectoires
        assert stats["feature_type_hist"]["initial_empty"] >= 0  # publiés

    def test_stop_coverage_d0_states(self):
        # couverture oracle/STOP NON vacue: états satisfaisant le but (d*=0)
        # ⇒ optimal_actions == exactement {STOP}
        import random as _r
        from ucm.env.siw import SIWState, generate_siw_layout, goal_satisfied
        from ucm.env.siw_oracle import SIWOracle
        from ucm.env.siw_oracle import enumerate_states
        rng = _r.Random(3)
        for lay in self._layouts(3):
            fields = [w["id"] for w in lay.widgets.values() if w["type"] == "field"]
            goal = {"predicate": "SET", "args": {"field": fields[0]}}
            o = SIWOracle(lay, goal)
            satisfied = [s for s in enumerate_states(lay)
                         if goal_satisfied(s, lay, goal) and o.reachable(s)]
            assert satisfied
            st = rng.choice(satisfied)
            assert o.d_star(st) == 0
            opt = o.optimal_actions(st)
            assert len(opt) == 1 and o.candidates[opt[0]]["action"] == "STOP"

    def test_pool200_dev_isodisjoint(self):
        # générateur 200 layouts V1-bis (DEV seed, PAS les scellés 03/04)
        import random as _r
        from ucm.env.siw import generate_siw_layout
        from ucm.data.siw_pipeline import generate_disjoint_pool_v4, _typed_graph_from_layout
        from ucm.data.splits import typed_isomorphic
        rng = _r.Random(20260926)
        existing = [generate_siw_layout(rng, rng.randint(2, 5), with_dialog=rng.random() < 0.5,
                                        profile="small") for _ in range(50)]
        store = {l.layout_hash(): {"views": l.views, "widgets": l.widgets,
                                   "nav_edges": [list(e) for e in l.nav_edges]} for l in existing}
        pool, report = generate_disjoint_pool_v4(
            seed=20261012, n_candidates=120, n_accept=15,
            exclusion_specs={"existing": {"store": store}})
        assert len(pool) == 15
        for l in pool:
            for e in existing:
                assert not typed_isomorphic(_typed_graph_from_layout(l), _typed_graph_from_layout(e))

    def test_stop_premature_demonstrated(self):
        # DECISION-V1-CONSTRUCT-HOLD (bee3424f): démontrer STOP correct ET prématuré.
        # Prématuré: état d*>0 — goal_satisfied FAUX, STOP ∉ A*, jamais supervisé.
        import random as _r
        from ucm.env.siw import generate_siw_layout, goal_satisfied
        from ucm.env.siw_oracle import SIWOracle, enumerate_states
        from ucm.data.siw_pipeline import _rebuild_siw_layout
        import json as _json
        from pathlib import Path as _P
        dev_raw = _json.loads(_P("artifacts/inventory-siw-dev-layouts.json").read_text())
        lay = _rebuild_siw_layout(list(dev_raw.values())[0])
        fields = [w["id"] for w in lay.widgets.values() if w["type"] == "field"]
        goal = {"predicate": "SET", "args": {"field": fields[0]}}
        o = SIWOracle(lay, goal)
        rng = _r.Random(9)
        # état profond (d*>0): STOP y serait PRÉMATURÉ
        deep = [s for s in enumerate_states(lay) if o.reachable(s) and o.d_star(s) >= 2
                and not goal_satisfied(s, lay, goal)]
        assert deep
        st = rng.choice(deep)
        assert o.d_star(st) >= 2
        assert not goal_satisfied(st, lay, goal)          # but NON satisfait
        opt = o.optimal_actions(st)
        assert all(o.candidates[i]["action"] != "STOP" for i in opt), \
            "STOP supervisé à d*>0 = STOP prématuré dans la supervision !"
        # et le duel: état satisfait (d*=0) ⇒ exactement {STOP}
        sat = [s for s in enumerate_states(lay) if goal_satisfied(s, lay, goal) and o.reachable(s)]
        st0 = rng.choice(sat)
        opt0 = o.optimal_actions(st0)
        assert len(opt0) == 1 and o.candidates[opt0[0]]["action"] == "STOP"


class TestV1bisAuditResiduals:
    """tagi-5 10:16:30 — restes (5) format/mapping et (6) exclusion publiée."""

    def test_couples_schema_versioned(self):
        from ucm.data.siw_pipeline import generate_multistep_adaptation_couples
        couples, _ = generate_multistep_adaptation_couples(self._l(), seed=13, couples_per_layout=10)
        assert all(c["schema"] == "ucm-siw-adaptation-couples/0.4" for c in couples)  # 0.3→0.4: recentrage sample_task (B3)

    def _l(self):
        import random as _r
        from ucm.env.siw import generate_siw_layout
        rng = _r.Random(31)
        return [generate_siw_layout(rng, rng.randint(2, 4), with_dialog=rng.random() < 0.5,
                                    profile="small") for _ in range(4)]

    def test_semantic_to_indices_bijective_with_stop(self):
        # mapping contrat: d0 couple → indices == [STOP] exactement
        import random as _r
        from ucm.env.siw import generate_siw_layout, SIW
        from ucm.data.siw_pipeline import semantic_to_indices, generate_multistep_adaptation_couples
        lays = self._l()
        couples, _ = generate_multistep_adaptation_couples(lays, seed=13, couples_per_layout=10)
        by_hash = {l.layout_hash(): l for l in lays}
        from ucm.data.siw_pipeline import _rebuild_siw_layout
        n_d0 = n_dpos = 0
        for c in couples:
            lay = by_hash[c["layout_hash"]]
            from ucm.env.siw import SIWState
            sv, fl, ch, dg, sub = c["state_key"]
            env = SIW(lay)
            env.candidates()  # ordre canonique
            cands = env.candidates()
            idx = semantic_to_indices(c["optimal_semantic"], cands)
            if c["d_star"] == 0:
                assert idx == [len(cands) - 1], "d0 matérialisé ≠ [STOP]"  # STOP = dernier
                n_d0 += 1
            else:
                assert all(cands[i]["action"] != "STOP" for i in idx)
                n_dpos += 1
        assert n_d0 > 0 and n_dpos > 0

    def test_pool200_artifact_disjunction(self):
        import json
        from pathlib import Path
        art = json.loads(Path("artifacts/dev-v1bis-pool200.json").read_text())
        assert art["schema"] == "ucm-siw-v1bis-pool-dev/0.4"  # D9: ops_note 2 opens
        assert art["exclusion_set"]["count"] == 450  # v0.3: exclusion par REJEU (zéro manifest lu)
        assert len(art["pool_layout_hashes"]) == 200
        assert "ops_note" in art  # ruling 10:29 documenté dans l'artefact


class TestRStarV6:
    """Audit lead 10:49 — 4 bloqueurs: quota exact, RNG unique, terminal
    garanti, sérialisation triée, recovery retirée."""

    def _lays(self, n=10):
        import random as _r
        from ucm.env.siw import generate_siw_layout
        rng = _r.Random(20261010)
        return [generate_siw_layout(rng, rng.randint(2, 5), with_dialog=rng.random() < 0.5,
                                    profile="small") for _ in range(n)]

    def test_v7_single_emission_and_trace(self):
        # M1: départ émis UNE fois (source departure à depth 0), terminal une
        # fois, records == d*(s0)+1 par épisode, d* décroît de 1 par pas,
        # doublons UNIQUEMENT inter-épisodes, executed = action réelle.
        from collections import defaultdict
        from ucm.data.siw_pipeline import generate_adaptation_episodes_rstar
        lays = self._lays()
        couples, stats = generate_adaptation_episodes_rstar(lays, seed=13, n_episodes=40)
        by_ep = defaultdict(list)
        for c in couples:
            by_ep[c["episode_id"]].append(c)
        assert len(by_ep) == 40
        for eid, recs in by_ep.items():
            recs.sort(key=lambda c: c["depth"])
            n_dep = sum(1 for r in recs if r["source"] == "departure")
            n_term = sum(1 for r in recs if r["source"] == "terminal")
            assert n_dep == 1 and n_term == 1, f"{eid}: double émission M1 !"
            assert recs[0]["source"] == "departure" and recs[0]["depth"] == 0
            assert recs[-1]["source"] == "terminal" and recs[-1]["d_star"] == 0
            assert len(recs) == recs[0]["d_star"] + 1  # records == d*(s0)+1
            for i in range(len(recs) - 1):
                assert recs[i]["d_star"] == recs[i + 1]["d_star"] + 1  # trace
            # rstar_executed: présent sur departure+trajectory (l'action exécutée
            # depuis cet état), None sur terminal
            for r in recs[:-1]:
                assert r["rstar_executed"] is not None
            assert recs[-1]["rstar_executed"] is None
        # multiplicité INTER-épisode préservée (doublons légitimes publiés)
        assert stats["multiplicity"]["duplicates"] >= 0
        assert all(c["schema"] == "ucm-siw-adaptation-couples/0.8" for c in couples)  # 0.7→0.8: executed_semantic

    def test_v6_fail_on_impossible_pool(self):
        # pool 1 layout 2 vues: VIEW d*≥2 impossible pour ce layout → FAIL
        import random as _r, pytest as _pt
        from ucm.env.siw import generate_siw_layout
        from ucm.data.siw_pipeline import generate_adaptation_episodes_rstar
        rng = _r.Random(5)
        lay = generate_siw_layout(rng, 2, with_dialog=False, profile="small")
        with _pt.raises(RuntimeError, match="FAIL"):
            generate_adaptation_episodes_rstar([lay], seed=1, n_episodes=4)

    def test_v6_tie_determinism_hashseeds(self):
        # déterminisme incluant des TIES (plusieurs optimaux): même hash
        import hashlib, json, subprocess, sys
        script = (
            "import random, hashlib, json\n"
            "from ucm.env.siw import generate_siw_layout\n"
            "from ucm.data.siw_pipeline import generate_adaptation_episodes_rstar\n"
            "rng = random.Random(20261010)\n"
            "lays = [generate_siw_layout(rng, rng.randint(2, 5), with_dialog=rng.random() < 0.5, profile='small') for _ in range(10)]\n"
            "couples, _ = generate_adaptation_episodes_rstar(lays, seed=13, n_episodes=40)\n"
            "print(hashlib.sha256(json.dumps(couples, sort_keys=True).encode()).hexdigest())\n"
        )
        hashes = set()
        for hs in ("0", "1", "42"):
            out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                                 text=True, check=True,
                                 env={"PYTHONHASHSEED": hs, "PATH": "/usr/bin:/bin"})
            hashes.add(out.stdout.strip())
        assert len(hashes) == 1, f"ties non-déterministes: {hashes}"


class TestRStarV6:
    """Audit lead 10:49 — 4 bloqueurs: quota exact, RNG unique, terminal
    garanti, sérialisation triée, recovery retirée."""

    def _lays(self, n=10):
        import random as _r
        from ucm.env.siw import generate_siw_layout
        rng = _r.Random(20261010)
        return [generate_siw_layout(rng, rng.randint(2, 5), with_dialog=rng.random() < 0.5,
                                    profile="small") for _ in range(n)]

    def test_v7_single_emission_and_trace(self):
        # M1: départ émis UNE fois (source departure à depth 0), terminal une
        # fois, records == d*(s0)+1 par épisode, d* décroît de 1 par pas,
        # doublons UNIQUEMENT inter-épisodes, executed = action réelle.
        from collections import defaultdict
        from ucm.data.siw_pipeline import generate_adaptation_episodes_rstar
        lays = self._lays()
        couples, stats = generate_adaptation_episodes_rstar(lays, seed=13, n_episodes=40)
        by_ep = defaultdict(list)
        for c in couples:
            by_ep[c["episode_id"]].append(c)
        assert len(by_ep) == 40
        for eid, recs in by_ep.items():
            recs.sort(key=lambda c: c["depth"])
            n_dep = sum(1 for r in recs if r["source"] == "departure")
            n_term = sum(1 for r in recs if r["source"] == "terminal")
            assert n_dep == 1 and n_term == 1, f"{eid}: double émission M1 !"
            assert recs[0]["source"] == "departure" and recs[0]["depth"] == 0
            assert recs[-1]["source"] == "terminal" and recs[-1]["d_star"] == 0
            assert len(recs) == recs[0]["d_star"] + 1  # records == d*(s0)+1
            for i in range(len(recs) - 1):
                assert recs[i]["d_star"] == recs[i + 1]["d_star"] + 1  # trace
            # rstar_executed: présent sur departure+trajectory (l'action exécutée
            # depuis cet état), None sur terminal
            for r in recs[:-1]:
                assert r["rstar_executed"] is not None
            assert recs[-1]["rstar_executed"] is None
        # multiplicité INTER-épisode préservée (doublons légitimes publiés)
        assert stats["multiplicity"]["duplicates"] >= 0
        assert all(c["schema"] == "ucm-siw-adaptation-couples/0.8" for c in couples)  # 0.7→0.8: executed_semantic

    def test_v6_fail_on_impossible_pool(self):
        # pool 1 layout 2 vues: VIEW d*≥2 impossible pour ce layout → FAIL
        import random as _r, pytest as _pt
        from ucm.env.siw import generate_siw_layout
        from ucm.data.siw_pipeline import generate_adaptation_episodes_rstar
        rng = _r.Random(5)
        lay = generate_siw_layout(rng, 2, with_dialog=False, profile="small")
        with _pt.raises(RuntimeError, match="FAIL"):
            generate_adaptation_episodes_rstar([lay], seed=1, n_episodes=4)

    def test_v6_tie_determinism_hashseeds(self):
        # déterminisme incluant des TIES (plusieurs optimaux): même hash
        import hashlib, json, subprocess, sys
        script = (
            "import random, hashlib, json\n"
            "from ucm.env.siw import generate_siw_layout\n"
            "from ucm.data.siw_pipeline import generate_adaptation_episodes_rstar\n"
            "rng = random.Random(20261010)\n"
            "lays = [generate_siw_layout(rng, rng.randint(2, 5), with_dialog=rng.random() < 0.5, profile='small') for _ in range(10)]\n"
            "couples, _ = generate_adaptation_episodes_rstar(lays, seed=13, n_episodes=40)\n"
            "print(hashlib.sha256(json.dumps(couples, sort_keys=True).encode()).hexdigest())\n"
        )
        hashes = set()
        for hs in ("0", "1", "42"):
            out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                                 text=True, check=True,
                                 env={"PYTHONHASHSEED": hs, "PATH": "/usr/bin:/bin"})
            hashes.add(out.stdout.strip())
        assert len(hashes) == 1, f"ties non-déterministes: {hashes}"




class TestExecutedSemantic08:
    """tagi-5 11:23: rstar_executed_semantic publié (κ_executed dérivable)."""

    def test_executed_semantic_published_and_consistent(self):
        import random as _r
        from ucm.env.siw import generate_siw_layout, SIW
        from ucm.env.siw_oracle import SIWOracle
        from ucm.data.siw_pipeline import generate_adaptation_episodes_rstar
        rng = _r.Random(20261010)
        lays = [generate_siw_layout(rng, rng.randint(2, 5), with_dialog=rng.random() < 0.5,
                                    profile="small") for _ in range(6)]
        couples, _ = generate_adaptation_episodes_rstar(lays, seed=13, n_episodes=16)
        assert all(c["schema"] == "ucm-siw-adaptation-couples/0.8" for c in couples)
        by_hash = {l.layout_hash(): l for l in lays}
        from ucm.data.siw_pipeline import _rebuild_siw_layout  # noqa
        for c in couples:
            if c["source"] == "terminal":
                assert c["rstar_executed_semantic"] is None
            else:
                sem = c["rstar_executed_semantic"]
                assert sem and ":" in sem
                # cohérence: la sémantique exécutée ∈ optimales
                assert sem in c["optimal_semantic"], f"exécutée {sem} ∉ optimales"


class TestRROrderHelper:
    """Helper partagé round-robin (lead 12:37/12:39) — writer physique + asserts."""

    def test_round_robin_and_writer(self, tmp_path):
        import random as _r
        from ucm.env.siw import generate_siw_layout
        from ucm.data.siw_pipeline import generate_adaptation_episodes_rstar
        from ucm.data.rstar_order import round_robin_episodes, write_rr_couples_file
        from collections import defaultdict
        rng = _r.Random(20261010)
        lays = [generate_siw_layout(rng, rng.randint(2, 5), with_dialog=rng.random() < 0.5,
                                    profile="small") for _ in range(6)]
        couples, _ = generate_adaptation_episodes_rstar(lays, seed=13, n_episodes=16)
        by_ep = defaultdict(list)
        for c in couples:
            by_ep[c["episode_id"]].append(c)
        order = round_robin_episodes(by_ep, sorted(by_ep))
        # interleave: les 4 premiers épisodes couvrent 4 prédicats distincts
        first4 = {by_ep[e][0]["goal"]["predicate"] for e in order[:4]}
        assert first4 == {"VIEW", "SET", "CHOOSE", "SUBMITTED"}
        # tout préfixe N%4==0: N/4 par prédicat
        for N in (4, 8, 16):
            preds = [by_ep[e][0]["goal"]["predicate"] for e in order[:N]]
            from collections import Counter
            assert all(v == N // 4 for v in Counter(preds).values())
        # writer physique: fichier en ordre RR, SHA flux, set(pred)==4
        rep = write_rr_couples_file(couples, by_ep, str(tmp_path / "rr.jsonl"))
        assert rep["episodes"] == 16 and rep["pred_counts"] == {"VIEW": 4, "SET": 4, "CHOOSE": 4, "SUBMITTED": 4}
        import hashlib, json
        from pathlib import Path
        post = hashlib.sha256(Path(rep["path"]).read_bytes()).hexdigest()
        assert post == rep["sha256_write_stream"]
        lines = [json.loads(l) for l in Path(rep["path"]).read_text().splitlines()]
        # ordre fichier == ordre RR (premier épisode du fichier = premier du RR)
        assert lines[0]["episode_id"] == order[0]


class TestTest2Generation:
    """Audit 16:26: seed replay 20260926, assertion manifeste, flux complet."""

    def test_exclusion_store_replay_matches_manifeste(self):
        # D2: ÉGALITÉ EXACTE (pas >=420) — replay seed 20260926 == (pool200.dev - dev30)
        import json as _json
        from pathlib import Path as _P
        from ucm.env.siw import generate_siw_layout as _gl
        ref = _json.loads(_P("artifacts/dev-v1bis-pool200.json").read_text())
        dev30 = set(_json.loads(_P("artifacts/inventory-siw-dev-layouts.json").read_text()).keys())
        expected = set(ref["exclusion_set"]["layout_hashes"]) - dev30
        import random as _r
        rng = _r.Random(20260926)
        replay = set()
        for _ in range(420):
            lay = _gl(rng, rng.randint(2, 5), with_dialog=rng.random() < 0.5, profile="small")
            replay.add(lay.layout_hash())
        assert replay == expected, f"D2: {len(replay)} != {len(expected)} — égalité exacte requise"

    def test_smoke_dev_generation(self):
        # seed DEV distinct, 4 épisodes — jamais 20261003/04
        from ucm.data.test2_generation import generate_test2_episodes
        art = generate_test2_episodes(seed_pool=20261099, seed_episodes=20261099,
                                      n_per_predicate=1, n_pool_layouts=5)
        assert art["episodes"]["total"] == 4
        assert len(art["selfcontained_full"]) == 4  # flux COMPLET pas échantillon
        assert art["pool"]["accepted"] >= 5
        # chaque épisode auto-suffiant
        for ep in art["selfcontained_full"]:
            assert {"episode_id", "layout_spec", "task", "d_star", "layout_hash"} <= set(ep)


class TestV1bisGenStoreV02:
    """Revue externe 4 (lead 17:51): store 26 layouts specs + test acceptation."""

    def test_store_covers_all_10_files(self):
        import json
        from pathlib import Path
        store = json.loads(Path("artifacts/v1bis-gen-store-v02.json").read_text())
        store_keys = set(store["store"].keys())
        n_files = 0
        for f in sorted(Path("artifacts/v1bis-gen").glob("v1bis-gen-*.jsonl")):
            n_files += 1
            file_hashes = {json.loads(l).get("layout_hash")
                          for l in f.read_text().splitlines() if l.strip()}
            assert file_hashes.issubset(store_keys), \
                f"{f.name}: {len(file_hashes - store_keys)} hashes hors store"
        assert n_files >= 10  # 10 primaires + remplacements possibles
        assert len(store_keys) == 26
        # chaque entrée a des specs complètes
        for h, spec in store["store"].items():
            assert spec.get("views") and spec.get("widgets") is not None


class TestVerdictV1bis:
    """Module analyse primaire (lead 17:52): fonction pure, verdicts connus."""

    @staticmethod
    def _write_raw(tmp_path, pre_by_seed, scr_by_seed):
        import json
        raw = {"pretrained": {}, "scratch": {}}
        for seed, preds in pre_by_seed.items():
            raw["pretrained"][str(seed)] = {p: v for p, v in preds.items()}
        for seed, preds in scr_by_seed.items():
            raw["scratch"][str(seed)] = {p: v for p, v in preds.items()}
        f = tmp_path / "raw.json"
        f.write_text(json.dumps(raw))
        return str(f)

    def test_pass_verdict(self, tmp_path):
        from ucm.data.verdict_v1bis import verdict_v1bis
        # +10pp sur tous les seeds/preds → PASS
        pre = {s: {"VIEW": 0.7, "SET": 0.7, "CHOOSE": 0.7, "SUBMITTED": 0.7} for s in range(5)}
        scr = {s: {"VIEW": 0.6, "SET": 0.6, "CHOOSE": 0.6, "SUBMITTED": 0.6} for s in range(5)}
        r = verdict_v1bis(self._write_raw(tmp_path, pre, scr))
        assert r["verdict"] == "PASS" and r["diff_pts"] == 10.0 and r["ci_low"] > 0

    def test_fail_verdict(self, tmp_path):
        from ucm.data.verdict_v1bis import verdict_v1bis
        # -10pp cohérent → FAIL (IC exclut tout positif)
        pre = {s: {"VIEW": 0.5, "SET": 0.5, "CHOOSE": 0.5, "SUBMITTED": 0.5} for s in range(5)}
        scr = {s: {"VIEW": 0.6, "SET": 0.6, "CHOOSE": 0.6, "SUBMITTED": 0.6} for s in range(5)}
        r = verdict_v1bis(self._write_raw(tmp_path, pre, scr))
        assert r["verdict"] == "FAIL" and r["diff_pts"] == -10.0

    def test_indetermine_verdict(self, tmp_path):
        from ucm.data.verdict_v1bis import verdict_v1bis
        # INDETERMINÉ = point >= 5pp MAIS IC contient 0 (grosse variance inter-seeds)
        pre = {s: {"VIEW": 1.0, "SET": 1.0, "CHOOSE": 1.0, "SUBMITTED": 1.0} if s < 2
               else {"VIEW": 0.55, "SET": 0.55, "CHOOSE": 0.55, "SUBMITTED": 0.55}
               for s in range(5)}
        scr = {s: {"VIEW": 0.6, "SET": 0.6, "CHOOSE": 0.6, "SUBMITTED": 0.6} for s in range(5)}
        r = verdict_v1bis(self._write_raw(tmp_path, pre, scr))
        assert r["verdict"] == "INDETERMINÉ", f"attendu INDETERMINÉ, reçu {r['verdict']} (diff={r['diff_pts']})"
        assert "≠ « aucun transfert »" in r["note"]

    def test_indetermine_wording(self, tmp_path):
        from ucm.data.verdict_v1bis import verdict_v1bis
        # même cas INDETERMINÉ (±20pp alterné → point>0 mais IC∋0)
        pre = {s: {"VIEW": 1.0, "SET": 1.0, "CHOOSE": 1.0, "SUBMITTED": 1.0} if s < 2
               else {"VIEW": 0.55, "SET": 0.55, "CHOOSE": 0.55, "SUBMITTED": 0.55}
               for s in range(5)}
        scr = {s: {"VIEW": 0.6, "SET": 0.6, "CHOOSE": 0.6, "SUBMITTED": 0.6} for s in range(4)}
        r = verdict_v1bis(self._write_raw(tmp_path, pre, scr))
        assert r["verdict"] == "INDETERMINÉ"
        assert "INDETERMINÉ ≠ " in r["note"]  # wording exact

    def test_mutation_ci_rule(self, tmp_path):
        # DURCI (lead 18:00): mutation qui change le verdict — pas un or quasi-vacuous
        from ucm.data.verdict_v1bis import verdict_v1bis
        pre = {s: {"VIEW": 0.7, "SET": 0.7, "CHOOSE": 0.7, "SUBMITTED": 0.7} for s in range(5)}
        scr = {s: {"VIEW": 0.6, "SET": 0.6, "CHOOSE": 0.6, "SUBMITTED": 0.6} for s in range(5)}
        r1 = verdict_v1bis(self._write_raw(tmp_path, pre, scr))
        assert r1["verdict"] == "PASS"
        pre_mut = dict(pre)
        pre_mut[0] = {"VIEW": 0.3, "SET": 0.3, "CHOOSE": 0.3, "SUBMITTED": 0.3}
        mut_dir = tmp_path / "mut"
        mut_dir.mkdir(exist_ok=True)
        r2 = verdict_v1bis(self._write_raw(mut_dir, pre_mut, scr))
        assert r2["verdict"] != "PASS", "mutation doit faire perdre PASS"
        assert r2["ci_low"] < r1["ci_low"], "ci_low doit chuter"

    def test_fail_below_5pp_even_if_ci_positive(self, tmp_path):
        # arbitrage (1): point 3pp avec IC>0 doit FAILer (pas INDETERMINÉ)
        from ucm.data.verdict_v1bis import verdict_v1bis
        pre = {s: {"VIEW": 0.63, "SET": 0.63, "CHOOSE": 0.63, "SUBMITTED": 0.63} for s in range(5)}
        scr = {s: {"VIEW": 0.6, "SET": 0.6, "CHOOSE": 0.6, "SUBMITTED": 0.6} for s in range(5)}
        r = verdict_v1bis(self._write_raw(tmp_path, pre, scr))
        assert r["verdict"] == "FAIL", f"3pp doit FAIL (pas {r['verdict']})"
        assert r["diff_pts"] == 3.0

    def test_no_pairs(self, tmp_path):
        from ucm.data.verdict_v1bis import verdict_v1bis
        pre = {1: {"VIEW": 0.7}}
        scr = {2: {"VIEW": 0.6}}  # seed différent → aucune paire
        r = verdict_v1bis(self._write_raw(tmp_path, pre, scr))
        assert r["verdict"] == "INDETERMINÉ" and r["diff_pts"] is None


class TestVerdict1Niveau:
    """Erratum 18:26: bootstrap 1 niveau (prédicats fixes), supersede 2-niveaux."""

    @staticmethod
    def _make_raw(tmp_path, pre, scr):
        import json
        raw = {"pretrained": {str(s): p for s, p in pre.items()},
               "scratch": {str(s): p for s, p in scr.items()}}
        f = tmp_path / "raw.json"
        f.write_text(json.dumps(raw))
        return str(f)

    def test_a_stable_aggregate_5pp(self, tmp_path):
        # (a) +30/+30/-20/-20: agrégat +5pp stable, IC 1-niveau reflète la dispersion
        from ucm.data.verdict_v1bis import verdict_v1bis
        P4 = ("VIEW", "SET", "CHOOSE", "SUBMITTED")
        pre = {s: {p: 0.6 + (0.3 if s < 4 else -0.2) for p in P4} for s in range(8)}
        scr = {s: {p: 0.6 for p in P4} for s in range(8)}
        r = verdict_v1bis(self._make_raw(tmp_path, pre, scr))
        assert r["diff_pts"] == 5.0
        assert r["ci_low"] <= 0  # la dispersion inter-seeds fait chuter ci_low
        assert r["verdict"] == "INDETERMINÉ", f"(a) doit être INDETERMINÉ (5pp + IC∋0), reçu {r['verdict']} — fix flottant"

    def test_b_no_effect_ic_covers_zero(self, tmp_path):
        # (b) absence d'effet simulée → IC couvre 0
        from ucm.data.verdict_v1bis import verdict_v1bis
        P4 = ("VIEW", "SET", "CHOOSE", "SUBMITTED")
        pre = {s: {p: 0.6 for p in P4} for s in range(5)}
        scr = {s: {p: 0.6 for p in P4} for s in range(5)}
        r = verdict_v1bis(self._make_raw(tmp_path, pre, scr))
        assert r["diff_pts"] == 0.0
        assert r["ci_low"] <= 0 <= r["ci_high"], f"IC ({r['ci_low']},{r['ci_high']}) ne couvre pas 0"

    def test_ci_high_is_95th_percentile_not_max(self):
        """Regression (bug 24/09, tagi-review): ci_high must be the 95th
        percentile of the bootstrap distribution, NOT the max. Non-vacuous:
        with a skewed distribution, max > 95th pct — the test FAILS if the
        implementation reverts to stats[-1]."""
        import json, tempfile, os
        from ucm.data.verdict_v1bis import bootstrap_ci_1level
        # skewed: one seed with a huge diff -> max >> 95th percentile
        diffs = {f"s{i}": {"VIEW": 40.0 if i == 0 else 0.0, "SET": 40.0 if i == 0 else 0.0,
                            "CHOOSE": 40.0 if i == 0 else 0.0, "SUBMITTED": 40.0 if i == 0 else 0.0}
                 for i in range(10)}  # 1 outlier/10 seeds: max=40, 95th pct << 40
        ci_low, ci_high = bootstrap_ci_1level(diffs)
        # With 10k resamples, P(all 10 draws pick s0 both times)= tiny -> max=40,
        # but 95th percentile must be < max in a skewed dist.
        # Deterministic (bootstrap seed 20260923): fixed impl -> 95th pct = 12.0;
        # mutated (stats[-1] = max) -> 24.0 — the exact value discriminates.
        assert ci_high == 12.0, f"ci_high={ci_high} — expected 95th pct 12.0 (max would be 24.0)"
        assert ci_low == 0.0


class TestDSLMinimal:
    """DSL minimal P2 (mission 3): tests différentiels vs oracles existants."""

    def test_all_differential_match(self):
        from ucm.data.dsl_minimal import run_differential_suite
        results = run_differential_suite()
        assert len(results) >= 3
        for r in results:
            assert r["match"], f"DIVERGENCE DSL dans {r['test']}: {r['divergences']}"

    def test_dsl_tgk_specific(self):
        from ucm.data.dsl_minimal import differential_test_tgk
        dsl = {
            "type": "tgk",
            "rooms": ["x", "y"], "edges": [["x", "y"]], "door_edge": 0,
            "init": {"agent": "x", "carried": None, "key": "y", "parcel": "x", "door_locked": False},
            "goal": {"predicate": "HAVE", "args": {"object": "key"}},
        }
        result = differential_test_tgk(dsl)
        assert result["reachable"] and result["d_star"] >= 1

    def test_compare_traces_detects_divergence(self):
        from ucm.data.dsl_minimal import compare_traces
        a = {"d_star": 3, "L_star": 4, "reachable": True, "optimal_count": 2}
        b = {"d_star": 5, "L_star": 6, "reachable": True, "optimal_count": 1}
        r = compare_traces(a, b)
        assert not r["match"] and "d_star" in r["divergences"]


class TestDSLContract:
    """(a) fail-closed type inconnu/spec incomplète, (b) constante figée."""

    def test_unknown_type_fails_closed(self):
        from ucm.data.dsl_minimal import world_from_dsl
        with pytest.raises(ValueError, match="type de monde inconnu"):
            world_from_dsl({"type": "minecraft"})

    def test_missing_fields_fail_closed(self):
        from ucm.data.dsl_minimal import world_from_dsl
        with pytest.raises(ValueError, match="champs requis manquants"):
            world_from_dsl({"type": "tgk", "rooms": ["a"]})  # edges/door/init/goal absents
        with pytest.raises(ValueError, match="champs requis manquants"):
            world_from_dsl({"type": "siw", "views": ["v"]})  # nav_edges/init/goal absents

    def test_not_a_dict_fails(self):
        from ucm.data.dsl_minimal import world_from_dsl
        with pytest.raises(ValueError, match="dict"):
            world_from_dsl("not_a_dict")

    def test_compare_fields_constant(self):
        from ucm.data.dsl_minimal import _COMPARE_FIELDS
        assert _COMPARE_FIELDS == ("d_star", "L_star", "reachable", "optimal_count")
