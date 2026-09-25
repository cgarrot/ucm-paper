"""Prototype fusion scellée B3+B4 — critères d'acceptation tagi-1 21:20."""

import pytest

from ucm.data.interactions_merge import (
    GENESIS,
    entry_seal,
    merge_worker_logs,
    merge_worker_logs_bytes,
    verify_merge,
    verify_worker_chain,
)
from ucm.data.schema import SchemaError


def _chain(worker_id, specs):
    """Construit une chaîne valide: specs = [(arm, k, motive)]."""
    entries, prev = [], GENESIS
    for seq, (arm, k, motive) in enumerate(specs):
        e = {"worker_id": worker_id, "seq": seq, "arm": arm, "k": k,
             "motive": motive, "target": "SIW-dev", "t": f"00:00:{seq:02d}"}
        e["seal"] = entry_seal(e, prev)
        prev = e["seal"]
        entries.append(e)
    return entries


def _sample_logs():
    return {
        "w0": _chain("w0", [("A", 100, "finetune"), ("A", 500, "finetune"), ("B", 100, "eval")]),
        "w1": _chain("w1", [("B", 100, "finetune"), ("A", 100, "eval")]),
        "w2": _chain("w2", [("A", 100, "finetune"), ("B", 2000, "eval")]),
    }


class TestAcceptanceB:
    def test_b4_scheduling_permutations_same_bytes(self):
        logs = _sample_logs()
        base = merge_worker_logs_bytes(logs)
        # permutations d'insertion des workers et ordre interne préservé
        import itertools
        for perm in itertools.permutations(sorted(logs)):
            shuffled = {w: logs[w] for w in perm}
            assert merge_worker_logs_bytes(shuffled) == base

    def test_b4_merge_replay_same_sha(self):
        logs = _sample_logs()
        m1 = merge_worker_logs(logs)
        m2 = merge_worker_logs(logs)
        assert m1["manifest_sha256"] == m2["manifest_sha256"]
        assert merge_worker_logs_bytes(logs) == merge_worker_logs_bytes(logs)

    def test_b3_missing_worker_fails(self):
        logs = _sample_logs()
        merge = merge_worker_logs(logs)
        # worker silencé côté vérification (roster le déclare, log absent) → échec
        reduced = {w: es for w, es in logs.items() if w != "w1"}
        with pytest.raises(SchemaError, match="roster déclare .* log absent"):
            verify_merge(merge, reduced)
        # cas dual: fusion produite SANS w1, vérification avec les logs w1
        # → w1 ABSENT du roster scellé → échec (le trou dans la fusion est lié)
        merge_silenced = merge_worker_logs(reduced)
        with pytest.raises(SchemaError, match="ABSENT du roster"):
            verify_merge(merge_silenced, logs)
        # worker inconnu fourni à la vérification → échec
        with pytest.raises(SchemaError, match="ABSENT du roster"):
            verify_merge(merge, {**logs, "w9": []})

    def test_tamper_entry_fails(self):
        logs = _sample_logs()
        merge = merge_worker_logs(logs)
        tampered = {w: [dict(e) for e in es] for w, es in logs.items()}
        tampered["w0"][1]["motive"] = "XX-falsifié"
        with pytest.raises(SchemaError, match="seal ROMPU"):
            verify_merge(merge, tampered)

    def test_tamper_merged_sequence_fails(self):
        import copy
        logs = _sample_logs()
        merge = merge_worker_logs(logs)
        bad = copy.deepcopy(merge)
        bad["merged_sequence_sha256"] = "0" * 64
        from ucm.data.writer import seal_manifest
        bad.pop("manifest_sha256")
        bad = seal_manifest(bad)
        with pytest.raises(SchemaError, match="rejeu"):
            verify_merge(bad, logs)

    def test_tamper_roster_count_fails(self):
        import copy
        logs = _sample_logs()
        merge = merge_worker_logs(logs)
        bad = copy.deepcopy(merge)
        bad["roster"][0]["count"] += 1
        bad.pop("manifest_sha256")
        from ucm.data.writer import seal_manifest
        bad = seal_manifest(bad)
        with pytest.raises(SchemaError):
            verify_merge(bad, logs)

    def test_b2_order_content_not_clock(self):
        # mêmes contenus, timestamps différents → même fusion
        logs1 = _sample_logs()
        logs2 = {w: [dict(e) for e in es] for w, es in logs1.items()}
        for es in logs2.values():
            for e in es:
                e["t"] = "99:99:99"
        # reseal (le payload v1 inclut t — ici v2 le garde comme métadonnée;
        # l'ORDRE ne dépend que du contenu)
        for w, es in logs2.items():
            prev = GENESIS
            for e in es:
                e["seal"] = entry_seal(e, prev)
                prev = e["seal"]
        m1, m2 = merge_worker_logs(logs1), merge_worker_logs(logs2)
        # l'ordre fusionné (clés) est identique ; les hashs d'entrées diffèrent
        # (t scellé dans la chaîne — métadonnée, pas décision d'ordre)
        assert m1["merged_count"] == m2["merged_count"]
        assert m1["merged_sequence_sha256"] != m2["merged_sequence_sha256"]

    def test_chain_verify_seq_hole_fails(self):
        entries = _chain("w0", [("A", 1, "m"), ("A", 2, "m"), ("A", 3, "m")])
        del entries[1]  # trou de seq
        with pytest.raises(SchemaError, match="seq"):
            verify_worker_chain(entries, "w0")


class TestContractV5:
    """N1/N2 (lead 16:11): fail-closed champs requis + contrat ordre documenté."""

    def test_missing_field_fails_closed(self):
        from ucm.data.interactions_merge import verify_worker_chain
        from ucm.data.schema import SchemaError
        # entrée sans 'arm' — doit FAIL, pas default silencieux à ""
        entries = [{"worker_id": "w0", "seq": 0, "k": 100, "motive": "m",
                    "target": "SIW", "t": "00:00:00", "seal": "x"}]
        with pytest.raises(SchemaError, match="champs requis manquants.*arm"):
            verify_worker_chain(entries, "w0")

    def test_perp_intra_worker_detected(self):
        # N2: permutation intra-worker casse la chaîne → échec de séquence
        from ucm.data.interactions_merge import verify_worker_chain
        from ucm.data.schema import SchemaError
        entries = _chain("w0", [("A", 100, "m1"), ("A", 200, "m2"), ("A", 300, "m3")])
        permuted = [entries[1], entries[0], entries[2]]  # permutation
        with pytest.raises(SchemaError):
            verify_worker_chain(permuted, "w0")
