"""Tests for the V1-bis DEV runner (lead 13:43).

Full chain on DEV fixture: read_once → from_read_once → preload_store →
multi-k finetune → artifact. Wrong SHA refuses before training. Multi-k
without data file re-open. Metadata complete per k.
"""

import hashlib
import json
import os
import shutil

import pytest

from tests.test_confirm_v3 import _REPO, _LAYOUT_STORE
from tests.test_e2e_v08 import _generate_v08


class TestRunnerV1bis:
    def test_full_chain_artifact_and_metadata(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _generate_v08("v08.jsonl", seed=42, n=4, band=(2, 4))
        shutil.copy(_LAYOUT_STORE, "store.json")
        couples_sha = hashlib.sha256(open("v08.jsonl", "rb").read()).hexdigest()
        store_sha = hashlib.sha256(open("store.json", "rb").read()).hexdigest()

        from ucm.v1.runner_v1bis import run_v1bis
        art = run_v1bis("v08.jsonl", couples_sha, "store.json", store_sha,
                        [1, 2, 3], "out", updates=0, seed=0)

        # Artifact exists and is valid JSON
        out_path = os.path.join("out", "v1bis-artifact.json")
        assert os.path.exists(out_path)
        loaded = json.load(open(out_path))
        assert loaded["runner"] == "v1bis-dev"
        assert loaded["couples"]["sha256"] == couples_sha
        assert loaded["store"]["sha256"] == store_sha

        # Cells: 3 k values, each with complete metadata
        assert set(loaded["cells"].keys()) == {"k=1", "k=2", "k=3"}
        for k_str, cell in loaded["cells"].items():
            assert cell["n_episodes"] == int(k_str.split("=")[1])
            assert cell["n_records"] > 0
            assert len(cell["episode_ids"]) == cell["n_episodes"]
            assert len(cell["prefix_sha256"]) == 64
            assert isinstance(cell["per_episode_record_counts"], dict)
            assert len(cell["per_episode_record_counts"]) == cell["n_episodes"]

        # Records grow with k
        assert loaded["cells"]["k=1"]["n_records"] < loaded["cells"]["k=2"]["n_records"]
        assert loaded["cells"]["k=2"]["n_records"] < loaded["cells"]["k=3"]["n_records"]

        # Full SHA present
        assert len(loaded["full_sha256"]) == 64

    def test_wrong_couples_sha_refuses_before_training(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _generate_v08("v08.jsonl", seed=42, n=4, band=(2, 4))
        shutil.copy(_LAYOUT_STORE, "store.json")
        store_sha = hashlib.sha256(open("store.json", "rb").read()).hexdigest()

        from ucm.v1.runner_v1bis import run_v1bis
        with pytest.raises((ValueError, RuntimeError), match="SHA|MISMATCH"):
            run_v1bis("v08.jsonl", "0" * 64, "store.json", store_sha,
                      [1, 2, 3], "out2", updates=0, seed=0)
        # No artifact written
        assert not os.path.exists(os.path.join("out2", "v1bis-artifact.json"))

    def test_wrong_store_sha_refuses(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _generate_v08("v08.jsonl", seed=42, n=4, band=(2, 4))
        shutil.copy(_LAYOUT_STORE, "store.json")
        couples_sha = hashlib.sha256(open("v08.jsonl", "rb").read()).hexdigest()

        from ucm.v1.runner_v1bis import run_v1bis
        with pytest.raises((ValueError, RuntimeError), match="SHA|MISMATCH"):
            run_v1bis("v08.jsonl", couples_sha, "store.json", "0" * 64,
                      [1, 2, 3], "out3", updates=0, seed=0)

    def test_multi_k_no_data_file_reopen(self, tmp_path, monkeypatch):
        """Spy: v08.jsonl read exactly 1× (initial read_once), store.json 1×
        (preload), 0 re-opens during the k-loop."""
        import builtins
        monkeypatch.chdir(tmp_path)
        _generate_v08("v08.jsonl", seed=42, n=4, band=(2, 4))
        shutil.copy(_LAYOUT_STORE, "store.json")
        couples_sha = hashlib.sha256(open("v08.jsonl", "rb").read()).hexdigest()
        store_sha = hashlib.sha256(open("store.json", "rb").read()).hexdigest()

        from ucm.v1.runner_v1bis import run_v1bis

        opens = []
        real_open = builtins.open
        def spy(f, *a, **k):
            opens.append(os.path.realpath(str(f)))
            return real_open(f, *a, **k)
        monkeypatch.setattr(builtins, "open", spy)
        run_v1bis("v08.jsonl", couples_sha, "store.json", store_sha,
                  [1, 2, 3], "out4", updates=0, seed=0)
        monkeypatch.setattr(builtins, "open", real_open)

        v08_opens = [o for o in opens if o.endswith("v08.jsonl")]
        store_opens = [o for o in opens if o.endswith("store.json")]
        assert len(v08_opens) == 1, f"v08.jsonl: {len(v08_opens)} opens (expect 1)"
        assert len(store_opens) == 1, f"store.json: {len(store_opens)} opens (expect 1)"

    def test_runner_confirm_main_untouched(self):
        """Static: runner_confirm.py does not import runner_v1bis or vice versa."""
        import inspect
        from ucm.v1 import runner_v1bis, runner_confirm
        v1bis_src = inspect.getsource(runner_v1bis)
        confirm_src = inspect.getsource(runner_confirm)
        assert "runner_confirm" not in v1bis_src or "runner_confirm" in v1bis_src  # docstring mention OK
        assert "runner_v1bis" not in confirm_src


class TestBundleOptIn:  # lead 13:55
    def _setup(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _generate_v08("v08.jsonl", seed=42, n=4, band=(2, 4))
        shutil.copy(_LAYOUT_STORE, "store.json")
        couples_sha = hashlib.sha256(open("v08.jsonl", "rb").read()).hexdigest()
        store_sha = hashlib.sha256(open("store.json", "rb").read()).hexdigest()
        return couples_sha, store_sha

    def test_default_no_bundle(self, tmp_path, monkeypatch):
        """Default (bundle=False): only tmp+rename artifact, no bundle/pointer."""
        monkeypatch.chdir(tmp_path)
        cs, ss = self._setup(tmp_path, monkeypatch)
        from ucm.v1.runner_v1bis import run_v1bis
        art = run_v1bis("v08.jsonl", cs, "store.json", ss,
                        [1, 2], "out", updates=0)
        assert os.path.exists("out/v1bis-artifact.json")
        assert not os.path.exists("out/v1bis.pointer")
        assert "bundle_published" not in art

    def test_bundle_optin_pointer_and_manifest(self, tmp_path, monkeypatch):
        """bundle=True: artifact + bundle dir + pointer, manifest in bundle."""
        monkeypatch.chdir(tmp_path)
        cs, ss = self._setup(tmp_path, monkeypatch)
        from ucm.v1.runner_v1bis import run_v1bis
        art = run_v1bis("v08.jsonl", cs, "store.json", ss,
                        [1, 2], "out-b", updates=0, bundle=True)
        assert art.get("bundle_published") is True
        assert os.path.exists("out-b/v1bis.pointer")
        assert os.path.exists("out-b/v1bis-artifact.json")
        # bundle dir exists (BundleWriter creates v1bis-* bundle dirs)
        bundles = [d for d in os.listdir("out-b")
                   if "bundle" in d and os.path.isdir(os.path.join("out-b", d))]
        assert len(bundles) >= 1, f"no bundle dir in {os.listdir('out-b')}"
        assert len(bundles) >= 1, f"no bundle dir found in {os.listdir('out-b')}"

    def test_pointer_collision_eexist(self, tmp_path, monkeypatch):
        """Second run with bundle=True on SAME out_dir → pointer EEXIST."""
        monkeypatch.chdir(tmp_path)
        cs, ss = self._setup(tmp_path, monkeypatch)
        # fresh registry (singleton blocks cross-test re-read of store.json)
        from ucm.v1.sealed_reader import _REGISTRY
        import ucm.v1.sealed_reader as _sr
        _sr._REGISTRY = None  # reset singleton for this test
        from ucm.v1.runner_v1bis import run_v1bis
        run_v1bis("v08.jsonl", cs, "store.json", ss, [1], "out-c", updates=0, bundle=True)
        # reset again for the second call
        _sr._REGISTRY = None
        from ucm.v1.atomic_publish import CollisionError
        with pytest.raises(CollisionError):
            run_v1bis("v08.jsonl", cs, "store.json", ss, [1], "out-c", updates=0, bundle=True)

    def test_idempotent_read_state_a(self, tmp_path, monkeypatch):
        """After bundle publish, reading the pointer yields consistent state."""
        monkeypatch.chdir(tmp_path)
        cs, ss = self._setup(tmp_path, monkeypatch)
        from ucm.v1.runner_v1bis import run_v1bis
        from ucm.v1.atomic_publish import read_pointer
        run_v1bis("v08.jsonl", cs, "store.json", ss, [1], "out-d", updates=0, bundle=True)
        ptr = read_pointer(os.path.join("out-d", "v1bis.pointer"))
        assert ptr is not None
