# Code snapshot

Frozen copy of the UCM source tree, taken from the working repository at commit `a47f7b7c36daa87a33fb3aa12506799d27024187` (2026-09-25 15:41). It is included for reading and audit; the paper does not depend on rebuilding it.

```
ucm/          library: env, model, dsl, eval, v1, p2, web, data
tests/        560 tests (pytest); most require only numpy, MLX-dependent tests are marked by import
scripts/      experiment launchers and profiling (dev_*, profile_*, launch_*)
configs/      v0a.yaml
pytest.ini
```

See `SNAPSHOT.json` for exact counts and hashes.

## Running the tests

```bash
python3.12 -m venv .venv
.venv/bin/pip install mlx numpy pytest      # MLX requires Apple silicon
.venv/bin/python -m pytest -q               # full suite
```

On non-Apple hardware, exclude MLX-dependent tests (model training/eval); the environment, DSL, data, and instrument tests are pure numpy and run anywhere.

## Provenance notes

- No secrets, no data beyond the toy worlds, no network access in the library (the single exception is `ucm/web/compiler_probe.py`, which fetches four public pages for the coverage probe).
- The snapshot intentionally excludes the working repo's `artifacts/`, `reports/`, `.goals/`, `.mesh/`, `.pi/` state and the `docs/` original French files (those are in `../archive/`).
- This is a snapshot, not a fork: development continues in the working repository.
