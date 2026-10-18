# Packaging the classical PIR engine for public install

Goal: make the **classical** `physics_engine` installable from a **public**
repo via `pip install git+<url>@<tag>`, so SRBench CI can pull it — without
shipping the JEPA/torch code.

## Why both a public repo AND this file are needed

- **Public repo** = SRBench CI (running on GitHub's machines) can *reach* the
  code. A private repo fails the clone with an auth error.
- **pyproject.toml** = pip can *install* the code once reached (tells pip the
  package name, deps, and which folders to ship).

Neither replaces the other. You need both.

## Steps

1. Put `pyproject.toml` at the **root** of the public engine repo (same level
   as the `physics_engine/` package directory).
2. Add a `README.md` and a `LICENSE` at the root. Set `license` in
   `pyproject.toml` to match.
3. Fill in `authors` and the `Homepage` URL in `pyproject.toml`.
4. Commit, then **tag an immutable ref**: `git tag v0.1.0 && git push --tags`.
   Use that tag (not a branch) in `install.sh`'s `PIR_REF`.

## Packaging sanity check — run before tagging (catches the torch trap)

From the engine repo root:

```bash
pip wheel . -w /tmp/dist --no-deps

# 1) confirm ONLY the classical package shipped (no discovery/, jepa_module/,
#    results*, feynman_data):
python -c "import zipfile,glob; \
print('\n'.join(zipfile.ZipFile(glob.glob('/tmp/dist/physics_engine-*.whl')[0]).namelist()))"

# 2) install into a CLEAN venv with no torch and confirm a torch-free import:
python -m venv /tmp/cleanenv
/tmp/cleanenv/bin/pip install /tmp/dist/physics_engine-*.whl
/tmp/cleanenv/bin/python -c "import torch"   # MUST fail: ModuleNotFoundError
/tmp/cleanenv/bin/python -c "from physics_engine.sklearn_adapter import PIRRegressor; print('OK')"
```

If step 2's import fails with a torch error, some module in `physics_engine`
does a **top-level** `import torch`. Fix it in the engine (move the import
inside the JEPA-gated code path / behind a try-except), do NOT add torch to
`dependencies` — that would drag the whole JEPA/torch stack into SRBench's env.

If `physics_engine/__init__.py` imports submodules that in turn import the bare
top-level `discovery/` package, those imports will break once `discovery/` is
excluded from the wheel. Make sure the classical import path
(`physics_engine.sklearn_adapter`) does not transitively import `discovery`.

## Then wire SRBench to it

In `algorithms/PIR/install.sh` set:

```bash
PIR_REPO_URL="https://github.com/USER/physics_engine"
PIR_REF="v0.1.0"   # the tag you pushed
```

`install.sh` already verifies the import after install, so a broken package
fails fast.

## Verified

This config was test-built into a wheel: it ships only
`physics_engine/{__init__,sklearn_adapter}.py`, excludes `discovery/`,
`jepa_module/`, `results*`, and `feynman_data/`, and imports cleanly in a
torch-free venv.
