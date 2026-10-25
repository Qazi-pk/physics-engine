# Changelog

## 3.7.0 — 2026-10-25

- `PIRRegressor(safeguard=False)`: optional safeguard after the engine and the
  gates. If the formula's training R² is below 0.999 and a ridge-regression
  linear model is better by more than 0.05 (5-fold cross-validated, so scored
  honestly against an in-sample formula), the linear model is returned; if both
  are worse than predicting the mean, the mean is returned. Rule fixed before
  benchmarking; off by default, so 3.6 behaviour is unchanged.
- `exp` gate: skipped when its candidate pool would exceed 30,000 terms (about
  10 or more inputs). Previously it could exhaust memory on wide data. Every
  Feynman dataset (at most 9 inputs, 24,836 terms) is unaffected.
- Column names that are not valid Python identifiers (e.g. `inv-nodes`) get a
  safe name (`inv_nodes`) in the model string, so it can always be parsed.
- The engine no longer writes `debug_franka.log` into the working directory;
  set `PIR_DEBUG_LOG=<file>` to get that diagnostic log.
- Tests: `tests/test_v37.py`.

## 3.6.0 — 2026-10-19

- `PIRRegressor(gates=...)`: optional fallback gates after the main engine —
  exact rational (`P/Q`), exact exponential, and noise-tolerant power-law
  (exponents snapped to simple fractions). Off by default; with `gates=None`
  behaviour is unchanged from 3.5.
- New modules `physics_engine/discovery/rational_gate.py`, `exp_gate.py`,
  `noisy_gate.py` (numpy, sympy and the standard library only).
- Tests (`tests/test_gates.py`) and GitHub Actions CI.
- README rewritten for users; previous packaging notes moved to
  `docs/PACKAGING.md`. Added `CITATION.cff`, `CHANGELOG.md`, `CONTRIBUTING.md`.
- Package metadata: author, project URLs now point to this repository.

## 3.5.0

- Core dependencies (sympy, scikit-learn) made required; `.so` package data;
  version used by the SRBench `pir` method.
