# Changelog

## 3.8.0 — 2026-10-30

- `PIRRegressor(separability=False)`: optional separability stage. If the
  engine's model is not exact, PIR tests whether `y = g(A) * h(B)` for a split
  of the inputs into groups A and B, solves g and h separately (recursively),
  and accepts the product only if it reproduces the training data **exactly**
  (max relative error < 1e-6) and beats the engine. Off by default, so 3.7
  behaviour is unchanged. Rules fixed before benchmarking:
  - detection by a deterministic cross-term test (quadratic fit of log|y| on
    the centred log-inputs; a split is multiplicative if its cross terms
    account for less than 0.5% of the spread of log|y|); at most 10 inputs;
  - inseparable variables are grouped into blocks, and the split "single
    variables | multi-variable blocks" is tried first;
  - at most 4 extraction attempts (a count, not a clock, so results do not
    depend on machine speed), each on a fixed 5,000-row subsample; the exact
    acceptance check uses all rows.
- New subpackage `physics_engine/discovery/separability/`.
- Order of stages: engine → separability → gates → safeguard.
- Tests: `tests/test_separability_crossterm.py`.

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
