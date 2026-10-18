# Changelog

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
