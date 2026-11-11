# Changelog

## 3.10.0 — 2026-11-12

New opt-in **noise-tolerant stage**, `PIRRegressor(noise_tolerant=True)`
(default `False`). All earlier stages accept a law only if it reproduces the
training data exactly, which noisy data never does. Rules fixed before benchmarking (V1–V9 in
`physics_engine/discovery/noise_tolerant/stage.py`):

- **When** (V1): only if no model so far is exact; an exact model is never
  replaced. It runs after the late separability pass and before the safeguard.
- **Families** (V3): separable `C*M(A)*h(B)` (a monomial times a short sum of
  power or sin/cos terms, or its root, square or inverse); power law times
  `exp` of a short sum of monomials; and short rationals `P/Q` (up to 5 inputs).
- **Validated acceptance** (V2, V4–V6): coefficients fitted in y space on 75% of
  up to 20,000 training rows; exponents and coefficient ratios snapped to
  simple fractions within 3 standard errors; accepted only if snapping costs
  less than 2% held-out error and the form predicts the held-out 25% within 5%
  of a flexible reference model.
- **Replacement** (V7): the best accepted form replaces the current model only
  if it predicts the held-out rows better.
- **Minimum data** (V9, amendment): the stage needs at least 2,000 training
  rows. Added after the first black-box check failed on a 100-row dataset,
  where it judged on about 19 held-out rows.
- **Output forms** (V8): written so that SRBench's scorer can verify them
  without assuming positive variables (SRBench issue #235): exact roots as
  monomials, a dominant term pulled out of a root, common factors out of sums.
  Each rewritten form is checked numerically against the fitted model.
- Tests: `tests/test_noise_tolerant_v310.py`.

## 3.9.0 — 2026-11-03

Separability (still opt-in, `separability=True`) recovers more laws. Rules fixed
before benchmarking (R5–R8 in `physics_engine/discovery/separability/stage.py`):

- **Sign-changing targets** (R5): when y takes both signs, splits are detected
  with a log-free derivative test, y·∂²y/∂a∂b − (∂y/∂a)(∂y/∂b) = 0, instead of
  the log|y| cross-term test (new module `separability/signaware.py`).
- **Reciprocal terms** (R6): the additive extraction retries with a constant and
  1/x terms when the 3.8 terms are not exact (e.g. 1/r2 − 1/r1).
- **Late pass** (R7): if nothing is exact after the gates, separability runs again
  with two more solvers for a factor h: h^±2 as a short sum of power terms
  (relativistic roots such as 1/√(1 − v²/c²)), and a small sin/cos library.
  Exact acceptance is unchanged; it runs after the gates, so it never displaces
  an exact rational or exponential form.
- **Readable forms**: a dominant monomial is pulled out of a root and roots of
  positive factors are merged, only if the result re-verifies numerically
  (e.g. `c**2*m/sqrt(1 - v**2/c**2)`).
- **Fix** (R8): the final check evaluates the verified model string; before, a
  correct product could be rejected because its factors printed fewer digits
  than they hold.
- Tests: `tests/test_separability_v39.py`.

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
