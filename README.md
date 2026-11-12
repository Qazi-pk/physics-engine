# PIR — Physics Intermediate Representation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23926418.svg)](https://doi.org/10.5281/zenodo.23926418)

This release: 10.5281/zenodo.23926418. All versions (always the latest): 10.5281/zenodo.23460310.

PIR is a torch-free symbolic regression engine for recovering closed-form
physical laws from tabular data. It fits interpretable expressions (power
laws, rational functions, exponentials, sums of such terms) with classical
least-squares methods, and exposes a scikit-learn style `PIRRegressor`.

PIR is research software (alpha). The core regressor and the fallback gates
are tested; other subpackages are experimental (see *Status* below).

## Install

```bash
pip install "git+https://github.com/Qazi-pk/physics-engine.git@v3.10.0"
```

Requires Python 3.9+ (CI tests 3.10 and 3.12). Dependencies: numpy, scipy,
pandas, sympy, scikit-learn, pyyaml. No PyTorch.

## Quick start

```python
import numpy as np
from physics_engine.sklearn_adapter import PIRRegressor

rng = np.random.default_rng(0)
X = rng.uniform(1, 5, (400, 3))
y = X[:, 0] * X[:, 1] / (X[:, 1] + X[:, 2])

model = PIRRegressor(gates="all", safeguard=True, separability=True,
                     noise_tolerant=True).fit(X, y)
print(model.model())        # x_0*x_1/(x_1 + x_2)
print(model.final_source_)  # "rational": found by the rational gate
y_hat = model.predict(X)
```

Columns of a NumPy array are named `x_0, x_1, ...`; a pandas DataFrame keeps
its column names (names that are not valid Python identifiers, such as
`inv-nodes`, become `inv_nodes`).

## Fallback gates (new in 3.6)

`PIRRegressor(gates=None)` (the default) behaves exactly as in 3.5. With
`gates="all"`, or a tuple such as `("rational", "noisy")`, three fallback
stages run after the main engine, in this fixed order, stopping at the first
one that is accepted:

| Gate | Form | Accepted only if |
|---|---|---|
| `rational` | `P(x)/Q(x)`, low-degree polynomials | the model reproduces the data **exactly** (max relative error < 1e-6) and beats the engine |
| `exp` | `C * exp(sum of monomials)` | **exact**, as above |
| `noisy` | `C * prod x_j^(p/q)`, exponents snapped to simple fractions (q ≤ 4) | its training MSE is no worse than the engine's |

The exact gates never replace the engine's model with an approximate one. The
noisy gate snaps exponents only within 3 standard errors and rejects data that
a richer smooth model explains better (a lack-of-fit test on held-out rows).
All thresholds were fixed before benchmarking. `model.gate_info_` records what
each gate did.

## Separability (new in 3.8)

Many laws are products of simpler parts, for example
`F = G*m1*m2 / ((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)`. With
`separability=True` (off by default), when the engine's model is not exact,
PIR looks for a split of the inputs into groups A and B with
`y = g(A) * h(B)`, solves each part separately, and multiplies them back:

1. **Detection**: a quadratic fit of log|y| on the log-inputs. If y is such a
   product, every cross term between a variable in A and one in B vanishes;
   a split counts as multiplicative if those cross terms explain less than
   0.5% of the spread of log|y|. Deterministic, at most 10 inputs.
2. **Order**: variables that cannot be separated form blocks (e.g. x and y in
   `sqrt(x**2 + y**2)`); the split "single variables | blocks" is tried first.
3. **Budget**: at most 4 extraction attempts, each on a fixed 5,000-row
   subsample, so results do not depend on the machine's speed.
4. **Acceptance**: the product must reproduce all training rows exactly
   (max relative error < 1e-6) and beat the engine.

`model.final_source_` is then `"separability"`, and
`model.gate_info_["separability"]` records the splits tried. Stages run in the
order engine → separability → gates → safeguard. Separability adds laws only
on exact (noise-free) data; on noisy data it does not fire.

New in 3.9: targets that change sign (detected with a log-free derivative
test), differences such as `1/r2 - 1/r1`, and a late pass after the gates
that solves a factor as a root of a short sum of power terms (relativistic
laws such as `m*c**2/sqrt(1 - v**2/c**2)`) or with a small sin/cos library
(`F*r*sin(theta)`). The late pass runs only when nothing is exact after the
gates, so it never replaces an exact rational or exponential form;
`model.gate_info_["separability_late"]` records it. Rules R5–R8 are in
`physics_engine/discovery/separability/stage.py`.

## Noise-tolerant stage (new in 3.10)

The stages above accept a law only if it reproduces the training data
exactly, which noisy data never does. With `noise_tolerant=True` (off by
default), when no model so far is exact, PIR tries three families with
*validated* acceptance: separable products `C*M(A)*h(B)`, power laws times
`exp(...)`, and short rationals `P/Q`.

1. **Fit**: coefficients are fitted on 75% of (up to 20,000) training rows.
2. **Snap**: exponents and coefficient ratios are snapped to simple fractions,
   but only within 3 standard errors of the fitted values.
3. **Validate**: a form is accepted only if snapping costs less than 2% error
   on the held-out 25%, and it predicts those rows within 5% of a flexible
   reference model.
4. **Replace**: it replaces the current model only if it predicts the
   held-out rows better. An exact model is never replaced.

The stage needs at least 2,000 training rows. `model.final_source_` is then
`"noise_tolerant"`, and `model.gate_info_["noise_tolerant"]` records each
family's decision. Rules V1–V9 are in
`physics_engine/discovery/noise_tolerant/stage.py`.

Output forms are chosen so that SRBench's scorer can verify them without
assuming positive variables (SRBench issue #235): for example `k*x**2`, not
`sqrt(k**2*x**4)`. Each rewritten form is checked numerically against the
fitted model.

## Safeguard (new in 3.7)

PIR looks for physical laws. On data without one, its best formula can fit
worse than a straight line. With `safeguard=True` (off by default) the final
formula is compared with a ridge-regression linear model and with the mean:

1. A formula with training R² ≥ 0.999 is always kept.
2. Otherwise the linear model replaces it only if its 5-fold cross-validated
   R² beats the formula's (in-sample) R² by more than 0.05.
3. If both are below 0, the mean is returned.

`model.final_source_` is then `"linear_fallback"` or `"mean_fallback"`, which
means "no law found", and `model.gate_info_["safeguard"]` records the scores.
The rule was fixed before benchmarking.

## Benchmark results (self-run)

On the 119 Feynman datasets of [SRBench](https://github.com/cavalab/srbench)
at noise 0, two seeds, scored with SRBench's own `assess_symbolic_model.py`
(including the fix from cavalab/srbench PR #231):

Symbolic solutions out of 119 (seeds 23654 / 78901):

| Target noise | 3.5 engine | 3.6.0 (`gates="all"`) | 3.7.0 (+ `safeguard=True`) | 3.8.0 (+ `separability=True`) | 3.9.0 | 3.10.0 (+ `noise_tolerant=True`) |
|---|---|---|---|---|---|---|
| 0 | 35 / 35 | 50 / 50 | 50 / 50 | 58 / 58 | 74 / 74 | **75 / 75** |
| 0.001 | 35 / 35 | 36 / 36 | 36 / 36 | 36 / 36 | 36 / 36 | **73 / 74** |
| 0.01 | 30 / 29 | 35 / 35 | 35 / 35 | 35 / 35 | 35 / 35 | **73 / 74** |
| 0.1 | 2 / 2 | 23 / 22 | 23 / 22 | 23 / 22 | 23 / 22 | **55 / 55** |

No dataset solved by an earlier version is lost at any noise level. The 3.6.0
to 3.10.0 columns were run with the packaged code inside SRBench's official
`pir` method (pre-release commits 5e6e306 for 3.7.0, 53d1d90 for 3.8.0,
af639aa for 3.9.0 and 3ceeb6c for 3.10.0, identical code apart from the
version number and, for 3.10.0, the minimum-data rule V9, which never applies
to these datasets of 75,000 training rows). The 8 laws
added by separability in 3.8 are I.9.18, I.13.4, I.24.6, II.6.15a, II.11.3,
II.11.27, II.37.1 and III.10.19.

The 16 laws added in 3.9 are I.10.7, I.12.11, I.13.12, I.15.10, I.18.12,
I.18.14, I.48.2, II.2.42, II.6.11, II.13.23, II.13.34, II.15.4, II.15.5,
III.17.37, test_1 and test_5. 3.9 was developed on a fixed half of the 119
datasets (chosen by the parity of the SHA-256 hash of each name, before any
3.9 work); it gained 10 laws on that half and 6 on the other half, which was
not inspected during 3.9 development (3.6–3.8 were developed on all 119).

3.10 roughly doubles the laws recovered from noisy data and adds I.32.17 at
noise 0. It was developed on the same half; on seed 23654 it gained 23, 23
and 18 laws there at noise 0.001, 0.01 and 0.1, and 14, 15 and 14 on the half
not inspected during 3.10 work, about the same rate per law.

On SRBench's 120 black-box (real-world and synthetic regression) datasets, the
safeguard raises PIR's median test R² from −0.02 (3.6.0, seed 23654) to 0.38
(3.7.0, median over seeds 23654, 15795 and 860; single seeds 0.36–0.44), about
the level of plain linear regression (0.375 in SRBench's published results),
and removes all out-of-memory failures; on the shared seed, no dataset got
worse than with 3.6.0. On such data PIR mostly returns the linear fallback: it
is a law-finder, not a general regressor.

3.8.0 (separability on) gives the same black-box results as 3.7.0 on seed
23654: 120/120 results, median test R² 0.436 for both, no dataset better or
worse by more than 0.01. Separability did not change a single final model
there (black-box data has no exact product law); median fit time is unchanged
(15 s), the slowest fit took 433 s (264 s with 3.7.0).

3.9.0 gives the same black-box results as 3.8.0 on seed 23654: 120/120
results, median test R² 0.436, no dataset better or worse by more than 0.01;
neither separability pass changed a final model. Median fit time 13 s; the
slowest fit took 1,019 s (433 s with 3.8.0), because the late pass searches to
its count limit when nothing is exact.

The first black-box check of 3.10 failed its pre-fixed rule: on one dataset
of 100 rows (624_fri_c0_100_5) test R² fell from 0.449 to 0.217, because the
noise-tolerant stage judged on about 19 held-out rows. The minimum-data rule
(at least 2,000 training rows) was then added; with it, 3.10.0 gives the same
black-box results as 3.9.0 on seed 23654: 120/120 results, median test R²
0.436, no dataset better or worse by more than 0.01.

On the 14 Strogatz datasets of SRBench, which PIR had never been run on before
(300 training rows each, rules fixed before the run), 3.10.0 gives the same
results as 3.9.0, because the minimum-data rule keeps the noise-tolerant stage
off; it loses no law. A first run without that rule gained three laws with
noise (lv1, lv2, shearflow2) and lost none. Strogatz counts use SRBench's
original scorer.

These are the author's own runs, not an official SRBench leaderboard result.

## Status

| Part | Status |
|---|---|
| `sklearn_adapter.PIRRegressor`, `discovery/` gates, separability, noise-tolerant stage, safeguard | tested (see `tests/`) |
| `discovery/` symbolic and power-law search | used by the regressor |
| `scientist/`, `active_inference/`, `knowledge_graph/`, `api/`, `interface/`, `plugins/robotics/` | experimental, not covered by this repository's tests |

## Citing

If you use PIR, please cite it using the metadata in
[`CITATION.cff`](CITATION.cff) (GitHub shows a "Cite this repository" button).

## Contributing and support

Bug reports and questions: [GitHub issues](https://github.com/Qazi-pk/physics-engine/issues).
See [CONTRIBUTING.md](CONTRIBUTING.md). Changes are listed in
[CHANGELOG.md](CHANGELOG.md).

## Development note

Parts of this code and its tests were written with AI assistance (Claude,
Anthropic). The author reviewed the code, chose the methods and thresholds,
and verified every reported result.

## License

MIT — see [LICENSE](LICENSE).
