# PIR — Physics Intermediate Representation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23631107.svg)](https://doi.org/10.5281/zenodo.23631107)

This release: 10.5281/zenodo.23631107. All versions (always the latest): 10.5281/zenodo.23460310.

PIR is a torch-free symbolic regression engine for recovering closed-form
physical laws from tabular data. It fits interpretable expressions (power
laws, rational functions, exponentials, sums of such terms) with classical
least-squares methods, and exposes a scikit-learn style `PIRRegressor`.

PIR is research software (alpha). The core regressor and the fallback gates
are tested; other subpackages are experimental (see *Status* below).

## Install

```bash
pip install "git+https://github.com/Qazi-pk/physics-engine.git@v3.8.0"
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

model = PIRRegressor(gates="all", safeguard=True, separability=True).fit(X, y)
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

| Target noise | 3.5 engine | 3.6.0 (`gates="all"`) | 3.7.0 (+ `safeguard=True`) | 3.8.0 (+ `separability=True`) |
|---|---|---|---|---|
| 0 | 35 / 35 | 50 / 50 | 50 / 50 | **58 / 58** |
| 0.001 | 35 / 35 | 36 / 36 | 36 / 36 | 36 / 36 |
| 0.01 | 30 / 29 | 35 / 35 | 35 / 35 | 35 / 35 |
| 0.1 | 2 / 2 | 23 / 22 | 23 / 22 | 23 / 22 |

No dataset solved by an earlier version is lost at any noise level. The 3.6.0,
3.7.0 and 3.8.0 columns were run with the packaged code inside SRBench's
official `pir` method (pre-release commits 5e6e306 for 3.7.0 and 53d1d90 for
3.8.0, identical code apart from the version number). The 8 laws added by
separability in 3.8 are I.9.18, I.13.4, I.24.6, II.6.15a, II.11.3, II.11.27,
II.37.1 and III.10.19.

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

These are the author's own runs, not an official SRBench leaderboard result.

## Status

| Part | Status |
|---|---|
| `sklearn_adapter.PIRRegressor`, `discovery/` gates, separability, safeguard | tested (see `tests/`) |
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
