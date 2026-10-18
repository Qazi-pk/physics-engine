# PIR — Physics Intermediate Representation

PIR is a torch-free symbolic regression engine for recovering closed-form
physical laws from tabular data. It fits interpretable expressions (power
laws, rational functions, exponentials, sums of such terms) with classical
least-squares methods, and exposes a scikit-learn style `PIRRegressor`.

PIR is research software (alpha). The core regressor and the fallback gates
are tested; other subpackages are experimental (see *Status* below).

## Install

```bash
pip install "git+https://github.com/Qazi-pk/physics-engine.git@v3.6.0"
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

model = PIRRegressor(gates="all").fit(X, y)
print(model.model())        # x_0*x_1/(x_1 + x_2)
print(model.final_source_)  # "rational": found by the rational gate
y_hat = model.predict(X)
```

Columns of a NumPy array are named `x_0, x_1, ...`; a pandas DataFrame keeps
its column names.

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

## Benchmark results (self-run)

On the 119 Feynman datasets of [SRBench](https://github.com/cavalab/srbench)
at noise 0, two seeds, scored with SRBench's own `assess_symbolic_model.py`
(including the fix from cavalab/srbench PR #231):

| Configuration | Solved / 119 |
|---|---|
| PIR 3.5 engine | 35 (seed 23654) |
| + rational, exp and noisy gates | 50 (both seeds) |

At noise levels 0.001, 0.01 and 0.1 the gates never lost a dataset that the
engine solved. These are the author's own runs, not an official SRBench
leaderboard result. They were produced with SRBench method wrappers that use
the same gate code as this release; a re-run with the packaged 3.6 release is
planned.

## Status

| Part | Status |
|---|---|
| `sklearn_adapter.PIRRegressor`, `discovery/` gates | tested (see `tests/`) |
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
