"""PIR 3.9 separability additions (rules R5-R8 in separability/stage.py)."""
import numpy as np

from physics_engine.sklearn_adapter import PIRRegressor


def _fit(f, ranges, n=4000, seed=3):
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.uniform(a, b, n) for a, b in ranges])
    y = f(X)
    m = PIRRegressor(gates="all", safeguard=True, separability=True).fit(X, y)
    err = float(np.max(np.abs(m.predict(X) - y)) / np.max(np.abs(y)))
    return m, err


def test_sign_changing_product_r5_r6():
    # y = x0*x1*(1/x3 - 1/x2) changes sign; needs R5 detection and R6 reciprocal atoms
    m, err = _fit(lambda X: X[:, 0] * X[:, 1] * (1 / X[:, 3] - 1 / X[:, 2]), [(1, 5)] * 4)
    assert m.final_source_ == "separability" and err < 1e-6, m.gate_info_.get("separability")


def test_relativistic_root_late_pass_r7():
    # y = x0*x1/sqrt(1 - x1**2/x2**2): 1/h**2 is a sum of two power atoms
    m, err = _fit(lambda X: X[:, 0] * X[:, 1] / np.sqrt(1 - X[:, 1] ** 2 / X[:, 2] ** 2),
                  [(1, 5), (1, 2), (3, 10)])
    assert m.final_source_ == "separability" and err < 1e-6, m.gate_info_.get("separability_late")
    s = str(m.model()).replace(" ", "")
    assert "x_1**2/x_2**2" in s and "sqrt(" in s and "x_0*x_1/" in s    # readable form (R7)


def test_trig_factor_late_pass_r7():
    m, err = _fit(lambda X: X[:, 0] * X[:, 1] * np.sin(X[:, 2]), [(1, 5), (1, 5), (1, 5)])
    assert m.final_source_ == "separability" and err < 1e-6
    assert "sin(x_2)" in str(m.model())


def test_late_pass_does_not_run_when_exact():
    m, _ = _fit(lambda X: X[:, 0] / (X[:, 1] + X[:, 1] * X[:, 2]), [(1, 5)] * 3)
    late = m.gate_info_.get("separability_late")
    assert late is None or late.get("decision") == "engine_exact"


def test_separability_still_off_by_default():
    assert PIRRegressor().separability is False
