"""Tests for the fallback gates (new in 3.6) and their wiring into PIRRegressor."""
import numpy as np
import pytest

from physics_engine.discovery.exp_gate import exp_gate
from physics_engine.discovery.noisy_gate import noisy_power_gate
from physics_engine.discovery.rational_gate import rational_gate
from physics_engine.sklearn_adapter import PIRRegressor

NAMES = ["x_0", "x_1", "x_2"]


@pytest.fixture(scope="module")
def X():
    return np.random.default_rng(0).uniform(1, 5, (400, 3))


# ---------------------------------------------------------------- gates alone
def test_rational_gate_exact(X):
    x0, x1, x2 = X.T
    res = rational_gate(X, x0 * x1 / (x1 + x2), NAMES)
    assert res is not None and res["exact"]


def test_exp_gate_exact(X):
    x0, x1, x2 = X.T
    res = exp_gate(X, x0 * np.exp(-x1 / x2), NAMES)
    assert res is not None and res["exact"]


def test_noisy_gate_snaps_cube_root(X):
    x0, x1, _ = X.T
    y = x0 * x1 ** (1 / 3)
    y = y + np.random.default_rng(1).normal(0, 0.01 * np.sqrt(np.mean(y ** 2)), len(y))
    res = noisy_power_gate(X, y, NAMES)
    assert res is not None and res["snapped"]
    assert res["exponents"] == {"x_0": "1", "x_1": "1/3"}


@pytest.mark.parametrize("make_y", [lambda x0, x1, x2: x0 + x1,
                                    lambda x0, x1, x2: np.exp(-x0),
                                    lambda x0, x1, x2: np.sin(x0) * x1])
def test_noisy_gate_rejects_non_monomials(X, make_y):
    assert noisy_power_gate(X, make_y(*X.T), NAMES) is None


# ------------------------------------------------------------ regressor wiring
def test_gates_off_by_default():
    assert PIRRegressor().gates is None


def test_unknown_gate_name_raises(X):
    with pytest.raises(ValueError):
        PIRRegressor(gates="bogus").fit(X, X[:, 0])


def test_regressor_adopts_rational_gate(X):
    x0, x1, x2 = X.T
    y = x0 * x1 / (x1 + x2)
    m = PIRRegressor(gates="all").fit(X, y)
    assert m.final_source_ == "rational"
    assert np.max(np.abs(m.predict(X) - y)) < 1e-6 * np.max(np.abs(y))
    assert m.gate_info_["decision"] == "rational_accepted"


def test_regressor_keeps_engine_when_no_gate_fits(X):
    x0, x1, _ = X.T
    y = np.sin(x0) + x1
    m = PIRRegressor(gates="all").fit(X, y)
    assert m.final_source_ == "engine"
    assert m.gate_info_["decision"] == "engine_kept"
