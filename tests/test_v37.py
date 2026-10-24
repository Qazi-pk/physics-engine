"""Tests for the 3.7 changes: exp-gate pool cap, safe column names, safeguard."""
import time

import numpy as np
import pandas as pd
import pytest
import sympy as sp

from physics_engine.discovery.exp_gate import MAX_POOL_TERMS, exp_gate, pool_size_upper_bound
from physics_engine.sklearn_adapter import PIRRegressor, _safe_names


# ------------------------------------------------------------- A: pool cap
def test_pool_size_matches_plan():
    assert pool_size_upper_bound(9) == 24_836          # largest Feynman case
    assert pool_size_upper_bound(9) <= MAX_POOL_TERMS  # so Feynman is unaffected
    assert pool_size_upper_bound(10) > MAX_POOL_TERMS


def test_exp_gate_skips_wide_data_quickly():
    rng = np.random.default_rng(0)
    X = rng.uniform(1, 5, (60, 15))
    t0 = time.time()
    assert exp_gate(X, np.exp(X[:, 0]), [f"x_{i}" for i in range(15)]) is None
    assert time.time() - t0 < 1.0


def test_exp_gate_still_works_at_nine_inputs():
    rng = np.random.default_rng(1)
    X = rng.uniform(1, 2, (300, 9))
    res = exp_gate(X, X[:, 0] * np.exp(-X[:, 1] / X[:, 2]), [f"x_{i}" for i in range(9)])
    assert res is not None and res["exact"]


# --------------------------------------------------------- B: safe names
def test_safe_names_only_change_invalid_ones():
    assert _safe_names(["age", "inv-nodes", "deg-malig", "2x", "lambda", "a b", "a_b"]) == \
        ["age", "inv_nodes", "deg_malig", "x_2x", "x_lambda", "a_b", "a_b_1"]


def test_hyphenated_columns_give_parseable_model():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"inv-nodes": rng.uniform(1, 5, 200), "deg-malig": rng.uniform(1, 5, 200),
                       "age": rng.uniform(1, 5, 200)})
    y = df["inv-nodes"].values * df["age"].values
    m = PIRRegressor(gates="all", safeguard=True).fit(df, y)
    expr = sp.sympify(m.model())          # must not raise
    assert "inv-nodes" not in m.model()
    assert {str(s) for s in expr.free_symbols} <= {"inv_nodes", "deg_malig", "age"}
    assert np.allclose(m.predict(df), y, rtol=1e-4)


# --------------------------------------------------------- C: safeguard
def _r2(y, p):
    return 1 - np.mean((y - p) ** 2) / np.var(y)


def test_safeguard_off_by_default():
    assert PIRRegressor().safeguard is False


def test_safeguard_keeps_exact_law():
    rng = np.random.default_rng(3)
    X = rng.uniform(1, 5, (400, 3))
    y = X[:, 0] * X[:, 1] / X[:, 2]
    m = PIRRegressor(gates="all", safeguard=True).fit(X, y)
    assert m.gate_info_["safeguard"]["decision"] == "kept_near_exact"
    assert m.final_source_ not in ("linear_fallback", "mean_fallback")


def test_safeguard_falls_back_to_linear():
    rng = np.random.default_rng(4)
    X = rng.uniform(-1, 1, (400, 4))
    y = 3 + 2 * X[:, 0] - X[:, 1] + 0.5 * X[:, 2] + rng.normal(0, 0.1, 400)
    m = PIRRegressor(gates="all", safeguard=True).fit(X, y)
    info = m.gate_info_["safeguard"]
    assert m.final_source_ == "linear_fallback", info
    assert _r2(y, m.predict(X)) > 0.95
    sp.sympify(m.model())


def test_safeguard_never_much_worse_than_mean():
    rng = np.random.default_rng(5)
    X = rng.uniform(1, 5, (300, 3))
    y = rng.normal(0, 1, 300)                     # no relation at all
    m = PIRRegressor(gates="all", safeguard=True).fit(X, y)
    assert _r2(y, m.predict(X)) > -0.01


def test_safeguard_false_matches_36_behaviour():
    rng = np.random.default_rng(6)
    X = rng.uniform(-1, 1, (300, 3))
    y = 1 + X[:, 0] + X[:, 1]
    a = PIRRegressor(gates="all").fit(X, y)
    b = PIRRegressor(gates="all", safeguard=False).fit(X, y)
    assert a.model() == b.model() and "safeguard" not in b.gate_info_
