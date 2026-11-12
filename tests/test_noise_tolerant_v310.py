"""PIR 3.10 (V): noise-tolerant stage. Small synthetic cases, fixed seeds."""
import numpy as np
import sympy as sp

from physics_engine.discovery.noise_tolerant import stage, vexp, vrat, vsep

RNG = np.random.default_rng(0)


def _noisy(y, level, seed=1):
    return y + np.random.default_rng(seed).normal(0, level * np.sqrt(np.mean(y ** 2)), len(y))


def _close(expr, names, X, f, rtol=2e-2):
    e = sp.sympify(expr, locals={n: sp.Symbol(n) for n in names})
    g = sp.lambdify([sp.Symbol(n) for n in names], e, "numpy")
    v = np.asarray(g(*X.T), dtype=float)
    return np.max(np.abs(v - f) / (np.abs(f) + 1e-12)) < rtol


def test_vsep_root_sum_under_noise():
    n = 6000
    X = RNG.uniform(1, 5, (n, 4)); names = ["mom", "Bx", "By", "Bz"]
    f = X[:, 0] * np.sqrt(X[:, 1] ** 2 + X[:, 2] ** 2 + X[:, 3] ** 2)
    best, info = vsep.v_sep(X, _noisy(f, 0.01), names)
    assert best is not None, info
    assert "sqrt(Bx**2 + By**2 + Bz**2)" in best["expr"]


def test_vexp_gaussian_under_noise():
    n = 6000
    X = np.c_[RNG.uniform(1, 3, n), RNG.uniform(1, 3, n)]; names = ["sigma", "theta"]
    f = np.exp(-(X[:, 1] / X[:, 0]) ** 2 / 2) / (np.sqrt(2 * np.pi) * X[:, 0])
    best, info = vexp.v_exp(X, _noisy(f, 0.01), names)
    assert best is not None, info
    assert _close(best["expr"], names, X, f)


def test_vrat_weighted_mean_under_noise():
    n = 6000
    X = RNG.uniform(1, 5, (n, 4)); names = ["m1", "m2", "r1", "r2"]
    f = (X[:, 0] * X[:, 2] + X[:, 1] * X[:, 3]) / (X[:, 0] + X[:, 1])
    best, info = vrat.v_rat(X, _noisy(f, 0.001), names)
    assert best is not None, info
    assert _close(best["expr"], names, X, f, rtol=5e-3)


def test_stage_keeps_exact_current_model():
    X = RNG.uniform(1, 2, (3000, 2)); y = X[:, 0] * X[:, 1]
    expr, info = stage.run_noise_tolerant(X, y, ["a", "b"], y.copy())
    assert expr is None and info["decision"] == "current_exact"


def test_stage_replaces_only_if_better():
    n = 4000
    X = RNG.uniform(1, 5, (n, 3)); names = ["a", "b", "c"]
    f = X[:, 0] * X[:, 1] ** 2 / X[:, 2]; y = _noisy(f, 0.01)
    expr, info = stage.run_noise_tolerant(X, y, names, f.copy())   # current = the truth itself
    # with the truth as the current model, the stage may only keep it, or return the same law with a
    # refitted scale that happens to score a little better on the held-out rows of this noisy sample
    assert expr is None or _close(expr, names, X, f, rtol=1e-3), (expr, info)


def test_stage_never_raises_on_messy_input():
    n = 2500
    X = RNG.normal(size=(n, 6)); X[:, 2] = 1.0; X[:, 3] = X[:, 0]
    y = X[:, 0] - X[:, 1] + RNG.normal(size=n)
    expr, info = stage.run_noise_tolerant(X, y, [f"x{i}" for i in range(6)], np.zeros(n))
    assert "decision" in info


def test_adapter_flag_off_by_default():
    from physics_engine.sklearn_adapter import PIRRegressor
    assert PIRRegressor().noise_tolerant is False


def test_exact_root_written_as_monomial():
    """sqrt(k**2*x**4) must come out as k*x**2 (the scorer cannot simplify roots without positivity)."""
    n = 6000
    X = RNG.uniform(1, 5, (n, 2)); names = ["k_spring", "x"]
    f = 0.5 * X[:, 0] * X[:, 1] ** 2
    expr, info = stage.run_noise_tolerant(X, _noisy(f, 0.001), names, np.zeros(n))
    assert expr is not None, info
    assert "sqrt" not in expr and "k_spring*x**2" in expr


def test_stage_skips_small_data():
    """V9: fewer than 2,000 training rows -> the stage does nothing."""
    X = RNG.uniform(1, 5, (1500, 3)); f = X[:, 0] * X[:, 1]
    expr, info = stage.run_noise_tolerant(X, _noisy(f, 0.01), ["a", "b", "c"], np.zeros(1500))
    assert expr is None and info["decision"] == "too_few_rows"
