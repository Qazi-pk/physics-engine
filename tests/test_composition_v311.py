"""PIR 3.11 (W): composition family and readable exact forms. Small synthetic cases, fixed seeds."""
import numpy as np
import sympy as sp

from physics_engine.discovery.noise_tolerant import stage, vcomp
from physics_engine.discovery.readable import readable_exact


def _noisy(y, level, seed=1):
    return y + np.random.default_rng(seed).normal(0, level * np.sqrt(np.mean(y ** 2)), len(y))


def _same_law(expr, true_expr, names):
    """SRBench-like: after rounding floats to 3 decimals, the ratio is a constant."""
    loc = {n: sp.Symbol(n) for n in names}
    m = sp.sympify(expr, locals=loc)
    m = m.xreplace({a: sp.Float(round(float(a), 3)) for a in m.atoms(sp.Float)})
    t = sp.sympify(true_expr, locals=loc)
    return sp.simplify(m / t).is_number


def test_comp_exact_planck_form():
    RNG = np.random.default_rng(0)
    n = 4000
    X = RNG.uniform(1, 5, (n, 4)); names = ["h", "omega", "T", "kb"]
    f = 1 / (np.exp(0.159 * X[:, 0] * X[:, 1] / (X[:, 2] * X[:, 3])) - 1)
    res, info = vcomp.comp_exact(X, f, names)
    assert res is not None, info
    assert res["max_err"] < 1e-6
    assert _same_law(res["expr"], "1/(exp(0.159*h*omega/(T*kb)) - 1)", names)


def test_comp_exact_arcsin_and_log():
    RNG = np.random.default_rng(0)
    n = 4000
    X = np.c_[RNG.uniform(1, 2, n), RNG.uniform(2, 5, n), RNG.uniform(1, 5, n)]; names = ["lambd", "d", "n"]
    f = np.arcsin(X[:, 0] / (X[:, 1] * X[:, 2]))
    res, info = vcomp.comp_exact(X, f, names)
    assert res is not None and _same_law(res["expr"], "asin(lambd/(d*n))", names), info
    X = RNG.uniform(1, 5, (n, 3)); names = ["T", "V1", "V2"]
    f = X[:, 0] * np.log(X[:, 2] / X[:, 1])
    res, info = vcomp.comp_exact(X, f, names)
    assert res is not None and _same_law(res["expr"], "T*log(V2/V1)", names), info


def test_comp_relativistic_factor_under_noise():
    RNG = np.random.default_rng(0)
    n = 6000
    X = np.c_[RNG.uniform(1, 5, n), RNG.uniform(1, 2, n), RNG.uniform(3, 10, n)]; names = ["m", "v", "c"]
    f = X[:, 0] * X[:, 2] ** 2 / np.sqrt(1 - X[:, 1] ** 2 / X[:, 2] ** 2)
    best, info = vcomp.v_comp(X, _noisy(f, 0.01), names)
    assert best is not None, info
    assert "sqrt(1 - v**2/c**2)" in best["expr"]


def test_comp_rejects_pure_noise():
    RNG = np.random.default_rng(0)
    n = 3000
    X = RNG.uniform(1, 5, (n, 3)); names = ["a", "b", "c"]
    best, info = vcomp.v_comp(X, RNG.normal(0, 1, n) + 5, names)
    assert best is None


def test_stage_composition_flag():
    RNG = np.random.default_rng(0)
    n = 4000
    X = np.c_[RNG.uniform(1, 2, n), RNG.uniform(2, 5, n), RNG.uniform(1, 5, n)]; names = ["lambd", "d", "n"]
    f = np.arcsin(X[:, 0] / (X[:, 1] * X[:, 2]))
    y = _noisy(f, 0.001)
    cur = np.full(n, y.mean())
    txt0, info0 = stage.run_noise_tolerant(X, y, names, cur, composition=False)
    txt1, info1 = stage.run_noise_tolerant(X, y, names, cur, composition=True)
    assert "comp" not in info0["tried"] and "comp" in info1["tried"]
    assert txt1 is not None and "asin" in txt1


def test_readable_merges_roots_and_perfect_squares():
    RNG = np.random.default_rng(0)
    n = 500
    names = ["gamma", "pr", "rho"]; S = [sp.Symbol(x) for x in names]
    X = RNG.uniform(1, 5, (n, 3)); y = np.sqrt(X[:, 0] * X[:, 1] / X[:, 2])
    e = sp.sympify("1.0*sqrt(gamma)*sqrt(pr)/sqrt(rho)", locals=dict(zip(names, S)))
    assert str(readable_exact(e, S, X, y)) == "1.0*sqrt(gamma*pr/rho)"
    names = ["q", "v", "c"]; S = [sp.Symbol(x) for x in names]
    X = np.c_[RNG.uniform(1, 5, n), RNG.uniform(1, 2, n), RNG.uniform(3, 10, n)]
    y = X[:, 0] / (1 - X[:, 1] / X[:, 2])
    e = sp.sympify("q/sqrt(1.86929359874611 - 3.73858719749222*v/c + 1.86929359874611*v**2/c**2)", locals=dict(zip(names, S)))
    r = readable_exact(e, S, X, y)
    assert r is not None and not r.atoms(sp.Pow) - {p for p in r.atoms(sp.Pow) if p.exp.is_Integer}, r
    fn = sp.lambdify(S, r, "numpy")
    assert np.allclose(fn(*X.T), y / np.sqrt(1.86929359874611), rtol=1e-6)   # same law, constant folded


def test_readable_leaves_clean_forms_alone():
    RNG = np.random.default_rng(0)
    names = ["a", "b"]; S = [sp.Symbol(x) for x in names]
    X = RNG.uniform(1, 5, (200, 2)); y = X[:, 0] * X[:, 1] ** 2
    assert readable_exact(sp.sympify("a*b**2", locals=dict(zip(names, S))), S, X, y) is None


def test_composition_flag_off_by_default():
    RNG = np.random.default_rng(0)
    from physics_engine.sklearn_adapter import PIRRegressor
    assert PIRRegressor().composition is False
