"""Unit tests for separability_crossterm (PIR 3.8 plan, rule R1). Run: python3 -m pytest -q test_separability_crossterm.py"""
import numpy as np
import pandas as pd

from physics_engine.discovery.separability.crossterm import TAU, crossterm_search


def _df(f, n=4000, k=4, seed=0, lo=1.0, hi=3.0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(lo, hi, (n, k))
    d = pd.DataFrame(X, columns=[f"a{i}" for i in range(k)])
    d["target"] = f(*X.T)
    return d


def _mult(res):
    return {frozenset(r[0]) for r in res if r[2] == "MULT"}


def test_pure_product_every_split_separable():
    best, res = crossterm_search(_df(lambda a, b, c, d: a * b ** 2 / (c * d)))
    assert len(_mult(res)) == len(res)          # a monomial separates every way


def test_inseparable_pair_is_kept_together():
    best, res = crossterm_search(_df(lambda a, b, c, d: a * c * np.sqrt(b ** 2 + d ** 2)))
    for A in _mult(res):
        assert {"a1", "a3"} <= A or not ({"a1", "a3"} & A)
    assert _mult(res)                           # and some valid split is found


def test_fully_inseparable_gives_nothing():
    best, res = crossterm_search(_df(lambda a, b, c, d: 1.0 / (a + b + c + d)))
    assert best is None and not _mult(res)


def test_sorted_and_deterministic():
    d = _df(lambda a, b, c, d: a * c * np.sqrt(b ** 2 + d ** 2))
    r1, r2 = crossterm_search(d)[1], crossterm_search(d.copy())[1]
    # same splits, same order, same MULT/NONE decisions; the scores may differ in the last
    # digits (BLAS results depend on memory alignment, seen on Python 3.10 in CI)
    assert [x[:3] for x in r1] == [x[:3] for x in r2]
    assert np.allclose([x[3] for x in r1], [x[3] for x in r2], rtol=1e-9, atol=0)
    assert all(r1[i][3] <= r1[i + 1][3] for i in range(len(r1) - 1))
    assert all((x[2] == "MULT") == (x[3] < TAU) for x in r1)


def test_zero_in_data_falls_back_without_error():
    d = _df(lambda a, b, c, d: a * b * c * d)
    d.loc[0, "a0"] = 0.0
    best, res = crossterm_search(d)             # v7 partition_search path
    assert isinstance(res, list)


def test_too_many_inputs_gives_no_candidates():
    best, res = crossterm_search(_df(lambda *x: np.prod(x, axis=0), k=11, n=500))
    assert best is None and res == []


def test_blocks_recover_physical_structure():
    from physics_engine.discovery.separability.crossterm import block_split, crossterm_blocks
    d = _df(lambda a, b, c, d: a * c * np.sqrt(b ** 2 + d ** 2))
    assert crossterm_blocks(d) == [["a0"], ["a2"], ["a1", "a3"]]
    assert block_split(d) == (["a0", "a2"], ["a1", "a3"])


def test_block_split_absent_for_pure_product_or_fully_coupled():
    from physics_engine.discovery.separability.crossterm import block_split
    assert block_split(_df(lambda a, b, c, d: a * b * c * d)) is None          # no multi-variable block
    assert block_split(_df(lambda a, b, c, d: 1.0 / (a + b + c + d))) is None  # no singleton block


# ------------------------------------------------- end to end through PIRRegressor
def test_regressor_separability_recovers_nested_product():
    from physics_engine.sklearn_adapter import PIRRegressor
    rng = np.random.default_rng(7)
    X = rng.uniform(1, 3, (3000, 4))
    y = X[:, 0] * X[:, 2] * np.sqrt(X[:, 1] ** 2 + X[:, 3] ** 2)
    m = PIRRegressor(separability=True, gates="all", safeguard=True).fit(X, y)
    assert m.final_source_ == "separability", m.gate_info_.get("separability")
    assert np.max(np.abs(m.predict(X) - y)) / np.max(np.abs(y)) < 1e-6


def test_regressor_separability_off_by_default():
    from physics_engine.sklearn_adapter import PIRRegressor
    assert PIRRegressor().separability is False


def test_stage_extracts_on_fixed_subsample_and_checks_all_rows():
    # R2b: extraction on at most EXTRACT_ROWS rows (fixed seed), acceptance on every row
    import pandas as pd
    from physics_engine.discovery.separability import stage
    rng = np.random.default_rng(11)
    X = pd.DataFrame(rng.uniform(1, 3, (12000, 4)), columns=["a0", "a1", "a2", "a3"])
    y = (X.a0 * X.a2 * np.sqrt(X.a1 ** 2 + X.a3 ** 2)).to_numpy()
    txt, factors, info = stage.run_separability(X, y, base_err=1.0)
    assert info["extract_rows"] == stage.EXTRACT_ROWS == 5000
    assert info["n_rows"] == 12000
    assert txt is not None and info["accepted_err_scaled"] < stage.EXACT_TOL
