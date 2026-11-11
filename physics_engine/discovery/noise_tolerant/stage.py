"""
Noise-tolerant stage (PIR 3.10, feature V; plan: claude/pir310_plan.md).

The 3.6-3.9 stages accept a law only if it reproduces the training rows to max scaled error < 1e-6,
which noisy data never meets. This stage accepts validated, not exact, forms. Rules (fixed at the
3.10 pre-registration commit, before the V2/V3 benchmark run):

V1  Trigger. Runs in PIRRegressor(noise_tolerant=True) (default False) after the late separability
    pass and before the safeguard, and only if the current model is not exact on the training rows
    (max |pred - y| / max |y| >= 1e-6). An exact model is never replaced; at noise 0 nothing changes
    for laws that are already exact.
V2  Rows. At most 20,000 training rows (numpy default_rng(0), sorted) and one fixed 75/25 split
    (default_rng(0) permutation). Every family fits on the 75% and is judged on the same 25%.
V3  Families.
    sep: y = C * M(A) * h(B); M a monomial of the inputs in A; h**p (p in +-1, +-2, +-1/2) a sum of
         <= 3 atoms from the 3.9 power and sin/cos libraries (|B| <= 3 power atoms, |B| <= 4 trig).
         Candidate splits: the additive-log split (inputs whose log-effect is curved go to B), the
         3.9 block split, the 3.9 cross-term splits; at most 4 orientations. Positive inputs only;
         constant and duplicated columns are left out.
    exp: y = s * C * prod x**a * exp(sum of <= 3 monomials from the 3.6 exp-gate pool); exhaustive
         subsets among the 150 pool terms most correlated with the log residual; the pool is built
         on a seeded row subset that keeps it under 400 MB (d <= 5: all rows).
    rat: y = P / Q with <= 3 and 2..3 monomials of degree <= 2 (<= 5 inputs); supports chosen by the
         linearized residual y*Q - P.
V4  Fitting. Coefficients refined by Gauss-Newton in y space (unbiased for additive noise). Best atom
    subsets by Gram matrix; among sizes, the smallest whose held-out MSE is within 2% of the best.
V5  Snapping. Exponents to p/q (q <= 4) within 3 standard errors + 0.01; coefficient ratios (and exp
    coefficients) to p/q (q <= 6) within 3 standard errors (delta method). The overall scale stays
    a fitted number.
V6  Acceptance (lack of fit). Snapped held-out MSE <= 1.02 x unsnapped, and <= 1.05 x the best
    flexible reference (same family with a polynomial in place of the snapped part).
V7  Replacement. Among accepted forms the lowest held-out MSE wins; near-ties (relative 1e-6, the
    same law written two ways) go to the expression with fewest operations, then to the family tried
    first (sep, exp, rat). It replaces the current model only if its held-out MSE is lower than the
    current model's on the same rows. The 3.7 safeguard runs afterwards as before.
V9  Minimum data (amendment, 11 Nov 2026, after the first black-box run failed on a 100-row
    dataset): the stage runs only with at least 2,000 training rows, so that its held-out set has
    at least 500 rows. Feynman training sets have 75,000 rows, so V2/V3 results are unaffected.
V8  Output form. Equivalent forms that SRBench's scorer can verify without assuming positive
    variables (SRBench issue #235), each checked numerically (relative 1e-9) against the fitted
    model before use: exact roots written as monomials; a dominant monomial pulled out of a
    two-term root; roots merged; one root for a single term and for half-integer power laws;
    common factors pulled out of sums; the exponent of exp factored.
"""
import numpy as np
import sympy as sp

from . import vsep, vexp, vrat

MAX_ROWS = 20000
MIN_TRAIN_ROWS = 2000   # V9
EXACT_TOL = 1e-6


def _rows(n, seed):
    """The rows and the 75/25 split every family uses (same rule as in vsep/vexp/vrat)."""
    keep = np.arange(n)
    if n > MAX_ROWS:
        keep = np.sort(np.random.default_rng(seed).choice(n, MAX_ROWS, replace=False))
    tr, va = vsep._split(len(keep), seed)
    return keep, tr, va


def run_noise_tolerant(X, y, names, current_pred, seed=0):
    """X: 2-D array (training rows), y: 1-D, names: column names (safe identifiers),
    current_pred: the current model's predictions on X. Returns (expr_string or None, info)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    info = {"tried": {}}
    if len(y) < MIN_TRAIN_ROWS:
        info["decision"] = "too_few_rows"
        return None, info
    scale = float(max(np.max(np.abs(y)), 1e-300))
    cur_err = float(np.max(np.abs(current_pred - y)) / scale)
    if np.isfinite(cur_err) and cur_err < EXACT_TOL:
        info["decision"] = "current_exact"
        return None, info
    keep, tr, va = _rows(len(y), seed)
    yk, pk = y[keep], np.asarray(current_pred, dtype=float)[keep]
    cur_mse = float(np.mean((pk[va] - yk[va]) ** 2))
    if not np.isfinite(cur_mse):
        cur_mse = float("inf")
    info["current_val_mse"] = cur_mse
    best = None
    for name, fn in (("sep", vsep.v_sep), ("exp", vexp.v_exp), ("rat", vrat.v_rat)):
        try:
            res, sub = fn(X, y, names, seed=seed)
        except Exception as e:  # a family must never break the fit
            info["tried"][name] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
            continue
        info["tried"][name] = {"decision": sub.get("decision"),
                               "val_mse": None if res is None else res.get("val_mse"),
                               "expr": None if res is None else res.get("expr")}
        if res is not None and res.get("expr") and res.get("val_mse") is not None:
            v = float(res["val_mse"])
            try:
                ops = int(sp.count_ops(sp.sympify(res["expr"])))
            except Exception:
                ops = 10 ** 6
            # lowest held-out MSE wins; near-ties (relative 1e-6: the same law written two ways) go to
            # the shorter expression (fewest operations), then to the family tried first
            if best is None or v < best[1] * (1 - 1e-6) or (v <= best[1] * (1 + 1e-6) and ops < best[3]):
                best = (name, v, res["expr"], ops)
    if best is None:
        info["decision"] = "none_accepted"
        return None, info
    info.update(best_family=best[0], best_val_mse=best[1])
    if not best[1] < cur_mse:
        info["decision"] = "not_better_than_current"
        return None, info
    info["decision"] = "accepted"
    return best[2], info
