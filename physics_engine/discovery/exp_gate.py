"""
exp_gate.py -- exponential ("semi-log") linearization gate for PIR.

Family of laws (first-order rate equations, Beer-Lambert, Boltzmann factors,
Gaussians):

        y = s * C * prod_i x_i^a_i * exp( sum_k b_k * m_k(x) )

where s = +-1, C > 0, and each m_k is a monomial (a product of up to 3
variables with integer powers in {-2,-1,1,2}). Taking logs:

        ln|y| = ln C + sum_i a_i ln x_i + sum_k b_k m_k(x)

which is LINEAR in (ln C, a_i, b_k). Same idea as the F3 log-linear gate and
the rational gate: turn the search into least squares, then accept only if the
result is EXACT on the data.

Search: the power-law part (constant + ln x_i of positive columns) is always
in the fit; exponent monomials are added greedily (orthogonal matching
pursuit), up to `max_terms`, stopping as soon as the fit is exact. Then
near-zero coefficients are pruned and coefficients are snapped to simple
fractions only if exactness survives.

ISOLATION: new file, numpy + sympy + stdlib only, imports nothing from the
engine. Intended as a FALLBACK ONLY (after the engine and the rational gate).
"""
from fractions import Fraction
from itertools import combinations, product

import numpy as np

EXACT_TOL = 1e-6
POWERS = (-2, -1, 1, 2)
MAX_VARS_PER_TERM = 3
MAX_ROWS = 4000


def _monomial_pool(X, names, nonzero):
    cols, labels = [], []
    d = X.shape[1]
    for r in range(1, MAX_VARS_PER_TERM + 1):
        for idx in combinations(range(d), r):
            for pw in product(POWERS, repeat=r):
                if any(p < 0 and not nonzero[i] for i, p in zip(idx, pw)):
                    continue
                col = np.ones(len(X))
                for i, p in zip(idx, pw):
                    col = col * X[:, i] ** p
                if not np.all(np.isfinite(col)) or np.std(col) == 0:
                    continue
                cols.append(col)
                labels.append(tuple(zip(idx, pw)))
    # single large products with exponents +-1 (e.g. m*g*x/(kb*T))
    for r in range(MAX_VARS_PER_TERM + 1, d + 1):
        for idx in combinations(range(d), r):
            for pw in product((-1, 1), repeat=r):
                if any(p < 0 and not nonzero[i] for i, p in zip(idx, pw)):
                    continue
                col = np.ones(len(X))
                for i, p in zip(idx, pw):
                    col = col * X[:, i] ** p
                if not np.all(np.isfinite(col)) or np.std(col) == 0:
                    continue
                cols.append(col)
                labels.append(tuple(zip(idx, pw)))
    return cols, labels


def _lstsq(A, z):
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    return coef, z - A @ coef


def _y_err(A, coef, s, y):
    with np.errstate(over="ignore", invalid="ignore"):
        yh = s * np.exp(A @ coef)
    if not np.all(np.isfinite(yh)):
        return np.inf
    return float(np.max(np.abs(yh - y)) / (np.max(np.abs(y)) or 1.0))


def _snap(c, tol=1e-9):
    f = Fraction(c).limit_denominator(12)
    return float(f) if abs(float(f) - c) <= tol * max(1.0, abs(c)) else c


def exp_gate(X, y, names, max_terms=4, exact_tol=EXACT_TOL, seed=0):
    """Return dict(expr, exact, max_rel_err, n_exp_terms) or None if the
    family does not apply (y changes sign or hits zero)."""
    import sympy as sp

    X = np.asarray(X, float)
    y = np.asarray(y, float)
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y = X[ok], y[ok]
    if len(y) < 10 or np.any(y == 0) or not (np.all(y > 0) or np.all(y < 0)):
        return None
    s = 1.0 if y[0] > 0 else -1.0
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(y), min(len(y), MAX_ROWS), replace=False)
    Xs, ys = X[sub], y[sub]
    z = np.log(np.abs(ys))

    positive = np.all(X > 0, axis=0)
    nonzero = np.all(X != 0, axis=0)

    def base_cols(Z):
        cols = [np.ones(len(Z))] + [np.log(Z[:, i]) for i in range(Z.shape[1]) if positive[i]]
        return cols
    base_labels = ["const"] + [("log", i) for i in range(X.shape[1]) if positive[i]]

    pool, pool_labels = _monomial_pool(Xs, names, nonzero)
    chosen = []                                   # indices into pool

    def design(cols_idx, Z=None, P=None):
        if Z is None:
            B, P_ = base_cols(Xs), pool
        else:
            B, P_ = base_cols(Z), P
        return np.column_stack(B + [P_[k] for k in cols_idx])

    A = design(chosen)
    coef, r = _lstsq(A, z)
    best = (_y_err(A, coef, s, ys), list(chosen))
    P = np.column_stack(pool) if pool else np.zeros((len(ys), 0))
    B0 = np.column_stack(base_cols(Xs))

    # (a) exhaustive: base + every single exponent term
    if best[0] >= exact_tol * 1e-3:
        for k in range(P.shape[1]):
            Ak = np.column_stack([B0, P[:, k]])
            ck, _ = _lstsq(Ak, z)
            e = _y_err(Ak, ck, s, ys)
            if e < best[0]:
                best = (e, [k])

    # (b) all terms at once + sequential thresholding (only if rows allow it)
    if best[0] >= exact_tol * 1e-3 and 0 < P.shape[1] and B0.shape[1] + P.shape[1] < len(ys) // 2:
        active = list(range(P.shape[1]))
        for thr in (1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2):
            Aa = np.column_stack([B0] + [P[:, k] for k in active])
            ca, _ = _lstsq(Aa, z)
            if _y_err(Aa, ca, s, ys) >= exact_tol:
                break
            mag = np.abs(ca[B0.shape[1]:]) * np.linalg.norm(P[:, active], axis=0)
            mag = mag / (np.max(np.abs(z)) * np.sqrt(len(z)) + 1e-300)
            nxt = [k for k, m in zip(active, mag) if m > thr]
            if not nxt or len(nxt) == len(active) and thr > 1e-10:
                pass
            Ab = np.column_stack([B0] + [P[:, k] for k in nxt]) if nxt else B0
            cb, _ = _lstsq(Ab, z)
            eb = _y_err(Ab, cb, s, ys)
            if eb < exact_tol and len(nxt) <= max_terms and (len(nxt) < len(best[1]) or best[0] >= exact_tol):
                best = (eb, nxt)
            if eb >= exact_tol:
                break
            active = nxt

    # (c) greedy orthogonal matching pursuit
    if best[0] >= exact_tol * 1e-3 and P.shape[1]:
        Pn = P / (np.linalg.norm(P, axis=0) + 1e-300)
        chosen = []
        A = B0
        coef, r = _lstsq(A, z)
        for _ in range(max_terms):
            Q, _ = np.linalg.qr(A)
            Rp = Pn - Q @ (Q.T @ Pn)
            nrm = np.linalg.norm(Rp, axis=0)
            score = np.where(nrm > 1e-8, np.abs(Rp.T @ r) / (nrm + 1e-300), 0.0)
            score[chosen] = 0.0
            k = int(np.argmax(score))
            if score[k] <= 0:
                break
            chosen.append(k)
            A = np.column_stack([B0] + [P[:, j] for j in chosen])
            coef, r = _lstsq(A, z)
            err = _y_err(A, coef, s, ys)
            if err < best[0]:
                best = (err, list(chosen))
            if err < exact_tol * 1e-3:
                break

    chosen = best[1]
    A = design(chosen)
    coef, _ = _lstsq(A, z)

    # prune near-zero coefficients (power-law exponents and exp terms), keep exactness
    labels = base_labels + [pool_labels[k] for k in chosen]
    keep = [j for j, c in enumerate(coef) if j == 0 or abs(c) > 1e-8]
    A2 = A[:, keep]
    c2, _ = _lstsq(A2, z)
    if _y_err(A2, c2, s, ys) <= max(best[0], 1e-14) * 10:
        A, coef, labels = A2, c2, [labels[j] for j in keep]
    # snap to simple fractions (not the constant) if exactness survives
    snapped = np.array([coef[0]] + [_snap(c) for c in coef[1:]])
    if _y_err(A, snapped, s, ys) <= max(_y_err(A, coef, s, ys), 1e-14) * 10:
        coef = snapped

    # build expression
    syms = sp.symbols(list(names))
    expo, prefactor = sp.Integer(0), sp.Integer(1)
    for c, lab in zip(coef, labels):
        cs = sp.Rational(Fraction(c).limit_denominator(12)) if c == _snap(c) and lab != "const" else sp.Float(repr(float(c)))
        if lab == "const":
            prefactor = prefactor * sp.Float(repr(float(np.exp(c))))
        elif isinstance(lab, tuple) and lab[0] == "log":
            prefactor = prefactor * syms[lab[1]] ** cs
        else:
            m = sp.Integer(1)
            for i, p in lab:
                m = m * syms[i] ** p
            expo = expo + cs * m
    expr = (sp.Integer(int(s)) * prefactor * sp.exp(expo)) if expo != 0 else sp.Integer(int(s)) * prefactor

    # verify the expression itself on ALL rows
    f = sp.lambdify(syms, expr, "numpy")
    with np.errstate(over="ignore", invalid="ignore"):
        yh = np.broadcast_to(np.asarray(f(*X.T), float), y.shape)
    err = float(np.max(np.abs(yh - y)) / (np.max(np.abs(y)) or 1.0)) if np.all(np.isfinite(yh)) else np.inf
    n_exp = sum(1 for lab in labels if isinstance(lab, tuple) and lab and lab[0] != "log")
    return dict(expr=str(expr), exact=bool(err < exact_tol), max_rel_err=err, n_exp_terms=n_exp)


# ---------------------------------------------------------------- self-test
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N = 2000
    U = lambda a, b: rng.uniform(a, b, N)
    cases = {
        # Feynman-style exponential laws (positives)
        "I.6.2a  exp(-th^2/2)/sqrt(2pi)": (["theta"], lambda th: np.exp(-th**2 / 2) / np.sqrt(2 * np.pi), [U(1, 3)]),
        "I.6.2   gaussian(theta, sigma)": (["sigma", "theta"],
                                           lambda sg, th: np.exp(-(th / sg)**2 / 2) / (np.sqrt(2 * np.pi) * sg),
                                           [U(1, 3), U(1, 3)]),
        "I.6.2b  gaussian(theta-theta1)": (["sigma", "theta", "theta1"],
                                           lambda sg, th, t1: np.exp(-((th - t1) / sg)**2 / 2) / (np.sqrt(2 * np.pi) * sg),
                                           [U(1, 3), U(1, 3), U(1, 3)]),
        "I.40.1  n0*exp(-m g x/(kb T))": (["n_0", "m", "x", "T", "g", "kb"],
                                           lambda n0, m, x, T, g, kb: n0 * np.exp(-m * g * x / (kb * T)),
                                           [U(1, 3), U(1, 3), U(1, 3), U(1, 3), U(1, 3), U(1, 3)]),
        "Beer-Lambert I0*exp(-a z)":      (["I0", "alpha", "z"], lambda I0, a, z: I0 * np.exp(-a * z),
                                           [U(1, 5), U(0.1, 2), U(0.1, 3)]),
        "Arrhenius A*T^0.5*exp(-E/T)":    (["A", "E", "T"], lambda A, E, T: A * np.sqrt(T) * np.exp(-E / T),
                                           [U(1, 5), U(1, 5), U(1, 10)]),
    }
    negatives = {
        "rational x/(1+x)":   (["x"], lambda x: x / (1 + x), [U(0.5, 3)]),
        "sin(x)*y":           (["x", "y"], lambda x, yv: np.sin(x) * yv, [U(0.2, 1.4), U(1, 3)]),
        "Planck x^3/(e^x-1)": (["x"], lambda x: x**3 / (np.exp(x) - 1), [U(0.5, 5)]),
        "sum a+b":            (["a", "b"], lambda a, b: a + b, [U(1, 3), U(1, 3)]),
    }
    fails = 0
    print("=== positives (expect EXACT) ===")
    for name, (nm, fn, cols) in cases.items():
        X = np.column_stack(cols); yv = fn(*cols)
        r = exp_gate(X, yv, nm)
        ok = r is not None and r["exact"]
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'} {name:32s} err={r and r['max_rel_err']:.1e}\n      {r and r['expr']}")
    print("\n=== negatives (expect NOT exact -> gate does not fire) ===")
    for name, (nm, fn, cols) in negatives.items():
        X = np.column_stack(cols); yv = fn(*cols)
        r = exp_gate(X, yv, nm)
        fired = r is not None and r["exact"]
        fails += fired
        print(f"{'FAIL (false positive)' if fired else 'PASS'} {name:22s} "
              f"err={(r['max_rel_err'] if r else float('nan')):.1e}")
    print(f"\nOVERALL: {'ALL PASSED' if fails == 0 else f'{fails} FAILURE(S)'}")
