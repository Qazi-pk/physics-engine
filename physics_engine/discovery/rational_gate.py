"""
rational_gate.py -- equation-error rational linearization gate for PIR.

Fifth member of the linearization family (log-, reciprocal-, square-,
rational-sum): if  y = P(x)/Q(x)  with P, Q polynomials, then

        y*Q(x) - P(x) = 0

is LINEAR and HOMOGENEOUS in the coefficients of P and Q. Stacking
columns [y*phi(x) | -phi(x)] and taking the right singular vector with
the smallest singular value recovers (q, p) up to scale. Deterministic,
no nonlinear solver, no seeds.

Covers targets the separability module cannot reach because they are not
multiplicatively separable, e.g. normalized differences
(a-b)/(a+b) (NDVI, NBR, MNDWI) and EVI-type ratios.

ISOLATION: new file; imports nothing from the engine. Intended to be
called as a FALLBACK ONLY, after the existing pipeline fails, so every
established recovery keeps its exact code path.

Usage:
    from rational_gate import rational_gate
    res = rational_gate(X, y, names)        # X: (n, d) array
    if res is not None and res["exact"]:
        print(res["expr"], res["max_rel_err"])
"""
from fractions import Fraction
from itertools import combinations_with_replacement

import numpy as np

EXACT_TOL = 1e-6          # same full-recovery threshold as separability
DEN_MIN_RATIO = 1e-6      # min |Q| / max |Q| on data; below = near-pole
PRUNE_THRESHOLDS = (1e-8, 1e-6, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2)
PRUNE_ERR_SLACK = 1.05    # sparse fit may be at most 5% worse (RMS)
MAX_ROWS = 20_000         # subsample for speed; final check uses all rows
MAX_DEG2_VARS = 8         # degree-2 basis size grows ~d^2/2


# ---------------------------------------------------------------- basis
def _basis(X, names, degree):
    cols, labs = [np.ones(len(X))], ["1"]
    for i, n in enumerate(names):
        cols.append(X[:, i]); labs.append(n)
    if degree >= 2:
        for i, j in combinations_with_replacement(range(len(names)), 2):
            cols.append(X[:, i] * X[:, j])
            labs.append(f"{names[i]}**2" if i == j else f"{names[i]}*{names[j]}")
    return np.column_stack(cols), labs


# ---------------------------------------------------------- core solve
def _null_vector(y, Phi, support):
    """Null vector of [y*Phi | -Phi] restricted to `support` (indices
    into the 2k stacked coefficient vector). Returns full-length v and
    the two smallest singular values."""
    A = np.hstack([y[:, None] * Phi, -Phi])[:, support]
    norms = np.linalg.norm(A, axis=0)
    norms[norms == 0] = 1.0
    _, s, Vt = np.linalg.svd(A / norms, full_matrices=False)
    v = np.zeros(2 * Phi.shape[1])
    v[support] = Vt[-1] / norms
    s2 = s[-2] if len(s) > 1 else np.inf
    return v, s[-1], s2


def _evaluate(v, Phi, y):
    k = Phi.shape[1]
    q, p = v[:k], v[k:]
    if not np.any(q):
        return None
    Q = Phi @ q
    aQ = np.abs(Q)
    if aQ.max() == 0 or aQ.min() / aQ.max() < DEN_MIN_RATIO:
        return None                               # pole on data
    if np.sign(Q).min() != np.sign(Q).max():
        return None                               # denominator crosses zero
    yhat = (Phi @ p) / Q
    scale = np.max(np.abs(y)) or 1.0
    max_rel = np.max(np.abs(yhat - y)) / scale
    rms_rel = np.sqrt(np.mean((yhat - y) ** 2)) / (np.std(y) or 1.0)
    return max_rel, rms_rel


def _scaled_magnitudes(v, Phi, y):
    """Coefficient size times its column's typical magnitude, so
    thresholds are unit-free."""
    A = np.hstack([y[:, None] * Phi, -Phi])
    m = np.abs(v) * np.linalg.norm(A, axis=0)
    return m / (m.max() or 1.0)


# -------------------------------------------------------- formatting
def _nice(c, tol=1e-8):
    f = Fraction(c).limit_denominator(64)
    if abs(float(f) - c) <= tol * max(1.0, abs(c)):
        return f
    return c


def _poly_str(coefs, labs):
    terms = []
    for c, l in zip(coefs, labs):
        if c == 0:
            continue
        c = _nice(c)
        cs = str(c) if isinstance(c, Fraction) else repr(float(c))  # full precision: the string IS the model
        if isinstance(c, Fraction) and c.denominator != 1:
            cs = f"({cs})"
        terms.append(cs if l == "1" else f"{cs}*{l}")
    return " + ".join(terms).replace("+ -", "- ") if terms else "0"


def _normalize(v, k):
    """Scale so the largest-|.| denominator coefficient equals 1."""
    q = v[:k]
    return v / q[np.argmax(np.abs(q))]


# --------------------------------------------------------------- gate
def rational_gate(X, y, names, max_degree=2, exact_tol=EXACT_TOL, seed=0):
    """Try y = P(x)/Q(x) at degree 1, then degree 2.

    Returns dict(expr, degree, n_terms, max_rel_err, rms_rel_err, exact,
    null_dim_gap) for the best sparse fit, or None if no admissible
    rational form exists (pole on data, or no fit at all).
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y = X[ok], y[ok]
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(y), min(len(y), MAX_ROWS), replace=False)

    best = None
    for degree in range(1, max_degree + 1):
        if degree == 2 and X.shape[1] > MAX_DEG2_VARS:
            break
        Phi_all, labs = _basis(X, names, degree)
        Phi, ys = Phi_all[sub], y[sub]
        k = Phi.shape[1]

        full = np.arange(2 * k)
        v_full, s_min, s_next = _null_vector(ys, Phi, full)
        ev_full = _evaluate(v_full, Phi, ys)
        if ev_full is None:
            continue
        rms_full = ev_full[1]

        # sequential thresholding: keep the sparsest support whose RMS
        # error stays within PRUNE_ERR_SLACK of the full fit
        chosen = (v_full, ev_full)
        mags = _scaled_magnitudes(v_full, Phi, ys)
        for thr in PRUNE_THRESHOLDS:
            support = np.where(mags >= thr)[0]
            if len(support) < 2 or not np.any(support < k) or not np.any(support >= k):
                break
            v_s, _, _ = _null_vector(ys, Phi, support)
            ev = _evaluate(v_s, Phi, ys)
            if ev is None:
                break
            if ev[1] <= PRUNE_ERR_SLACK * rms_full + 1e-14:
                chosen = (v_s, ev)
            else:
                break

        v = _normalize(chosen[0], k)
        ev_all = _evaluate(v, Phi_all, y)            # verify on ALL rows
        if ev_all is None:
            continue
        n_terms = int(np.count_nonzero(v))
        res = dict(
            expr=f"({_poly_str(v[k:], labs)}) / ({_poly_str(v[:k], labs)})",
            degree=degree,
            n_terms=n_terms,
            max_rel_err=float(ev_all[0]),
            rms_rel_err=float(ev_all[1]),
            exact=bool(ev_all[0] < exact_tol),
            null_dim_gap=float(s_min / s_next) if np.isfinite(s_next) else None,
        )
        if best is None or (res["exact"] and not best["exact"]) or \
           (res["exact"] == best["exact"] and res["rms_rel_err"] < 0.5 * best["rms_rel_err"]):
            best = res
        if best["exact"]:
            break                                    # lowest degree wins
    return best


# ---------------------------------------------------------- self-test
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N = 10_000
    names = ["B2", "B3", "B4", "B8", "B11", "B12"]
    B = {b: rng.uniform(0.02, 0.6, N) for b in names}
    B["B2"] = rng.uniform(0.01, 0.12, N)          # realistic blue
    X = np.column_stack([B[n] for n in names])

    cases = {
        "NDVI":  (B["B8"] - B["B4"]) / (B["B8"] + B["B4"]),
        "NBR":   (B["B8"] - B["B12"]) / (B["B8"] + B["B12"]),
        "MNDWI": (B["B3"] - B["B11"]) / (B["B3"] + B["B11"]),
        "EVI":   2.5 * (B["B8"] - B["B4"]) / (B["B8"] + 6 * B["B4"] - 7.5 * B["B2"] + 1),
        # degree-2 case: II_11_27-like shape, n*a/(1 - n*a/3)
        "RAT2":  B["B3"] * B["B4"] / (1 - B["B3"] * B["B4"] / 3),
    }
    negatives = {
        "monomial a*b*c": B["B3"] * B["B4"] * B["B8"],
        "exp":            np.exp(B["B8"] / B["B4"]),
        "sqrt-sum":       np.sqrt(B["B3"] ** 2 + B["B4"] ** 2 + B["B8"] ** 2),
    }

    print("=== Positives, clean (expect exact) ===")
    fails = 0
    for t, y in cases.items():
        r = rational_gate(X, y, names)
        good = r is not None and r["exact"]
        fails += not good
        print(f"{'PASS' if good else 'FAIL'} {t:6s} deg={r and r['degree']} "
              f"err={r and r['max_rel_err']:.2e}\n      {r and r['expr']}")

    print("\n=== Positives, 1% noise (expect right form, sparse) ===")
    for t, y in cases.items():
        yn = y * (1 + 0.01 * rng.standard_normal(N))
        r = rational_gate(X, yn, names)
        print(f"{t:6s} terms={r and r['n_terms']} rms={r and r['rms_rel_err']:.3f}\n"
              f"      {r and r['expr']}")

    print("\n=== Negatives (expect NOT exact -> gate does not fire) ===")
    for t, y in negatives.items():
        r = rational_gate(X, y, names)
        fired = r is not None and r["exact"]
        fails += fired
        print(f"{'FAIL (false positive)' if fired else 'PASS'} {t:15s} "
              f"err={r['max_rel_err'] if r else float('nan'):.2e}")

    print(f"\nOVERALL: {'ALL PASSED' if fails == 0 else f'{fails} FAILURE(S)'}")
