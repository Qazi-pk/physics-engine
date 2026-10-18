"""
noisy_gate.py -- noise-tolerant power-law ("monomial") gate for PIR.

    y ~= C * prod_j x_j ** a_j ,   a_j snapped to simple fractions p/q

Why: the rational and exp gates accept only EXACT fits (<1e-6), so on noisy or
rounded real data they never fire, and the baseline has only a few fixed
exponents (no 1/3, 3/4, ...). Example that motivated this: QTc Fridericia
(QT * RR^(-1/3)) on PTB-XL+ ECG data; the motivating data is NOT used to set
any threshold below.

Method (all thresholds fixed a priori, not tuned on any benchmark):
  1. y must keep one sign. Rows of the other sign are allowed only if they
     are few (<= 10%) and all small (|y| below the 25th percentile of the
     majority), i.e. noise around zero; they are dropped.
     Only strictly positive (or strictly negative) x columns enter; others get
     exponent 0.
  2. Split rows 75/25 (seeded). On the 75%: start from weighted least squares
     of ln|y| on [1, ln|x_j|] (weights y^2), then refine by nonlinear least
     squares in y-space (Gauss-Newton). NLS is the right estimator for additive
     noise; the log fit alone is biased (exponents shrink toward 0 at 10% noise).
     Exponent standard errors come from the NLS Jacobian.
  3. Snap each exponent a_j to the nearest p/q (q <= 4, |p/q| <= 3) if
     |a_j - p/q| <= 3 * SE(a_j) (+1e-6). If any exponent cannot be snapped,
     the target is not a simple monomial: return no form (unless allow_free).
  4. Refit C in y-space by least squares on the 75%.
  5. On the held-out 25%: the snapped model must be no worse than the
     free-exponent model by more than 2% in MSE (snapping must not cost fit),
     AND no worse than a flexible log-quadratic model (all (ln x)^2 and
     ln x_i ln x_j terms) by more than 5% (lack-of-fit test: a monomial must
     explain the data as well as a richer smooth model does).
Returns a dict with expr, exponents, val_mse, val_r2, snapped, or None.
The CALLER decides acceptance vs its baseline on held-out rows (see
gate_accept_vs_baseline).
"""
from fractions import Fraction

import numpy as np

MAX_DEN, MAX_ABS, K_SE, SNAP_SLACK, LOF_SLACK = 4, 3, 3.0, 0.02, 0.05
FRACTIONS = sorted({Fraction(p, q) for q in range(1, MAX_DEN + 1)
                    for p in range(-MAX_ABS * q, MAX_ABS * q + 1)})
_FR_ARR = np.array([float(f) for f in FRACTIONS])


def _split(n, seed, frac=0.75):
    perm = np.random.default_rng(seed).permutation(n)
    cut = int(frac * n)
    return perm[:cut], perm[cut:]


def _mono(Xa, expo):
    with np.errstate(all="ignore"):
        return np.exp(np.log(Xa) @ expo)


def _nls(F, y, beta0, iters=50):
    """Gauss-Newton for y ~ s * exp(F @ beta) (s = +-1 folded into y). Returns beta, cov or None."""
    beta = beta0.copy()
    lam = 1e-3
    def sse(b):
        with np.errstate(all="ignore"):
            r = y - np.exp(F @ b)
        v = float(np.dot(r, r))
        return v if np.isfinite(v) else np.inf
    cur = sse(beta)
    for _ in range(iters):
        with np.errstate(all="ignore"):
            f = np.exp(F @ beta)
        J = F * f[:, None]
        r = y - f
        JtJ = J.T @ J
        g = J.T @ r
        improved = False
        for _ in range(10):
            try:
                step = np.linalg.solve(JtJ + lam * np.diag(np.diag(JtJ) + 1e-12), g)
            except np.linalg.LinAlgError:
                return None
            nb = beta + step
            nv = sse(nb)
            if nv < cur:
                beta, cur, lam, improved = nb, nv, max(lam / 3, 1e-9), True
                break
            lam *= 10
        if not improved or np.max(np.abs(step)) < 1e-12:
            break
    with np.errstate(all="ignore"):
        f = np.exp(F @ beta)
    J = F * f[:, None]
    dof = max(len(y) - F.shape[1], 1)
    try:
        cov = (cur / dof) * np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        return None
    return beta, cov


def noisy_power_gate(X, y, names, seed=0, allow_free=False, max_rows=20000):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim != 2 or len(y) < 20:
        return None
    if len(y) > max_rows:
        idx = np.random.default_rng(seed).choice(len(y), max_rows, replace=False)
        X, y = X[idx], y[idx]
    # 1. one sign for y
    s = np.sign(np.median(y))
    if s == 0:
        return None
    keep = np.sign(y) == s
    if not keep.all():
        if keep.mean() < 0.90:
            return None
        if np.max(np.abs(y[~keep])) > np.percentile(np.abs(y[keep]), 25):
            return None
    X, y = X[keep], y[keep]
    # usable columns: one strict sign
    cols, xsign = [], []
    for j in range(X.shape[1]):
        c = X[:, j]
        if np.all(c > 0):
            cols.append(j); xsign.append(1)
        elif np.all(c < 0):
            cols.append(j); xsign.append(-1)
    if not cols:
        return None
    Xa = np.abs(X[:, cols])
    tr, va = _split(len(y), seed)
    # 2. weighted least squares in log space
    A = np.column_stack([np.ones(len(tr)), np.log(Xa[tr])])
    z = np.log(np.abs(y[tr]))
    w = y[tr] ** 2
    w = w / w.mean()
    sw = np.sqrt(w)
    coef0, *_ = np.linalg.lstsq(A * sw[:, None], z * sw, rcond=None)
    out = _nls(A, np.abs(y[tr]), coef0)
    if out is None:
        return None
    coef, cov = out
    se = np.sqrt(np.clip(np.diag(cov)[1:], 0, None))
    a = coef[1:]
    # 3. snap
    near = _FR_ARR[np.argmin(np.abs(a[:, None] - _FR_ARR[None, :]), axis=1)]
    ok = np.abs(a - near) <= K_SE * se + 1e-6
    snapped = bool(np.all(ok))
    if not snapped and not allow_free:
        return None
    expo = near if snapped else np.round(a, 3)
    # 4. C in y-space
    def fit_C(e):
        m = _mono(Xa[tr], e)
        den = float(np.dot(m, m))
        return float(np.dot(m, y[tr]) / den) if den > 0 else np.nan
    C = fit_C(expo)
    if not np.isfinite(C):
        return None
    # 5. held-out check vs free exponents
    def vmse(e, c):
        return float(np.mean((c * _mono(Xa[va], e) - y[va]) ** 2))
    mse = vmse(expo, C)
    mse_free = vmse(a, fit_C(a))
    tiny = 1e-12 * float(np.mean(y[va] ** 2))
    if snapped and mse > mse_free * (1 + SNAP_SLACK) + tiny:
        if not allow_free:
            return None
        expo, C, mse, snapped = np.round(a, 3), fit_C(np.round(a, 3)), vmse(np.round(a, 3), fit_C(np.round(a, 3))), False
    # lack-of-fit: flexible log-quadratic reference on the same split
    L = np.log(Xa)
    feats = [np.ones(len(y)), *L.T, *(L ** 2).T]
    feats += [L[:, i] * L[:, k] for i in range(L.shape[1]) for k in range(i + 1, L.shape[1])]
    F = np.column_stack(feats)
    if F.shape[1] < len(tr) // 5:
        cf0, *_ = np.linalg.lstsq(F[tr] * sw[:, None], z * sw, rcond=None)
        outf = _nls(F[tr], np.abs(y[tr]), cf0)
        cf = outf[0] if outf is not None else cf0
        with np.errstate(all="ignore"):
            g = s * np.exp(F @ cf)
        mse_flex = float(np.mean((g[va] - y[va]) ** 2))
        if np.isfinite(mse_flex) and mse > mse_flex * (1 + LOF_SLACK) + tiny:
            return None
    else:
        mse_flex = float("nan")
    yv = y[va]
    vy = float(np.mean((yv - yv.mean()) ** 2)) if len(yv) > 1 else 0.0
    r2 = 1 - mse / vy if vy > 0 else float("nan")
    # expression (sign of negative columns folded into C * (-1)^a only for integer a)
    parts, sign_fix = [], 1.0
    for e, j, sg in zip(expo, cols, xsign):
        if abs(e) < 1e-12:
            continue
        fr = Fraction(float(e)).limit_denominator(1000) if snapped else float(e)
        base = names[j] if sg > 0 else f"(-{names[j]})"
        parts.append(f"{base}**({fr})")
    expr = repr(float(C)) + ("*" + "*".join(parts) if parts else "")
    return dict(expr=expr, C=float(C), snapped=snapped,
                exponents={names[j]: (str(Fraction(float(e)).limit_denominator(1000)) if snapped else float(e))
                           for e, j in zip(expo, cols) if abs(e) > 1e-12},
                se={names[j]: float(sj) for sj, j in zip(se, cols)},
                val_mse=mse, val_mse_flex=mse_flex, val_r2=float(r2), n_rows=int(len(y)), seed=seed)


def gate_val_rows(n, seed=0, max_rows=20000):
    """Row indices (into the original y) the gate held out, for a fair baseline comparison."""
    return None  # caller compares on its own split; kept for API symmetry


if __name__ == "__main__":
    import time
    rng = np.random.default_rng(1)
    n = 2000
    X = rng.uniform(1, 5, (n, 4))
    x0, x1, x2, x3 = X.T
    cases = {
        "x0*x1/x2 (Feynman-like)": (x0 * x1 / x2, True),
        "x0*x1**(-1/3) (Fridericia-like)": (x0 * x1 ** (-1 / 3), True),
        "3.7*x0**2*sqrt(x1)/x3**3": (3.7 * x0 ** 2 * np.sqrt(x1) / x3 ** 3, True),
        "-x0**(3/4)*x2 (negative)": (-x0 ** 0.75 * x2, True),
        "x0+x1 (not monomial)": (x0 + x1, False),
        "x0/(x0+x1) (rational)": (x0 / (x0 + x1), False),
        "exp(-x0) (exponential)": (np.exp(-x0), False),
        "sin(x0)*x1 (sign change)": (np.sin(x0) * x1, False),
        "x0**0.342 (non-simple exp.)": (x0 ** 0.342, None),   # None: snapping to 1/3 is acceptable at high noise
    }
    fails = 0
    for noise in (0.0, 0.001, 0.01, 0.1):
        print(f"\n=== noise {noise} (sigma = noise * RMS(y), as SRBench) ===")
        for name, (y, should) in cases.items():
            yn = y + rng.normal(0, noise * np.sqrt(np.mean(y ** 2)), n)
            t = time.time(); res = noisy_power_gate(X, yn, ["x0", "x1", "x2", "x3"])
            got = res is not None
            # for monomials at high noise, "no form" is acceptable (too noisy to decide); a WRONG form is not
            if should is None:
                tag = "ok (ambiguous)" if (not got or noise >= 0.1) else "FAIL (over-snapped)"
                fails += tag.startswith("FAIL")
            elif got and not should:
                fails += 1; tag = "FAIL (false form)"
            elif got and should:
                tag = "ok"
            elif should and noise <= 0.01:
                fails += 1; tag = "FAIL (missed)"
            else:
                tag = "ok (no form)"
            print(f"  {tag:18s} {name:34s} -> {res['expr'] if got else None}  ({time.time()-t:.2f}s)")
    print(f"\nOVERALL: {'ALL PASSED' if fails == 0 else f'{fails} FAILURE(S)'}")
