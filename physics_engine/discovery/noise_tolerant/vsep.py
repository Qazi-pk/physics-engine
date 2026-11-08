"""vsep.py -- PIR 3.10 prototype V-sep: noise-tolerant separability.

y = C * M(A) * h(B),  M(A) = prod a**e_a (monomial, exponents snapped),
                      h(B)**p = sum of <= 3 library atoms (coefficient ratios snapped).

Acceptance is NOT exactness. On a fixed 75/25 split of the training rows:
  * snapping must not cost fit (held-out MSE <= (1 + SNAP_SLACK) * unsnapped);
  * lack of fit: held-out MSE <= (1 + LOF_SLACK) * that of a rich reference
    C * M(A) * poly_deg(B) (same monomial, flexible h);
  * the caller compares with the engine on the same held-out rows.
"""
from fractions import Fraction
from itertools import combinations

import numpy as np
import pandas as pd
import sympy as sp

from physics_engine.discovery.separability.signaware import block_split, crossterm_search
from physics_engine.discovery.separability.exact import _power_atoms, _trig_atoms, _evaluate

K_SE, SNAP_SLACK, LOF_SLACK, PARSIMONY = 3.0, 0.02, 0.05, 0.02
NONLIN_FRAC = 0.01   # additive-log split: a variable is non-power if its curvature part > 1% of std(log|y|)
EXP_FLOOR = 0.01   # approximation error of the flexible h surrogate; exponent grid spacing is >= 1/12
MAX_ATTEMPTS, MAX_ROWS = 4, 20000
EXP_FR = sorted({Fraction(p, q) for q in (1, 2, 3, 4) for p in range(-6 * q, 6 * q + 1)})
EXP_ARR = np.array([float(f) for f in EXP_FR])
RAT_FR = sorted({Fraction(p, q) for q in range(1, 7) for p in range(-12 * q, 12 * q + 1) if p != 0})
RAT_ARR = np.array([float(f) for f in RAT_FR])


def _split(n, seed=0):
    perm = np.random.default_rng(seed).permutation(n)
    c = int(0.75 * n)
    return np.sort(perm[:c]), np.sort(perm[c:])


def _poly_basis(Z, deg):
    """All monomials of the standardized columns of Z up to total degree deg (incl. 1)."""
    n, d = Z.shape
    cols = [np.ones(n)]
    from itertools import combinations_with_replacement
    for k in range(1, deg + 1):
        for idx in combinations_with_replacement(range(d), k):
            cols.append(np.prod(Z[:, idx], axis=1))
    return np.column_stack(cols)


def _std(M, ref):
    mu, sd = ref.mean(axis=0), ref.std(axis=0)
    sd[sd == 0] = 1
    return (M - mu) / sd


def _monomial_exponents(df, A, B, y, tr):
    """Weighted LS of log|y| on log a (a in A) + flexible poly(B); weights y**2. Snap within K_SE."""
    LA = np.log(df[A].to_numpy(dtype=float))
    ZB = _std(df[B].to_numpy(dtype=float), df[B].to_numpy(dtype=float)[tr])
    F = _poly_basis(ZB, 3 if len(B) <= 4 else 2)
    D = np.column_stack([LA, F])[tr]
    z = np.log(np.abs(y[tr]) + 1e-300)
    w = y[tr] ** 2; w = w / w.mean(); sw = np.sqrt(w)   # var(log|y|) ~ sigma^2 / y^2
    coef, *_ = np.linalg.lstsq(D * sw[:, None], z * sw, rcond=None)
    res = (D @ coef - z) * sw
    dof = max(len(z) - D.shape[1], 1)
    s2 = float(res @ res) / dof
    try:
        cov = s2 * np.linalg.pinv((D * sw[:, None]).T @ (D * sw[:, None]))
    except np.linalg.LinAlgError:
        return None
    e = coef[:len(A)]
    se = np.sqrt(np.clip(np.diag(cov)[:len(A)], 0, None))
    near = EXP_ARR[np.argmin(np.abs(e[:, None] - EXP_ARR[None, :]), axis=1)]
    return near, e, se


def _monomial_nls(df, A, B, y, tr, va, iters=25):
    """y = prod a**e_a * (F(B) beta), F = polynomial basis of the standardized B columns, fitted by
    Gauss-Newton in y space (unbiased for additive noise), started from the weighted log fit.
    Returns (snapped e or None, e, SE(e), held-out MSE of this rich model = the LOF reference)."""
    _, e0, _ = _monomial_exponents(df, A, B, y, tr)
    LA = np.log(df[A].to_numpy(dtype=float))
    ZB = _std(df[B].to_numpy(dtype=float), df[B].to_numpy(dtype=float)[tr])
    deg = {1: 6, 2: 5, 3: 4, 4: 3}.get(len(B), 3 if len(B) <= 6 else 2)
    F = _poly_basis(ZB, deg)
    Ft, yt, Lt = F[tr], y[tr], LA[tr]

    def fit_beta(e_):
        M_ = np.exp(Lt @ e_)
        b_, *_ = np.linalg.lstsq(Ft * M_[:, None], yt, rcond=None)
        return b_

    e = np.array(e0, dtype=float)
    b = fit_beta(e)
    f = np.exp(Lt @ e) * (Ft @ b)
    sse = float(np.sum((f - yt) ** 2))
    for _ in range(iters):
        M_ = np.exp(Lt @ e)
        J = np.column_stack([f[:, None] * Lt, Ft * M_[:, None]])
        step, *_ = np.linalg.lstsq(J, yt - f, rcond=None)
        lam = 1.0
        while lam > 1e-4:
            e2, b2 = e + lam * step[:len(A)], b + lam * step[len(A):]
            f2 = np.exp(Lt @ e2) * (Ft @ b2)
            s2_ = float(np.sum((f2 - yt) ** 2))
            if np.isfinite(s2_) and s2_ <= sse:
                break
            lam /= 2
        if lam <= 1e-4:
            break
        conv = s2_ > sse * (1 - 1e-10)
        e, b, f, sse = e2, b2, f2, s2_
        if conv:
            break
    M_ = np.exp(Lt @ e)
    J = np.column_stack([f[:, None] * Lt, Ft * M_[:, None]])
    s2 = sse / max(len(yt) - J.shape[1], 1)
    cov = s2 * np.linalg.pinv(J.T @ J)
    se = np.sqrt(np.clip(np.diag(cov)[:len(A)], 0, None))
    fv = np.exp(LA[va] @ e) * (F[va] @ b)
    mse_ref = float(np.mean((fv - y[va]) ** 2))
    near = EXP_ARR[np.argmin(np.abs(e[:, None] - EXP_ARR[None, :]), axis=1)]
    ok = bool(np.all(np.abs(e - near) <= K_SE * se + EXP_FLOOR))
    return (near if ok else None), e, se, mse_ref


def _best_subsets(Phi, t, kmax=3, top=5):
    """Exhaustive best subsets (sizes 1..kmax) by training SSE; vectorized per size."""
    sc = np.sqrt((Phi ** 2).mean(axis=0)); sc[sc == 0] = 1
    P = Phi / sc
    G = P.T @ P; b = P.T @ t; tt = float(t @ t)
    out = {}
    m = P.shape[1]
    for k in range(1, kmax + 1):
        idx = np.array(list(combinations(range(m), k)))
        if len(idx) == 0:
            continue
        best = []
        for s in range(0, len(idx), 20000):
            I = idx[s:s + 20000]
            Gs = G[I[:, :, None], I[:, None, :]] + 1e-12 * np.eye(k)
            bs = b[I]
            try:
                x = np.linalg.solve(Gs, bs[..., None])[..., 0]
            except np.linalg.LinAlgError:
                continue
            sse = tt - np.einsum("ij,ij->i", bs, x)
            o = np.argsort(sse)[:top]
            best += [(float(sse[j]), tuple(I[j])) for j in o]
        best.sort()
        out[k] = [i for _, i in best[:top]]
    return out


def _nls_w(P, M, y, w0, p, s_out, iters=30):
    """y = s_out * M * (P w)**(1/p); Gauss-Newton with step halving. Returns (w, SE(w))."""
    w = np.array(w0, dtype=float)

    def pred(w_):
        Q = P @ w_
        if p != 1 and np.any(Q <= 0):
            return None, None
        with np.errstate(all="ignore"):
            hq = Q ** (1.0 / p) if p != 1 else Q
        return s_out * M * hq, Q

    f, Q = pred(w)
    if f is None:
        return None, None
    sse = float(np.sum((f - y) ** 2))
    for _ in range(iters):
        J = (s_out * M * (1.0 / p) * np.abs(Q) ** (1.0 / p - 1))[:, None] * P if p != 1 else M[:, None] * P
        step, *_ = np.linalg.lstsq(J, y - f, rcond=None)
        lam, done = 1.0, False
        while lam > 1e-4:
            f2, Q2 = pred(w + lam * step)
            if f2 is not None:
                s2_ = float(np.sum((f2 - y) ** 2))
                if s2_ <= sse:
                    w, f, Q, done = w + lam * step, f2, Q2, s2_ > sse * (1 - 1e-12)
                    sse = s2_
                    break
            lam /= 2
        if lam <= 1e-4 or done:
            break
    J = (s_out * M * (1.0 / p) * np.abs(Q) ** (1.0 / p - 1))[:, None] * P if p != 1 else M[:, None] * P
    s2 = sse / max(len(y) - len(w), 1)
    cov = s2 * np.linalg.pinv(J.T @ J)
    return w, cov


def _num(expr, df):
    ss = sorted(expr.free_symbols, key=str)
    f = sp.lambdify(ss, expr, "numpy")
    return np.asarray(f(*[df[str(s)].to_numpy(dtype=float) for s in ss]), dtype=float) * np.ones(len(df))


def _readable(plain, Cf, Msym, terms, pr, df, target):
    """Equivalent forms for positive inputs that SRBench's scorer can simplify (it cannot use
    positivity): (1) with two terms under a root, one dominant on every row, pull the dominant
    monomial out: Q = T_d (1 + T_o/T_d); (2) merge roots and powers (powsimp, force). Every form
    is re-verified numerically against the plain form; the first that matches is returned."""
    pos = {s: sp.Symbol(str(s), positive=True) for s in plain.free_symbols}
    back = {v: k for k, v in pos.items()}
    forms = []
    if pr != 1 and len(terms) == 2:
        v = [_num(tm, df) for tm in terms]
        for i, j in ((0, 1), (1, 0)):
            if np.all(np.abs(v[j]) < np.abs(v[i])):
                Td, To = terms[i].xreplace(pos), terms[j].xreplace(pos)
                h = sp.powsimp(sp.expand_power_base(Td ** (1 / pr), force=True), force=True) * \
                    (1 + sp.powsimp(To / Td, force=True)) ** (1 / pr)
                forms.append(("dominant_out", Cf * sp.powsimp(Msym.xreplace(pos) * h, force=True)))
                break
    if len(terms) == 1 and abs(pr) == 2:
        # one root over everything: (M**2 * T**sign(p))**(1/2), e.g. sqrt(gamma*pr/rho)
        T = terms[0] if pr > 0 else 1 / terms[0]
        inner = sp.powsimp(Msym ** 2 * T, force=True)
        ex = inner.as_powers_dict()
        if all(sp.Integer(2) * (e_ / 2) == e_ and (e_ / 2).is_integer for e_ in ex.values()):
            # the root is exact (all exponents even): write the plain monomial, e.g. k_spring*x**2,
            # never sqrt(k_spring**2*x**4), which the scorer cannot simplify without positivity
            forms.insert(0, ("monomial", Cf * sp.Mul(*[b ** (e_ / 2) for b, e_ in ex.items()])))
        else:
            forms.insert(0, ("single_root", Cf * sp.sqrt(inner)))
    if pr < 0 and len(terms) == 1:
        # single term under a negative power: write (1/T)**(1/|p|), e.g. sqrt(gamma*pr/rho)
        forms.append(("inverted_single", Cf * Msym * (1 / terms[0]) ** (-1 / pr)))
    if pr != 1:
        ex = Msym.as_powers_dict()
        inner = [b for b, e_ in ex.items() if e_ == 1 / pr]
        if inner:
            rest = sp.Mul(*[b ** e_ for b, e_ in ex.items() if e_ != 1 / pr])
            forms.append(("merged_root", Cf * rest * (sp.Mul(*inner) * sp.Add(*terms)) ** (1 / pr)))
        forms.append(("merged", sp.powsimp(Cf * Msym.xreplace(pos) * sp.Add(*terms).xreplace(pos) ** (1 / pr), force=True)))
    ref = _num(plain, df)
    for name, f in forms:
        f = f.xreplace(back)
        try:
            if np.max(np.abs(_num(f, df) - ref)) <= 1e-9 * np.max(np.abs(ref)):
                return str(f), name
        except Exception:
            continue
    return str(plain), "plain"


def _snap_ratios(c, cov):
    """Ratios c_k / c_j (c_j = largest |c|) to simple rationals within K_SE * SE (delta method)."""
    j = int(np.argmax(np.abs(c)))
    r = c / c[j]
    var = (np.diag(cov) - 2 * r * cov[:, j] + r ** 2 * cov[j, j]) / c[j] ** 2
    rse = np.sqrt(np.clip(var, 0, None))
    out = np.array(r, dtype=float)
    fr = [None] * len(c)
    for k in range(len(c)):
        if k == j:
            fr[k] = Fraction(1); out[k] = 1.0; continue
        near = RAT_ARR[np.argmin(np.abs(r[k] - RAT_ARR))]
        if abs(r[k] - near) <= K_SE * rse[k] + 1e-9:
            out[k] = near; fr[k] = Fraction(near).limit_denominator(6)
        else:
            return None
    s = 1.0 if c[j] > 0 else -1.0
    return out * s, [f * int(s) for f in fr], j


def _eval_h(Phi_cols, w, p):
    Q = Phi_cols @ w
    if p == 1:
        return Q
    if np.any(Q <= 0) and float(p) != int(p):
        return None
    with np.errstate(all="ignore"):
        h = np.sign(Q) * np.abs(Q) ** (1.0 / p) if float(p) == int(p) and int(p) % 2 else Q ** (1.0 / p)
    return h if np.all(np.isfinite(h)) else None


def _additive_log_split(df, y, tr, bs):
    """If y keeps one sign: log|y| = sum_v (e_v L_v + q_v L_v^2 + c_v L_v^3) over variables that are
    not in a multi-variable block. Variables with q_v or c_v beyond K_SE standard errors (or in a
    block) go to B (the h side); the rest, power-law variables, to A (the monomial side)."""
    s = float(np.sign(np.median(y)))
    good = np.sign(y) == s
    if good.mean() < 0.9:          # not (mostly) one-signed; small noisy rows may flip sign
        return None
    tr = tr[good[tr]]
    names = list(df.columns)
    blockvars = set(bs[1]) if bs is not None else set()
    L = np.log(df[names].to_numpy(dtype=float))
    Lz = _std(L, L[tr])
    cols = [np.ones(len(y))]
    for j in range(len(names)):
        cols += [Lz[:, j], Lz[:, j] ** 2, Lz[:, j] ** 3]
    sw = np.abs(y[tr]) / np.sqrt(np.mean(y[tr] ** 2))
    D = np.column_stack(cols)[tr] * sw[:, None]
    z = np.log(np.abs(y[tr])) * sw
    coef, *_ = np.linalg.lstsq(D, z, rcond=None)
    res = D @ coef - z
    s2 = float(res @ res) / max(len(z) - D.shape[1], 1)
    se = np.sqrt(np.clip(np.diag(s2 * np.linalg.pinv(D.T @ D)), 0, None))
    A, B = [], []
    zs = float(np.std(np.log(np.abs(y[tr]))))
    for j, nm in enumerate(names):
        q, c = coef[2 + 3 * j], coef[3 + 3 * j]
        nl = float(np.std(q * Lz[tr, j] ** 2 + c * Lz[tr, j] ** 3))
        nonlin = nl > NONLIN_FRAC * zs
        (B if (nonlin or nm in blockvars) else A).append(nm)
    if not A or not B or len(B) > 4:
        return None
    return A, B


def v_sep(X, y, names, seed=0, verbose=False):
    df = pd.DataFrame(np.asarray(X, dtype=float), columns=names)
    y = np.asarray(y, dtype=float).ravel()
    if len(y) > MAX_ROWS:
        keep = np.sort(np.random.default_rng(seed).choice(len(y), MAX_ROWS, replace=False))
        df, y = df.iloc[keep].reset_index(drop=True), y[keep]
    if not all(np.all(df[c] > 0) for c in names):
        return None, {"decision": "nonpositive_input"}
    # constant or duplicated columns carry no structure and make the fits singular: leave them out
    keep_c, seen = [], set()
    for c in names:
        v = df[c].to_numpy()
        key = v.tobytes()
        if np.ptp(v) > 0 and key not in seen:
            keep_c.append(c); seen.add(key)
    if len(keep_c) < 2:
        return None, {"decision": "too_few_varying_inputs"}
    df = df[keep_c]
    tr, va = _split(len(y), seed)
    d2 = df.copy(); d2["target"] = y
    _, ranked = crossterm_search(d2.iloc[tr].reset_index(drop=True), target="target")
    cands = [(list(r[0]), list(r[1])) for r in ranked if r[2] == "MULT"][:8]
    bs = block_split(d2.iloc[tr].reset_index(drop=True), target="target")
    if bs is not None:
        bs = (list(bs[0]), list(bs[1]))
        cands = [bs] + [c for c in cands if set(c[0]) != set(bs[0]) and set(c[1]) != set(bs[0])]
    al = _additive_log_split(df, y, tr, bs)
    if al is not None:
        cands = [al] + [c for c in cands if set(c[0]) != set(al[0])]
    info = {"cands": [" | ".join("+".join(g) for g in c) for c in cands], "tried": []}
    best = None
    att = 0
    for A0, B0 in cands:
        for A, B in ((A0, B0), (B0, A0)):
            if att >= MAX_ATTEMPTS:
                break
            att += 1
            r = _try(df, y, A, B, tr, va, verbose)
            info["tried"].append({"A": "+".join(A), "B": "+".join(B), **{k: v for k, v in (r or {}).items() if k != "fn"}})
    refs = [r["mse_ref"] for r in info["tried"] if "mse_ref" in r]
    ref = min(refs) if refs else float("inf")
    info["mse_ref_global"] = ref
    for r in info["tried"]:
        if r.get("ok_snap") and "val_mse" in r:
            r["ok_lof"] = r["val_mse"] <= ref * (1 + LOF_SLACK) + 1e-30
            r["accepted"] = bool(r["ok_lof"])
            if r["accepted"] and (best is None or r["val_mse"] < best["val_mse"]):
                best = r
    info["decision"] = "accepted" if best else "none"
    return best, info


def _try(df, y, A, B, tr, va, verbose):
    e, e_raw, e_se, mse_ref = _monomial_nls(df, A, B, y, tr, va)
    if e_raw is None:
        return {"stage": "monomial_fit_failed"}
    if e is None:
        return {"stage": "monomial_not_snapped", "mse_ref": mse_ref,
                "e_raw": [round(float(x), 4) for x in e_raw], "e_se": [float(x) for x in e_se]}
    M = np.prod(df[A].to_numpy(dtype=float) ** e, axis=1)
    r = y / M
    syms = {c: sp.Symbol(c) for c in df.columns}
    atoms = [sp.Integer(1)]
    if len(B) <= 3:
        atoms += _power_atoms(B, df)[1]
    if len(B) <= 4:
        atoms += _trig_atoms(B, df)[1]
    seen, keep, cols = set(), [], []
    sub = df[B]
    for a in atoms:
        if a in seen:
            continue
        seen.add(a)
        v = np.asarray(_evaluate(a, sub), dtype=float) * np.ones(len(sub))
        if np.all(np.isfinite(v)) and (np.std(v) > 0 or a == 1):
            keep.append(a); cols.append(v)
    if not keep:
        return {"stage": "no_atoms"}
    Phi = np.column_stack(cols)
    sg = float(np.sign(np.median(r)))
    pos = sg * r > 0
    # with noise, small targets can flip sign: allow roots/powers if <= 10% of rows flip and all
    # flipped rows are small; such rows are left out of the t-space start only (NLS uses all rows)
    # roots/powers of h: start values from the rows where r has its main sign (noise can flip
    # small targets); the y-space NLS then uses all rows, whatever their sign
    one_sign = bool(pos.mean() >= 0.5)
    trp = tr[pos[tr]]
    cands = []
    for p in (1, -1, 2, -2, 0.5, -0.5):
        if p != 1 and not one_sign:
            continue
        rows = tr if p == 1 else trp
        with np.errstate(all="ignore"):
            t = (sg * r) ** p if p != 1 else r
        if not np.all(np.isfinite(t[rows])):
            continue
        subsets = _best_subsets(Phi[rows], t[rows])
        s_out = sg if p != 1 else 1.0
        for k, lst in subsets.items():
            for idx in lst:
                P = Phi[:, idx]
                w0, *_ = np.linalg.lstsq(P[rows], t[rows], rcond=None)
                if _eval_h(P[tr], w0, p) is None:
                    continue
                # refine in y space (unbiased for additive noise); covariance from the Jacobian
                w, cov = _nls_w(P[tr], M[tr], y[tr], w0, p, s_out)
                if w is None:
                    continue
                h = _eval_h(P, w, p)
                if h is None:
                    continue
                mse = float(np.mean((M[va] * s_out * h[va] - y[va]) ** 2))
                cands.append((mse, k, p, idx, w, cov))
    if not cands:
        return {"stage": "no_h", "mse_ref": mse_ref}
    mmin = min(c[0] for c in cands)
    # parsimony: smallest k within PARSIMONY of the best; then best mse at that k
    kbest = min(c[1] for c in cands if c[0] <= mmin * (1 + PARSIMONY))
    mse, k, p, idx, w, se = min((c for c in cands if c[1] == kbest), key=lambda c: c[0])
    P = Phi[:, idx]
    sn = _snap_ratios(w, se)  # se is the covariance of w
    out = {"e": [str(Fraction(float(x)).limit_denominator(4)) for x in e], "p": p, "k": k,
           "atoms": [str(keep[i]) for i in idx], "w": [float(x) for x in w],
           "mse_unsnapped": mse, "mse_ref": mse_ref}
    if sn is None:
        out["stage"] = "ratios_not_snapped"
        return out
    ws, fr, j = sn
    h = _eval_h(P, ws, p)
    if h is None:
        out["stage"] = "snapped_invalid"; return out
    base = M * (sg if p != 1 else 1.0) * h
    C = float(base[tr] @ y[tr] / (base[tr] @ base[tr]))
    mse_s = float(np.mean((C * base[va] - y[va]) ** 2))
    out.update(val_mse=mse_s, ratios=[str(f) for f in fr])
    ok_snap = mse_s <= mse * (1 + SNAP_SLACK) + 1e-30
    out.update(ok_snap=ok_snap)
    # expression
    Msym = sp.Mul(*[sp.Symbol(a) ** sp.Rational(Fraction(float(x)).limit_denominator(4).numerator,
                                                 Fraction(float(x)).limit_denominator(4).denominator)
                    for a, x in zip(A, e) if x != 0])
    Q = sum(sp.Rational(f.numerator, f.denominator) * keep[i] for f, i in zip(fr, idx))
    pr = sp.Rational(Fraction(p).limit_denominator(2).numerator, Fraction(p).limit_denominator(2).denominator)
    # scale: h uses ws (ratios with w_j -> 1); C already absorbs everything
    Cf = sp.Float(C * (sg if p != 1 else 1.0), 10)
    # common factors out of a sum, e.g. m1/r2 - m1/r1 -> m1*(1/r2 - 1/r1)
    Q = sp.factor_terms(Q) if pr == 1 else Q
    plain = Cf * Msym * Q ** (1 / pr)
    target = C * base
    out["expr"], out["form"] = _readable(plain, Cf, Msym, [sp.Rational(f.numerator, f.denominator) * keep[i] for f, i in zip(fr, idx)],
                                         pr, df, target)
    out["stage"] = "done"
    return out
