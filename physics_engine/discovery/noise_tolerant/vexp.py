"""vexp.py -- PIR 3.10 prototype V-exp: noise-tolerant version of the 3.6 exp gate.

y = s * C * prod x_i**a_i * exp(sum_{k<=3} b_k * m_k(x)),  m_k from the exp gate's monomial pool.

Selection by forward steps in log space (weights y**2), every candidate refined by Gauss-Newton in
y space; a_i snapped to p/q (q <= 4), b_k to p/q (q <= 6), both within 3 SE (+ small floor);
accepted on a fixed 75/25 split if snapping costs < 2% held-out MSE and the held-out MSE is within
5% of a flexible reference s*exp(poly3(x) + sum c_i ln x_i).
"""
from fractions import Fraction

import numpy as np
import sympy as sp

from .vsep import _best_subsets
from physics_engine.discovery.exp_gate import _monomial_pool, pool_size_upper_bound, MAX_POOL_TERMS

K_SE, SNAP_SLACK, LOF_SLACK, PARSIMONY, EXP_FLOOR = 3.0, 0.02, 0.05, 0.02, 0.01
A_FR = np.array(sorted({float(Fraction(p, q)) for q in (1, 2, 3, 4) for p in range(-6 * q, 6 * q + 1)}))
B_FR = np.array(sorted({float(Fraction(p, q)) for q in range(1, 7) for p in range(-12 * q, 12 * q + 1)}))
MAX_ROWS, TOPN = 20000, 150
MEM_BYTES, MIN_SEL_ROWS = 4e8, 2000   # pool matrix budget (with its weighted copy) and least rows


def _split(n, seed=0):
    perm = np.random.default_rng(seed).permutation(n)
    c = int(0.75 * n)
    return np.sort(perm[:c]), np.sort(perm[c:])


def _gn(D, y, s, beta0, iters=40):
    """y = s * exp(D beta), Gauss-Newton with step halving. Returns beta, cov."""
    beta = np.array(beta0, float)
    with np.errstate(all="ignore"):
        f = s * np.exp(D @ beta)
    if not np.all(np.isfinite(f)):
        return None, None
    sse = float(np.sum((f - y) ** 2))
    for _ in range(iters):
        J = f[:, None] * D
        step, *_ = np.linalg.lstsq(J, y - f, rcond=None)
        lam = 1.0
        while lam > 1e-4:
            b2 = beta + lam * step
            with np.errstate(all="ignore"):
                f2 = s * np.exp(D @ b2)
            s2 = float(np.sum((f2 - y) ** 2))
            if np.isfinite(s2) and s2 <= sse:
                break
            lam /= 2
        if lam <= 1e-4:
            break
        conv = s2 > sse * (1 - 1e-10)
        beta, f, sse = b2, f2, s2
        if conv:
            break
    J = f[:, None] * D
    cov = sse / max(len(y) - D.shape[1], 1) * np.linalg.pinv(J.T @ J)
    return beta, cov


def _poly(Z, deg):
    from itertools import combinations_with_replacement
    cols = [np.ones(len(Z))]
    for k in range(1, deg + 1):
        for idx in combinations_with_replacement(range(Z.shape[1]), k):
            cols.append(np.prod(Z[:, idx], axis=1))
    return np.column_stack(cols)


def v_exp(X, y, names, seed=0):
    X = np.asarray(X, float); y = np.asarray(y, float).ravel()
    if len(y) > MAX_ROWS:
        k = np.sort(np.random.default_rng(seed).choice(len(y), MAX_ROWS, replace=False)); X, y = X[k], y[k]
    d = X.shape[1]
    if pool_size_upper_bound(d) > MAX_POOL_TERMS:
        return None, {"decision": "pool_too_large"}
    s = float(np.sign(np.median(y)))
    good = np.sign(y) == s
    if good.mean() < 0.5:          # small targets may flip sign under noise; log fit uses main-sign rows
        return None, {"decision": "not_one_sign"}
    pos = [j for j in range(d) if np.all(X[:, j] > 0)]
    nonzero = [bool(np.all(X[:, j] != 0)) for j in range(d)]
    tr, va = _split(len(y), seed)
    trg = tr[good[tr]]
    L = np.log(X[:, pos]) if pos else np.zeros((len(y), 0))
    base = np.column_stack([np.ones(len(y)), L])
    # memory guard: the term pool is built only on a seeded subset of the training rows, sized so
    # that the pool matrix stays below MEM_BYTES; chosen terms are then evaluated on all rows
    bound = max(pool_size_upper_bound(d), 1)
    m_sel = min(len(trg), int(MEM_BYTES / (16.0 * bound)))
    if m_sel < min(MIN_SEL_ROWS, len(trg)):
        return None, {"decision": "pool_too_large_for_memory"}
    sel = trg if m_sel >= len(trg) else np.sort(np.random.default_rng(seed).choice(trg, m_sel, replace=False))
    cols, labels = _monomial_pool(X[sel], names, nonzero)
    if not cols:
        return None, {"decision": "empty_pool"}
    P = np.column_stack(cols); del cols

    def col(j):
        return np.prod([X[:, i] ** pw for i, pw in labels[j]], axis=0) * np.ones(len(y))

    # weighted log fit on the selection rows
    z = np.log(np.abs(y[sel])); w_s = np.sqrt(y[sel] ** 2 / np.mean(y[sel] ** 2))
    Bw, Pw, zw = base[sel] * w_s[:, None], P * w_s[:, None], z * w_s
    del P
    # project the power-law part out, keep the TOPN pool columns most correlated with the rest,
    # then exhaustive best subsets of 1..3 of them (Gram matrix), each refined in y space
    Qb, _ = np.linalg.qr(Bw)
    r = zw - Qb @ (Qb.T @ zw)
    Po = Pw - Qb @ (Qb.T @ Pw)
    nrm = np.linalg.norm(Po, axis=0)
    score = np.where(nrm > 1e-10 * (np.linalg.norm(Pw, axis=0) + 1e-300), np.abs(Po.T @ r) / (nrm + 1e-300), 0)
    top = np.argsort(-score)[:TOPN]
    subsets = {0: [()]}
    sub = _best_subsets(Po[:, top], r, kmax=3, top=5)
    for kk, lst in sub.items():
        subsets[kk] = [tuple(int(top[i]) for i in idx) for idx in lst]
    del Pw, Po
    w = np.sqrt(y[trg] ** 2 / np.mean(y[trg] ** 2)); zt = np.log(np.abs(y[trg])) * w
    cache = {}
    cands = []
    for k, lst in subsets.items():
        for chosen in lst:
            for j in chosen:
                if j not in cache:
                    cache[j] = col(j)
            D_all = np.column_stack([base] + [cache[j] for j in chosen])
            c0, *_ = np.linalg.lstsq(D_all[trg] * w[:, None], zt, rcond=None)
            beta, cov = _gn(D_all[tr], y[tr], s, c0)
            if beta is None:
                continue
            with np.errstate(all="ignore"):
                mse = float(np.mean((s * np.exp(D_all[va] @ beta) - y[va]) ** 2))
            if np.isfinite(mse):
                cands.append((mse, k, list(chosen), beta, cov))
    if not cands:
        return None, {"decision": "no_fit"}
    mmin = min(c[0] for c in cands)
    kb = min(c[1] for c in cands if c[0] <= mmin * (1 + PARSIMONY))
    mse, k, ch, beta, cov = min((c for c in cands if c[1] == kb), key=lambda c: c[0])
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    na = len(pos)
    a, b = beta[1:1 + na], beta[1 + na:]
    a_s = A_FR[np.argmin(np.abs(a[:, None] - A_FR[None, :]), axis=1)] if na else a
    b_s = B_FR[np.argmin(np.abs(b[:, None] - B_FR[None, :]), axis=1)] if len(b) else b
    info = {"k": k, "terms": [labels[j] for j in ch], "a": a.round(4).tolist(), "b": b.round(4).tolist(), "mse": mse}
    ok_a = np.all(np.abs(a - a_s) <= K_SE * se[1:1 + na] + EXP_FLOOR) if na else True
    ok_b = np.all(np.abs(b - b_s) <= K_SE * se[1 + na:] + 1e-3 * np.abs(b)) if len(b) else True
    if not (ok_a and ok_b):
        info["decision"] = "not_snapped"; return None, info
    D_all = np.column_stack([base] + [cache[j] for j in ch])
    off = D_all[:, 1:] @ np.r_[a_s, b_s]
    # refit ln C only (one parameter): y = s*exp(lnC + off)
    g = s * np.exp(off)
    C = float(g[tr] @ y[tr] / (g[tr] @ g[tr]))
    if not C > 0:
        info["decision"] = "bad_C"; return None, info
    mse_s = float(np.mean((C * g[va] - y[va]) ** 2))
    # flexible reference
    Zs = (X - X[tr].mean(0)) / (X[tr].std(0) + 1e-300)
    F = np.column_stack([_poly(Zs, 3 if d <= 3 else 2), L])
    cf0, *_ = np.linalg.lstsq(F[trg] * w[:, None], zt, rcond=None)
    cf, _ = _gn(F[tr], y[tr], s, cf0)
    with np.errstate(all="ignore"):
        mse_ref = float(np.mean((s * np.exp(F[va] @ cf) - y[va]) ** 2)) if cf is not None else float("inf")
    info.update(mse_snapped=mse_s, mse_ref=mse_ref)
    if mse_s > mse * (1 + SNAP_SLACK) or not mse_s <= mse_ref * (1 + LOF_SLACK):
        info["decision"] = "rejected"; return None, info
    sy = [sp.Symbol(n) for n in names]
    expo = sum(sp.Rational(Fraction(float(bk)).limit_denominator(6)) * sp.Mul(*[sy[i] ** p for i, p in labels[j]])
               for bk, j in zip(b_s, ch))
    mono = sp.Mul(*[sy[pos[i]] ** sp.Rational(Fraction(float(ai)).limit_denominator(4)) for i, ai in enumerate(a_s) if ai != 0])
    if not ch and any(Fraction(float(ai)).limit_denominator(4).denominator == 2 for ai in a_s):
        # pure power law with half-integer exponents: one root, e.g. sqrt(gamma*pr/rho)
        expr = sp.Float(s * C, 10) * sp.sqrt(sp.Mul(*[sy[pos[i]] ** sp.Rational(Fraction(2 * float(ai)).limit_denominator(4))
                                                       for i, ai in enumerate(a_s) if ai != 0]))
    else:
        expr = sp.Float(s * C, 10) * mono * sp.exp(sp.factor(expo))   # e.g. -(theta - theta1)**2/(2*sigma**2)
    info["decision"] = "accepted"
    return {"expr": str(expr), "val_mse": mse_s}, info
