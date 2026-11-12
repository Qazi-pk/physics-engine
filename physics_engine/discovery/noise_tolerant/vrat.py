"""vrat.py -- PIR 3.10 prototype V-rat: noise-tolerant version of the 3.6 rational gate.

y = P(x) / Q(x), P with <= 3 and Q with 2 or 3 monomials of total degree <= 2 (d <= 5 inputs).
Supports: exhaustive over (P, Q) supports with the linearized residual y*Q - P (smallest
eigenvector of the 2x2..6x6 Gram matrix, batched); the best supports are refined by
Gauss-Newton in y space; coefficient ratios snapped to p/q (q <= 6) within 3 SE; accepted on a
fixed 75/25 split if snapping costs < 2% held-out MSE and the held-out MSE is within 5% of a
flexible reference (polynomial of degree 3 in the standardized inputs, and the best free
rational found)."""
from fractions import Fraction
from itertools import combinations, combinations_with_replacement

import numpy as np
import sympy as sp

K_SE, SNAP_SLACK, LOF_SLACK, PARSIMONY = 3.0, 0.02, 0.05, 0.02
MAX_D, MAX_ROWS, TOP = 5, 20000, 30
R_FR = np.array(sorted({float(Fraction(p, q)) for q in range(1, 7) for p in range(-12 * q, 12 * q + 1)}))


def _split(n, seed=0):
    perm = np.random.default_rng(seed).permutation(n)
    c = int(0.75 * n)
    return np.sort(perm[:c]), np.sort(perm[c:])


def _basis(X, names):
    d = X.shape[1]
    cols, labs = [np.ones(len(X))], [()]
    for k in (1, 2):
        for idx in combinations_with_replacement(range(d), k):
            cols.append(np.prod(X[:, idx], axis=1)); labs.append(idx)
    return np.column_stack(cols), labs


def _gn(Pm, Qm, y, cp, cq, iters=40):
    """y = (Pm cp) / (Qm cq), cq[0] fixed to 1. Returns (cp, cq, cov of free params) or None."""
    def f(cp_, cq_):
        q = Qm @ cq_
        if np.any(np.abs(q) < 1e-12 * np.max(np.abs(q))) or np.any(np.sign(q) != np.sign(q[0])):
            return None, None
        return (Pm @ cp_) / q, q
    fv, q = f(cp, cq)
    if fv is None:
        return None
    sse = float(np.sum((fv - y) ** 2))
    for _ in range(iters):
        J = np.column_stack([Pm / q[:, None], -(fv / q)[:, None] * Qm[:, 1:]])
        step, *_ = np.linalg.lstsq(J, y - fv, rcond=None)
        lam = 1.0
        while lam > 1e-4:
            cp2 = cp + lam * step[:len(cp)]
            cq2 = np.r_[1.0, cq[1:] + lam * step[len(cp):]]
            f2, q2 = f(cp2, cq2)
            if f2 is not None:
                s2 = float(np.sum((f2 - y) ** 2))
                if s2 <= sse:
                    break
            lam /= 2
        if lam <= 1e-4:
            break
        conv = s2 > sse * (1 - 1e-10)
        cp, cq, fv, q, sse = cp2, cq2, f2, q2, s2
        if conv:
            break
    J = np.column_stack([Pm / q[:, None], -(fv / q)[:, None] * Qm[:, 1:]])
    cov = sse / max(len(y) - J.shape[1], 1) * np.linalg.pinv(J.T @ J)
    return cp, cq, cov


def v_rat(X, y, names, seed=0):
    X = np.asarray(X, float); y = np.asarray(y, float).ravel()
    if X.shape[1] > MAX_D:
        return None, {"decision": "too_many_inputs"}
    if len(y) > MAX_ROWS:
        k = np.sort(np.random.default_rng(seed).choice(len(y), MAX_ROWS, replace=False)); X, y = X[k], y[k]
    tr, va = _split(len(y), seed)
    Phi, labs = _basis(X, names)
    sc = np.sqrt(np.mean(Phi[tr] ** 2, axis=0)); sc[sc == 0] = 1
    Ph = Phi / sc
    m = Ph.shape[1]
    yt = y[tr]; ys = yt / np.sqrt(np.mean(yt ** 2))
    # Gram blocks for the linearized residual [y*phi_Q, -phi_P]
    A_q = Ph[tr] * ys[:, None]
    A_p = Ph[tr]
    G_qq, G_pp, G_qp = A_q.T @ A_q, A_p.T @ A_p, A_q.T @ A_p
    found = []
    for nq in (2, 3):
        Qs = list(combinations(range(m), nq))
        for npp in (1, 2, 3):
            Ps = list(combinations(range(m), npp))
            for qi in Qs:
                if 0 not in qi and nq == 2 and False:
                    pass
                Qi = np.array(qi)
                Pi = np.array(Ps)
                k = nq + npp
                G = np.zeros((len(Pi), k, k))
                G[:, :nq, :nq] = G_qq[np.ix_(Qi, Qi)]
                G[:, nq:, nq:] = G_pp[Pi[:, :, None], Pi[:, None, :]]
                G[:, :nq, nq:] = -G_qp[Qi][:, Pi].transpose(1, 0, 2)
                G[:, nq:, :nq] = np.transpose(G[:, :nq, nq:], (0, 2, 1))
                ev = np.linalg.eigvalsh(G)
                ratio = ev[:, 0] / (ev[:, 1] + 1e-300)
                for j in np.argsort(ev[:, 0])[:3]:
                    found.append((float(ev[j, 0]), float(ratio[j]), qi, tuple(Pi[j])))
    found.sort()
    cands = []
    seen = set()
    for _, _, qi, pi in found:
        if (qi, pi) in seen:
            continue
        seen.add((qi, pi))
        if len(seen) > TOP * 6:
            break
        Pm, Qm = Phi[:, list(pi)], Phi[:, list(qi)]
        # start: smallest eigenvector of the support's Gram
        A = np.column_stack([Qm[tr] * yt[:, None], -Pm[tr]])
        A = A / (np.sqrt(np.mean(A ** 2, axis=0)) + 1e-300)
        _, _, Vt = np.linalg.svd(A, full_matrices=False)
        v = Vt[-1] / (np.sqrt(np.mean(np.column_stack([Qm[tr] * yt[:, None], -Pm[tr]]) ** 2, axis=0)) + 1e-300)
        if abs(v[0]) < 1e-12:
            continue
        cq0, cp0 = v[:len(qi)] / v[0], v[len(qi):] / v[0]
        out = _gn(Pm[tr], Qm[tr], yt, cp0, cq0)
        if out is None:
            continue
        cp, cq, cov = out
        q = Qm[va] @ cq
        if np.any(np.sign(q) != np.sign(q[0])):
            continue
        mse = float(np.mean(((Pm[va] @ cp) / q - y[va]) ** 2))
        cands.append((mse, len(pi) + len(qi), qi, pi, cp, cq, cov))
    if not cands:
        return None, {"decision": "no_fit"}
    mmin = min(c[0] for c in cands)
    nb = min(c[1] for c in cands if c[0] <= mmin * (1 + PARSIMONY))
    mse, n_t, qi, pi, cp, cq, cov = min((c for c in cands if c[1] == nb), key=lambda c: c[0])
    # snap: all coefficients relative to the largest |cp| (scale is free)
    theta = np.r_[cp, cq[1:]]
    j = int(np.argmax(np.abs(cp)))
    # covariance of ratios theta/theta_j by the delta method (cq[0] = 1 is fixed)
    r = theta / theta[j]
    var = (np.diag(cov) - 2 * r * cov[:, j] + r ** 2 * cov[j, j]) / theta[j] ** 2
    se = np.sqrt(np.clip(var, 0, None))
    near = R_FR[np.argmin(np.abs(r[:, None] - R_FR[None, :]), axis=1)]
    info = {"P": [labs[i] for i in pi], "Q": [labs[i] for i in qi], "ratios": r.round(5).tolist(), "mse": mse, "mmin": mmin}
    rq0 = 1.0 / theta[j]
    nq0 = R_FR[np.argmin(np.abs(R_FR - rq0))]
    if not (np.all(np.abs(r - near) <= K_SE * se + 1e-9)):
        info["decision"] = "not_snapped"; return None, info
    cp_s = near[:len(cp)]; cq_s_free = near[len(cp):]
    # Q = cq0 + ...: cq0 relative = 1/theta_j; refit only the overall scale on that structure
    Pm, Qm = Phi[:, list(pi)], Phi[:, list(qi)]
    # with ratios fixed, y = C * (Pm cp_s) / (Qm [q0, cq_s_free]) where q0 is free: fit q0 and C by GN
    best = None
    for q0 in sorted({nq0, float(Fraction(rq0).limit_denominator(6))}):
        if q0 == 0:
            continue
        cq_s = np.r_[q0, cq_s_free]
        qv = Qm @ cq_s
        if np.any(np.sign(qv) != np.sign(qv[0])):
            continue
        g = (Pm @ cp_s) / qv
        C = float(g[tr] @ y[tr] / (g[tr] @ g[tr]))
        mse_s = float(np.mean((C * g[va] - y[va]) ** 2))
        if best is None or mse_s < best[0]:
            best = (mse_s, C, cq_s)
    if best is None:
        info["decision"] = "snapped_invalid"; return None, info
    mse_s, C, cq_s = best
    # reference: polynomial degree 3 in standardized inputs
    Z = (X - X[tr].mean(0)) / (X[tr].std(0) + 1e-300)
    cols = [np.ones(len(Z))]
    for k in (1, 2, 3):
        for idx in combinations_with_replacement(range(Z.shape[1]), k):
            cols.append(np.prod(Z[:, idx], axis=1))
    F = np.column_stack(cols)
    cf, *_ = np.linalg.lstsq(F[tr], y[tr], rcond=None)
    mse_ref = min(float(np.mean((F[va] @ cf - y[va]) ** 2)), mmin)
    info.update(mse_snapped=mse_s, mse_ref=mse_ref)
    if mse_s > mse * (1 + SNAP_SLACK) or mse_s > mse_ref * (1 + LOF_SLACK):
        info["decision"] = "rejected"; return None, info
    sy = [sp.Symbol(n) for n in names]
    mono = lambda idx: sp.Mul(*[sy[i] for i in idx])
    rat = lambda v: sp.Rational(Fraction(float(v)).limit_denominator(6))
    Pe = sum(rat(c) * mono(labs[i]) for c, i in zip(cp_s, pi))
    Qe = sum(rat(c) * mono(labs[i]) for c, i in zip(cq_s, qi))
    info["decision"] = "accepted"
    return {"expr": str(sp.Float(C, 10) * Pe / Qe), "val_mse": mse_s}, info
