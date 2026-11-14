"""vcomp.py -- PIR 3.11 prototype W: the composition family.

    y = C * prod_i x_i**a_i * g(k * u),   u = prod_j x_j**p_j  (a monomial from the exp gate's pool),

g from a fixed library of one-variable functions (exp(t), 1/(exp(t)-1), 1/(exp(t)+1),
1/(exp(t)+exp(-t)), tanh, sin, cos, sin**2, cos**2, arcsin, arccos, arctan, log, (1+t)**p for
p in {1, 2, -1, -2, 1/2, -1/2}), k a scale inside g. The 3.6 exp gate is the special case g = exp;
the relativistic factor 1/sqrt(1 - v**2/c**2) is g = (1+t)**(-1/2) with k = -1, u = v**2/c**2.

Procedure (one procedure, two acceptance rules):
  1. Screen (log space, weights |y|/rms, a seeded row subset): for every pool monomial u, every
     g and a fixed grid of k (scaled to u's median or maximum), regress log|y| - log|g(k u)| on
     [1, log x_i]; keep the best (g, u) pairs.
  2. Refine each in y space (least squares over C, a_i, k), on the 75% training rows of the
     fixed split; standard errors from the Jacobian.
  3. Snap a_i to p/q (q <= 4) within 3 SE + 0.01; k to p/q (q <= 6) within 3 SE + 0.1% |k|,
     otherwise k stays a fitted number (written with 10 significant digits). Refit C.
  4. Exact use (noise 0): accepted only if max scaled error < 1e-6 (checked again by the
     regressor on all rows). Noisy use (inside V): V6 lack of fit (snapped <= 1.02 x unsnapped,
     <= 1.05 x the best flexible reference), plus W4: held-out R**2 >= 0.5 (a library function
     fitted to pure noise can pass the lack-of-fit test by chance; a law explains the data), then
     the V7 replacement rules as for the other families.
Positive inputs only (logs); u must be positive on all rows.
"""
from fractions import Fraction

import numpy as np
import sympy as sp
from scipy.optimize import least_squares

from physics_engine.discovery.exp_gate import _monomial_pool, pool_size_upper_bound
from .vexp import _split, _poly

MAX_ROWS, SCREEN_ROWS, MAX_POOL = 20000, 1500, 5000
TOP_PAIRS, N_GRID = 12, 33
K_SE, SNAP_SLACK, LOF_SLACK, EXP_FLOOR, K_FLOOR = 3.0, 0.02, 0.05, 0.01, 1e-3
MIN_R2 = 0.5   # W4: a composition form must explain at least half of the held-out variance
EXACT_TOL = 1e-6
A_FR = sorted({Fraction(p, q) for q in (1, 2, 3, 4) for p in range(-6 * q, 6 * q + 1)})
A_ARR = np.array([float(f) for f in A_FR])
K_FR = sorted({Fraction(p, q) for q in range(1, 7) for p in range(-12 * q, 12 * q + 1) if p != 0})
K_ARR = np.array([float(f) for f in K_FR])


def _lib():
    """name -> (numpy g(t), sympy g(t), k-grid kind). Grid kinds: 'scale+' k = kappa/median(u);
    'scale+-' both signs; 'bounded+' k = f/max(u), f < 1; 'bounded+-'; 'mixed' = scale+ and bounded-."""
    L = {
        "exp": (np.exp, sp.exp, "scale+-"),
        "bose": (lambda t: 1.0 / np.expm1(t), lambda t: 1 / (sp.exp(t) - 1), "scale+-"),
        "fermi": (lambda t: 1.0 / (np.exp(t) + 1.0), lambda t: 1 / (sp.exp(t) + 1), "scale+-"),
        "sech": (lambda t: 1.0 / (np.exp(t) + np.exp(-t)), lambda t: 1 / (sp.exp(t) + sp.exp(-t)), "scale+"),
        "tanh": (np.tanh, sp.tanh, "scale+"),
        "sin": (np.sin, sp.sin, "scale+"),
        "cos": (np.cos, sp.cos, "scale+"),
        "sin2": (lambda t: np.sin(t) ** 2, lambda t: sp.sin(t) ** 2, "scale+"),
        "cos2": (lambda t: np.cos(t) ** 2, lambda t: sp.cos(t) ** 2, "scale+"),
        "asin": (np.arcsin, sp.asin, "bounded+"),
        "acos": (np.arccos, sp.acos, "bounded+-"),
        "atan": (np.arctan, sp.atan, "scale+"),
        "log": (np.log, sp.log, "scale+"),
    }
    for p in (1, 2, -1, -2, Fraction(1, 2), Fraction(-1, 2)):
        L[f"pow{p}"] = ((lambda t, p=float(p): (1.0 + t) ** p),
                        (lambda t, p=sp.Rational(p): (1 + t) ** p), "mixed")
    return L


LIB = _lib()


def _grid(kind, med, umax):
    kap = np.geomspace(1e-2, 1e2, N_GRID) / med
    frac = np.geomspace(0.01, 0.995, N_GRID) / umax
    return {"scale+": kap, "scale+-": np.r_[kap, -kap], "bounded+": frac,
            "bounded+-": np.r_[frac, -frac], "mixed": np.r_[kap, -frac]}[kind]


def _k_bounds(kind, umax):
    lim = (1.0 - 1e-9) / umax
    if kind == "bounded+":
        return 1e-300, lim
    if kind == "bounded+-":
        return -lim, lim
    if kind == "mixed":
        return -lim, np.inf
    if kind == "scale+":
        return 1e-300, np.inf
    return -np.inf, np.inf


def _generic_ref(X, y, L, tr, va):
    """Flexible reference in y space: polynomial (deg 3 if d <= 3 else 2) in standardized inputs + logs."""
    Zs = (X - X[tr].mean(0)) / (X[tr].std(0) + 1e-300)
    F = np.column_stack([_poly(Zs, 3 if X.shape[1] <= 3 else 2), L])
    c, *_ = np.linalg.lstsq(F[tr], y[tr], rcond=None)
    return float(np.mean((F[va] @ c - y[va]) ** 2))


def candidates(X, y, names, seed=0, verbose=False):
    """Refined, snapped composition candidates. Returns (list of dicts, info); each dict has
    expr, val_mse, mse_unsnapped, max_err (scaled, all rows given), ops, g, u."""
    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()
    info = {}
    if len(y) > MAX_ROWS:
        k = np.sort(np.random.default_rng(seed).choice(len(y), MAX_ROWS, replace=False))
        X, y = X[k], y[k]
    n, d = X.shape
    if pool_size_upper_bound(d) > MAX_POOL:
        info["decision"] = "pool_too_large"
        return [], info
    pos = [j for j in range(d) if np.all(X[:, j] > 0)]
    if not pos:
        info["decision"] = "no_positive_inputs"
        return [], info
    scale = float(max(np.max(np.abs(y)), 1e-300))
    tr, va = _split(n, seed)
    nz = y != 0
    trg = tr[nz[tr]]
    if len(trg) < 50:
        info["decision"] = "too_few_rows"
        return [], info
    L = np.log(X[:, pos])
    # pool monomials, positive on all rows
    nonzero = [bool(np.all(X[:, j] != 0)) for j in range(d)]
    cols, labels = _monomial_pool(X, names, nonzero)
    keep = [i for i, c in enumerate(cols) if np.all(c > 0) and np.std(c) > 0]
    if not keep:
        info["decision"] = "empty_pool"
        return [], info
    U = np.column_stack([cols[i] for i in keep])
    labels = [labels[i] for i in keep]
    del cols
    # ---- screen
    sel = trg if len(trg) <= SCREEN_ROWS else np.sort(np.random.default_rng(seed).choice(trg, SCREEN_ROWS, replace=False))
    w = np.abs(y[sel]) / np.sqrt(np.mean(y[sel] ** 2))
    B = np.column_stack([np.ones(len(sel)), L[sel]]) * w[:, None]
    Q, _ = np.linalg.qr(B)
    z = np.log(np.abs(y[sel])) * w
    rz = z - Q @ (Q.T @ z)
    rz2 = float(rz @ rz)
    Us = U[sel]
    med = np.median(Us, axis=0)
    umax = U.max(axis=0)
    best = {}   # (g, j) -> (sse, k)
    with np.errstate(all="ignore"):
        for g, (f, _, kind) in LIB.items():
            for kap in _grid(kind, 1.0, 1.0):
                # k = kap / med  (scale grids)  or  kap / umax  (bounded grids): both encoded per column
                if kind in ("scale+", "scale+-") or (kind == "mixed" and kap > 0):
                    kk = kap / med
                else:
                    kk = kap / umax
                T = np.log(np.abs(f(Us * kk[None, :])))
                ok = np.all(np.isfinite(T), axis=0)
                if not np.any(ok):
                    continue
                Tw = T * w[:, None]
                R = Tw - Q @ (Q.T @ Tw)
                sse = rz2 - 2.0 * (rz @ R) + np.sum(R * R, axis=0)
                sse = np.where(ok, sse, np.inf)
                for j in np.argsort(sse)[:3]:
                    if np.isfinite(sse[j]) and sse[j] < best.get((g, j), (np.inf,))[0]:
                        best[(g, j)] = (float(sse[j]), float(kk[j]))
    order = sorted(best.items(), key=lambda kv: kv[1][0])[:TOP_PAIRS]
    info["screened"] = [(g, labels[j], round(kv[0] / rz2, 6)) for (g, j), kv in order[:5]]
    if not order:
        info["decision"] = "no_candidate"
        return [], info
    # ---- refine, snap
    out = []
    sign_major = float(np.sign(np.median(y[trg])))
    for (g, j), (sse, k0) in order:
        f, fs, kind = LIB[g]
        u = U[:, j]
        lo, hi = _k_bounds(kind, float(umax[j]))

        def model(th, rows):
            with np.errstate(all="ignore"):
                return th[0] * np.exp(L[rows] @ th[1:-1]) * f(th[-1] * u[rows])

        def resid(th):
            r = model(th, tr) - y[tr]
            return np.where(np.isfinite(r), r, 1e6)

        # start: regression of log|y| - log|g(k0 u)| on [1, log x] (training rows)
        with np.errstate(all="ignore"):
            t0 = np.log(np.abs(f(k0 * u[trg])))
        okr = np.isfinite(t0)
        if okr.sum() < 20:
            continue
        D = np.column_stack([np.ones(okr.sum()), L[trg][okr]])
        c0, *_ = np.linalg.lstsq(D, np.log(np.abs(y[trg][okr])) - t0[okr], rcond=None)
        with np.errstate(all="ignore"):
            sg = np.sign(f(k0 * u[trg][okr])) * np.sign(y[trg][okr])
        s0 = float(np.sign(np.sum(sg))) or sign_major
        th0 = np.r_[s0 * np.exp(c0[0]), c0[1:], k0]
        lb = np.r_[-np.inf * np.ones(1 + len(pos)), lo]
        ub = np.r_[np.inf * np.ones(1 + len(pos)), hi]
        th0[-1] = np.clip(th0[-1], lo if np.isfinite(lo) else -np.inf, hi if np.isfinite(hi) else np.inf)
        try:
            res = least_squares(resid, th0, bounds=(lb, ub), x_scale="jac", max_nfev=200)
        except Exception:
            continue
        th = res.x
        r = res.fun
        sse = float(r @ r)
        try:
            cov = sse / max(len(tr) - len(th), 1) * np.linalg.pinv(res.jac.T @ res.jac)
            se = np.sqrt(np.clip(np.diag(cov), 0, None))
        except Exception:
            continue
        with np.errstate(all="ignore"):
            mse_un = float(np.mean((model(th, va) - y[va]) ** 2))
        if not np.isfinite(mse_un):
            continue
        a, k = th[1:-1], float(th[-1])
        a_s = A_ARR[np.argmin(np.abs(a[:, None] - A_ARR[None, :]), axis=1)]
        if not np.all(np.abs(a - a_s) <= K_SE * se[1:-1] + EXP_FLOOR):
            continue
        k_i = int(np.argmin(np.abs(K_ARR - k)))
        k_snapped = abs(K_ARR[k_i] - k) <= K_SE * se[-1] + K_FLOOR * abs(k)
        k_s = float(K_ARR[k_i]) if k_snapped else k
        if not (lo <= k_s <= hi):
            k_snapped, k_s = False, k
        # refit C with snapped shape
        with np.errstate(all="ignore"):
            shape = np.exp(L @ a_s) * f(k_s * u)
        if not np.all(np.isfinite(shape[tr])):
            continue
        C = float(shape[tr] @ y[tr] / (shape[tr] @ shape[tr]))
        pred = C * shape
        if not np.all(np.isfinite(pred)):
            continue
        mse_s = float(np.mean((pred[va] - y[va]) ** 2))
        max_err = float(np.max(np.abs(pred - y)) / scale)
        # expression
        sy = [sp.Symbol(nm) for nm in names]
        mono = sp.Mul(*[sy[pos[i]] ** sp.Rational(Fraction(float(ai)).limit_denominator(4))
                        for i, ai in enumerate(a_s) if ai != 0])
        usym = sp.Mul(*[sy[i] ** p for i, p in labels[j]])
        ksym = sp.Rational(K_FR[k_i]) if k_snapped else sp.Float(k_s, 10)
        expr = sp.Float(C, 10) * mono * fs(ksym * usym)
        try:
            ops = int(sp.count_ops(expr))
        except Exception:
            ops = 10 ** 6
        out.append(dict(expr=str(expr), val_mse=mse_s, mse_unsnapped=mse_un, max_err=max_err, ops=ops,
                        g=g, u=labels[j], k=k_s, k_snapped=bool(k_snapped), a=a_s.tolist()))
    if not out:
        info["decision"] = "none_fitted"
        return [], info
    try:
        info["ref_generic"] = _generic_ref(X, y, L, tr, va)
    except Exception:
        info["ref_generic"] = float("inf")
    info["ref_family"] = min(c["mse_unsnapped"] for c in out)
    info["var_val"] = float(np.var(y[va]))
    out.sort(key=lambda c: c["val_mse"])
    return out, info


def comp_exact(X, y, names, seed=0):
    """Exact use: the simplest candidate with max scaled error < 1e-6 (on the rows used), or None."""
    cands, info = candidates(X, y, names, seed)
    ex = [c for c in cands if c["max_err"] < EXACT_TOL]
    if not ex:
        info["decision"] = info.get("decision", "none_exact")
        return None, info
    ex.sort(key=lambda c: (c["ops"], c["max_err"]))
    info["decision"] = "exact"
    info["expr"] = ex[0]["expr"]
    return ex[0], info


def v_comp(X, y, names, seed=0):
    """Noisy use inside V: lowest held-out MSE among candidates passing V6; same return as the other families."""
    cands, info = candidates(X, y, names, seed)
    if not cands:
        return None, info
    ref = min(info.get("ref_generic", float("inf")), info["ref_family"])
    var_va = info["var_val"]
    acc = [c for c in cands if c["val_mse"] <= c["mse_unsnapped"] * (1 + SNAP_SLACK)
           and c["val_mse"] <= ref * (1 + LOF_SLACK) and c["val_mse"] <= (1 - MIN_R2) * var_va]
    if acc:   # V7 near-ties (relative 1e-6: the same law written two ways) go to the shorter form
        m0 = acc[0]["val_mse"]
        acc = sorted([c for c in acc if c["val_mse"] <= m0 * (1 + 1e-6)], key=lambda c: c["ops"]) + \
              [c for c in acc if c["val_mse"] > m0 * (1 + 1e-6)]
    if not acc:
        info["decision"] = "rejected"
        info["best_tried"] = cands[0]["expr"]
        return None, info
    b = acc[0]
    info.update(decision="accepted", g=b["g"], u=b["u"], k=b["k"], k_snapped=b["k_snapped"],
                mse_snapped=b["val_mse"], mse_unsnapped=b["mse_unsnapped"], ref=ref)
    return {"expr": b["expr"], "val_mse": b["val_mse"]}, info
