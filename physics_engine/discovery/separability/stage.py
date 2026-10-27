"""
Separability stage for PIRRegressor (PIR 3.8): if y = g(A) * h(B), solve g and h separately.

Port of the research method "pirsep8" (research repo: pirsep_regressor.py v7 + pirsep8_regressor.py,
verified V1 2026-10-28: 12 gain laws x 12 seeds, 144/144). Rules (claude/pir38_plan.md, section 2):
  R1  detection by the cross-term test (crossterm.py), also inside the recursion;
  R1b the block split 'single-variable blocks | multi-variable blocks' is tried first;
  R2  at most MAX_ATTEMPTS = 4 extraction attempts (counted, no wall-clock budget);
  R3  accept only an EXACT recombination (max scaled error < 1e-6) that beats the engine.
The helper functions below are copied verbatim from pirsep_regressor.py v7.
"""
import time

import numpy as np
import pandas as pd
import sympy as sp

from .crossterm import block_split, crossterm_search
from .exact import _evaluate
from .recursive import extract_recursive

EXACT_TOL = 1e-6
TOPK = 8
MAX_ATTEMPTS = 4
CANON = True
CANON_MAX_OPS = 4000


def _numeric(exprs, d):
    v = np.ones(len(d), dtype=float)
    for e in exprs:
        v = v * np.asarray(_evaluate(e, d), dtype=float)
    return v


def _positive_simplify(e, df):
    """sqrt(n**2) -> n for columns that are strictly positive in the training data.
    Pure representation fix (same function on the data); SRBench's symbolic scorer cannot
    simplify sqrt(n**2) unless n is declared positive."""
    try:
        pos = {}
        for s in e.free_symbols:
            nm = str(s)
            if nm in df.columns and float(df[nm].min()) > 0:
                pos[s] = sp.Dummy(nm, positive=True)
        if not pos:
            return e
        back = {v: k for k, v in pos.items()}
        return e.xreplace(pos).xreplace(back)
    except Exception:
        return e


def _clean(e):
    """Zero out float coefficients that are numerically nothing (~1e-17 fit noise)."""
    fl = list(e.atoms(sp.Float))
    if not fl:
        return e
    mx = max(abs(float(f)) for f in fl)
    drop = {f: sp.Integer(0) for f in fl if abs(float(f)) < 1e-10 * mx}
    return e.xreplace(drop) if drop else e





def _canonicalize(e, rel_drop=1e-8, tol=1e-9, max_den=1000, max_num=10**6):
    """Name-free canonical form of a recovered expression (representation only).
    1. drop float coefficients below rel_drop * max|float| (fit noise)
    2. put over one denominator; divide numerator and denominator by the smallest-magnitude
       float coefficient of the denominator
    3. turn floats that equal a small rational within tol into exact Rationals.
    The caller re-verifies the result numerically against the data."""
    fl = list(e.atoms(sp.Float))
    if not fl:
        return e
    # Upstream solvers leave Floats of MIXED precision (e.g. 5.331 stored with 4-digit
    # precision next to a 12-digit value); SymPy arithmetic then degrades to the lowest
    # precision (5.331*sqrt(0.0351870319179) evaluated to 0.99992 instead of 1.00000).
    # Rebuild every Float at high precision from its printed decimal digits -- exactly the
    # literal that lambdify / the numeric data check uses.
    e = e.xreplace({f: sp.Float(str(f), 30) for f in fl})
    fl = list(e.atoms(sp.Float))
    mx = max(abs(float(f)) for f in fl)
    e = e.xreplace({f: sp.Integer(0) for f in fl if abs(float(f)) < rel_drop * mx})
    e = sp.together(e)
    n, d = sp.fraction(e)
    n, d = sp.expand(n), sp.expand(d)
    dfl = sorted(abs(float(f)) for f in d.atoms(sp.Float))
    if dfl and dfl[0] > 0:
        s = sp.Float(dfl[0], 30)
        n, d = sp.expand(n / s), sp.expand(d / s)
    e = n / d

    def rat(x):
        r = sp.nsimplify(x, tolerance=tol * max(1.0, abs(float(x))), rational=True)
        if r.is_Rational and abs(r.q) <= max_den and abs(r.p) <= max_num:
            return r
        return x
    return e.xreplace({f: rat(f) for f in e.atoms(sp.Float)})


def _parse_err(txt, df, yv, scale):
    """Error of a model STRING exactly as a consumer (SRBench) would read it: parse it back."""
    syms = {c: sp.Symbol(c) for c in df.columns if c != "target"}
    P = sp.sympify(txt, locals=syms)
    ss = sorted(P.free_symbols, key=str)
    f = sp.lambdify(ss, P, "numpy")
    v = np.broadcast_to(np.asarray(f(*[df[str(x)].to_numpy(dtype=float) for x in ss]), dtype=float), (len(df),))
    if not np.all(np.isfinite(v)):
        return float("inf")
    return float(np.max(np.abs(v - yv)) / scale)


def _final_string(use, df, yv, scale):
    """Pick the model string. Every candidate is PARSED BACK and checked against the training
    data (< EXACT_TOL); the first that passes is used. Returns (string, form, log)."""
    log = []
    E = None
    # ROOT CAUSE (found by diagnostics): upstream solvers leave Floats whose PRINTED decimal
    # (what lambdify evaluates, e.g. 5.331) differs from the stored binary value (4-digit
    # precision -> up to ~1e-5 relative). Mul-merging, float(), expand etc. use the stored
    # value, so the string drifted from the data. Rebuild every Float from its printed
    # decimal BEFORE any arithmetic; this is exactly the literal the accepted model used.
    try:
        use = [e.xreplace({f: sp.Float(str(f), 30) for f in e.atoms(sp.Float)}) for e in use]
    except Exception as ex:
        log.append(f"literal_rebuild_exc {ex}")
    try:
        E = sp.Mul(*use)
    except Exception as ex:
        log.append(f"mul_exception {ex}")
    if CANON and E is not None:
        try:
            ops = int(sp.count_ops(E))
            if ops > CANON_MAX_OPS:
                log.append(f"canon_skipped_ops={ops}")
            else:
                for tol in (1e-9, 1e-7, 1e-5, 1e-4):
                    try:
                        C = _canonicalize(E, tol=tol)
                        txt = str(C)
                        err = _parse_err(txt, df, yv, scale)
                        log.append(f"canon tol={tol:g} err={err:.2e}")
                        if err < EXACT_TOL:
                            return txt, f"canonical(tol={tol:g})", log
                    except Exception as ex:
                        log.append(f"canon tol={tol:g} exc {type(ex).__name__}: {str(ex)[:80]}")
        except Exception as ex:
            log.append(f"canon_exc {type(ex).__name__}: {str(ex)[:80]}")
    # plain forms: (a) full-precision Floats in one product, (b) factors joined textually
    try:
        def hp(e):
            fl = list(e.atoms(sp.Float))
            return e.xreplace({f: sp.Float(float(f), 17) for f in fl})
        cands = []
        if E is not None:
            cands.append(("plain_fullprec_mul", sp.sstr(hp(E), full_prec=True)))
        cands.append(("plain_fullprec_factors", "*".join("(%s)" % sp.sstr(hp(e), full_prec=True) for e in use)))
        for name, txt in cands:
            err = _parse_err(txt, df, yv, scale)
            log.append(f"{name} err={err:.2e}")
            if err < EXACT_TOL:
                return txt, name, log
    except Exception as ex:
        log.append(f"plain_exc {type(ex).__name__}: {str(ex)[:80]}")
    return None, "none", log


def run_separability(X_df, y, base_err):
    """X_df: DataFrame with safe column names; y: 1-D array; base_err: engine's max scaled error.
    Returns (model_string, factor_exprs, info). model_string is None if nothing was accepted."""
    cols = list(X_df.columns)
    df = pd.DataFrame(np.asarray(X_df, dtype=float), columns=cols)
    yv = np.asarray(y, dtype=float).ravel()
    scale = float(max(np.max(np.abs(yv)), 1e-300))
    info = {"decision": None, "tried": [], "n_rows": len(df), "n_features": len(cols)}
    if base_err < EXACT_TOL:
        info["decision"] = "engine_exact"
        return None, None, info
    df["target"] = yv
    t0 = time.time()
    _, ranked = crossterm_search(df, target="target")
    info["search_secs"] = round(time.time() - t0, 2)
    cands = [(list(r[0]), list(r[1])) for r in ranked if r[2] == "MULT"][:TOPK]
    bs = block_split(df, target="target")
    info["block_split"] = None if bs is None else " | ".join("+".join(g) for g in bs)
    if bs is not None:
        bs = (list(bs[0]), list(bs[1]))
        cands = [bs] + [c for c in cands if set(c[0]) != set(bs[0]) and set(c[1]) != set(bs[0])]
    info["n_mult_candidates"] = len(cands)
    attempts = 0
    for A, B in cands:
        for ga, gb in ((A, B), (B, A)):
            if attempts >= MAX_ATTEMPTS:
                info["decision"] = "attempts_exhausted"
                return None, None, info
            attempts += 1
            t1 = time.time()
            res = extract_recursive(df, ga, gb, target_col="target")
            rec = {"split": "+".join(ga) + " | " + "+".join(gb), "secs": round(time.time() - t1, 1)}
            if res is None:
                rec["result"] = "None"
                info["tried"].append(rec)
                continue
            rec.update(recovered=bool(res["recovered"]), err=float(res["recombined_max_rel_err"]))
            info["tried"].append(rec)
            if not res["recovered"]:
                continue
            syms = {c: sp.Symbol(c) for c in cols}
            exprs = [sp.sympify(e, locals=syms) if isinstance(e, str) else e for _, e in res["factors"]]
            cleaned = [_positive_simplify(_clean(e), df) for e in exprs]
            err_c = float(np.max(np.abs(_numeric(cleaned, df) - yv)) / scale)
            err_o = float(np.max(np.abs(_numeric(exprs, df) - yv)) / scale)
            use, err = (cleaned, err_c) if err_c < EXACT_TOL else (exprs, err_o)
            if not (err < EXACT_TOL and err < base_err):
                info["decision"] = "rejected_not_better"
                continue
            txt, form, log = _final_string(use, df, yv, scale)
            # as in v7: if no string form re-verifies, the plain product is used (predictions
            # always come from the verified factors themselves)
            info.update(decision="separability_accepted", accepted_split=rec["split"],
                        accepted_err_scaled=err, string_form=form, string_verified=txt is not None)
            if txt is None:
                info["string_log"] = log
                txt = str(sp.Mul(*use))
            return txt, use, info
    info["decision"] = info["decision"] or "no_recovery"
    return None, None, info
