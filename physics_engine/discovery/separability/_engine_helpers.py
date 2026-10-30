"""
Engine helpers used by separability, copied VERBATIM from the PIR research repository's
physics_engine/discovery/symbolic_search.py (they are not part of the public engine, whose
_log_linear_gate is an older version). Kept private to the separability package so the public
engine's behaviour is unchanged.
"""
import numpy as np

_F3_LOG_LINEAR_THRESHOLD = 0.99
_F3_LOG_LINEAR_ROUND_TO = 0.5


def _drop_collinear_log_columns(log_X, variables, tol=1e-6):
    """
    Greedily keep only log-space columns that add new rank beyond the
    intercept and whatever's already been kept, dropping any column that is
    an exact (or near-exact) deterministic function of columns already kept
    -- e.g. inv_r2=r**-2 once r is kept, or m1m2=m1*m2 once m1 and m2 are
    kept. This is a generic numerical check (project each candidate onto the
    current column space, drop it if the residual is ~0 relative to its own
    norm) -- it does NOT hardcode which feature names are "derived"; it
    detects the redundancy directly from the data.

    Without this, _log_linear_gate's OLS fit over a rank-deficient design
    matrix silently returns numpy's minimum-norm solution, which smears the
    true exponent across every collinear column instead of concentrating it
    on one -- confirmed root cause of the gravity r^-2.5 bug (see
    GRAVITY_LOG_LINEAR_COLLINEARITY_NOTE.md). Deduplicating here removes the
    rank-deficiency before the fit ever runs, at the source.

    Returns (kept_variables, kept_log_X) -- same row count, fewer columns.
    """
    n = log_X.shape[0]
    kept_cols = [np.ones(n)]
    kept_vars = []
    kept_idx = []

    for j, name in enumerate(variables):
        candidate = log_X[:, j]
        basis = np.column_stack(kept_cols)
        proj_coeffs, _, _, _ = np.linalg.lstsq(basis, candidate, rcond=None)
        residual = candidate - basis @ proj_coeffs
        cand_norm = np.linalg.norm(candidate)
        resid_norm = np.linalg.norm(residual)
        if cand_norm > 0 and (resid_norm / cand_norm) < tol:
            continue  # exact deterministic function of already-kept columns
        kept_cols.append(candidate)
        kept_vars.append(name)
        kept_idx.append(j)

    return kept_vars, log_X[:, kept_idx]


def _log_linear_gate(df, target_var, threshold=None):
    """
    Check if target looks like a power-law monomial: y = C * x1^a1 * x2^a2 * ...
    by fitting log(y) = log(C) + sum(ai * log(xi)) via OLS.

    Returns sympy expression if R² >= threshold, else None (fall through).
    """
    import numpy as np
    import sympy as sp
    from numpy.linalg import lstsq

    if threshold is None:
        threshold = _F3_LOG_LINEAR_THRESHOLD

    try:
        variables = [c for c in df.columns if c != target_var]
        if len(variables) < 1:
            return None

        y = df[target_var].to_numpy(dtype=float)
        X = df[variables].to_numpy(dtype=float)

        # Need all positive for log transform
        valid = (y > 0) & np.all(X > 0, axis=1) & np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        if valid.sum() < max(20, 2 * len(variables)):
            return None

        yv = y[valid]
        Xv = X[valid]

        log_y = np.log(yv)
        log_X = np.log(Xv)

        # Drop columns that are exact deterministic functions of other
        # columns already present (e.g. inv_r2 when r is present, m1m2 when
        # m1 and m2 are present) -- removes rank-deficiency before fitting.
        variables, log_X = _drop_collinear_log_columns(log_X, variables)
        if len(variables) < 1:
            return None

        A = np.column_stack([np.ones(len(yv)), log_X])
        coeffs, _, _, _ = lstsq(A, log_y, rcond=None)

        y_pred = A @ coeffs
        ss_res = np.sum((log_y - y_pred) ** 2)
        ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        if r2 < threshold:
            return None

        # Recover constant and exponents, round to nearest half-integer
        log_C = coeffs[0]
        exponents = coeffs[1:]
        step = _F3_LOG_LINEAR_ROUND_TO

        C_val = float(np.exp(log_C))
        expr = sp.Float(C_val, 4)

        for vname, exp_raw in zip(variables, exponents):
            exp_r = round(exp_raw / step) * step
            if abs(exp_r) < 1e-6:
                continue
            sym = sp.Symbol(vname)
            if exp_r == 1.0:
                expr = expr * sym
            elif exp_r == -1.0:
                expr = expr / sym
            else:
                expr = expr * sym ** sp.Rational(int(exp_r * 2), 2)

        return sp.simplify(expr)

    except Exception:
        return None


def _select_atoms_residual_projection(exprs, symbols, data_df, target_var, max_terms):
    """
    Greedy residual-projection (OMP-style) atom selection for Franka mass
    tasks. Replaces sp.srepr alphabetical sort for these tasks only.

    Picks the atom most correlated with the CURRENT RESIDUAL at each step
    (not the raw target), refits via least squares, updates the residual,
    and repeats. This surfaces atoms whose importance is conditional on
    other atoms already being selected -- e.g. sin(2*q2)*cos(q3) for
    Franka M11, whose marginal correlation with the raw target is
    unremarkable but whose correlation with the residual after cos(q4)
    and cos(q2)*cos(q4) are fit is high.

    Returns at most `max_terms` expressions, in selection order. Never
    returns more than max_terms -- this preserves the existing
    max_basis_terms cap exactly.
    """
    import numpy as np
    import sympy as sp

    if max_terms <= 0 or not exprs:
        return []

    y = data_df[target_var].to_numpy(dtype=float)
    sym_to_col = {v: k for k, v in symbols.items()}

    cols = []
    valid_exprs = []
    for expr in exprs:
        free_syms = list(expr.free_symbols)
        try:
            arg_cols = [sym_to_col[s] for s in free_syms]
            if free_syms:
                f = sp.lambdify(free_syms, expr, modules="numpy")
                args = [data_df[c].to_numpy(dtype=float) for c in arg_cols]
                vals = np.asarray(f(*args), dtype=float)
                if vals.shape == ():
                    vals = np.full(len(data_df), float(vals))
            else:
                vals = np.full(len(data_df), float(expr))
        except Exception:
            continue
        if not np.all(np.isfinite(vals)):
            continue
        cols.append(vals)
        valid_exprs.append(expr)

    if not cols:
        return []

    X = np.column_stack(cols)
    max_terms = min(max_terms, X.shape[1])

    residual = y.copy()
    selected = []
    remaining = list(range(X.shape[1]))

    for _ in range(max_terms):
        best_score, best_idx = -1.0, None
        for idx in remaining:
            col = X[:, idx]
            norm = np.linalg.norm(col)
            if norm < 1e-12:
                continue
            score = abs(np.dot(col, residual)) / norm
            if score > best_score:
                best_score, best_idx = score, idx
        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
        X_sel = X[:, selected]
        coeffs, _, _, _ = np.linalg.lstsq(X_sel, y, rcond=None)
        residual = y - X_sel @ coeffs

    return [valid_exprs[i] for i in selected]
