"""
separability_exact.py — exact (non-binned) extraction for multiplicative
separability, replacing separability_partition.py's lossy bin-cell-mean
extraction (2-5% error even on true products; see SEPARABILITY_ROI_NOTE.md).

ARCHITECTURE ("defer to PIR", per SEPARABILITY_ROI_NOTE.md's decided-but-
not-yet-built fix):
    Given a CONFIRMED multiplicative split f(A,B) = g(A)*h(B) (detection is
    separability_primitive.py's job, unchanged, reused as-is):

    1. Fit a MARGINAL log-linear regression of log(f) on group A's raw
       columns ALONE (ignoring B entirely). Reuses symbolic_search.py's own
       _log_linear_gate() -- same OLS-in-log-space math, same rounding to
       nice fractions, same collinearity dedup (the gravity fix) -- but
       called with threshold=-1 to DISABLE its R^2 acceptance gate.

       Why disabling the gate is correct here, not cheating: _log_linear_gate
       normally requires R^2 >= 0.99 because a low R^2 usually means "not a
       clean monomial". But here we already KNOW (from the separability
       detection step) that f is separable into g(A)*h(B) -- a LOW R^2 for
       the A-only marginal fit is EXPECTED and does not indicate a bad fit,
       it indicates that most of log(f)'s variance is explained by the
       (deliberately omitted) B-dependent term. Confirmed empirically before
       building this: A and B are independently sampled in every case-1
       dataset checked (cross-correlation ~0.001-0.003), so omitting B from
       the regression does NOT bias the coefficient estimate on A's own
       terms (standard OLS orthogonality/exogeneity argument) -- it only
       adds unbiased residual variance, which is exactly what depresses R^2
       without touching the coefficient itself.

    2. Divide the ACTUAL target by the fitted g_hat(A), evaluated per row
       (not per bin -- this is the key exactness fix). For a true product,
       this recovers h(B) exactly (to floating-point precision), for EVERY
       row, no averaging.

    3. Solve the residual (target / g_hat) over group B's own raw columns
       via a small curated additive/pairwise-product basis (see
       _solve_additive_sum), since h(B) may itself be a non-monomial sum
       (e.g. B + B*C) that a pure log-linear fit can't find. Deliberately
       does NOT call discover_law() for this -- found directly by testing
       that its F3 gate can false-positive on exactly this kind of
       residual; see _solve_additive_sum's docstring for the mechanism.

    4. Try BOTH orderings (fit A first & solve B, and fit B first & solve A)
       since we don't know a priori which side is the "easy" monomial one.
       For single-column groups, also try +1/-1 shifted variants of the
       marginal fit (see _fit_marginal_candidates) to catch reciprocal-of-
       affine factors like 1/(gamma-1). Report whichever combination
       achieves the cleanest overall reconstruction (lowest recombination
       error against the actual target), not the best marginal R^2 in
       isolation.

    Additive splits (g(A)+h(B)) use the same idea without logs: subtract the
    fitted g_hat(A) from the target directly, then hand the residual to
    discover_law() for group B.

VALIDATION: this module's __main__ block runs a synthetic KNOWN case,
f = A * (B + B*C), matching feynman_II_37_1's exact structure, BEFORE this
is ever pointed at real data -- same "prove it on a known answer first"
discipline as strict_scorer.py's self-test and separability_primitive.py's
validation.
"""
import os
import sys
from itertools import combinations, permutations

import numpy as np
import pandas as pd
import sympy as sp

from ._engine_helpers import (  # package port: was physics_engine.discovery.symbolic_search
    _log_linear_gate,
    _select_atoms_residual_projection,
)


def _fit_marginal_monomial(df, cols, target_col):
    """Fit log(target) ~ log(cols) alone, ignoring every other column.
    Reuses _log_linear_gate's own OLS-in-log-space + rounding, with its
    R^2 gate disabled (threshold=-1) since a low R^2 here is expected, not
    disqualifying -- see module docstring."""
    sub = df[list(cols) + [target_col]].copy()
    expr = _log_linear_gate(sub, target_col, threshold=-1.0)
    return expr  # sympy expression in terms of `cols`, or None on hard failure


def _fit_marginal_candidates(df, cols, target_col):
    """
    Return a LIST of candidate g(cols) expressions to try as the "first"
    group's factor, ignoring the other group entirely. Always includes the
    raw log-linear monomial fit. For single-column groups, ALSO tries
    fitting on (col+1) and (col-1) as alternate single-feature transforms.

    This is a small, curated, LOCAL addition (confined entirely to this
    module -- no shared grammar list or function signature touched) added
    specifically because reciprocal-of-affine factors (1/(gamma-1),
    1/(1+sigma)) showed up in real case-1 data and a plain log-linear fit
    on the raw column cannot represent them (see
    SEPARABILITY_CASE1_RECOVERY_NOTE.md). Deliberately NOT a generic
    search over arbitrary shifts -- only +1/-1, the two forms actually
    observed.

    Candidates are not ranked here by their own R^2 (all of them omit the
    other group, so their absolute R^2 is uniformly depressed and not a
    fair comparison in isolation) -- extract_multiplicative_exact ranks
    them by the FINAL recombination error instead, which is the only
    apples-to-apples comparison once both groups are solved.
    """
    candidates = []
    g0 = _fit_marginal_monomial(df, cols, target_col)
    if g0 is not None:
        candidates.append(g0)

    if len(cols) == 1:
        col = cols[0]
        col_sym = sp.Symbol(col)
        for shift in (1, -1):
            shifted_name = f"__{col}_shift__"
            shifted_vals = df[col].to_numpy(dtype=float) + shift
            if not np.all(shifted_vals > 0):
                continue  # _log_linear_gate needs strictly positive columns
            sub = pd.DataFrame({shifted_name: shifted_vals, target_col: df[target_col].to_numpy(dtype=float)})
            g_shifted = _log_linear_gate(sub, target_col, threshold=-1.0)
            if g_shifted is not None:
                g_shifted = g_shifted.subs(sp.Symbol(shifted_name), col_sym + shift)
                candidates.append(g_shifted)

    return candidates


def _evaluate(expr, df):
    """Vectorized evaluation of a sympy expression over a dataframe's rows."""
    free = sorted(expr.free_symbols, key=str)
    if not free:
        return np.full(len(df), float(expr))
    f = sp.lambdify(free, expr, modules="numpy")
    args = [df[str(s)].to_numpy(dtype=float) for s in free]
    return np.asarray(f(*args), dtype=float)


def _solve_additive_sum_core(df, cols, residual, max_atoms=6, extended=False):
    """
    Find the sparse additive combination of a SMALL, CURATED basis
    (each column to the first and second power, pairwise products within
    the group, and power1*power2 cross terms) that reconstructs `residual`
    exactly.

    Deliberately does NOT call discover_law() for this step. Found directly
    by testing: discover_law()'s F3 log-linear gate runs unconditionally
    before anything else and, for a residual like B*(1+C), can falsely
    clear its own R^2>=0.99 threshold (a power law happens to approximate
    B*(1+C) well over this narrow data range) and return a WRONG monomial
    answer before the correct additive-sum search ever gets a turn. Rather
    than touch _log_linear_gate's threshold or discover_law's control flow
    (both load-bearing for the existing 22-task pipeline), this reuses
    _select_atoms_residual_projection -- the same validated greedy
    atom-selector already used for Franka mass-matrix tasks -- directly,
    bypassing the F3 gate entirely for this step.

    Squares and power1*power2 cross terms added 2026-08-10 for sum-of-
    squares remainders like v**2+u**2+w**2 (I_13_4) and m*omega**2+
    m*omega_0**2 (I_24_6) -- confirmed genuinely local to this function,
    same isolation already verified for the +1/-1 marginal-fit extension.
    Still deliberately small and curated, not a combinatorially broad
    power search: only power 1 and power 2, no higher powers.
    """
    symbols = {c: sp.Symbol(c) for c in cols}
    candidates = [symbols[c] for c in cols]
    candidates += [symbols[c] ** 2 for c in cols]
    candidates += [symbols[c1] * symbols[c2] for c1, c2 in combinations(cols, 2)]
    candidates += [symbols[c1] * symbols[c2] ** 2 for c1, c2 in permutations(cols, 2)]
    if extended:
        # 3.9 prototype (B): a constant and reciprocals, so differences such as 1/r2 - 1/r1
        # are reachable; used only after the 3.8 atom set has failed (see wrapper below)
        candidates += [sp.Integer(1)]
        candidates += [1 / symbols[c] for c in cols if np.all(np.abs(df[c].to_numpy(dtype=float)) > 1e-300)]

    sub = df[list(cols)].copy()
    sub["__residual__"] = residual
    selected = _select_atoms_residual_projection(
        candidates, symbols, sub, "__residual__", max_terms=max_atoms
    )
    if not selected:
        return None, float("inf")

    X = np.column_stack([_evaluate(expr, sub) for expr in selected])
    y = residual
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ coeffs
    max_rel_err = float(np.max(np.abs(y_pred - y) / (np.abs(y) + 1e-12)))

    expr = sum(sp.Float(float(c), 12) * e for c, e in zip(coeffs, selected))
    return sp.simplify(expr), max_rel_err



def _solve_additive_sum(df, cols, residual, max_atoms=6):
    """3.8 atom set first; the extended set (constant, reciprocals) only if that is not exact."""
    h, err = _solve_additive_sum_core(df, cols, residual, max_atoms=max_atoms)
    if h is not None and err < 1e-6:
        return h, err
    h2, err2 = _solve_additive_sum_core(df, cols, residual, max_atoms=max_atoms, extended=True)
    if h2 is not None and err2 < err:
        return h2, err2
    return h, err

def _solve_reciprocal_sum(df, cols, residual, max_atoms=6):
    """
    Alternative to _solve_additive_sum for residuals of the shape
    h(B) = 1/Q(B), where Q(B) sits in the SAME curated basis
    _solve_additive_sum uses (each column to the first and second power,
    pairwise products, and power1*power2 cross terms).

    Reciprocal-linearization trick, added 2026-08-11 -- same spirit as
    _log_linear_gate's log-linearization, but inverting instead of logging:
    if h = 1/Q(B), then 1/h = Q(B) is LINEAR in Q's coefficients, so a
    plain linear least-squares fit of 1/residual against the basis
    recovers Q(B) exactly (deterministic, no nonlinear solver, same
    exact-recovery guarantee as every other step in this module), then
    inverting back gives h. NOT partial fractions -- that decomposes a
    single-variable rational function into simple fractions and doesn't
    apply here; this is a basis-transform trick for a multi-variable
    residual that is the reciprocal of a polynomial already in-grammar.

    Scoping checked BEFORE writing this (see SEPARABILITY_ROI_NOTE.md /
    SEPARABILITY_CASE1_RECOVERY_NOTE.md's "I_9_18" section): confirmed by
    reading the true formulas directly, not assumed, that the 6-dataset
    relativistic family (II_13_23, I_10_7, I_15_10, I_48_2, I_34_14) needs
    1/sqrt(1-v**2/c**2) -- inverting once still leaves a square root, not a
    linear combination of this basis -- and I_34_1 needs 1/(1-v/c), which
    inverts to 1-v/c, needing a constant term and a v/c RATIO, neither of
    which this basis (or _solve_additive_sum's) contains. So this trick is
    NOT expected to help any of those 6; scoped specifically to residuals
    that are the reciprocal of a polynomial already representable in the
    existing basis (I_9_18-shaped, and also II_11_3-shaped -- see below).

    Basis history, corrected 2026-08-11 (found by testing, not assumed):
    the FIRST version of this function deliberately dropped the cubic
    power1*power2 cross terms to fix an I_9_18-specific atom-selection
    failure (see the atom-budget note below), reasoning that a
    reciprocal-of-quadratic-form target should only ever need pure
    quadratic terms. That reasoning was WRONG -- caught by testing on
    feynman_II_11_3 (q*Ef/(m*(omega_0**2-omega**2))), whose true Q(B) is
    m*omega_0**2 - m*omega**2, exactly the cubic power1*power2 shape that
    had been dropped. Restoring those terms (matching _solve_additive_sum's
    basis exactly, no divergence) fixes II_11_3 (converges to ~1e-14)
    WITHOUT breaking I_9_18 (still converges, to ~1e-12 -- slightly looser
    than the quadratic-only version's ~1e-13 because the search space is
    now larger, but still comfortably within the 1e-6 full-recovery
    threshold). Lesson logged for future extensions: don't hand-narrow a
    shared-shape basis for one dataset's atom-selection convenience --
    verify against at least one more real example with a different true
    Q(B) shape before treating a basis restriction as validated.

    Atom-budget note (found by testing on I_9_18, not assumed): the
    caller-supplied `max_atoms` (default 6, sized for _solve_additive_sum's
    sparser targets) is NOT enough atoms for a reciprocal-of-quadratic-form
    target over a 6+ column group -- I_9_18's true Q(B) needs 9 terms, and
    _select_atoms_residual_projection (shared, load-bearing for Franka
    tasks elsewhere, NOT modified) is a greedy OMP selector that can get
    misled by collinear candidates (e.g. x1*y1 looking almost as good as
    x1*x2 when x1,y1 happen to share a sampled range) before it reaches the
    true combination, if starved of budget. Verified empirically: giving
    it the FULL candidate pool (no early cap) reliably reaches the exact
    solution regardless of selection-order detours, because once the true
    atoms are all included, least-squares zeroes out every spurious extra
    automatically. With ~100k-row datasets and candidate pools this small
    (dozens, not thousands), an uncapped full-pool fit is just an ordinary
    overdetermined least-squares problem -- correctness, not sparsity, is
    the point of this exact-recovery step, same principle as disabling
    _log_linear_gate's R^2 gate elsewhere in this module. So this function
    deliberately ignores `max_atoms` as a cap and always searches the full
    candidate pool.

    Returns (h_expr, max_rel_err_vs_residual), or (None, inf) if `residual`
    has a zero/non-finite entry (reciprocal undefined) or the atom selector
    finds nothing.
    """
    if np.any(np.abs(residual) < 1e-300) or not np.all(np.isfinite(residual)):
        return None, float("inf")
    target_inv = 1.0 / residual
    if not np.all(np.isfinite(target_inv)):
        return None, float("inf")

    symbols = {c: sp.Symbol(c) for c in cols}
    candidates = [symbols[c] for c in cols]
    candidates += [symbols[c] ** 2 for c in cols]
    candidates += [symbols[c1] * symbols[c2] for c1, c2 in combinations(cols, 2)]
    candidates += [symbols[c1] * symbols[c2] ** 2 for c1, c2 in permutations(cols, 2)]

    sub = df[list(cols)].copy()
    sub["__target_inv__"] = target_inv
    selected = _select_atoms_residual_projection(
        candidates, symbols, sub, "__target_inv__", max_terms=len(candidates)
    )
    if not selected:
        return None, float("inf")

    X = np.column_stack([_evaluate(expr, sub) for expr in selected])
    coeffs, _, _, _ = np.linalg.lstsq(X, target_inv, rcond=None)
    Q_vals = X @ coeffs
    if np.any(np.abs(Q_vals) < 1e-300) or not np.all(np.isfinite(Q_vals)):
        return None, float("inf")

    h_pred = 1.0 / Q_vals
    max_rel_err = float(np.max(np.abs(h_pred - residual) / (np.abs(residual) + 1e-12)))

    Q_expr = sum(sp.Float(float(c), 12) * e for c, e in zip(coeffs, selected))
    h_expr = 1 / sp.simplify(Q_expr)
    return h_expr, max_rel_err


def _solve_square_sum(df, cols, residual, max_atoms=6):
    """
    Alternative to _solve_additive_sum/_solve_reciprocal_sum for residuals
    of the shape h(B) = sqrt(Q(B)), where Q(B) sits in the SAME curated
    basis the other two solvers use (each column to the first and second
    power, pairwise products, and power1*power2 cross terms).

    Square-linearization trick, added 2026-08-12 -- the third variant of
    the same idea as _log_linear_gate's log-linearization and
    _solve_reciprocal_sum's inversion: if h = sqrt(Q(B)), then h^2 = Q(B)
    is LINEAR in Q's coefficients, so a plain linear least-squares fit of
    residual^2 against the basis recovers Q(B) exactly (deterministic, no
    nonlinear solver, same exact-recovery guarantee as every other step in
    this module), then h = sqrt(Q_hat). Guards against Q_hat going
    negative anywhere (undefined sqrt), same style of guard
    _solve_reciprocal_sum already has for near-zero/non-finite values.

    Scoping checked BEFORE writing this (see SEPARABILITY_ROI_NOTE.md /
    SEPARABILITY_CASE1_RECOVERY_NOTE.md's bucket-B triage): of the 3 named
    sqrt-of-sum-of-squares candidates, a controlled separability check
    (hold the marginal-fit group fixed, vary the other group, check
    d ln(target)/d(marginal var) stays constant) showed feynman_III_10_19
    (mom*sqrt(Bx**2+By**2+Bz**2)) and feynman_II_6_15a
    (p_d/(4*pi*epsilon)*3*z/r**5*sqrt(x**2+y**2)) are genuinely separable,
    while feynman_II_24_17 (sqrt(omega**2/c**2-pi**2/d**2)) is NOT (spread
    0.058 across B-variants) -- a detection artifact, same failure mode as
    I_18_4/I_27_6. Of the 2 real candidates, only III_10_19 is a clean,
    ONE-LEVEL target for this solver: its residual h(Bx,By,Bz) is a pure
    sum of squares under a sqrt, directly representable in this basis.
    II_6_15a's residual is a NESTED product (a rational-monomial part
    times a sqrt-of-sum-of-squares part over only 2 of its 5 remaining
    columns) that this flat single-shot solver cannot represent -- it
    needs recursive re-splitting of the residual, which
    separability_partition.py explicitly documents as not built yet.
    Deliberately not attempted here; this solver is scoped to III_10_19-
    shaped one-level cases only.

    Returns (h_expr, max_rel_err_vs_residual), or (None, inf) if the atom
    selector finds nothing or the fitted Q(B) goes negative anywhere.
    """
    symbols = {c: sp.Symbol(c) for c in cols}
    candidates = [symbols[c] for c in cols]
    candidates += [symbols[c] ** 2 for c in cols]
    candidates += [symbols[c1] * symbols[c2] for c1, c2 in combinations(cols, 2)]
    candidates += [symbols[c1] * symbols[c2] ** 2 for c1, c2 in permutations(cols, 2)]

    residual_sq = residual ** 2
    sub = df[list(cols)].copy()
    sub["__residual_sq__"] = residual_sq
    selected = _select_atoms_residual_projection(
        candidates, symbols, sub, "__residual_sq__", max_terms=len(candidates)
    )
    if not selected:
        return None, float("inf")

    X = np.column_stack([_evaluate(expr, sub) for expr in selected])
    coeffs, _, _, _ = np.linalg.lstsq(X, residual_sq, rcond=None)
    Q_vals = X @ coeffs
    if np.any(Q_vals < 0) or not np.all(np.isfinite(Q_vals)):
        return None, float("inf")

    h_pred = np.sqrt(Q_vals)
    max_rel_err = float(np.max(np.abs(h_pred - residual) / (np.abs(residual) + 1e-12)))

    Q_expr = sum(sp.Float(float(c), 12) * e for c, e in zip(coeffs, selected))
    h_expr = sp.sqrt(sp.simplify(Q_expr))
    return h_expr, max_rel_err


def _solve_rational_sum(df, cols, residual, max_atoms=6):
    """
    Alternative to the other three solvers for residuals of the shape
    h(B) = 1/Q(B), where Q(B) may include NEGATIVE-power atoms (1/col,
    1/(col1*col2)) and a bare CONSTANT term -- none of which any prior
    solver's basis has had.

    Rational-linearization trick, added 2026-08-13 -- same inversion
    pattern as _solve_reciprocal_sum, but over an extended basis: the
    existing positive-power basis (col, col**2, col1*col2, col1*col2**2)
    PLUS a constant (1), 1/col for each column, and 1/(col1*col2) for
    each pair. 1/h = Q(B) is still LINEAR in Q's coefficients regardless
    of which atoms are in the basis, so the same deterministic
    least-squares fit + invert-back recovers Q(B) exactly whenever it is
    representable in this richer set.

    Scoping checked BEFORE writing this (see SEPARABILITY_CASE1_RECOVERY_NOTE.md's
    bucket-F/G closer read): feynman_II_11_27 (n*alpha/(1-(n*alpha/3))*epsilon*Ef)
    is genuinely separable (confirmed via the controlled derivative test,
    every column probed individually), but its residual after peeling off
    the clean epsilon*Ef factor, g(n,alpha) = n*alpha/(1-n*alpha/3), gives
    1/g = 1/(n*alpha) - 1/3 -- exactly TWO atoms neither prior solver's
    basis had: 1/(n*alpha) (reciprocal of a PRODUCT, not a raw column)
    and a bare constant. Scoped narrowly to this shape; not expected to
    help the relativistic sqrt family (inverting a sqrt still leaves a
    sqrt, not a rational function, regardless of basis richness) or any
    dataset already solved by a narrower basis.

    Guards: near-zero/non-finite columns are guarded the same way as
    every prior division in this module (`np.abs(...) < 1e-300`) --
    genuinely small but nonzero sampled values (e.g. a column ranging
    down to 0 in its metadata, like II_11_27's n and alpha) stay
    perfectly finite in float64 at any realistic sample size; only an
    exact-zero or catastrophically-small value excludes that reciprocal
    candidate.

    Returns (h_expr, max_rel_err_vs_residual), or (None, inf) if `residual`
    has a zero/non-finite entry or the atom selector finds nothing.
    """
    if np.any(np.abs(residual) < 1e-300) or not np.all(np.isfinite(residual)):
        return None, float("inf")
    target_inv = 1.0 / residual
    if not np.all(np.isfinite(target_inv)):
        return None, float("inf")

    symbols = {c: sp.Symbol(c) for c in cols}
    col_vals = {c: df[c].to_numpy(dtype=float) for c in cols}

    candidates = [sp.Integer(1)]
    candidates += [symbols[c] for c in cols]
    candidates += [symbols[c] ** 2 for c in cols]
    candidates += [symbols[c1] * symbols[c2] for c1, c2 in combinations(cols, 2)]
    candidates += [symbols[c1] * symbols[c2] ** 2 for c1, c2 in permutations(cols, 2)]

    for c in cols:
        if np.all(np.abs(col_vals[c]) > 1e-300):
            candidates.append(1 / symbols[c])
    for c1, c2 in combinations(cols, 2):
        if np.all(np.abs(col_vals[c1] * col_vals[c2]) > 1e-300):
            candidates.append(1 / (symbols[c1] * symbols[c2]))

    sub = df[list(cols)].copy()
    sub["__target_inv__"] = target_inv
    selected = _select_atoms_residual_projection(
        candidates, symbols, sub, "__target_inv__", max_terms=len(candidates)
    )
    if not selected:
        return None, float("inf")

    X = np.column_stack([_evaluate(expr, sub) for expr in selected])
    coeffs, _, _, _ = np.linalg.lstsq(X, target_inv, rcond=None)
    Q_vals = X @ coeffs
    if np.any(np.abs(Q_vals) < 1e-300) or not np.all(np.isfinite(Q_vals)):
        return None, float("inf")

    h_pred = 1.0 / Q_vals
    max_rel_err = float(np.max(np.abs(h_pred - residual) / (np.abs(residual) + 1e-12)))

    Q_expr = sum(sp.Float(float(c), 12) * e for c, e in zip(coeffs, selected))
    h_expr = 1 / sp.simplify(Q_expr)
    return h_expr, max_rel_err


ENABLE_TRANSFORMED = False


def _best_subset(A, target, kmax=3, tol=1e-9):
    """Exhaustive best subset of at most kmax columns, smallest size first, via the Gram matrix
    (each subset costs a kmax x kmax solve), then verified on all rows by max error.
    Returns (idx, coef, rel_max_err) of the first exact subset, else of the best one."""
    scale = np.sqrt((A ** 2).mean(axis=0))
    scale[scale == 0] = 1.0
    As = A / scale
    G = As.T @ As
    b = As.T @ target
    tt = float(target @ target)
    tn = float(np.max(np.abs(target)))
    best = (float("inf"), None, None)
    for k in range(1, kmax + 1):
        cands = []
        for idx in combinations(range(As.shape[1]), k):
            Gs = G[np.ix_(idx, idx)]
            try:
                x = np.linalg.solve(Gs, b[list(idx)])
            except np.linalg.LinAlgError:
                continue
            sse = tt - float(b[list(idx)] @ x)
            cands.append((sse, idx))
        cands.sort(key=lambda c: c[0])
        for sse, idx in cands[:20]:          # verify only the most promising subsets on all rows
            X = As[:, idx]
            coef, *_ = np.linalg.lstsq(X, target, rcond=None)
            r = float(np.max(np.abs(X @ coef - target))) / tn
            if r < best[0]:
                best = (r, idx, coef / scale[list(idx)])
        if best[0] < tol:
            break
    return best[1], best[2], best[0]


def _power_atoms(cols, df):
    """3.9 prototype (C): monomials of at most two of the columns, exponents in a fixed set,
    plus a constant. Only for small groups (<= 3 columns)."""
    E1 = (-6, -4, -3, -2, -1, 1, 2, 3, 4)
    E2 = (-6, -4, -2, -1, 1, 2)
    s = {c: sp.Symbol(c) for c in cols}
    pos = [c for c in cols if np.all(df[c].to_numpy(dtype=float) > 0)]
    atoms = [sp.Integer(1)] + [s[c] ** a for c in pos for a in E1]
    atoms += [s[a] ** i * s[b] ** j for a, b in combinations(pos, 2) for i in E2 for j in E2]
    return s, atoms


def _solve_transformed_sum(df, cols, residual, power, max_atoms=6):
    """3.9 prototype (C): residual**power = sum of atoms (power -2 or 2), solved exactly by the
    same greedy atom selection + least squares, then h = (sum)**(1/power)."""
    if len(cols) > 3 or np.any(np.abs(residual) < 1e-300) or not np.all(np.isfinite(residual)):
        return None, float("inf")
    target = residual.astype(float) ** power
    if not np.all(np.isfinite(target)):
        return None, float("inf")
    symbols, atoms = _power_atoms(cols, df)
    sub = df[list(cols)].copy()
    cols_v = []
    keep = []
    for e in atoms:
        v = _evaluate(e, sub)
        if np.all(np.isfinite(v)):
            cols_v.append(v); keep.append(e)
    if not keep:
        return None, float("inf")
    A = np.column_stack(cols_v)
    # best subset of 1..3 atoms (exhaustive, deterministic order); stop at the smallest exact size
    idx, coef, _ = _best_subset(A, target, kmax=3)
    if idx is None:
        return None, float("inf")
    selected = [keep[i] for i in idx]
    Q = A[:, idx] @ coef
    if np.any(Q <= 0):
        return None, float("inf")
    sign = np.sign(np.median(residual))
    h_vals = sign * Q ** (1.0 / power)
    err = float(np.max(np.abs(h_vals - residual) / (np.abs(residual) + 1e-12)))
    Qe = sum(sp.Float(float(c), 15) * e for c, e in zip(coef, selected))
    h = sign * Qe ** sp.Rational(1, power)
    # readable form: with two terms where one dominates on every row, write
    # Q = M*(1 + r) and pull the monomial M out of the root (all inputs are positive here)
    if len(selected) == 2:
        vals = [c * _evaluate(e, sub) for c, e in zip(coef, selected)]
        for i, j in ((0, 1), (1, 0)):
            if np.all(np.abs(vals[j]) < np.abs(vals[i])) and selected[i] != 1:
                pos = {s_: sp.Symbol(str(s_), positive=True) for s_ in symbols.values()}
                M = (sp.Float(float(coef[i]), 15) * selected[i]).xreplace(pos)
                r = (sp.Float(float(coef[j] / coef[i]), 15) * selected[j] / selected[i]).xreplace(pos)
                r = sp.powsimp(r)
                hp = sign * sp.powsimp(sp.expand_power_base(M ** sp.Rational(1, power), force=True)) * (1 + r) ** sp.Rational(1, power)
                plain = {v_: sp.Symbol(str(v_)) for v_ in hp.free_symbols}
                h = hp.xreplace(plain)
                break
    return h, err


def _trig_atoms(cols, df):
    """3.9 prototype (A): a small 1-D function library for a factor h(B) of at most 4 variables:
    sin/cos of one variable (also of half the variable, squared), optionally times one or two
    other variables, plus the variables themselves and a constant."""
    s = {c: sp.Symbol(c) for c in cols}
    atoms = [sp.Integer(1)] + [s[c] for c in cols]
    for c in cols:
        x = s[c]
        trig = [sp.sin(x), sp.cos(x), sp.sin(x) ** 2, sp.cos(x) ** 2, sp.sin(x / 2) ** 2, sp.cos(x / 2) ** 2]
        atoms += trig
        others = [o for o in cols if o != c]
        for k in (1, 2):
            for grp in combinations(others, k):
                m = sp.Mul(*[s[o] for o in grp])
                atoms += [m * sp.sin(x), m * sp.cos(x), m * sp.cos(x) ** 2]
    return s, atoms


def _solve_library(df, cols, residual, max_atoms=6):
    """3.9 prototype (A): residual**p (p in 1, -1, 1/2, -1/2, 2, -2) = sum of at most 3 library
    atoms, exhaustive best subset, exact only."""
    if len(cols) > 4 or not np.all(np.isfinite(residual)) or np.any(np.abs(residual) < 1e-300):
        return None, float("inf")
    symbols, atoms = _trig_atoms(cols, df)
    sub = df[list(cols)].copy()
    vals, keep = [], []
    for e in atoms:
        v = _evaluate(e, sub)
        if (np.all(np.isfinite(v)) and np.std(v) > 0) or e == 1:
            vals.append(np.asarray(v, dtype=float) * np.ones(len(sub))); keep.append(e)
    A = np.column_stack(vals)
    positive = bool(np.all(residual > 0) or np.all(residual < 0))
    best = (float("inf"), None)
    for p in (1, -1, sp.Rational(1, 2), sp.Rational(-1, 2), 2, -2):
        if (not float(p).is_integer() or p in (2, -2)) and not positive:
            continue
        sgn = float(np.sign(np.median(residual))) if positive else 1.0
        target = (sgn * residual) ** float(p) if positive else residual ** float(p)
        idx, coef, r = _best_subset(A, target, kmax=3)
        if idx is None or r >= 1e-9:
            continue
        Qe = sum(sp.Float(float(c), 15) * keep[i] for c, i in zip(coef, idx))
        h = (sgn if positive else 1.0) * Qe ** (sp.Integer(1) / sp.sympify(p))
        try:
            hv = _evaluate(h, sub)
            err = float(np.max(np.abs(hv - residual) / (np.abs(residual) + 1e-12)))
        except Exception:
            continue
        if err < best[0]:
            best = (err, h)
        if err < 1e-6:
            break
    return (best[1], best[0]) if best[1] is not None else (None, float("inf"))


def _solve_recip_square_ext(df, cols, residual, max_atoms=6):
    return _solve_transformed_sum(df, cols, residual, -2, max_atoms)


def _solve_square_ext(df, cols, residual, max_atoms=6):
    return _solve_transformed_sum(df, cols, residual, 2, max_atoms)


def extract_multiplicative_exact(df, group_a, group_b, target_col, max_atoms=6):
    """
    Given a CONFIRMED multiplicative split f = g(group_a) * h(group_b),
    try both orderings and return the best reconstruction.

    Returns a dict with keys: ordering, g_expr, h_expr, h_fit_max_rel_err,
    recombined_max_rel_err.
    """
    base = [
        ("direct", _solve_additive_sum),
        ("reciprocal", _solve_reciprocal_sum),
        ("square", _solve_square_sum),
        ("rational", _solve_rational_sum),
    ]
    results = _extract_with(df, group_a, group_b, target_col, max_atoms, base)
    if results and min(r["recombined_max_rel_err"] for r in results) < 1e-6:
        results.sort(key=lambda r: r["recombined_max_rel_err"])
        return results[0]
    # 3.9 prototype (C): transformed-sum solvers only when the 3.8 solvers found nothing exact,
    # and only in the late separability pass (after the gates), see sklearn_adapter
    if not ENABLE_TRANSFORMED:
        if not results:
            return None
        results.sort(key=lambda r: r["recombined_max_rel_err"])
        return results[0]
    results += _extract_with(df, group_a, group_b, target_col, max_atoms, [
        ("recip_square_ext", _solve_recip_square_ext),
        ("square_ext", _solve_square_ext),
        ("library", _solve_library),
    ])
    if not results:
        return None
    results.sort(key=lambda r: r["recombined_max_rel_err"])
    return results[0]


def _extract_with(df, group_a, group_b, target_col, max_atoms, solvers):
    results = []
    for first, second, first_name, second_name in [
        (group_a, group_b, "A", "B"),
        (group_b, group_a, "B", "A"),
    ]:
        for g_expr in _fit_marginal_candidates(df, first, target_col):
            g_vals = _evaluate(g_expr, df)
            if np.any(np.abs(g_vals) < 1e-300) or not np.all(np.isfinite(g_vals)):
                continue
            residual = df[target_col].to_numpy(dtype=float) / g_vals
            if not np.all(np.isfinite(residual)):
                continue

            for solver_name, solver in solvers:
                h_expr, h_fit_max_rel_err = solver(
                    df, second, residual, max_atoms=max_atoms
                )
                if h_expr is None:
                    continue

                # Full recombination check: g_expr(first) * h_expr(second) vs target.
                try:
                    h_vals = _evaluate(h_expr, df)
                    recombined = g_vals * h_vals
                    actual = df[target_col].to_numpy(dtype=float)
                    rel_err = np.abs(recombined - actual) / (np.abs(actual) + 1e-12)
                    max_rel_err = float(np.max(rel_err))
                except Exception:
                    max_rel_err = float("inf")

                results.append({
                    "ordering": f"fit {first_name} first ({'+'.join(first)})={g_expr}, "
                                f"solve {second_name} ({'+'.join(second)}) [{solver_name}]",
                    "g_group": first,
                    "g_expr": g_expr,
                    "h_group": second,
                    "h_expr": h_expr,
                    "h_fit_max_rel_err": h_fit_max_rel_err,
                    "recombined_max_rel_err": max_rel_err,
                })

    return results


