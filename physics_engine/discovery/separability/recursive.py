"""
separability_recursive.py -- recursive multiplicative re-splitting for
residuals that extract_multiplicative_exact's flat solvers can't represent
in one shot (e.g. feynman_II_6_15a's h(epsilon,p_d,x,y,z), which is itself
a nested product of a rational-monomial part and sqrt(x**2+y**2)).

Deliberately a brand-new file, not a modification to separability_exact.py
or separability_partition.py -- see "Future work: recursive re-splitting
for II_6_15a" in SEPARABILITY_CASE1_RECOVERY_NOTE.md for the isolation
rationale (interleaving detection inside extraction is new cross-module
coupling; keeping it in a separate module makes `git diff --stat` on the
two existing modules show zero lines). Both extract_multiplicative_exact
and partition_search are imported and called UNMODIFIED.

ARCHITECTURE (per the note's plan): extract_recursive() calls the existing
extract_multiplicative_exact() as a base attempt. If the base attempt's
h(group_b) doesn't fully recover the target (recombined_max_rel_err above
FULL_RECOVERY_THRESHOLD) and depth budget remains, it injects the numeric
residual (target / g_hat(group_a)) as a pseudo-column -- same idiom
_solve_additive_sum already uses -- and runs partition_search() on
group_b's own columns to look for a clean sub-split; if found (and it's a
genuine MULT split, not ADD or NONE), it recurses on the residual with the
detected sub-groups. Combines g(A) * h(B) * ... from whatever depth it
bottoms out at.

TERMINATION: two independent guarantees -- group_b's column count strictly
shrinks each recursive call (partition_search can't find a split below 2
columns, so recursion structurally bottoms out regardless of any explicit
limit), plus an explicit max_depth cap (default 1) as a defensive/scope
limiter -- no dataset characterized so far shows evidence of needing more
than one extra level.

HONESTY: the public entry point never trusts an intermediate per-level
error as the final answer. Whatever depth the recursion bottoms out at,
it recombines ALL collected factors and re-checks against the ORIGINAL
target column in the ORIGINAL dataframe before reporting "recovered".
"""
import os
import sys

import numpy as np

from .exact import (  # package port: was separability_exact
    extract_multiplicative_exact, _evaluate, _fit_marginal_monomial,
    _solve_additive_sum, _solve_reciprocal_sum, _solve_square_sum, _solve_rational_sum,
)
# package port, rule R1 (PIR 3.8): the cross-term detector replaces partition_search here too
from .crossterm import crossterm_search as partition_search

FULL_RECOVERY_THRESHOLD = 1e-6

# Cap on how many ranked partition_search candidates _extract_recursive_inner
# will try per level before giving up -- see the fallback comment at its call
# site. Bounds cost on genuinely non-separable residuals (every candidate
# tried, none recover) while covering the II_6_15a case (true split at rank 6
# of the top-6 shown).
_MAX_FALLBACK_CANDIDATES = 8

_SOLVERS = (_solve_additive_sum, _solve_reciprocal_sum, _solve_square_sum, _solve_rational_sum)


def _fit_pinned_ordering(df, prefactor_group, residual_group, target_col, max_atoms=6):
    """
    Like ONE direction of extract_multiplicative_exact's loop, but PINNED:
    always fits prefactor_group as g first, solves residual_group as h
    second -- never tries the reverse.

    Needed (found by testing, not assumed): extract_multiplicative_exact
    tries BOTH orderings and returns whichever scores lowest. On a
    synthetic f = A * (B * sqrt(C**2+D**2)) case, the 'wrong' ordering (fit
    a 3-column log-linear approximation on {B,C,D} ignoring A, solve A as
    the residual) scored a LOWER recombination error than the 'right'
    ordering (exact 1-column fit on A, unsolved residual on {B,C,D}) --
    because a merely-approximate multi-column monomial fit can beat an
    exact single-column fit paired with a residual none of the 4 flat
    solvers can represent. That silently handed back the small group
    (length 1) as h_group, which can't be recursed into further, defeating
    recursion on exactly the case it needs to handle. Pinning the ordering
    keeps the caller-specified residual_group as h_group always, so
    recursion always has a well-defined side to keep splitting.

    Reuses _fit_marginal_monomial, _evaluate, and all 4 solvers unmodified
    from separability_exact.py. Deliberately uses ONLY the plain
    log-linear monomial candidate (_fit_marginal_monomial), NOT the full
    _fit_marginal_candidates list that also tries +1/-1 shifted variants
    (found by testing: ranking the +1/-1 shift candidates against a
    residual that's out-of-grammar for all 4 solvers is unreliable -- the
    max-relative-error metric can favor a spurious shifted candidate whose
    error happens to spread more evenly, over the numerically-exact plain
    candidate, precisely in the not-yet-fully-recovered regime recursion
    operates in). The +1/-1 shift extension was built for a different,
    narrower purpose (reciprocal-of-affine factors on a group already
    confirmed to fully recover); recursion's job is peeling off a clean
    monomial prefactor, which is exactly what the plain candidate is for.

    Only the "try both orderings, keep best" orchestration from
    extract_multiplicative_exact is not reused, since that orchestration
    is exactly what needs to NOT happen here.
    """
    g_expr = _fit_marginal_monomial(df, prefactor_group, target_col)
    if g_expr is None:
        return None

    g_vals = _evaluate(g_expr, df)
    if np.any(np.abs(g_vals) < 1e-300) or not np.all(np.isfinite(g_vals)):
        return None
    residual = df[target_col].to_numpy(dtype=float) / g_vals
    if not np.all(np.isfinite(residual)):
        return None

    best = None
    for solver in _SOLVERS:
        h_expr, _h_fit_err = solver(df, residual_group, residual, max_atoms=max_atoms)
        if h_expr is None:
            continue
        try:
            h_vals = _evaluate(h_expr, df)
            recombined = g_vals * h_vals
            actual = df[target_col].to_numpy(dtype=float)
            rel_err = np.abs(recombined - actual) / (np.abs(actual) + 1e-12)
            max_rel_err = float(np.max(rel_err))
        except Exception:
            max_rel_err = float("inf")

        if best is None or max_rel_err < best["recombined_max_rel_err"]:
            best = {
                "g_group": list(prefactor_group), "g_expr": g_expr,
                "h_group": list(residual_group), "h_expr": h_expr,
                "recombined_max_rel_err": max_rel_err,
            }
    return best


def _extract_recursive_inner(df, group_a, group_b, target_col, max_atoms,
                              max_depth, nbins, tol, depth):
    """Collects factors level by level. Each level's own recombined error is
    used only to DECIDE whether to recurse further -- the caller-facing
    extract_recursive() recomputes the authoritative error at the end.

    Fast path: try extract_multiplicative_exact (both orderings) first --
    if it fully recovers in one shot, take it as-is, no need to care which
    side it called g vs h. Only fall back to the ordering-PINNED fit (see
    _fit_pinned_ordering) when the fast path does NOT fully recover, so
    group_b is guaranteed to be the side available for recursion."""
    fast = extract_multiplicative_exact(df, group_a, group_b, target_col, max_atoms=max_atoms)
    if fast is not None and fast["recombined_max_rel_err"] < FULL_RECOVERY_THRESHOLD:
        return {
            "factors": [(fast["g_group"], fast["g_expr"]), (fast["h_group"], fast["h_expr"])],
            "depth_reached": depth, "note": "",
        }

    base = _fit_pinned_ordering(df, group_a, group_b, target_col, max_atoms=max_atoms)
    if base is None:
        return None

    g_group, g_expr = base["g_group"], base["g_expr"]
    h_group, h_expr = base["h_group"], base["h_expr"]
    err = base["recombined_max_rel_err"]
    factors_so_far = [(g_group, g_expr), (h_group, h_expr)]

    if err < FULL_RECOVERY_THRESHOLD:
        return {"factors": factors_so_far, "depth_reached": depth, "note": ""}

    if depth >= max_depth:
        return {
            "factors": factors_so_far, "depth_reached": depth,
            "note": f"NOT RECOVERED at max_depth={max_depth} -- partial result only (local err {err:.3e})",
        }

    if len(h_group) < 2:
        return {
            "factors": factors_so_far, "depth_reached": depth,
            "note": "NOT RECOVERED -- leaf group has <2 columns, cannot split further",
        }

    g_vals = _evaluate(g_expr, df)
    if np.any(np.abs(g_vals) < 1e-300) or not np.all(np.isfinite(g_vals)):
        return {
            "factors": factors_so_far, "depth_reached": depth,
            "note": "NOT RECOVERED -- g(group_a) not usable for residual division",
        }
    residual = df[target_col].to_numpy(dtype=float) / g_vals
    if not np.all(np.isfinite(residual)):
        return {
            "factors": factors_so_far, "depth_reached": depth,
            "note": "NOT RECOVERED -- residual not finite",
        }

    # Inject the residual as a pseudo-column (same idiom _solve_additive_sum
    # uses) and search for a clean sub-split within group_b alone.
    sub = df[list(h_group)].copy()
    sub["__residual__"] = residual

    top_best, results = partition_search(sub, target="__residual__", nbins=nbins, tol=tol, verbose=False)
    if top_best is None:
        return {
            "factors": factors_so_far, "depth_reached": depth,
            "note": "NOT RECOVERED -- no clean sub-split found in residual",
        }

    # Fallback over ranked candidates (found by testing on feynman_II_6_15a,
    # 2026-08-19): partition_search's top-ranked (lowest-score) split can be a
    # detection artifact from _group_coord's geometric-mean projection on
    # multi-column groups, not the genuine structure. On II_6_15a the TRUE
    # sub-split ({x,y} sqrt part vs {epsilon,p_d,z} monomial part) scored
    # mult=0.0105, just above `tol=0.01`, so classify() labeled it NONE and
    # it never became `best` -- while a spurious 1-column split scored
    # 0.0030 and won, sending recursion down a dead-end branch that then
    # exhausted the whole max_depth budget. Trusting only rank 0 (the prior
    # behavior) can't recover in that situation.
    #
    # This is safe to broaden: the ultimate arbiter of correctness is the
    # numeric recombination check against the real residual (ends up
    # re-verified against the ORIGINAL target by extract_recursive()'s
    # caller-facing wrapper too), not partition_search's score/classification
    # -- a spurious split reaching FULL_RECOVERY_THRESHOLD agreement on real
    # data by coincidence is not a realistic risk (confirmed empirically:
    # known non-separable artifacts feynman_I_18_4/I_11_19/II_36_38 still
    # correctly report NOT RECOVERED after this change). ADD-classified
    # candidates are still skipped -- this recursive path only composes
    # g(A)*h(B) products. Capped at _MAX_FALLBACK_CANDIDATES to bound cost on
    # genuinely non-separable inputs, where every candidate is tried and none
    # recover.
    candidates = [r for r in results if r[2] != "ADD"][:_MAX_FALLBACK_CANDIDATES]
    if not candidates:
        return {
            "factors": factors_so_far, "depth_reached": depth,
            "note": "NOT RECOVERED -- no MULT/NONE sub-split candidates in residual",
        }

    best_partial = None
    for subA, subB, kind, score, mult, add in candidates:
        sub_result = _extract_recursive_inner(
            sub, list(subA), list(subB), "__residual__", max_atoms, max_depth, nbins, tol, depth + 1,
        )
        if sub_result is None:
            continue
        candidate_result = {
            "factors": [(g_group, g_expr)] + sub_result["factors"],
            "depth_reached": sub_result["depth_reached"],
            "note": sub_result["note"],
        }
        if not sub_result["note"]:
            return candidate_result
        if best_partial is None:
            best_partial = candidate_result

    if best_partial is not None:
        return best_partial

    return {
        "factors": factors_so_far, "depth_reached": depth,
        "note": "NOT RECOVERED -- no candidate sub-split recovered",
    }


def extract_recursive(df, group_a, group_b, target_col="target", max_atoms=6,
                       max_depth=1, nbins=12, tol=0.01):
    """
    Recursive multiplicative extraction: f = g(group_a) * h(group_b), where
    h(group_b) may itself be a nested product needing further splitting.

    Returns a dict:
      {
        "recovered": bool,                # True iff recombined_max_rel_err < 1e-6
        "factors": [(cols, expr), ...],    # one entry per level, in extraction order
        "recombined_max_rel_err": float,   # against the ORIGINAL target_col in df
        "depth_reached": int,
        "note": str,                       # empty on full recovery, else why it stopped
      }
    or None if the depth-0 base extraction itself produced no usable ordering
    at all (mirrors extract_multiplicative_exact's own None contract).

    On datasets that don't need recursion (base extraction fully recovers on
    the first try), this is IDENTICAL to calling extract_multiplicative_exact
    directly -- recursion never fires, depth_reached is 0.
    """
    result = _extract_recursive_inner(
        df, group_a, group_b, target_col, max_atoms, max_depth, nbins, tol, depth=0,
    )
    if result is None:
        return None

    # Never trust an intermediate per-level error as the final answer --
    # recombine ALL collected factors and re-check against the ORIGINAL
    # target column in the ORIGINAL dataframe. Evaluate each factor
    # SEPARATELY as a numeric array and multiply the arrays -- found by
    # testing that evaluating one combined symbolic product (factor_1 *
    # factor_2 * ...) loses precision when one factor is a large
    # least-squares-fit expression carrying many near-zero (~1e-17) noise
    # coefficient terms: sympy's evaluation of the merged symbolic product
    # amplified a 3.7e-13 per-factor error to 2.5e-4 in testing, while
    # multiplying the independently-evaluated numeric arrays stayed at
    # 3.7e-13. Same-magnitude tiny coefficients are already present in
    # every one of the 11 solved cases' h_expr (see
    # independent_separability_verification.py) -- they only became a
    # visible problem here because recursion multiplies TWO such
    # expressions together instead of evaluating just one.
    combined_vals = np.ones(len(df), dtype=float)
    for cols, expr in result["factors"]:
        combined_vals = combined_vals * _evaluate(expr, df)
    actual = df[target_col].to_numpy(dtype=float)
    rel_err = np.abs(combined_vals - actual) / (np.abs(actual) + 1e-12)
    overall_err = float(np.max(rel_err))

    recovered = overall_err < FULL_RECOVERY_THRESHOLD
    note = result["note"]
    if recovered:
        note = ""
    elif not note:
        note = "NOT RECOVERED -- combined recombination above threshold"

    return {
        "recovered": recovered,
        "factors": result["factors"],
        "recombined_max_rel_err": overall_err,
        "depth_reached": result["depth_reached"],
        "note": note,
    }


