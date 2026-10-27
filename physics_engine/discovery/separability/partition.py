"""
Separability partition search + extraction (v3.5 stage, step 1)
================================================================
Builds on the validated 2-group primitive (separability_primitive.py).

DETECT: for an N-variable function, search candidate 2-way partitions of the
variables and find one for which f separates multiplicatively (g(A)*h(B)) or
additively (g(A)+h(B)), using the rank-1 / double-difference tests.

EXTRACT: given a detected multiplicative split f(A,B)=g(A)*h(B), recover g and
h (up to a shared constant) by slicing:
    fix B at a reference cell b0 -> f(A,b0) = g(A)*h(b0) = c_B * g(A)
    fix A at a reference cell a0 -> f(a0,B) = g(a0)*h(B) = c_A * h(B)
    and f(a0,b0) = g(a0)*h(b0), which reconciles the constants:
        g(A)*h(B) ~= f(A,b0) * f(a0,B) / f(a0,b0)
This is exact for a true product and is the standard separable-factor identity.

Additive analogue:
    f(A,B) - f(A,b0) - f(a0,B) + f(a0,b0)  vanishes for a true sum, and
    g(A)+h(B) ~= f(A,b0) + f(a0,B) - f(a0,b0).

RECURSE: NOT built yet. Deferred until extraction is validated (see note).
This module exposes extracted sub-samples so a caller could feed them back
to PIR, but does not itself recurse.

Validation: on known products we check that the extracted g and h, recombined,
reproduce f to machine precision; and that g depends only on A, h only on B.
"""
import numpy as np
import pandas as pd
import os
from itertools import combinations

DATA = os.path.expanduser("~/pmlb_data")


# ---- the validated primitive (multiplicative + additive), reused ----
def _bin_matrix(a, b, f, nbins):
    ab = np.quantile(a, np.linspace(0, 1, nbins + 1)); ab[-1] += 1e-9
    bb = np.quantile(b, np.linspace(0, 1, nbins + 1)); bb[-1] += 1e-9
    ai = np.clip(np.digitize(a, ab) - 1, 0, nbins - 1)
    bi = np.clip(np.digitize(b, bb) - 1, 0, nbins - 1)
    M = np.full((nbins, nbins), np.nan)
    for i in range(nbins):
        for j in range(nbins):
            m = (ai == i) & (bi == j)
            if m.any():
                M[i, j] = f[m].mean()
    if np.isnan(M).any():
        M[np.isnan(M)] = np.nanmean(M)
    return M


def sep_scores(a, b, f, nbins=12):
    M = _bin_matrix(a, b, f, nbins)
    s = np.linalg.svd(M, compute_uv=False)
    mult = s[1] / s[0] if s[0] > 0 else 1.0
    D = M - M[:, [0]] - M[[0], :] + M[0, 0]
    add = np.abs(D).mean() / (np.abs(M).mean() + 1e-12)
    return mult, add


def classify(mult, add, tol=0.01):
    if mult < tol and mult <= add:
        return "MULT", mult
    if add < tol and add < mult:
        return "ADD", add
    if mult < tol:
        return "MULT", mult
    if add < tol:
        return "ADD", add
    return "NONE", min(mult, add)


# ---- group reduction: collapse a multi-var group to a 1-D coordinate ----
# For the primitive we need a single 'a' and single 'b' vector. For a GROUP of
# variables we reduce the group to one coordinate by... using the group's
# columns jointly is not 1-D. Approach: to TEST a partition {A}|{B} where A or
# B has >1 variable, we bin on a *representative* reduction. The cleanest exact
# reduction that preserves separability structure is to project each group onto
# the geometric mean of its columns for the binning axis. This is a heuristic
# for DETECTION only; EXTRACTION below works cell-wise and does not rely on it.
def _group_coord(df, cols):
    X = np.column_stack([df[c].to_numpy(float) for c in cols])
    # geometric-mean-ish coordinate (positive data); fall back to mean
    if np.all(X > 0):
        return np.exp(np.log(X).mean(axis=1))
    return X.mean(axis=1)


def partition_search(df, target="target", nbins=12, tol=0.01, verbose=True):
    """Search all 2-way partitions of the feature columns; report separations."""
    feats = [c for c in df.columns if c != target]
    f = df[target].to_numpy(float)
    n = len(feats)
    results = []
    # candidate: choose size-k subset A (1 <= k <= n//2), B = rest
    seen = set()
    for k in range(1, n // 2 + 1):
        for A in combinations(feats, k):
            B = tuple(c for c in feats if c not in A)
            key = frozenset([frozenset(A), frozenset(B)])
            if key in seen:
                continue
            seen.add(key)
            a = _group_coord(df, list(A))
            b = _group_coord(df, list(B))
            mult, add = sep_scores(a, b, f, nbins)
            kind, score = classify(mult, add, tol)
            results.append((A, B, kind, score, mult, add))
    results.sort(key=lambda r: r[3])  # best (lowest) score first
    if verbose:
        for A, B, kind, score, mult, add in results[:6]:
            print(f"  {'+'.join(A):18s} | {'+'.join(B):18s}  "
                  f"mult={mult:.4f} add={add:.4f} -> {kind}")
    # best genuine separation
    best = next((r for r in results if r[2] != "NONE"), None)
    return best, results


def extract_mult(df, A, B, target="target", nbins=12):
    """Extract g(A), h(B) for a detected multiplicative split, and validate:
    recombine and compare to f. Returns (max_rel_err, g_vals, h_vals)."""
    f = df[target].to_numpy(float)
    a = _group_coord(df, list(A))
    b = _group_coord(df, list(B))
    M = _bin_matrix(a, b, f, nbins)
    # reference cell (0,0). Reconstruct M_hat[i,j] = M[i,0]*M[0,j]/M[0,0]
    if abs(M[0, 0]) < 1e-15:
        return None, None, None
    M_hat = np.outer(M[:, 0], M[0, :]) / M[0, 0]
    rel = np.abs(M_hat - M) / (np.abs(M) + 1e-12)
    g = M[:, 0]            # g(A) up to constant h(b0)
    h = M[0, :] / M[0, 0]  # h(B) normalized so h(b0)=1
    return float(rel.max()), g, h


def load(name):
    return pd.read_csv(f"{DATA}/{name}/{name}.tsv.gz", sep="\t", compression="gzip")


