"""
separability_crossterm.py -- deterministic multiplicative-separability detector (PIR 3.8 candidate).

Replaces the binned-SVD score of separability_partition.partition_search, which sits at its own
noise floor (diagnosis: claude/separability_diagnosis_2026-10-26.md). Same call signature and
return shape, so it is a drop-in replacement.

Idea. If y = g(A) * h(B), then log|y| = log|g(A)| + log|h(B)|, so in a smooth fit of log|y| on the
log inputs every CROSS term between a variable in A and a variable in B must vanish. We fit a full
quadratic of t = log|y| on z_i = log|x_i| (centered) by least squares and score a split A|B by

    S(A|B) = std( sum_{i in A, j in B} beta_ij * z_i * z_j ) / std(t)

S is ~0 for a true multiplicative split and clearly positive otherwise.

Rules (FIXED A PRIORI for 3.8, before any benchmark run; see claude/pir38_plan.md):
  * applies only if every input column and the target are nonzero (logs defined);
    otherwise falls back to the original partition_search (v7 behaviour);
  * at most N_MAX = 10 inputs (2**9 - 1 = 511 splits); more inputs -> no candidates;
  * seeded subsample of at most MAX_ROWS = 20,000 rows (numpy default_rng(0));
  * a split is MULT if S < TAU = 5e-3, else NONE; results sorted by S ascending,
    ties broken by the split's sorted column names (fully deterministic).
  * R1b (added 27 Oct 2026, before V1 completed -- see claude/pir38_plan.md): variables
    joined by a pair cross-term score >= TAU form one inseparable BLOCK (connected
    components); block_split() gives the split 'all single-variable blocks | all
    multi-variable blocks', which pirsep8 tries FIRST.
"""
import itertools

import numpy as np

TAU = 5e-3
N_MAX = 10
MAX_ROWS = 20_000


def _fallback(df, target, nbins, tol, verbose):
    from .partition import partition_search
    return partition_search(df, target=target, nbins=nbins, tol=tol, verbose=verbose)


def crossterm_scores(df, target="target"):
    """Return (feats, list of (A, B, S)) or None if the test does not apply."""
    feats = [c for c in df.columns if c != target]
    n = len(feats)
    if n < 2 or n > N_MAX:
        return feats, []
    X = df[feats].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=float)
    if not (np.all(np.isfinite(X)) and np.all(np.isfinite(y))):
        return None
    if np.any(X == 0) or np.any(y == 0):
        return None
    if len(y) > MAX_ROWS:
        idx = np.random.default_rng(0).choice(len(y), MAX_ROWS, replace=False)
        X, y = X[idx], y[idx]
    Z = np.log(np.abs(X))
    Z = Z - Z.mean(axis=0)
    t = np.log(np.abs(y))
    sd = float(np.std(t))
    if not np.isfinite(sd) or sd == 0:
        return None
    pairs = list(itertools.combinations(range(n), 2))
    cols = ([np.ones(len(t))] + [Z[:, i] for i in range(n)] + [Z[:, i] ** 2 for i in range(n)]
            + [Z[:, i] * Z[:, j] for i, j in pairs])
    beta, *_ = np.linalg.lstsq(np.column_stack(cols), t, rcond=None)
    cross = {p: beta[1 + 2 * n + k] * Z[:, p[0]] * Z[:, p[1]] for k, p in enumerate(pairs)}
    out = []
    for k in range(1, n // 2 + 1):
        for A in itertools.combinations(range(n), k):
            if 2 * k == n and 0 not in A:
                continue  # each balanced split once
            Aset = set(A)
            c = np.zeros(len(t))
            for (i, j), v in cross.items():
                if (i in Aset) != (j in Aset):
                    c = c + v
            B = tuple(feats[j] for j in range(n) if j not in Aset)
            out.append((tuple(feats[i] for i in A), B, float(np.std(c) / sd)))
    return feats, out


def crossterm_search(df, target="target", nbins=12, tol=0.01, verbose=False):
    """Drop-in replacement for partition_search: returns (best, results) with
    results = [(A, B, kind, score, mult, add)] sorted by score; best = first MULT or None."""
    r = crossterm_scores(df, target)
    if r is None:
        return _fallback(df, target, nbins, tol, verbose)
    _, scored = r
    results = [(A, B, "MULT" if S < TAU else "NONE", S, S, float("nan")) for A, B, S in scored]
    results.sort(key=lambda x: (x[3], tuple(sorted(x[0])), tuple(sorted(x[1]))))
    if verbose:
        for A, B, kind, S, _, _ in results[:6]:
            print(f"  {'+'.join(A):18s} | {'+'.join(B):18s}  S={S:.2e} -> {kind}")
    best = next((x for x in results if x[2] == "MULT"), None)
    return best, results


def crossterm_blocks(df, target="target", tau_edge=TAU):
    """Variables linked by a non-negligible cross term (pair score >= tau_edge) form one
    inseparable BLOCK (connected components). Returns list of blocks (sorted) or None."""
    feats = [c for c in df.columns if c != target]
    n = len(feats)
    if n < 2 or n > N_MAX:
        return None
    X = df[feats].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=float)
    if not (np.all(np.isfinite(X)) and np.all(np.isfinite(y))) or np.any(X == 0) or np.any(y == 0):
        return None
    if len(y) > MAX_ROWS:
        idx = np.random.default_rng(0).choice(len(y), MAX_ROWS, replace=False)
        X, y = X[idx], y[idx]
    Z = np.log(np.abs(X)); Z = Z - Z.mean(axis=0); t = np.log(np.abs(y)); sd = float(np.std(t))
    if not np.isfinite(sd) or sd == 0:
        return None
    pairs = list(itertools.combinations(range(n), 2))
    cols = ([np.ones(len(t))] + [Z[:, i] for i in range(n)] + [Z[:, i] ** 2 for i in range(n)]
            + [Z[:, i] * Z[:, j] for i, j in pairs])
    beta, *_ = np.linalg.lstsq(np.column_stack(cols), t, rcond=None)
    parent = list(range(n))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for k, (i, j) in enumerate(pairs):
        if np.std(beta[1 + 2 * n + k] * Z[:, i] * Z[:, j]) / sd >= tau_edge:
            parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(feats[i])
    return sorted((sorted(g) for g in groups.values()), key=lambda g: (len(g), g))


def block_split(df, target="target"):
    """The split 'all single-variable blocks | all multi-variable blocks' (the monomial part vs the
    rest), or None if it does not exist (no singleton block or no multi-variable block)."""
    b = crossterm_blocks(df, target)
    if not b:
        return None
    single = [g[0] for g in b if len(g) == 1]
    multi = [v for g in b if len(g) > 1 for v in g]
    if not single or not multi:
        return None
    return sorted(single), sorted(multi)
