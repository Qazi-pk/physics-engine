"""
Sign-aware separability detection (PIR 3.9, rule R5; prototype B).

When y takes both signs, log|y| diverges near the zero crossing and the cross-term test
(crossterm.py) links wrong variables. For y = g(A) * h(B) the identity

    y * d2y/(da db) - (dy/da) * (dy/db) = 0      for every a in A, b in B

holds with no logarithm and no division. The derivatives come from a degree-4 polynomial
fit of y in standardized log-inputs (inputs must be positive). Pair score = rms of the
identity, normalized; a pair is linked if its score >= TAU_D. Linked pairs form blocks
(connected components), exactly as in crossterm.block_split.

Used only when y has both signs and every input is positive; otherwise the 3.8 functions
are returned unchanged.
"""
import itertools

import numpy as np

from .crossterm import N_MAX, block_split as _block_split_38, crossterm_search as _search_38

TAU_D = 0.5
DEG = 4
MAX_ROWS = 20_000


def _applies(df, target):
    y = df[target].to_numpy(dtype=float)
    feats = [c for c in df.columns if c != target]
    if not (np.any(y > 0) and np.any(y < 0)) or not (2 <= len(feats) <= N_MAX):
        return False
    X = df[feats].to_numpy(dtype=float)
    return bool(np.all(np.isfinite(X)) and np.all(X > 0) and np.all(np.isfinite(y)))


def pair_scores(df, target="target"):
    feats = [c for c in df.columns if c != target]
    n = len(feats)
    X = df[feats].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=float)
    if len(y) > MAX_ROWS:
        idx = np.random.default_rng(0).choice(len(y), MAX_ROWS, replace=False)
        X, y = X[idx], y[idx]
    Z = np.log(X)
    Z = (Z - Z.mean(0)) / Z.std(0)
    terms = [()]
    for d in range(1, DEG + 1):
        terms += list(itertools.combinations_with_replacement(range(n), d))

    def mono(t, dv=()):
        c = np.ones(len(Z))
        cnt = {k: t.count(k) for k in set(t)}
        for v in dv:
            if cnt.get(v, 0) == 0:
                return np.zeros(len(Z))
            c = c * cnt[v]
            cnt[v] -= 1
        for k, p in cnt.items():
            if p:
                c = c * Z[:, k] ** p
        return c

    A = np.column_stack([mono(t) for t in terms])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    fit = A @ beta
    d1 = [np.column_stack([mono(t, (i,)) for t in terms]) @ beta for i in range(n)]
    out = {}
    for i, j in itertools.combinations(range(n), 2):
        dij = np.column_stack([mono(t, (i, j)) for t in terms]) @ beta
        num = fit * dij - d1[i] * d1[j]
        den = np.sqrt(np.mean((fit * dij) ** 2)) + np.sqrt(np.mean((d1[i] * d1[j]) ** 2))
        out[(feats[i], feats[j])] = float(np.sqrt(np.mean(num ** 2)) / den) if den > 0 else 0.0
    return out


def _blocks(df, target):
    s = pair_scores(df, target)
    feats = [c for c in df.columns if c != target]
    par = {f: f for f in feats}

    def find(a):
        while par[a] != a:
            a = par[a]
        return a

    for (a, b), v in sorted(s.items()):
        if v >= TAU_D:
            par[find(a)] = find(b)
    groups = {}
    for f in feats:
        groups.setdefault(find(f), []).append(f)
    return sorted((sorted(g) for g in groups.values()), key=lambda g: (len(g), g)), s


def crossterm_search(df, target="target", nbins=12, tol=0.01, verbose=False):
    if not _applies(df, target):
        return _search_38(df, target=target, nbins=nbins, tol=tol, verbose=verbose)
    blocks, s = _blocks(df, target)
    feats = [c for c in df.columns if c != target]
    res = []
    for r in range(1, len(blocks) // 2 + 1 if len(blocks) > 1 else 1):
        for pick in itertools.combinations(range(len(blocks)), r):
            ga = sorted(sum((blocks[i] for i in pick), []))
            gb = sorted(f for f in feats if f not in ga)
            score = max(v for (a, b), v in s.items() if (a in ga) != (b in ga))
            res.append((tuple(ga), tuple(gb), "MULT" if score < TAU_D else "NONE", score, score, float("nan")))
    res.sort(key=lambda x: (x[3], x[0], x[1]))
    best = next((x for x in res if x[2] == "MULT"), None)
    return best, res


def block_split(df, target="target"):
    if not _applies(df, target):
        return _block_split_38(df, target=target)
    blocks, _ = _blocks(df, target)
    single = [b[0] for b in blocks if len(b) == 1]
    multi = sorted(sum((b for b in blocks if len(b) > 1), []))
    return (sorted(single), multi) if single and multi else None
