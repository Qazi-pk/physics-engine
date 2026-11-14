"""readable.py -- PIR 3.11 (W1): readable forms for EXACT models.

The 3.6-3.9 exact stages sometimes return a correct law in a form that SRBench's scorer cannot
verify without assuming positive variables (SRBench issue #235), e.g.

    1.0*sqrt(gamma)*sqrt(pr)/sqrt(rho)                 instead of  sqrt(gamma*pr/rho)
    q/(eps*r*sqrt(1.869 - 3.739*v/c + 1.869*v**2/c**2)) instead of  q/(eps*r*(1 - v/c))

The 3.10 noise-tolerant stage already writes its own output in verifiable forms (rule V8) but, by
rule V1, never touches an exact model. This module applies the same idea to any exact final model:

  W1a  merge roots of a product            (powsimp/powdenest with force=True)
  W1b  a perfect square under a root       sqrt(c*(poly)**2) -> sqrt(c)*poly  (numeric coefficients
       are rationalised relative to the largest one, tolerance 1e-6, then factored)
  W1c  numeric constants folded            sqrt(1.869) -> 1.367...

A rewrite is used only if it evaluates to the current model on the training rows (max absolute
difference <= 1e-9 x max|y|) and has no more operations than the current form. The rewrite changes
how the law is written, never which law is returned.
"""
import numpy as np
import sympy as sp

REL_TOL = 1e-9


def _fold(expr):
    """Fold numeric powers (Float**Rational) into Floats."""
    return expr.replace(lambda e: e.is_Pow and e.base.is_Float and e.exp.is_Rational,
                        lambda e: sp.Float(float(e), 15))


def _merge_roots(expr):
    """W1a: x**(1/2)*y**(3/2)/z**(1/2) -> sqrt(x*y**3/z) (one root per product; other factors kept)."""
    def one(m):
        if not isinstance(m, sp.Mul):
            return m
        inner, rest = [], []
        for f in sp.Mul.make_args(m):
            if isinstance(f, sp.Pow) and f.exp.is_Rational and f.exp.q == 2:
                inner.append(f.base ** (f.exp * 2))
            else:
                rest.append(f)
        if len(inner) < 2:
            return m
        return sp.Mul(*rest) * sp.sqrt(sp.Mul(*inner))
    return expr.replace(lambda e: isinstance(e, sp.Mul), one)


def _unsquare(expr):
    """W1b: sqrt(poly) -> sqrt(c)*p (and the sign-flipped variant) when poly == c*p**2 after
    rationalising the coefficients. Returns the list of rewritten variants (possibly empty)."""
    outs = []
    for pw in sorted(expr.atoms(sp.Pow), key=sp.count_ops, reverse=True):
        ex = pw.exp
        if not (ex.is_Rational and ex.q == 2 and isinstance(pw.base, sp.Add) and pw.base.free_symbols):
            continue
        base = pw.base
        coeffs = [abs(float(t.as_coeff_Mul()[0])) for t in sp.Add.make_args(base)]
        lead = max(coeffs)
        if not lead > 0:
            continue
        try:
            nb = sp.nsimplify(sp.expand(base / lead), rational=True, tolerance=1e-6)
            fac = sp.factor(nb)
        except Exception:
            continue
        const, rest = fac.as_coeff_Mul()
        roots = []
        for f in sp.Mul.make_args(rest):
            if isinstance(f, sp.Pow) and f.exp.is_Integer and f.exp % 2 == 0:
                roots.append(f.base ** (f.exp // 2))
            else:
                roots = None
                break
        if not roots or const < 0:
            continue
        root = sp.Mul(*roots)
        for sgn in (1, -1):
            new = sp.sqrt(sp.Float(lead, 15) * const) * sgn * root
            outs.append(expr.xreplace({pw: new ** (2 * ex)}))
    return outs


def readable_exact(expr, symbols, Xv, yv, tol=REL_TOL):
    """Return a simpler, numerically identical form of `expr`, or None."""
    Xv = np.asarray(Xv, float)
    yv = np.asarray(yv, float).ravel()
    n = len(yv)
    scale = float(max(np.max(np.abs(yv)), 1e-300))

    def ev(e):
        fn = sp.lambdify(symbols, e, modules="numpy")
        with np.errstate(all="ignore"):
            return np.broadcast_to(np.asarray(fn(*Xv.T), dtype=float), (n,))

    try:
        base = ev(expr)
        ops0 = int(sp.count_ops(expr))
    except Exception:
        return None
    cands = []
    try:
        e1 = sp.powsimp(sp.powdenest(expr, force=True), force=True)          # W1a
        cands += [e1, _merge_roots(e1)]
        for e2 in _unsquare(e1):                                             # W1b
            cands += [e2, _merge_roots(_fold(e2)), _fold(e2)]                # W1c
    except Exception:
        pass
    seen, scored = set(), []
    for c in cands:
        s = str(c)
        if s in seen or s == str(expr):
            continue
        seen.add(s)
        try:
            scored.append((int(sp.count_ops(c)), len(seen), c))
        except Exception:
            continue
    for ops, _, c in sorted(scored, key=lambda z: z[:2]):     # simplest verified form wins
        try:
            if ops > ops0:
                continue
            p = ev(c)
            if np.all(np.isfinite(p)) and float(np.max(np.abs(p - base))) <= tol * scale:
                return c
        except Exception:
            continue
    return None
