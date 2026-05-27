from __future__ import annotations

from fractions import Fraction
from itertools import combinations_with_replacement, product as iproduct

import sympy as sp


PRODUCT_EXPONENTS = [
    Fraction(-2),
    Fraction(-1),
    Fraction(1),
    Fraction(2),
    Fraction(1, 2),
    Fraction(3, 2),
]


COMPOUND_ANGLE_TEMPLATES = [
    "l1 * sin(q1)",
    "l1 * cos(q1)",
    "l2 * sin(q2)",
    "l2 * cos(q2)",
    "l2 * sin(q1 + q2)",
    "l2 * cos(q1 + q2)",
    "l1 * sin(q1 + q2)",
    "l1 * cos(q1 + q2)",
    "l1 * sin(q1) + l2 * sin(q1 + q2)",
    "l1 * cos(q1) + l2 * cos(q1 + q2)",
    "-l1 * sin(q1) - l2 * sin(q1 + q2)",
    "-l1 * cos(q1) - l2 * cos(q1 + q2)",
    "-l2 * sin(q1 + q2)",
    "l2 * cos(q1 + q2)",
    "-l2 * cos(q1 + q2)",
    "sin(q1) + sin(q1 + q2)",
    "cos(q1) + cos(q1 + q2)",
    "-sin(q1) - sin(q1 + q2)",
    "-cos(q1) - cos(q1 + q2)",
]


def build_basis_3var_products(
    variables: list[str],
    symbols: dict[str, sp.Symbol],
    exponent_set: list | tuple | None = None,
) -> list[sp.Expr]:
    """
    Build symbolic 3-variable product terms:
        phi(x) = x_i^a * x_j^b * x_k^c

    Terms use variable triples with replacement (e.g., i, i, j), while
    excluding pure cubic self-products (i, i, i).
    """
    if exponent_set is None:
        exponent_set = PRODUCT_EXPONENTS

    exponent_syms = [sp.Rational(e) for e in exponent_set]
    terms: list[sp.Expr] = []

    for var1, var2, var3 in combinations_with_replacement(variables, 3):
        if var1 == var2 == var3:
            continue
        for a, b, c in iproduct(exponent_syms, repeat=3):
            if a == 0 or b == 0 or c == 0:
                continue
            expr = (symbols[var1] ** a) * (symbols[var2] ** b) * (symbols[var3] ** c)
            terms.append(sp.simplify(expr))

    return terms


def build_compound_angle_library(
    variables: list[str] | None = None,
    include_negatives: bool = True,
    include_sums: bool = True,
) -> list[str]:
    """Build candidate expressions containing compound angles for planar robot tasks."""
    if variables is None:
        variables = ["l1", "l2", "q1", "q2"]

    candidates = list(COMPOUND_ANGLE_TEMPLATES)

    lengths = ["l1", "l2"]
    angles = ["q1", "q2"]
    compound_angles = ["q1 + q2", "q1 - q2", "q2 - q1"]
    trig_fns = ["sin", "cos"]

    for length in lengths:
        for fn in trig_fns:
            for angle in angles + compound_angles:
                expr = f"{length} * {fn}({angle})"
                if expr not in candidates:
                    candidates.append(expr)
                if include_negatives:
                    neg_expr = f"-{length} * {fn}({angle})"
                    if neg_expr not in candidates:
                        candidates.append(neg_expr)

    if include_sums:
        for left in lengths:
            for right in lengths:
                for fn in trig_fns:
                    for angle_a in angles:
                        for angle_b in compound_angles:
                            expr = f"{left} * {fn}({angle_a}) + {right} * {fn}({angle_b})"
                            if expr not in candidates:
                                candidates.append(expr)
                            if include_negatives:
                                neg_expr = f"-{left} * {fn}({angle_a}) - {right} * {fn}({angle_b})"
                                if neg_expr not in candidates:
                                    candidates.append(neg_expr)

    return candidates
