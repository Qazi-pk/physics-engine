from __future__ import annotations

from typing import Optional, Union

import sympy as sp

from .representation import Equation


ExpressionLike = Union[str, sp.Expr]
EquationLike = Union[Equation, sp.Equality, str]


def _as_expression(expr: ExpressionLike) -> sp.Expr:
    if isinstance(expr, str):
        return sp.sympify(expr)
    return expr


def _as_sympy_equation(equation: EquationLike) -> sp.Equality:
    if isinstance(equation, Equation):
        return sp.Eq(sp.sympify(equation.lhs), sp.sympify(equation.rhs))

    if isinstance(equation, sp.Equality):
        return equation

    if "=" in equation:
        lhs_text, rhs_text = equation.split("=", maxsplit=1)
        return sp.Eq(sp.sympify(lhs_text.strip()), sp.sympify(rhs_text.strip()))

    return sp.Eq(sp.sympify(equation), 0)


def canonicalize_expression(expr: ExpressionLike) -> sp.Expr:
    parsed = _as_expression(expr)
    return sp.simplify(sp.expand_mul(parsed))


def canonicalize_equation(equation: EquationLike) -> sp.Equality:
    eq = _as_sympy_equation(equation)
    lhs = canonicalize_expression(eq.lhs)
    rhs = canonicalize_expression(eq.rhs)
    return sp.Eq(lhs, rhs)


def normalize_equation(equation: EquationLike, target: Optional[str] = None) -> sp.Equality:
    eq = canonicalize_equation(equation)
    if target is None:
        if isinstance(eq.lhs, sp.Symbol):
            target_symbol = eq.lhs
        else:
            ordered_symbols = sorted(eq.free_symbols, key=lambda sym: sym.name)
            if not ordered_symbols:
                return eq
            target_symbol = ordered_symbols[0]
    else:
        target_symbol = sp.Symbol(target)

    solved = sp.solve(eq, target_symbol, dict=False)
    if not solved:
        return eq

    solution = canonicalize_expression(solved[0])
    return sp.Eq(target_symbol, solution)


def transform_equation(
    equation: Equation,
    normalize_to: Optional[str] = None,
) -> Equation:
    transformed = normalize_equation(equation, target=normalize_to)
    metadata = dict(equation.metadata)
    metadata["canonicalized"] = "true"
    metadata["normalized"] = "true" if normalize_to is not None else "false"
    if normalize_to is not None:
        metadata["normalize_target"] = normalize_to
    return Equation(
        lhs=str(transformed.lhs),
        rhs=str(transformed.rhs),
        dimensions=equation.dimensions,
        regime=equation.regime,
        metadata=metadata,
    )
