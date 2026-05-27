"""
units.py

Expression-level dimensional evaluation using canonical Dimension objects.

This module:
- Traverses SymPy expressions
- Computes dimensions using provided symbol → Dimension mappings
- Enforces dimensional consistency as a HARD constraint
"""

from sympy import Add, Mul, Pow, Symbol, Eq
from .dimensions import Dimension, DimensionError


def dimension_of(expr, symbol_dimensions: dict[Symbol, Dimension]) -> Dimension:
    """
    Recursively compute the physical dimension of a SymPy expression.

    Parameters
    ----------
    expr : sympy expression
        The expression whose dimension is to be evaluated.
    symbol_dimensions : dict[sympy.Symbol, Dimension]
        Mapping of symbols to their physical dimensions.

    Returns
    -------
    Dimension
        The resulting physical dimension.

    Raises
    ------
    DimensionError
        If dimensions are missing or inconsistent.
    """

    # Symbol
    if isinstance(expr, Symbol):
        if expr not in symbol_dimensions:
            raise DimensionError(f"Missing dimension for symbol: {expr}")
        return symbol_dimensions[expr]

    # Number (dimensionless)
    if expr.is_Number:
        return Dimension()

    # Addition / subtraction: all terms must match
    if isinstance(expr, Add):
        dims = [dimension_of(arg, symbol_dimensions) for arg in expr.args]
        base = dims[0]
        for d in dims[1:]:
            if d != base:
                raise DimensionError(
                    f"Dimension mismatch in addition: {expr}"
                )
        return base

    # Multiplication
    if isinstance(expr, Mul):
        result = Dimension()
        for arg in expr.args:
            result *= dimension_of(arg, symbol_dimensions)
        return result

    # Power
    if isinstance(expr, Pow):
        base, exponent = expr.args
        if not exponent.is_Number:
            raise DimensionError(
                f"Non-numeric exponent in expression: {expr}"
            )
        return dimension_of(base, symbol_dimensions) ** float(exponent)

    raise DimensionError(f"Unsupported expression type: {expr}")


def check_equation(eq: Eq, symbol_dimensions: dict):
    """
    Enforce dimensional consistency for a SymPy equation.
    HARD GATE: raises DimensionError on mismatch.
    """

    lhs_dim = dimension_of(eq.lhs, symbol_dimensions)
    rhs_dim = dimension_of(eq.rhs, symbol_dimensions)

    if lhs_dim != rhs_dim:
        raise DimensionError(
            f"Dimension mismatch in equation {eq}: "
            f"LHS [{lhs_dim}] != RHS [{rhs_dim}]"
        )
