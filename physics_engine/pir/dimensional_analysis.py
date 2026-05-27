import sympy as sp


def _add(d1, d2):
    return tuple(a + b for a, b in zip(d1, d2))


def dimensions_of_expression(expr, symbol_dimensions):
    if isinstance(expr, str):
        expr = sp.sympify(expr)

    if expr.is_Symbol:
        return symbol_dimensions.get(str(expr), (0, 0, 0))

    if expr.is_Number:
        return (0, 0, 0)

    if expr.is_Mul:
        out = (0, 0, 0)
        for arg in expr.args:
            out = _add(out, dimensions_of_expression(arg, symbol_dimensions))
        return out

    if expr.is_Pow:
        base, exponent = expr.args
        base_dim = dimensions_of_expression(base, symbol_dimensions)
        # Use the raw exponent (may be sp.Rational) instead of int() to
        # preserve fractional powers such as r^(3/2) needed for Kepler.
        return tuple(exponent * d for d in base_dim)

    return (0, 0, 0)


def dimensional_filter(candidates, variable_dimensions, target_dimension):
    """
    Remove expressions that violate dimensional consistency.
    """

    valid = []
    for expr in candidates:
        try:
            dim = dimensions_of_expression(expr, variable_dimensions)
            if dim == target_dimension:
                valid.append(expr)
        except Exception:
            continue

    return valid
