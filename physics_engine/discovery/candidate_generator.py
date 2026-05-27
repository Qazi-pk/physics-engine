import itertools


def _is_numeric(value):
    return isinstance(value, (int, float))


def _column_values(rows, column_name):
    return [row.get(column_name) for row in rows]


def _is_numeric_column(rows, column_name):
    values = [value for value in _column_values(rows, column_name) if value is not None]
    if len(values) < 2:
        return False
    return all(_is_numeric(value) for value in values)


def _generate_tabular_candidates(headers, rows, max_candidates=20):
    numeric_columns = [header for header in headers if _is_numeric_column(rows, header)]

    candidates = []
    for target in numeric_columns:
        for feature in numeric_columns:
            if target == feature:
                continue
            candidates.append({"target": target, "feature": feature})
            if len(candidates) >= max_candidates:
                return candidates

    return candidates


def _generate_symbolic_candidates(variables, max_power=3, include_half_powers=True):
    candidates = []

    exponents = [p for p in range(-max_power, max_power + 1) if p != 0]
    if include_half_powers:
        half_exponents = [p / 2 for p in range(-2 * max_power, 2 * max_power + 1) if p % 2 != 0]
        exponents.extend(half_exponents)

    for var in variables:
        for p in sorted(set(exponents)):
            candidates.append(var**p)

    for v1, v2 in itertools.combinations(variables, 2):
        candidates.append(v1 * v2)
        candidates.append(v1 / v2)

    return candidates


def generate_candidates(headers_or_variables, rows=None, max_candidates=20, max_power=3):
    """
    Backward-compatible candidate generator.

    Modes:
    1) Tabular mode (existing behavior):
       generate_candidates(headers, rows, max_candidates=20)
    2) Symbolic mode (new behavior):
       generate_candidates(variables, max_power=3)
    """

    if rows is None:
        return _generate_symbolic_candidates(headers_or_variables, max_power=max_power)

    return _generate_tabular_candidates(headers_or_variables, rows, max_candidates=max_candidates)