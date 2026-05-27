import numpy as np
from sklearn.linear_model import LinearRegression
from sympy import Eq, Float, Symbol

from .candidate_generator import generate_candidates


def fit_model(feature, target):
    feature = np.asarray(feature, dtype=float).reshape(-1, 1)
    target = np.asarray(target, dtype=float)

    model = LinearRegression()
    model.fit(feature, target)

    predictions = model.predict(feature)
    mse = float(np.mean((target - predictions) ** 2))

    return float(model.coef_[0]), float(model.intercept_), mse


def _linear_fit(xs, ys):
    n = len(xs)
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xx = sum(x * x for x in xs)
    sum_xy = sum(x * y for x, y in zip(xs, ys))

    denominator = n * sum_xx - sum_x * sum_x
    if denominator == 0:
        return None

    slope = (n * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / n

    y_mean = sum_y / n
    ss_total = sum((y - y_mean) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    score = 1.0 if ss_total == 0 else 1 - (ss_res / ss_total)

    return slope, intercept, score


def fit_symbolic_equation(rows, target, feature):
    pairs = []
    for row in rows:
        x_value = row.get(feature)
        y_value = row.get(target)
        if isinstance(x_value, (int, float)) and isinstance(y_value, (int, float)):
            pairs.append((float(x_value), float(y_value)))

    if len(pairs) < 2:
        return None

    xs = [x for x, _ in pairs]
    ys = [y for _, y in pairs]
    fitted = _linear_fit(xs, ys)
    if fitted is None:
        return None

    slope, intercept, score = fitted
    target_symbol = Symbol(target)
    feature_symbol = Symbol(feature)

    if abs(intercept) < 1e-10:
        equation = Eq(target_symbol, Float(slope) * feature_symbol)
    else:
        equation = Eq(target_symbol, Float(slope) * feature_symbol + Float(intercept))

    return {
        "target": target,
        "feature": feature,
        "equation": equation,
        "score": float(score),
    }


def discover_equations(headers, rows, max_results=5):
    candidates = generate_candidates(headers, rows)
    discovered = []

    for candidate in candidates:
        fitted = fit_symbolic_equation(rows, candidate["target"], candidate["feature"])
        if fitted is not None:
            discovered.append(fitted)

    discovered.sort(key=lambda item: item["score"], reverse=True)
    return discovered[:max_results]