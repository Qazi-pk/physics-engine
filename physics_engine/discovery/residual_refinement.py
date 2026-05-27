import numpy as np
import sympy as sp


def _predict_with_model(model, data, variable_names):
    symbols = [sp.Symbol(v) for v in variable_names]
    expr_fn = sp.lambdify(symbols, model["expression"], "numpy")
    features = expr_fn(*[data[v].values for v in variable_names])
    features = np.asarray(features, dtype=float).reshape(-1)
    predictions = model["coefficient"] * features + model["intercept"]
    return features, predictions


def refine_with_residuals(
    data,
    target_name,
    variable_names,
    base_model,
    candidates,
    min_correlation=0.6,
    min_improvement_ratio=0.01,
):
    """
    Residual-driven one-step refinement:
    1) Compute residuals from base model
    2) Find candidate term with strongest residual correlation
    3) Add one corrective term if it improves MSE enough
    """

    if base_model is None:
        return base_model

    target = np.asarray(data[target_name].values, dtype=float)
    _, base_pred = _predict_with_model(base_model, data, variable_names)

    mask_base = np.isfinite(base_pred) & np.isfinite(target)
    if np.sum(mask_base) < 3:
        return base_model

    residuals = target[mask_base] - base_pred[mask_base]
    base_mse = float(np.mean((target[mask_base] - base_pred[mask_base]) ** 2))

    symbols = [sp.Symbol(v) for v in variable_names]
    best = None

    for term in candidates:
        if sp.simplify(term - base_model["expression"]) == 0:
            continue

        try:
            term_fn = sp.lambdify(symbols, term, "numpy")
            term_values = term_fn(*[data[v].values for v in variable_names])
            term_values = np.asarray(term_values, dtype=float).reshape(-1)
        except Exception:
            continue

        mask = mask_base & np.isfinite(term_values)
        if np.sum(mask) < 3:
            continue

        residual_subset = target[mask] - base_pred[mask]
        term_subset = term_values[mask]

        residual_std = float(np.std(residual_subset))
        term_std = float(np.std(term_subset))
        if residual_std <= 1e-12 or term_std <= 1e-12:
            continue

        corr = float(np.corrcoef(residual_subset, term_subset)[0, 1])
        if not np.isfinite(corr) or abs(corr) < min_correlation:
            continue

        X = np.column_stack([base_pred[mask], term_subset, np.ones(np.sum(mask))])
        y = target[mask]

        try:
            coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            continue

        alpha, beta, gamma = coeffs
        y_hat = alpha * base_pred[mask] + beta * term_subset + gamma
        mse = float(np.mean((y - y_hat) ** 2))

        if best is None or mse < best["mse"]:
            best = {
                "term": term,
                "correlation": corr,
                "mse": mse,
                "alpha": float(alpha),
                "beta": float(beta),
                "gamma": float(gamma),
            }

    if best is None:
        return base_model

    improvement_ratio = (base_mse - best["mse"]) / max(base_mse, 1e-12)
    if improvement_ratio < min_improvement_ratio:
        return base_model

    refined_expression = sp.simplify(best["alpha"] * base_model["expression"] + best["beta"] * best["term"])

    refined = dict(base_model)
    refined.update(
        {
            "expression": refined_expression,
            "coefficient": 1.0,
            "intercept": best["gamma"],
            "mse": best["mse"],
            "refined": True,
            "refinement": {
                "added_term": best["term"],
                "correlation": best["correlation"],
                "improvement_ratio": float(improvement_ratio),
            },
        }
    )

    return refined
