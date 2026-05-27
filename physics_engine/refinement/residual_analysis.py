import numpy as np


def residual_summary(y_true, y_pred):
    residuals = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return {
        "mae": float(np.mean(np.abs(residuals))),
        "mean": float(np.mean(residuals)),
        "std": float(np.std(residuals)),
        "max_abs": float(np.max(np.abs(residuals))),
    }


def _valid_correlation_inputs(residual, feature):
    residual = np.asarray(residual, dtype=float).reshape(-1)
    feature = np.asarray(feature, dtype=float).reshape(-1)

    if residual.size < 2 or feature.size != residual.size:
        return None

    mask = np.isfinite(residual) & np.isfinite(feature)
    if np.sum(mask) < 2:
        return None

    residual_masked = residual[mask]
    feature_masked = feature[mask]
    if float(np.std(residual_masked)) <= 1e-12 or float(np.std(feature_masked)) <= 1e-12:
        return None

    return residual_masked, feature_masked


def find_residual_correlations(residual, feature_dict):
    correlations = {}

    for name, feature in feature_dict.items():
        prepared = _valid_correlation_inputs(residual, feature)
        if prepared is None:
            continue

        residual_masked, feature_masked = prepared
        corr = float(np.corrcoef(residual_masked, feature_masked)[0, 1])
        if np.isfinite(corr):
            correlations[name] = abs(corr)

    return sorted(correlations.items(), key=lambda x: x[1], reverse=True)
