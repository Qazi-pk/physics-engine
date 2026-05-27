import numpy as np

from .residual_analysis import find_residual_correlations


def detect_missing_signal(residuals, feature_values, threshold=0.7):
    residuals = np.asarray(residuals, dtype=float)
    feature_values = np.asarray(feature_values, dtype=float)

    if len(residuals) < 2 or len(feature_values) != len(residuals):
        return {"correlation": 0.0, "missing_signal": False}

    if float(np.std(residuals)) == 0.0 or float(np.std(feature_values)) == 0.0:
        return {"correlation": 0.0, "missing_signal": False}

    corr = float(np.corrcoef(residuals, feature_values)[0, 1])
    return {"correlation": corr, "missing_signal": abs(corr) >= threshold}


def suggest_missing_term(residual, features, threshold=0.6):
    ranked = find_residual_correlations(residual, features)
    if not ranked:
        return None

    best_feature, corr = ranked[0]
    if corr > threshold:
        return best_feature

    return None
