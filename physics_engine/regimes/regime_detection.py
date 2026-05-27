import numpy as np


def find_best_regime_split(feature_values, residuals, min_group_size=20):
    values = np.asarray(feature_values, dtype=float)
    residuals = np.asarray(residuals, dtype=float)

    if len(values) != len(residuals) or len(values) < 2 * min_group_size:
        return {"threshold": None, "gap": 0.0}

    best_gap = 0.0
    best_threshold = None

    for threshold in np.percentile(values, np.linspace(10, 90, 9)):
        left = residuals[values < threshold]
        right = residuals[values >= threshold]

        if len(left) < min_group_size or len(right) < min_group_size:
            continue

        gap = abs(float(np.mean(left) - np.mean(right)))
        if gap > best_gap:
            best_gap = gap
            best_threshold = float(threshold)

    return {"threshold": best_threshold, "gap": best_gap}
