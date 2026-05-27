from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr


def accuracy_score(y_true, y_pred) -> float:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    if y_true_arr.size == 0 or y_pred_arr.size == 0 or y_true_arr.shape != y_pred_arr.shape:
        return 0.0

    finite_mask = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
    if not np.any(finite_mask):
        return 0.0

    y_true_arr = y_true_arr[finite_mask]
    y_pred_arr = y_pred_arr[finite_mask]

    mse = float(np.mean((y_true_arr - y_pred_arr) ** 2))
    variance = float(np.var(y_true_arr))

    if variance <= 0.0:
        return 1.0 if mse == 0.0 else 0.0

    return float(np.clip(1.0 - (mse / variance), 0.0, 1.0))


def simplicity_score(complexity: float) -> float:
    value = float(max(0.0, complexity))
    return float(1.0 / (1.0 + value))


def residual_score(residual, feature_matrix) -> float:
    residual_arr = np.asarray(residual, dtype=float)
    feature_arr = np.asarray(feature_matrix, dtype=float)

    if residual_arr.ndim != 1 or residual_arr.size < 2:
        return 1.0

    if feature_arr.ndim == 1:
        feature_arr = feature_arr.reshape(-1, 1)

    if feature_arr.ndim != 2 or feature_arr.shape[0] != residual_arr.shape[0]:
        return 1.0

    finite_residual = np.isfinite(residual_arr)
    max_corr = 0.0

    for idx in range(feature_arr.shape[1]):
        feature_col = feature_arr[:, idx]
        finite_mask = finite_residual & np.isfinite(feature_col)
        if np.count_nonzero(finite_mask) < 2:
            continue

        x = residual_arr[finite_mask]
        y = feature_col[finite_mask]

        if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
            continue

        corr, _ = pearsonr(x, y)
        if np.isfinite(corr):
            max_corr = max(max_corr, abs(float(corr)))

    return float(np.clip(1.0 - max_corr, 0.0, 1.0))


def robustness_score(successes: int, runs: int) -> float:
    if runs <= 0:
        return 0.0
    return float(np.clip(float(successes) / float(runs), 0.0, 1.0))


def discovery_confidence(
    A: float,
    S: float,
    R: float,
    N: float,
    w1: float = 0.4,
    w2: float = 0.2,
    w3: float = 0.2,
    w4: float = 0.2,
) -> float:
    components = np.asarray([A, S, R, N], dtype=float)
    weights = np.asarray([w1, w2, w3, w4], dtype=float)

    components = np.nan_to_num(components, nan=0.0, posinf=0.0, neginf=0.0)
    components = np.clip(components, 0.0, 1.0)

    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        raise ValueError("DCS weights must sum to a positive value.")

    normalized_weights = weights / weight_sum
    return float(np.clip(np.dot(normalized_weights, components), 0.0, 1.0))
