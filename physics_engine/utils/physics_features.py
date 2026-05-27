import numpy as np


def _safe_inverse(values, power=1):
    values = np.asarray(values, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / np.power(values, power)
    return np.where(np.isfinite(inv), inv, np.nan)


def _safe_power(values, exponent):
    values = np.asarray(values, dtype=float)
    with np.errstate(invalid="ignore"):
        powered = np.power(values, exponent)
    return np.where(np.isfinite(powered), powered, np.nan)


def _build_radial_feature_family(r_values):
    r_values = np.asarray(r_values, dtype=float)
    return {
        "r2": _safe_power(r_values, 2),
        "r3": _safe_power(r_values, 3),
        "inv_r": _safe_inverse(r_values, power=1),
        "inv_r2": _safe_inverse(r_values, power=2),
        "inv_r3": _safe_inverse(r_values, power=3),
        "sqrt_r": _safe_power(r_values, 0.5),
        "r_3_2": _safe_power(r_values, 1.5),
    }


def detect_invariants(dataframe):
    """
    Detect simple algebraic invariants from available state variables.

    Current invariant set:
      - Cartesian radial invariant from x, y: x^2 + y^2 -> r^2
    """

    invariants = {}

    if "x" in dataframe.columns and "y" in dataframe.columns:
        x = np.asarray(dataframe["x"], dtype=float)
        y = np.asarray(dataframe["y"], dtype=float)
        invariants["r2"] = _safe_power(x, 2) + _safe_power(y, 2)

    return invariants


def generate_trigonometric_features(dataframe):
    features = {}

    if "theta1" in dataframe.columns:
        t1 = np.asarray(dataframe["theta1"], dtype=float)
        features["sin_theta1"] = np.sin(t1)
        features["cos_theta1"] = np.cos(t1)

    if "theta2" in dataframe.columns:
        t2 = np.asarray(dataframe["theta2"], dtype=float)
        features["sin_theta2"] = np.sin(t2)
        features["cos_theta2"] = np.cos(t2)

    if "theta1" in dataframe.columns and "theta2" in dataframe.columns:
        t12 = np.asarray(dataframe["theta1"], dtype=float) + np.asarray(dataframe["theta2"], dtype=float)
        features["sin_theta12"] = np.sin(t12)
        features["cos_theta12"] = np.cos(t12)

    return features


def generate_physics_features(dataframe):
    """
    Generate physics-guided candidate features from available columns.

    Works with common radial/orbital patterns and avoids crashes on invalid
    values by returning NaN for non-finite points.
    """

    features = {}
    features.update(generate_trigonometric_features(dataframe))

    invariants = detect_invariants(dataframe)

    if "r2" in invariants and "r" not in dataframe.columns:
        derived_r = _safe_power(invariants["r2"], 0.5)
        features["r"] = derived_r
        features.update(_build_radial_feature_family(derived_r))

    if "r" in dataframe.columns:
        r = np.asarray(dataframe["r"], dtype=float)
        features.update(_build_radial_feature_family(r))

    if "x" in dataframe.columns and "y" in dataframe.columns:
        x = np.asarray(dataframe["x"], dtype=float)
        y = np.asarray(dataframe["y"], dtype=float)
        if "r" in dataframe.columns:
            r = np.asarray(dataframe["r"], dtype=float)
        elif "r" in features:
            r = np.asarray(features["r"], dtype=float)
        else:
            r = _safe_power(_safe_power(x, 2) + _safe_power(y, 2), 0.5)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_over_r3 = x / np.power(r, 3)
            y_over_r3 = y / np.power(r, 3)
        features["x_over_r3"] = np.where(np.isfinite(x_over_r3), x_over_r3, np.nan)
        features["y_over_r3"] = np.where(np.isfinite(y_over_r3), y_over_r3, np.nan)

    return features
