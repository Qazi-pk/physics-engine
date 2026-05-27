from __future__ import annotations

import numpy as np


def _safe_inverse(values, power: int = 1):
    values = np.asarray(values, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / np.power(values, power)
    return np.where(np.isfinite(inv), inv, np.nan)


def _safe_sqrt(values):
    values = np.asarray(values, dtype=float)
    with np.errstate(invalid="ignore"):
        root = np.sqrt(values)
    return np.where(np.isfinite(root), root, np.nan)


def _safe_divide(numerator, denominator):
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = numerator / denominator
    return np.where(np.isfinite(ratio), ratio, np.nan)


def _safe_atan2(y_values, x_values):
    y_values = np.asarray(y_values, dtype=float)
    x_values = np.asarray(x_values, dtype=float)
    angles = np.arctan2(y_values, x_values)
    return np.where(np.isfinite(angles), angles, np.nan)


def generate_latent_variables(dataframe):
    """
    Generate latent/hidden physics variables from observed state columns.

    Current latent families:
    - radial invariants from Cartesian position: x, y -> r2, r, inv_r, inv_r2, inv_r3, x_over_r3, y_over_r3
    - speed invariants from Cartesian velocity: vx, vy -> v2, v, inv_v, inv_v2
    - mixed hidden variables from position+velocity: radial_velocity_numerator, angular_momentum_z,
      specific_kinetic_energy, specific_energy_proxy
    """

    features = {}

    if "x" in dataframe.columns and "y" in dataframe.columns:
        x = np.asarray(dataframe["x"], dtype=float)
        y = np.asarray(dataframe["y"], dtype=float)
        r2 = x**2 + y**2
        r = _safe_sqrt(r2)

        features["r2"] = r2
        features["r"] = r
        features["inv_r"] = _safe_inverse(r, power=1)
        features["inv_r2"] = _safe_inverse(r, power=2)
        features["inv_r3"] = _safe_inverse(r, power=3)
        features["x_sq"] = x**2
        features["y_sq"] = y**2
        features["xy"] = x * y
        features["theta"] = _safe_atan2(y, x)
        features["x_over_r"] = _safe_divide(x, r)
        features["y_over_r"] = _safe_divide(y, r)

        features["x_over_r3"] = _safe_divide(x, np.power(r, 3))
        features["y_over_r3"] = _safe_divide(y, np.power(r, 3))

    if "vx" in dataframe.columns and "vy" in dataframe.columns:
        vx = np.asarray(dataframe["vx"], dtype=float)
        vy = np.asarray(dataframe["vy"], dtype=float)
        v2 = vx**2 + vy**2
        v = _safe_sqrt(v2)

        features["v2"] = v2
        features["v"] = v
        features["inv_v"] = _safe_inverse(v, power=1)
        features["inv_v2"] = _safe_inverse(v, power=2)
        features["specific_kinetic_energy"] = 0.5 * v2

    if all(col in dataframe.columns for col in ("x", "y", "vx", "vy")):
        x = np.asarray(dataframe["x"], dtype=float)
        y = np.asarray(dataframe["y"], dtype=float)
        vx = np.asarray(dataframe["vx"], dtype=float)
        vy = np.asarray(dataframe["vy"], dtype=float)

        features["radial_velocity_numerator"] = x * vx + y * vy
        features["angular_momentum_z"] = x * vy - y * vx

        if "v2" in features:
            inv_r_source = features.get("inv_r")
            if inv_r_source is not None:
                features["specific_energy_proxy"] = 0.5 * np.asarray(features["v2"], dtype=float) - np.asarray(inv_r_source, dtype=float)

    return features


def generate_hidden_variables(dataframe):
    return generate_latent_variables(dataframe)
