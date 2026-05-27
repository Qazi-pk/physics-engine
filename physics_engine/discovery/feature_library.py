from __future__ import annotations

import numpy as np

from physics_engine.core.dataset import Dataset


class FeatureLibrary:
    def __init__(self, profile: str = "basic"):
        valid_profiles = {"basic", "polynomial", "cross_terms"}
        if profile not in valid_profiles:
            raise ValueError(f"Invalid profile '{profile}'. Choose from {valid_profiles}.")
        self.profile = profile

    def build(self, dataset: Dataset) -> tuple[np.ndarray, list[str]]:
        theta = dataset.get("theta")
        omega = dataset.get("omega")
        alpha = dataset.get("alpha")

        if theta is None or omega is None or alpha is None:
            raise ValueError("Dataset must contain theta, omega, and alpha.")

        theta_arr = np.asarray(theta, dtype=float)
        omega_arr = np.asarray(omega, dtype=float)
        alpha_arr = np.asarray(alpha, dtype=float)

        features: dict[str, np.ndarray] = {
            "alpha": alpha_arr,
            "omega": omega_arr,
            "theta": theta_arr,
        }

        if self.profile in ("basic", "polynomial"):
            features["omega2"] = omega_arr**2
            features["theta2"] = theta_arr**2
            features["sin_theta"] = np.sin(theta_arr)

        if self.profile == "polynomial":
            features["alpha2"] = alpha_arr**2
            features["omega3"] = omega_arr**3
            features["theta3"] = theta_arr**3

        if self.profile == "cross_terms":
            features["alpha_omega"] = alpha_arr * omega_arr
            features["alpha_theta"] = alpha_arr * theta_arr
            features["omega_theta"] = omega_arr * theta_arr
            features["sin_theta"] = np.sin(theta_arr)
            features["cos_theta"] = np.cos(theta_arr)

        names = list(features.keys())
        X = np.column_stack([features[name] for name in names])
        return X, names
