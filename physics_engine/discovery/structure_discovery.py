from __future__ import annotations

import numpy as np

from physics_engine.discovery.parameter_estimation import ParameterEstimator


class StructureDiscovery:
    def __init__(self, threshold: float = 0.05):
        self.estimator = ParameterEstimator()
        self.threshold = float(threshold)

    def discover(self, X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> list[tuple[float, str]]:
        if X.shape[1] != len(feature_names):
            raise ValueError("feature_names length must match number of feature columns.")

        params = self.estimator.estimate_linear(X, y)

        equation: list[tuple[float, str]] = []
        for coef, name in zip(params, feature_names):
            if abs(float(coef)) > self.threshold:
                equation.append((float(coef), name))

        return equation
