"""
Robotics-specific feature libraries for PIR structure discovery.

The standard FeatureLibrary is designed for single-joint dynamics
(theta, omega, alpha).  Jacobian and multi-link kinematics require
*composite* trigonometric terms that the basic library omits:

    sin(θ₁ + θ₂),  cos(θ₁ + θ₂)

Without these terms a symbolic regression algorithm **cannot** reconstruct
the planar 2-link Jacobian equations, which is the root cause of 0 % success
in short-sample experiments.

Usage::

    from physics_engine.discovery.feature_library_robotics import (
        FeatureLibraryRobotics,
        build_robot_jacobian_library,
    )

    # From a pandas DataFrame / dict-like
    X, names = build_robot_jacobian_library(df)

    # Or using the object API (mirrors FeatureLibrary)
    lib = FeatureLibraryRobotics(profile="jacobian")
    X, names = lib.build_from_dict({
        "theta1": ..., "theta2": ..., "l1": ..., "l2": ...
    })
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Functional API (stateless helpers)
# ---------------------------------------------------------------------------

def build_robot_jacobian_library(
    dataset: Union[Dict[str, np.ndarray], object],
    *,
    include_link_lengths: bool = True,
    include_raw_angles: bool = True,
    include_products: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build the feature matrix for 2-link planar robot Jacobian discovery.

    The Jacobian of a 2-link planar arm has the analytical form::

        J11 = -l1*sin(θ1) - l2*sin(θ1+θ2)
        J12 = -l2*sin(θ1+θ2)
        J21 =  l1*cos(θ1) + l2*cos(θ1+θ2)
        J22 =  l2*cos(θ1+θ2)

    These equations require the **composite** terms sin(θ1+θ2) and
    cos(θ1+θ2).  Without them no sparse regression method can reconstruct
    the correct Jacobian coefficients.

    Args:
        dataset:               A dict-like object with columns ``theta1``,
                               ``theta2``, and optionally ``l1``, ``l2``.
        include_link_lengths:  Add l1 and l2 as features (required for
                               coefficient recovery; default True).
        include_raw_angles:    Add raw θ1, θ2 values (default True).
        include_products:      Add interaction products l1*sin(θ1) etc.
                               (improves discovery with small datasets;
                               default False to keep the library sparse).

    Returns:
        (X, feature_names) where X has shape (n_samples, n_features).

    Raises:
        ValueError: If ``theta1`` or ``theta2`` are missing from *dataset*.
    """
    theta1 = _get(dataset, "theta1")
    theta2 = _get(dataset, "theta2")

    if theta1 is None or theta2 is None:
        raise ValueError(
            "dataset must contain 'theta1' and 'theta2' columns for "
            "Jacobian feature library construction."
        )

    theta1 = np.asarray(theta1, dtype=float)
    theta2 = np.asarray(theta2, dtype=float)
    theta12 = theta1 + theta2   # composite angle

    l1 = _get(dataset, "l1")
    l2 = _get(dataset, "l2")

    features: Dict[str, np.ndarray] = {}

    # ---- raw angles -------------------------------------------------------
    if include_raw_angles:
        features["theta1"] = theta1
        features["theta2"] = theta2

    # ---- single-angle trig ------------------------------------------------
    features["sin_theta1"] = np.sin(theta1)
    features["cos_theta1"] = np.cos(theta1)
    features["sin_theta2"] = np.sin(theta2)
    features["cos_theta2"] = np.cos(theta2)

    # ---- composite trig (KEY for Jacobian discovery) ---------------------
    features["sin_theta1_theta2"] = np.sin(theta12)
    features["cos_theta1_theta2"] = np.cos(theta12)

    # ---- link lengths -------------------------------------------------------
    if include_link_lengths and l1 is not None:
        features["l1"] = np.asarray(l1, dtype=float)
    if include_link_lengths and l2 is not None:
        features["l2"] = np.asarray(l2, dtype=float)

    # ---- optional interaction products ------------------------------------
    if include_products:
        if l1 is not None:
            l1a = np.asarray(l1, dtype=float)
            features["l1_sin_theta1"]       = l1a * np.sin(theta1)
            features["l1_cos_theta1"]       = l1a * np.cos(theta1)
            features["l1_sin_theta1_theta2"] = l1a * np.sin(theta12)
            features["l1_cos_theta1_theta2"] = l1a * np.cos(theta12)
        if l2 is not None:
            l2a = np.asarray(l2, dtype=float)
            features["l2_sin_theta1_theta2"] = l2a * np.sin(theta12)
            features["l2_cos_theta1_theta2"] = l2a * np.cos(theta12)

    names = list(features.keys())
    X = np.column_stack([features[n] for n in names])
    return X, names


def build_robot_dynamics_library(
    dataset: Union[Dict[str, np.ndarray], object],
) -> Tuple[np.ndarray, List[str]]:
    """
    Feature library for single-joint dynamics discovery::

        τ = I·α + b·ω + k·θ   (and higher-order terms)

    Wraps the standard FeatureLibrary 'cross_terms' profile and adds
    sin/cos of theta for completeness.
    """
    theta = _get(dataset, "theta")
    omega = _get(dataset, "omega")
    alpha = _get(dataset, "alpha")

    if theta is None or omega is None or alpha is None:
        raise ValueError(
            "dataset must contain 'theta', 'omega', 'alpha' for "
            "robot dynamics discovery."
        )

    theta = np.asarray(theta, dtype=float)
    omega = np.asarray(omega, dtype=float)
    alpha = np.asarray(alpha, dtype=float)

    features: Dict[str, np.ndarray] = {
        "alpha":        alpha,
        "omega":        omega,
        "theta":        theta,
        "omega2":       omega ** 2,
        "theta2":       theta ** 2,
        "sin_theta":    np.sin(theta),
        "cos_theta":    np.cos(theta),
        "alpha_omega":  alpha * omega,
        "alpha_theta":  alpha * theta,
        "omega_theta":  omega * theta,
    }

    names = list(features.keys())
    X = np.column_stack([features[n] for n in names])
    return X, names


# ---------------------------------------------------------------------------
# Object API (mirrors FeatureLibrary)
# ---------------------------------------------------------------------------

class FeatureLibraryRobotics:
    """
    Robotics-specific feature library with profile selection.

    Profiles
    --------
    ``jacobian``  (default)
        Composite trig terms for 2-link Jacobian discovery.
        Includes sin(θ₁+θ₂) and cos(θ₁+θ₂).

    ``jacobian_products``
        Same plus l·trig interaction terms.  Better for small datasets
        at the cost of a larger feature matrix.

    ``dynamics``
        Standard single-joint dynamics features (θ, ω, α, cross terms).

    Examples
    --------
    >>> lib = FeatureLibraryRobotics(profile="jacobian")
    >>> X, names = lib.build_from_dict({"theta1": ..., "theta2": ...})
    """

    PROFILES = ("jacobian", "jacobian_products", "dynamics")

    def __init__(self, profile: str = "jacobian") -> None:
        if profile not in self.PROFILES:
            raise ValueError(
                f"Invalid profile '{profile}'. Choose from {self.PROFILES}."
            )
        self.profile = profile

    def build_from_dict(
        self,
        data: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Build feature matrix from a plain column dictionary.

        Args:
            data: Mapping column_name → 1-D array.

        Returns:
            (X, feature_names)
        """
        if self.profile == "jacobian":
            return build_robot_jacobian_library(
                data,
                include_link_lengths=True,
                include_raw_angles=True,
                include_products=False,
            )
        elif self.profile == "jacobian_products":
            return build_robot_jacobian_library(
                data,
                include_link_lengths=True,
                include_raw_angles=True,
                include_products=True,
            )
        else:  # dynamics
            return build_robot_dynamics_library(data)

    def build_from_dataframe(self, df: object) -> Tuple[np.ndarray, List[str]]:
        """
        Build feature matrix from a pandas DataFrame.

        Args:
            df: pandas DataFrame with appropriate column names.

        Returns:
            (X, feature_names)
        """
        return self.build_from_dict({col: df[col].values for col in df.columns})  # type: ignore[attr-defined]

    def feature_names(self, sample_data: Optional[Dict[str, np.ndarray]] = None) -> List[str]:
        """Return expected feature names for this profile (requires sample data to infer shapes)."""
        if sample_data is not None:
            _, names = self.build_from_dict(sample_data)
            return names
        if self.profile == "jacobian":
            return [
                "theta1", "theta2",
                "sin_theta1", "cos_theta1",
                "sin_theta2", "cos_theta2",
                "sin_theta1_theta2", "cos_theta1_theta2",
                "l1", "l2",
            ]
        if self.profile == "dynamics":
            return [
                "alpha", "omega", "theta",
                "omega2", "theta2",
                "sin_theta", "cos_theta",
                "alpha_omega", "alpha_theta", "omega_theta",
            ]
        return self.feature_names()  # products variant—call with sample_data


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _get(obj: Any, key: str) -> Optional[np.ndarray]:
    """Extract a column from a dict or pandas DataFrame (or any attribute)."""
    if isinstance(obj, dict):
        return obj.get(key)
    try:
        return obj[key]  # type: ignore[index]
    except (KeyError, TypeError):
        pass
    try:
        return getattr(obj, key)
    except AttributeError:
        return None


from typing import Any  # noqa: E402  (moved after usage to avoid circular import noise)


__all__ = [
    "FeatureLibraryRobotics",
    "build_robot_jacobian_library",
    "build_robot_dynamics_library",
]
