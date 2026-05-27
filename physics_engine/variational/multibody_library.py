"""Feature library for 2-DOF multibody Lagrangian models.

This module provides basis terms over generalized coordinates (q1, q2) used to
parameterize:

    M11(q), M12(q), M22(q), V(q)

with closed-form partial derivatives required by Euler-Lagrange equations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np


@dataclass(frozen=True)
class MultiBodyBasisTerm:
    """Single 2-variable basis term with first partial derivatives."""

    name: str
    value_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]
    dq1_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]
    dq2_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]

    def value(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        return np.asarray(self.value_fn(q1, q2), dtype=float)

    def d_q1(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        return np.asarray(self.dq1_fn(q1, q2), dtype=float)

    def d_q2(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        return np.asarray(self.dq2_fn(q1, q2), dtype=float)


def multibody_basis_terms() -> List[MultiBodyBasisTerm]:
    """Default trigonometric + constant basis suitable for many robot systems."""
    return [
        MultiBodyBasisTerm(
            name="1",
            value_fn=lambda q1, q2: np.ones_like(q1),
            dq1_fn=lambda q1, q2: np.zeros_like(q1),
            dq2_fn=lambda q1, q2: np.zeros_like(q1),
        ),
        MultiBodyBasisTerm(
            name="cos_q1",
            value_fn=lambda q1, q2: np.cos(q1),
            dq1_fn=lambda q1, q2: -np.sin(q1),
            dq2_fn=lambda q1, q2: np.zeros_like(q1),
        ),
        MultiBodyBasisTerm(
            name="cos_q2",
            value_fn=lambda q1, q2: np.cos(q2),
            dq1_fn=lambda q1, q2: np.zeros_like(q1),
            dq2_fn=lambda q1, q2: -np.sin(q2),
        ),
        MultiBodyBasisTerm(
            name="cos_q1q2",
            value_fn=lambda q1, q2: np.cos(q1 + q2),
            dq1_fn=lambda q1, q2: -np.sin(q1 + q2),
            dq2_fn=lambda q1, q2: -np.sin(q1 + q2),
        ),
        MultiBodyBasisTerm(
            name="sin_q1",
            value_fn=lambda q1, q2: np.sin(q1),
            dq1_fn=lambda q1, q2: np.cos(q1),
            dq2_fn=lambda q1, q2: np.zeros_like(q1),
        ),
        MultiBodyBasisTerm(
            name="sin_q2",
            value_fn=lambda q1, q2: np.sin(q2),
            dq1_fn=lambda q1, q2: np.zeros_like(q1),
            dq2_fn=lambda q1, q2: np.cos(q2),
        ),
        MultiBodyBasisTerm(
            name="sin_q1q2",
            value_fn=lambda q1, q2: np.sin(q1 + q2),
            dq1_fn=lambda q1, q2: np.cos(q1 + q2),
            dq2_fn=lambda q1, q2: np.cos(q1 + q2),
        ),
    ]


def multibody_features(q1: np.ndarray, q2: np.ndarray) -> Dict[str, np.ndarray]:
    """Convenience map of basis name -> evaluated values."""
    q1_arr = np.asarray(q1, dtype=float).reshape(-1)
    q2_arr = np.asarray(q2, dtype=float).reshape(-1)
    if q1_arr.shape[0] != q2_arr.shape[0]:
        raise ValueError("q1 and q2 must have same length")

    return {term.name: term.value(q1_arr, q2_arr) for term in multibody_basis_terms()}


__all__ = [
    "MultiBodyBasisTerm",
    "multibody_basis_terms",
    "multibody_features",
]
