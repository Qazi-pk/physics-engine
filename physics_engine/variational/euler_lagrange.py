"""Euler-Lagrange residual utilities for structured Lagrangian discovery.

For a structured model

    L(q, dq) = T(dq) - V(q)

the Euler-Lagrange residual becomes

    R = d/dt(∂T/∂dq) + ∂V/∂q

which is linear in basis coefficients when T and V are linear combinations
of predefined basis terms.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .lagrangian_library import ScalarBasisTerm


def build_el_regression_matrix(
    q: np.ndarray,
    dq: np.ndarray,
    ddq: np.ndarray,
    kinetic_terms: List[ScalarBasisTerm],
    potential_terms: List[ScalarBasisTerm],
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Build linear matrix A for residual A @ coeffs ≈ 0.

    Kinetic column i: d²phi_i/d(dq)² * ddq
    Potential column j: dpsi_j/dq

    Coeff vector layout is [a_kinetic..., b_potential...].
    """
    q_arr = np.asarray(q, dtype=float).reshape(-1)
    dq_arr = np.asarray(dq, dtype=float).reshape(-1)
    ddq_arr = np.asarray(ddq, dtype=float).reshape(-1)

    if not (len(q_arr) == len(dq_arr) == len(ddq_arr)):
        raise ValueError("q, dq, ddq must have same length")

    k_cols = [term.d2(dq_arr) * ddq_arr for term in kinetic_terms]
    p_cols = [term.d1(q_arr) for term in potential_terms]

    all_cols = k_cols + p_cols
    if not all_cols:
        raise ValueError("No basis terms provided for Euler-Lagrange matrix")

    A = np.column_stack(all_cols)
    return A, [t.name for t in kinetic_terms], [t.name for t in potential_terms]


def solve_nullspace_coefficients(A: np.ndarray) -> np.ndarray:
    """Find non-trivial coeff vector minimizing ||A c||² via SVD nullspace."""
    _, _, vt = np.linalg.svd(A, full_matrices=False)
    coeffs = vt[-1, :]
    scale = float(np.max(np.abs(coeffs)))
    if scale > 0:
        coeffs = coeffs / scale
    return coeffs


def evaluate_el_residual(A: np.ndarray, coeffs: np.ndarray) -> Dict[str, float]:
    """Compute residual statistics for Euler-Lagrange fit."""
    residual = A @ coeffs
    mse = float(np.mean(residual**2))
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(mse))
    max_abs = float(np.max(np.abs(residual))) if residual.size else 0.0
    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "max_abs": max_abs,
    }


__all__ = [
    "build_el_regression_matrix",
    "solve_nullspace_coefficients",
    "evaluate_el_residual",
]
