"""Structured feature libraries for variational discovery.

This module builds basis terms for structured Lagrangian parameterization:

    L(q, dq) = T(dq) - V(q)

Instead of searching arbitrary symbolic expressions over (q, dq), we split the
problem into kinetic and potential components, dramatically reducing search
complexity and improving stability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np


@dataclass(frozen=True)
class ScalarBasisTerm:
    """Single scalar basis term with derivative callbacks."""

    name: str
    value_fn: Callable[[np.ndarray], np.ndarray]
    first_derivative_fn: Callable[[np.ndarray], np.ndarray]
    second_derivative_fn: Callable[[np.ndarray], np.ndarray]

    def value(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.value_fn(x), dtype=float)

    def d1(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.first_derivative_fn(x), dtype=float)

    def d2(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.second_derivative_fn(x), dtype=float)


def kinetic_terms(max_even_power: int = 4) -> List[ScalarBasisTerm]:
    """Build kinetic basis terms that depend only on velocity dq.

    Args:
        max_even_power: Highest even power for polynomial kinetic terms.

    Returns:
        List of basis terms (e.g., dq2, dq4).
    """
    terms: List[ScalarBasisTerm] = []
    for power in range(2, max_even_power + 1, 2):
        terms.append(
            ScalarBasisTerm(
                name=f"dq{power}",
                value_fn=lambda dq, p=power: dq ** p,
                first_derivative_fn=lambda dq, p=power: p * (dq ** (p - 1)),
                second_derivative_fn=lambda dq, p=power: p * (p - 1) * (dq ** (p - 2)),
            )
        )
    return terms


def potential_terms(include_trig: bool = True, max_even_power: int = 4) -> List[ScalarBasisTerm]:
    """Build potential basis terms that depend only on position q."""
    terms: List[ScalarBasisTerm] = []
    for power in range(2, max_even_power + 1, 2):
        terms.append(
            ScalarBasisTerm(
                name=f"q{power}",
                value_fn=lambda q, p=power: q ** p,
                first_derivative_fn=lambda q, p=power: p * (q ** (p - 1)),
                second_derivative_fn=lambda q, p=power: p * (p - 1) * (q ** (p - 2)),
            )
        )

    if include_trig:
        terms.extend(
            [
                ScalarBasisTerm(
                    name="sin_q",
                    value_fn=np.sin,
                    first_derivative_fn=np.cos,
                    second_derivative_fn=lambda q: -np.sin(q),
                ),
                ScalarBasisTerm(
                    name="cos_q",
                    value_fn=np.cos,
                    first_derivative_fn=lambda q: -np.sin(q),
                    second_derivative_fn=lambda q: -np.cos(q),
                ),
            ]
        )

    return terms


def build_structured_library(
    q: np.ndarray,
    dq: np.ndarray,
    *,
    kinetic_max_even_power: int = 4,
    potential_max_even_power: int = 4,
    include_trig: bool = True,
) -> Dict[str, object]:
    """Build a complete structured T(dq)-V(q) feature library.

    Returns dictionary with term lists and evaluated design matrices for T and V.
    """
    q_arr = np.asarray(q, dtype=float).reshape(-1)
    dq_arr = np.asarray(dq, dtype=float).reshape(-1)
    if q_arr.shape[0] != dq_arr.shape[0]:
        raise ValueError("q and dq must have same length")

    t_terms = kinetic_terms(max_even_power=kinetic_max_even_power)
    v_terms = potential_terms(include_trig=include_trig, max_even_power=potential_max_even_power)

    t_matrix = np.column_stack([t.value(dq_arr) for t in t_terms]) if t_terms else np.empty((len(q_arr), 0))
    v_matrix = np.column_stack([v.value(q_arr) for v in v_terms]) if v_terms else np.empty((len(q_arr), 0))

    return {
        "q": q_arr,
        "dq": dq_arr,
        "kinetic_terms": t_terms,
        "potential_terms": v_terms,
        "kinetic_features": t_matrix,
        "potential_features": v_matrix,
        "kinetic_names": [t.name for t in t_terms],
        "potential_names": [v.name for v in v_terms],
    }


def lagrangian_features(
    q: np.ndarray,
    dq: np.ndarray,
    *,
    kinetic_max_even_power: int = 4,
    potential_max_even_power: int = 4,
    include_trig: bool = True,
) -> Dict[str, np.ndarray]:
    """Return flattened structured features for L(q, dq) = T(dq) - V(q).

    This helper provides a compact dictionary interface often used in pipelines:

        {
            "dq2": dq**2,
            "dq4": dq**4,
            "q2": q**2,
            "cos_q": cos(q),
            "sin_q": sin(q),
        }

    The exact keys depend on the selected basis settings.
    """
    library = build_structured_library(
        q=q,
        dq=dq,
        kinetic_max_even_power=kinetic_max_even_power,
        potential_max_even_power=potential_max_even_power,
        include_trig=include_trig,
    )

    features: Dict[str, np.ndarray] = {}
    for name, values in zip(library["kinetic_names"], library["kinetic_features"].T, strict=False):
        features[str(name)] = np.asarray(values, dtype=float)
    for name, values in zip(library["potential_names"], library["potential_features"].T, strict=False):
        features[str(name)] = np.asarray(values, dtype=float)

    return features


__all__ = [
    "ScalarBasisTerm",
    "kinetic_terms",
    "potential_terms",
    "build_structured_library",
    "lagrangian_features",
]
