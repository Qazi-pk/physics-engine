"""Structured Lagrangian discovery using T(dq) - V(q) parameterization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .euler_lagrange import (
    build_el_regression_matrix,
    evaluate_el_residual,
    solve_nullspace_coefficients,
)
from .lagrangian_library import build_structured_library


@dataclass(frozen=True)
class StructuredLagrangianResult:
    """Result object for structured variational discovery."""

    lagrangian: str
    kinetic_energy: str
    potential_energy: str
    residual_mse: float
    residual_rmse: float
    kinetic_coefficients: Dict[str, float]
    potential_coefficients: Dict[str, float]


def _term_string(coeff_map: Dict[str, float]) -> str:
    items = [(name, coeff) for name, coeff in coeff_map.items() if abs(coeff) > 1e-8]
    if not items:
        return "0"
    parts: List[str] = []
    for name, coeff in items:
        parts.append(f"{coeff:.6g}*{name}")
    return " + ".join(parts)


def _normalize_gauge(kinetic: Dict[str, float], potential: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """Normalize coefficient scale; additive constants are already excluded by design."""
    all_vals = list(kinetic.values()) + list(potential.values())
    max_abs = max((abs(v) for v in all_vals), default=0.0)
    if max_abs <= 0:
        return {"kinetic": kinetic, "potential": potential}

    k = {n: v / max_abs for n, v in kinetic.items()}
    p = {n: v / max_abs for n, v in potential.items()}

    # Sign normalization: prefer positive leading kinetic coefficient when possible.
    lead = next((v for _, v in sorted(k.items()) if abs(v) > 1e-8), None)
    if lead is not None and lead < 0:
        k = {n: -v for n, v in k.items()}
        p = {n: -v for n, v in p.items()}

    return {"kinetic": k, "potential": p}


def discover_structured_lagrangian(
    q: np.ndarray,
    dq: np.ndarray,
    ddq: np.ndarray,
    *,
    kinetic_max_even_power: int = 4,
    potential_max_even_power: int = 4,
    include_trig: bool = True,
) -> StructuredLagrangianResult:
    """Discover L(q,dq)=T(dq)-V(q) from trajectory arrays."""
    q_arr = np.asarray(q, dtype=float).reshape(-1)
    dq_arr = np.asarray(dq, dtype=float).reshape(-1)
    ddq_arr = np.asarray(ddq, dtype=float).reshape(-1)

    if not (len(q_arr) == len(dq_arr) == len(ddq_arr)):
        raise ValueError("q, dq, and ddq must have same length")
    if len(q_arr) < 6:
        raise ValueError("Need at least 6 samples for structured Lagrangian discovery")

    library = build_structured_library(
        q_arr,
        dq_arr,
        kinetic_max_even_power=kinetic_max_even_power,
        potential_max_even_power=potential_max_even_power,
        include_trig=include_trig,
    )

    kinetic_terms = library["kinetic_terms"]
    potential_terms = library["potential_terms"]

    A, kinetic_names, potential_names = build_el_regression_matrix(
        q_arr,
        dq_arr,
        ddq_arr,
        kinetic_terms=kinetic_terms,
        potential_terms=potential_terms,
    )

    coeffs = solve_nullspace_coefficients(A)
    stats = evaluate_el_residual(A, coeffs)

    k_count = len(kinetic_names)
    k_raw = {name: float(coeffs[i]) for i, name in enumerate(kinetic_names)}
    p_raw = {name: float(coeffs[k_count + j]) for j, name in enumerate(potential_names)}

    normalized = _normalize_gauge(k_raw, p_raw)
    k_map = normalized["kinetic"]
    p_map = normalized["potential"]

    t_expr = _term_string(k_map)
    v_expr = _term_string(p_map)
    lag_expr = f"({t_expr}) - ({v_expr})"

    return StructuredLagrangianResult(
        lagrangian=lag_expr,
        kinetic_energy=t_expr,
        potential_energy=v_expr,
        residual_mse=stats["mse"],
        residual_rmse=stats["rmse"],
        kinetic_coefficients=k_map,
        potential_coefficients=p_map,
    )


def discover_structured_lagrangian_from_dataframe(
    df: pd.DataFrame,
    q_col: str = "theta",
    dq_col: str = "omega",
    ddq_col: str = "alpha",
    **kwargs,
) -> StructuredLagrangianResult:
    """Convenience wrapper using DataFrame columns."""
    for col in (q_col, dq_col, ddq_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' missing from dataframe")

    return discover_structured_lagrangian(
        q=df[q_col].to_numpy(dtype=float),
        dq=df[dq_col].to_numpy(dtype=float),
        ddq=df[ddq_col].to_numpy(dtype=float),
        **kwargs,
    )


def discover_structured_lagrangian_from_csv(
    csv_path: str | Path,
    q_col: str = "theta",
    dq_col: str = "omega",
    ddq_col: str = "alpha",
    **kwargs,
) -> StructuredLagrangianResult:
    """Convenience wrapper loading trajectory data from CSV."""
    df = pd.read_csv(csv_path)
    return discover_structured_lagrangian_from_dataframe(
        df=df,
        q_col=q_col,
        dq_col=dq_col,
        ddq_col=ddq_col,
        **kwargs,
    )


__all__ = [
    "StructuredLagrangianResult",
    "discover_structured_lagrangian",
    "discover_structured_lagrangian_from_dataframe",
    "discover_structured_lagrangian_from_csv",
]
