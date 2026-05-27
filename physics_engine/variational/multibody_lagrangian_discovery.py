"""Structured 2-DOF multibody Lagrangian discovery.

Model form:

    L(q, dq) = 0.5 * dq^T M(q) dq - V(q)

where M(q) is symmetric with entries M11, M12, M22 parameterized by basis
features in (q1, q2), and V(q) is likewise basis-parameterized.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .multibody_library import MultiBodyBasisTerm, multibody_basis_terms


@dataclass(frozen=True)
class MultiBodyLagrangianResult:
    lagrangian: str
    mass_matrix: Dict[str, str]
    potential_energy: str
    tau_mse: float
    tau_rmse: float
    mass_coefficients: Dict[str, Dict[str, float]]
    potential_coefficients: Dict[str, float]
    mass_spd_fraction: float


def _term_string(coeff_map: Dict[str, float]) -> str:
    items = [(name, coeff) for name, coeff in coeff_map.items() if abs(coeff) > 1e-8]
    if not items:
        return "0"
    return " + ".join(f"{coeff:.6g}*{name}" for name, coeff in items)


def _normalize_coefficients(
    m11: Dict[str, float],
    m12: Dict[str, float],
    m22: Dict[str, float],
    v: Dict[str, float],
) -> Dict[str, Dict[str, float]]:
    all_vals = list(m11.values()) + list(m12.values()) + list(m22.values()) + list(v.values())
    max_abs = max((abs(x) for x in all_vals), default=0.0)
    if max_abs <= 0.0:
        return {"m11": m11, "m12": m12, "m22": m22, "v": v}

    m11n = {k: val / max_abs for k, val in m11.items()}
    m12n = {k: val / max_abs for k, val in m12.items()}
    m22n = {k: val / max_abs for k, val in m22.items()}
    vn = {k: val / max_abs for k, val in v.items()}

    lead = next((x for _, x in sorted(m11n.items()) if abs(x) > 1e-8), None)
    if lead is not None and lead < 0:
        m11n = {k: -val for k, val in m11n.items()}
        m12n = {k: -val for k, val in m12n.items()}
        m22n = {k: -val for k, val in m22n.items()}
        vn = {k: -val for k, val in vn.items()}

    return {"m11": m11n, "m12": m12n, "m22": m22n, "v": vn}


def _build_design_matrix(
    q1: np.ndarray,
    q2: np.ndarray,
    dq1: np.ndarray,
    dq2: np.ndarray,
    ddq1: np.ndarray,
    ddq2: np.ndarray,
    basis_terms: List[MultiBodyBasisTerm],
) -> tuple[np.ndarray, List[str]]:
    n = q1.shape[0]
    feat = len(basis_terms)

    phi = np.column_stack([t.value(q1, q2) for t in basis_terms])
    dphi_q1 = np.column_stack([t.d_q1(q1, q2) for t in basis_terms])
    dphi_q2 = np.column_stack([t.d_q2(q1, q2) for t in basis_terms])

    x = np.zeros((2 * n, 4 * feat), dtype=float)
    a = slice(0, feat)
    b = slice(feat, 2 * feat)
    c = slice(2 * feat, 3 * feat)
    g = slice(3 * feat, 4 * feat)

    # tau1 rows
    x[:n, a] = (
        phi * ddq1[:, None]
        + 0.5 * dphi_q1 * (dq1[:, None] ** 2)
        + dphi_q2 * (dq1 * dq2)[:, None]
    )
    x[:n, b] = phi * ddq2[:, None] + dphi_q2 * (dq2[:, None] ** 2)
    x[:n, c] = -0.5 * dphi_q1 * (dq2[:, None] ** 2)
    x[:n, g] = dphi_q1

    # tau2 rows
    x[n:, a] = -0.5 * dphi_q2 * (dq1[:, None] ** 2)
    x[n:, b] = phi * ddq1[:, None] + dphi_q1 * (dq1[:, None] ** 2)
    x[n:, c] = (
        phi * ddq2[:, None]
        + dphi_q1 * (dq1 * dq2)[:, None]
        + 0.5 * dphi_q2 * (dq2[:, None] ** 2)
    )
    x[n:, g] = dphi_q2

    names = [t.name for t in basis_terms]
    return x, names


def discover_multibody_lagrangian(
    q1: np.ndarray,
    q2: np.ndarray,
    dq1: np.ndarray,
    dq2: np.ndarray,
    ddq1: np.ndarray,
    ddq2: np.ndarray,
    tau1: np.ndarray,
    tau2: np.ndarray,
) -> MultiBodyLagrangianResult:
    """Fit 2-DOF structured multibody Lagrangian to torque data."""
    arrays = [q1, q2, dq1, dq2, ddq1, ddq2, tau1, tau2]
    vecs = [np.asarray(v, dtype=float).reshape(-1) for v in arrays]
    n = vecs[0].shape[0]
    if any(v.shape[0] != n for v in vecs):
        raise ValueError("All input arrays must have the same length")
    if n < 8:
        raise ValueError("Need at least 8 samples for multibody discovery")

    q1a, q2a, dq1a, dq2a, ddq1a, ddq2a, tau1a, tau2a = vecs
    basis_terms = multibody_basis_terms()

    x, names = _build_design_matrix(
        q1=q1a,
        q2=q2a,
        dq1=dq1a,
        dq2=dq2a,
        ddq1=ddq1a,
        ddq2=ddq2a,
        basis_terms=basis_terms,
    )
    y = np.concatenate([tau1a, tau2a])

    theta, *_ = np.linalg.lstsq(x, y, rcond=None)
    pred = x @ theta
    resid = pred - y
    mse = float(np.mean(resid**2))
    rmse = float(np.sqrt(mse))

    feat = len(names)
    m11 = {name: float(theta[i]) for i, name in enumerate(names)}
    m12 = {name: float(theta[feat + i]) for i, name in enumerate(names)}
    m22 = {name: float(theta[2 * feat + i]) for i, name in enumerate(names)}
    v = {name: float(theta[3 * feat + i]) for i, name in enumerate(names)}

    normalized = _normalize_coefficients(m11, m12, m22, v)
    m11n = normalized["m11"]
    m12n = normalized["m12"]
    m22n = normalized["m22"]
    vn = normalized["v"]

    # SPD sanity metric on fitted M(q)
    phi = np.column_stack([t.value(q1a, q2a) for t in basis_terms])
    m11v = phi @ np.array([m11n[k] for k in names], dtype=float)
    m12v = phi @ np.array([m12n[k] for k in names], dtype=float)
    m22v = phi @ np.array([m22n[k] for k in names], dtype=float)
    det = m11v * m22v - m12v**2
    spd_fraction = float(np.mean((m11v > 0.0) & (m22v > 0.0) & (det > 0.0)))

    m11_expr = _term_string(m11n)
    m12_expr = _term_string(m12n)
    m22_expr = _term_string(m22n)
    v_expr = _term_string(vn)

    t_expr = f"0.5*({m11_expr})*dq1**2 + ({m12_expr})*dq1*dq2 + 0.5*({m22_expr})*dq2**2"
    lag_expr = f"({t_expr}) - ({v_expr})"

    return MultiBodyLagrangianResult(
        lagrangian=lag_expr,
        mass_matrix={"M11": m11_expr, "M12": m12_expr, "M22": m22_expr},
        potential_energy=v_expr,
        tau_mse=mse,
        tau_rmse=rmse,
        mass_coefficients={"M11": m11n, "M12": m12n, "M22": m22n},
        potential_coefficients=vn,
        mass_spd_fraction=spd_fraction,
    )


def discover_multibody_lagrangian_from_dataframe(
    df: pd.DataFrame,
    *,
    q1_col: str = "q1",
    q2_col: str = "q2",
    dq1_col: str = "dq1",
    dq2_col: str = "dq2",
    ddq1_col: str = "ddq1",
    ddq2_col: str = "ddq2",
    tau1_col: str = "tau1",
    tau2_col: str = "tau2",
) -> MultiBodyLagrangianResult:
    for col in (q1_col, q2_col, dq1_col, dq2_col, ddq1_col, ddq2_col, tau1_col, tau2_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' missing from dataframe")

    return discover_multibody_lagrangian(
        q1=df[q1_col].to_numpy(dtype=float),
        q2=df[q2_col].to_numpy(dtype=float),
        dq1=df[dq1_col].to_numpy(dtype=float),
        dq2=df[dq2_col].to_numpy(dtype=float),
        ddq1=df[ddq1_col].to_numpy(dtype=float),
        ddq2=df[ddq2_col].to_numpy(dtype=float),
        tau1=df[tau1_col].to_numpy(dtype=float),
        tau2=df[tau2_col].to_numpy(dtype=float),
    )


def discover_multibody_lagrangian_from_csv(
    csv_path: str | Path,
    **kwargs,
) -> MultiBodyLagrangianResult:
    df = pd.read_csv(csv_path)
    return discover_multibody_lagrangian_from_dataframe(df, **kwargs)


__all__ = [
    "MultiBodyLagrangianResult",
    "discover_multibody_lagrangian",
    "discover_multibody_lagrangian_from_dataframe",
    "discover_multibody_lagrangian_from_csv",
]
