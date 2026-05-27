from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp


@dataclass(frozen=True)
class LagrangianDiscoveryResult:
    lagrangian: str
    error: float
    euler_lagrange_mse: float


def _build_basis(q_symbol: sp.Symbol, v_symbol: sp.Symbol, max_power: int, include_cross_terms: bool) -> list[sp.Expr]:
    basis: list[sp.Expr] = []
    for power in range(1, max_power + 1):
        basis.append(q_symbol**power)
        if power != 1:
            basis.append(v_symbol**power)

    if include_cross_terms:
        for i in range(1, max_power + 1):
            for j in range(1, max_power + 1):
                basis.append((q_symbol**i) * (v_symbol**j))

    return sorted(set(basis), key=sp.srepr)


def _evaluate(expr: sp.Expr, q_values: np.ndarray, v_values: np.ndarray, q_symbol: sp.Symbol, v_symbol: sp.Symbol) -> np.ndarray:
    fn = sp.lambdify((q_symbol, v_symbol), expr, "numpy")
    values = fn(q_values, v_values)
    array_values = np.asarray(values, dtype=float).reshape(-1)
    if array_values.size == 1:
        return np.full(q_values.shape[0], float(array_values[0]), dtype=float)
    return array_values


def _euler_lagrange_feature(term: sp.Expr, q_values: np.ndarray, v_values: np.ndarray, a_values: np.ndarray, q_symbol: sp.Symbol, v_symbol: sp.Symbol) -> np.ndarray:
    d_term_dq = sp.diff(term, q_symbol)
    d2_term_dqdv = sp.diff(sp.diff(term, v_symbol), q_symbol)
    d2_term_dv2 = sp.diff(sp.diff(term, v_symbol), v_symbol)

    eval_d_term_dq = _evaluate(d_term_dq, q_values, v_values, q_symbol, v_symbol)
    eval_d2_term_dqdv = _evaluate(d2_term_dqdv, q_values, v_values, q_symbol, v_symbol)
    eval_d2_term_dv2 = _evaluate(d2_term_dv2, q_values, v_values, q_symbol, v_symbol)

    return eval_d2_term_dqdv * v_values + eval_d2_term_dv2 * a_values - eval_d_term_dq


def discover_lagrangian(
    q: np.ndarray,
    dqdt: np.ndarray,
    d2qdt2: np.ndarray,
    max_power: int = 2,
    include_cross_terms: bool = True,
) -> LagrangianDiscoveryResult:
    q_values = np.asarray(q, dtype=float).reshape(-1)
    v_values = np.asarray(dqdt, dtype=float).reshape(-1)
    a_values = np.asarray(d2qdt2, dtype=float).reshape(-1)

    if not (len(q_values) == len(v_values) == len(a_values)):
        raise ValueError("q, dqdt, and d2qdt2 must have the same length")
    if len(q_values) < 4:
        raise ValueError("Need at least 4 samples for Lagrangian discovery")

    q_symbol = sp.Symbol("q")
    v_symbol = sp.Symbol("dqdt")

    basis = _build_basis(q_symbol, v_symbol, max_power=max_power, include_cross_terms=include_cross_terms)
    if not basis:
        raise RuntimeError("No basis terms generated for Lagrangian discovery")

    feature_matrix = np.column_stack(
        [
            _euler_lagrange_feature(term, q_values, v_values, a_values, q_symbol, v_symbol)
            for term in basis
        ]
    )

    _, _, vt = np.linalg.svd(feature_matrix, full_matrices=False)
    coeffs = vt[-1, :]

    max_abs = float(np.max(np.abs(coeffs)))
    if max_abs > 0.0:
        coeffs = coeffs / max_abs

    lagrangian_expr = sp.simplify(sum(float(coeff) * term for coeff, term in zip(coeffs, basis)))
    residual = feature_matrix @ coeffs
    mse = float(np.mean(residual**2))

    return LagrangianDiscoveryResult(
        lagrangian=str(lagrangian_expr),
        error=mse,
        euler_lagrange_mse=mse,
    )


def discover_lagrangian_from_dataframe(
    df: pd.DataFrame,
    q_col: str,
    dqdt_col: str,
    d2qdt2_col: str,
    **kwargs,
) -> LagrangianDiscoveryResult:
    for col in (q_col, dqdt_col, d2qdt2_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' missing from dataframe")

    return discover_lagrangian(
        q=df[q_col].to_numpy(dtype=float),
        dqdt=df[dqdt_col].to_numpy(dtype=float),
        d2qdt2=df[d2qdt2_col].to_numpy(dtype=float),
        **kwargs,
    )


def discover_lagrangian_from_csv(
    csv_path: str | Path,
    q_col: str,
    dqdt_col: str,
    d2qdt2_col: str,
    **kwargs,
) -> LagrangianDiscoveryResult:
    df = pd.read_csv(csv_path)
    return discover_lagrangian_from_dataframe(
        df=df,
        q_col=q_col,
        dqdt_col=dqdt_col,
        d2qdt2_col=d2qdt2_col,
        **kwargs,
    )
