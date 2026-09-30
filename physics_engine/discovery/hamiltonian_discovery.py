from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp


@dataclass(frozen=True)
class HamiltonianDiscoveryResult:
    hamiltonian: str
    error: float
    dqdt_mse: float
    dpdt_mse: float


def _build_basis(q_symbol: sp.Symbol, p_symbol: sp.Symbol, max_power: int, include_cross_terms: bool) -> list[sp.Expr]:
    basis: list[sp.Expr] = []
    for power in range(1, max_power + 1):
        basis.append(q_symbol**power)
        basis.append(p_symbol**power)

    if include_cross_terms:
        for i in range(1, max_power + 1):
            for j in range(1, max_power + 1):
                basis.append((q_symbol**i) * (p_symbol**j))

    unique_basis = sorted(set(basis), key=sp.srepr)
    return unique_basis


def _evaluate(expr: sp.Expr, q_values: np.ndarray, p_values: np.ndarray, q_symbol: sp.Symbol, p_symbol: sp.Symbol) -> np.ndarray:
    fn = sp.lambdify((q_symbol, p_symbol), expr, "numpy")
    values = fn(q_values, p_values)
    array_values = np.asarray(values, dtype=float).reshape(-1)
    if array_values.size == 1:
        return np.full(q_values.shape[0], float(array_values[0]), dtype=float)
    return array_values


def discover_hamiltonian(
    q: np.ndarray,
    p: np.ndarray,
    dqdt: np.ndarray,
    dpdt: np.ndarray,
    max_power: int = 2,
    include_cross_terms: bool = True,
) -> HamiltonianDiscoveryResult:
    q_values = np.asarray(q, dtype=float).reshape(-1)
    p_values = np.asarray(p, dtype=float).reshape(-1)
    dqdt_values = np.asarray(dqdt, dtype=float).reshape(-1)
    dpdt_values = np.asarray(dpdt, dtype=float).reshape(-1)

    if not (len(q_values) == len(p_values) == len(dqdt_values) == len(dpdt_values)):
        raise ValueError("q, p, dqdt, and dpdt must have the same length")
    if len(q_values) < 4:
        raise ValueError("Need at least 4 samples for Hamiltonian discovery")

    q_symbol = sp.Symbol("q")
    p_symbol = sp.Symbol("p")

    basis = _build_basis(q_symbol, p_symbol, max_power=max_power, include_cross_terms=include_cross_terms)
    if not basis:
        raise RuntimeError("No basis terms generated for Hamiltonian discovery")

    dphi_dp = [sp.diff(term, p_symbol) for term in basis]
    neg_dphi_dq = [-sp.diff(term, q_symbol) for term in basis]

    matrix_top = np.column_stack([_evaluate(term, q_values, p_values, q_symbol, p_symbol) for term in dphi_dp])
    matrix_bottom = np.column_stack([_evaluate(term, q_values, p_values, q_symbol, p_symbol) for term in neg_dphi_dq])
    matrix_a = np.vstack([matrix_top, matrix_bottom])
    vector_b = np.concatenate([dqdt_values, dpdt_values])

    coeffs, *_ = np.linalg.lstsq(matrix_a, vector_b, rcond=None)

    h_expr = sp.simplify(sum(float(coeff) * term for coeff, term in zip(coeffs, basis)))

    dq_pred = _evaluate(sp.diff(h_expr, p_symbol), q_values, p_values, q_symbol, p_symbol)
    dp_pred = _evaluate(-sp.diff(h_expr, q_symbol), q_values, p_values, q_symbol, p_symbol)

    dqdt_mse = float(np.mean((dqdt_values - dq_pred) ** 2))
    dpdt_mse = float(np.mean((dpdt_values - dp_pred) ** 2))
    error = 0.5 * (dqdt_mse + dpdt_mse)

    return HamiltonianDiscoveryResult(
        hamiltonian=str(h_expr),
        error=error,
        dqdt_mse=dqdt_mse,
        dpdt_mse=dpdt_mse,
    )


def discover_hamiltonian_from_dataframe(
    df: pd.DataFrame,
    q_col: str,
    p_col: str,
    dqdt_col: str,
    dpdt_col: str,
    **kwargs,
) -> HamiltonianDiscoveryResult:
    for col in (q_col, p_col, dqdt_col, dpdt_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' missing from dataframe")

    return discover_hamiltonian(
        q=df[q_col].to_numpy(dtype=float),
        p=df[p_col].to_numpy(dtype=float),
        dqdt=df[dqdt_col].to_numpy(dtype=float),
        dpdt=df[dpdt_col].to_numpy(dtype=float),
        **kwargs,
    )


def discover_hamiltonian_from_csv(
    csv_path: str | Path,
    q_col: str,
    p_col: str,
    dqdt_col: str,
    dpdt_col: str,
    **kwargs,
) -> HamiltonianDiscoveryResult:
    df = pd.read_csv(csv_path)
    return discover_hamiltonian_from_dataframe(
        df=df,
        q_col=q_col,
        p_col=p_col,
        dqdt_col=dqdt_col,
        dpdt_col=dpdt_col,
        **kwargs,
    )
