from dataclasses import dataclass

import numpy as np
import sympy as sp


@dataclass(frozen=True)
class HamiltonianValidationResult:
    dqdt_mse: float
    dpdt_mse: float
    total_mse: float


def validate_hamiltonian(
    hamiltonian_expr: str,
    q: np.ndarray,
    p: np.ndarray,
    dqdt: np.ndarray,
    dpdt: np.ndarray,
) -> HamiltonianValidationResult:
    q_values = np.asarray(q, dtype=float).reshape(-1)
    p_values = np.asarray(p, dtype=float).reshape(-1)
    dqdt_values = np.asarray(dqdt, dtype=float).reshape(-1)
    dpdt_values = np.asarray(dpdt, dtype=float).reshape(-1)

    if not (len(q_values) == len(p_values) == len(dqdt_values) == len(dpdt_values)):
        raise ValueError("q, p, dqdt, and dpdt must have equal lengths")

    q_symbol = sp.Symbol("q")
    p_symbol = sp.Symbol("p")
    h_expr = sp.sympify(hamiltonian_expr)

    dq_fn = sp.lambdify((q_symbol, p_symbol), sp.diff(h_expr, p_symbol), "numpy")
    dp_fn = sp.lambdify((q_symbol, p_symbol), -sp.diff(h_expr, q_symbol), "numpy")

    dq_pred = np.asarray(dq_fn(q_values, p_values), dtype=float).reshape(-1)
    dp_pred = np.asarray(dp_fn(q_values, p_values), dtype=float).reshape(-1)

    dqdt_mse = float(np.mean((dqdt_values - dq_pred) ** 2))
    dpdt_mse = float(np.mean((dpdt_values - dp_pred) ** 2))
    total_mse = 0.5 * (dqdt_mse + dpdt_mse)

    return HamiltonianValidationResult(
        dqdt_mse=dqdt_mse,
        dpdt_mse=dpdt_mse,
        total_mse=total_mse,
    )
