from dataclasses import dataclass

import numpy as np
import sympy as sp


@dataclass(frozen=True)
class LagrangianValidationResult:
    euler_lagrange_mse: float


def validate_lagrangian(
    lagrangian_expr: str,
    q: np.ndarray,
    dqdt: np.ndarray,
    d2qdt2: np.ndarray,
) -> LagrangianValidationResult:
    q_values = np.asarray(q, dtype=float).reshape(-1)
    v_values = np.asarray(dqdt, dtype=float).reshape(-1)
    a_values = np.asarray(d2qdt2, dtype=float).reshape(-1)

    if not (len(q_values) == len(v_values) == len(a_values)):
        raise ValueError("q, dqdt, and d2qdt2 must have equal lengths")

    q_symbol = sp.Symbol("q")
    v_symbol = sp.Symbol("dqdt")
    l_expr = sp.sympify(lagrangian_expr)

    d_l_dq = sp.diff(l_expr, q_symbol)
    d2_l_dqdv = sp.diff(sp.diff(l_expr, v_symbol), q_symbol)
    d2_l_dv2 = sp.diff(sp.diff(l_expr, v_symbol), v_symbol)

    d_l_dq_fn = sp.lambdify((q_symbol, v_symbol), d_l_dq, "numpy")
    d2_l_dqdv_fn = sp.lambdify((q_symbol, v_symbol), d2_l_dqdv, "numpy")
    d2_l_dv2_fn = sp.lambdify((q_symbol, v_symbol), d2_l_dv2, "numpy")

    eval_d_l_dq = np.asarray(d_l_dq_fn(q_values, v_values), dtype=float).reshape(-1)
    eval_d2_l_dqdv = np.asarray(d2_l_dqdv_fn(q_values, v_values), dtype=float).reshape(-1)
    eval_d2_l_dv2 = np.asarray(d2_l_dv2_fn(q_values, v_values), dtype=float).reshape(-1)

    if eval_d_l_dq.size == 1:
        eval_d_l_dq = np.full(q_values.shape[0], float(eval_d_l_dq[0]), dtype=float)
    if eval_d2_l_dqdv.size == 1:
        eval_d2_l_dqdv = np.full(q_values.shape[0], float(eval_d2_l_dqdv[0]), dtype=float)
    if eval_d2_l_dv2.size == 1:
        eval_d2_l_dv2 = np.full(q_values.shape[0], float(eval_d2_l_dv2[0]), dtype=float)

    residual = eval_d2_l_dqdv * v_values + eval_d2_l_dv2 * a_values - eval_d_l_dq
    mse = float(np.mean(residual**2))

    return LagrangianValidationResult(euler_lagrange_mse=mse)
