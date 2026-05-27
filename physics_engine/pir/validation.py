from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence, Tuple, Union

import sympy as sp

from .dimensional_analysis import dimensions_of_expression
from .representation import Equation, PIRHamiltonian


ExpressionLike = Union[str, sp.Expr]
EquationLike = Union[Equation, sp.Equality]
DimensionVector = Tuple[Union[int, sp.Rational], Union[int, sp.Rational], Union[int, sp.Rational]]


@dataclass(frozen=True)
class PIRValidationPassResult:
    name: str
    passed: bool
    details: str


def _as_expression(expr: ExpressionLike) -> sp.Expr:
    if isinstance(expr, str):
        return sp.sympify(expr)
    return expr


def _as_equation(equation: EquationLike) -> sp.Equality:
    if isinstance(equation, Equation):
        return sp.Eq(sp.sympify(equation.lhs), sp.sympify(equation.rhs))
    return equation


def validate_dimensional_consistency(
    equation: EquationLike,
    symbol_dimensions: Mapping[str, DimensionVector],
) -> PIRValidationPassResult:
    eq = _as_equation(equation)
    lhs_dim = dimensions_of_expression(eq.lhs, symbol_dimensions)
    rhs_dim = dimensions_of_expression(eq.rhs, symbol_dimensions)
    passed = lhs_dim == rhs_dim
    details = f"lhs={lhs_dim}, rhs={rhs_dim}"
    return PIRValidationPassResult(name="dimensional_consistency", passed=passed, details=details)


def detect_exchange_symmetry(
    expression: ExpressionLike,
    var_a: str,
    var_b: str,
) -> PIRValidationPassResult:
    expr = _as_expression(expression)
    a_symbol = sp.Symbol(var_a)
    b_symbol = sp.Symbol(var_b)
    swapped = expr.subs({a_symbol: b_symbol, b_symbol: a_symbol}, simultaneous=True)
    symmetric = sp.simplify(expr - swapped) == 0
    details = f"swap({var_a},{var_b})={'symmetric' if symmetric else 'not_symmetric'}"
    return PIRValidationPassResult(name="exchange_symmetry", passed=bool(symmetric), details=details)


def detect_time_invariance(
    expression: ExpressionLike,
    time_variable: str = "t",
) -> PIRValidationPassResult:
    expr = _as_expression(expression)
    time_symbol = sp.Symbol(time_variable)
    invariant = time_symbol not in expr.free_symbols
    details = f"time_symbol_present={not invariant}"
    return PIRValidationPassResult(name="time_invariance", passed=invariant, details=details)


def detect_energy_conservation_candidate(
    hamiltonian: Union[PIRHamiltonian, ExpressionLike],
    time_variable: str = "t",
) -> PIRValidationPassResult:
    if isinstance(hamiltonian, PIRHamiltonian):
        expression = hamiltonian.expression
    else:
        expression = hamiltonian
    base_result = detect_time_invariance(expression, time_variable=time_variable)
    details = (
        "Hamiltonian has no explicit time dependence"
        if base_result.passed
        else "Hamiltonian depends explicitly on time"
    )
    return PIRValidationPassResult(
        name="energy_conservation_candidate",
        passed=base_result.passed,
        details=details,
    )


def run_validation_passes(
    equation: Optional[EquationLike] = None,
    expression: Optional[ExpressionLike] = None,
    symbol_dimensions: Optional[Mapping[str, DimensionVector]] = None,
    exchange_pairs: Optional[Sequence[Tuple[str, str]]] = None,
    check_time_invariance: bool = False,
    time_variable: str = "t",
) -> list[PIRValidationPassResult]:
    results: list[PIRValidationPassResult] = []

    if equation is not None and symbol_dimensions is not None:
        results.append(validate_dimensional_consistency(equation, symbol_dimensions))

    if expression is not None and exchange_pairs:
        for var_a, var_b in exchange_pairs:
            results.append(detect_exchange_symmetry(expression, var_a=var_a, var_b=var_b))

    if expression is not None and check_time_invariance:
        results.append(detect_time_invariance(expression, time_variable=time_variable))

    return results


def all_passed(results: Iterable[PIRValidationPassResult]) -> bool:
    return all(result.passed for result in results)
