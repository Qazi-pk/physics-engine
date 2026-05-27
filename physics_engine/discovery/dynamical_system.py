from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from physics_engine.pir import PIRDifferentialEquation, PIRSystem

from .symbolic_search import discover_law


@dataclass(frozen=True)
class DiscoveredEquationResult:
    target: str
    law: str
    error: float
    significant: dict[str, float]


def _target_to_lhs(target: str) -> str:
    if target.endswith("_dot"):
        state = target.removesuffix("_dot")
        return f"d{state}/dt"
    return target


def discover_dynamical_system(
    csv_path: str,
    targets: list[str],
    **discover_kwargs: Any,
) -> tuple[PIRSystem, list[DiscoveredEquationResult]]:
    equations = []
    details: list[DiscoveredEquationResult] = []

    for target in targets:
        law, error, significant = discover_law(
            csv_path,
            target,
            **discover_kwargs,
        )
        lhs = _target_to_lhs(target)
        equation = PIRDifferentialEquation(lhs=lhs, rhs=str(law), order=1, metadata={"target": target})
        equations.append(equation)
        details.append(
            DiscoveredEquationResult(
                target=target,
                law=str(law),
                error=float(error),
                significant={str(k): float(v) for k, v in significant.items()},
            )
        )

    return PIRSystem(equations=equations), details
