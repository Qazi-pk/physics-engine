from __future__ import annotations

from dataclasses import dataclass

from .structure_detection import StructureDetectionResult


@dataclass(frozen=True)
class DecompositionPlan:
    structure: str
    variables: tuple[str, str] | None
    blocked_interactions: tuple[tuple[str, str], ...]


def decompose_problem(result: StructureDetectionResult) -> DecompositionPlan:
    if result.structure == "additive" and result.variables is not None:
        a, b = result.variables
        return DecompositionPlan(
            structure="additive",
            variables=(a, b),
            blocked_interactions=((a, b),),
        )

    return DecompositionPlan(
        structure="unknown",
        variables=result.variables,
        blocked_interactions=(),
    )
