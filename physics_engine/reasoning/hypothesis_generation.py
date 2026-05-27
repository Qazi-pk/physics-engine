from __future__ import annotations

from dataclasses import dataclass

from physics_engine.knowledge.models import KnowledgeLaw


@dataclass(frozen=True)
class Hypothesis:
    statement: str
    candidate_law: str | None
    confidence_hint: float


def generate_hypotheses(
    question: str,
    known_laws: list[KnowledgeLaw] | None = None,
    discovered_equation: str | None = None,
) -> list[Hypothesis]:
    hypotheses: list[Hypothesis] = []

    if known_laws:
        top = known_laws[0]
        hypotheses.append(
            Hypothesis(
                statement=f"The query may correspond to {top.law}: {top.equation}",
                candidate_law=top.law,
                confidence_hint=0.8,
            )
        )

    if discovered_equation:
        hypotheses.append(
            Hypothesis(
                statement=f"Data-driven discovery suggests equation: {discovered_equation}",
                candidate_law=None,
                confidence_hint=0.7,
            )
        )

    if not hypotheses:
        hypotheses.append(
            Hypothesis(
                statement="Insufficient signal for a specific law; gather more constraints or data.",
                candidate_law=None,
                confidence_hint=0.2,
            )
        )

    return hypotheses
