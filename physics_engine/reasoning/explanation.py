from __future__ import annotations

from .hypothesis_generation import Hypothesis
from .law_validation import ValidationResult


def generate_hypothesis_explanation(
    hypotheses: list[Hypothesis],
    validation: ValidationResult | None = None,
) -> str:
    if not hypotheses:
        base = "No hypothesis generated."
    else:
        base = hypotheses[0].statement

    if validation is None:
        return base

    return f"{base} {validation.explanation}"
