from __future__ import annotations

import re
from dataclasses import dataclass

from physics_engine.knowledge.models import KnowledgeLaw


@dataclass(frozen=True)
class ValidationResult:
    is_dimensionally_plausible: bool
    closest_known_law: str | None
    token_overlap_score: float
    residual_quality_score: float
    overall_score: float
    explanation: str


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", (text or "").lower()) if len(token) > 0}


def _best_known_law_match(discovered_equation: str, known_laws: list[KnowledgeLaw]) -> tuple[str | None, float]:
    eq_tokens = _tokens(discovered_equation)
    if not eq_tokens or not known_laws:
        return None, 0.0

    best_name = None
    best_score = 0.0
    for law in known_laws:
        law_tokens = _tokens(law.equation) | _tokens(law.law) | {v.lower() for v in law.variables}
        overlap = len(eq_tokens & law_tokens)
        score = overlap / max(len(eq_tokens), 1)
        if score > best_score:
            best_score = score
            best_name = law.law

    return best_name, float(best_score)


def _residual_quality(validation_error: float, significant_correlations: dict[str, float] | None = None) -> float:
    error_component = max(0.0, 1.0 - float(validation_error)) if validation_error >= 0 else 0.0
    significant_correlations = significant_correlations or {}

    max_corr = 0.0
    for value in significant_correlations.values():
        try:
            max_corr = max(max_corr, abs(float(value)))
        except (TypeError, ValueError):
            continue

    residual_component = max(0.0, 1.0 - max_corr)
    return 0.6 * error_component + 0.4 * residual_component


def validate_candidate_law(
    discovered_equation: str,
    validation_error: float,
    known_laws: list[KnowledgeLaw],
    significant_correlations: dict[str, float] | None = None,
    dimensionally_plausible: bool = True,
) -> ValidationResult:
    closest_law, overlap = _best_known_law_match(discovered_equation, known_laws)
    residual_score = _residual_quality(validation_error, significant_correlations)

    overall = 0.4 * (1.0 if dimensionally_plausible else 0.0) + 0.35 * overlap + 0.25 * residual_score
    overall = max(0.0, min(1.0, overall))

    if closest_law:
        explanation = (
            f"Candidate equation aligns most with '{closest_law}' (token overlap={overlap:.2f}) "
            f"with residual quality={residual_score:.2f}."
        )
    else:
        explanation = (
            f"No close known-law match found; residual quality={residual_score:.2f}."
        )

    return ValidationResult(
        is_dimensionally_plausible=bool(dimensionally_plausible),
        closest_known_law=closest_law,
        token_overlap_score=overlap,
        residual_quality_score=residual_score,
        overall_score=overall,
        explanation=explanation,
    )
