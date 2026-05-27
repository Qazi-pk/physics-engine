from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentProposal:
    objective: str
    suggestion: str
    expected_information_gain: float


def design_next_experiment(
    cycle_index: int,
    confidence: float,
    validation_error: float,
) -> ExperimentProposal:
    if confidence >= 0.9 and validation_error <= 0.1:
        return ExperimentProposal(
            objective="stress_test_theory",
            suggestion="Test the discovered law on a noisier or wider-range dataset.",
            expected_information_gain=0.35,
        )

    if confidence < 0.6:
        return ExperimentProposal(
            objective="reduce_model_uncertainty",
            suggestion="Collect data in regimes with stronger signal and broader variable ranges.",
            expected_information_gain=0.8,
        )

    return ExperimentProposal(
        objective="resolve_parameter_uncertainty",
        suggestion="Increase samples near high-curvature regions to refine coefficients/exponents.",
        expected_information_gain=0.55,
    )
