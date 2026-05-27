from __future__ import annotations

import numpy as np

from .experiment_registry import ExperimentConfig


def compute_discovery_confidence(
    config: ExperimentConfig,
    law_text: str,
    error: float,
    reason: str,
    exception: str | None = None,
) -> float:
    """
    Compute a normalized discovery confidence score in [0, 1].

    Components:
    - token coverage (how many expected law tokens appear)
    - normalized error margin against benchmark threshold
    - execution stability (penalize exceptions)
    """

    if exception is not None:
        return 0.0

    compact = (law_text or "").replace(" ", "")
    expected = tuple(config.expected_tokens)

    if len(expected) == 0:
        token_coverage = 1.0
    else:
        matched = sum(1 for token in expected if token in compact)
        token_coverage = float(matched) / float(len(expected))

    if np.isfinite(error) and config.error_threshold > 0:
        normalized_error = max(0.0, 1.0 - (float(error) / float(config.error_threshold)))
    else:
        normalized_error = 0.0

    execution_stability = 1.0

    score = 0.55 * token_coverage + 0.35 * normalized_error + 0.10 * execution_stability

    if reason == "expected_tokens_and_error":
        score = max(score, 0.8)
    elif reason == "expected_tokens":
        score = max(score, 0.75)
    elif reason == "error_threshold":
        score = max(score, 0.6)

    return float(np.clip(score, 0.0, 1.0))
