"""
Experiment hashing utilities for reproducible benchmark IDs.

The hash is deterministic across runs for the same canonical experiment config.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict


def normalize_experiment_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return a canonicalized config dictionary for stable hashing."""
    normalized = dict(config)
    if "noise_level" in normalized:
        normalized["noise_level"] = float(normalized["noise_level"])
    if "dataset_size" in normalized:
        normalized["dataset_size"] = int(normalized["dataset_size"])
    if "seed" in normalized:
        normalized["seed"] = int(normalized["seed"])
    return normalized


def compute_experiment_hash(config: Dict[str, Any], length: int = 16) -> str:
    """
    Compute deterministic experiment hash from canonical config.

    Args:
        config: Experiment configuration dictionary
        length: Optional short hash length for filenames (default: 16)

    Returns:
        SHA256-based experiment identifier
    """
    canonical = normalize_experiment_config(config)
    config_str = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(config_str.encode("utf-8")).hexdigest()
    return digest[:length] if length and length < len(digest) else digest


__all__ = [
    "normalize_experiment_config",
    "compute_experiment_hash",
]
