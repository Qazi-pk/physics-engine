"""
Data-model for nodes in the Physics Knowledge Graph.

Each node type carries its own metadata schema.  All nodes share a
``node_id``, ``kind``, and an optional ``metadata`` dict for
extensible annotations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


NodeKind = str  # "law" | "dataset" | "experiment" | "variable" | "invariant" | "algorithm"


@dataclass
class LawNode:
    """
    A discovered or known physics law.

    Attributes:
        node_id:     Unique identifier, e.g. ``"newton_second_law"``
        equation:    Symbolic string, e.g. ``"F = m * a"``
        variables:   List of variable names appearing in the equation
        domain:      Physics domain: ``"mechanics"``, ``"robotics"``, …
        source:      How it was produced: ``"known"``, ``"discovered"``, …
        confidence:  0–1 confidence score from the discovery algorithm
        error:       Residual / fit error (lower is better)
        metadata:    Arbitrary extra annotations
    """

    node_id: str
    equation: str
    variables: List[str] = field(default_factory=list)
    domain: str = "unknown"
    source: str = "discovered"
    confidence: float = 1.0
    error: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def kind(self) -> NodeKind:
        return "law"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "kind": self.kind,
            "equation": self.equation,
            "variables": self.variables,
            "domain": self.domain,
            "source": self.source,
            "confidence": self.confidence,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class DatasetNode:
    """A dataset used for discovery or validation."""

    node_id: str
    path: str = ""
    samples: int = 0
    noise_level: float = 0.0
    domain: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def kind(self) -> NodeKind:
        return "dataset"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "kind": self.kind,
            "path": self.path,
            "samples": self.samples,
            "noise_level": self.noise_level,
            "domain": self.domain,
            "metadata": self.metadata,
        }


@dataclass
class ExperimentNode:
    """A single benchmark experiment run."""

    node_id: str
    experiment_id: str = ""
    algorithm: str = ""
    dataset: str = ""
    seed: int = 0
    success: bool = False
    runtime_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def kind(self) -> NodeKind:
        return "experiment"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "kind": self.kind,
            "experiment_id": self.experiment_id,
            "algorithm": self.algorithm,
            "dataset": self.dataset,
            "seed": self.seed,
            "success": self.success,
            "runtime_seconds": self.runtime_seconds,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class VariableNode:
    """A physical variable (e.g. theta, omega, torque)."""

    node_id: str
    symbol: str = ""
    unit: str = ""
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def kind(self) -> NodeKind:
        return "variable"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "kind": self.kind,
            "symbol": self.symbol,
            "unit": self.unit,
            "description": self.description,
            "metadata": self.metadata,
        }


@dataclass
class InvariantNode:
    """A conserved or invariant quantity (e.g. total energy)."""

    node_id: str
    expression: str = ""
    conservation_law: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def kind(self) -> NodeKind:
        return "invariant"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "kind": self.kind,
            "expression": self.expression,
            "conservation_law": self.conservation_law,
            "metadata": self.metadata,
        }


@dataclass
class AlgorithmNode:
    """A discovery or identification algorithm."""

    node_id: str
    display_name: str = ""
    family: str = ""          # "symbolic", "neural", "regression", …
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def kind(self) -> NodeKind:
        return "algorithm"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "kind": self.kind,
            "display_name": self.display_name,
            "family": self.family,
            "description": self.description,
            "metadata": self.metadata,
        }


# Union type for any graph node
AnyNode = LawNode | DatasetNode | ExperimentNode | VariableNode | InvariantNode | AlgorithmNode


__all__ = [
    "NodeKind",
    "AnyNode",
    "LawNode",
    "DatasetNode",
    "ExperimentNode",
    "VariableNode",
    "InvariantNode",
    "AlgorithmNode",
]
