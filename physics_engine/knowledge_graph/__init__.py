"""
physics_engine.knowledge_graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A graph-based store for accumulated physics knowledge.

Nodes represent physical entities:
  - LawNode        - discovered or known physics laws
  - DatasetNode    - experimental datasets
  - ExperimentNode - individual benchmark runs
  - VariableNode   - physical variables (theta, omega, …)
  - InvariantNode  - conserved quantities
  - AlgorithmNode  - discovery/identification algorithms

Edges carry typed relations (RelationType) such as:
  - validated_on, derived_from, consistent_with, conserves, …

Typical usage::

    from physics_engine.knowledge_graph import PhysicsKnowledgeGraph, RelationType

    g = PhysicsKnowledgeGraph()

    g.add_law(
        "robot_dynamics",
        "tau = I*alpha + b*omega + k*theta",
        variables=["tau", "alpha", "omega", "theta", "I", "b", "k"],
        domain="robotics",
        source="discovered",
        confidence=0.95,
    )

    g.add_dataset("robot_joint_1dof", samples=500, noise_level=0.01, domain="robotics")

    g.add_relation("robot_dynamics", RelationType.VALIDATED_ON, "robot_joint_1dof")

    # Persist and reload
    g.save("results/knowledge_graph.json")
    g2 = PhysicsKnowledgeGraph.load("results/knowledge_graph.json")

    print(g.to_markdown())
"""

from .graph_store import Edge, PhysicsKnowledgeGraph
from .law_node import (
    AlgorithmNode,
    AnyNode,
    DatasetNode,
    ExperimentNode,
    InvariantNode,
    LawNode,
    NodeKind,
    VariableNode,
)
from .relation_types import RELATION_LABELS, RelationType, describe_relation

__all__ = [
    # Graph
    "PhysicsKnowledgeGraph",
    "Edge",
    # Node types
    "AnyNode",
    "NodeKind",
    "LawNode",
    "DatasetNode",
    "ExperimentNode",
    "VariableNode",
    "InvariantNode",
    "AlgorithmNode",
    # Relations
    "RelationType",
    "RELATION_LABELS",
    "describe_relation",
]
