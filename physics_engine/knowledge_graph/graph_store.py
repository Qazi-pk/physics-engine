"""
Core Physics Knowledge Graph store.

Provides an in-memory graph of physics nodes (laws, datasets,
experiments, variables, invariants, algorithms) linked by typed
relations.  The graph can be persisted to / loaded from JSON and
queried by node kind, relation type, or domain.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple

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
from .relation_types import RelationType


# ---------------------------------------------------------------------------
# Edge
# ---------------------------------------------------------------------------

class Edge:
    """A directed, typed relationship between two graph nodes."""

    def __init__(
        self,
        source: str,
        relation: RelationType | str,
        target: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.source = source
        self.relation = RelationType(relation) if not isinstance(relation, RelationType) else relation
        self.target = target
        self.metadata: Dict[str, Any] = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "relation": self.relation.value,
            "target": self.target,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return f"Edge({self.source!r} --[{self.relation.value}]--> {self.target!r})"


# ---------------------------------------------------------------------------
# PhysicsKnowledgeGraph
# ---------------------------------------------------------------------------

class PhysicsKnowledgeGraph:
    """
    In-memory knowledge graph for PIR.

    Nodes are any of: LawNode, DatasetNode, ExperimentNode,
    VariableNode, InvariantNode, AlgorithmNode.

    Usage
    -----
    >>> g = PhysicsKnowledgeGraph()
    >>> g.add_law("robot_dynamics", "tau = I*alpha + b*omega + k*theta",
    ...           variables=["tau", "alpha", "omega", "theta"],
    ...           domain="robotics")
    >>> g.add_dataset("robot_joint_1dof", samples=500, noise_level=0.01)
    >>> g.add_relation("robot_dynamics", RelationType.VALIDATED_ON, "robot_joint_1dof")
    >>> g.find_relations("robot_dynamics")
    [Edge('robot_dynamics' --[validated_on]--> 'robot_joint_1dof')]
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, AnyNode] = {}
        self._edges: List[Edge] = []

    # ------------------------------------------------------------------
    # Node creation helpers
    # ------------------------------------------------------------------

    def add_law(
        self,
        node_id: str,
        equation: str,
        *,
        variables: Optional[List[str]] = None,
        domain: str = "unknown",
        source: str = "discovered",
        confidence: float = 1.0,
        error: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LawNode:
        """Add or replace a LawNode."""
        node = LawNode(
            node_id=node_id,
            equation=equation,
            variables=variables or [],
            domain=domain,
            source=source,
            confidence=confidence,
            error=error,
            metadata=metadata or {},
        )
        self._nodes[node_id] = node
        return node

    def add_dataset(
        self,
        node_id: str,
        *,
        path: str = "",
        samples: int = 0,
        noise_level: float = 0.0,
        domain: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DatasetNode:
        """Add or replace a DatasetNode."""
        node = DatasetNode(
            node_id=node_id,
            path=path,
            samples=samples,
            noise_level=noise_level,
            domain=domain,
            metadata=metadata or {},
        )
        self._nodes[node_id] = node
        return node

    def add_experiment(
        self,
        node_id: str,
        *,
        experiment_id: str = "",
        algorithm: str = "",
        dataset: str = "",
        seed: int = 0,
        success: bool = False,
        runtime_seconds: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ExperimentNode:
        """Add or replace an ExperimentNode."""
        node = ExperimentNode(
            node_id=node_id,
            experiment_id=experiment_id,
            algorithm=algorithm,
            dataset=dataset,
            seed=seed,
            success=success,
            runtime_seconds=runtime_seconds,
            metadata=metadata or {},
        )
        self._nodes[node_id] = node
        return node

    def add_variable(
        self,
        node_id: str,
        *,
        symbol: str = "",
        unit: str = "",
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> VariableNode:
        """Add or replace a VariableNode."""
        node = VariableNode(
            node_id=node_id,
            symbol=symbol,
            unit=unit,
            description=description,
            metadata=metadata or {},
        )
        self._nodes[node_id] = node
        return node

    def add_invariant(
        self,
        node_id: str,
        *,
        expression: str = "",
        conservation_law: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> InvariantNode:
        """Add or replace an InvariantNode."""
        node = InvariantNode(
            node_id=node_id,
            expression=expression,
            conservation_law=conservation_law,
            metadata=metadata or {},
        )
        self._nodes[node_id] = node
        return node

    def add_algorithm(
        self,
        node_id: str,
        *,
        display_name: str = "",
        family: str = "",
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AlgorithmNode:
        """Add or replace an AlgorithmNode."""
        node = AlgorithmNode(
            node_id=node_id,
            display_name=display_name or node_id,
            family=family,
            description=description,
            metadata=metadata or {},
        )
        self._nodes[node_id] = node
        return node

    def add_node(self, node: AnyNode) -> AnyNode:
        """Add a pre-constructed node object."""
        self._nodes[node.node_id] = node
        return node

    # ------------------------------------------------------------------
    # Edge creation
    # ------------------------------------------------------------------

    def add_relation(
        self,
        source: str,
        relation: RelationType | str,
        target: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Edge:
        """
        Add a directed relation edge.

        Args:
            source:   source node_id (need not pre-exist)
            relation: RelationType or string key
            target:   target node_id (need not pre-exist)
            metadata: optional annotation dict
        """
        edge = Edge(source, relation, target, metadata)
        self._edges.append(edge)
        return edge

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_node(self, node_id: str) -> Optional[AnyNode]:
        """Return node by id, or None."""
        return self._nodes.get(node_id)

    def nodes(self, kind: Optional[NodeKind] = None) -> List[AnyNode]:
        """Return all nodes, optionally filtered by kind."""
        if kind is None:
            return list(self._nodes.values())
        return [n for n in self._nodes.values() if n.kind == kind]

    def laws(self) -> List[LawNode]:
        return [n for n in self._nodes.values() if isinstance(n, LawNode)]

    def datasets(self) -> List[DatasetNode]:
        return [n for n in self._nodes.values() if isinstance(n, DatasetNode)]

    def experiments(self) -> List[ExperimentNode]:
        return [n for n in self._nodes.values() if isinstance(n, ExperimentNode)]

    def edges(
        self,
        relation: Optional[RelationType | str] = None,
    ) -> List[Edge]:
        """Return all edges, optionally filtered by relation type."""
        if relation is None:
            return list(self._edges)
        rel = RelationType(relation) if not isinstance(relation, RelationType) else relation
        return [e for e in self._edges if e.relation == rel]

    def find_relations(
        self,
        node_id: str,
        direction: str = "out",
        relation: Optional[RelationType | str] = None,
    ) -> List[Edge]:
        """
        Find edges connected to a node.

        Args:
            node_id:   node to search from
            direction: ``"out"`` (source), ``"in"`` (target), or ``"both"``
            relation:  optional filter by relation type
        """
        rel = RelationType(relation) if relation and not isinstance(relation, RelationType) else relation

        result: List[Edge] = []
        for e in self._edges:
            if rel and e.relation != rel:
                continue
            if direction in ("out", "both") and e.source == node_id:
                result.append(e)
            elif direction in ("in", "both") and e.target == node_id:
                result.append(e)
        return result

    def neighbors(self, node_id: str) -> List[AnyNode]:
        """Return all nodes directly connected to node_id (both directions)."""
        ids: set[str] = set()
        for e in self._edges:
            if e.source == node_id:
                ids.add(e.target)
            elif e.target == node_id:
                ids.add(e.source)
        return [self._nodes[i] for i in ids if i in self._nodes]

    def search_laws(
        self,
        *,
        domain: Optional[str] = None,
        variable: Optional[str] = None,
        source: Optional[str] = None,
    ) -> List[LawNode]:
        """
        Search law nodes by domain, variable name, or source.

        All provided filters must match (AND semantics).
        """
        results = self.laws()
        if domain:
            results = [n for n in results if n.domain == domain]
        if variable:
            results = [n for n in results if variable in n.variables]
        if source:
            results = [n for n in results if n.source == source]
        return results

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Return a summary of graph size by kind."""
        kinds: Dict[str, int] = {}
        for n in self._nodes.values():
            kinds[n.kind] = kinds.get(n.kind, 0) + 1
        rel_counts: Dict[str, int] = {}
        for e in self._edges:
            key = e.relation.value
            rel_counts[key] = rel_counts.get(key, 0) + 1
        return {
            "total_nodes": len(self._nodes),
            "total_edges": len(self._edges),
            "nodes_by_kind": kinds,
            "edges_by_relation": rel_counts,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [e.to_dict() for e in self._edges],
        }

    def save(self, path: str | Path) -> Path:
        """Persist the graph to a JSON file."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        return out

    @classmethod
    def load(cls, path: str | Path) -> "PhysicsKnowledgeGraph":
        """Load a previously saved graph from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        graph = cls()
        _KIND_MAP = {
            "law": LawNode,
            "dataset": DatasetNode,
            "experiment": ExperimentNode,
            "variable": VariableNode,
            "invariant": InvariantNode,
            "algorithm": AlgorithmNode,
        }
        for nd in data.get("nodes", []):
            kind = nd.get("kind", "law")
            klass = _KIND_MAP.get(kind)
            if klass:
                # Build node from dict, drop 'kind' key since it's a property
                init_data = {k: v for k, v in nd.items() if k != "kind"}
                try:
                    node = klass(**init_data)
                    graph._nodes[node.node_id] = node
                except TypeError:
                    pass  # schema mismatch – skip gracefully
        for ed in data.get("edges", []):
            try:
                edge = Edge(ed["source"], ed["relation"], ed["target"], ed.get("metadata"))
                graph._edges.append(edge)
            except (KeyError, ValueError):
                pass
        return graph

    def to_markdown(self) -> str:
        """Return a Markdown summary of the graph for reports."""
        lines: List[str] = ["# Physics Knowledge Graph\n"]
        s = self.stats()
        lines.append(f"**Nodes:** {s['total_nodes']}  |  **Edges:** {s['total_edges']}\n")

        for kind, count in sorted(s["nodes_by_kind"].items()):
            lines.append(f"- {kind}: {count}")
        lines.append("")

        if self.laws():
            lines.append("## Discovered Laws\n")
            lines.append("| ID | Domain | Equation | Confidence |")
            lines.append("|---|---|---|---|")
            for law in sorted(self.laws(), key=lambda l: l.node_id):
                conf = f"{law.confidence:.2f}" if law.confidence is not None else "—"
                lines.append(f"| {law.node_id} | {law.domain} | `{law.equation}` | {conf} |")
            lines.append("")

        if self._edges:
            lines.append("## Relations\n")
            lines.append("| Source | Relation | Target |")
            lines.append("|---|---|---|")
            for e in self._edges:
                lines.append(f"| {e.source} | {e.relation.value} | {e.target} |")
        return "\n".join(lines)

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"PhysicsKnowledgeGraph("
            f"nodes={s['total_nodes']}, edges={s['total_edges']})"
        )


__all__ = [
    "Edge",
    "PhysicsKnowledgeGraph",
]
