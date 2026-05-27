"""
Relation type definitions for the Physics Knowledge Graph.

Edges between nodes carry a typed relation that describes how two
physics objects (laws, datasets, experiments, …) are connected.
"""

from __future__ import annotations

from enum import Enum


class RelationType(str, Enum):
    """All first-class relation types understood by PIR."""

    # Discovery provenance
    DERIVED_FROM = "derived_from"           # law derived from another law
    DISCOVERED_BY = "discovered_by"         # law discovered by an algorithm
    VALIDATED_ON = "validated_on"           # law validated on a dataset
    DISCOVERED_FROM = "discovered_from"     # law found in an experiment

    # Physics structure
    CONSISTENT_WITH = "consistent_with"     # laws are mutually consistent
    CONTRADICTS = "contradicts"             # laws are in tension
    SPECIAL_CASE_OF = "special_case_of"     # one is a limiting case of another
    GENERALIZES = "generalizes"             # one generalizes another
    CONSERVES = "conserves"                 # law implies a conservation law
    RELATED_TO = "related_to"               # loose semantic relationship

    # Dataset / experiment links
    GENERATED_FROM = "generated_from"       # dataset generated from a law
    TESTED_WITH = "tested_with"             # algorithm tested on dataset


# Human-readable labels for reports
RELATION_LABELS: dict[RelationType, str] = {
    RelationType.DERIVED_FROM:    "derived from",
    RelationType.DISCOVERED_BY:   "discovered by",
    RelationType.VALIDATED_ON:    "validated on",
    RelationType.DISCOVERED_FROM: "discovered from",
    RelationType.CONSISTENT_WITH: "consistent with",
    RelationType.CONTRADICTS:     "contradicts",
    RelationType.SPECIAL_CASE_OF: "special case of",
    RelationType.GENERALIZES:     "generalizes",
    RelationType.CONSERVES:       "conserves",
    RelationType.RELATED_TO:      "related to",
    RelationType.GENERATED_FROM:  "generated from",
    RelationType.TESTED_WITH:     "tested with",
}


def describe_relation(rel: RelationType | str) -> str:
    """Return a human-readable string for *rel*."""
    try:
        return RELATION_LABELS[RelationType(rel)]
    except (KeyError, ValueError):
        return str(rel)


__all__ = [
    "RelationType",
    "RELATION_LABELS",
    "describe_relation",
]
