# physics_engine package
"""
Physics Intermediate Representation (PIR) Engine

A comprehensive framework for automated physics law discovery from data.
"""

__version__ = "0.1.0"

# Logging configuration
from .logging_config import setup_logging, get_logger, logger

# Data utilities
from .data_loader import load_example, list_datasets, AVAILABLE_DATASETS

# Core components
from .core import Dataset, PhysicsModel

# Simulation
from .simulation import simulate

# Pipelines
from .pipelines import (
    run_system_identification,
    RobotJointIdentifier,
    RobotJointIdentificationResult,
    run_robot_joint_identification,
    RobotStructureDiscovery,
)

# Knowledge graph
from .knowledge_graph import PhysicsKnowledgeGraph, RelationType

# Variational mechanics
from .variational import (
    MultiBodyLagrangianResult,
    StructuredLagrangianResult,
    discover_multibody_lagrangian,
    discover_multibody_lagrangian_from_csv,
    discover_multibody_lagrangian_from_dataframe,
    discover_structured_lagrangian,
    discover_structured_lagrangian_from_csv,
    discover_structured_lagrangian_from_dataframe,
    lagrangian_features,
    multibody_features,
)

# Module-level exports for advanced users
from . import core, discovery, knowledge_graph, pipelines, plugins, simulation, validation, variational, active_inference

# Active inference
from .active_inference import (
    compute_free_energy,
    batch_compute_free_energy,
    FreeEnergyResult,
    BeliefState,
    ModelCandidate,
    ExperimentSelector,
    SelectionStrategy,
    ExperimentScore,
    ExperimentMetadataLogger,
    ExperimentRecord,
    DatasetMetadata,
    AlgorithmMetadata,
    DiscoveredLawMetadata,
    PerformanceMetrics,
    SystemMetadata,
)

__all__ = [
    # Version
    "__version__",
    # Logging
    "setup_logging",
    "get_logger",
    "logger",
    # Data utilities
    "load_example",
    "list_datasets",
    "AVAILABLE_DATASETS",
    # Core classes
    "Dataset",
    "PhysicsModel",
    # Simulation
    "simulate",
    # Pipelines
    "run_system_identification",
    "RobotJointIdentifier",
    "RobotJointIdentificationResult",
    "run_robot_joint_identification",
    "RobotStructureDiscovery",
    # Knowledge graph
    "PhysicsKnowledgeGraph",
    "RelationType",
    # Variational mechanics
    "StructuredLagrangianResult",
    "MultiBodyLagrangianResult",
    "discover_structured_lagrangian",
    "discover_structured_lagrangian_from_dataframe",
    "discover_structured_lagrangian_from_csv",
    "discover_multibody_lagrangian",
    "discover_multibody_lagrangian_from_dataframe",
    "discover_multibody_lagrangian_from_csv",
    "lagrangian_features",
    "multibody_features",
    # Active inference
    "compute_free_energy",
    "batch_compute_free_energy",
    "FreeEnergyResult",
    "BeliefState",
    "ModelCandidate",
    "ExperimentSelector",
    "SelectionStrategy",
    "ExperimentScore",
    # Metadata logging
    "ExperimentMetadataLogger",
    "ExperimentRecord",
    "DatasetMetadata",
    "AlgorithmMetadata",
    "DiscoveredLawMetadata",
    "PerformanceMetrics",
    "SystemMetadata",
    # Modules (for advanced users)
    "core",
    "discovery",
    "knowledge_graph",
    "pipelines",
    "plugins",
    "simulation",
    "validation",
    "variational",
    "active_inference",
]
