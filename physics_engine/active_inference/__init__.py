"""
physics_engine.active_inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Lightweight active inference layer for autonomous experiment selection.

This module provides:
  - Free energy computation (model fit + complexity trade-off)
  - Belief state (model repository with uncertainties)
  - Experiment selector (choose next experiment via epistemic uncertainty)

No modifications to existing discovery engines. Sits as lightweight decision layer above.

Key classes:
  - BeliefState: Repository of candidate models
  - ExperimentSelector: Choose next experiment
  - FreeEnergyResult: Result of free energy computation

Key functions:
  - compute_free_energy(): Compute F from observed vs predicted data
  - batch_compute_free_energy(): Compute F for multiple models

Example::

    from physics_engine.active_inference import (
        compute_free_energy,
        BeliefState,
        ExperimentSelector,
    )
    
    # Create belief state
    belief = BeliefState()
    
    # After discovery, compute free energy
    f_result = compute_free_energy(observed_tau, predicted_tau, n_params=8)
    
    # Add to belief
    belief.add_model("cycle_1", law_dict, free_energy=f_result.f_value, ...)
    
    # Select next experiment
    selector = ExperimentSelector(strategy="uncertainty_sampling")
    next_exp, score = selector.choose_experiment(belief, ["exp1", "exp2", "exp3"])
"""

from .free_energy import compute_free_energy, batch_compute_free_energy, FreeEnergyResult
from .belief_state import BeliefState, ModelCandidate
from .experiment_selector import ExperimentSelector, SelectionStrategy, ExperimentScore

__all__ = [
    # Free energy
    "compute_free_energy",
    "batch_compute_free_energy",
    "FreeEnergyResult",
    # Belief state
    "BeliefState",
    "ModelCandidate",
    # Experiment selection
    "ExperimentSelector",
    "SelectionStrategy",
    "ExperimentScore",
]

# Metadata logging
from .metadata_logging import (
  ExperimentMetadataLogger,
  ExperimentRecord,
  DatasetMetadata,
  AlgorithmMetadata,
  DiscoveredLawMetadata,
  PerformanceMetrics,
  SystemMetadata,
)

__all__.extend([
  # Metadata logging
  "ExperimentMetadataLogger",
  "ExperimentRecord",
  "DatasetMetadata",
  "AlgorithmMetadata",
  "DiscoveredLawMetadata",
  "PerformanceMetrics",
  "SystemMetadata",
])
