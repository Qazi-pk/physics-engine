"""
metadata_logging.py
~~~~~~~~~~~~~~~~~~~

Comprehensive experiment metadata logging for reproducibility and publication.

Captures:
  - Dataset specifications (path, size, noise level, domain)
  - Algorithm (discovery method, hyperparameters)
  - Discovered law (equation, coefficients, confidence)
  - Performance metrics (MSE, F, validation error)
  - System info (Python, library versions, runtime)
  - Timestamps and reproducibility info

Exports to JSON for archival and paper reproduction.

Usage::

    from physics_engine.active_inference import ExperimentMetadataLogger
    
    logger = ExperimentMetadataLogger()
    
    # Log experiment metadata
    metadata = logger.create_experiment_record(
        experiment_name="2dof_arm_discovery",
        dataset_name="robotics_joint_identification",
        discovery_method="multibody_lagrangian",
        discovered_law={...},
        metrics={"mse": 1e-4, "free_energy": 15.3},
    )
    
    # Save to JSON
    logger.save("results/experiment_metadata.json")
    
    # Load for reproduction
    loaded = ExperimentMetadataLogger.load("results/experiment_metadata.json")
"""

import json
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List, Optional
import platform
import sys

try:
    import numpy as np
except ImportError:
    np = None

try:
    import pandas as pd
except ImportError:
    pd = None


@dataclass
class DatasetMetadata:
    """Metadata about the dataset used."""
    
    name: str
    """Name/identifier of dataset."""
    
    path: Optional[str] = None
    """Path to dataset file (if file-based)."""
    
    num_samples: int = 0
    """Number of data points."""
    
    num_features: int = 0
    """Number of inputs/features."""
    
    num_outputs: int = 0
    """Number of outputs/targets."""
    
    domain: str = "general"
    """Domain (robotics, orbital, pendulum, etc.)."""
    
    noise_level: float = 0.0
    """Estimated or known noise level."""
    
    split_ratio: float = 1.0
    """Fraction used for training (vs validation/test)."""
    
    notes: str = ""
    """Additional notes about the dataset."""


@dataclass
class AlgorithmMetadata:
    """Metadata about the discovery algorithm."""
    
    name: str
    """Algorithm name (e.g., 'multibody_lagrangian', 'symbolic_diso')."""
    
    discovery_mode: str = "standard"
    """Discovery mode (standard, hamiltonian, lagrangian, multibody_lagrangian, etc.)."""
    
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    """Hyperparameters used (population_size, generations, etc.)."""
    
    regularization: float = 1.0
    """Regularization strength."""
    
    basis_functions: Optional[List[str]] = None
    """Basis functions used (for structured discovery)."""
    
    notes: str = ""
    """Additional notes about configuration."""


@dataclass
class DiscoveredLawMetadata:
    """Metadata about the discovered law."""
    
    equation: str
    """String representation of the equation."""
    
    latex_equation: Optional[str] = None
    """LaTeX representation (for papers)."""
    
    variables: List[str] = field(default_factory=list)
    """Variable names in the equation."""
    
    coefficients: Dict[str, float] = field(default_factory=dict)
    """Fitted coefficient values."""
    
    n_parameters: int = 0
    """Number of fitted parameters."""
    
    domain: str = "general"
    """Domain of applicability."""
    
    notes: str = ""
    """Additional notes."""


@dataclass
class PerformanceMetrics:
    """Discovered law performance metrics."""
    
    training_mse: float = 0.0
    """Mean squared error on training data."""
    
    training_rmse: float = 0.0
    """Root mean squared error on training data."""
    
    validation_mse: Optional[float] = None
    """MSE on validation set (if available)."""
    
    free_energy: float = 0.0
    """Free energy (prediction error + complexity)."""
    
    r_squared: float = 0.0
    """R² score (if applicable)."""
    
    confidence: float = 1.0
    """Confidence in the discovered law."""
    
    additional: Dict[str, float] = field(default_factory=dict)
    """Additional metrics (domain-specific)."""


@dataclass
class SystemMetadata:
    """System information for reproducibility."""
    
    python_version: str = ""
    """Python version."""
    
    platform: str = ""
    """OS platform (Linux, Darwin, Windows)."""
    
    numpy_version: Optional[str] = None
    """NumPy version."""
    
    pandas_version: Optional[str] = None
    """Pandas version."""
    
    pir_version: str = "0.1.0"
    """PIR package version."""
    
    modules: Dict[str, str] = field(default_factory=dict)
    """Other installed module versions."""


@dataclass
class ExperimentRecord:
    """Complete experiment record for archival and reproduction."""
    
    experiment_name: str
    """Name/identifier for this experiment."""
    
    timestamp: str
    """ISO format timestamp when experiment was run."""
    
    dataset: DatasetMetadata
    """Dataset used."""
    
    algorithm: AlgorithmMetadata
    """Discovery algorithm used."""
    
    discovered_law: DiscoveredLawMetadata
    """The discovered law."""
    
    metrics: PerformanceMetrics
    """Performance metrics."""
    
    system: SystemMetadata
    """System information."""
    
    runtime_seconds: float = 0.0
    """Total runtime in seconds."""
    
    notes: str = ""
    """Experiment notes (purpose, observations, etc.)."""
    
    seed: Optional[int] = None
    """Random seed (for reproducibility)."""


class ExperimentMetadataLogger:
    """Logger for experiment metadata."""
    
    def __init__(self):
        """Initialize metadata logger."""
        self.records: List[ExperimentRecord] = []
    
    def create_experiment_record(
        self,
        experiment_name: str,
        dataset_name: str,
        discovery_method: str,
        discovered_law: Dict[str, Any],
        metrics: Dict[str, float],
        dataset_metadata: Dict[str, Any] = None,
        algorithm_metadata: Dict[str, Any] = None,
        runtime_seconds: float = 0.0,
        notes: str = "",
        seed: Optional[int] = None,
    ) -> ExperimentRecord:
        """
        Create an experiment record from discovered law and metrics.
        
        Args:
            experiment_name: Identifier for this experiment
            dataset_name: Name of the dataset
            discovery_method: Algorithm name (multibody_lagrangian, etc.)
            discovered_law: Dict with equation, variables, coefficients, etc.
            metrics: Dict with training_mse, validation_mse, free_energy, etc.
            dataset_metadata: Optional augmentation of dataset info
            algorithm_metadata: Optional augmentation of algorithm info
            runtime_seconds: Total runtime
            notes: Experiment notes
            seed: Random seed used
        
        Returns:
            ExperimentRecord object
        """
        # Dataset metadata
        ds_meta = DatasetMetadata(name=dataset_name)
        if dataset_metadata:
            for key, val in dataset_metadata.items():
                if hasattr(ds_meta, key):
                    setattr(ds_meta, key, val)
        
        # Algorithm metadata
        algo_meta = AlgorithmMetadata(
            name=discovery_method,
            discovery_mode=discovery_method,
        )
        if algorithm_metadata:
            for key, val in algorithm_metadata.items():
                if hasattr(algo_meta, key):
                    setattr(algo_meta, key, val)
        
        # Discovered law metadata
        law_meta = DiscoveredLawMetadata(
            equation=discovered_law.get("equation", ""),
            variables=discovered_law.get("variables", []),
            coefficients=discovered_law.get("coefficients", {}),
            n_parameters=len(discovered_law.get("coefficients", {})),
            latex_equation=discovered_law.get("latex_equation"),
        )
        
        # Performance metrics
        perf = PerformanceMetrics(
            training_mse=metrics.get("mse", metrics.get("training_mse", 0.0)),
            training_rmse=metrics.get("rmse", metrics.get("training_rmse", 0.0)),
            validation_mse=metrics.get("validation_mse"),
            free_energy=metrics.get("free_energy", 0.0),
            r_squared=metrics.get("r_squared", 0.0),
            confidence=metrics.get("confidence", 1.0),
        )
        
        # System info
        sys_meta = SystemMetadata(
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            platform=platform.system(),
            pir_version="0.1.0",
        )
        if np:
            sys_meta.numpy_version = np.__version__
        if pd:
            sys_meta.pandas_version = pd.__version__
        
        # Create record
        record = ExperimentRecord(
            experiment_name=experiment_name,
            timestamp=datetime.utcnow().isoformat() + "Z",
            dataset=ds_meta,
            algorithm=algo_meta,
            discovered_law=law_meta,
            metrics=perf,
            system=sys_meta,
            runtime_seconds=runtime_seconds,
            notes=notes,
            seed=seed,
        )
        
        self.records.append(record)
        return record
    
    def add_record(self, record: ExperimentRecord) -> None:
        """Add a pre-constructed record."""
        self.records.append(record)
    
    def save(self, filepath: str) -> None:
        """
        Save experiment records to JSON file.
        
        Args:
            filepath: Path to output JSON file
        """
        data = {
            "records": [asdict(r) for r in self.records],
            "total_experiments": len(self.records),
            "export_timestamp": datetime.utcnow().isoformat() + "Z",
        }
        
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
    
    @classmethod
    def load(cls, filepath: str) -> "ExperimentMetadataLogger":
        """
        Load experiment records from JSON file.
        
        Args:
            filepath: Path to JSON file
        
        Returns:
            ExperimentMetadataLogger with loaded records
        """
        logger = cls()
        
        with open(filepath, "r") as f:
            data = json.load(f)
        
        for record_dict in data.get("records", []):
            # Reconstruct nested dataclasses
            record = ExperimentRecord(
                experiment_name=record_dict["experiment_name"],
                timestamp=record_dict["timestamp"],
                dataset=DatasetMetadata(**record_dict["dataset"]),
                algorithm=AlgorithmMetadata(**record_dict["algorithm"]),
                discovered_law=DiscoveredLawMetadata(**record_dict["discovered_law"]),
                metrics=PerformanceMetrics(**record_dict["metrics"]),
                system=SystemMetadata(**record_dict["system"]),
                runtime_seconds=record_dict.get("runtime_seconds", 0.0),
                notes=record_dict.get("notes", ""),
                seed=record_dict.get("seed"),
            )
            logger.records.append(record)
        
        return logger
    
    def to_dict(self) -> Dict[str, Any]:
        """Export all records as dictionary."""
        return {
            "records": [asdict(r) for r in self.records],
            "total_experiments": len(self.records),
        }
    
    def summary(self) -> str:
        """
        Get human-readable summary of all logged experiments.
        
        Returns:
            Formatted string summary
        """
        if not self.records:
            return "No experiments logged."
        
        lines = [
            f"Experiment Metadata Log ({len(self.records)} experiments)",
            "=" * 70,
        ]
        
        for i, record in enumerate(self.records, 1):
            lines.append(f"\n[{i}] {record.experiment_name}")
            lines.append(f"    Dataset: {record.dataset.name}")
            lines.append(f"    Algorithm: {record.algorithm.name}")
            lines.append(f"    Samples: {record.dataset.num_samples}")
            lines.append(f"    MSE: {record.metrics.training_mse:.6e}")
            lines.append(f"    Free Energy: {record.metrics.free_energy:.4f}")
            lines.append(f"    Runtime: {record.runtime_seconds:.2f}s")
            lines.append(f"    Equation: {record.discovered_law.equation}")
        
        return "\n".join(lines)
