"""
Experiment configuration management for PIR benchmarking.

Handles loading and validation of experiment configurations from YAML files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ModuleNotFoundError:  # optional until YAML-backed config loading is used
    yaml = None


def _require_yaml() -> None:
    if yaml is None:
        raise ModuleNotFoundError(
            "PyYAML is required for benchmark YAML configuration loading/saving. "
            "Install it with 'pip install pyyaml'."
        )


@dataclass
class ExperimentConfig:
    """
    Configuration for a benchmark experiment sweep.
    
    Defines the parameter space for experiments including datasets,
    algorithms, noise levels, dataset sizes, and random seeds.
    
    Attributes:
        datasets: List of dataset names to test
        algorithms: List of algorithm/pipeline names to test
        noise_levels: List of noise levels (stddev)
        dataset_sizes: List of dataset sizes (number of samples)
        seeds: List of random seeds for reproducibility
        output_dir: Directory for results
        parallel: Enable parallel execution
        max_workers: Maximum parallel workers (None = auto)
        timeout: Timeout per experiment in seconds (None = no timeout)
        metrics: List of metrics to compute
        metadata: Additional experiment metadata
    """
    
    name: str = "benchmark"
    datasets: List[str] = field(default_factory=list)
    algorithms: List[str] = field(default_factory=list)
    noise_levels: List[float] = field(default_factory=lambda: [0.0, 0.01, 0.05])
    dataset_sizes: List[int] = field(default_factory=lambda: [200, 500])
    seeds: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    
    output_dir: str = "results_benchmark"
    parallel: bool = True
    max_workers: Optional[int] = None
    timeout: Optional[float] = None
    cache_enabled: bool = True
    cache_dir: str = "results/cache"
    force_rerun: bool = False
    
    metrics: List[str] = field(default_factory=lambda: ["equation_error", "runtime"])
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def total_experiments(self) -> int:
        """Calculate total number of experiments in the grid."""
        return (
            len(self.datasets) *
            len(self.algorithms) *
            len(self.noise_levels) *
            len(self.dataset_sizes) *
            len(self.seeds)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary."""
        return {
            "name": self.name,
            "datasets": self.datasets,
            "algorithms": self.algorithms,
            "noise_levels": self.noise_levels,
            "dataset_sizes": self.dataset_sizes,
            "seeds": self.seeds,
            "output_dir": self.output_dir,
            "parallel": self.parallel,
            "max_workers": self.max_workers,
            "timeout": self.timeout,
            "cache_enabled": self.cache_enabled,
            "cache_dir": self.cache_dir,
            "force_rerun": self.force_rerun,
            "metrics": self.metrics,
            "metadata": self.metadata,
            "total_experiments": self.total_experiments(),
        }
    
    def save(self, path: Path) -> None:
        """Save configuration to YAML file."""
        _require_yaml()
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def validate(self) -> List[str]:
        """
        Validate configuration.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        if not self.datasets:
            errors.append("No datasets specified")
        
        if not self.algorithms:
            errors.append("No algorithms specified")
        
        if any(n < 0 for n in self.noise_levels):
            errors.append("Noise levels must be non-negative")
        
        if any(s <= 0 for s in self.dataset_sizes):
            errors.append("Dataset sizes must be positive")
        
        if not self.seeds:
            errors.append("No seeds specified")
        
        if self.max_workers is not None and self.max_workers <= 0:
            errors.append("max_workers must be positive")
        
        if self.timeout is not None and self.timeout <= 0:
            errors.append("timeout must be positive")

        if not self.cache_dir:
            errors.append("cache_dir must be non-empty")
        
        return errors

    @classmethod
    def from_yaml(cls, path: Path | str) -> "ExperimentConfig":
        """Load configuration from YAML file."""
        return load_config(path)


def load_config(path: Path | str) -> ExperimentConfig:
    """
    Load experiment configuration from YAML file.
    
    Args:
        path: Path to YAML configuration file
    
    Returns:
        ExperimentConfig instance
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    
    Examples:
        >>> config = load_config("experiments/benchmark_configs/robotics.yaml")
        >>> print(f"Total experiments: {config.total_experiments()}")
        Total experiments: 900
    """
    path = Path(path)
    _require_yaml()
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path) as f:
        data = yaml.safe_load(f)
    
    if not data:
        raise ValueError(f"Empty or invalid config file: {path}")
    
    # Create config from YAML data
    config = ExperimentConfig(
        name=data.get("name", "benchmark"),
        datasets=data.get("datasets", []),
        algorithms=data.get("algorithms", []),
        noise_levels=data.get("noise_levels", [0.0, 0.01, 0.05]),
        dataset_sizes=data.get("dataset_sizes", [200, 500]),
        seeds=data.get("seeds", [1, 2, 3, 4, 5]),
        output_dir=data.get("output_dir", "results_benchmark"),
        parallel=data.get("parallel", True),
        max_workers=data.get("max_workers"),
        timeout=data.get("timeout"),
        cache_enabled=data.get("cache_enabled", True),
        cache_dir=data.get("cache_dir", "results/cache"),
        force_rerun=data.get("force_rerun", False),
        metrics=data.get("metrics", ["equation_error", "runtime"]),
        metadata=data.get("metadata", {}),
    )
    
    # Validate
    errors = config.validate()
    if errors:
        error_msg = "\n".join(f"  - {e}" for e in errors)
        raise ValueError(f"Invalid configuration:\n{error_msg}")
    
    return config


def create_default_config(
    name: str = "default",
    output_dir: str = "results_benchmark"
) -> ExperimentConfig:
    """
    Create a default experiment configuration.
    
    Args:
        name: Configuration name (for metadata)
        output_dir: Output directory for results
    
    Returns:
        Default ExperimentConfig instance
    """
    return ExperimentConfig(
        name=name,
        datasets=["pendulum", "robot_joint"],
        algorithms=["system_identification", "robot_structure_discovery"],
        noise_levels=[0.0, 0.01, 0.05],
        dataset_sizes=[200, 500],
        seeds=[1, 2, 3, 4, 5],
        output_dir=output_dir,
        parallel=True,
        metrics=["equation_error", "parameter_error", "runtime"],
        metadata={"name": name, "description": "Default benchmark configuration"},
    )


__all__ = [
    "ExperimentConfig",
    "load_config",
    "create_default_config",
]
