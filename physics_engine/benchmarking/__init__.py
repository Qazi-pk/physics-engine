"""
Benchmarking module for PIR Engine.

Provides tools for running large-scale experiment grids including:
- Experiment configuration management
- Automated experiment execution
- Result logging and metrics
- Summary report generation
- Parallel execution support

This module enables scaling from single experiments to thousands of runs
for research validation and industrial benchmarking.
"""

from .experiment_config import ExperimentConfig, load_config
from .experiment_hash import compute_experiment_hash, normalize_experiment_config
from .experiment_runner import ExperimentRunner, run_single_experiment, set_seed
from .metrics import (
    compute_metrics,
    equation_error,
    parameter_error,
    runtime_metric,
)
from .cache_manager import CacheManager
from .diagnostics import (
    analyze_failures,
    suggest_fixes,
    format_diagnostics_md,
    format_diagnostics_dict,
)
from .result_aggregator import aggregate_results, export_paper_tables, summarize_for_paper
from .result_logger import ResultLogger
from .summary_generator import SummaryGenerator
from physics_engine.experiment_tracking import ArtifactStore, ExperimentDB, build_experiment_queue

__all__ = [
    "ExperimentConfig",
    "load_config",
    "compute_experiment_hash",
    "normalize_experiment_config",
    "ExperimentRunner",
    "run_single_experiment",
    "set_seed",
    "compute_metrics",
    "equation_error",
    "parameter_error",
    "runtime_metric",
    "CacheManager",
    "analyze_failures",
    "suggest_fixes",
    "format_diagnostics_md",
    "format_diagnostics_dict",
    "aggregate_results",
    "summarize_for_paper",
    "export_paper_tables",
    "ResultLogger",
    "SummaryGenerator",
    "ExperimentDB",
    "ArtifactStore",
    "build_experiment_queue",
]
