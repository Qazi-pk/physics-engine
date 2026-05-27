"""
Experiment runner for PIR benchmarking.

Orchestrates the execution of large-scale experiment grids with support
for parallel execution and result logging.
"""

from __future__ import annotations

import itertools
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional

import numpy as np

from physics_engine.benchmarking.cache_manager import CacheManager
from physics_engine.benchmarking.experiment_config import ExperimentConfig
from physics_engine.benchmarking.experiment_hash import compute_experiment_hash
from physics_engine.benchmarking.metrics import compute_metrics
from physics_engine.benchmarking.result_logger import ResultLogger
from physics_engine.experiment_tracking import ExperimentDB, build_experiment_queue, drain_queue
from physics_engine.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ExperimentRun:
    """
    Single experiment run configuration.
    
    Represents one point in the experiment grid.
    """
    
    dataset: str
    algorithm: str
    noise_level: float
    dataset_size: int
    seed: int
    run_id: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary."""
        return {
            "dataset": self.dataset,
            "algorithm": self.algorithm,
            "noise_level": self.noise_level,
            "dataset_size": self.dataset_size,
            "seed": self.seed,
            "run_id": self.run_id,
        }
    
    def __str__(self) -> str:
        """Human-readable representation."""
        return (
            f"Run {self.run_id}: {self.algorithm} on {self.dataset} "
            f"(size={self.dataset_size}, noise={self.noise_level}, seed={self.seed})"
        )


class ExperimentRunner:
    """
    Orchestrates execution of benchmark experiment grids.
    
    Generates all combinations of experiment parameters, executes them
    (optionally in parallel), and logs results.
    
    Examples:
        >>> from physics_engine.benchmarking import ExperimentRunner, load_config
        >>> 
        >>> config = load_config("experiments/benchmark_configs/robotics.yaml")
        >>> runner = ExperimentRunner(config)
        >>> 
        >>> # Run all experiments
        >>> results = runner.run()
        >>> 
        >>> # Generate summary
        >>> summary = runner.generate_summary()
    """
    
    def __init__(
        self,
        config: ExperimentConfig,
        algorithm_registry: Optional[Dict[str, Callable]] = None,
    ):
        """
        Initialize experiment runner.
        
        Args:
            config: Experiment configuration
            algorithm_registry: Dictionary mapping algorithm names to functions
        """
        self.config = config
        self.algorithm_registry = algorithm_registry or {}
        self.result_logger = ResultLogger(Path(config.output_dir))
        self.experiment_db = ExperimentDB(Path(config.output_dir) / "experiments.db")
        self.cache_manager = CacheManager(config.cache_dir)
        self._runs: List[ExperimentRun] = []
        self._results: List[Dict[str, Any]] = []
    
    def generate_runs(self) -> Generator[ExperimentRun, None, None]:
        """
        Generate all experiment runs from config.
        
        Yields:
            ExperimentRun instances for each parameter combination
        """
        # Create parameter grid
        param_grid = itertools.product(
            self.config.datasets,
            self.config.algorithms,
            self.config.noise_levels,
            self.config.dataset_sizes,
            self.config.seeds,
        )
        
        # Generate runs
        for run_id, (dataset, algorithm, noise, size, seed) in enumerate(param_grid, start=1):
            yield ExperimentRun(
                dataset=dataset,
                algorithm=algorithm,
                noise_level=noise,
                dataset_size=size,
                seed=seed,
                run_id=run_id,
            )
    
    def run(self, progress_callback: Optional[Callable[[int, int], None]] = None) -> List[Dict[str, Any]]:
        """
        Execute all experiments in the grid.
        
        Args:
            progress_callback: Optional callback(completed, total) for progress updates
        
        Returns:
            List of result dictionaries
        """
        # Generate all runs and push through explicit queue abstraction
        runs = list(self.generate_runs())
        queue = build_experiment_queue(runs)
        queued_runs = drain_queue(queue)
        total = len(runs)
        
        logger.info(f"Starting benchmark with {total} experiments")
        logger.info(f"Configuration: {self.config.to_dict()}")
        self.result_logger.save_config(self.config.to_dict())
        
        # Skip duplicate experiments based on deterministic hash index
        duplicate_results: List[Dict[str, Any]] = []
        pending_runs: List[ExperimentRun] = []
        for run in queued_runs:
            experiment_id = _experiment_id_for_run(run)
            if not self.config.force_rerun and self.experiment_db.has_experiment(experiment_id):
                skipped_result = {
                    "run": run.to_dict(),
                    "experiment_id": experiment_id,
                    "status": "skipped_duplicate",
                    "error": None,
                    "runtime_seconds": 0.0,
                    "from_cache": True,
                }
                duplicate_results.append(skipped_result)
                self.result_logger.save_run(run, skipped_result)
            else:
                pending_runs.append(run)

        # Execute pending experiments
        if self.config.parallel:
            results = self._run_parallel(pending_runs, progress_callback)
        else:
            results = self._run_sequential(pending_runs, progress_callback)
        
        self._results = duplicate_results + results
        logger.info(f"Completed {len(results)} experiments")
        
        return self._results
    
    def _run_sequential(
        self,
        runs: List[ExperimentRun],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute experiments sequentially."""
        results = []
        total = len(runs)
        
        for i, run in enumerate(runs, start=1):
            try:
                logger.info(f"[{i}/{total}] {run}")
                result = run_single_experiment(
                    run=run,
                    algorithm_registry=self.algorithm_registry,
                    timeout=self.config.timeout,
                    cache_enabled=self.config.cache_enabled,
                    force_rerun=self.config.force_rerun,
                    cache_dir=self.config.cache_dir,
                )
                results.append(result)
                
                # Log result
                self.result_logger.save_run(run, result)
                
                if progress_callback:
                    progress_callback(i, total)
                    
            except Exception as e:
                logger.error(f"Experiment {run.run_id} failed: {e}")
                results.append({
                    "run": run.to_dict(),
                    "experiment_id": _experiment_id_for_run(run),
                    "status": "failed",
                    "error": str(e),
                    "from_cache": False,
                })
        
        return results
    
    def _run_parallel(
        self,
        runs: List[ExperimentRun],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute experiments in parallel."""
        results = []
        total = len(runs)
        completed = 0
        
        max_workers = self.config.max_workers
        logger.info(f"Running experiments in parallel (max_workers={max_workers})")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_run = {
                executor.submit(
                    run_single_experiment,
                    run,
                    self.algorithm_registry,
                    self.config.timeout,
                    self.config.cache_enabled,
                    self.config.force_rerun,
                    self.config.cache_dir,
                ): run
                for run in runs
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_run):
                run = future_to_run[future]
                completed += 1
                
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Log result
                    self.result_logger.save_run(run, result)
                    
                    logger.info(f"[{completed}/{total}] Completed {run}")
                    
                    if progress_callback:
                        progress_callback(completed, total)
                        
                except Exception as e:
                    logger.error(f"Experiment {run.run_id} failed: {e}")
                    results.append({
                        "run": run.to_dict(),
                        "experiment_id": _experiment_id_for_run(run),
                        "status": "failed",
                        "error": str(e),
                        "from_cache": False,
                    })
        
        return results
    
    def generate_summary(self) -> Dict[str, Any]:
        """
        Generate summary statistics from results.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self._results:
            logger.warning("No results available for summary")
            return {}
        
        # Count successful/failed
        successful = sum(1 for r in self._results if r.get("status") != "failed")
        failed = len(self._results) - successful
        cache_hits = sum(1 for r in self._results if r.get("from_cache", False))
        
        # Aggregate metrics
        summary = {
            "total_experiments": len(self._results),
            "successful": successful,
            "failed": failed,
            "skipped_duplicate": sum(1 for r in self._results if r.get("status") == "skipped_duplicate"),
            "success_rate": successful / len(self._results) if self._results else 0,
            "cache_hits": cache_hits,
            "cache_hit_rate": cache_hits / len(self._results) if self._results else 0,
            "config": self.config.to_dict(),
            "cache": self.cache_manager.stats(),
            "experiment_index": self.experiment_db.stats(),
        }
        
        # Save summary
        self.result_logger.save_summary(summary)
        
        return summary


def run_single_experiment(
    run: ExperimentRun,
    algorithm_registry: Optional[Dict[str, Callable]] = None,
    timeout: Optional[float] = None,
    cache_enabled: bool = True,
    force_rerun: bool = False,
    cache_dir: str = "results/cache",
) -> Dict[str, Any]:
    """
    Execute a single experiment.
    
    Args:
        run: Experiment run configuration
        algorithm_registry: Dictionary of available algorithms
        timeout: Timeout in seconds
    
    Returns:
        Dictionary with experiment results and metrics
    """
    algorithm_registry = algorithm_registry or {}
    experiment_id = _experiment_id_for_run(run)
    cache = CacheManager(cache_dir)

    if cache_enabled and not force_rerun and cache.exists(experiment_id):
        cached_result = cache.load(experiment_id)
        cached_result["from_cache"] = True
        cached_result.setdefault("experiment_id", experiment_id)
        logger.info(f"Loading cached result: {experiment_id}")
        return cached_result
    
    # Set random seed for reproducibility
    set_seed(run.seed)
    
    # Start timer
    start_time = time.time()
    
    try:
        # Get algorithm
        if run.algorithm not in algorithm_registry:
            # Try to import standard PIR algorithms
            algorithm_func = _get_default_algorithm(run.algorithm)
        else:
            algorithm_func = algorithm_registry[run.algorithm]
        
        # Load/generate dataset
        dataset = _load_dataset(run.dataset, run.dataset_size, run.noise_level, run.seed)
        
        # Run algorithm
        if timeout:
            # TODO: Implement timeout mechanism
            result = algorithm_func(dataset)
        else:
            result = algorithm_func(dataset)
        
        # Compute metrics
        runtime = time.time() - start_time
        metrics = compute_metrics(result, ground_truth=None)  # TODO: Add ground truth
        metrics["runtime"] = runtime

        result_payload = {
            "experiment_id": experiment_id,
            "dataset": run.dataset,
            "algorithm": run.algorithm,
            "noise_level": run.noise_level,
            "dataset_size": run.dataset_size,
            "seed": run.seed,
            "runtime_seconds": runtime,
            "run": run.to_dict(),
            "status": "success",
            "result": result if hasattr(result, "to_dict") else str(result),
            "metrics": metrics,
            "from_cache": False,
        }

        if cache_enabled:
            cache.save(experiment_id, result_payload)

        return result_payload
        
    except Exception as e:
        runtime = time.time() - start_time
        logger.error(f"Experiment failed: {e}")
        failure_payload = {
            "experiment_id": experiment_id,
            "dataset": run.dataset,
            "algorithm": run.algorithm,
            "noise_level": run.noise_level,
            "dataset_size": run.dataset_size,
            "seed": run.seed,
            "runtime_seconds": runtime,
            "run": run.to_dict(),
            "status": "failed",
            "error": str(e),
            "runtime": runtime,
            "from_cache": False,
        }

        if cache_enabled:
            cache.save(experiment_id, failure_payload)

        return failure_payload


def set_seed(seed: int) -> None:
    """Set deterministic random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def _experiment_id_for_run(run: ExperimentRun) -> str:
    """Compute deterministic experiment id from run config."""
    return compute_experiment_hash(
        {
            "dataset": run.dataset,
            "algorithm": run.algorithm,
            "noise_level": run.noise_level,
            "dataset_size": run.dataset_size,
            "seed": run.seed,
        }
    )


def _get_default_algorithm(name: str) -> Callable:
    """Get default PIR algorithm by name."""
    from physics_engine.pipelines import (
        run_system_identification,
        RobotStructureDiscovery,
    )
    
    algorithms = {
        "system_identification": run_system_identification,
        "robot_structure_discovery": lambda ds: RobotStructureDiscovery().run(ds),
    }
    
    if name not in algorithms:
        raise ValueError(f"Unknown algorithm: {name}")
    
    return algorithms[name]


def _load_dataset(name: str, size: int, noise: float, seed: int):
    """Load or generate dataset."""
    from physics_engine import load_example
    from physics_engine.core import Dataset
    import pandas as pd
    
    # Try to load example dataset
    try:
        df = load_example(name)
        
        # Subsample if needed
        if len(df) > size:
            np.random.seed(seed)
            df = df.sample(n=size, random_state=seed)
        
        # Add noise if specified
        if noise > 0:
            for col in df.select_dtypes(include=[np.number]).columns:
                if col != "time":
                    df[col] = df[col] + np.random.normal(0, noise, len(df))
        
        # Convert pandas DataFrame to Dataset format
        variables = [col for col in df.columns if col != "time"]
        data = {col: np.asarray(df[col].values) for col in variables}
        time = np.asarray(df["time"].values) if "time" in df.columns else None
        return Dataset(variables=variables, data=data, time=time)
        
    except Exception as e:
        logger.warning(f"Could not load dataset '{name}': {e}")
        # Return dummy dataset
        np.random.seed(seed)
        time_arr = np.linspace(0, 10, size)
        x_arr = np.random.randn(size)
        return Dataset(
            variables=["x"],
            data={"x": x_arr},
            time=time_arr
        )


__all__ = [
    "ExperimentRun",
    "ExperimentRunner",
    "run_single_experiment",
    "set_seed",
]
