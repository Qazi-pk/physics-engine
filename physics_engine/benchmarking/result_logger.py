"""
Result logging for PIR benchmarking.

Handles saving experiment results in organized directory structures.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from physics_engine.experiment_tracking import ArtifactStore, ExperimentDB
from physics_engine.logging_config import get_logger

logger = get_logger(__name__)


class ResultLogger:
    """
    Manages logging of experiment results.
    
    Creates organized directory structure:
    
    output_dir/
        runs/
            run_0001.json
            run_0002.json
            ...
        summary.json
        config.json
        timestamp.txt
    
    Examples:
        >>> logger = ResultLogger(Path("results_benchmark"))
        >>> logger.save_run(run, result)
        >>> logger.save_summary(summary)
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize result logger.
        
        Args:
            output_dir: Directory for saving results
        """
        self.output_dir = output_dir
        self.runs_dir = output_dir / "runs"
        self.artifact_store = ArtifactStore(output_dir)
        self.experiment_db = ExperimentDB(output_dir / "experiments.db")
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(exist_ok=True)
        
        # Save timestamp
        timestamp_file = self.output_dir / "timestamp.txt"
        timestamp_file.write_text(datetime.now().isoformat())
        
        logger.info(f"Initialized result logger at: {self.output_dir}")
    
    def save_run(self, run, result: Dict[str, Any]) -> Path:
        """
        Save results for a single experiment run.
        
        Args:
            run: ExperimentRun instance
            result: Result dictionary
        
        Returns:
            Path to saved file
        """
        # Save full per-run artifact bundle
        run_dir = self.artifact_store.save_run_artifacts(run, result)
        filepath = run_dir / "result.json"
        
        # Add metadata
        result_with_meta = {
            "run_id": run.run_id,
            "timestamp": datetime.now().isoformat(),
            "experiment_id": result.get("experiment_id"),
            "dataset": run.dataset,
            "algorithm": run.algorithm,
            "noise_level": run.noise_level,
            "dataset_size": run.dataset_size,
            "seed": run.seed,
            "runtime_seconds": result.get("runtime_seconds", result.get("runtime")),
            "success": result.get("status") == "success",
            "status": result.get("status"),
            "from_cache": result.get("from_cache", False),
            "run_config": run.to_dict(),
            "result": result,
        }

        # Overwrite result.json with enriched metadata
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result_with_meta, f, indent=2, default=str)

        metrics = result.get("metrics", {}) if isinstance(result.get("metrics", {}), dict) else {}
        error_value = metrics.get("equation_error", metrics.get("mse", 0.0))
        db_record = {
            "experiment_id": result.get("experiment_id"),
            "run_id": run.run_id,
            "dataset": run.dataset,
            "algorithm": run.algorithm,
            "noise": run.noise_level,
            "dataset_size": run.dataset_size,
            "seed": run.seed,
            "status": result.get("status", "unknown"),
            "success": result.get("status") == "success",
            "error": float(error_value or 0.0),
            "runtime": float(result.get("runtime_seconds", result.get("runtime", 0.0)) or 0.0),
            "from_cache": bool(result.get("from_cache", False)),
            "artifact_dir": str(run_dir),
        }
        self.experiment_db.insert_or_update(db_record)
        
        logger.debug(f"Saved run {run.run_id} to {filepath}")
        return filepath
    
    def save_summary(self, summary: Dict[str, Any]) -> Path:
        """
        Save summary statistics.
        
        Args:
            summary: Summary dictionary
        
        Returns:
            Path to saved file
        """
        filepath = self.output_dir / "summary.json"
        
        # Add metadata
        summary_with_meta = {
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(summary_with_meta, f, indent=2, default=str)

        # Also persist in structured summary directory
        self.artifact_store.save_summary_json(summary_with_meta)
        
        logger.info(f"Saved summary to {filepath}")
        return filepath
    
    def save_config(self, config: Dict[str, Any]) -> Path:
        """
        Save experiment configuration.
        
        Args:
            config: Configuration dictionary
        
        Returns:
            Path to saved file
        """
        filepath = self.output_dir / "config.json"
        
        with open(filepath, "w") as f:
            json.dump(config, f, indent=2, default=str)
        
        logger.info(f"Saved config to {filepath}")
        return filepath
    
    def load_run(self, run_id: int) -> Dict[str, Any]:
        """
        Load results for a specific run.
        
        Args:
            run_id: Run identifier
        
        Returns:
            Result dictionary
        
        Raises:
            FileNotFoundError: If run file doesn't exist
        """
        run_dir = self.runs_dir / f"run_{run_id:05d}"
        filepath = run_dir / "result.json"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Run file not found: {filepath}")
        
        with open(filepath) as f:
            return json.load(f)
    
    def load_all_runs(self) -> list[Dict[str, Any]]:
        """
        Load all experiment runs.
        
        Returns:
            List of result dictionaries
        """
        results = []
        
        for filepath in sorted(self.runs_dir.glob("run_*/result.json")):
            try:
                with open(filepath, encoding="utf-8") as f:
                    results.append(json.load(f))
            except Exception as e:
                logger.warning(f"Failed to load {filepath}: {e}")
        
        return results
    
    def load_summary(self) -> Dict[str, Any]:
        """
        Load summary statistics.
        
        Returns:
            Summary dictionary
        
        Raises:
            FileNotFoundError: If summary file doesn't exist
        """
        filepath = self.output_dir / "summary.json"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Summary file not found: {filepath}")
        
        with open(filepath) as f:
            return json.load(f)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about logged results.
        
        Returns:
            Dictionary with statistics
        """
        run_files = list(self.runs_dir.glob("run_*.json"))
        
        return {
            "output_dir": str(self.output_dir),
            "total_runs": len(list(self.runs_dir.glob("run_*"))),
            "runs_dir_size_mb": sum(f.stat().st_size for f in self.runs_dir.rglob("*") if f.is_file()) / 1024 / 1024,
            "has_summary": (self.output_dir / "summary.json").exists(),
            "has_config": (self.output_dir / "config.json").exists(),
            "has_experiment_db": (self.output_dir / "experiments.db").exists(),
        }


def create_markdown_report(summary: Dict[str, Any], output_path: Path) -> None:
    """
    Create a human-readable markdown report from summary.
    
    Args:
        summary: Summary dictionary
        output_path: Path to save markdown file
    """
    lines = [
        "# PIR Benchmark Report",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        f"- Total experiments: {summary.get('total_experiments', 0)}",
        f"- Successful: {summary.get('successful', 0)}",
        f"- Failed: {summary.get('failed', 0)}",
        f"- Success rate: {summary.get('success_rate', 0):.2%}",
        "",
    ]
    
    # Add configuration
    if "config" in summary:
        config = summary["config"]
        lines.extend([
            "## Configuration",
            "",
            f"- Datasets: {', '.join(config.get('datasets', []))}",
            f"- Algorithms: {', '.join(config.get('algorithms', []))}",
            f"- Noise levels: {config.get('noise_levels', [])}",
            f"- Dataset sizes: {config.get('dataset_sizes', [])}",
            f"- Seeds: {config.get('seeds', [])}",
            "",
        ])
    
    # Add metrics if available
    if "metrics" in summary:
        lines.extend([
            "## Metrics",
            "",
        ])
        for metric_name, stats in summary["metrics"].items():
            lines.append(f"### {metric_name}")
            lines.append("")
            for stat_name, value in stats.items():
                lines.append(f"- {stat_name}: {value}")
            lines.append("")
    
    # Write to file
    output_path.write_text("\n".join(lines))
    logger.info(f"Created markdown report: {output_path}")


__all__ = [
    "ResultLogger",
    "create_markdown_report",
]
