"""
Summary report generation for PIR benchmarking.

Creates comprehensive summaries and comparison tables from experiment results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from physics_engine.benchmarking.diagnostics import (
    analyze_failures,
    format_diagnostics_dict,
    format_diagnostics_md,
    suggest_fixes,
)
from physics_engine.benchmarking.metrics import aggregate_metrics
from physics_engine.benchmarking.result_aggregator import export_paper_tables, summarize_for_paper
from physics_engine.benchmarking.result_logger import create_markdown_report
from physics_engine.logging_config import get_logger

logger = get_logger(__name__)


class SummaryGenerator:
    """
    Generates summary reports from benchmark results.
    
    Creates tables, charts, and markdown reports for analysis.
    
    Examples:
        >>> generator = SummaryGenerator(results)
        >>> summary = generator.generate()
        >>> generator.save_markdown("summary.md")
        >>> comparison = generator.compare_algorithms()
    """
    
    def __init__(self, results: List[Dict[str, Any]]):
        """
        Initialize summary generator.
        
        Args:
            results: List of experiment result dictionaries
        """
        self.results = results
        self._df = None
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert results to pandas DataFrame for analysis.
        
        Returns:
            DataFrame with one row per experiment
        """
        if self._df is not None:
            return self._df
        
        rows = []
        for result in self.results:
            row = {}
            
            # Add run configuration
            if "run" in result:
                row.update(result["run"])
            elif "run_config" in result:
                row.update(result["run_config"])
            
            # Add status
            row["status"] = result.get("status", "unknown")
            
            # Add metrics
            if "metrics" in result:
                for metric_name, value in result["metrics"].items():
                    row[f"metric_{metric_name}"] = value
            
            rows.append(row)
        
        self._df = pd.DataFrame(rows)
        return self._df
    
    def generate(self) -> Dict[str, Any]:
        """
        Generate comprehensive summary.
        
        Returns:
            Dictionary with summary statistics and tables
        """
        df = self.to_dataframe()
        
        summary = {
            "total_experiments": len(self.results),
            "successful": int((df["status"] == "success").sum()),
            "failed": int((df["status"] == "failed").sum()),
            "success_rate": float((df["status"] == "success").mean()),
        }
        
        # Aggregate metrics
        if "metrics" in self.results[0]:
            summary["metrics"] = aggregate_metrics(self.results)
        
        # Group by algorithm
        if "algorithm" in df.columns:
            summary["by_algorithm"] = self._summarize_by_algorithm(df)
        
        # Group by dataset
        if "dataset" in df.columns:
            summary["by_dataset"] = self._summarize_by_dataset(df)
        
        # Group by noise level
        if "noise_level" in df.columns:
            summary["by_noise"] = self._summarize_by_noise(df)

        # Automatic diagnostics and recommendations
        summary["diagnostics"] = format_diagnostics_dict(self.results)

        return summary
    
    def _summarize_by_algorithm(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Summarize results grouped by algorithm."""
        summary = {}
        
        for algorithm in df["algorithm"].unique():
            algo_df = df[df["algorithm"] == algorithm]
            
            summary[algorithm] = {
                "total": len(algo_df),
                "successful": int((algo_df["status"] == "success").sum()),
                "success_rate": float((algo_df["status"] == "success").mean()),
            }
            
            # Add metric means
            metric_cols = [c for c in algo_df.columns if c.startswith("metric_")]
            for col in metric_cols:
                metric_name = col.replace("metric_", "")
                summary[algorithm][f"mean_{metric_name}"] = float(algo_df[col].mean())
        
        return summary
    
    def _summarize_by_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Summarize results grouped by dataset."""
        summary = {}
        
        for dataset in df["dataset"].unique():
            ds_df = df[df["dataset"] == dataset]
            
            summary[dataset] = {
                "total": len(ds_df),
                "successful": int((ds_df["status"] == "success").sum()),
                "success_rate": float((ds_df["status"] == "success").mean()),
            }
        
        return summary
    
    def _summarize_by_noise(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Summarize results grouped by noise level."""
        summary = {}
        
        for noise in sorted(df["noise_level"].unique()):
            noise_df = df[df["noise_level"] == noise]
            
            summary[float(noise)] = {
                "total": len(noise_df),
                "successful": int((noise_df["status"] == "success").sum()),
                "success_rate": float((noise_df["status"] == "success").mean()),
            }
        
        return summary
    
    def compare_algorithms(self) -> pd.DataFrame:
        """
        Create algorithm comparison table.
        
        Returns:
            DataFrame comparing algorithms across metrics
        """
        df = self.to_dataframe()
        
        if "algorithm" not in df.columns:
            logger.warning("No algorithm column found")
            return pd.DataFrame()
        
        # Pivot table with algorithms as rows, metrics as columns
        metric_cols = [c for c in df.columns if c.startswith("metric_")]
        
        comparison = df.groupby("algorithm")[metric_cols].agg(["mean", "std", "count"])
        
        # Clean up column names
        comparison.columns = ["_".join(col).replace("metric_", "") for col in comparison.columns]
        
        return comparison
    
    def diagnose(self) -> Dict[str, Any]:
        """
        Run the diagnostics engine and return a structured report.

        Returns:
            Dict with keys ``diagnostics`` (list of issue strings) and
            ``recommendations`` (list of fix strings).
        """
        return format_diagnostics_dict(self.results)

    def save_markdown(self, output_path: Path | str) -> None:
        """
        Save summary as markdown file, including automatic diagnostics
        and recommended actions when the success rate is below 100 %.

        Args:
            output_path: Path to save markdown file
        """
        output_path = Path(output_path)
        summary = self.generate()
        create_markdown_report(summary, output_path)
        # Append diagnostics block when experiments had failures
        n_total   = summary.get("total_experiments", len(self.results))
        n_success = summary.get("successful", 0)
        rate      = summary.get("success_rate", 0.0)
        if rate < 1.0 and self.results:
            diag_md = format_diagnostics_md(
                self.results,
                success_rate=rate,
                n_success=n_success,
                n_total=n_total,
            )
            with open(output_path, "a", encoding="utf-8") as fh:
                fh.write("\n" + diag_md)
            logger.info("Appended diagnostics to %s", output_path)
    
    def save_csv(self, output_path: Path | str) -> None:
        """
        Save results as CSV file.
        
        Args:
            output_path: Path to save CSV file
        """
        output_path = Path(output_path)
        df = self.to_dataframe()
        df.to_csv(output_path, index=False)
        logger.info(f"Saved results to CSV: {output_path}")
    
    def save_comparison_table(self, output_path: Path | str) -> None:
        """
        Save algorithm comparison table as CSV.
        
        Args:
            output_path: Path to save CSV file
        """
        output_path = Path(output_path)
        comparison = self.compare_algorithms()
        comparison.to_csv(output_path)
        logger.info(f"Saved comparison table: {output_path}")

    def paper_table(self) -> pd.DataFrame:
        """
        Create paper-ready success-rate table.

        Columns: Algorithm | Dataset | Noise | SuccessRate
        """
        df = self.to_dataframe()
        return summarize_for_paper(df)

    def save_paper_tables(self, output_dir: Path | str) -> Dict[str, Path]:
        """
        Export paper-ready tables to CSV/Markdown/LaTeX.

        Files:
        - summary.csv
        - summary.md
        - summary.tex
        """
        df = self.to_dataframe()
        paths = export_paper_tables(df, output_dir)
        logger.info(
            "Saved paper tables: %s",
            {name: str(path) for name, path in paths.items()},
        )
        return paths
    
    def print_summary(self) -> None:
        """Print summary to console."""
        summary = self.generate()
        
        print("\n" + "=" * 70)
        print("PIR Benchmark Summary")
        print("=" * 70)
        print(f"\nTotal experiments: {summary['total_experiments']}")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")
        print(f"Success rate: {summary['success_rate']:.2%}")

        # Diagnostics
        diag = summary.get("diagnostics", {})
        issues = diag.get("diagnostics", [])
        recs   = diag.get("recommendations", [])
        if issues:
            print("\n" + "-" * 70)
            print("Diagnostics")
            print("-" * 70)
            for msg in issues:
                print(f"  • {msg}")
        if recs:
            print("\n" + "-" * 70)
            print("Recommended Actions")
            print("-" * 70)
            for rec in recs:
                print(f"  → {rec}")
        
        # Print algorithm comparison
        if "by_algorithm" in summary:
            print("\n" + "-" * 70)
            print("Algorithm Comparison")
            print("-" * 70)
            for algo, stats in summary["by_algorithm"].items():
                print(f"\n{algo}:")
                for key, value in stats.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.4f}")
                    else:
                        print(f"  {key}: {value}")
        
        print("\n" + "=" * 70)


def generate_comparison_table(
    results: List[Dict[str, Any]],
    group_by: str = "algorithm",
    metrics: List[str] = None,
) -> pd.DataFrame:
    """
    Generate comparison table grouped by specified column.
    
    Args:
        results: List of result dictionaries
        group_by: Column to group by ('algorithm', 'dataset', etc.)
        metrics: List of metrics to include (None = all)
    
    Returns:
        DataFrame with comparison statistics
    """
    generator = SummaryGenerator(results)
    df = generator.to_dataframe()
    
    if group_by not in df.columns:
        logger.warning(f"Column '{group_by}' not found in results")
        return pd.DataFrame()
    
    # Select metric columns
    metric_cols = [c for c in df.columns if c.startswith("metric_")]
    if metrics:
        metric_cols = [c for c in metric_cols if any(m in c for m in metrics)]
    
    # Create comparison table
    comparison = df.groupby(group_by)[metric_cols].agg(["mean", "std", "count"])
    
    return comparison


__all__ = [
    "SummaryGenerator",
    "generate_comparison_table",
]
