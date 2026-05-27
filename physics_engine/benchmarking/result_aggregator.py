"""
Result aggregation utilities for cached benchmark artifacts.

Provides programmatic and export helpers for paper-ready summaries.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def aggregate_results(cache_dir: Path | str) -> pd.DataFrame:
    """
    Aggregate cached JSON results into a DataFrame.

    Args:
        cache_dir: Directory containing cached experiment JSON files

    Returns:
        DataFrame of flattened experiment rows
    """
    rows: List[Dict[str, Any]] = []
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        return pd.DataFrame()

    for file in sorted(cache_path.glob("*.json")):
        with open(file, "r", encoding="utf-8") as f:
            payload = json.load(f)

        row: Dict[str, Any] = {
            "experiment_id": payload.get("experiment_id", file.stem),
            "algorithm": payload.get("algorithm"),
            "dataset": payload.get("dataset"),
            "noise_level": payload.get("noise_level"),
            "dataset_size": payload.get("dataset_size"),
            "seed": payload.get("seed"),
            "status": payload.get("status"),
            "runtime_seconds": payload.get("runtime_seconds"),
            "equation": payload.get("equation"),
            "from_cache": payload.get("from_cache", False),
        }

        metrics = payload.get("metrics", {}) or {}
        for name, value in metrics.items():
            row[f"metric_{name}"] = value

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_for_paper(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build paper-ready summary table.

    Columns: Algorithm | Dataset | Noise | SuccessRate
    """
    if df.empty:
        return pd.DataFrame(columns=["Algorithm", "Dataset", "Noise", "SuccessRate"])

    local = df.copy()
    local["success"] = local["status"].eq("success")

    grouped = (
        local.groupby(["algorithm", "dataset", "noise_level"], dropna=False)["success"]
        .mean()
        .reset_index()
    )

    grouped["SuccessRate"] = grouped["success"].round(4)
    grouped = grouped.drop(columns=["success"])
    grouped = grouped.rename(
        columns={
            "algorithm": "Algorithm",
            "dataset": "Dataset",
            "noise_level": "Noise",
        }
    )
    return grouped[["Algorithm", "Dataset", "Noise", "SuccessRate"]]


def export_paper_tables(df: pd.DataFrame, output_dir: Path | str) -> Dict[str, Path]:
    """
    Export summary tables for papers and reports.

    Creates `summary.csv`, `summary.md`, and `summary.tex`.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary = summarize_for_paper(df)

    csv_path = output_path / "summary.csv"
    md_path = output_path / "summary.md"
    tex_path = output_path / "summary.tex"

    summary.to_csv(csv_path, index=False)
    summary.to_markdown(md_path, index=False)

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(summary.to_latex(index=False))

    return {
        "summary_csv": csv_path,
        "summary_md": md_path,
        "summary_tex": tex_path,
    }


__all__ = [
    "aggregate_results",
    "summarize_for_paper",
    "export_paper_tables",
]
