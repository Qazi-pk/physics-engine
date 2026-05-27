"""
Artifact storage for benchmark experiments.

Writes per-run artifacts under a structured directory layout.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class ArtifactStore:
    """Stores experiment artifacts under results/runs/run_XXXXX/."""

    def __init__(self, output_dir: Path | str):
        self.output_dir = Path(output_dir)
        self.runs_dir = self.output_dir / "runs"
        self.summaries_dir = self.output_dir / "summaries"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.summaries_dir.mkdir(parents=True, exist_ok=True)

    def run_dir_for(self, run_id: int) -> Path:
        return self.runs_dir / f"run_{int(run_id):05d}"

    def save_run_artifacts(self, run: Any, result: Dict[str, Any]) -> Path:
        run_dir = self.run_dir_for(getattr(run, "run_id", result.get("run_id", 0)))
        run_dir.mkdir(parents=True, exist_ok=True)

        config_payload = run.to_dict() if hasattr(run, "to_dict") else dict(result.get("run", {}))
        with open(run_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config_payload, f, indent=2, default=str)

        metrics_payload = result.get("metrics", {}) if isinstance(result.get("metrics", {}), dict) else {}
        with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, indent=2, default=str)

        with open(run_dir / "result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)

        equation = result.get("equation")
        if equation is None:
            payload_result = result.get("result")
            equation = payload_result if isinstance(payload_result, str) else ""
        (run_dir / "discovered_equation.txt").write_text(str(equation), encoding="utf-8")

        return run_dir

    def save_summary_json(self, summary: Dict[str, Any]) -> Path:
        out = self.summaries_dir / "summary.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        return out

    def save_summary_markdown(self, markdown: str) -> Path:
        out = self.summaries_dir / "summary.md"
        out.write_text(markdown, encoding="utf-8")
        return out
