from __future__ import annotations

import json
import re
from collections import Counter
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from physics_engine.evaluation.discovery_confidence import (
    discovery_confidence,
    robustness_score,
    simplicity_score,
)

from .experiment_registry import EXPERIMENTS
from .experiment_runner import RunResult


EXPERIMENT_CONFIG_BY_NAME = {config.name: config for config in EXPERIMENTS}


def _law_complexity(law_text: str) -> int:
    compact = (law_text or "").replace(" ", "")
    if not compact:
        return 8
    return len(re.findall(r"\+|\-|\*|/|\^", compact))


def _residual_randomness_from_significant(significant: dict) -> float:
    if not significant:
        return 1.0

    max_corr = 0.0
    for value in significant.values():
        try:
            corr = abs(float(value))
        except (TypeError, ValueError):
            continue
        if np.isfinite(corr):
            max_corr = max(max_corr, corr)

    return float(np.clip(1.0 - max_corr, 0.0, 1.0))


def generate_summary_report(results: list[RunResult], output_dir: str | Path = "results") -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list[RunResult]] = defaultdict(list)
    for item in results:
        grouped[item.experiment].append(item)

    experiments = {}
    for name, items in grouped.items():
        config = EXPERIMENT_CONFIG_BY_NAME.get(name)
        success_count = sum(1 for item in items if item.success)
        total = len(items)
        success_rate = (100.0 * success_count / total) if total > 0 else 0.0
        mean_error = float(np.mean([item.error for item in items])) if items else float("nan")
        mean_confidence = float(np.mean([item.confidence for item in items])) if items else float("nan")

        if config is not None and np.isfinite(mean_error) and config.error_threshold > 0:
            accuracy_component = float(np.clip(1.0 - (mean_error / config.error_threshold), 0.0, 1.0))
        else:
            accuracy_component = 0.0

        simplicity_component = float(
            np.mean([simplicity_score(_law_complexity(item.law)) for item in items])
        ) if items else 0.0
        residual_component = float(
            np.mean([_residual_randomness_from_significant(item.significant) for item in items])
        ) if items else 0.0
        robustness_component = robustness_score(success_count, total)
        dcs = discovery_confidence(
            A=accuracy_component,
            S=simplicity_component,
            R=residual_component,
            N=robustness_component,
        )

        experiments[name] = {
            "runs": total,
            "successes": success_count,
            "success_rate_percent": round(success_rate, 2),
            "mean_error": mean_error,
            "mean_confidence": mean_confidence,
            "discovery_confidence": dcs,
            "structure_detection_counts": dict(
                Counter(
                    str(item.discovery_metadata.get("detected_structure", "unknown"))
                    for item in items
                    if item.discovery_metadata
                )
            ),
            "mean_pruned_interactions": float(
                np.mean([
                    float(item.discovery_metadata.get("pruned_interaction_count", 0) or 0)
                    for item in items
                ])
            ) if items else 0.0,
            "physics_feature_usage_counts": dict(
                Counter(feature for item in items for feature in item.physics_features_used)
            ),
            "latent_feature_usage_counts": dict(
                Counter(feature for item in items for feature in item.latent_features_used)
            ),
            "latent_discovery_log_counts": dict(
                Counter(message for item in items for message in item.latent_discovery_log)
            ),
        }

    summary = {
        "total_runs": len(results),
        "experiments": experiments,
        "raw_results": [asdict(item) for item in results],
    }

    summary_json_path = output_path / "summary.json"
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Discovery Success Summary",
        "",
        "| System | Runs | Successes | Success Rate | Mean Error | Mean Confidence | Discovery Confidence |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for name in sorted(experiments):
        row = experiments[name]
        lines.append(
            f"| {name} | {row['runs']} | {row['successes']} | {row['success_rate_percent']}% | {row['mean_error']:.6f} | {row['mean_confidence']:.4f} | {row['discovery_confidence']:.4f} |"
        )

    lines.extend(["", "## Physics Feature Usage", ""])
    for name in sorted(experiments):
        usage = experiments[name].get("physics_feature_usage_counts", {})
        if usage:
            usage_text = ", ".join(f"{feature}:{count}" for feature, count in sorted(usage.items()))
        else:
            usage_text = "none"
        lines.append(f"- {name}: {usage_text}")

    lines.extend(["", "## Latent Feature Usage", ""])
    for name in sorted(experiments):
        usage = experiments[name].get("latent_feature_usage_counts", {})
        if usage:
            usage_text = ", ".join(f"{feature}:{count}" for feature, count in sorted(usage.items()))
        else:
            usage_text = "none"
        lines.append(f"- {name}: {usage_text}")

    lines.extend(["", "## Latent Variable Discovery Logs", ""])
    for name in sorted(experiments):
        usage = experiments[name].get("latent_discovery_log_counts", {})
        if usage:
            usage_text = ", ".join(f"{message}:{count}" for message, count in sorted(usage.items()))
        else:
            usage_text = "none"
        lines.append(f"- {name}: {usage_text}")

    lines.extend(["", "## Structure-Guided Discovery", ""])
    for name in sorted(experiments):
        counts = experiments[name].get("structure_detection_counts", {})
        mean_pruned = float(experiments[name].get("mean_pruned_interactions", 0.0))
        if counts:
            counts_text = ", ".join(f"{key}:{value}" for key, value in sorted(counts.items()))
        else:
            counts_text = "none"
        lines.append(
            f"- {name}: structures={counts_text}; mean_pruned_interactions={mean_pruned:.3f}"
        )

    summary_md_path = output_path / "summary.md"
    summary_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return summary
