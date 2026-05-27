from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from physics_engine.benchmarks import generate_harmonic_oscillator_dataset
from physics_engine.discovery import discover_dynamical_system
from physics_engine.evaluation.discovery_confidence import (
    discovery_confidence,
    robustness_score,
    simplicity_score,
)

from .confidence import compute_discovery_confidence
from .experiment_registry import EXPERIMENTS


@dataclass
class EquationRunResult:
    target: str
    law: str
    error: float
    significant: dict[str, float]
    success: bool
    reason: str
    confidence: float


@dataclass
class DynamicalSystemRunResult:
    system_name: str
    seed: int
    noise: float
    dataset_size: int
    dataset_path: str
    equations: list[EquationRunResult]
    system_success: bool
    system_confidence: float
    system_discovery_confidence: float
    exception: str | None = None


def _law_complexity(law_text: str) -> int:
    compact = (law_text or "").replace(" ", "")
    if not compact:
        return 8
    return len(re.findall(r"\+|\-|\*|/|\^", compact))


def _residual_randomness(significant: dict[str, float]) -> float:
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


def _is_success(expected_tokens: tuple[str, ...], error_threshold: float, law_text: str, error: float) -> tuple[bool, str]:
    compact = (law_text or "").replace(" ", "")
    has_tokens = all(token in compact for token in expected_tokens)

    if has_tokens:
        return True, "expected_tokens"

    if np.isfinite(error) and error <= error_threshold:
        return True, "error_threshold"

    return False, "no_match"


def _config_by_name(name: str):
    for config in EXPERIMENTS:
        if config.name == name:
            return config
    raise ValueError(f"Missing experiment config: {name}")


def _run_level_dcs(equation_results: list[EquationRunResult], robustness_component: float) -> float:
    if not equation_results:
        return 0.0

    configs = {
        "x_dot": _config_by_name("harmonic_oscillator_xdot"),
        "v_dot": _config_by_name("harmonic_oscillator_vdot"),
    }

    accuracy_values = []
    simplicity_values = []
    residual_values = []

    for result in equation_results:
        config = configs[result.target]
        if np.isfinite(result.error) and config.error_threshold > 0:
            accuracy_values.append(float(np.clip(1.0 - (result.error / config.error_threshold), 0.0, 1.0)))
        else:
            accuracy_values.append(0.0)
        simplicity_values.append(simplicity_score(_law_complexity(result.law)))
        residual_values.append(_residual_randomness(result.significant))

    return discovery_confidence(
        A=float(np.mean(accuracy_values)),
        S=float(np.mean(simplicity_values)),
        R=float(np.mean(residual_values)),
        N=float(np.clip(robustness_component, 0.0, 1.0)),
    )


def generate_dynamical_system_summary(
    results: list[DynamicalSystemRunResult],
    output_dir: str | Path,
) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    total = len(results)
    successes = sum(1 for item in results if item.system_success)
    success_rate = (100.0 * successes / total) if total > 0 else 0.0
    mean_system_confidence = float(np.mean([item.system_confidence for item in results])) if results else 0.0

    robustness_component = robustness_score(successes, total)
    mean_system_dcs = float(np.mean([item.system_discovery_confidence for item in results])) if results else 0.0

    per_target: dict[str, dict] = {}
    for target in ("x_dot", "v_dot"):
        target_rows = [eq for run in results for eq in run.equations if eq.target == target]
        target_successes = sum(1 for row in target_rows if row.success)
        target_total = len(target_rows)
        target_rate = (100.0 * target_successes / target_total) if target_total > 0 else 0.0
        target_mean_error = float(np.mean([row.error for row in target_rows])) if target_rows else float("nan")
        target_mean_confidence = (
            float(np.mean([row.confidence for row in target_rows])) if target_rows else float("nan")
        )
        per_target[target] = {
            "runs": target_total,
            "successes": target_successes,
            "success_rate_percent": round(target_rate, 2),
            "mean_error": target_mean_error,
            "mean_confidence": target_mean_confidence,
        }

    summary = {
        "total_runs": total,
        "system": {
            "name": "harmonic_oscillator_system",
            "runs": total,
            "successes": successes,
            "success_rate_percent": round(success_rate, 2),
            "mean_system_confidence": mean_system_confidence,
            "mean_system_discovery_confidence": mean_system_dcs,
            "robustness_score": robustness_component,
        },
        "equations": per_target,
        "raw_results": [asdict(item) for item in results],
    }

    summary_json_path = output_path / "summary.json"
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# L5 Dynamical System Discovery Summary",
        "",
        "| System | Runs | Successes | Success Rate | Mean System Confidence | Mean System DCS |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| harmonic_oscillator_system | {total} | {successes} | {round(success_rate, 2)}% | "
            f"{mean_system_confidence:.4f} | {mean_system_dcs:.4f} |"
        ),
        "",
        "## Equation-Level Metrics",
        "",
        "| Equation Target | Runs | Successes | Success Rate | Mean Error | Mean Confidence |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for target in ("x_dot", "v_dot"):
        row = per_target[target]
        lines.append(
            f"| {target} | {row['runs']} | {row['successes']} | {row['success_rate_percent']}% | {row['mean_error']:.6f} | {row['mean_confidence']:.4f} |"
        )

    summary_md_path = output_path / "summary.md"
    summary_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return summary


def run_harmonic_oscillator_system_benchmark(
    output_dir: str | Path = "results_dynamical_system",
    repeats: int = 5,
    dataset_sizes: Iterable[int] = (200, 500),
    noise_levels: Iterable[float] = (0.0, 0.01, 0.05),
    base_seed: int = 123,
    show_progress: bool = False,
) -> list[DynamicalSystemRunResult]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset_dir = output_path / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    config_xdot = _config_by_name("harmonic_oscillator_xdot")
    config_vdot = _config_by_name("harmonic_oscillator_vdot")

    results: list[DynamicalSystemRunResult] = []
    total_runs = len(tuple(dataset_sizes)) * len(tuple(noise_levels)) * repeats
    completed_runs = 0

    for size in tuple(dataset_sizes):
        for noise in tuple(noise_levels):
            for trial in range(repeats):
                seed = base_seed + trial
                dataset_file = dataset_dir / f"harmonic_oscillator_n{size}_noise{str(noise).replace('.', 'p')}_seed{seed}.csv"

                system_success = False
                system_confidence = 0.0
                system_dcs = 0.0
                equation_results: list[EquationRunResult] = []
                exception_text = None

                try:
                    df = generate_harmonic_oscillator_dataset(
                        steps=int(size),
                        dt=0.01,
                        omega=1.0,
                        noise_std=float(noise),
                        seed=seed,
                    )
                    df.to_csv(dataset_file, index=False)

                    _, details = discover_dynamical_system(
                        str(dataset_file),
                        targets=["x_dot", "v_dot"],
                        max_basis_terms=6,
                        max_iterations=1,
                        allowed_powers=[1],
                        enforce_dimensions=False,
                        unary_functions=[],
                    )

                    for detail in details:
                        config = config_xdot if detail.target == "x_dot" else config_vdot
                        success, reason = _is_success(
                            expected_tokens=config.expected_tokens,
                            error_threshold=config.error_threshold,
                            law_text=detail.law,
                            error=detail.error,
                        )
                        confidence = compute_discovery_confidence(
                            config=config,
                            law_text=detail.law,
                            error=detail.error,
                            reason=reason,
                            exception=None,
                        )
                        equation_results.append(
                            EquationRunResult(
                                target=detail.target,
                                law=detail.law,
                                error=detail.error,
                                significant=detail.significant,
                                success=success,
                                reason=reason,
                                confidence=confidence,
                            )
                        )

                    system_success = all(item.success for item in equation_results)
                    system_confidence = float(np.mean([item.confidence for item in equation_results]))
                    system_dcs = _run_level_dcs(
                        equation_results,
                        robustness_component=1.0 if system_success else 0.0,
                    )
                except Exception as exc:
                    exception_text = f"{type(exc).__name__}: {exc}"

                run_result = DynamicalSystemRunResult(
                    system_name="harmonic_oscillator_system",
                    seed=seed,
                    noise=float(noise),
                    dataset_size=int(size),
                    dataset_path=str(dataset_file),
                    equations=equation_results,
                    system_success=system_success,
                    system_confidence=system_confidence,
                    system_discovery_confidence=system_dcs,
                    exception=exception_text,
                )
                results.append(run_result)

                result_file = output_path / f"harmonic_oscillator_n{size}_noise{str(noise).replace('.', 'p')}_seed{seed}.json"
                result_file.write_text(json.dumps(asdict(run_result), indent=2), encoding="utf-8")

                completed_runs += 1
                if show_progress and (completed_runs == 1 or completed_runs == total_runs or completed_runs % 5 == 0):
                    status = "ok" if system_success else "partial"
                    print(
                        f"[{completed_runs}/{total_runs}] harmonic_oscillator n={size} noise={noise} seed={seed} -> {status}",
                        flush=True,
                    )

    return results
