"""
Metrics for evaluating discovery experiments.

Provides standardized metrics for comparing algorithms and validating results.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def compute_metrics(
    result: Any,
    ground_truth: Optional[Any] = None,
    **kwargs
) -> Dict[str, float]:
    """
    Compute all applicable metrics for an experiment result.
    
    Args:
        result: Experiment result (model, equation, etc.)
        ground_truth: Optional ground truth for comparison
        **kwargs: Additional parameters for metric computation
    
    Returns:
        Dictionary of metric names to values
    """
    metrics = {}
    
    # Try to extract metrics from result
    if hasattr(result, "score"):
        metrics["score"] = float(result.score)
    
    if hasattr(result, "r2"):
        metrics["r2"] = float(result.r2)
    
    if hasattr(result, "mse"):
        metrics["mse"] = float(result.mse)
    
    if hasattr(result, "equation"):
        metrics["equation_complexity"] = equation_complexity(str(result.equation))
    
    # Compare with ground truth if available
    if ground_truth is not None:
        if hasattr(result, "equation") and hasattr(ground_truth, "equation"):
            metrics["equation_error"] = equation_error(result.equation, ground_truth.equation)
        
        if hasattr(result, "parameters") and hasattr(ground_truth, "parameters"):
            metrics["parameter_error"] = parameter_error(
                result.parameters,
                ground_truth.parameters
            )
    
    return metrics


def equation_error(predicted: str | Any, true: str | Any) -> float:
    """
    Compute error between predicted and true equations.
    
    For symbolic equations, computes symbolic difference.
    For strings, uses edit distance.
    
    Args:
        predicted: Predicted equation
        true: True/ground truth equation
    
    Returns:
        Error metric (0 = perfect match, higher = more different)
    """
    # Convert to strings for comparison
    pred_str = str(predicted)
    true_str = str(true)
    
    # Exact match
    if pred_str == true_str:
        return 0.0
    
    # Normalized edit distance
    import difflib
    similarity = difflib.SequenceMatcher(None, pred_str, true_str).ratio()
    return 1.0 - similarity


def parameter_error(
    predicted: Dict[str, float],
    true: Dict[str, float],
    metric: str = "relative_l2"
) -> float:
    """
    Compute error between predicted and true parameters.
    
    Args:
        predicted: Predicted parameter values
        true: True parameter values
        metric: Error metric type ('relative_l2', 'absolute', 'max')
    
    Returns:
        Parameter error value
    
    Examples:
        >>> pred = {"a": 1.1, "b": 2.05}
        >>> true = {"a": 1.0, "b": 2.0}
        >>> error = parameter_error(pred, true)
        >>> print(f"Error: {error:.4f}")
        Error: 0.0707
    """
    # Get common parameters
    common_keys = set(predicted.keys()) & set(true.keys())
    
    if not common_keys:
        return float("inf")
    
    if metric == "relative_l2":
        # Relative L2 norm
        errors = []
        for key in common_keys:
            pred_val = predicted[key]
            true_val = true[key]
            if abs(true_val) > 1e-10:
                rel_error = abs(pred_val - true_val) / abs(true_val)
            else:
                rel_error = abs(pred_val - true_val)
            errors.append(rel_error ** 2)
        return np.sqrt(np.mean(errors))
    
    elif metric == "absolute":
        # Absolute L2 norm
        errors = [(predicted[k] - true[k]) ** 2 for k in common_keys]
        return np.sqrt(np.mean(errors))
    
    elif metric == "max":
        # Maximum absolute error
        errors = [abs(predicted[k] - true[k]) for k in common_keys]
        return max(errors)
    
    else:
        raise ValueError(f"Unknown metric: {metric}")


def runtime_metric(runtime: float) -> Dict[str, float]:
    """
    Compute runtime metrics.
    
    Args:
        runtime: Execution time in seconds
    
    Returns:
        Dictionary with runtime statistics
    """
    return {
        "runtime_seconds": runtime,
        "runtime_minutes": runtime / 60,
    }


def equation_complexity(equation: str) -> int:
    """
    Compute complexity score for an equation.
    
    Counts number of operators, variables, and constants.
    
    Args:
        equation: Equation as string
    
    Returns:
        Complexity score (lower = simpler)
    
    Examples:
        >>> complexity = equation_complexity("y = a*x + b")
        >>> print(complexity)
        5
    """
    # Count operators
    operators = ["+", "-", "*", "/", "**", "^", "sin", "cos", "exp", "log"]
    complexity = sum(equation.count(op) for op in operators)
    
    # Count terms (rough estimate)
    terms = len([t for t in equation.split() if t not in ["=", "(", ")", ","]])
    
    return complexity + terms


def success_rate(results: list) -> float:
    """
    Compute success rate from list of results.
    
    Args:
        results: List of result dictionaries
    
    Returns:
        Success rate (0-1)
    """
    if not results:
        return 0.0
    
    successful = sum(1 for r in results if r.get("status") != "failed")
    return successful / len(results)


def mean_metric(results: list, metric_name: str) -> float:
    """
    Compute mean value of a metric across results.
    
    Args:
        results: List of result dictionaries
        metric_name: Name of metric to average
    
    Returns:
        Mean metric value
    """
    values = []
    for result in results:
        if "metrics" in result and metric_name in result["metrics"]:
            values.append(result["metrics"][metric_name])
    
    if not values:
        return float("nan")
    
    return np.mean(values)


def median_metric(results: list, metric_name: str) -> float:
    """
    Compute median value of a metric across results.
    
    Args:
        results: List of result dictionaries
        metric_name: Name of metric
    
    Returns:
        Median metric value
    """
    values = []
    for result in results:
        if "metrics" in result and metric_name in result["metrics"]:
            values.append(result["metrics"][metric_name])
    
    if not values:
        return float("nan")
    
    return float(np.median(values))


def aggregate_metrics(results: list) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics across all results.
    
    Computes mean, median, std, min, max for each metric.
    
    Args:
        results: List of result dictionaries
    
    Returns:
        Dictionary mapping metric names to statistics
    """
    # Collect all metrics
    all_metrics = {}
    for result in results:
        if "metrics" in result:
            for metric_name, value in result["metrics"].items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)
    
    # Compute statistics
    aggregated = {}
    for metric_name, values in all_metrics.items():
        if values:
            aggregated[metric_name] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "count": len(values),
            }
    
    return aggregated


__all__ = [
    "compute_metrics",
    "equation_error",
    "parameter_error",
    "runtime_metric",
    "equation_complexity",
    "success_rate",
    "mean_metric",
    "median_metric",
    "aggregate_metrics",
]
