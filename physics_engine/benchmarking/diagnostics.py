"""
Automatic failure diagnostics and recommendation engine for PIR benchmarking.

When experiments have low or zero success rates PIR calls this module to
produce human-readable diagnostics and concrete recommended fixes—so the user
never has to manually diagnose why discovery failed.

Typical usage (automatically called by SummaryGenerator)::

    from physics_engine.benchmarking.diagnostics import analyze_failures, suggest_fixes

    results = [...]   # list of run result dicts
    diags   = analyze_failures(results)
    fixes   = suggest_fixes(results)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Internal thresholds (tweak without breaking the public API)
# ---------------------------------------------------------------------------
_MIN_SAMPLES_SYMBOLIC   = 200    # below this → dataset too small warning
_MIN_SAMPLES_JACOBIAN   = 400    # above to reliably discover Jacobians
_LARGE_ERROR_THRESHOLD  = 5.0    # residual error that flags "library mismatch"
_TIMEOUT_MARKER         = "KeyboardInterrupt"
_INFINITY               = float("inf")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_failures(results: List[Dict[str, Any]]) -> List[str]:
    """
    Inspect a list of experiment results and return diagnostic strings.

    Args:
        results: List of result dicts, each containing at minimum
                 ``status``, ``error`` (float or None), ``dataset_size``
                 (int), and optionally ``law`` (string).

    Returns:
        List of human-readable diagnostic messages (may be empty if all ok).
    """
    if not results:
        return []

    diagnostics: List[str] = []

    # ---- collect raw numbers -----------------------------------------
    errors:   List[float] = []
    timeouts: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for r in results:
        law_str = r.get("law", "") or ""
        err     = r.get("error")

        if _TIMEOUT_MARKER in str(law_str):
            timeouts.append(r)

        if r.get("status", "").lower() not in ("success", "ok", "true", True):
            failures.append(r)

        if err is not None and not _is_inf_or_nan(err):
            errors.append(float(err))

    total      = len(results)
    n_failed   = len(failures)
    success_rate = 1.0 - n_failed / total if total else 0.0

    # ---- dataset size --------------------------------------------------
    dataset_size = _extract_dataset_size(results)
    if dataset_size is not None:
        if dataset_size < _MIN_SAMPLES_SYMBOLIC:
            diagnostics.append(
                f"Dataset size ({dataset_size} samples) is below the recommended "
                f"minimum ({_MIN_SAMPLES_SYMBOLIC}) for reliable symbolic discovery."
            )
        elif dataset_size < _MIN_SAMPLES_JACOBIAN:
            diagnostics.append(
                f"Dataset size ({dataset_size} samples) may be insufficient for "
                f"multi-target discovery such as robot Jacobians "
                f"(recommended: {_MIN_SAMPLES_JACOBIAN}+)."
            )

    # ---- error magnitude -----------------------------------------------
    if errors:
        max_error = max(errors)
        mean_error = sum(errors) / len(errors)
        if max_error > _LARGE_ERROR_THRESHOLD:
            diagnostics.append(
                f"Large residual errors detected (max={max_error:.2f}, "
                f"mean={mean_error:.2f}). The feature library likely does not "
                f"contain the composite terms needed by the target equations."
            )
    else:
        diagnostics.append(
            "No finite errors recorded—all runs may have terminated before "
            "producing a valid residual. Check for timeouts or algorithm crashes."
        )

    # ---- timeouts ------------------------------------------------------
    if timeouts:
        n = len(timeouts)
        diagnostics.append(
            f"{n} run(s) terminated early (KeyboardInterrupt / timeout). "
            "Increase the per-experiment timeout or reduce algorithm complexity."
        )

    # ---- complete failure ----------------------------------------------
    if success_rate == 0.0 and total > 0:
        diagnostics.append(
            "Zero successful discoveries. All targets failed validation—"
            "this typically indicates a mismatch between the feature library "
            "and the true equation structure."
        )

    # ---- variable coverage hint ----------------------------------------
    law_strings = [str(r.get("law", "")) for r in results]
    _check_missing_trig_features(law_strings, results, diagnostics)

    return diagnostics


def suggest_fixes(results: List[Dict[str, Any]]) -> List[str]:
    """
    Return a prioritised list of concrete recommended actions.

    Args:
        results: Same format as :func:`analyze_failures`.

    Returns:
        List of actionable recommendation strings.
    """
    if not results:
        return []

    suggestions: List[str] = []

    dataset_size = _extract_dataset_size(results)
    errors = [
        float(r["error"])
        for r in results
        if r.get("error") is not None and not _is_inf_or_nan(r.get("error", _INFINITY))
    ]
    timeouts = [r for r in results if _TIMEOUT_MARKER in str(r.get("law", ""))]

    # ---- sample count --------------------------------------------------
    if dataset_size is not None and dataset_size < _MIN_SAMPLES_JACOBIAN:
        target = 1000 if dataset_size < 200 else 500
        suggestions.append(
            f"Increase dataset_sizes to {target}–2000 samples for more reliable "
            "symbolic regression."
        )

    # ---- feature library -----------------------------------------------
    law_strings = [str(r.get("law", "")) for r in results]
    if _laws_lack_composite_trig(law_strings, results):
        suggestions.append(
            'Add composite trigonometric features (e.g. sin(θ₁+θ₂), cos(θ₁+θ₂)) '
            "to the feature library. Use the 'jacobian' profile in "
            "FeatureLibraryRobotics or set library_profile='jacobian' in the "
            "benchmark config."
        )
    else:
        suggestions.append(
            "Try a richer feature library profile: 'cross_terms' or 'extended'."
        )

    # ---- error magnitude -----------------------------------------------
    if errors and max(errors) > _LARGE_ERROR_THRESHOLD:
        suggestions.append(
            "Increase symbolic search iterations (e.g. max_iterations=1000) "
            "to allow finer coefficient optimisation."
        )

    # ---- timeouts ------------------------------------------------------
    if timeouts:
        suggestions.append(
            "Increase the per-experiment timeout (e.g. timeout=300 seconds) "
            "to allow long-running symbolic searches to complete."
        )

    # ---- noise ---------------------------------------------------------
    noise_levels = list({
        r.get("noise_level") for r in results
        if r.get("noise_level") is not None
    })
    if noise_levels and any(n > 0.05 for n in noise_levels if isinstance(n, (int, float))):
        suggestions.append(
            "High noise levels detected. Consider reducing noise or applying a "
            "data-smoothing pre-processing step before discovery."
        )

    # ---- general fallback ----------------------------------------------
    if not suggestions:
        suggestions.append(
            "Re-run with more random seeds (seeds: [1,2,3,4,5]) to obtain "
            "statistically robust results."
        )

    return suggestions


def format_diagnostics_md(
    results: List[Dict[str, Any]],
    *,
    success_rate: Optional[float] = None,
    n_success: Optional[int] = None,
    n_total: Optional[int] = None,
) -> str:
    """
    Render a full Markdown diagnostics block suitable for appending to
    a ``summary.md`` report.

    Args:
        results:      Raw experiment result list.
        success_rate: Pre-computed rate (0–1); computed from *results* if None.
        n_success:    Pre-computed count; computed if None.
        n_total:      Pre-computed total; computed if None.

    Returns:
        Multi-line Markdown string.
    """
    diags = analyze_failures(results)
    fixes = suggest_fixes(results)

    if n_total is None:
        n_total = len(results)
    if n_success is None:
        n_success = sum(
            1 for r in results
            if r.get("status", "").lower() in ("success", "ok", "true")
        )
    if success_rate is None:
        success_rate = n_success / n_total if n_total else 0.0

    pct = success_rate * 100
    lines: List[str] = [
        "",
        "---",
        "",
        "## Diagnostics",
        "",
        f"**Success rate: {pct:.1f}% ({n_success}/{n_total})**",
        "",
    ]

    if diags:
        lines.append("### Issues Detected")
        lines.append("")
        for d in diags:
            lines.append(f"- {d}")
        lines.append("")
    else:
        lines.append("_No issues detected._")
        lines.append("")

    if fixes:
        lines.append("### Recommended Actions")
        lines.append("")
        for f in fixes:
            lines.append(f"- {f}")
        lines.append("")

    return "\n".join(lines)


def format_diagnostics_dict(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Return diagnostics and recommendations as a structured dictionary,
    ready for embedding in ``summary.json``.
    """
    n_total   = len(results)
    n_success = sum(
        1 for r in results
        if r.get("status", "").lower() in ("success", "ok", "true")
    )
    return {
        "success_rate": n_success / n_total if n_total else 0.0,
        "n_success": n_success,
        "n_total": n_total,
        "diagnostics": analyze_failures(results),
        "recommendations": suggest_fixes(results),
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _is_inf_or_nan(v: Any) -> bool:
    try:
        return math.isnan(float(v)) or math.isinf(float(v))
    except (TypeError, ValueError):
        return True


def _extract_dataset_size(results: List[Dict[str, Any]]) -> Optional[int]:
    for r in results:
        ds = r.get("dataset_size") or (r.get("run") or {}).get("dataset_size")
        if ds is not None:
            try:
                return int(ds)
            except (TypeError, ValueError):
                pass
    return None


def _laws_lack_composite_trig(
    law_strings: List[str],
    results: List[Dict[str, Any]],
) -> bool:
    """Return True if results look like a Jacobian problem missing composite trig."""
    targets = {(r.get("target") or "") for r in results}
    jacobian_targets = {"J11", "J12", "J21", "J22"}
    is_jacobian = bool(targets & jacobian_targets)
    has_composite = any(
        ("theta1 + theta2" in s or "theta1+theta2" in s or "_theta2" in s)
        for s in law_strings
    )
    return is_jacobian and not has_composite


def _check_missing_trig_features(
    law_strings: List[str],
    results: List[Dict[str, Any]],
    diagnostics: List[str],
) -> None:
    """Append a diagnostic if Jacobian composite trig features appear missing."""
    if _laws_lack_composite_trig(law_strings, results):
        diagnostics.append(
            "Jacobian targets (J11–J22) require composite trigonometric features "
            "sin(θ₁+θ₂) and cos(θ₁+θ₂). These are absent from the current feature "
            "library, which is the primary cause of failure."
        )


__all__ = [
    "analyze_failures",
    "suggest_fixes",
    "format_diagnostics_md",
    "format_diagnostics_dict",
]
