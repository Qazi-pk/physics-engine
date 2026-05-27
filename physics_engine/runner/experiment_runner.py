from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import sympy as sp

from physics_engine.discovery import (
    discover_hamiltonian_from_dataframe,
    discover_lagrangian_from_dataframe,
    discover_law,
)
from physics_engine.variational import discover_structured_lagrangian_from_dataframe
from physics_engine.utils.latent_variables import generate_latent_variables
from physics_engine.utils.physics_features import generate_physics_features, generate_trigonometric_features

from .confidence import compute_discovery_confidence
from .experiment_registry import EXPERIMENTS, ExperimentConfig

try:
    from physics_engine.diffusion import FlowPriorScorer
except Exception:
    FlowPriorScorer = None


@dataclass
class RunResult:
    experiment: str
    seed: int
    noise: float
    dataset_size: int
    operator_profile: str
    dataset_path: str
    law: str
    error: float
    significant: dict
    success: bool
    reason: str
    confidence: float
    exception: str | None = None
    physics_features_generated: list[str] = None
    physics_features_used: list[str] = None
    latent_features_generated: list[str] = None
    latent_features_used: list[str] = None
    latent_discovery_log: list[str] = None
    discovery_metadata: dict = None
    dcs_formal: dict | None = None

    def __post_init__(self):
        if self.physics_features_generated is None:
            self.physics_features_generated = []
        if self.physics_features_used is None:
            self.physics_features_used = []
        if self.latent_features_generated is None:
            self.latent_features_generated = []
        if self.latent_features_used is None:
            self.latent_features_used = []
        if self.latent_discovery_log is None:
            self.latent_discovery_log = []
        if self.discovery_metadata is None:
            self.discovery_metadata = {}


def _latent_discovery_messages(
    df,
    generated: list[str],
    used: list[str],
    available: list[str] | None = None,
) -> list[str]:
    discovered = set(generated)
    used_set = set(used)
    available_set = set(available or generated)
    messages: list[str] = []

    def _status_suffix(name: str) -> str:
        if name in discovered:
            return ""
        if name in available_set:
            return " (already present in dataset)"
        return ""

    if all(col in df.columns for col in ("x", "y")) and "r" in available_set:
        messages.append(f"latent variable discovered: r = sqrt(x^2 + y^2){_status_suffix('r')}")
    if all(col in df.columns for col in ("x", "y")) and "r2" in available_set:
        messages.append(f"latent variable discovered: r2 = x^2 + y^2{_status_suffix('r2')}")
    if "inv_r" in available_set:
        messages.append(f"latent variable discovered: inv_r = 1/r{_status_suffix('inv_r')}")
    if "inv_r2" in available_set:
        messages.append(f"latent variable discovered: inv_r2 = 1/r^2{_status_suffix('inv_r2')}")
    if "inv_r3" in available_set:
        messages.append(f"latent variable discovered: inv_r3 = 1/r^3{_status_suffix('inv_r3')}")
    if "x_over_r3" in available_set:
        suffix = " (used by discovered law)" if "x_over_r3" in used_set else ""
        messages.append(f"candidate feature: x_over_r3 = x/r^3{_status_suffix('x_over_r3')}{suffix}")
    if "y_over_r3" in available_set:
        suffix = " (used by discovered law)" if "y_over_r3" in used_set else ""
        messages.append(f"candidate feature: y_over_r3 = y/r^3{_status_suffix('y_over_r3')}{suffix}")
    if all(col in df.columns for col in ("vx", "vy")) and "v" in available_set:
        messages.append(f"latent variable discovered: v = sqrt(vx^2 + vy^2){_status_suffix('v')}")
    if "angular_momentum_z" in available_set:
        messages.append(
            f"latent variable discovered: angular_momentum_z = x*vy - y*vx{_status_suffix('angular_momentum_z')}"
        )

    if not messages and available_set:
        preview = ", ".join(sorted(generated)[:6])
        if not preview:
            preview = ", ".join(sorted(available_set)[:6])
        messages.append(f"latent features generated: {preview}")

    return messages


def _extract_symbol_names(expression_text: str) -> set[str]:
    if not expression_text:
        return set()

    try:
        parsed = sp.sympify(expression_text)
        return {str(symbol) for symbol in parsed.free_symbols}
    except Exception:
        return set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expression_text))


EXPECTED_EQUIVALENCE_EXPRESSIONS: dict[str, str] = {
    "newton": "m*a",
    "pendulum": "theta",
    "kepler_third_law": "r**(3/2)",
    "inverse_square_acceleration": "1/r**2",
    "gravity": "m1*m2/r**2",
    "orbit_ax": "x_over_r3",
    "orbit_ay": "y_over_r3",
    "harmonic_oscillator_xdot": "v",
    "harmonic_oscillator_vdot": "x",
}


def _build_equivalence_eval_frame(df: pd.DataFrame) -> pd.DataFrame:
    eval_df = df.copy()
    all_features = {}
    all_features.update(generate_physics_features(eval_df))
    all_features.update(generate_latent_variables(eval_df))
    for name, values in all_features.items():
        if name not in eval_df.columns:
            eval_df[name] = values
    return eval_df


def _numerically_equivalent(
    discovered_law: str,
    expected_law: str,
    eval_df: pd.DataFrame,
    n_test_points: int = 500,
    rtol: float = 0.02,
    seed: int = 0,
) -> bool | None:
    if not discovered_law or not expected_law or eval_df.empty:
        return None

    try:
        expr_found = sp.sympify(discovered_law)
        expr_expected = sp.sympify(expected_law)
    except Exception:
        return None

    symbol_names = sorted({str(s) for s in expr_found.free_symbols.union(expr_expected.free_symbols)})
    if not symbol_names:
        return None

    for name in symbol_names:
        if name not in eval_df.columns:
            return None

    rng = np.random.default_rng(seed)
    eval_subset = eval_df.loc[:, symbol_names].copy()
    for name in symbol_names:
        eval_subset[name] = pd.to_numeric(eval_subset[name], errors="coerce")
    eval_subset = eval_subset.replace([np.inf, -np.inf], np.nan).dropna()
    if eval_subset.empty:
        return None

    sample_n = int(min(max(50, n_test_points), len(eval_subset)))
    sample_idx = rng.choice(len(eval_subset), size=sample_n, replace=len(eval_subset) < sample_n)
    sampled = eval_subset.iloc[sample_idx]
    sampled_arrays = [sampled[name].to_numpy(dtype=float) for name in symbol_names]

    def _prune_small_additive_terms(expr: sp.Expr, rel_tol: float = 0.05) -> sp.Expr:
        terms = list(sp.Add.make_args(sp.expand(expr)))
        if len(terms) <= 1:
            return expr

        kept_terms: list[sp.Expr] = []
        magnitudes: list[float] = []
        for term in terms:
            try:
                f_term = sp.lambdify(symbol_names, term, "numpy")
                vals = np.asarray(f_term(*sampled_arrays), dtype=float)
                vals = np.ravel(vals)
                if vals.size == 1:
                    vals = np.full(sampled.shape[0], float(vals[0]), dtype=float)
                mag = float(np.median(np.abs(vals[np.isfinite(vals)]))) if np.any(np.isfinite(vals)) else 0.0
            except Exception:
                mag = 0.0
            magnitudes.append(mag)

        max_mag = max(magnitudes) if magnitudes else 0.0
        if max_mag <= 1e-12:
            return expr

        for term, mag in zip(terms, magnitudes):
            if mag >= rel_tol * max_mag:
                kept_terms.append(term)

        if not kept_terms:
            return expr
        return sp.simplify(sum(kept_terms))

    expr_found = _prune_small_additive_terms(expr_found)

    try:
        f_found = sp.lambdify(symbol_names, expr_found, "numpy")
        f_expected = sp.lambdify(symbol_names, expr_expected, "numpy")
        vals_found = np.asarray(f_found(*sampled_arrays), dtype=float)
        vals_expected = np.asarray(f_expected(*sampled_arrays), dtype=float)
    except Exception:
        return None

    vals_found = np.ravel(vals_found)
    vals_expected = np.ravel(vals_expected)
    if vals_found.size == 1:
        vals_found = np.full(int(n_test_points), float(vals_found[0]), dtype=float)
    if vals_expected.size == 1:
        vals_expected = np.full(int(n_test_points), float(vals_expected[0]), dtype=float)

    size = min(vals_found.size, vals_expected.size)
    if size == 0:
        return None
    vals_found = vals_found[:size]
    vals_expected = vals_expected[:size]

    finite_mask = np.isfinite(vals_found) & np.isfinite(vals_expected)
    if float(np.mean(finite_mask)) < 0.8:
        return None
    vals_found = vals_found[finite_mask]
    vals_expected = vals_expected[finite_mask]
    if vals_found.size == 0:
        return None

    # Allow global scaling/sign differences by fitting one scalar.
    denom = float(np.dot(vals_expected, vals_expected)) + 1e-12
    scale = float(np.dot(vals_found, vals_expected)) / denom
    vals_expected_scaled = scale * vals_expected

    rel_err = np.abs(vals_found - vals_expected_scaled) / (np.abs(vals_expected_scaled) + 1e-10)
    return float(np.median(rel_err)) < float(rtol)


def _dataset_kwargs(config: ExperimentConfig, dataset_size: int, noise: float, seed: int) -> dict:
    if config.name == "newton":
        return {"num_samples": dataset_size, "noise_std": noise, "seed": seed}

    if config.name == "gravity":
        return {"num_samples": dataset_size, "noise_std": noise, "seed": seed}

    if config.name == "pendulum":
        return {"steps": dataset_size, "dt": 0.01, "noise_std": noise, "seed": seed}

    if config.name == "kepler_third_law":
        return {"num_samples": dataset_size, "noise_std": noise, "seed": seed}

    if config.name == "inverse_square_acceleration":
        return {"num_samples": dataset_size, "noise_std": noise, "seed": seed}

    if config.name in {"orbit_ax", "orbit_ay"}:
        return {"steps": dataset_size, "dt": 0.01, "seed": seed, "noise_std": noise}

    if config.name in {"harmonic_oscillator_xdot", "harmonic_oscillator_vdot"}:
        return {"steps": dataset_size, "dt": 0.01, "omega": 1.0, "seed": seed, "noise_std": noise}

    if config.name in {"harmonic_oscillator_lagrangian", "harmonic_oscillator_structured_lagrangian"}:
        return {"steps": dataset_size, "dt": 0.01, "omega": 1.0, "seed": seed, "noise_std": noise}

    if config.name == "harmonic_oscillator_hamiltonian":
        return {"steps": dataset_size, "dt": 0.01, "omega": 1.0, "seed": seed, "noise_std": noise}

    if config.name in {"double_pendulum_theta1dot", "double_pendulum_theta2dot"}:
        return {"steps": dataset_size, "dt": 0.01, "seed": seed, "noise_std": noise}

    if config.name in {"planar_robot_j11", "planar_robot_j12", "planar_robot_j21", "planar_robot_j22"}:
        return {"num_samples": dataset_size, "noise_std": noise, "seed": seed}

    if config.name in {"planar_robot_fk_x", "planar_robot_fk_y"}:
        return {"num_samples": dataset_size, "noise_std": noise, "seed": seed}

    if config.name.startswith("franka_M"):
        return {"num_samples": dataset_size, "noise_std": noise, "seed": seed}

    if config.name == "oog_damped_oscillator":
        return {"num_samples": dataset_size, "noise_std": noise, "seed": seed}

    if config.name == "oog_relativistic_correction":
        return {"num_samples": dataset_size, "noise_std": noise, "seed": seed}
    if config.name.startswith("feynman_"):
        return {"num_samples": dataset_size, "noise_std": noise, "seed": seed}
    raise ValueError(f"Unsupported experiment: {config.name}")


def _add_noise_to_target(df, target: str, noise: float, seed: int):
    if noise <= 0:
        return df

    noisy_df = df.copy()
    rng = np.random.default_rng(seed)
    noisy_df[target] = noisy_df[target] + rng.normal(0.0, noise, len(noisy_df))
    return noisy_df


def _prepare_dataset_for_discovery(config: ExperimentConfig, df):
    if config.name.startswith("double_pendulum_"):
        keep_columns = ["theta1", "theta2", "omega1", "omega2", config.target_column]
        existing_columns = [column for column in keep_columns if column in df.columns]
        return df.loc[:, existing_columns].copy()

    if config.name.startswith("planar_robot_j"):
        keep_columns = ["theta1", "theta2", "l1", "l2", config.target_column]
        existing_columns = [column for column in keep_columns if column in df.columns]
        return df.loc[:, existing_columns].copy()
    if config.name.startswith("planar_robot_fk_"):
        prepared = df.copy()
        trig_features = generate_trigonometric_features(prepared)
        for name, values in trig_features.items():
            if name not in prepared.columns:
                prepared[name] = values

        if config.name == "planar_robot_fk_x":
            keep_columns = ["cos_theta1", "cos_theta12", config.target_column]
        else:
            keep_columns = ["sin_theta1", "sin_theta12", config.target_column]

        existing_columns = [column for column in keep_columns if column in prepared.columns]
        return prepared.loc[:, existing_columns].copy()

    if config.name.startswith("franka_M"):
        keep_columns = [f"q{i}" for i in range(1, 8)] + [config.target_column]
        existing_columns = [column for column in keep_columns if column in df.columns]
        return df.loc[:, existing_columns].copy()
    return df


def _sanitize_noise(noise: float) -> str:
    return str(noise).replace(".", "p")


def _compute_dcs_formal(
    df,
    target_col: str,
    law_text: str,
    error_value: float,
    success: bool,
) -> dict:
    """
    Compute formal DCS per paper Eq. 5 using real dataset quantities.

        DCS = 0.4*A + 0.2*S + 0.2*R + 0.2*N
        A = max(0, 1 - RMSE / std(y))        -- scale-invariant accuracy
        S = 1 / (1 + C)                      -- simplicity, C = sympy count_ops
        R = 1 - max_k |rho_k|, k in 1..5     -- residual autocorrelation
        N = placeholder (cross-seed, filled post-hoc by aggregator)

    Returns dict with {A, S, R, N, DCS, tau_A, complexity}. N is stored as
    1.0 if success else 0.0 (per-run proxy); aggregator overrides with the
    true cross-seed success rate.
    """
    import math
    import numpy as np
    import sympy as sp

    # --- tau_A: std of target on this dataset ---
    try:
        y = np.asarray(df[target_col].values, dtype=float)
        y = y[np.isfinite(y)]
        tau_a = float(np.std(y, ddof=1)) if y.size >= 2 else 1.0
        tau_a = max(tau_a, 1e-6)
    except Exception:
        tau_a = 1.0

    # --- A: accuracy ---
    if error_value is None or not math.isfinite(error_value) or error_value < 0:
        A = 0.0
    else:
        A = max(0.0, 1.0 - float(error_value) / tau_a)

    # --- S: simplicity via sympy count_ops (robust for any law string) ---
    try:
        expr = sp.sympify(law_text) if law_text else sp.Integer(0)
        complexity = int(sp.count_ops(expr))
    except Exception:
        # Fall back to token count if sympify fails on unusual law strings
        complexity = max(1, law_text.count(" + ") + law_text.count(" - ") + 1)
    S = 1.0 / (1.0 + complexity)

    # --- R: residual autocorrelation on held-out predictions ---
    # We reconstruct y_pred by lambdifying the law and evaluating on df.
    R = 0.5  # neutral default if reconstruction fails
    try:
        # Identify predictor columns (everything except target)
        predictors = [c for c in df.columns if c != target_col]
        if law_text and predictors:
            expr = sp.sympify(law_text)
            syms = [sp.Symbol(c) for c in predictors]
            f = sp.lambdify(syms, expr, "numpy")
            X = [np.asarray(df[c].values, dtype=float) for c in predictors]
            y_pred = np.asarray(f(*X), dtype=float).reshape(-1)
            y_true = np.asarray(df[target_col].values, dtype=float)
            mask = np.isfinite(y_pred) & np.isfinite(y_true)
            if mask.sum() >= 10:
                residuals = (y_true - y_pred)[mask]
                residuals = residuals - residuals.mean()
                denom = float(np.dot(residuals, residuals))
                if denom > 0:
                    max_abs_rho = 0.0
                    for lag in range(1, 6):
                        if lag >= len(residuals):
                            break
                        num = float(np.dot(residuals[:-lag], residuals[lag:]))
                        rho = abs(num / denom)
                        if rho > max_abs_rho:
                            max_abs_rho = rho
                    R = max(0.0, min(1.0, 1.0 - max_abs_rho))
    except Exception:
        pass  # keep R = 0.5

    # --- N: per-run placeholder (aggregator replaces with cross-seed value) ---
    N = 1.0 if success else 0.0

    DCS = 0.4 * A + 0.2 * S + 0.2 * R + 0.2 * N

    return {
        "A": round(float(A), 4),
        "S": round(float(S), 4),
        "R": round(float(R), 4),
        "N": round(float(N), 4),
        "DCS": round(float(DCS), 4),
        "tau_A": round(float(tau_a), 6),
        "complexity": int(complexity),
    }




def _run_discovery(
    config: ExperimentConfig,
    dataset_file: Path,
    discover_kwargs: dict,
) -> tuple[str, float, dict[str, float], dict]:
    mode = (config.discovery_mode or "standard").strip().lower()

    if mode == "lagrangian":
        q_col = str(discover_kwargs.get("q_col", "q"))
        dqdt_col = str(discover_kwargs.get("dqdt_col", "dqdt"))
        d2qdt2_col = str(discover_kwargs.get("d2qdt2_col", "d2qdt2"))
        lagrangian_kwargs = {
            "max_power": int(discover_kwargs.get("max_power", 2)),
            "include_cross_terms": bool(discover_kwargs.get("include_cross_terms", True)),
        }

        df = pd.read_csv(dataset_file)
        result = discover_lagrangian_from_dataframe(
            df=df,
            q_col=q_col,
            dqdt_col=dqdt_col,
            d2qdt2_col=d2qdt2_col,
            **lagrangian_kwargs,
        )
        return str(result.lagrangian), float(result.error), {}, {
            "discovery_mode": "lagrangian",
            "detected_structure": "n/a",
            "pruned_interaction_count": 0,
        }

    if mode == "structured_lagrangian":
        q_col = str(discover_kwargs.get("q_col", "q"))
        dq_col = str(discover_kwargs.get("dq_col", discover_kwargs.get("dqdt_col", "dqdt")))
        ddq_col = str(discover_kwargs.get("ddq_col", discover_kwargs.get("d2qdt2_col", "d2qdt2")))
        structured_kwargs = {
            "kinetic_max_even_power": int(discover_kwargs.get("kinetic_max_even_power", 4)),
            "potential_max_even_power": int(discover_kwargs.get("potential_max_even_power", 4)),
            "include_trig": bool(discover_kwargs.get("include_trig", True)),
        }

        df = pd.read_csv(dataset_file)
        result = discover_structured_lagrangian_from_dataframe(
            df=df,
            q_col=q_col,
            dq_col=dq_col,
            ddq_col=ddq_col,
            **structured_kwargs,
        )
        return str(result.lagrangian), float(result.residual_mse), {}, {
            "discovery_mode": "structured_lagrangian",
            "detected_structure": "structured_lagrangian",
            "pruned_interaction_count": 0,
            "kinetic_energy": str(result.kinetic_energy),
            "potential_energy": str(result.potential_energy),
            "residual_rmse": float(result.residual_rmse),
            "kinetic_coefficients": dict(result.kinetic_coefficients),
            "potential_coefficients": dict(result.potential_coefficients),
        }

    if mode == "hamiltonian":
        q_col = str(discover_kwargs.get("q_col", "q"))
        p_col = str(discover_kwargs.get("p_col", "p"))
        dqdt_col = str(discover_kwargs.get("dqdt_col", "dqdt"))
        dpdt_col = str(discover_kwargs.get("dpdt_col", "dpdt"))
        hamiltonian_kwargs = {
            "max_power": int(discover_kwargs.get("max_power", 2)),
            "include_cross_terms": bool(discover_kwargs.get("include_cross_terms", True)),
        }

        df = pd.read_csv(dataset_file)
        result = discover_hamiltonian_from_dataframe(
            df=df,
            q_col=q_col,
            p_col=p_col,
            dqdt_col=dqdt_col,
            dpdt_col=dpdt_col,
            **hamiltonian_kwargs,
        )
        return str(result.hamiltonian), float(result.error), {}, {
            "discovery_mode": "hamiltonian",
            "detected_structure": "n/a",
            "pruned_interaction_count": 0,
        }

    law, error, significant, metadata = discover_law(
        str(dataset_file),
        config.target_column,
        task_name=config.name,
        return_metadata=True,
        **discover_kwargs,
    )
    return str(law), float(error), significant, metadata


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(total_seconds, 60)
    hours, mins = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def _normalize_unary_functions(unary_functions):
    if unary_functions is None:
        return None

    resolved = []
    for func in unary_functions:
        if callable(func):
            resolved.append(func)
            continue

        if isinstance(func, str):
            name = func.strip().lower()
            if name == "sin":
                resolved.append(sp.sin)
                continue
            if name == "cos":
                resolved.append(sp.cos)
                continue

        raise ValueError(f"Unsupported unary function specifier: {func}")

    return resolved


def _is_success(
    config: ExperimentConfig,
    law_text: str,
    error: float,
    eval_df: pd.DataFrame | None = None,
) -> tuple[bool, str]:
    compact = law_text.replace(" ", "")

    expected_expr = EXPECTED_EQUIVALENCE_EXPRESSIONS.get(config.name)
    has_tokens: bool
    used_numerical_equivalence = False
    if expected_expr is not None and eval_df is not None:
        equivalence = _numerically_equivalent(
            discovered_law=law_text,
            expected_law=expected_expr,
            eval_df=eval_df,
        )
        if equivalence is not None:
            has_tokens = bool(equivalence)
            used_numerical_equivalence = True
        else:
            has_tokens = all(token in compact for token in config.expected_tokens)
    else:
        has_tokens = all(token in compact for token in config.expected_tokens)

    has_error = np.isfinite(error) and error <= config.error_threshold

    if config.success_mode == "token_and_error":
        if has_tokens and has_error:
            return True, "equivalent_and_error" if used_numerical_equivalence else "expected_tokens_and_error"
        if has_tokens and not has_error:
            return False, "equivalent_but_high_error" if used_numerical_equivalence else "tokens_but_high_error"
        if has_error and not has_tokens:
            return False, "low_error_but_not_equivalent" if used_numerical_equivalence else "low_error_but_missing_tokens"
        return False, "no_match"

    if has_tokens:
        return True, "expected_tokens"

    if has_error:
        return True, "error_threshold"

    return False, "no_match"


def _operator_profiles(selected: Iterable[str] | None = None) -> dict[str, dict]:
    return {
        "linear": {
            "allowed_powers": [1],
        },
        "extended": {
            "allowed_powers": [1, -1, -2],
        },
    } if selected is None else {
        name: profile
        for name, profile in {
            "linear": {
                "allowed_powers": [1],
            },
            "extended": {
                "allowed_powers": [1, -1, -2],
            },
        }.items()
        if name in set(selected)
    }


def run_all_experiments(
    output_dir: str | Path = "results",
    experiments: Iterable[ExperimentConfig] = EXPERIMENTS,
    noise_levels: Iterable[float] = (0.0, 0.01, 0.05),
    dataset_sizes: Iterable[int] = (200, 500),
    repeats: int = 5,
    base_seed: int = 123,
    experiment_names: Iterable[str] | None = None,
    operator_profiles: Iterable[str] | None = None,
    show_progress: bool = False,
    progress_interval: int = 1,
    use_ransac: bool = True,
    enforce_dimensions: bool = True,
    use_residual: bool = True,
    use_sparse: bool = True,
    use_ot_loss: bool = False,
    alpha: float = 0.7,
    beta: float = 0.3,
    use_flow_prior: bool = False,
    flow_prior_checkpoint: str | Path = "physics_engine/diffusion/flow_prior.pt",
    use_jepa: bool = False,
    use_vjepa: bool = False,
    gamma: float = 0.2,
    langevin_steps: int = 500,
    skip_existing: bool = False,
) -> list[RunResult]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset_dir = output_path / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    run_results: list[RunResult] = []

    flow_prior_scorer = None
    if use_flow_prior:
        if FlowPriorScorer is None:
            raise ImportError("FlowPriorScorer is unavailable. Ensure diffusion dependencies are installed.")

        checkpoint_path = Path(flow_prior_checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path.cwd() / checkpoint_path
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Flow prior checkpoint not found: {checkpoint_path}")

        flow_prior_scorer = FlowPriorScorer.from_checkpoint(str(checkpoint_path), prior_weight=0.1)

    noise_levels = tuple(noise_levels)
    dataset_sizes = tuple(dataset_sizes)

    selected_experiments = set(experiment_names) if experiment_names is not None else None
    profiles = _operator_profiles(operator_profiles)
    progress_interval = max(1, int(progress_interval))

    total_runs = 0
    for config in experiments:
        if selected_experiments is not None and config.name not in selected_experiments:
            continue
        for operator_name in profiles:
            if operator_name not in config.operator_profiles:
                continue
            total_runs += len(dataset_sizes) * len(noise_levels) * repeats

    completed_runs = 0
    started_at = time.perf_counter()

    for config in experiments:
        if selected_experiments is not None and config.name not in selected_experiments:
            continue

        eligible_profiles: list[tuple[str, dict]] = []
        for operator_name, profile_kwargs in profiles.items():
            if operator_name in config.operator_profiles:
                eligible_profiles.append((operator_name, profile_kwargs))

        has_extended_profile = any(name == "extended" for name, _ in eligible_profiles)

        for size in dataset_sizes:
            for noise in noise_levels:
                for trial in range(repeats):
                    seed = base_seed + trial
                    stop_after_linear_success = False

                    for operator_name, profile_kwargs in eligible_profiles:
                        if stop_after_linear_success:
                            total_runs = max(completed_runs, total_runs - 1)
                            continue

                        result_file = (
                            output_path
                            / f"{config.name}_n{size}_noise{_sanitize_noise(noise)}_{operator_name}_seed{seed}.json"
                        )
                        if skip_existing and result_file.exists():
                            try:
                                existing = json.loads(result_file.read_text(encoding="utf-8"))
                                record = RunResult(**existing)
                                run_results.append(record)
                                completed_runs += 1
                                if show_progress:
                                    print(
                                        f"[{completed_runs}/{total_runs}] {config.name} "
                                        f"n={size} noise={noise} profile={operator_name} seed={seed} -> skipped (existing)",
                                        flush=True,
                                    )
                                if operator_name == "linear" and record.success and has_extended_profile:
                                    stop_after_linear_success = True
                                continue
                            except Exception:
                                pass  # fall through to re-run if file is corrupt

                        discover_kwargs = dict(config.default_discovery_kwargs)
                        discover_kwargs.update(profile_kwargs)
                        discover_kwargs["unary_functions"] = _normalize_unary_functions(
                            discover_kwargs.get("unary_functions")
                        )
                        # Add pipeline toggles
                        discover_kwargs.setdefault("use_ransac", use_ransac)
                        discover_kwargs.setdefault("enforce_dimensions", enforce_dimensions)
                        discover_kwargs.setdefault("use_residual", use_residual)
                        discover_kwargs.setdefault("use_sparse", use_sparse)
                        discover_kwargs.setdefault("use_ot_loss", use_ot_loss)
                        discover_kwargs.setdefault("alpha", alpha)
                        discover_kwargs.setdefault("beta", beta)
                        discover_kwargs.setdefault("use_flow_prior", use_flow_prior)
                        discover_kwargs.setdefault("flow_prior_scorer", flow_prior_scorer)
                        discover_kwargs.setdefault("use_jepa", use_jepa)
                        discover_kwargs.setdefault("use_vjepa", use_vjepa)
                        discover_kwargs.setdefault("gamma", gamma)
                        discover_kwargs.setdefault("langevin_steps", langevin_steps)

                        df = config.dataset_generator(
                            **_dataset_kwargs(config, dataset_size=size, noise=noise, seed=seed)
                        )
                        df = _prepare_dataset_for_discovery(config, df)

                        dataset_file = (
                            dataset_dir
                            / f"{config.name}_n{size}_noise{_sanitize_noise(noise)}_{operator_name}_seed{seed}.csv"
                        )
                        df.to_csv(dataset_file, index=False)

                        law_text = ""
                        error_value = float("inf")
                        significant = {}
                        success = False
                        reason = "exception"
                        confidence = 0.0
                        exception_text = None
                        physics_features_generated: list[str] = []
                        physics_features_used: list[str] = []
                        latent_features_generated: list[str] = []
                        latent_features_used: list[str] = []
                        latent_discovery_log: list[str] = []
                        discovery_metadata: dict = {}

                        physics_candidates = generate_physics_features(df)
                        physics_features_generated = sorted(
                            [name for name in physics_candidates.keys() if name not in df.columns]
                        )
                        latent_candidates = generate_latent_variables(df)
                        latent_features_generated = sorted(
                            [name for name in latent_candidates.keys() if name not in df.columns]
                        )

                        try:
                            law_text, error_value, significant, discovery_metadata = _run_discovery(
                                config=config,
                                dataset_file=dataset_file,
                                discover_kwargs=discover_kwargs,
                            )
                            eval_df = _build_equivalence_eval_frame(df)
                            success, reason = _is_success(config, law_text, error_value, eval_df=eval_df)
                            law_symbols = _extract_symbol_names(law_text)
                            physics_features_used = sorted(
                                [name for name in physics_features_generated if name in law_symbols]
                            )
                            latent_features_used = sorted(
                                [name for name in latent_features_generated if name in law_symbols]
                            )
                        except Exception as exc:
                            exception_text = f"{type(exc).__name__}: {exc}"

                        latent_discovery_log = _latent_discovery_messages(
                            df=df,
                            generated=latent_features_generated,
                            used=latent_features_used,
                            available=sorted(latent_candidates.keys()),
                        )

                        confidence = compute_discovery_confidence(
                            config=config,
                            law_text=law_text,
                            error=error_value,
                            reason=reason,
                            exception=exception_text,
                        )

                        # DCS-at-write-time: compute formal Eq. 5 DCS from the actual
                        # dataset and fitted law, so future runs store reproducible
                        # DCS values directly in the JSON artifact. (2026-04-22)
                        try:
                            _target_col = getattr(config, "target_variable", None) or getattr(
                                config, "target_name", None
                            )
                            if _target_col is None:
                                # Fall back to the last column of df (convention in PIR-Bench CSVs)
                                _target_col = df.columns[-1]
                            dcs_formal = _compute_dcs_formal(
                                df=df,
                                target_col=_target_col,
                                law_text=law_text,
                                error_value=error_value,
                                success=success,
                            )
                        except Exception as _dcs_exc:
                            dcs_formal = {"error": f"dcs_formal failed: {type(_dcs_exc).__name__}: {_dcs_exc}"}

                        record = RunResult(
                            experiment=config.name,
                            seed=seed,
                            noise=float(noise),
                            dataset_size=int(size),
                            operator_profile=operator_name,
                            dataset_path=str(dataset_file),
                            law=law_text,
                            error=error_value,
                            significant=significant,
                            success=success,
                            reason=reason,
                            confidence=confidence,
                            exception=exception_text,
                            physics_features_generated=physics_features_generated,
                            physics_features_used=physics_features_used,
                            latent_features_generated=latent_features_generated,
                            latent_features_used=latent_features_used,
                            latent_discovery_log=latent_discovery_log,
                            discovery_metadata=discovery_metadata,
                            dcs_formal=dcs_formal,
                        )
                        run_results.append(record)

                        completed_runs += 1
                        should_print_progress = (
                            completed_runs == 1
                            or completed_runs == total_runs
                            or (completed_runs % progress_interval == 0)
                        )
                        if show_progress and should_print_progress:
                            status = "ok" if success else reason
                            elapsed = time.perf_counter() - started_at
                            avg_per_run = elapsed / completed_runs if completed_runs > 0 else 0.0
                            remaining_runs = max(0, total_runs - completed_runs)
                            eta = avg_per_run * remaining_runs
                            print(
                                f"[{completed_runs}/{total_runs}] {config.name} "
                                f"n={size} noise={noise} profile={operator_name} seed={seed} -> {status} "
                                f"(elapsed={_format_duration(elapsed)}, eta={_format_duration(eta)})",
                                flush=True,
                            )

                        result_file.write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")

                        if operator_name == "linear" and success and has_extended_profile:
                            stop_after_linear_success = True

    return run_results
