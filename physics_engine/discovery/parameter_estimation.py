from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from physics_engine.core.dataset import Dataset
from physics_engine.core.model import PhysicsModel


@dataclass(frozen=True)
class ParameterEstimationResult:
    estimated_parameters: dict[str, float]
    success: bool
    mse: float
    cost: float
    nfev: int
    message: str


class ParameterEstimator:
    def estimate_linear(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)

        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D design matrix.")
        if y_arr.ndim != 1:
            raise ValueError("y must be a 1D target array.")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have the same number of rows.")
        if X_arr.shape[0] < 2:
            raise ValueError("At least 2 samples are required for linear estimation.")

        solution, *_ = np.linalg.lstsq(X_arr, y_arr, rcond=None)
        return np.asarray(solution, dtype=float)

    def fit(
        self,
        model: PhysicsModel,
        dataset: Dataset,
        initial_parameters: dict[str, float],
        observed_variables: list[str] | None = None,
        bounds: tuple[dict[str, float], dict[str, float]] | None = None,
    ) -> ParameterEstimationResult:
        return estimate_parameters(
            model=model,
            dataset=dataset,
            initial_parameters=initial_parameters,
            observed_variables=observed_variables,
            bounds=bounds,
        )


def estimate_parameters(
    model: PhysicsModel,
    dataset: Dataset,
    initial_parameters: dict[str, float],
    observed_variables: list[str] | None = None,
    bounds: tuple[dict[str, float], dict[str, float]] | None = None,
) -> ParameterEstimationResult:
    if dataset.time is None:
        raise ValueError("Dataset.time is required for parameter estimation.")

    time = np.asarray(dataset.time, dtype=float)
    if time.ndim != 1 or len(time) < 2:
        raise ValueError("Dataset.time must be a 1D array with at least 2 samples.")

    observed_variables = list(observed_variables or model.variables)
    for var in observed_variables:
        if dataset.get(var) is None:
            raise ValueError(f"Observed variable '{var}' not found in dataset.")

    parameter_names = list(initial_parameters.keys())
    full_theta0 = np.array([float(initial_parameters[name]) for name in parameter_names], dtype=float)

    if bounds is None:
        full_lower_bounds = np.full_like(full_theta0, -np.inf, dtype=float)
        full_upper_bounds = np.full_like(full_theta0, np.inf, dtype=float)
    else:
        lower_map, upper_map = bounds
        full_lower_bounds = np.array([float(lower_map.get(name, -np.inf)) for name in parameter_names], dtype=float)
        full_upper_bounds = np.array([float(upper_map.get(name, np.inf)) for name in parameter_names], dtype=float)

    fixed_mask = np.isfinite(full_lower_bounds) & np.isfinite(full_upper_bounds) & (
        np.isclose(full_lower_bounds, full_upper_bounds)
    )
    free_mask = ~fixed_mask
    fixed_values = full_lower_bounds.copy()

    free_parameter_names = [name for i, name in enumerate(parameter_names) if free_mask[i]]
    theta0 = full_theta0[free_mask]
    lower_bounds = full_lower_bounds[free_mask]
    upper_bounds = full_upper_bounds[free_mask]

    variable_to_index = {name: idx for idx, name in enumerate(model.variables)}
    observed_indices = [variable_to_index[name] for name in observed_variables]
    observed_data = np.column_stack([np.asarray(dataset.get(name), dtype=float) for name in observed_variables])

    initial_state = [float(np.asarray(dataset.get(name), dtype=float)[0]) for name in model.variables]
    t_span = (float(time[0]), float(time[-1]))

    def _compose_full_theta(theta_free: np.ndarray) -> np.ndarray:
        theta_full = full_theta0.copy()
        theta_full[fixed_mask] = fixed_values[fixed_mask]
        theta_full[free_mask] = theta_free
        return theta_full

    def residuals(theta_free: np.ndarray) -> np.ndarray:
        theta_full = _compose_full_theta(theta_free)
        params = {name: float(value) for name, value in zip(parameter_names, theta_full)}

        def _rhs(t, state):
            return model.equation_function(state, t, params)

        solution = solve_ivp(
            fun=_rhs,
            t_span=t_span,
            y0=initial_state,
            t_eval=time,
            rtol=1e-9,
            atol=1e-9,
        )

        if (not solution.success) or solution.y.shape[1] != len(time):
            return np.full(observed_data.size, 1e6, dtype=float)

        simulated = solution.y[observed_indices, :].T
        return (simulated - observed_data).reshape(-1)

    if theta0.size == 0:
        resid = residuals(theta0)
        mse = float(np.mean(resid**2)) if resid.size > 0 else 0.0
        fitted_params = {
            name: float(value)
            for name, value in zip(parameter_names, _compose_full_theta(theta0))
        }
        return ParameterEstimationResult(
            estimated_parameters=fitted_params,
            success=True,
            mse=mse,
            cost=float(0.5 * np.sum(resid**2)),
            nfev=0,
            message="No free parameters to optimize.",
        )

    start_points: list[np.ndarray] = [theta0]
    finite_bounds = np.isfinite(lower_bounds) & np.isfinite(upper_bounds)
    if np.any(finite_bounds):
        if theta0.size == 1 and finite_bounds[0]:
            grid = np.linspace(lower_bounds[0], upper_bounds[0], 7)
            start_points.extend(np.array([g], dtype=float) for g in grid)
        else:
            rng = np.random.default_rng(123)
            for _ in range(8):
                candidate = theta0.copy()
                for i in range(theta0.size):
                    if finite_bounds[i]:
                        candidate[i] = rng.uniform(lower_bounds[i], upper_bounds[i])
                start_points.append(candidate)

    best_result = None
    best_cost = np.inf
    for x0 in start_points:
        trial = least_squares(
            fun=residuals,
            x0=x0,
            bounds=(lower_bounds, upper_bounds),
        )
        if float(trial.cost) < best_cost:
            best_cost = float(trial.cost)
            best_result = trial

    result = best_result
    if result is None:
        raise RuntimeError("Parameter estimation failed to produce an optimization result.")

    theta_full = _compose_full_theta(result.x)
    fitted_params = {name: float(value) for name, value in zip(parameter_names, theta_full)}
    resid = residuals(result.x)
    mse = float(np.mean(resid**2)) if resid.size > 0 else 0.0

    return ParameterEstimationResult(
        estimated_parameters=fitted_params,
        success=bool(result.success),
        mse=mse,
        cost=float(result.cost),
        nfev=int(result.nfev),
        message=str(result.message),
    )
