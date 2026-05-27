from __future__ import annotations

import numpy as np

from physics_engine.core.dataset import Dataset
from physics_engine.core.model import PhysicsModel
from physics_engine.discovery.parameter_estimation import estimate_parameters
from physics_engine.simulation.integrators import simulate


def run_system_identification(
    dataset: Dataset,
    model: PhysicsModel,
    initial_parameters: dict[str, float],
    observed_variables: list[str] | None = None,
    bounds: tuple[dict[str, float], dict[str, float]] | None = None,
) -> dict:
    fit_result = estimate_parameters(
        model=model,
        dataset=dataset,
        initial_parameters=initial_parameters,
        observed_variables=observed_variables,
        bounds=bounds,
    )

    fitted_model = PhysicsModel(
        variables=model.variables,
        parameters=fit_result.estimated_parameters,
        equation_function=model.equation_function,
    )

    if dataset.time is None:
        raise ValueError("Dataset.time is required for system identification pipeline.")

    time = np.asarray(dataset.time, dtype=float)
    initial_state = [float(np.asarray(dataset.get(name), dtype=float)[0]) for name in model.variables]
    sim = simulate(
        model=fitted_model,
        initial_state=initial_state,
        t_span=(float(time[0]), float(time[-1])),
        steps=len(time),
    )

    observed_variables = list(observed_variables or model.variables)
    var_index = {name: i for i, name in enumerate(model.variables)}

    per_variable_mse: dict[str, float] = {}
    for var in observed_variables:
        i = var_index[var]
        true = np.asarray(dataset.get(var), dtype=float)
        pred = np.asarray(sim["states"][i], dtype=float)
        per_variable_mse[var] = float(np.mean((pred - true) ** 2))

    overall_mse = float(np.mean(list(per_variable_mse.values()))) if per_variable_mse else 0.0

    return {
        "estimated_parameters": fit_result.estimated_parameters,
        "fit_success": fit_result.success,
        "fit_message": fit_result.message,
        "fit_mse": fit_result.mse,
        "simulation": sim,
        "per_variable_mse": per_variable_mse,
        "overall_mse": overall_mse,
    }
