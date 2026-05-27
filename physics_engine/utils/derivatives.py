import numpy as np
import pandas as pd


def _validate_inputs(values, times):
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)

    if values.ndim != 1 or times.ndim != 1:
        raise ValueError("values and times must be 1D arrays")

    if values.shape != times.shape:
        raise ValueError("values and times must have the same shape")

    if len(values) < 3:
        raise ValueError("At least 3 samples are required for stable derivatives")

    return values, times


def first_derivative(values, times):
    values, times = _validate_inputs(values, times)
    return np.gradient(values, times, edge_order=2)


def second_derivative(values, times):
    values, times = _validate_inputs(values, times)
    first = np.gradient(values, times, edge_order=2)
    return np.gradient(first, times, edge_order=2)


def compute_second_derivative(values, dt):
    values = np.asarray(values, dtype=float)
    first = np.gradient(values, dt, edge_order=2)
    return np.gradient(first, dt, edge_order=2)


def prepare_orbit_discovery_dataset(df: pd.DataFrame, dt: float):
    data = df.copy()
    data["ax"] = compute_second_derivative(data["x"].to_numpy(dtype=float), dt)
    data["ay"] = compute_second_derivative(data["y"].to_numpy(dtype=float), dt)
    return data
