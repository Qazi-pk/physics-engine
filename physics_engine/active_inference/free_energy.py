"""
free_energy.py
~~~~~~~~~~~~~~

Compute free energy (Kullback-Leibler divergence) between observed and predicted distributions.
This quantifies how well a model explains the observed data, enabling model comparison.

Free Energy (F) in active inference:
    F = KL(Q(x|o) || P(x|o))  ≈ prediction error + model complexity

In this context:
    - Prediction error: sum of squared residuals (tau_mse for torque, etc.)
    - Model complexity: number of parameters (regularization)

Usage::

    from physics_engine.active_inference import compute_free_energy
    
    # After discovery yields predicted torques
    tau_observed = np.array([...])
    tau_predicted = np.array([...])
    n_params = 8  # number of fitted coefficients
    
    F = compute_free_energy(tau_observed, tau_predicted, sigma=0.1, n_params=n_params)
    # Lower F indicates better model fit and simpler explanation
"""

import numpy as np
from typing import Union, Tuple
from dataclasses import dataclass


@dataclass
class FreeEnergyResult:
    """Result of free energy computation."""
    
    f_value: float
    """Total free energy (lower is better)."""
    
    prediction_error: float
    """Sum of squared residuals."""
    
    complexity_penalty: float
    """Model complexity cost (function of n_params)."""
    
    kl_divergence: float
    """KL divergence between model and data."""
    
    mse: float
    """Mean squared error of prediction."""


def compute_free_energy(
    observed: np.ndarray,
    predicted: np.ndarray,
    sigma: float = 1.0,
    n_params: int = 0,
    regularization: float = 1.0,
) -> FreeEnergyResult:
    """
    Compute free energy (prediction error + complexity penalty).
    
    Args:
        observed: Observed data (e.g., measured torques) of shape (n_samples,) or (n_samples, n_outputs)
        predicted: Model predictions of same shape as observed
        sigma: Noise level (std dev) for likelihood scaling
        n_params: Number of free parameters in the model (for complexity penalty)
        regularization: Weight on complexity penalty (default 1.0 means balanced trade-off)
    
    Returns:
        FreeEnergyResult with F value and components.
    
    Notes:
        Free energy approximation:
            F ≈ ||o - m||² / (2σ²) + (n_params / 2) * log(N)
        
        - First term: prediction error normalized by noise
        - Second term: model complexity via BIC-like penalty
        - Lower F indicates better balance of fit and simplicity
    """
    observed = np.asarray(observed).reshape(-1)
    predicted = np.asarray(predicted).reshape(-1)
    
    if observed.shape != predicted.shape:
        raise ValueError(
            f"observed and predicted shapes must match: {observed.shape} vs {predicted.shape}"
        )
    
    n_samples = len(observed)
    
    # Prediction error (sum of squared residuals)
    residuals = observed - predicted
    sse = np.sum(residuals ** 2)
    mse = sse / n_samples
    
    # KL divergence component (normalized prediction error)
    kl_divergence = sse / (2 * sigma ** 2)
    
    # Complexity penalty (BIC-like: penalize number of parameters)
    # Use log(n_samples) to scale complexity with data size
    if n_samples > 1:
        bic_penalty = (n_params / 2) * np.log(n_samples)
    else:
        bic_penalty = (n_params / 2) * 1.0
    
    complexity_penalty = regularization * bic_penalty
    
    # Total free energy
    f_value = kl_divergence + complexity_penalty
    
    return FreeEnergyResult(
        f_value=float(f_value),
        prediction_error=float(sse),
        complexity_penalty=float(complexity_penalty),
        kl_divergence=float(kl_divergence),
        mse=float(mse),
    )


def batch_compute_free_energy(
    observed_list: list,
    predicted_list: list,
    n_params_list: list,
    sigma: float = 1.0,
    regularization: float = 1.0,
) -> list:
    """
    Compute free energy for multiple models.
    
    Args:
        observed_list: List of observed data arrays
        predicted_list: List of predicted data arrays
        n_params_list: List of parameter counts for each model
        sigma: Noise level
        regularization: Complexity weight
    
    Returns:
        List of FreeEnergyResult objects (one per model)
    """
    results = []
    for obs, pred, n_params in zip(observed_list, predicted_list, n_params_list):
        result = compute_free_energy(obs, pred, sigma=sigma, n_params=n_params, regularization=regularization)
        results.append(result)
    return results
