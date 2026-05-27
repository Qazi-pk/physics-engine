import numpy as np
import pandas as pd


def generate_planar_robot_kinematics_dataset(
    num_samples: int = 500,
    l1: float = 1.0,
    l2: float = 0.7,
    noise_std: float = 0.0,
    seed: int = 123,
):
    rng = np.random.default_rng(seed)

    theta1 = rng.uniform(-np.pi, np.pi, num_samples)
    theta2 = rng.uniform(-np.pi, np.pi, num_samples)

    x = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
    y = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2)

    if noise_std > 0:
        x = x + rng.normal(0.0, noise_std, num_samples)
        y = y + rng.normal(0.0, noise_std, num_samples)

    return pd.DataFrame(
        {
            "theta1": theta1,
            "theta2": theta2,
            "l1": np.full(num_samples, l1, dtype=float),
            "l2": np.full(num_samples, l2, dtype=float),
            "x": x,
            "y": y,
        }
    )


def generate_planar_robot_jacobian_dataset(
    num_samples: int = 500,
    l1: float = 1.0,
    l2: float = 0.7,
    noise_std: float = 0.0,
    seed: int = 123,
):
    rng = np.random.default_rng(seed)

    theta1 = rng.uniform(-np.pi, np.pi, num_samples)
    theta2 = rng.uniform(-np.pi, np.pi, num_samples)

    j11 = -l1 * np.sin(theta1) - l2 * np.sin(theta1 + theta2)
    j12 = -l2 * np.sin(theta1 + theta2)
    j21 = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
    j22 = l2 * np.cos(theta1 + theta2)

    if noise_std > 0:
        j11 = j11 + rng.normal(0.0, noise_std, num_samples)
        j12 = j12 + rng.normal(0.0, noise_std, num_samples)
        j21 = j21 + rng.normal(0.0, noise_std, num_samples)
        j22 = j22 + rng.normal(0.0, noise_std, num_samples)

    return pd.DataFrame(
        {
            "theta1": theta1,
            "theta2": theta2,
            "l1": np.full(num_samples, l1, dtype=float),
            "l2": np.full(num_samples, l2, dtype=float),
            "J11": j11,
            "J12": j12,
            "J21": j21,
            "J22": j22,
        }
    )
