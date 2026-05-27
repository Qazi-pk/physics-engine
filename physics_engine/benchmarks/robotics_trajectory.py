import numpy as np
import pandas as pd


def generate_robot_trajectory_dataset(
    num_samples: int = 500,
    l1: float = 1.0,
    l2: float = 1.0,
    seed: int = 0,
    noise_std: float = 0.0,
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
            "x": x,
            "y": y,
        }
    )
