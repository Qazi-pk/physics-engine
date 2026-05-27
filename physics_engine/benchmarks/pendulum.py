import numpy as np
import pandas as pd


def generate_pendulum_dataset(steps=2000, dt=0.01, noise_std=0.02, seed=123):
    rng = np.random.default_rng(seed)
    g = 9.81
    length = 1.0

    theta = 0.5
    omega = 0.0
    rows = []

    for _ in range(steps):
        alpha = -(g / length) * np.sin(theta)
        alpha_noisy = alpha + rng.normal(0.0, noise_std)
        omega += alpha * dt
        theta += omega * dt
        rows.append((theta, omega, alpha_noisy))

    return pd.DataFrame(rows, columns=["theta", "omega", "alpha"])
