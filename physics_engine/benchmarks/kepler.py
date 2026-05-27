import numpy as np
import pandas as pd


def generate_kepler_third_law_dataset(num_samples=1200, noise_std=0.0, seed=123):
    rng = np.random.default_rng(seed)
    r = rng.uniform(0.4, 6.0, size=int(num_samples))
    T = np.sqrt(r**3)

    if noise_std > 0:
        T = T + rng.normal(0.0, float(noise_std), size=len(T))

    return pd.DataFrame({"r": r, "T": T})


def generate_inverse_square_acceleration_dataset(num_samples=1200, noise_std=0.0, seed=123):
    rng = np.random.default_rng(seed)
    r = rng.uniform(0.4, 6.0, size=int(num_samples))
    a = 1.0 / (r**2)

    if noise_std > 0:
        a = a + rng.normal(0.0, float(noise_std), size=len(a))

    return pd.DataFrame(
        {
            "r": r,
            "a": a,
            "inv_r": 1.0 / r,
            "inv_r2": 1.0 / (r**2),
            "inv_r3": 1.0 / (r**3),
        }
    )


def generate_kepler_dataset(steps=4000, dt=0.01, seed=123):
    """
    Backward-compatible legacy Kepler generator used by older scripts.
    This produces an inverse-square acceleration proxy dataset.
    """
    rng = np.random.default_rng(seed)
    g_const = 1.0
    mass = 1.0

    x, y = 1.0, 0.0
    vx, vy = 0.0, 1.0 + rng.normal(0, 1e-4)

    rows = []
    for _ in range(steps):
        r = np.sqrt(x ** 2 + y ** 2)
        ax = -g_const * mass * x / (r ** 3)
        ay = -g_const * mass * y / (r ** 3)
        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt
        r = np.sqrt(x ** 2 + y ** 2)
        a = np.sqrt(ax ** 2 + ay ** 2)
        rows.append((r, a, 1.0 / r, 1.0 / (r ** 2), 1.0 / (r ** 3)))

    return pd.DataFrame(rows, columns=["r", "a", "inv_r", "inv_r2", "inv_r3"])
