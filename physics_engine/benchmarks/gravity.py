import numpy as np
import pandas as pd


def generate_gravity_dataset(num_samples=1200, noise_std=0.01, seed=123):
    rng = np.random.default_rng(seed)
    g_const = 1.0

    m1 = rng.uniform(0.5, 5.0, num_samples)
    m2 = rng.uniform(0.5, 5.0, num_samples)
    r = rng.uniform(0.8, 5.0, num_samples)
    force = g_const * m1 * m2 / (r ** 2)
    force_noisy = force + rng.normal(0.0, noise_std, num_samples)

    return pd.DataFrame(
        {
            "m1": m1,
            "m2": m2,
            "r": r,
            "inv_r2": 1.0 / (r ** 2),
            "m1m2": m1 * m2,
            "F": force_noisy,
        }
    )
