import numpy as np
import pandas as pd


def generate_newton_dataset(num_samples=500, noise_std=0.02, seed=123):
    rng = np.random.default_rng(seed)
    m = rng.uniform(0.5, 5.0, num_samples)
    a = rng.uniform(-3.0, 3.0, num_samples)
    force = m * a + rng.normal(0.0, noise_std, num_samples)
    return pd.DataFrame({"m": m, "a": a, "F": force})
