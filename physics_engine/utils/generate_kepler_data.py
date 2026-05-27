import numpy as np
import pandas as pd


def generate_kepler_dataset(noise_level=0.02, save_path=None):
    """
    Generate planetary dataset obeying Kepler's third law.

    T^2 = r^3  (using astronomical units)
    """

    planets = [
        ("Mercury", 0.387),
        ("Venus", 0.723),
        ("Earth", 1.000),
        ("Mars", 1.524),
        ("Jupiter", 5.203),
        ("Saturn", 9.537),
        ("Uranus", 19.191),
        ("Neptune", 30.07),
    ]

    data = []

    for name, r in planets:
        T_true = np.sqrt(r**3)

        noise = np.random.normal(0, noise_level * T_true)
        T_obs = T_true + noise

        data.append({
            "planet": name,
            "r": r,
            "T": T_obs,
        })

    df = pd.DataFrame(data)

    if save_path:
        df.to_csv(save_path, index=False)

    return df
