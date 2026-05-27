import numpy as np
import pandas as pd

from physics_engine.utils.derivatives import first_derivative, second_derivative


def _simulate_orbit_xy(steps: int = 1500, dt: float = 0.01, gm: float = 1.0, seed: int = 123, velocity_noise: float = 1e-4):
    rng = np.random.default_rng(seed)

    x, y = 1.0, 0.0
    vx, vy = 0.0, 1.0 + rng.normal(0.0, velocity_noise)

    rows = []
    for _ in range(steps):
        r = np.sqrt(x**2 + y**2)
        ax = -gm * x / (r**3)
        ay = -gm * y / (r**3)

        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt

        rows.append((x, y))

    return np.asarray(rows, dtype=float)


def _build_orbit_features(steps: int = 1500, dt: float = 0.01, gm: float = 1.0, seed: int = 123, velocity_noise: float = 1e-4):
    xy = _simulate_orbit_xy(steps=steps, dt=dt, gm=gm, seed=seed, velocity_noise=velocity_noise)
    x = xy[:, 0]
    y = xy[:, 1]
    t = np.arange(len(x), dtype=float) * dt

    ax = second_derivative(x, t)
    ay = second_derivative(y, t)

    r = np.sqrt(x**2 + y**2)
    inv_r3 = 1.0 / np.maximum(r**3, 1e-9)

    return pd.DataFrame(
        {
            "x_over_r3": x * inv_r3,
            "y_over_r3": y * inv_r3,
            "ax": ax,
            "ay": ay,
        }
    )


def generate_orbit_ax_dataset(steps: int = 1500, dt: float = 0.01, gm: float = 1.0, seed: int = 123, noise_std: float = 0.0):
    df = _build_orbit_features(steps=steps, dt=dt, gm=gm, seed=seed)
    out = df[["x_over_r3", "ax"]].copy()
    if noise_std > 0:
        rng = np.random.default_rng(seed)
        out["ax"] = out["ax"] + rng.normal(0.0, noise_std, len(out))
    return out


def generate_orbit_ay_dataset(steps: int = 1500, dt: float = 0.01, gm: float = 1.0, seed: int = 123, noise_std: float = 0.0):
    df = _build_orbit_features(steps=steps, dt=dt, gm=gm, seed=seed)
    out = df[["y_over_r3", "ay"]].copy()
    if noise_std > 0:
        rng = np.random.default_rng(seed)
        out["ay"] = out["ay"] + rng.normal(0.0, noise_std, len(out))
    return out
