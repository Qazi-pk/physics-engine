import numpy as np
import pandas as pd


def generate_harmonic_oscillator_dataset(
    steps: int = 2000,
    dt: float = 0.01,
    omega: float = 1.0,
    noise_std: float = 0.0,
    seed: int = 123,
):
    rng = np.random.default_rng(seed)

    x = 1.0
    v = 0.0
    rows = []

    for _ in range(steps):
        x_dot = v
        v_dot = -(omega**2) * x

        if noise_std > 0.0:
            x_dot = x_dot + rng.normal(0.0, noise_std)
            v_dot = v_dot + rng.normal(0.0, noise_std)

        rows.append((x, v, x_dot, v_dot))

        v = v + (-(omega**2) * x) * dt
        x = x + v * dt

    return pd.DataFrame(rows, columns=["x", "v", "x_dot", "v_dot"])


def generate_harmonic_oscillator_lagrangian_dataset(
    steps: int = 2000,
    dt: float = 0.01,
    omega: float = 1.0,
    noise_std: float = 0.0,
    seed: int = 123,
):
    base = generate_harmonic_oscillator_dataset(
        steps=steps,
        dt=dt,
        omega=omega,
        noise_std=noise_std,
        seed=seed,
    )

    q = base["x"].to_numpy(dtype=float)
    dqdt = base["v"].to_numpy(dtype=float)
    d2qdt2 = base["v_dot"].to_numpy(dtype=float)
    lagrangian = 0.5 * (dqdt**2) - 0.5 * (omega**2) * (q**2)

    return pd.DataFrame(
        {
            "q": q,
            "dqdt": dqdt,
            "d2qdt2": d2qdt2,
            "L": lagrangian,
        }
    )


def generate_harmonic_oscillator_hamiltonian_dataset(
    steps: int = 2000,
    dt: float = 0.01,
    omega: float = 1.0,
    noise_std: float = 0.0,
    seed: int = 123,
):
    base = generate_harmonic_oscillator_dataset(
        steps=steps,
        dt=dt,
        omega=omega,
        noise_std=noise_std,
        seed=seed,
    )

    q = base["x"].to_numpy(dtype=float)
    p = base["v"].to_numpy(dtype=float)
    dqdt = p
    dpdt = base["v_dot"].to_numpy(dtype=float)
    hamiltonian = 0.5 * (p**2) + 0.5 * (omega**2) * (q**2)

    return pd.DataFrame(
        {
            "q": q,
            "p": p,
            "dqdt": dqdt,
            "dpdt": dpdt,
            "H": hamiltonian,
        }
    )
