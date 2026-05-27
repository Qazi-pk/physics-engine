import numpy as np
import pandas as pd


def generate_double_pendulum_dataset(
    steps: int = 2000,
    dt: float = 0.01,
    g: float = 9.81,
    l1: float = 1.0,
    l2: float = 1.0,
    m1: float = 1.0,
    m2: float = 1.0,
    noise_std: float = 0.0,
    seed: int = 123,
):
    rng = np.random.default_rng(seed)

    theta1 = np.pi / 2.0
    theta2 = np.pi / 2.0 + 0.1
    omega1 = 0.0
    omega2 = 0.0

    rows = []

    for _ in range(steps):
        delta = theta1 - theta2
        sin_delta = np.sin(delta)
        cos_delta = np.cos(delta)

        denom1 = l1 * (2 * m1 + m2 - m2 * np.cos(2 * theta1 - 2 * theta2))
        denom2 = l2 * (2 * m1 + m2 - m2 * np.cos(2 * theta1 - 2 * theta2))

        alpha1_num = (
            -g * (2 * m1 + m2) * np.sin(theta1)
            - m2 * g * np.sin(theta1 - 2 * theta2)
            - 2 * sin_delta * m2 * (omega2**2 * l2 + omega1**2 * l1 * cos_delta)
        )
        alpha2_num = (
            2
            * sin_delta
            * (
                omega1**2 * l1 * (m1 + m2)
                + g * (m1 + m2) * np.cos(theta1)
                + omega2**2 * l2 * m2 * cos_delta
            )
        )

        alpha1 = alpha1_num / denom1
        alpha2 = alpha2_num / denom2

        theta1_dot = omega1
        theta2_dot = omega2

        if noise_std > 0.0:
            theta1_dot = theta1_dot + rng.normal(0.0, noise_std)
            theta2_dot = theta2_dot + rng.normal(0.0, noise_std)

        rows.append(
            (
                theta1,
                theta2,
                omega1,
                omega2,
                theta1_dot,
                theta2_dot,
                alpha1,
                alpha2,
            )
        )

        omega1 = omega1 + alpha1 * dt
        omega2 = omega2 + alpha2 * dt
        theta1 = theta1 + omega1 * dt
        theta2 = theta2 + omega2 * dt

    return pd.DataFrame(
        rows,
        columns=[
            "theta1",
            "theta2",
            "omega1",
            "omega2",
            "theta1_dot",
            "theta2_dot",
            "alpha1",
            "alpha2",
        ],
    )
