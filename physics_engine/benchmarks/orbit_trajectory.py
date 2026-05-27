import numpy as np
import pandas as pd


def generate_orbit_trajectory_dataset(
    num_steps=2000,
    dt=0.01,
    GM=1.0,
    seed=0,
):
    rng = np.random.default_rng(seed)

    x = 1.0
    y = 0.0
    vx = 0.0
    vy = 1.0 + rng.normal(0.0, 0.0)

    xs, ys, vxs, vys, ts = [], [], [], [], []

    for i in range(num_steps):
        r = np.sqrt(x * x + y * y)

        ax = -GM * x / r**3
        ay = -GM * y / r**3

        vx += ax * dt
        vy += ay * dt

        x += vx * dt
        y += vy * dt

        xs.append(x)
        ys.append(y)
        vxs.append(vx)
        vys.append(vy)
        ts.append(i * dt)

    df = pd.DataFrame(
        {
            "t": ts,
            "x": xs,
            "y": ys,
            "vx": vxs,
            "vy": vys,
        }
    )

    return df