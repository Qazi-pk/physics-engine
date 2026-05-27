from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


def simulate(model, initial_state, t_span, steps: int = 500) -> dict:
    t_eval = np.linspace(t_span[0], t_span[1], int(steps))

    sol = solve_ivp(
        fun=model.derivatives,
        t_span=t_span,
        y0=initial_state,
        t_eval=t_eval,
    )

    return {
        "t": sol.t,
        "states": sol.y,
    }
