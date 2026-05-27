from __future__ import annotations

import numpy as np


def is_conserved(quantity, tolerance: float = 1e-3):
    dqdt = np.gradient(quantity)
    return float(np.mean(np.abs(dqdt))) < float(tolerance)
