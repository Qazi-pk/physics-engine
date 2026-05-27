from __future__ import annotations

import numpy as np


def is_stable(jacobian):
    eigenvalues = np.linalg.eigvals(jacobian)
    return bool(np.all(np.real(eigenvalues) < 0))
