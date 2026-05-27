import numpy as np


def discover_power_law(x, y):
    """
    Discover power law relationship:

    y = k * x^a
    """

    logx = np.log(x)
    logy = np.log(y)

    coeffs = np.polyfit(logx, logy, 1)

    a = coeffs[0]
    k = np.exp(coeffs[1])

    return f"T = {k:.3f} * r^{a:.3f}"
