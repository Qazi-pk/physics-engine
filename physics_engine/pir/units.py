UNIT_DIMENSIONS = {
    "kg": (1, 0, 0),
    "m": (0, 1, 0),
    "s": (0, 0, 1),
    "N": (1, 1, -2),
    "J": (1, 2, -2),
}


def dimension_signature(dim):
    return f"M^{dim[0]} L^{dim[1]} T^{dim[2]}"
