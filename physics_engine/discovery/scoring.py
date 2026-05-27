def score_model(mse, complexity):
    """
    Penalize overly complex models.
    """

    return float(mse) + 0.01 * float(complexity)
