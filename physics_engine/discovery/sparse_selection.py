def select_top_terms(coefficients, basis_terms, max_terms=5, threshold=1e-10):
    ranked = [
        (abs(float(coef)), coef, term)
        for coef, term in zip(coefficients, basis_terms)
        if abs(float(coef)) > threshold
    ]
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [(coef, term) for _, coef, term in ranked[:max_terms]]
