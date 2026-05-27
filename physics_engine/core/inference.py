from sympy import solve


PRIORITY_SYMBOLS = {"v", "F", "a", "T", "t", "E", "m", "u"}


def infer_goal(equations, known_values):
    unknowns = set()

    for eq in equations:
        symbols = eq.free_symbols
        for sym in symbols:
            if sym not in known_values:
                unknowns.add(sym)

    return list(unknowns)


def rank_goals(equations, known_values, goals=None):
    candidates = goals or infer_goal(equations, known_values)
    scores = {goal: 0 for goal in candidates}

    for eq in equations:
        unknowns = [symbol for symbol in eq.free_symbols if symbol not in known_values]
        if len(unknowns) == 1 and unknowns[0] in scores:
            scores[unknowns[0]] += 2

    for goal in candidates:
        if str(goal) in PRIORITY_SYMBOLS:
            scores[goal] += 1

    return sorted(candidates, key=lambda goal: (-scores[goal], str(goal)))


def solve_equations(equations, known_values, target):
    substituted = [eq.subs(known_values) for eq in equations]

    for eq in substituted:
        try:
            sol = solve(eq, target)
            if sol:
                return sol[0]
        except Exception:
            continue

    return None


def build_dependency_graph(equations):
    graph = {}

    for eq in equations:
        lhs = list(eq.lhs.free_symbols)
        rhs = list(eq.rhs.free_symbols)

        for symbol in lhs:
            graph.setdefault(symbol, set()).update(rhs)

    return graph


def reason(equations, known_values):
    inferred_values = dict(known_values)
    updated = True

    while updated:
        updated = False

        for eq in equations:
            symbols = eq.free_symbols
            unknowns = [symbol for symbol in symbols if symbol not in inferred_values]

            if len(unknowns) == 1:
                target = unknowns[0]
                sol = solve(eq.subs(inferred_values), target)

                if sol:
                    inferred_values[target] = sol[0]
                    updated = True

    return inferred_values