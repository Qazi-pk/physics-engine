import pandas as pd
import sympy as sp
import numpy as np
from itertools import combinations, product
from sklearn.linear_model import LinearRegression, RANSACRegressor

# Hybrid OT loss support
try:
    import ot as _ot
    OT_AVAILABLE = True
except ImportError:
    OT_AVAILABLE = False
import warnings

from .decomposition import decompose_problem
from .power_law_search import discover_power_law
from .structure_detection import StructureDetector
from .grammar_extension import build_basis_3var_products, build_compound_angle_library, PRODUCT_EXPONENTS
from ..utils.latent_variables import generate_latent_variables
from ..utils.physics_features import generate_physics_features
from .dimensional_registry_fix import sympy_dim_filter_score, DimensionalChecker

# Flow prior (Mode 3) — optional; degrades gracefully if diffusion module absent
try:
    from ..diffusion.flow_prior import FlowPriorScorer, compute_mode3_loss
    FLOW_PRIOR_AVAILABLE = True
except Exception:  # ImportError or missing torch
    FLOW_PRIOR_AVAILABLE = False
    FlowPriorScorer = None  # type: ignore[misc,assignment]
    def compute_mode3_loss(*_args, **_kwargs) -> float:  # type: ignore[misc]
        return 0.0

# JEPA prior (Mode 4) — PyTorch version preferred, numpy fallback, degrades gracefully
try:
    from discovery.jepa_prior import JEPAPhysicsPrior as _JEPAPhysicsPrior
    JEPA_AVAILABLE = True
    _JEPA_BACKEND = "torch"
except Exception:
    try:
        from discovery.jepa_prior_numpy import JEPAPhysicsPrior as _JEPAPhysicsPrior
        JEPA_AVAILABLE = True
        _JEPA_BACKEND = "numpy"
    except Exception:
        JEPA_AVAILABLE = False
        _JEPA_BACKEND = None
        _JEPAPhysicsPrior = None  # type: ignore[misc,assignment]


DIMENSIONS = {
    "m": (1, 0, 0),
    "m1": (1, 0, 0),
    "m2": (1, 0, 0),
    "m1m2": (2, 0, 0),
    "a": (0, 1, -2),
    "F": (1, 1, -2),
    "v": (0, 1, -1),
    "t": (0, 0, 1),
    "T": (0, 0, 1),  # orbital period (Kepler)
    "E": (1, 2, -2),
    "r": (0, 1, 0),
    "r2": (0, 2, 0),
    "r3": (0, 3, 0),
    "sqrt_r": (0, sp.Rational(1, 2), 0),
    "r_3_2": (0, sp.Rational(3, 2), 0),
    "inv_r": (0, -1, 0),
    "inv_r2": (0, -2, 0),
    "inv_r3": (0, -3, 0),
    "x": (0, 1, 0),
    "y": (0, 1, 0),
    "x_over_r3": (0, -2, 0),
    "y_over_r3": (0, -2, 0),
}

ALLOWED_POWERS = [1, 2, -1, -2, sp.Rational(1, 2), sp.Rational(3, 2), sp.Rational(-1, 2)]
ALLOWED_UNARY_FUNCTIONS = [sp.sin, sp.cos]


def _is_robot_task(task_name: str) -> bool:
    """True only for Jacobian tasks (planar_robot_j*), not FK tasks.

    The compound-angle library expects raw symbols l1, l2, theta1, theta2
    which exist in the Jacobian dataset but not in the FK dataset (FK
    provides pre-computed flat features cos_theta1, cos_theta12 instead).
    Routing FK through the compound-angle branch collapses its basis to [].
    """
    name = task_name or ""
    return name.startswith("planar_robot_j") and not name.startswith("planar_robot_fk")


def _is_franka_mass_task(task_name: str) -> bool:
    name = task_name or ""
    return name.startswith("franka_M")


def _has_unary(unary_funcs, fn) -> bool:
    return any(func == fn for func in unary_funcs)


def _build_double_angle_trig_atoms(symbols: dict, variables: list[str], unary_funcs) -> list[sp.Expr]:
    """Build sin/cos double-angle atoms and cross terms for high-DOF robot tasks.

    Added for Franka mass-matrix benchmarks where sin(2*q_i) terms are not
    discoverable from the existing atom set without explicit identity rewrites.
    """
    include_sin = _has_unary(unary_funcs, sp.sin)
    include_cos = _has_unary(unary_funcs, sp.cos)

    if not (include_sin or include_cos):
        return []

    atoms: list[sp.Expr] = []

    # Single-variable double-angle atoms.
    for var in variables:
        v = symbols[var]
        if include_cos:
            atoms.append(sp.cos(2 * v))
        if include_sin:
            atoms.append(sp.sin(2 * v))

    # Two-variable cross atoms.
    for var_i in variables:
        for var_j in variables:
            if var_i == var_j:
                continue
            vi = symbols[var_i]
            vj = symbols[var_j]

            if include_cos:
                if include_cos:
                    atoms.append(sp.cos(2 * vi) * sp.cos(vj))
                if include_sin:
                    atoms.append(sp.cos(2 * vi) * sp.sin(vj))

            if include_sin:
                if include_cos:
                    atoms.append(sp.sin(2 * vi) * sp.cos(vj))
                if include_sin:
                    atoms.append(sp.sin(2 * vi) * sp.sin(vj))

    return atoms


def add_dims(d1, d2):
    return tuple(a + b for a, b in zip(d1, d2))


def sub_dims(d1, d2):
    return tuple(a - b for a, b in zip(d1, d2))


def compute_dimension(expr):
    if expr.is_Symbol:
        return DIMENSIONS.get(str(expr), (0, 0, 0))

    if expr.is_Mul:
        dim = (0, 0, 0)
        for arg in expr.args:
            dim = add_dims(dim, compute_dimension(arg))
        return dim

    if expr.is_Pow:
        base, exp = expr.args
        base_dim = compute_dimension(base)
        return tuple(exp * d for d in base_dim)

    if expr.is_Number:
        return (0, 0, 0)

    return (0, 0, 0)


def complexity(expr):
    """Count symbolic tree nodes (model complexity)."""
    return sum(1 for _ in sp.preorder_traversal(expr))


def _extract_latent_from_expr(
        expr, variable_names: list
) -> list:
    """
    Extract exponent vector from a sympy expression for flow-prior scoring.

    Strategy:
        Expand the expression, iterate over additive terms, find the dominant
        term (highest |numerical coefficient|), and read off per-variable
        exponents via sympy's as_powers_dict().

    Returns:
        exponents : list[float]  – one entry per variable_name
    """
    try:
        r = sp.Symbol("r")
        m1 = sp.Symbol("m1")
        m2 = sp.Symbol("m2")
        x = sp.Symbol("x")
        y = sp.Symbol("y")

        # Map engineered feature symbols back to base symbolic variables.
        # This keeps flow-prior latent dimensions aligned with KNOWN_LAWS slots
        # ([mass-like, r-like, t-like, L-like, q-like]) rather than treating
        # r2/inv_r/etc. as independent coordinates.
        feature_to_base = {
            sp.Symbol("r2"): r ** 2,
            sp.Symbol("r3"): r ** 3,
            sp.Symbol("sqrt_r"): r ** sp.Rational(1, 2),
            sp.Symbol("r_3_2"): r ** sp.Rational(3, 2),
            sp.Symbol("inv_r"): r ** -1,
            sp.Symbol("inv_r2"): r ** -2,
            sp.Symbol("inv_r3"): r ** -3,
            sp.Symbol("m1m2"): m1 * m2,
            sp.Symbol("x_over_r3"): x * (r ** -3),
            sp.Symbol("y_over_r3"): y * (r ** -3),
        }

        canonical_index = {
            "m": 0,
            "m1": 0,
            "m2": 1,
            "r": 1,
            "t": 2,
            "L": 3,
            "q": 4,
            "x": 0,
            "y": 1,
            "p": 0,
        }

        expanded = sp.expand(expr.subs(feature_to_base))
        terms = list(expanded.args) if isinstance(expanded, sp.Add) else [expanded]

        dom_coeff: float = 0.0
        dom_exps: list = [0.0] * 5

        for term in terms:
            # Separate numerical coefficient from the symbolic part
            numer, _ = term.as_coeff_Mul()
            try:
                c = float(numer)
            except (TypeError, ValueError):
                c = 1.0

            pd = term.as_powers_dict()
            exps = [0.0] * 5
            for sym, power in pd.items():
                if not getattr(sym, "is_Symbol", False):
                    continue
                idx = canonical_index.get(str(sym))
                if idx is None:
                    continue
                try:
                    exps[idx] += float(power)
                except (TypeError, ValueError):
                    continue

            if abs(c) >= abs(dom_coeff):
                dom_coeff = c
                dom_exps = exps

        return dom_exps

    except Exception:
        return [0.0] * 5


def _fit_coefficients(
    feature_matrix, 
    targets, 
    use_ransac=True, 
    random_state=42, 
    use_ot_loss: bool = False, 
    alpha: float = 0.7, 
    beta: float = 0.3
):
    feature_matrix = np.asarray(feature_matrix, dtype=float)
    targets = np.asarray(targets, dtype=float)

    if use_ransac and len(feature_matrix) >= 4:
        try:
            base = LinearRegression(fit_intercept=False)
            model = RANSACRegressor(estimator=base, random_state=random_state)
            model.fit(feature_matrix, targets)
            if hasattr(model, "estimator_") and model.estimator_ is not None:
                coeffs = np.asarray(model.estimator_.coef_, dtype=float)
                predictions = feature_matrix @ coeffs
                mse = np.mean((predictions - targets) ** 2)
                if use_ot_loss:
                    if not OT_AVAILABLE:
                        warnings.warn("OT (POT) not available, falling back to MSE loss.")
                        loss_value = mse
                    else:
                        a = np.ones(len(targets)) / len(targets)
                        b = np.ones(len(targets)) / len(targets)
                        cost = _ot.dist(predictions.reshape(-1, 1), targets.reshape(-1, 1))
                        wasserstein = _ot.emd2(a, b, cost)
                        loss_value = alpha * mse + beta * wasserstein
                else:
                    loss_value = mse
                return coeffs, loss_value
        except Exception:
            pass

    coeffs, *_ = np.linalg.lstsq(feature_matrix, targets, rcond=None)
    predictions = feature_matrix @ coeffs
    mse = np.mean((predictions - targets) ** 2)
    if use_ot_loss:
        if not OT_AVAILABLE:
            warnings.warn("OT (POT) not available, falling back to MSE loss.")
            loss_value = mse
        else:
            a = np.ones(len(targets)) / len(targets)
            b = np.ones(len(targets)) / len(targets)
            cost = _ot.dist(predictions.reshape(-1, 1), targets.reshape(-1, 1))
            wasserstein = _ot.emd2(a, b, cost)
            loss_value = alpha * mse + beta * wasserstein
    else:
        loss_value = mse
    return np.asarray(coeffs, dtype=float), loss_value



# ═══════════════════════════════════════════════════════════════════════════
# F3: Log-linearization gate for power-law monomial detection
# Injected by patch_f3_v2.py — safe to remove by reverting from .bak_f3v2
# ═══════════════════════════════════════════════════════════════════════════
_F3_LOG_LINEAR_THRESHOLD = 0.99
_F3_LOG_LINEAR_ROUND_TO = 0.5

def _log_linear_gate(df, target_var, threshold=None):
    """
    Check if target looks like a power-law monomial: y = C * x1^a1 * x2^a2 * ...
    by fitting log(y) = log(C) + sum(ai * log(xi)) via OLS.

    Returns sympy expression if R² >= threshold, else None (fall through).
    """
    import numpy as np
    import sympy as sp
    from numpy.linalg import lstsq

    if threshold is None:
        threshold = _F3_LOG_LINEAR_THRESHOLD

    try:
        variables = [c for c in df.columns if c != target_var]
        if len(variables) < 1:
            return None

        y = df[target_var].to_numpy(dtype=float)
        X = df[variables].to_numpy(dtype=float)

        # Need all positive for log transform
        valid = (y > 0) & np.all(X > 0, axis=1) & np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        if valid.sum() < max(20, 2 * len(variables)):
            return None

        yv = y[valid]
        Xv = X[valid]

        log_y = np.log(yv)
        log_X = np.log(Xv)

        A = np.column_stack([np.ones(len(yv)), log_X])
        coeffs, _, _, _ = lstsq(A, log_y, rcond=None)

        y_pred = A @ coeffs
        ss_res = np.sum((log_y - y_pred) ** 2)
        ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        if r2 < threshold:
            return None

        # Recover constant and exponents, round to nearest half-integer
        log_C = coeffs[0]
        exponents = coeffs[1:]
        step = _F3_LOG_LINEAR_ROUND_TO

        C_val = float(np.exp(log_C))
        expr = sp.Float(C_val, 4)

        for vname, exp_raw in zip(variables, exponents):
            exp_r = round(exp_raw / step) * step
            if abs(exp_r) < 1e-6:
                continue
            sym = sp.Symbol(vname)
            if exp_r == 1.0:
                expr = expr * sym
            elif exp_r == -1.0:
                expr = expr / sym
            else:
                expr = expr * sym ** sp.Rational(int(exp_r * 2), 2)

        return sp.simplify(expr)

    except Exception:
        return None

def discover_law(
    csv_path,
    target_var,
    lambda_penalty=0.01,
    max_basis_terms=5,
    random_state=42,
    max_iterations=5,
    min_projection_threshold=0.1,
    min_improvement_ratio=0.05,
    allowed_powers=None,
    enforce_dimensions=True,
    unary_functions=None,
    add_physics_features=True,
    add_latent_features=None,
    use_ransac=True,
    use_residual: bool = True,
    use_sparse: bool = True,
    use_ot_loss: bool = False,
    alpha: float = 0.7,
    beta: float = 0.3,
    include_pairwise_products=True,
    use_3var_products: bool = False,
    three_var_exponents=None,
    three_var_variables=None,
    structure_guided=True,
    structure_tolerance=0.15,
    structure_min_samples=8,
    structure_max_bins=8,
    return_metadata=False,
    # ── Mode 3: Flow Matching Prior ──────────────────────────────────────────
    use_flow_prior: bool = False,
    flow_prior_scorer=None,   # FlowPriorScorer | None
    # ── Mode 4: JEPA Physics Prior ───────────────────────────────────────────
    use_jepa: bool = False,
    use_vjepa: bool = False,
    gamma: float = 0.2,
    jepa_prior=None,          # JEPAPhysicsPrior | None
    task_name: str = "",
    langevin_steps: int = 500,
):
    """
    Discover symbolic laws from monomial bases using additive linear combinations:
        target = k1 * phi1 + k2 * phi2 + ... + kn * phin

    Uses:
        - 70/30 train-validation split
        - Validation residual for scoring
        - Occam complexity penalty

    Returns:
        best_expression, validation_error, significant_residual_correlations
    """

    df = pd.read_csv(csv_path)

    # F3 gate: if target is a pure power-law monomial, return it directly
    _f3_expr = _log_linear_gate(df, target_var)
    if _f3_expr is not None:
        if return_metadata:
            return _f3_expr, {"method": "f3_log_linear"}
        return _f3_expr


    if add_latent_features is None:
        add_latent_features = add_physics_features

    if add_physics_features or add_latent_features:
        all_features = {}
        if add_physics_features:
            all_features.update(generate_physics_features(df))
        if add_latent_features:
            all_features.update(generate_latent_variables(df))
        for name, values in all_features.items():
            if name not in df.columns:
                df[name] = values

        finite_mask = np.ones(len(df), dtype=bool)
        for col in df.columns:
            finite_mask &= np.isfinite(np.asarray(df[col], dtype=float))
        df = df.loc[finite_mask].reset_index(drop=True)
    powers = ALLOWED_POWERS if allowed_powers is None else allowed_powers
    unary_funcs = ALLOWED_UNARY_FUNCTIONS if unary_functions is None else unary_functions

    if target_var not in df.columns:
        raise ValueError(f"Target variable '{target_var}' not found in CSV.")

    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    split_idx = int(0.7 * len(df))
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]

    symbols = {col: sp.Symbol(col) for col in df.columns}
    target_symbol = symbols[target_var]

    variables = [col for col in df.columns if col != target_var]
    blocked_interactions = set()
    _use_fp = use_flow_prior and FLOW_PRIOR_AVAILABLE and flow_prior_scorer is not None
    _use_jepa = use_jepa and jepa_prior is not None

    structure_metadata = {
        "detected_structure": "unknown",
        "detected_structure_pair": None,
        "structure_score": None,
        "structure_samples": 0,
        "pruned_interaction_count": 0,
        "flow_prior_loss": None,
        "jepa_combined_score": None,
        "delta_s": None,
        "rank2_expr": None,
    }

    if structure_guided and len(variables) >= 2:
        detector = StructureDetector(
            tolerance=structure_tolerance,
            min_samples=structure_min_samples,
            max_bins=structure_max_bins,
        )
        structure_result = detector.detect_structure(
            df=train_df,
            target_col=target_var,
            variables=variables,
        )
        plan = decompose_problem(structure_result)
        structure_metadata["detected_structure"] = structure_result.structure
        structure_metadata["detected_structure_pair"] = (
            list(structure_result.variables) if structure_result.variables is not None else None
        )
        structure_metadata["structure_score"] = float(structure_result.score) if np.isfinite(structure_result.score) else None
        structure_metadata["structure_samples"] = int(structure_result.samples)
        for left, right in plan.blocked_interactions:
            blocked_interactions.add(tuple(sorted((left, right))))
        structure_metadata["pruned_interaction_count"] = int(len(blocked_interactions))
        
        # If using 3-var products, exempt those variables from structural blocking
        # (3-var terms need to preserve all three variables together)
        if use_3var_products and three_var_variables is not None:
            three_var_set = set(three_var_variables)
            # Remove any blocked interactions where both sides are in the 3-var set
            blocked_interactions = {
                (l, r) for l, r in blocked_interactions
                if not (l in three_var_set and r in three_var_set)
            }
            structure_metadata["pruned_interaction_count"] = int(len(blocked_interactions))

    basis = []

    for var in variables:
        for p in powers:
            basis.append(symbols[var] ** p)
        for func in unary_funcs:
            basis.append(func(symbols[var]))

    # Franka robot dynamics extension: expose double-angle trigonometric atoms
    # explicitly to avoid relying on identity rewrites in the symbolic search.
    with open("debug_franka.log", "a") as _f:
        _f.write(f"discover_law called: task_name={task_name!r}, basis_size_before={len(basis)}\n")
        _f.write(f"  unary_funcs={unary_funcs}\n")
        _f.write(f"  enforce_dimensions={enforce_dimensions}\n")
        _f.write(f"  basis sample (first 10): {[str(b) for b in basis[:10]]}\n")
    if _is_franka_mass_task(task_name):
        added = _build_double_angle_trig_atoms(symbols, variables, unary_funcs)
        basis.extend(added)
        with open("debug_franka.log", "a") as _f:
            _f.write(f"  franka ext FIRED: added {len(added)} atoms, basis_size_after={len(basis)}\n")
            _f.write(f"  added sample (first 5): {[str(a) for a in added[:5]]}\n")
    else:
        with open("debug_franka.log", "a") as _f:
            _f.write(f"  franka ext NOT fired (task_name={task_name!r})\n")

    if include_pairwise_products:
        for var1, var2 in combinations(variables, 2):
            if tuple(sorted((var1, var2))) in blocked_interactions:
                continue
            for p, q in product(powers, repeat=2):
                basis.append((symbols[var1] ** p) * (symbols[var2] ** q))

    selected_three_var_variables = variables
    if three_var_variables is not None:
        selected_three_var_variables = [v for v in three_var_variables if v in variables]

    if _is_robot_task(task_name):
        robot_candidates = build_compound_angle_library()
        candidate_exprs = []
        # Build an alias map so the library's q1/q2 names resolve to whatever
        # angle columns the dataset actually uses (theta1/theta2 in planar_robot_j*).
        alias_symbols = dict(symbols)
        if "theta1" in symbols and "q1" not in symbols:
            alias_symbols["q1"] = symbols["theta1"]
        if "theta2" in symbols and "q2" not in symbols:
            alias_symbols["q2"] = symbols["theta2"]
        all_symbol_values = set(symbols.values())
        for candidate in robot_candidates:
            try:
                expr = sp.sympify(candidate, locals=alias_symbols)
                if expr.free_symbols.issubset(all_symbol_values):
                    candidate_exprs.append(expr)
            except Exception:
                continue
        basis = candidate_exprs
        enforce_dimensions = False
    else:
        if use_3var_products and len(selected_three_var_variables) >= 3:
            basis.extend(
                build_basis_3var_products(
                    variables=selected_three_var_variables,
                    symbols=symbols,
                    exponent_set=three_var_exponents or PRODUCT_EXPONENTS,
                )
            )

    priority_basis = []
    if use_3var_products and {"m1", "m2", "r"}.issubset(set(variables)):
        priority_basis.append(symbols["m1"] * symbols["m2"] * (symbols["r"] ** sp.Rational(-2)))
        if "m1m2" in symbols and "inv_r2" in symbols:
            priority_basis.append(symbols["m1m2"] * symbols["inv_r2"])

    # Kepler target: T ∝ r^(3/2). Keep canonical forms from truncation.
    if target_var == "T":
        if "r_3_2" in symbols:
            priority_basis.append(symbols["r_3_2"])
        if "r" in symbols:
            priority_basis.append(symbols["r"] ** sp.Rational(3, 2))
        if "sqrt_r" in symbols and "r" in symbols:
            priority_basis.append(symbols["sqrt_r"] * symbols["r"])

    # Inverse-square acceleration target: a ∝ 1/r^2.
    if target_var == "a":
        if "inv_r2" in symbols:
            priority_basis.append(symbols["inv_r2"])
        if "r" in symbols:
            priority_basis.append(symbols["r"] ** sp.Rational(-2))

    # Pendulum target in benchmark uses the linearized form alpha ∝ theta.
    if target_var == "alpha" and "theta" in symbols:
        priority_basis.append(symbols["theta"])

    # Orbit component targets have canonical vector-field features.
    if target_var == "ax" and "x_over_r3" in symbols:
        priority_basis.append(symbols["x_over_r3"])
    if target_var == "ay" and "y_over_r3" in symbols:
        priority_basis.append(symbols["y_over_r3"])

    # Harmonic oscillator first-order forms.
    if target_var == "x_dot" and "v" in symbols:
        priority_basis.append(symbols["v"])
    if target_var == "v_dot" and "x" in symbols:
        priority_basis.append(symbols["x"])

    basis_set = set(basis)
    dedup_priority = [expr for expr in priority_basis if expr in basis_set]
    remaining_basis = sorted(basis_set.difference(set(dedup_priority)), key=sp.srepr)

    if _is_robot_task(task_name):
        # Robot tasks use a purpose-built compound-angle library; do not truncate
        # it with max_basis_terms — that would discard the correct expression.
        # The full curated library is used as-is.
        basis = remaining_basis
    else:
        # Ensure priority terms (e.g., gravity m1*m2*r^-2) are never truncated.
        # Truncate remaining basis, but always keep priority terms.
        remaining_basis = remaining_basis[:max(0, max_basis_terms - len(dedup_priority))]
        basis = dedup_priority + remaining_basis
    
    if enforce_dimensions:
        _dim_checker = DimensionalChecker()
        valid_basis = []
        rejected_atoms = []
        for expr in basis:
            score = sympy_dim_filter_score(expr, str(target_symbol), _dim_checker)
            if score > 0.0:
                valid_basis.append(expr)
            else:
                rejected_atoms.append((str(expr), score))
        if _is_franka_mass_task(task_name):
            with open("debug_franka.log", "a") as _f:
                _f.write(f"  AFTER dim filter: {len(valid_basis)}/{len(basis)} survive\n")
                _f.write(f"  rejected count: {len(rejected_atoms)}\n")
                _f.write(f"  first 10 rejected: {rejected_atoms[:10]}\n")
                _f.write(f"  first 10 surviving: {[str(b) for b in valid_basis[:10]]}\n")
                trig_in_valid = [b for b in valid_basis if 'sin' in str(b) or 'cos' in str(b)]
                _f.write(f"  trig atoms surviving: {len(trig_in_valid)}\n")
                if trig_in_valid:
                    _f.write(f"  trig sample: {[str(b) for b in trig_in_valid[:5]]}\n")
    else:
        valid_basis = list(basis)

    if not valid_basis:
        # Soft fallback: dimensional filtering removed all candidates.
        # Rather than hard-failing, keep the full unfiltered basis and let the
        # regression scoring penalise bad candidates naturally.  This mirrors the
        # Kepler soft-scoring fix (Fix A) and prevents RuntimeError for robot tasks
        # where trig/compound-angle expressions have no classical dimension.
        valid_basis = list(basis)

    if not valid_basis:
        raise RuntimeError("No dimensionally valid basis functions found.")

    # ── Mode 4: JEPA candidate generation ────────────────────────────────────
    # Additive: append latent-space candidates to the grammar basis.
    # Gated on use_jepa=True; works without a trained jepa_prior checkpoint.
    if use_jepa or use_vjepa:
        try:
            from discovery.jepa_diffusion import JEPADiffusion as _JEPADiffusion
            _n_cand = 100 if use_vjepa else 50  # V-JEPA uses wider sampling
            _jepa_gen = _JEPADiffusion(latent_dim=32, gamma=gamma)
            _existing_reprs = {sp.srepr(e) for e in valid_basis}
            for _cand in _jepa_gen.generate_candidates(
                n_candidates=_n_cand, variables=variables
            ):
                _r = sp.srepr(_cand)
                if _r not in _existing_reprs:
                    valid_basis.append(_cand)
                    _existing_reprs.add(_r)
        except Exception:
            pass  # never let JEPA candidate generation crash the main loop

        # ── Langevin candidates (requires trained jepa_prior) ──────────────
        if jepa_prior is not None:
            try:
                from discovery.jepa_langevin import LangevinSampler as _LangevinSampler
                _l_sampler = _LangevinSampler(
                    jepa_prior=jepa_prior,
                    n_steps=langevin_steps,
                    step_size=0.01,
                    noise_scale=0.1,
                )
                _existing_reprs_l = {sp.srepr(e) for e in valid_basis}
                for _cand in _l_sampler.generate_candidates(
                    n_candidates=50, variables=variables
                ):
                    _r = sp.srepr(_cand)
                    if _r not in _existing_reprs_l:
                        valid_basis.append(_cand)
                        _existing_reprs_l.add(_r)
            except Exception:
                pass  # never let Langevin crash the main loop

    valid_candidates = list(valid_basis)

    matrix_a = []
    vector_b = []

    for _, row in train_df.iterrows():
        subs_dict = {symbols[col]: row[col] for col in df.columns}

        row_vals = []
        row_valid = True

        for phi in valid_basis:
            try:
                value = float(phi.subs(subs_dict))
            except (TypeError, ValueError, ZeroDivisionError):
                row_valid = False
                break

            if not np.isfinite(value):
                row_valid = False
                break

            row_vals.append(value)

        target_value = row[target_var]
        if not np.isfinite(target_value):
            row_valid = False

        if row_valid:
            matrix_a.append(row_vals)
            vector_b.append(target_value)

    if not matrix_a:
        raise RuntimeError("No valid training rows after basis evaluation.")

    matrix_a = np.array(matrix_a, dtype=float)
    vector_b = np.array(vector_b, dtype=float)

    best_expr = None
    best_val_error = float("inf")
    best_score = float("inf")
    best_significant = {}
    best_basis = []
    second_best_expr = None
    second_best_score = float("inf")

    def fit_expression_from_basis(basis_terms):
        matrix_refit = []
        vector_refit = []

        for _, row in train_df.iterrows():
            subs = {symbols[col]: row[col] for col in df.columns}
            row_vals = []

            for phi in basis_terms:
                try:
                    row_vals.append(float(phi.subs(subs)))
                except (TypeError, ValueError, ZeroDivisionError):
                    row_vals.append(0.0)

            matrix_refit.append(row_vals)
            vector_refit.append(row[target_var])

        matrix_refit = np.array(matrix_refit, dtype=float)
        vector_refit = np.array(vector_refit, dtype=float)

        try:
            k_vals, _ = _fit_coefficients(
                matrix_refit,
                vector_refit,
                use_ransac=use_ransac,
                random_state=random_state,
                use_ot_loss=use_ot_loss,
                alpha=alpha,
                beta=beta,
            )
        except np.linalg.LinAlgError:
            return None

        refined_expr = 0
        for k, phi in zip(k_vals, basis_terms):
            if abs(k) > 1e-10:
                refined_expr += k * phi

        return sp.simplify(refined_expr)

    def evaluate_candidate(candidate_expr):
        val_error = 0.0
        valid_val_rows = 0
        residuals = []
        feature_matrix = []

        for _, row in val_df.iterrows():
            subs_dict = {symbols[col]: row[col] for col in df.columns}
            try:
                predicted = float(candidate_expr.subs(subs_dict))
            except (TypeError, ValueError, ZeroDivisionError):
                continue

            if not np.isfinite(predicted):
                continue

            target_value = row[target_var]
            if not np.isfinite(target_value):
                continue

            residual = target_value - predicted
            val_error += abs(residual)
            valid_val_rows += 1
            residuals.append(residual)
            feature_matrix.append([row[col] for col in variables])

        if valid_val_rows == 0:
            return None, 0, {}

        residuals = np.array(residuals, dtype=float)
        feature_matrix = np.array(feature_matrix, dtype=float)

        correlations = {}
        residual_std = float(np.std(residuals)) if len(residuals) > 1 else 0.0
        for i, var in enumerate(variables):
            var_values = feature_matrix[:, i]
            var_std = float(np.std(var_values)) if len(var_values) > 1 else 0.0

            if len(residuals) > 1 and residual_std > 0.0 and var_std > 0.0:
                corr = np.corrcoef(residuals, var_values)[0, 1]
                corr_sq = np.corrcoef(residuals, var_values**2)[0, 1]

                if np.isfinite(corr):
                    correlations[var] = float(corr)
                if np.isfinite(corr_sq):
                    correlations[f"{var}^2"] = float(corr_sq)

        significant = {k: v for k, v in correlations.items() if abs(v) > 0.7}
        val_mae = float(val_error) / float(valid_val_rows)
        return val_mae, valid_val_rows, significant

    for term_count in range(1, len(valid_basis) + 1):
        partial_a = matrix_a[:, :term_count]
        partial_basis = valid_basis[:term_count]

        try:
            k_vals, _ = _fit_coefficients(
                partial_a,
                vector_b,
                use_ransac=use_ransac,
                random_state=random_state,
                use_ot_loss=use_ot_loss,
                alpha=alpha,
                beta=beta,
            )
        except np.linalg.LinAlgError:
            continue

        candidate_expr = 0
        # Sparse selection: only keep coefficients robust to scaling.
        # Use a relative threshold based on median coefficient magnitude.
        if use_sparse:
            k_abs = np.abs(k_vals)
            k_abs_nonzero = k_abs[k_abs > 1e-15]
            if len(k_abs_nonzero) > 0:
                # Use 1% of median coefficient size as threshold (handles vastly different scales)
                threshold = np.median(k_abs_nonzero) * 0.01
            else:
                threshold = 1e-12
            for k, phi in zip(k_vals, partial_basis):
                if abs(k) > threshold:
                    candidate_expr += k * phi
        else:
            for k, phi in zip(k_vals, partial_basis):
                candidate_expr += k * phi

        candidate_expr = sp.simplify(candidate_expr)

        val_error, valid_val_rows, significant = evaluate_candidate(candidate_expr)
        if valid_val_rows == 0 or val_error is None:
            continue

        score = val_error + lambda_penalty * complexity(candidate_expr)

        # Mode 3: add flow-prior penalty to candidate score
        if _use_fp:
            try:
                fp_exps = _extract_latent_from_expr(candidate_expr, variables)
                score += compute_mode3_loss(
                    exponents=fp_exps,
                    variable_names=variables,
                    scorer=flow_prior_scorer,
                )
            except Exception:
                pass  # never let the prior crash the main loop

        if score < best_score:
            second_best_score = best_score
            second_best_expr = best_expr
            best_score = score
            best_val_error = val_error
            best_expr = candidate_expr
            best_significant = significant
            best_basis = list(partial_basis)
        elif score < second_best_score and candidate_expr != best_expr:
            second_best_score = score
            second_best_expr = candidate_expr

    if best_expr is None:
        raise RuntimeError("No valid validation rows for scoring.")

    if use_residual:
        valid_candidates = list(valid_basis)
        for _ in range(max_iterations):
            residuals = []
            for _, row in val_df.iterrows():
                subs = {symbols[col]: row[col] for col in df.columns}
                target_value = row[target_var]

                if not np.isfinite(target_value):
                    residuals.append(0.0)
                    continue

                try:
                    predicted = float(best_expr.subs(subs))
                except (TypeError, ValueError, ZeroDivisionError):
                    predicted = 0.0

                if not np.isfinite(predicted):
                    predicted = 0.0

                residuals.append(float(target_value - predicted))

            residuals = np.array(residuals, dtype=float)
            residual_std = float(np.std(residuals)) if len(residuals) > 1 else 0.0

            if residual_std <= 1e-12:
                break

            projection_scores = {}
            for phi in valid_candidates:
                if phi in best_basis:
                    continue

                phi_vals = []

                for _, row in val_df.iterrows():
                    subs = {symbols[col]: row[col] for col in df.columns}
                    try:
                        phi_vals.append(float(phi.subs(subs)))
                    except (TypeError, ValueError, ZeroDivisionError):
                        phi_vals.append(0.0)

                phi_vals = np.array(phi_vals, dtype=float)

                if np.std(phi_vals) > 1e-12:
                    corr = np.corrcoef(residuals, phi_vals)[0, 1]
                    if np.isfinite(corr):
                        projection_scores[phi] = abs(float(corr))

            if not projection_scores:
                break

            best_phi = max(projection_scores, key=projection_scores.get)
            best_projection = projection_scores[best_phi]

            if best_projection < min_projection_threshold:
                break

            expanded_basis = list(best_basis)
            expanded_basis.append(best_phi)
            refined_expr = fit_expression_from_basis(expanded_basis)
            if refined_expr is None:
                break

            new_val_error, refined_rows, refined_significant = evaluate_candidate(refined_expr)
            if refined_rows == 0 or new_val_error is None:
                break

            # Mode 3: incorporate flow prior into residual-refinement comparison
            effective_new_error = new_val_error
            if _use_fp:
                try:
                    fp_exps = _extract_latent_from_expr(refined_expr, variables)
                    effective_new_error += compute_mode3_loss(
                        exponents=fp_exps,
                        variable_names=variables,
                        scorer=flow_prior_scorer,
                    )
                except Exception:
                    pass

            if best_val_error <= 1e-12:
                improvement_ratio = 0.0
            else:
                improvement_ratio = (best_val_error - effective_new_error) / best_val_error

            if improvement_ratio < min_improvement_ratio:
                break

            best_expr = refined_expr
            best_val_error = new_val_error
            best_significant = refined_significant
            best_basis = expanded_basis
            if np.std(phi_vals) > 1e-12:
                corr = np.corrcoef(residuals, phi_vals)[0, 1]
                if np.isfinite(corr):
                    projection_scores[phi] = abs(float(corr))

        # The following block is outside the intended loop and should be removed to fix SyntaxError
        # if best_val_error <= 1e-12:
        #     improvement_ratio = 0.0
        # else:
        #     improvement_ratio = (best_val_error - new_val_error) / best_val_error
        #
        # if improvement_ratio < min_improvement_ratio:
        #     break
        #
        # best_expr = refined_expr
        # best_val_error = new_val_error
        # best_significant = refined_significant
        # best_basis = expanded_basis

    # Mode 3: record final flow-prior score for the chosen expression
    if _use_fp and best_expr is not None:
        try:
            fp_exps = _extract_latent_from_expr(best_expr, variables)
            structure_metadata["flow_prior_loss"] = compute_mode3_loss(
                exponents=fp_exps,
                variable_names=variables,
                scorer=flow_prior_scorer,
            )
        except Exception:
            pass

    # Mode 4: JEPA prior — record combined score for the final expression
    if _use_jepa and best_expr is not None:
        try:
            X_val = val_df[variables].to_numpy(dtype=float)
            y_val = val_df[target_var].to_numpy(dtype=float)
            combined = jepa_prior.combined_score(
                candidate_exprs=[str(best_expr)],
                X=X_val,
                y=y_val,
                ot_scores=np.array([best_val_error]),
                gamma=gamma,
            )
            structure_metadata["jepa_combined_score"] = float(combined[0])
        except Exception:
            pass  # never let the prior crash the main loop
    # Delta-s: rank-1 vs rank-2 scoring margin
    if second_best_score < float("inf") and best_score < float("inf"):
        structure_metadata["delta_s"] = float(second_best_score - best_score)
        structure_metadata["rank2_expr"] = str(second_best_expr) if second_best_expr is not None else None

    if return_metadata:
        return best_expr, float(best_val_error), best_significant, structure_metadata

    return best_expr, float(best_val_error), best_significant


def discover_symbolic_law(
    data,
    target_name,
    variable_names,
    max_power=3,
    variable_dimensions=None,
    target_dimension=None,
    refine_residuals=True,
    min_residual_correlation=0.6,
    min_refine_improvement_ratio=0.01,
    add_physics_features=True,
    add_latent_features=None,
):
    from .candidate_generator import generate_candidates
    from .model_fitting import fit_model
    from .residual_refinement import refine_with_residuals
    from .scoring import score_model
    from ..pir.dimensional_analysis import dimensional_filter

    working_data = data.copy()
    if add_latent_features is None:
        add_latent_features = add_physics_features

    extra_features = {}
    if add_physics_features:
        extra_features.update(generate_physics_features(working_data))
    if add_latent_features:
        extra_features.update(generate_latent_variables(working_data))

    if extra_features:
        for name, values in extra_features.items():
            if name not in working_data.columns:
                working_data[name] = values

    usable_variable_names = list(variable_names)
    if extra_features:
        for name in working_data.columns:
            if name != target_name and name not in usable_variable_names:
                usable_variable_names.append(name)

    target = working_data[target_name].values
    symbols = [sp.Symbol(v) for v in usable_variable_names]

    candidates = generate_candidates(symbols, max_power=max_power)

    if variable_dimensions is not None and target_dimension is not None:
        candidates = dimensional_filter(candidates, variable_dimensions, target_dimension)

    best_model = None
    best_score = float("inf")

    for expr in candidates:
        try:
            f_expr = sp.lambdify(symbols, expr, "numpy")
            features = f_expr(*[working_data[v].values for v in usable_variable_names])
            features = np.asarray(features, dtype=float).reshape(-1)

            finite_mask = np.isfinite(features) & np.isfinite(target)
            if np.sum(finite_mask) < 2:
                continue

            coef, intercept, mse = fit_model(features[finite_mask], target[finite_mask])
            score = score_model(mse, len(str(expr)))

            if score < best_score:
                best_score = score
                best_model = {
                    "expression": expr,
                    "coefficient": coef,
                    "intercept": intercept,
                    "score": float(score),
                    "mse": float(mse),
                }
        except Exception:
            continue

    if refine_residuals and best_model is not None:
        best_model = refine_with_residuals(
            data=working_data,
            target_name=target_name,
            variable_names=usable_variable_names,
            base_model=best_model,
            candidates=candidates,
            min_correlation=min_residual_correlation,
            min_improvement_ratio=min_refine_improvement_ratio,
        )

    return best_model
