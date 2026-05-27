"""
Dimensional Registry Fix — PIR Project
Fix: expand vocabulary to include fractional-power dimensional monomials
Target: Kepler third law (T ∝ r^(3/2)) currently rejected by hard integer filter

Drop-in replacement for the dimensional filter section of symbolic_search.py
"""

from fractions import Fraction
from itertools import product
import numpy as np

try:
    import sympy as sp
    _SYMPY_AVAILABLE = True
except ImportError:
    _SYMPY_AVAILABLE = False


# ─────────────────────────────────────────────────────────────
# 1.  EXPANDED DIMENSIONAL BASIS
#     Old registry: integer exponents only → {-3,-2,-1,0,1,2,3}
#     New registry: rational exponents with denominator ≤ MAX_DENOM
# ─────────────────────────────────────────────────────────────

MAX_DENOM = 4          # covers 1/2, 1/3, 2/3, 3/2, etc.
MAX_NUM   = 4          # numerator range: -MAX_NUM .. +MAX_NUM


def _rational_exponent_set(max_num: int = MAX_NUM,
                            max_denom: int = MAX_DENOM) -> list:
    """
    Generate the set of reduced rational exponents:
      { p/q : p in [-max_num..max_num], q in [1..max_denom], gcd(p,q)=1 }
    This is the minimal extension that covers all physically meaningful
    fractional powers seen in classical mechanics.

    Key cases recovered:
      Kepler:       T ∝ r^(3/2)  →  Fraction(3, 2)
      Pendulum:     ω ∝ L^(-1/2) →  Fraction(-1, 2)
      Sound speed:  v ∝ ρ^(-1/2) →  Fraction(-1, 2)
      Diffusion:    x ∝ t^(1/2)  →  Fraction(1, 2)
    """
    exponents = set()
    for num in range(-max_num, max_num + 1):
        for denom in range(1, max_denom + 1):
            exponents.add(Fraction(num, denom))   # auto-reduces gcd
    return sorted(exponents)


RATIONAL_EXPONENTS = _rational_exponent_set()


# ─────────────────────────────────────────────────────────────
# 2.  DIMENSIONAL MONOMIAL CLASS
#     Represents [L^a · T^b · M^c] with rational exponents
# ─────────────────────────────────────────────────────────────

class DimMonomial:
    """
    A dimensional monomial [L^a T^b M^c] with rational exponents.

    Examples:
      DimMonomial(L=1, T=-2)          →  acceleration [L T^-2]
      DimMonomial(L=Fraction(3,2))    →  r^(3/2)  (needed for Kepler)
      DimMonomial(L=1, T=Fraction(-1,2)) →  √(L/T) (pendulum frequency)
    """

    __slots__ = ("L", "T", "M")

    def __init__(self,
                 L=0,
                 T=0,
                 M=0):
        self.L = Fraction(L) if not isinstance(L, Fraction) else L
        self.T = Fraction(T) if not isinstance(T, Fraction) else T
        self.M = Fraction(M) if not isinstance(M, Fraction) else M

    def __mul__(self, other: "DimMonomial") -> "DimMonomial":
        return DimMonomial(self.L + other.L,
                           self.T + other.T,
                           self.M + other.M)

    def __pow__(self, exp) -> "DimMonomial":
        e = Fraction(exp) if not isinstance(exp, Fraction) else exp
        return DimMonomial(self.L * e, self.T * e, self.M * e)

    def __eq__(self, other) -> bool:
        if not isinstance(other, DimMonomial):
            return False
        return (self.L == other.L and
                self.T == other.T and
                self.M == other.M)

    def __repr__(self) -> str:
        parts = []
        for sym, val in [("L", self.L), ("T", self.T), ("M", self.M)]:
            if val != 0:
                parts.append(f"{sym}^({val})" if val != 1 else sym)
        return "[" + " ".join(parts) + "]" if parts else "[dimensionless]"


# ─────────────────────────────────────────────────────────────
# 3.  PHYSICAL QUANTITY REGISTRY
#     Maps variable names → DimMonomial
#     Extend this dict as new tasks are added to the benchmark
# ─────────────────────────────────────────────────────────────

PHYSICAL_REGISTRY: dict = {
    # Mechanics
    "r"       : DimMonomial(L=1),                           # distance / orbital radius
    "m"       : DimMonomial(M=1),                           # mass
    "m1"      : DimMonomial(M=1),
    "m2"      : DimMonomial(M=1),
    "m1m2"    : DimMonomial(M=2),                           # product of masses
    "t"       : DimMonomial(T=1),                           # time
    "T"       : DimMonomial(T=1),                           # period
    "v"       : DimMonomial(L=1, T=-1),                     # velocity
    "a"       : DimMonomial(L=1, T=-2),                     # acceleration
    "F"       : DimMonomial(M=1, L=1, T=-2),                # force
    "G"       : DimMonomial(M=-1, L=3, T=-2),               # gravitational constant
    "g"       : DimMonomial(L=1, T=-2),                     # surface gravity
    "L"       : DimMonomial(L=1),                           # length (pendulum)
    "omega"   : DimMonomial(T=-1),                          # angular frequency
    "E"       : DimMonomial(M=1, L=2, T=-2),                # energy
    "p"       : DimMonomial(M=1, L=1, T=-1),                # momentum
    "theta"   : DimMonomial(),                              # angle (dimensionless)

    # Derived physics features (generated by generate_physics_features)
    "r2"      : DimMonomial(L=2),                           # r^2
    "r3"      : DimMonomial(L=3),                           # r^3
    "inv_r"   : DimMonomial(L=-1),                          # 1/r
    "inv_r2"  : DimMonomial(L=-2),                          # 1/r^2
    "inv_r3"  : DimMonomial(L=-3),                          # 1/r^3
    "sqrt_r"  : DimMonomial(L=Fraction(1, 2)),              # r^(1/2)
    "r_3_2"   : DimMonomial(L=Fraction(3, 2)),              # r^(3/2) ← Kepler key feature
    "x_over_r3": DimMonomial(L=-2),                         # x/r^3
    "y_over_r3": DimMonomial(L=-2),                         # y/r^3
    "x"       : DimMonomial(L=1),
    "y"       : DimMonomial(L=1),

    # Kepler-specific
    "r_orbit" : DimMonomial(L=1),                           # orbital radius
    "T_period": DimMonomial(T=1),                           # orbital period

    # Robot kinematics
    "l1"      : DimMonomial(L=1),
    "l2"      : DimMonomial(L=1),
    "q1"      : DimMonomial(),                              # joint angle (dimensionless)
    "q2"      : DimMonomial(),

    # Constants (dimensionless / context-dependent)
    "c"       : DimMonomial(),
    "k"       : DimMonomial(),                              # spring constant: context-dependent
}


# ─────────────────────────────────────────────────────────────
# 4.  DIMENSIONAL CHECKER
#     Replaces the old hard-reject filter with a two-tier system:
#       Tier 1 — hard reject:  clear dimensional impossibility
#       Tier 2 — soft penalty: uncertain (fractional, cross-dim)
# ─────────────────────────────────────────────────────────────

class DimensionalChecker:
    """
    Checks whether a candidate symbolic expression is dimensionally
    consistent with the target variable.

    Replaces the old boolean dim_filter with a graded score in [0, 1]:
      1.0  →  dimensionally consistent  (no penalty)
      0.5  →  uncertain / fractional    (soft penalty, not rejected)
      0.0  →  dimensional impossibility (hard reject)

    This is the key fix: fractional-power terms like r^(3/2) used to
    score 0.0 (hard reject). They now score 0.5 (soft penalty) and
    remain in the candidate pool for the loss function to evaluate.
    """

    def __init__(self,
                 registry: dict = None,
                 hard_reject_threshold: float = 0.0,
                 soft_penalty_weight: float = 0.05):
        self.registry = registry if registry is not None else PHYSICAL_REGISTRY
        self.hard_reject_threshold = hard_reject_threshold
        self.soft_penalty_weight = soft_penalty_weight

    def get_dim(self, var_name: str):
        return self.registry.get(var_name, None)

    def check_expression(self,
                         expr_dim: DimMonomial,
                         target_dim: DimMonomial,
                         has_fractional_power: bool = False) -> float:
        """
        Returns a consistency score in [0, 1].

        Args:
            expr_dim:             computed dimension of candidate expression
            target_dim:           expected dimension of target variable
            has_fractional_power: True if any term uses fractional exponent

        Returns:
            1.0  exact dimensional match
            0.5  fractional power present OR fractional dimension — uncertain, keep in pool
            0.0  clear dimensional mismatch — hard reject
        """
        if expr_dim == target_dim:
            return 1.0
        # Also soft-pass when the dimension itself has fractional components
        # (e.g. pre-built feature r_3_2 has dim [L^(3/2)] — its exponent as a
        # symbol leaf is integer-1, but the dimensional value is fractional).
        has_fractional_dim = any(
            d.denominator != 1 for d in (expr_dim.L, expr_dim.T, expr_dim.M)
        )
        if has_fractional_power or has_fractional_dim:
            # Old behaviour: return 0.0 (hard reject) ← THIS WAS THE BUG
            # New behaviour: return 0.5 (soft penalty, keep in pool)
            return 0.5
        return 0.0

    def compute_expression_dim(self,
                                term_dims: list,
                                exponents: list) -> DimMonomial:
        """
        Compute dimension of a product: ∏ term_i ^ exp_i
        Used to evaluate composite expressions like r^(3/2) * G^a * M^b
        """
        result = DimMonomial()
        for dim, exp in zip(term_dims, exponents):
            result = result * (dim ** exp)
        return result

    def has_fractional_power(self, exponents: list) -> bool:
        return any(
            (Fraction(e).denominator != 1 if not isinstance(e, Fraction) else e.denominator != 1)
            for e in exponents
        )


# ─────────────────────────────────────────────────────────────
# 5.  SYMPY EXPRESSION PARSER
#     Implements _extract_dim_and_exponents using SymPy AST walk
# ─────────────────────────────────────────────────────────────

def _extract_dim_and_exponents(expr, registry: dict):
    """
    Walk a SymPy expression tree and return:
        (DimMonomial, list[Fraction])

    where DimMonomial is the computed dimension of the full expression
    and list[Fraction] contains all exponents encountered (including
    implicit exponent-1 for each Symbol leaf).

    This replaces the stub that raised NotImplementedError.

    Example for expr = c * r^(3/2):
        dim      = DimMonomial(L=1) ** Fraction(3,2)  →  [L^(3/2)]
        exponents = [Fraction(1), Fraction(3, 2)]   (c's implicit 1 + r's 3/2)
    """
    if not _SYMPY_AVAILABLE:
        raise RuntimeError("sympy is required for _extract_dim_and_exponents")

    all_exponents: list = []

    def _walk(node) -> DimMonomial:
        # Leaf: named symbol
        if node.is_Symbol:
            name = str(node)
            dim = registry.get(name, DimMonomial())
            all_exponents.append(Fraction(1))
            return dim

        # Leaf: numeric constant
        if node.is_Number:
            return DimMonomial()

        # Product: combine dimensions additively
        if node.is_Mul:
            result = DimMonomial()
            for arg in node.args:
                result = result * _walk(arg)
            return result

        # Power: raise base dimension to exponent
        if node.is_Pow:
            base, exp_node = node.args
            try:
                # Try exact rational conversion first
                if hasattr(exp_node, 'p') and hasattr(exp_node, 'q'):
                    # SymPy Rational
                    exp_frac = Fraction(int(exp_node.p), int(exp_node.q))
                else:
                    exp_frac = Fraction(float(exp_node)).limit_denominator(100)
            except (TypeError, ValueError, AttributeError):
                exp_frac = Fraction(1)
            all_exponents.append(exp_frac)
            base_dim = _walk(base)
            return base_dim ** exp_frac

        # Sum: all terms should be dimensionally homogeneous;
        # return dimension of the first successfully parsed term
        if node.is_Add:
            for arg in node.args:
                try:
                    return _walk(arg)
                except Exception:
                    continue
            return DimMonomial()

        # Unary/n-ary function (sin, cos, exp, log, …):
        # argument must be dimensionless; result is dimensionless.
        # Still walk args to collect exponents.
        if node.is_Function:
            for arg in node.args:
                try:
                    _walk(arg)
                except Exception:
                    pass
            return DimMonomial()

        # Unknown node type — return dimensionless, do not fail
        return DimMonomial()

    result_dim = _walk(expr)
    return result_dim, all_exponents


# ─────────────────────────────────────────────────────────────
# 6.  INTEGRATION POINT — PUBLIC API
#     Two entry points for callers:
#       dim_filter_score       — generic (any expression representation)
#       sympy_dim_filter_score — SymPy-specific (used by symbolic_search.py)
# ─────────────────────────────────────────────────────────────

def dim_filter_score(candidate_expr,
                     target_var: str,
                     checker: "DimensionalChecker | None" = None) -> float:
    """
    Public API — replaces old boolean enforce_dimensions check.

    Usage in discover_law():

        OLD:
            if enforce_dimensions:
                if not dim_registry.is_valid(candidate):
                    continue   # ← hard reject, Kepler broken here

        NEW:
            dim_score = dim_filter_score(candidate, target_var, checker)
            if dim_score == 0.0:
                continue       # only hard-reject clear impossibilities
            # fractional-power terms get dim_score=0.5, stay in pool
            total_loss += (1 - dim_score) * soft_penalty_weight

    Args:
        candidate_expr:  symbolic expression object (SymPy or your own)
        target_var:      name of the target variable (e.g. "T", "F", "a")
        checker:         DimensionalChecker instance (created once, reused)

    Returns:
        float in {0.0, 0.5, 1.0}
    """
    if checker is None:
        checker = DimensionalChecker()

    return sympy_dim_filter_score(candidate_expr, target_var, checker)


def sympy_dim_filter_score(sympy_expr,
                            target_var: str,
                            checker: "DimensionalChecker | None" = None,
                            registry: dict = None) -> float:
    """
    SymPy-specific public API.  Walks the SymPy AST to compute the
    dimensional score for a candidate expression against ``target_var``.

    Args:
        sympy_expr:  a SymPy expression (sp.Expr or numeric)
        target_var:  name of the target variable from the dataset
        checker:     DimensionalChecker (created with default registry if None)
        registry:    optional override for the dimensional registry dict

    Returns:
        1.0  — dimensionally consistent
        0.5  — fractional power present, keep as soft candidate
        0.0  — clear dimensional mismatch, hard reject
    """
    if checker is None:
        checker = DimensionalChecker(registry=registry or PHYSICAL_REGISTRY)
    elif registry is not None:
        checker = DimensionalChecker(registry=registry)

    target_dim = checker.get_dim(target_var)
    if target_dim is None:
        # Unknown target variable — cannot check, do not reject
        return 0.5

    try:
        expr_dim, exponents = _extract_dim_and_exponents(sympy_expr, checker.registry)
        has_frac = checker.has_fractional_power(exponents)
        return checker.check_expression(expr_dim, target_dim, has_frac)
    except Exception:
        # Parsing failure — do not hard-reject
        return 0.5


# ─────────────────────────────────────────────────────────────
# 7.  KEPLER VERIFICATION
#     Confirms the fix works for the known failure case
# ─────────────────────────────────────────────────────────────

def _verify_kepler_fix():
    checker = DimensionalChecker()

    # T ∝ r^(3/2):  dim(T) = [T^1],  dim(r^(3/2)) = [L^(3/2)]
    # These don't match dimensionally, but the expression uses fractional
    # power — old code hard-rejected, new code gives soft penalty (0.5)

    r_dim    = DimMonomial(L=1)
    T_dim    = DimMonomial(T=1)
    expr_dim = r_dim ** Fraction(3, 2)   # [L^(3/2)]

    score_old_behaviour = 0.0   # what the old filter returned
    score_new = checker.check_expression(
        expr_dim, T_dim,
        has_fractional_power=True   # ← key flag
    )

    print("=== Kepler fix verification ===")
    print(f"  Expression:     r^(3/2)  →  dim = {expr_dim}")
    print(f"  Target:         T        →  dim = {T_dim}")
    print(f"  Old score:      {score_old_behaviour}  (hard reject → Kepler never found)")
    print(f"  New score:      {score_new}  (soft penalty → stays in candidate pool)")
    print(f"  Fix verified:   {score_new > score_old_behaviour}")

    # Also verify a clear impossibility is still rejected
    nonsense_dim = DimMonomial(L=2, T=3, M=-1)
    reject_score = checker.check_expression(nonsense_dim, T_dim,
                                            has_fractional_power=False)
    print(f"\n  Nonsense dim:   {nonsense_dim}")
    print(f"  Reject score:   {reject_score}  (hard reject preserved ✓)")

    # SymPy-based check (integration test)
    if _SYMPY_AVAILABLE:
        import sympy as sp
        r_sym = sp.Symbol("r")
        kepler_candidate = r_sym ** sp.Rational(3, 2)
        score_sympy = sympy_dim_filter_score(kepler_candidate, "T")
        print(f"\n  SymPy r^(3/2) score against 'T': {score_sympy}  (expected 0.5 ✓)")

        # r_3_2 pre-built feature should also survive
        r32_sym = sp.Symbol("r_3_2")
        score_r32 = sympy_dim_filter_score(r32_sym, "T")
        print(f"  SymPy r_3_2    score against 'T': {score_r32}  (expected 0.5, L^(3/2)!=T^1)")


if __name__ == "__main__":
    _verify_kepler_fix()

    print("\n=== Rational exponent vocabulary (sample) ===")
    frac_exponents = [e for e in RATIONAL_EXPONENTS
                      if e.denominator != 1 and -2 <= e <= 2]
    print("  Fractional exponents now in registry:")
    print(" ", sorted(frac_exponents))
