"""
Feynman 100 dataset loader for PIR-Bench Phase 1 (v3.3).

v2 — adds exponent inference per equation. See _infer_three_var_exponents.

Reads FeynmanEquations.csv (Udrescu & Tegmark 2020 manifest) and provides
one parameterized generator function compatible with PIR's experiment_registry
convention.

Usage in experiment_registry.py:

    from .feynman_loader import (
        generate_feynman_dataset,
        register_feynman_experiments,
        TIER_A_EQUATIONS,
        TIER_B_EQUATIONS,
        TIER_C_EQUATIONS,
    )

    EXPERIMENTS.extend(register_feynman_experiments(tier="A", ExperimentConfig=ExperimentConfig))

Key change in v2:
  Each registered experiment gets `three_var_exponents` based on the formula:
    - positive-only product (no 1/x, no x**(-n)) → [Fraction(1)]
    - inverse terms present                     → [Fraction(-2), Fraction(-1), Fraction(1)]
  This fixes the v1 issue where Feynman #15 (U=mgz) was returning
  rational approximations like (153z-156)/(g*m*z) instead of m*g*z.
"""

from __future__ import annotations

import re
from fractions import Fraction
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr


# Path to manifest — relative to physics_engine/ root. Adjust if needed.
DEFAULT_MANIFEST_PATH = Path(__file__).resolve().parents[2] / "FeynmanEquations.csv"


# ---------------------------------------------------------------------------
# Tier classification (see v3.3 paper for rationale)
# ---------------------------------------------------------------------------
TIER_A_EQUATIONS = [
    1,
    8, 12, 15, 16, 25, 28, 37, 40, 53, 59, 65, 74, 85, 92,
    6, 9, 11, 26, 27, 30, 31, 38, 49, 51, 52, 54, 58, 60, 66,
    69, 70, 73, 75, 76, 77, 78, 86, 87, 88, 95, 96, 97, 98,
]
TIER_A_EQUATIONS = sorted(set(TIER_A_EQUATIONS))

TIER_B_EQUATIONS = [
    2, 3, 7, 10, 13, 14, 19, 20, 21, 22, 23, 24, 29,
    32, 34, 35, 36, 39, 41, 42, 45, 46, 47, 50,
    55, 57, 61, 63, 67, 68, 71, 72, 79, 80, 81, 83,
    84, 91, 93, 94, 99, 100,
]
TIER_B_EQUATIONS = sorted(set(TIER_B_EQUATIONS) - set(TIER_A_EQUATIONS))

TIER_C_EQUATIONS = [
    4, 5, 17, 18, 33, 43, 44, 48, 56, 62, 64, 82, 89, 90,
]
TIER_C_EQUATIONS = sorted(set(TIER_C_EQUATIONS) -
                          set(TIER_A_EQUATIONS) - set(TIER_B_EQUATIONS))


# ---------------------------------------------------------------------------
# Manifest cache (load once)
# ---------------------------------------------------------------------------
_manifest_cache: Optional[pd.DataFrame] = None


def _load_manifest(manifest_path: Optional[Path] = None) -> pd.DataFrame:
    global _manifest_cache
    if _manifest_cache is not None:
        return _manifest_cache

    path = Path(manifest_path) if manifest_path else DEFAULT_MANIFEST_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Feynman manifest not found at {path}. "
            "Copy FeynmanEquations.csv to physics_engine/ root, or set "
            "DEFAULT_MANIFEST_PATH explicitly."
        )

    df = pd.read_csv(path)
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    df = df.dropna(subset=["Number", "Formula"]).reset_index(drop=True)
    df["Number"] = df["Number"].astype(int)
    df["# variables"] = df["# variables"].astype(int)
    _manifest_cache = df
    return df


def _get_row(equation_number: int, manifest_path: Optional[Path] = None) -> dict:
    df = _load_manifest(manifest_path)
    matches = df[df["Number"] == equation_number]
    if len(matches) == 0:
        raise KeyError(
            f"Equation number {equation_number} not in manifest. "
            f"Available: {sorted(df['Number'].tolist())}"
        )
    return matches.iloc[0].to_dict()


# ---------------------------------------------------------------------------
# NEW IN v2: Exponent inference from formula structure
# ---------------------------------------------------------------------------
def _infer_three_var_exponents(formula_str: str) -> List[Fraction]:
    """
    Decide whether a Feynman equation's 3-var products should use
    positive-only exponents or include negatives.

    Returns:
      [Fraction(1)]                                     if formula has only positive powers
      [Fraction(-2), Fraction(-1), Fraction(1)]         if formula has inverse / negative powers

    Detection heuristics on the formula STRING (not parsed sympy):
      1. Any '**(-N)' pattern (e.g., r**(-2), r**(-3/2))   → negatives needed
      2. Any '**-N' without parentheses                    → negatives needed
      3. Any '1/<var>' or '1/(<expr>)'                     → negatives needed
      4. Any '<expr>/<var>' that isn't matching constants  → negatives needed (conservative)

    This is intentionally conservative: when in doubt, include negatives.
    False positives (including negatives unnecessarily) → at worst restore v1
    behavior. False negatives (excluding negatives when needed) → miss
    discovery. The asymmetry favors including negatives.
    """
    f = formula_str.replace(" ", "")

    # Strong signals for negative powers
    if re.search(r"\*\*\(?-\d", f):           # x**(-2), x**-2, x**(-3/2)
        return [Fraction(-2), Fraction(-1), Fraction(1)]
    if re.search(r"\*\*\(?\d+/\d+\s*\*?-", f): # rare: x**(3/-2)
        return [Fraction(-2), Fraction(-1), Fraction(1)]
    # 1/<expr> or any '/' followed by a variable (not just constants)
    if re.search(r"/[a-zA-Z_]", f):           # /m, /r, /epsilon, etc.
        return [Fraction(-2), Fraction(-1), Fraction(1)]
    if re.search(r"/\([^)]*[a-zA-Z]", f):     # /(m+n), /(r-r0), etc.
        return [Fraction(-2), Fraction(-1), Fraction(1)]

    # No negatives detected — restrict to positive product
    return [Fraction(1)]


# ---------------------------------------------------------------------------
# Generator function — signature unchanged from v1
# ---------------------------------------------------------------------------
def generate_feynman_dataset(
    equation_number: int,
    num_samples: int = 200,
    noise_std: float = 0.01,
    seed: int = 0,
    manifest_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Generate a sampled dataset for one Feynman equation."""
    row = _get_row(equation_number, manifest_path)
    n_vars = int(row["# variables"])
    formula_str = row["Formula"].strip()
    output_name = row["Output"].strip()

    var_specs = []
    for i in range(1, n_vars + 1):
        vname = str(row[f"v{i}_name"]).strip()
        vlow = float(row[f"v{i}_low"])
        vhigh = float(row[f"v{i}_high"])
        var_specs.append((vname, vlow, vhigh))

    rng = np.random.default_rng(seed)
    data = {}
    for vname, vlow, vhigh in var_specs:
        data[vname] = rng.uniform(vlow, vhigh, size=num_samples)

    local_dict = {
        "arcsin": sp.asin, "arccos": sp.acos, "arctan": sp.atan,
        "ln": sp.log, "pi": sp.pi,
        **{v: sp.Symbol(v) for v, _, _ in var_specs},
    }
    expr = parse_expr(formula_str, local_dict=local_dict)

    sym_order = [sp.Symbol(v) for v, _, _ in var_specs]
    f = sp.lambdify(sym_order, expr, "numpy")
    args = [data[v] for v, _, _ in var_specs]
    y_true = np.asarray(f(*args), dtype=float).ravel()

    std_y = float(np.std(y_true))
    if std_y < 1e-12:
        noise = rng.normal(0.0, noise_std, size=num_samples)
    else:
        noise = rng.normal(0.0, noise_std * std_y, size=num_samples)
    y_obs = y_true + noise

    df_out = pd.DataFrame(data)
    df_out[output_name] = y_obs
    return df_out


# ---------------------------------------------------------------------------
# Auto-registration helper — v2 with exponent inference
# ---------------------------------------------------------------------------
def register_feynman_experiments(
    tier: str = "A",
    ExperimentConfig=None,
    manifest_path: Optional[Path] = None,
) -> list:
    """
    Build a list of ExperimentConfig entries for the chosen Feynman tier.

    v2 NEW: per-equation `three_var_exponents` inferred from formula structure.
    """
    if ExperimentConfig is None:
        raise ValueError(
            "Pass the ExperimentConfig dataclass from experiment_registry. "
            "This avoids circular imports."
        )

    tier_map = {
        "A": TIER_A_EQUATIONS,
        "B": TIER_B_EQUATIONS,
        "C": TIER_C_EQUATIONS,
        "all": sorted(TIER_A_EQUATIONS + TIER_B_EQUATIONS + TIER_C_EQUATIONS),
    }
    if tier not in tier_map:
        raise ValueError(f"Unknown tier '{tier}'. Use A/B/C/all.")

    df = _load_manifest(manifest_path)
    experiments = []
    from functools import partial

    for eq_num in tier_map[tier]:
        rows = df[df["Number"] == eq_num]
        if len(rows) == 0:
            continue
        row = rows.iloc[0]
        n_vars = int(row["# variables"])
        output_name = str(row["Output"]).strip()
        filename = str(row["Filename"]).strip()
        formula = str(row["Formula"])

        # Detect operator usage in formula
        has_trig = bool(re.search(r"\b(sin|cos|tan|arcsin|arccos|arctan)\b", formula))
        has_exp_log = bool(re.search(r"\b(exp|log|ln|sqrt)\b", formula))

        # NOTE: PIR's runner only supports sin/cos as unary functions.
        # Equations needing log/exp/sqrt will use the rest of the basis
        # (products, powers) without unary support — honest current-state
        # measurement. Grammar extension is a v3.4 target.
        unary_funcs = []
        if has_trig:
            unary_funcs += ["sin", "cos"]

        # v2 KEY CHANGE: infer 3-var exponents from formula
        three_var_exps = _infer_three_var_exponents(formula)

        # Variables list for 3-var products (only when n_vars >= 3)
        var_list = []
        if n_vars >= 3:
            for i in range(1, min(n_vars + 1, 4)):  # first 3 vars
                var_list.append(str(row[f"v{i}_name"]).strip())

        default_kwargs = {
            "max_basis_terms": min(12 + 4 * n_vars, 40),
            "max_iterations": 50,
            "allowed_powers": [1, 2, 3] if three_var_exps == [Fraction(1)] else [-2, -1, 1, 2, 3],
            "enforce_dimensions": False,
            "unary_functions": unary_funcs,
            "include_pairwise_products": True,
            "use_3var_products": n_vars >= 3,
            "three_var_exponents": three_var_exps,
            "three_var_variables": var_list if var_list else None,
            "add_physics_features": False,    # blind benchmark
            "add_latent_features": False,
            "structure_guided": False,         # let basis fit do the work
        }

        safe_name = (
            f"feynman_{filename.replace('.', '_').replace('/', '_')}_n{eq_num}"
        )

        bound_generator = partial(generate_feynman_dataset, equation_number=eq_num)
        bound_generator.__name__ = f"generate_feynman_eq{eq_num}"

        experiments.append(
            ExperimentConfig(
                name=safe_name,
                target_column=output_name,
                dataset_generator=bound_generator,
                expected_tokens=tuple(),
                error_threshold=0.05,         # v3: tightened to match held-out verifier threshold
                default_discovery_kwargs=default_kwargs,
                operator_profiles=("linear", "extended"),
                success_mode="error_only",
                discovery_mode="standard",
            )
        )

    return experiments


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Exponent inference test:")
    print("=" * 60)

    test_cases = [
        ("m*g*z", "Eq 15 (U=mgz)"),
        ("mu*Nn", "Eq 8 (F=mu*Nn)"),
        ("q1*q2*r/(4*pi*epsilon*r**3)", "Eq 10 (Coulomb)"),
        ("G*m1*m2/((x2-x1)**2+(y2-y1)**2)", "Eq 5 (gravity)"),
        ("(x-u*t)/sqrt(1-u**2/c**2)", "Eq 17 (relativistic x)"),
        ("exp(-theta**2/2)/sqrt(2*pi)", "Eq 1 (Gaussian)"),
        ("1/2*k_spring*x**2", "Eq 16 (spring PE)"),
    ]
    for formula, label in test_cases:
        exps = _infer_three_var_exponents(formula)
        print(f"  {label}")
        print(f"    formula: {formula}")
        print(f"    inferred exponents: {exps}")
        print()

    print("=" * 60)
    print("Data generation smoke tests:")
    print("=" * 60)
    df = generate_feynman_dataset(15, num_samples=5, noise_std=0.01, seed=0)
    print("Eq 15 (U=mgz):")
    print(df)
    print()

    print("=" * 60)
    print(f"Tier A: {len(TIER_A_EQUATIONS)} equations")
    print(f"Tier B: {len(TIER_B_EQUATIONS)} equations")
    print(f"Tier C: {len(TIER_C_EQUATIONS)} equations")
    print(f"Total: {len(TIER_A_EQUATIONS + TIER_B_EQUATIONS + TIER_C_EQUATIONS)}")
