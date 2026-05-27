"""Variational mechanics tools for PIR.

Includes structured Lagrangian discovery with T(dq) - V(q) parameterization,
Euler-Lagrange residual utilities, and basis library builders.
"""

from .euler_lagrange import (
    build_el_regression_matrix,
    evaluate_el_residual,
    solve_nullspace_coefficients,
)
from .lagrangian_discovery import (
    StructuredLagrangianResult,
    discover_structured_lagrangian,
    discover_structured_lagrangian_from_csv,
    discover_structured_lagrangian_from_dataframe,
)
from .lagrangian_library import (
    ScalarBasisTerm,
    build_structured_library,
    kinetic_terms,
    lagrangian_features,
    potential_terms,
)
from .multibody_lagrangian_discovery import (
    MultiBodyLagrangianResult,
    discover_multibody_lagrangian,
    discover_multibody_lagrangian_from_csv,
    discover_multibody_lagrangian_from_dataframe,
)
from .multibody_library import (
    MultiBodyBasisTerm,
    multibody_basis_terms,
    multibody_features,
)

__all__ = [
    "ScalarBasisTerm",
    "kinetic_terms",
    "potential_terms",
    "build_structured_library",
    "lagrangian_features",
    "MultiBodyBasisTerm",
    "multibody_basis_terms",
    "multibody_features",
    "build_el_regression_matrix",
    "solve_nullspace_coefficients",
    "evaluate_el_residual",
    "StructuredLagrangianResult",
    "discover_structured_lagrangian",
    "discover_structured_lagrangian_from_dataframe",
    "discover_structured_lagrangian_from_csv",
    "MultiBodyLagrangianResult",
    "discover_multibody_lagrangian",
    "discover_multibody_lagrangian_from_dataframe",
    "discover_multibody_lagrangian_from_csv",
]
