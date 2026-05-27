from dataclasses import dataclass
from functools import partial
from typing import Callable

import pandas as pd

from physics_engine.benchmarks import (
    generate_double_pendulum_dataset,
    generate_gravity_dataset,
    generate_harmonic_oscillator_dataset,
    generate_harmonic_oscillator_hamiltonian_dataset,
    generate_harmonic_oscillator_lagrangian_dataset,
    generate_inverse_square_acceleration_dataset,
    generate_kepler_third_law_dataset,
    generate_newton_dataset,
    generate_orbit_ax_dataset,
    generate_orbit_ay_dataset,
    generate_planar_robot_jacobian_dataset,
    generate_robot_trajectory_dataset,
    generate_pendulum_dataset,
    generate_franka_mass_dataset,
    generate_oog_damped_oscillator_dataset,
    generate_oog_relativistic_correction_dataset,
)


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    target_column: str
    dataset_generator: Callable[..., pd.DataFrame]
    expected_tokens: tuple[str, ...]
    error_threshold: float
    default_discovery_kwargs: dict
    operator_profiles: tuple[str, ...] = ("linear", "extended")
    success_mode: str = "token_or_error"
    discovery_mode: str = "standard"


EXPERIMENTS: list[ExperimentConfig] = [
    ExperimentConfig(
        name="newton",
        target_column="F",
        dataset_generator=generate_newton_dataset,
        expected_tokens=("m", "a"),
        error_threshold=0.35,
        default_discovery_kwargs={
            "max_basis_terms": 8,
            "max_iterations": 50,  # L1
            "allowed_powers": [1],
            "unary_functions": [],
        },
        success_mode="token_and_error",
    ),
    ExperimentConfig(
        name="pendulum",
        target_column="alpha",
        dataset_generator=generate_pendulum_dataset,
        expected_tokens=("theta",),
        error_threshold=0.7,
        default_discovery_kwargs={
            "max_basis_terms": 8,
            "max_iterations": 50,  # L3
            "allowed_powers": [1],
            "enforce_dimensions": False,
            "unary_functions": [],
        },
        success_mode="token_and_error",
    ),
    ExperimentConfig(
        name="kepler_third_law",
        target_column="T",
        dataset_generator=generate_kepler_third_law_dataset,
        expected_tokens=("r_3_2",),
        error_threshold=0.4,
        default_discovery_kwargs={
            "max_basis_terms": 8,
            "max_iterations": 50,  # L2
            "allowed_powers": [1],
            "enforce_dimensions": False,
            "unary_functions": [],
        },
        success_mode="token_and_error",
    ),
    ExperimentConfig(
        name="inverse_square_acceleration",
        target_column="a",
        dataset_generator=generate_inverse_square_acceleration_dataset,
        expected_tokens=("inv_r2",),
        error_threshold=0.4,
        default_discovery_kwargs={
            "max_basis_terms": 6,
            "max_iterations": 50,  # L2
            "allowed_powers": [1],
            "enforce_dimensions": False,
            "unary_functions": [],
        },
        success_mode="token_and_error",
    ),
    ExperimentConfig(
        name="gravity",
        target_column="F",
        dataset_generator=generate_gravity_dataset,
        expected_tokens=("m1", "m2", "r**(-2)"),
        error_threshold=0.25,
        default_discovery_kwargs={
            "max_basis_terms": 12,
            "max_iterations": 50,  # L2
            "allowed_powers": [1],
            "use_3var_products": True,
            "three_var_variables": ["m1", "m2", "r"],
            "enforce_dimensions": False,
            "unary_functions": [],
            "include_pairwise_products": True,
            "add_physics_features": False,
            "add_latent_features": False,
            "use_ot_loss": True,
            "alpha": 0.7,
            "beta": 0.3,
        },
        success_mode="token_and_error",
    ),
    ExperimentConfig(
        name="orbit_ax",
        target_column="ax",
        dataset_generator=generate_orbit_ax_dataset,
        expected_tokens=("x_over_r3",),
        error_threshold=0.08,
        default_discovery_kwargs={
            "max_basis_terms": 4,
            "max_iterations": 100,  # L4
            "allowed_powers": [1],
            "enforce_dimensions": False,
            "unary_functions": [],
        },
        operator_profiles=("linear",),
        success_mode="token_and_error",
    ),
    ExperimentConfig(
        name="orbit_ay",
        target_column="ay",
        dataset_generator=generate_orbit_ay_dataset,
        expected_tokens=("y_over_r3",),
        error_threshold=0.08,
        default_discovery_kwargs={
            "max_basis_terms": 4,
            "max_iterations": 100,  # L4
            "allowed_powers": [1],
            "enforce_dimensions": False,
            "unary_functions": [],
        },
        operator_profiles=("linear",),
        success_mode="token_and_error",
    ),
    ExperimentConfig(
        name="harmonic_oscillator_xdot",
        target_column="x_dot",
        dataset_generator=generate_harmonic_oscillator_dataset,
        expected_tokens=("v",),
        error_threshold=0.2,
        default_discovery_kwargs={
            "max_basis_terms": 6,
            "max_iterations": 50,  # L3
            "allowed_powers": [1],
            "enforce_dimensions": False,
            "unary_functions": [],
        },
        operator_profiles=("linear",),
    ),
    ExperimentConfig(
        name="harmonic_oscillator_vdot",
        target_column="v_dot",
        dataset_generator=generate_harmonic_oscillator_dataset,
        expected_tokens=("x",),
        error_threshold=0.2,
        default_discovery_kwargs={
            "max_basis_terms": 6,
            "max_iterations": 50,  # L3
            "allowed_powers": [1],
            "enforce_dimensions": False,
            "unary_functions": [],
        },
        operator_profiles=("linear",),
    ),
    ExperimentConfig(
        name="harmonic_oscillator_lagrangian",
        target_column="L",
        dataset_generator=generate_harmonic_oscillator_lagrangian_dataset,
        expected_tokens=("q**2", "dqdt**2"),
        error_threshold=0.2,
        default_discovery_kwargs={
            "q_col": "q",
            "dqdt_col": "dqdt",
            "d2qdt2_col": "d2qdt2",
            "max_power": 2,
            "include_cross_terms": True,
        },
        operator_profiles=("linear",),
        success_mode="token_and_error",
        discovery_mode="lagrangian",
    ),
    ExperimentConfig(
        name="harmonic_oscillator_structured_lagrangian",
        target_column="L",
        dataset_generator=generate_harmonic_oscillator_lagrangian_dataset,
        expected_tokens=("q2", "dq2"),
        error_threshold=0.2,
        default_discovery_kwargs={
            "q_col": "q",
            "dq_col": "dqdt",
            "ddq_col": "d2qdt2",
            "kinetic_max_even_power": 2,
            "potential_max_even_power": 2,
            "include_trig": False,
        },
        operator_profiles=("linear",),
        success_mode="token_and_error",
        discovery_mode="structured_lagrangian",
    ),
    ExperimentConfig(
        name="harmonic_oscillator_hamiltonian",
        target_column="H",
        dataset_generator=generate_harmonic_oscillator_hamiltonian_dataset,
        expected_tokens=("q**2", "p**2"),
        error_threshold=0.2,
        default_discovery_kwargs={
            "q_col": "q",
            "p_col": "p",
            "dqdt_col": "dqdt",
            "dpdt_col": "dpdt",
            "max_power": 2,
            "include_cross_terms": True,
        },
        operator_profiles=("linear",),
        success_mode="token_and_error",
        discovery_mode="hamiltonian",
    ),
    ExperimentConfig(
        name="double_pendulum_theta1dot",
        target_column="theta1_dot",
        dataset_generator=generate_double_pendulum_dataset,
        expected_tokens=("omega1",),
        error_threshold=0.2,
        default_discovery_kwargs={
            "max_basis_terms": 2,
            "max_iterations": 100,  # L5
            "allowed_powers": [1],
            "enforce_dimensions": False,
            "unary_functions": [],
            "include_pairwise_products": False,
            "add_physics_features": False,
            "add_latent_features": False,
        },
        operator_profiles=("linear",),
        success_mode="token_and_error",
    ),
    ExperimentConfig(
        name="double_pendulum_theta2dot",
        target_column="theta2_dot",
        dataset_generator=generate_double_pendulum_dataset,
        expected_tokens=("omega2",),
        error_threshold=0.2,
        default_discovery_kwargs={
            "max_basis_terms": 2,
            "max_iterations": 100,  # L5
            "allowed_powers": [1],
            "enforce_dimensions": False,
            "unary_functions": [],
            "include_pairwise_products": False,
            "add_physics_features": False,
            "add_latent_features": False,
        },
        operator_profiles=("linear",),
        success_mode="token_and_error",
    ),
    ExperimentConfig(
        name="planar_robot_j11",
        target_column="J11",
        dataset_generator=generate_planar_robot_jacobian_dataset,
        expected_tokens=("sin(theta1)", "sin(theta1+theta2)"),
        error_threshold=0.2,
        default_discovery_kwargs={
            "max_basis_terms": 20,
            "max_iterations": 100,  # L5
            "allowed_powers": [1],
            "enforce_dimensions": False,
            "unary_functions": ["sin", "cos"],
        },
        operator_profiles=("linear",),
        success_mode="token_and_error",
    ),
    ExperimentConfig(
        name="planar_robot_j12",
        target_column="J12",
        dataset_generator=generate_planar_robot_jacobian_dataset,
        expected_tokens=("sin(theta1+theta2)",),
        error_threshold=0.2,
        default_discovery_kwargs={
            "max_basis_terms": 20,
            "max_iterations": 100,  # L5
            "allowed_powers": [1],
            "enforce_dimensions": False,
            "unary_functions": ["sin", "cos"],
        },
        operator_profiles=("linear",),
        success_mode="token_and_error",
    ),
    ExperimentConfig(
        name="planar_robot_j21",
        target_column="J21",
        dataset_generator=generate_planar_robot_jacobian_dataset,
        expected_tokens=("cos(theta1)", "cos(theta1+theta2)"),
        error_threshold=0.2,
        default_discovery_kwargs={
            "max_basis_terms": 20,
            "max_iterations": 100,  # L5
            "allowed_powers": [1],
            "enforce_dimensions": False,
            "unary_functions": ["sin", "cos"],
        },
        operator_profiles=("linear",),
        success_mode="token_and_error",
    ),
    ExperimentConfig(
        name="planar_robot_j22",
        target_column="J22",
        dataset_generator=generate_planar_robot_jacobian_dataset,
        expected_tokens=("cos(theta1+theta2)",),
        error_threshold=0.2,
        default_discovery_kwargs={
            "max_basis_terms": 20,
            "max_iterations": 100,  # L5
            "allowed_powers": [1],
            "enforce_dimensions": False,
            "unary_functions": ["sin", "cos"],
        },
        operator_profiles=("linear",),
        success_mode="token_and_error",
    ),
    ExperimentConfig(
        name="planar_robot_fk_x",
        target_column="x",
        dataset_generator=generate_robot_trajectory_dataset,
        expected_tokens=("cos_theta1", "cos_theta12"),
        error_threshold=0.2,
        default_discovery_kwargs={
            "max_basis_terms": 2,
            "max_iterations": 100,  # L4
            "allowed_powers": [1],
            "enforce_dimensions": False,
            "unary_functions": [],
            "include_pairwise_products": False,
        },
        operator_profiles=("linear",),
        success_mode="token_and_error",
    ),
    ExperimentConfig(
        name="planar_robot_fk_y",
        target_column="y",
        dataset_generator=generate_robot_trajectory_dataset,
        expected_tokens=("sin_theta1", "sin_theta12"),
        error_threshold=0.2,
        default_discovery_kwargs={
            "max_basis_terms": 2,
            "max_iterations": 100,  # L4
            "allowed_powers": [1],
            "enforce_dimensions": False,
            "unary_functions": [],
            "include_pairwise_products": False,
        },
        operator_profiles=("linear",),
        success_mode="token_and_error",
    ),
    # ── Out-of-Grammar benchmark task ─────────────────────────────────────────
    # True law: F = -k*x - b*v*|v|
    # The Abs(v) term cannot be expressed by the standard sin/cos grammar.
    # OT-only (baseline) fails; OT+JEPA (--use-jepa) should succeed.
    ExperimentConfig(
        name="oog_damped_oscillator",
        target_column="F",
        dataset_generator=generate_oog_damped_oscillator_dataset,
        expected_tokens=("Abs(v)",),
        error_threshold=0.5,
        default_discovery_kwargs={
            "max_basis_terms": 10,
            "max_iterations": 100,
            "allowed_powers": [1, 2, -1],
            "enforce_dimensions": False,
            "unary_functions": [],       # Abs not in standard grammar
            "include_pairwise_products": True,
            "add_physics_features": False,
            "add_latent_features": False,
            "use_ot_loss": True,
            "alpha": 0.7,
            "beta": 0.3,
        },
        operator_profiles=("linear",),
        success_mode="token_and_error",
    ),

    ExperimentConfig(
        name="oog_relativistic_correction",
        target_column="F",
        dataset_generator=generate_oog_relativistic_correction_dataset,
        expected_tokens=("r",),
        error_threshold=0.05,
        default_discovery_kwargs={
            "enforce_dimensions": False,
            "use_ot_loss": True,
            "alpha": 0.7,
            "beta": 0.3,
            "allowed_powers": [1, 2, 3, -1, -2, -3],
            "include_pairwise_products": False,
            "add_physics_features": False,
            "add_latent_features": False,
        },
        operator_profiles=("linear",),
        success_mode="token_or_error",
    ),
]


# ── Phase 1 (v3.3): Feynman 100 — Tier A (44 easiest equations) ──────────────
from .feynman_loader import register_feynman_experiments
EXPERIMENTS.extend(
    register_feynman_experiments(tier="A", ExperimentConfig=ExperimentConfig)
)


def _build_franka_experiments() -> list[ExperimentConfig]:
    components = ("M11", "M22", "M33", "M44", "M55", "M66", "M77")
    variants = ("baseline", "payload", "rotor4", "link5mass")

    component_tokens = {
        "M11": ("cos(q2)",),
        "M22": ("cos(q2)",),
        "M33": ("cos(q3)",),
        "M44": ("q4",),
        "M55": ("q5",),
        "M66": ("q6",),
        "M77": ("q7",),
    }

    configs: list[ExperimentConfig] = []
    for component in components:
        for variant in variants:
            suffix = "" if variant == "baseline" else f"_{variant}"
            task_name = f"franka_{component}{suffix}"
            configs.append(
                ExperimentConfig(
                    name=task_name,
                    target_column=component,
                    dataset_generator=partial(
                        generate_franka_mass_dataset,
                        component=component,
                        variant=variant,
                    ),
                    expected_tokens=component_tokens[component],
                    error_threshold=0.2,
                    default_discovery_kwargs={
                        "max_basis_terms": 24,
                        "max_iterations": 120,
                        "allowed_powers": [1],
                        "enforce_dimensions": False,
                        "unary_functions": ["sin", "cos"],
                        "include_pairwise_products": True,
                    },
                    operator_profiles=("linear",),
                    success_mode="token_or_error",
                )
            )
    return configs


EXPERIMENTS.extend(_build_franka_experiments())
