from __future__ import annotations

import json
from pathlib import Path


BENCHMARK_METADATA = {
    "newton": {
        "category": "algebraic",
        "difficulty_level": "L1",
        "target_law": "F = m * a",
        "description": "Linear force law discovery from static samples.",
    },
    "pendulum": {
        "category": "differential",
        "difficulty_level": "L3",
        "target_law": "theta'' + (g/L) * theta = 0",
        "description": "Second-order dynamical law discovery from trajectory data.",
    },
    "kepler_third_law": {
        "category": "power_law",
        "difficulty_level": "L2",
        "target_law": "T^2 ∝ r^3",
        "description": "Kepler third-law power scaling benchmark.",
    },
    "inverse_square_acceleration": {
        "category": "power_law",
        "difficulty_level": "L2",
        "target_law": "a ∝ 1 / r^2",
        "description": "Inverse-square radial acceleration benchmark.",
    },
    "gravity": {
        "category": "power_law",
        "difficulty_level": "L2",
        "target_law": "F ∝ m1 * m2 / r^2",
        "description": "Mass-coupled inverse-square force discovery.",
    },
    "orbit_ax": {
        "category": "vector_field",
        "difficulty_level": "L4",
        "target_law": "ax = -GM * x / r^3",
        "description": "Cartesian x-component central-force field discovery.",
    },
    "orbit_ay": {
        "category": "vector_field",
        "difficulty_level": "L4",
        "target_law": "ay = -GM * y / r^3",
        "description": "Cartesian y-component central-force field discovery.",
    },
    "orbit_gravity_discovery": {
        "category": "vector_field",
        "difficulty_level": "L4",
        "target_law": "ax, ay ∝ -x/r^3, -y/r^3",
        "description": "Composite orbit trajectory discovery benchmark for both Cartesian acceleration components.",
    },
    "harmonic_oscillator_xdot": {
        "category": "dynamical_system",
        "difficulty_level": "L5",
        "target_law": "dx/dt = v",
        "description": "Harmonic oscillator state equation discovery for x'.",
    },
    "harmonic_oscillator_vdot": {
        "category": "dynamical_system",
        "difficulty_level": "L5",
        "target_law": "dv/dt = -ω^2 x",
        "description": "Harmonic oscillator state equation discovery for v'.",
    },
    "harmonic_oscillator_lagrangian": {
        "category": "lagrangian",
        "difficulty_level": "L5",
        "target_law": "L = 0.5*dqdt^2 - 0.5*ω^2*q^2",
        "description": "Harmonic oscillator Lagrangian discovery via Euler-Lagrange residual minimization.",
    },
    "harmonic_oscillator_structured_lagrangian": {
        "category": "lagrangian",
        "difficulty_level": "L5",
        "target_law": "L(q, dq) = T(dq) - V(q)",
        "implemented_target": "(1*dq2) - (1*q2)",
        "description": "Harmonic oscillator structured Lagrangian discovery with split kinetic and potential libraries.",
    },
    "harmonic_oscillator_hamiltonian": {
        "category": "hamiltonian",
        "difficulty_level": "L5",
        "target_law": "H = 0.5*p^2 + 0.5*ω^2*q^2",
        "description": "Harmonic oscillator Hamiltonian discovery from canonical state derivatives.",
    },
    "double_pendulum_theta1dot": {
        "category": "dynamical_system",
        "difficulty_level": "L5",
        "target_law": "dtheta1/dt = omega1",
        "description": "Double pendulum state equation discovery for theta1'.",
    },
    "double_pendulum_theta2dot": {
        "category": "dynamical_system",
        "difficulty_level": "L5",
        "target_law": "dtheta2/dt = omega2",
        "description": "Double pendulum state equation discovery for theta2'.",
    },
    "planar_robot_j11": {
        "category": "robotics",
        "difficulty_level": "L5",
        "target_law": "J11 = -l1 sin(theta1) - l2 sin(theta1 + theta2)",
        "description": "Planar 2-link robot Jacobian element J11 discovery.",
    },
    "planar_robot_j12": {
        "category": "robotics",
        "difficulty_level": "L5",
        "target_law": "J12 = -l2 sin(theta1 + theta2)",
        "description": "Planar 2-link robot Jacobian element J12 discovery.",
    },
    "planar_robot_j21": {
        "category": "robotics",
        "difficulty_level": "L5",
        "target_law": "J21 = l1 cos(theta1) + l2 cos(theta1 + theta2)",
        "description": "Planar 2-link robot Jacobian element J21 discovery.",
    },
    "planar_robot_j22": {
        "category": "robotics",
        "difficulty_level": "L5",
        "target_law": "J22 = l2 cos(theta1 + theta2)",
        "description": "Planar 2-link robot Jacobian element J22 discovery.",
    },
    "planar_robot_fk_x": {
        "category": "robotics",
        "difficulty_level": "L4",
        "target_law": "x = l1 cos(theta1) + l2 cos(theta1 + theta2)",
        "description": "Planar 2-link forward kinematics discovery for x from trajectories.",
    },
    "planar_robot_fk_y": {
        "category": "robotics",
        "difficulty_level": "L4",
        "target_law": "y = l1 sin(theta1) + l2 sin(theta1 + theta2)",
        "description": "Planar 2-link forward kinematics discovery for y from trajectories.",
    },
}


for _component in ("M11", "M22", "M33", "M44", "M55", "M66", "M77"):
    BENCHMARK_METADATA[f"franka_{_component}"] = {
        "category": "robotics_dynamics",
        "difficulty_level": "L5",
        "target_law": f"Franka Panda mass-matrix diagonal {_component}(q)",
        "description": f"Rediscovery of Franka Panda {_component}(q) from Pinocchio-generated samples.",
    }
    BENCHMARK_METADATA[f"franka_{_component}_payload"] = {
        "category": "robotics_dynamics_perturbed",
        "difficulty_level": "L5",
        "target_law": f"Franka Panda {_component}(q) with +0.5kg end-effector payload",
        "description": "Perturbed benchmark for hidden-physics detection under payload shift.",
    }
    BENCHMARK_METADATA[f"franka_{_component}_rotor4"] = {
        "category": "robotics_dynamics_perturbed",
        "difficulty_level": "L5",
        "target_law": f"Franka Panda {_component}(q) with joint-4 rotor inertia perturbation",
        "description": "Perturbed benchmark for hidden-physics detection under rotor-inertia shift.",
    }
    BENCHMARK_METADATA[f"franka_{_component}_link5mass"] = {
        "category": "robotics_dynamics_perturbed",
        "difficulty_level": "L5",
        "target_law": f"Franka Panda {_component}(q) with +10% link-5 mass",
        "description": "Perturbed benchmark for hidden-physics detection under local mass shift.",
    }


def build_pir_bench_manifest():
    from physics_engine.runner.experiment_registry import EXPERIMENTS

    tasks = []
    for config in EXPERIMENTS:
        metadata = BENCHMARK_METADATA.get(
            config.name,
            {
                "category": "unknown",
                "difficulty_level": "L5",
                "target_law": "unspecified",
                "description": "No benchmark metadata provided.",
            },
        )

        task = {
            "name": config.name,
            "category": metadata["category"],
            "difficulty_level": metadata["difficulty_level"],
            "target_column": config.target_column,
            "target_law": metadata["target_law"],
            "description": metadata["description"],
            "implemented_target": metadata.get("implemented_target", metadata["target_law"]),
            "expected_tokens": list(config.expected_tokens),
            "error_threshold": float(config.error_threshold),
            "operator_profiles": list(config.operator_profiles),
            "default_discovery_kwargs": dict(config.default_discovery_kwargs),
            "dataset_parameters": {
                "dataset_size": "mapped by runner to num_samples/steps",
                "noise_level": "mapped by runner to noise_std when supported",
                "random_seed": "seed",
            },
        }
        tasks.append(task)

    return {
        "suite_name": "PIR-Bench",
        "version": "0.1.0",
        "description": "PIR-Bench evaluates automated physical law discovery across algebraic, differential, power-law, and vector-field discovery tasks under varying noise levels and dataset sizes.",
        "difficulty_levels": {
            "L1": "Linear relations",
            "L2": "Nonlinear algebraic and power-law relations",
            "L3": "Differential equation discovery",
            "L4": "Spatial/vector field laws",
            "L5": "Multi-equation coupled systems",
        },
        "recommended_sweep": {
            "repeats": 5,
            "dataset_sizes": [200, 500, 1000],
            "noise_levels": [0.0, 0.01, 0.05],
        },
        "tasks": tasks,
    }


def write_pir_bench_manifest(output_dir: str | Path = "pir_bench"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    manifest = build_pir_bench_manifest()
    manifest_path = output_path / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)

    return manifest_path


def write_pir_bench_markdown(output_dir: str | Path = "pir_bench"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    manifest = build_pir_bench_manifest()
    lines = [
        "# PIR-Bench",
        "",
        manifest["description"],
        "",
        "| Task | Category | Level | Target Law |",
        "| --- | --- | --- | --- |",
    ]

    for task in manifest["tasks"]:
        law = task["target_law"]
        if task["name"] == "newton":
            law = "$F = ma$"
        elif task["name"] == "pendulum":
            law = "$\\theta'' + (g/L)\\theta = 0$"
        elif task["name"] == "kepler_third_law":
            law = "$T^2 \\propto r^3$"
        elif task["name"] == "inverse_square_acceleration":
            law = "$a \\propto 1/r^2$"
        elif task["name"] == "gravity":
            law = "$F \\propto m_1 m_2 / r^2$"
        elif task["name"] == "orbit_ax":
            law = "$a_x = -GMx/r^3$"
        elif task["name"] == "orbit_ay":
            law = "$a_y = -GMy/r^3$"
        elif task["name"] == "harmonic_oscillator_xdot":
            law = "$\\dot{x} = v$"
        elif task["name"] == "harmonic_oscillator_vdot":
            law = "$\\dot{v} = -\\omega^2 x$"
        elif task["name"] == "harmonic_oscillator_lagrangian":
            law = "$L = 0.5\\dot{q}^2 - 0.5\\omega^2 q^2$"
        elif task["name"] == "harmonic_oscillator_structured_lagrangian":
            law = "$L(q, \\dot{q}) = T(\\dot{q}) - V(q)$"
        elif task["name"] == "harmonic_oscillator_hamiltonian":
            law = "$H = 0.5p^2 + 0.5\\omega^2 q^2$"
        elif task["name"] == "double_pendulum_theta1dot":
            law = "$\\dot{\\theta_1} = \\omega_1$"
        elif task["name"] == "double_pendulum_theta2dot":
            law = "$\\dot{\\theta_2} = \\omega_2$"
        elif task["name"] == "planar_robot_j11":
            law = "$J_{11} = -l_1\\sin(\\theta_1) - l_2\\sin(\\theta_1 + \\theta_2)$"
        elif task["name"] == "planar_robot_j12":
            law = "$J_{12} = -l_2\\sin(\\theta_1 + \\theta_2)$"
        elif task["name"] == "planar_robot_j21":
            law = "$J_{21} = l_1\\cos(\\theta_1) + l_2\\cos(\\theta_1 + \\theta_2)$"
        elif task["name"] == "planar_robot_j22":
            law = "$J_{22} = l_2\\cos(\\theta_1 + \\theta_2)$"
        elif task["name"] == "planar_robot_fk_x":
            law = "$x = l_1\\cos(\\theta_1) + l_2\\cos(\\theta_1 + \\theta_2)$"
        elif task["name"] == "planar_robot_fk_y":
            law = "$y = l_1\\sin(\\theta_1) + l_2\\sin(\\theta_1 + \\theta_2)$"

        lines.append(
            f"| {task['name']} | {task['category']} | {task['difficulty_level']} | {law} |"
        )

    if {"orbit_ax", "orbit_ay"}.issubset({task["name"] for task in manifest["tasks"]}):
        lines.append("| orbit_gravity_discovery | vector_field | L4 | $a_x, a_y \\propto -x/r^3, -y/r^3$ |")

    lines.extend(
        [
            "",
            "## Discovery Task Definition",
            "",
            "Given a dataset generated from an unknown physical process,",
            "the discovery system must infer a symbolic equation whose",
            "structure matches the ground-truth law up to multiplicative",
            "constants and algebraic rearrangements.",
            "",
            "## Dataset Generation",
            "",
            "Each benchmark dataset is generated synthetically from a ground-truth physical law and optionally corrupted with Gaussian noise.",
            "",
            "Parameters:",
            "",
            "- `dataset_size`: number of samples or simulation steps",
            "- `noise_level`: Gaussian noise $\\sigma$ applied to targets when supported",
            "- `random_seed`: reproducibility control",
            "",
            "## Evaluation",
            "",
            "Discovery success is determined by structure matching and/or thresholded prediction error, depending on the task.",
            "",
            "Structure matching is performed using normalized equation tokens",
            "(e.g., variables, operators, and exponents) ignoring multiplicative",
            "constants and commutative ordering.",
            "",
            "Reported metrics:",
            "",
            "- `success_rate_percent`",
            "- `mean_error`",
            "- `mean_confidence`",
            "- `discovery_confidence`",
            "",
            "Each discovered law is assigned a Discovery Confidence Score (DCS)",
            "based on prediction accuracy, model simplicity, residual randomness,",
            "and robustness across noise levels.",
            "",
            "Optional extension metric for future tasks:",
            "",
            "- `parameter_error` when true parameters are explicitly encoded",
            "",
            "## Difficulty Levels",
            "",
            "- L1 - linear algebraic laws",
            "- L2 - nonlinear power laws",
            "- L3 - differential equations",
            "- L4 - vector field laws",
            "- L5 - multi-equation dynamical systems",
            "",
            "## Benchmark Layout",
            "",
            "```text",
            "pir_bench/",
            "  newton/",
            "  pendulum/",
            "  kepler_third_law/",
            "  inverse_square_acceleration/",
            "  gravity/",
            "  orbit_ax/",
            "  orbit_ay/",
            "  harmonic_oscillator_xdot/",
            "  harmonic_oscillator_vdot/",
            "  harmonic_oscillator_lagrangian/",
            "  harmonic_oscillator_structured_lagrangian/",
            "  harmonic_oscillator_hamiltonian/",
            "  double_pendulum_theta1dot/",
            "  double_pendulum_theta2dot/",
            "  planar_robot_j11/",
            "  planar_robot_j12/",
            "  planar_robot_j21/",
            "  planar_robot_j22/",
            "  planar_robot_fk_x/",
            "  planar_robot_fk_y/",
            "```",
            "",
            "Each task folder contains:",
            "",
            "- `ground_truth.json`",
            "- `benchmark_config.yaml`",
            "- `dataset_generator.py` (reference stub)",
            "",
            "## Example Output",
            "",
            "| System | Runs | Successes | Success Rate |",
            "| --- | ---: | ---: | ---: |",
            "| newton | 45 | 45 | 100.0% |",
            "| pendulum | 45 | 45 | 100.0% |",
            "| kepler_third_law | 45 | 39 | 86.7% |",
            "| inverse_square_acceleration | 45 | 43 | 95.6% |",
            "| gravity | 45 | 41 | 91.1% |",
            "| orbit_ax | 45 | 45 | 100.0% |",
            "| orbit_ay | 45 | 45 | 100.0% |",
            "| harmonic_oscillator_xdot | 45 | 45 | 100.0% |",
            "| harmonic_oscillator_vdot | 45 | 45 | 100.0% |",
            "| harmonic_oscillator_lagrangian | 45 | 42 | 93.3% |",
            "| harmonic_oscillator_structured_lagrangian | 45 | 43 | 95.6% |",
            "| harmonic_oscillator_hamiltonian | 45 | 41 | 91.1% |",
            "| double_pendulum_theta1dot | 45 | 44 | 97.8% |",
            "| double_pendulum_theta2dot | 45 | 44 | 97.8% |",
            "",
            "## Recommended Sweep",
            "",
            "- repeats: 5",
            "- dataset sizes: 200, 500, 1000",
            "- noise levels: 0.0, 0.01, 0.05",
            "",
            "Run:",
            "",
            "```powershell",
            ".\\.venv\\Scripts\\python.exe run_discovery_benchmark.py --repeats 5 --dataset-sizes 200 500 1000 --noise-levels 0 0.01 0.05",
            "```",
        ]
    )

    markdown_path = output_path / "README.md"
    with markdown_path.open("w", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")

    return markdown_path


def write_pir_bench_layout(output_dir: str | Path = "pir_bench"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    manifest = build_pir_bench_manifest()
    created = []
    for task in manifest["tasks"]:
        task_dir = output_path / task["name"]
        task_dir.mkdir(parents=True, exist_ok=True)

        ground_truth = {
            "task": task["name"],
            "target_law": task["target_law"],
            "implemented_target": task["implemented_target"],
            "target_column": task["target_column"],
            "expected_tokens": task["expected_tokens"],
        }
        ground_truth_path = task_dir / "ground_truth.json"
        ground_truth_path.write_text(json.dumps(ground_truth, indent=2), encoding="utf-8")

        yaml_lines = [
            f"name: {task['name']}",
            f"category: {task['category']}",
            f"difficulty_level: {task['difficulty_level']}",
            f"target_column: {task['target_column']}",
            f"error_threshold: {task['error_threshold']}",
            "operator_profiles:",
        ]
        for profile in task["operator_profiles"]:
            yaml_lines.append(f"  - {profile}")
        yaml_lines.extend(
            [
                "dataset_parameters:",
                "  - dataset_size",
                "  - noise_level",
                "  - random_seed",
            ]
        )
        (task_dir / "benchmark_config.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

        dataset_generator_stub = (
            "# Reference dataset generator stub for PIR-Bench task.\n"
            "# The canonical implementation is registered in physics_engine/runner/experiment_registry.py\n"
            "\n"
            "def generate_dataset(dataset_size, noise_level, random_seed):\n"
            "    raise NotImplementedError('Use registered generator from experiment registry.')\n"
        )
        (task_dir / "dataset_generator.py").write_text(dataset_generator_stub, encoding="utf-8")
        created.append(task_dir)

    return created
