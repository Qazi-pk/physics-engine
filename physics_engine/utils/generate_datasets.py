from pathlib import Path

from physics_engine.benchmarks.gravity import generate_gravity_dataset
from physics_engine.benchmarks.kepler import generate_kepler_dataset
from physics_engine.benchmarks.newton import generate_newton_dataset
from physics_engine.benchmarks.pendulum import generate_pendulum_dataset
from physics_engine.benchmarks.robotics import (
    generate_planar_robot_jacobian_dataset,
    generate_planar_robot_kinematics_dataset,
)
from physics_engine.benchmarks.robotics_trajectory import generate_robot_trajectory_dataset


def generate_all_benchmarks(output_dir="datasets"):
    out = Path(output_dir)
    synthetic = out / "synthetic"
    out.mkdir(parents=True, exist_ok=True)
    synthetic.mkdir(parents=True, exist_ok=True)

    generate_newton_dataset().to_csv(synthetic / "newton.csv", index=False)
    generate_pendulum_dataset().to_csv(out / "pendulum.csv", index=False)
    generate_kepler_dataset().to_csv(out / "orbital_data.csv", index=False)
    generate_gravity_dataset().to_csv(synthetic / "gravity.csv", index=False)
    generate_planar_robot_kinematics_dataset().to_csv(synthetic / "planar_robot.csv", index=False)
    generate_planar_robot_jacobian_dataset().to_csv(synthetic / "planar_robot_jacobian.csv", index=False)
    generate_robot_trajectory_dataset().to_csv(synthetic / "planar_robot_trajectory.csv", index=False)
