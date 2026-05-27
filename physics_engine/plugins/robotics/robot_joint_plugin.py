from __future__ import annotations

from physics_engine.plugins.plugin_base import PIRPlugin


class RobotJointPlugin(PIRPlugin):
    """
    Plugin for 1-DOF robot joint identification and structure discovery.
    """

    name = "robotics_joint"

    def register_datasets(self) -> dict[str, str]:
        return {
            "robot_joint_1dof": "Synthetic 1-DOF joint dataset with theta, omega, alpha, torque",
            "robot_joint_dynamics": "Real or simulated joint motion data",
        }

    def register_models(self) -> dict[str, str]:
        return {
            "robot_joint_linear": "Linear damped spring-inertia model: I*α + b*ω + k*θ = τ",
            "robot_joint_nonlinear": "Nonlinear joint dynamics with friction and backlash",
        }

    def register_pipelines(self) -> dict[str, str]:
        return {
            "robot_joint_identification": "physics_engine.pipelines.RobotJointIdentifier",
            "robot_structure_discovery": "physics_engine.pipelines.RobotStructureDiscovery",
        }
