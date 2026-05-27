from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from physics_engine.core.dataset import Dataset


@dataclass(frozen=True)
class RobotJointIdentificationResult:
    estimated_parameters: dict[str, float]
    torque_mse: float
    torque_r2: float
    n_samples: int


class RobotJointIdentifier:
    def identify(self, dataset: Dataset) -> RobotJointIdentificationResult:
        required = ["theta", "omega", "alpha", "tau"]
        for name in required:
            if dataset.get(name) is None:
                raise ValueError(f"Dataset missing required variable: {name}")

        theta = np.asarray(dataset.get("theta"), dtype=float)
        omega = np.asarray(dataset.get("omega"), dtype=float)
        alpha = np.asarray(dataset.get("alpha"), dtype=float)
        tau = np.asarray(dataset.get("tau"), dtype=float)

        n = len(tau)
        if n < 3:
            raise ValueError("Dataset must contain at least 3 samples.")

        design = np.column_stack([alpha, omega, theta])
        solution, *_ = np.linalg.lstsq(design, tau, rcond=None)
        i_est, b_est, k_est = [float(v) for v in solution]

        tau_hat = design @ solution
        residual = tau_hat - tau
        mse = float(np.mean(residual**2))

        total_var = float(np.sum((tau - float(np.mean(tau))) ** 2))
        if total_var <= 0.0:
            r2 = 1.0
        else:
            r2 = float(1.0 - np.sum(residual**2) / total_var)

        return RobotJointIdentificationResult(
            estimated_parameters={
                "I": i_est,
                "b": b_est,
                "k": k_est,
            },
            torque_mse=mse,
            torque_r2=r2,
            n_samples=n,
        )


def run_robot_joint_identification(dataset: Dataset) -> dict:
    result = RobotJointIdentifier().identify(dataset)
    return {
        "estimated_parameters": result.estimated_parameters,
        "torque_mse": result.torque_mse,
        "torque_r2": result.torque_r2,
        "n_samples": result.n_samples,
    }
