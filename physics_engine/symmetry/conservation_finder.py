from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConservationCheckResult:
    conserved: bool
    quantity: str
    score: float
    mean_abs_derivative: float
    values: np.ndarray


class ConservationFinder:
    """Find candidate conserved quantities from trajectories."""

    def __init__(self, tolerance: float = 1e-3) -> None:
        self.tolerance = float(tolerance)

    @staticmethod
    def _as_array(values: np.ndarray | list[float]) -> np.ndarray:
        return np.asarray(values, dtype=float).reshape(-1)

    def angular_momentum(
        self,
        x: np.ndarray | list[float],
        y: np.ndarray | list[float],
        vx: np.ndarray | list[float],
        vy: np.ndarray | list[float],
        mass: float = 1.0,
    ) -> np.ndarray:
        x_arr = self._as_array(x)
        y_arr = self._as_array(y)
        vx_arr = self._as_array(vx)
        vy_arr = self._as_array(vy)
        return float(mass) * (x_arr * vy_arr - y_arr * vx_arr)

    def linear_momentum(
        self,
        vx: np.ndarray | list[float],
        vy: np.ndarray | list[float],
        mass: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        vx_arr = self._as_array(vx)
        vy_arr = self._as_array(vy)
        return float(mass) * vx_arr, float(mass) * vy_arr

    def energy(
        self,
        vx: np.ndarray | list[float],
        vy: np.ndarray | list[float],
        potential: np.ndarray | list[float],
        mass: float = 1.0,
    ) -> np.ndarray:
        vx_arr = self._as_array(vx)
        vy_arr = self._as_array(vy)
        potential_arr = self._as_array(potential)
        kinetic = 0.5 * float(mass) * (vx_arr * vx_arr + vy_arr * vy_arr)
        return kinetic + potential_arr

    def is_conserved(self, quantity: np.ndarray | list[float], dt: float = 1.0) -> tuple[bool, float, float]:
        values = self._as_array(quantity)
        if len(values) < 2:
            return False, float("inf"), float("inf")

        derivative = np.gradient(values, float(dt))
        mean_abs_derivative = float(np.mean(np.abs(derivative)))
        scale = float(np.mean(np.abs(values))) + 1e-12
        score = float(mean_abs_derivative / scale)
        return bool(score <= self.tolerance), score, mean_abs_derivative

    def detect_angular_momentum(
        self,
        x: np.ndarray | list[float],
        y: np.ndarray | list[float],
        vx: np.ndarray | list[float],
        vy: np.ndarray | list[float],
        dt: float = 1.0,
        mass: float = 1.0,
    ) -> ConservationCheckResult:
        values = self.angular_momentum(x=x, y=y, vx=vx, vy=vy, mass=mass)
        conserved, score, mean_abs_derivative = self.is_conserved(values, dt=dt)
        return ConservationCheckResult(
            conserved=conserved,
            quantity="angular_momentum",
            score=score,
            mean_abs_derivative=mean_abs_derivative,
            values=values,
        )
