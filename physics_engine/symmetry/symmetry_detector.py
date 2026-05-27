from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SymmetryCheckResult:
    detected: bool
    score: float
    evaluated_points: int
    coverage: float


class SymmetryDetector:
    """Detect simple geometric symmetries from observed state/dynamics pairs."""

    def __init__(self, tolerance: float = 0.15, min_coverage: float = 0.6) -> None:
        self.tolerance = float(tolerance)
        self.min_coverage = float(min_coverage)

    @staticmethod
    def _as_array(values: np.ndarray | list[float]) -> np.ndarray:
        arr = np.asarray(values, dtype=float).reshape(-1)
        return arr

    @staticmethod
    def _rotate_2d(x: np.ndarray, y: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
        c = float(np.cos(angle))
        s = float(np.sin(angle))
        xr = c * x - s * y
        yr = s * x + c * y
        return xr, yr

    @staticmethod
    def _typical_spacing(x: np.ndarray, y: np.ndarray) -> float:
        points = np.column_stack([x, y])
        count = len(points)
        if count < 3:
            return float("inf")

        nearest_distances: list[float] = []
        for i in range(count):
            deltas = points - points[i]
            distances = np.sqrt(np.sum(deltas * deltas, axis=1))
            distances[i] = np.inf
            nearest_distances.append(float(np.min(distances)))

        return float(np.median(nearest_distances))

    @staticmethod
    def _nearest_indices(x: np.ndarray, y: np.ndarray, xr: np.ndarray, yr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = np.column_stack([x, y])
        query = np.column_stack([xr, yr])
        nearest_idx = np.empty(len(query), dtype=int)
        nearest_dist = np.empty(len(query), dtype=float)

        for i, q in enumerate(query):
            deltas = points - q
            distances = np.sqrt(np.sum(deltas * deltas, axis=1))
            idx = int(np.argmin(distances))
            nearest_idx[i] = idx
            nearest_dist[i] = float(distances[idx])

        return nearest_idx, nearest_dist

    def detect_rotational_symmetry(
        self,
        x: np.ndarray | list[float],
        y: np.ndarray | list[float],
        ax: np.ndarray | list[float],
        ay: np.ndarray | list[float],
        angles: tuple[float, ...] = (math.pi / 6, math.pi / 4, math.pi / 3),
    ) -> SymmetryCheckResult:
        """
        Check equivariance F(Rx) ~= R F(x) by rotating states and matching nearest
        observed states in the original dataset.
        """

        x_arr = self._as_array(x)
        y_arr = self._as_array(y)
        ax_arr = self._as_array(ax)
        ay_arr = self._as_array(ay)

        n = len(x_arr)
        if not (len(y_arr) == len(ax_arr) == len(ay_arr) == n) or n == 0:
            return SymmetryCheckResult(detected=False, score=float("inf"), evaluated_points=0, coverage=0.0)

        spacing = self._typical_spacing(x_arr, y_arr)
        distance_limit = float("inf") if not np.isfinite(spacing) else 2.5 * spacing

        errors: list[float] = []
        used_points = 0
        total_queries = 0

        for angle in angles:
            xr, yr = self._rotate_2d(x_arr, y_arr, angle)
            ax_rot, ay_rot = self._rotate_2d(ax_arr, ay_arr, angle)
            nearest_idx, nearest_dist = self._nearest_indices(x_arr, y_arr, xr, yr)

            total_queries += n
            for i, j in enumerate(nearest_idx):
                if nearest_dist[i] > distance_limit:
                    continue

                denom = float(np.hypot(ax_arr[j], ay_arr[j])) + 1e-12
                err = float(np.hypot(ax_arr[j] - ax_rot[i], ay_arr[j] - ay_rot[i]) / denom)
                errors.append(err)
                used_points += 1

        if used_points == 0:
            return SymmetryCheckResult(detected=False, score=float("inf"), evaluated_points=0, coverage=0.0)

        score = float(np.mean(errors))
        coverage = float(used_points / max(total_queries, 1))
        detected = bool(score <= self.tolerance and coverage >= self.min_coverage)
        return SymmetryCheckResult(detected=detected, score=score, evaluated_points=used_points, coverage=coverage)

    def detect_translation_symmetry(
        self,
        x: np.ndarray | list[float],
        ax: np.ndarray | list[float],
        shift: float | None = None,
    ) -> SymmetryCheckResult:
        """Check 1D translation invariance a(x + c) ~= a(x)."""

        x_arr = self._as_array(x)
        ax_arr = self._as_array(ax)

        n = len(x_arr)
        if len(ax_arr) != n or n == 0:
            return SymmetryCheckResult(detected=False, score=float("inf"), evaluated_points=0, coverage=0.0)

        if shift is None:
            shift = float(np.std(x_arr))
            if not np.isfinite(shift) or shift == 0.0:
                shift = 1.0

        shifted = x_arr + float(shift)

        order = np.argsort(x_arr)
        x_sorted = x_arr[order]
        ax_sorted = ax_arr[order]

        min_x = float(x_sorted[0])
        max_x = float(x_sorted[-1])

        mask = (shifted >= min_x) & (shifted <= max_x)
        if not np.any(mask):
            return SymmetryCheckResult(detected=False, score=float("inf"), evaluated_points=0, coverage=0.0)

        interpolated = np.interp(shifted[mask], x_sorted, ax_sorted)
        reference = ax_arr[mask]
        denom = float(np.mean(np.abs(reference))) + 1e-12
        score = float(np.mean(np.abs(reference - interpolated)) / denom)
        coverage = float(np.mean(mask.astype(float)))
        detected = bool(score <= self.tolerance and coverage >= self.min_coverage)
        return SymmetryCheckResult(detected=detected, score=score, evaluated_points=int(np.sum(mask)), coverage=coverage)
